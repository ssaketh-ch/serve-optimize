import json
import signal
import socket
import sqlite3
from datetime import datetime, timezone

import pytest

from serve_optimize.backends.factory import (
    MANAGED_BACKEND_CHOICES,
    SUPPORTED_MANAGED_BACKENDS,
    UnsupportedManagedBackendError,
    create_managed_backend_adapter,
)
from serve_optimize.backends.sglang import (
    SGLangArgumentCapabilities,
    _select_sglang_help_text,
    detect_sglang_argument_capabilities,
    parse_sglang_argument_capabilities,
    render_sglang_launch,
)
from serve_optimize.backends.vllm import (
    VllmAdapter,
    VLLMArgumentCapabilities,
    allocate_port,
    detect_vllm_argument_capabilities,
    parse_vllm_argument_capabilities,
    render_vllm_launch,
    validate_port_available,
    vllm_command,
)
from serve_optimize.cli import main
from serve_optimize.evidence import launch_config_hash, workload_config_hash
from serve_optimize.managed import (
    _generate_managed_candidates,
    build_managed_preflight,
    group_candidates_by_launch_config,
    run_managed_evaluation,
    serving_config_to_launch_config,
    serving_config_to_workload_config,
)
from serve_optimize.managed_candidates import CapabilityContext, generate_managed_candidates_from_capabilities
from serve_optimize.modeling import infer_model_capability_metadata
from serve_optimize.priors import ManagedPriorPolicy, apply_managed_prior_policy
from serve_optimize.schemas import (
    Goal,
    GpuDevice,
    HardwareSnapshot,
    HealthCheckResult,
    ManagedLifecycleRecord,
    ManagedRunSummary,
    PowerSampleRecord,
    PriorCandidate,
    PriorResult,
    PriorSource,
    RequestRecord,
    ServerHandle,
    ServerLaunchSpec,
    ServingConfig,
    WorkloadProfile,
)
from serve_optimize.synthesis import SYNTHESIS_SOURCE, CandidateSynthesisResult
from serve_optimize.telemetry import TelemetryCapture
from serve_optimize.validation import normalize_quantization, validate_managed_candidate


def test_vllm_managed_launch_spec_builds_command_and_logs(tmp_path) -> None:
    config = _config(dtype="bf16", quantization="awq-int4")
    port = allocate_port("127.0.0.1")

    spec = VllmAdapter(argument_capabilities=_all_engine_caps()).build_launch_spec(
        config,
        host="127.0.0.1",
        port=port,
        log_dir=tmp_path / "logs",
    )

    assert spec.command[:3] == ["vllm", "serve", "model-path"]
    assert spec.command[spec.command.index("--dtype") + 1] == "bfloat16"
    assert spec.command[spec.command.index("--host") + 1] == "127.0.0.1"
    assert spec.command[spec.command.index("--port") + 1] == str(port)
    assert spec.command[spec.command.index("--quantization") + 1] == "awq"
    assert spec.stdout_log_path == str(tmp_path / "logs" / config.id / "stdout.log")
    assert spec.stderr_log_path == str(tmp_path / "logs" / config.id / "stderr.log")


def test_managed_backend_factory_supports_vllm_and_sglang() -> None:
    adapter = create_managed_backend_adapter("vllm")
    sglang = create_managed_backend_adapter("sglang")

    assert adapter.name == "vllm"
    assert sglang.name == "sglang"
    assert SUPPORTED_MANAGED_BACKENDS == ("vllm", "sglang")
    assert "sglang" in MANAGED_BACKEND_CHOICES


def test_sglang_capability_detection_unavailable_nonfatal(monkeypatch) -> None:
    def missing_run(*args, **kwargs):
        del args, kwargs
        raise OSError("missing")

    monkeypatch.setattr("serve_optimize.backends.sglang.shutil.which", lambda name: None)
    monkeypatch.setattr("serve_optimize.backends.sglang.subprocess.run", missing_run)

    capabilities = detect_sglang_argument_capabilities(timeout_s=0.01)

    assert capabilities.detection_status in {"unavailable", "error"}
    assert capabilities.to_artifact()["backend"] == "sglang"
    assert "help_text" not in capabilities.to_artifact()


def test_sglang_capability_detection_supports_serve_subcommand(monkeypatch) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_which(name):
        return "/bin/sglang" if name == "sglang" else None

    def fake_run(command, **kwargs):
        del kwargs
        if command == ["python", "-m", "sglang.launch_server", "--help"]:
            return Completed(1, stderr="No module named sglang")
        if command == ["/bin/sglang", "version"]:
            return Completed(0, stdout="0.5.13.post1\n")
        if command == ["/bin/sglang", "serve", "--help"]:
            return Completed(0, stdout=_sglang_help_text())
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("serve_optimize.backends.sglang.sys.executable", "python")
    monkeypatch.setattr("serve_optimize.backends.sglang.shutil.which", fake_which)
    monkeypatch.setattr("serve_optimize.backends.sglang.subprocess.run", fake_run)

    capabilities = detect_sglang_argument_capabilities(timeout_s=0.01)

    assert capabilities.detection_status == "success"
    assert capabilities.launch_command == ("/bin/sglang", "serve")
    assert capabilities.version == "0.5.13.post1"
    assert capabilities.supports("--model-path")


def test_managed_backend_factory_rejects_unknown_backend() -> None:
    with pytest.raises(UnsupportedManagedBackendError) as exc:
        create_managed_backend_adapter("unknown")

    assert "Unsupported managed backend 'unknown'. Currently supported: vllm, sglang." == str(exc.value)


def test_sglang_help_parser_detects_flags_and_hash() -> None:
    capabilities = parse_sglang_argument_capabilities(_sglang_help_text())

    assert capabilities.detection_status == "success"
    assert capabilities.supports("--model-path")
    assert capabilities.supports("--context-length")
    assert capabilities.tensor_parallel_flag() == "--tp-size"
    assert capabilities.choices_for("--dtype") == frozenset({"auto", "float16", "bfloat16"})
    assert capabilities.choices_for("--quantization") == frozenset({"awq", "gptq"})
    assert capabilities.help_hash == parse_sglang_argument_capabilities(_sglang_help_text()).help_hash


def test_sglang_help_hash_excludes_timestamped_stderr_warnings() -> None:
    stdout = _sglang_help_text()
    first = _select_sglang_help_text(stdout, "[W615 03:54:06] warning")
    second = _select_sglang_help_text(stdout, "[W615 03:55:07] warning")

    assert parse_sglang_argument_capabilities(first).help_hash == parse_sglang_argument_capabilities(second).help_hash


def test_sglang_help_hash_normalizes_choice_order() -> None:
    first = _sglang_help_text() + "\n  --sampling-backend {pytorch,flashinfer,ascend}\n"
    second = _sglang_help_text() + "\n  --sampling-backend {flashinfer,pytorch,ascend}\n"

    assert parse_sglang_argument_capabilities(first).help_hash == parse_sglang_argument_capabilities(second).help_hash


def test_sglang_command_renders_supported_fields() -> None:
    config = _config(
        backend="sglang",
        dtype="bf16",
        quantization="awq-int4",
        max_batch_size=2,
        gpu_memory_utilization=0.8,
        kv_cache_policy="backend-default",
        scheduler="backend-default",
        extra={
            "served_model_name": "served-test",
            "trust_remote_code": True,
            "disable_piecewise_cuda_graph": True,
            "chunked_prefill_size": 1024,
            "disable_radix_cache": True,
            "cuda_graph_max_bs": 16,
        },
    )

    rendered = render_sglang_launch(config, host="127.0.0.1", port=8000, capabilities=_sglang_caps())
    command = rendered.command

    assert command[:3] == ["python", "-m", "sglang.launch_server"]
    assert command[command.index("--model-path") + 1] == "model-path"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--context-length") + 1] == "2048"
    assert command[command.index("--max-running-requests") + 1] == "2"
    assert command[command.index("--mem-fraction-static") + 1] == "0.8"
    assert command[command.index("--quantization") + 1] == "awq"
    assert command[command.index("--chunked-prefill-size") + 1] == "1024"
    assert command[command.index("--cuda-graph-max-bs") + 1] == "16"
    assert command[command.index("--served-model-name") + 1] == "served-test"
    assert "--trust-remote-code" in command
    assert "--disable-radix-cache" in command
    assert "--disable-piecewise-cuda-graph" in command
    assert rendered.rendered_fields["disable_piecewise_cuda_graph"] is True
    assert rendered.canonical_config.extra["disable_piecewise_cuda_graph"] is True
    assert rendered.canonical_config.quantization == "awq"
    assert rendered.canonical_config.backend == "sglang"
    assert rendered.capabilities_help_hash == "sglang-help-hash"
    assert rendered.to_metadata()["command"] == command


def test_sglang_command_rejects_vllm_only_fields() -> None:
    config = _config(backend="sglang", block_size=16)

    rendered = render_sglang_launch(config, capabilities=_sglang_caps())

    assert "block_size" in rendered.unsupported_fields
    result = validate_managed_candidate(
        config,
        backend="sglang",
        model_metadata=infer_model_capability_metadata("model-path"),
        sglang_argument_capabilities=_sglang_caps(),
    )
    assert result.valid is False
    assert "without a direct SGLang translation" in str(result.reason)


def test_sglang_command_marks_unknown_capabilities_unavailable() -> None:
    capabilities = SGLangArgumentCapabilities(
        executable="python",
        launch_command=("python", "-m", "sglang.launch_server"),
        detection_status="unavailable",
        detection_error="help command failed",
    )

    rendered = render_sglang_launch(
        _config(backend="sglang", extra={"disable_piecewise_cuda_graph": True}),
        capabilities=capabilities,
    )

    assert rendered.command == ["python", "-m", "sglang.launch_server"]
    assert rendered.unsupported_fields == {}
    assert "model" in rendered.unavailable_fields
    assert "gpu_memory_utilization" in rendered.unavailable_fields
    assert "disable_piecewise_cuda_graph" in rendered.unavailable_fields
    assert rendered.backend_metadata["capability_detection_status"] == "unavailable"


def test_sglang_validation_rejects_incompatible_graph_options() -> None:
    config = _config(
        backend="sglang",
        extra={"disable_cuda_graph": True, "cuda_graph_max_bs": 16},
    )

    result = validate_managed_candidate(
        config,
        backend="sglang",
        model_metadata=infer_model_capability_metadata("model-path"),
        sglang_argument_capabilities=_sglang_caps(),
    )

    assert result.valid is False
    assert result.reason == "disable_cuda_graph=true cannot be combined with cuda_graph_max_bs."


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"served_model_name": ""}, "served_model_name must be a non-empty string."),
        ({"trust_remote_code": "yes"}, "trust_remote_code must be a boolean."),
    ],
)
def test_sglang_validation_rejects_invalid_direct_options(extra, reason) -> None:
    result = validate_managed_candidate(
        _config(backend="sglang", extra=extra),
        backend="sglang",
        model_metadata=infer_model_capability_metadata("model-path"),
        sglang_argument_capabilities=_sglang_caps(),
    )

    assert result.valid is False
    assert result.reason == reason


@pytest.mark.parametrize(
    ("extra", "field_name", "flag"),
    [
        ({"served_model_name": "served-test"}, "served_model_name", "--served-model-name"),
        ({"trust_remote_code": True}, "trust_remote_code", "--trust-remote-code"),
    ],
)
def test_sglang_validation_rejects_unsupported_direct_options(extra, field_name, flag) -> None:
    result = validate_managed_candidate(
        _config(
            backend="sglang",
            max_batch_size=1,
            gpu_memory_utilization=0.0,
            extra=extra,
        ),
        backend="sglang",
        model_metadata=infer_model_capability_metadata("model-path"),
        sglang_argument_capabilities=_sglang_caps("--model-path", "--dtype", "--context-length"),
    )

    assert result.valid is False
    assert result.reason == f"{field_name} requires detected SGLang support for {flag}."


def test_serving_config_new_engine_fields_default_to_none() -> None:
    config = _config()

    assert config.block_size is None
    assert config.kv_cache_dtype is None
    assert config.enforce_eager is None
    assert config.max_num_batched_tokens is None
    assert config.enable_chunked_prefill is None
    assert config.max_cudagraph_capture_size is None
    assert config.enable_prefix_caching is None


def test_vllm_command_renders_engine_options(tmp_path) -> None:
    config = _config(
        block_size=16,
        kv_cache_dtype="auto",
        enforce_eager=True,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )

    spec = VllmAdapter(argument_capabilities=_all_engine_caps()).build_launch_spec(
        config,
        host="127.0.0.1",
        port=allocate_port("127.0.0.1"),
        log_dir=tmp_path / "logs",
    )

    assert spec.command[spec.command.index("--block-size") + 1] == "16"
    assert spec.command[spec.command.index("--kv-cache-dtype") + 1] == "auto"
    assert "--enforce-eager" in spec.command
    assert spec.command[spec.command.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--enable-chunked-prefill" in spec.command
    assert "--enable-prefix-caching" in spec.command
    assert "--max-cudagraph-capture-size" not in spec.command


def test_vllm_command_renders_false_chunked_prefill_and_cudagraph(tmp_path) -> None:
    config = _config(
        max_num_batched_tokens=4096,
        enable_chunked_prefill=False,
        max_cudagraph_capture_size=32,
    )

    spec = VllmAdapter(argument_capabilities=_all_engine_caps()).build_launch_spec(
        config,
        host="127.0.0.1",
        port=allocate_port("127.0.0.1"),
        log_dir=tmp_path / "logs",
    )

    assert "--no-enable-chunked-prefill" in spec.command
    assert spec.command[spec.command.index("--max-cudagraph-capture-size") + 1] == "32"
    assert "--enforce-eager" not in spec.command


def test_vllm_command_renders_cuda_graph_sizes_alias(tmp_path) -> None:
    config = _config(max_cudagraph_capture_size=32)

    spec = VllmAdapter(argument_capabilities=_caps("--cuda-graph-sizes")).build_launch_spec(
        config,
        host="127.0.0.1",
        port=allocate_port("127.0.0.1"),
        log_dir=tmp_path / "logs",
    )

    assert "--max-cudagraph-capture-size" not in spec.command
    assert spec.command[spec.command.index("--cuda-graph-sizes") + 1] == "32"


def test_vllm_rendered_launch_records_cuda_graph_alias() -> None:
    rendered = render_vllm_launch(_config(max_cudagraph_capture_size=32), capabilities=_caps("--cuda-graph-sizes"))

    assert rendered.command[rendered.command.index("--cuda-graph-sizes") + 1] == "32"
    assert rendered.canonical_config.max_cudagraph_capture_size == 32
    assert rendered.rendered_fields["max_cudagraph_capture_size"] == 32
    assert rendered.flag_aliases == {"max_cudagraph_capture_size": "--cuda-graph-sizes"}


def test_vllm_command_omits_unsupported_engine_options(tmp_path) -> None:
    config = _config(block_size=16, max_num_batched_tokens=4096, enable_chunked_prefill=True)

    spec = VllmAdapter(argument_capabilities=_caps()).build_launch_spec(
        config,
        host="127.0.0.1",
        port=allocate_port("127.0.0.1"),
        log_dir=tmp_path / "logs",
    )

    assert "--block-size" not in spec.command
    assert "--max-num-batched-tokens" not in spec.command
    assert "--enable-chunked-prefill" not in spec.command


def test_vllm_command_omits_null_engine_options(tmp_path) -> None:
    spec = VllmAdapter(argument_capabilities=_all_engine_caps()).build_launch_spec(
        _config(),
        host="127.0.0.1",
        port=allocate_port("127.0.0.1"),
        log_dir=tmp_path / "logs",
    )

    assert "--block-size" not in spec.command
    assert "--kv-cache-dtype" not in spec.command
    assert "--enforce-eager" not in spec.command
    assert "--max-num-batched-tokens" not in spec.command
    assert "--enable-chunked-prefill" not in spec.command
    assert "--no-enable-chunked-prefill" not in spec.command
    assert "--max-cudagraph-capture-size" not in spec.command
    assert "--enable-prefix-caching" not in spec.command


def test_vllm_help_parser_detects_engine_flags_and_choices() -> None:
    caps = parse_vllm_argument_capabilities(
        """
        --block-size BLOCK_SIZE
        --kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2,fp8_inc}
        --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
        --enable-chunked-prefill
        --no-enable-chunked-prefill
        --cuda-graph-sizes CUDA_GRAPH_SIZES
        --enable-prefix-caching
        """
    )

    assert caps.detection_status == "success"
    assert caps.supports("--block-size")
    assert caps.supports("--cuda-graph-sizes")
    assert caps.cudagraph_capture_flag() == "--cuda-graph-sizes"
    assert caps.choices_for("--kv-cache-dtype") == frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"})


def test_vllm_help_hash_excludes_timestamped_runtime_messages() -> None:
    help_text = """
    --block-size BLOCK_SIZE
    --kv-cache-dtype {auto,bfloat16,float16}
    """
    first = f"INFO 06-16 01:09:43 [__init__.py:235] Automatically detected platform cuda.\n{help_text}"
    second = f"INFO 06-16 01:09:47 [__init__.py:235] Automatically detected platform cuda.\n{help_text}"

    assert parse_vllm_argument_capabilities(first).help_hash == parse_vllm_argument_capabilities(second).help_hash


def test_vllm_help_hash_normalizes_choice_order() -> None:
    first = "--kv-cache-dtype {auto,bfloat16,float16}"
    second = "--kv-cache-dtype {float16,auto,bfloat16}"

    assert parse_vllm_argument_capabilities(first).help_hash == parse_vllm_argument_capabilities(second).help_hash


def test_vllm_help_hash_changes_with_supported_flags() -> None:
    first = "--kv-cache-dtype {auto,bfloat16,float16}"
    second = f"{first}\n--enable-prefix-caching"

    assert parse_vllm_argument_capabilities(first).help_hash != parse_vllm_argument_capabilities(second).help_hash


def test_vllm_capability_detection_failure_does_not_raise(monkeypatch) -> None:
    def fail_run(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr("serve_optimize.backends.vllm.subprocess.run", fail_run)

    caps = detect_vllm_argument_capabilities(executable="/not/a/vllm", timeout_s=0.01)

    assert caps.detection_status in {"failed", "unavailable"}
    assert caps.supported_flags == frozenset()


def test_port_allocation_and_validation() -> None:
    port = allocate_port("127.0.0.1")
    validate_port_available("127.0.0.1", port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        used_port = int(sock.getsockname()[1])
        with pytest.raises(ValueError):
            validate_port_available("127.0.0.1", used_port)


def test_health_check_success_and_failure() -> None:
    adapter = VllmAdapter()
    handle = _handle()

    ok = adapter.wait_for_health(handle, model="model-path", timeout_s=0.01, request_fn=_ok_request)
    failed = adapter.wait_for_health(handle, model="model-path", timeout_s=0.01, request_fn=_failed_request)

    assert ok.healthy is True
    assert ok.status == "ok"
    assert failed.healthy is False
    assert failed.error == "not ready"


def test_stop_only_kills_launched_process_group() -> None:
    calls: list[tuple[int, int]] = []
    process = _FakeProcess(pid=111)

    def killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        process.returncode = 0

    adapter = VllmAdapter(killpg_fn=killpg)
    adapter._processes[111] = process
    adapter._launched_pgids.add(222)

    stopped = adapter.stop_server(_handle(pid=111, pgid=222), timeout_s=0.01)
    skipped = adapter.stop_server(_handle(pid=333, pgid=444), timeout_s=0.01)

    assert stopped.status == "stopped"
    assert skipped.status == "skipped"
    assert calls == [(222, signal.SIGTERM)]


def test_failed_launch_records_candidate_failure_and_continues(tmp_path) -> None:
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=_LaunchFailAdapter(),
        candidate_provider=lambda: [_config()],
        evidence_write=False,
    )

    run_dir = tmp_path / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.status == "failed"
    assert summary.failed_candidate_count == 1
    assert (run_dir / "managed_run.json").exists()
    assert (run_dir / "launch_specs.jsonl").exists()
    assert (run_dir / "server_lifecycle.jsonl").exists()
    assert failures[0]["config_id"] == "cfg-test"
    assert failures[0]["stage"] == "launch"
    assert "boom" in failures[0]["error"]


def test_health_failure_records_reason_and_stops_server(tmp_path) -> None:
    class HealthFailAdapter(_SuccessAdapter):
        def __init__(self) -> None:
            self.stop_count = 0

        def wait_for_health(self, handle, *, model, timeout_s, request_fn=None):
            del model, timeout_s, request_fn
            return HealthCheckResult(
                config_id=handle.config_id,
                backend=handle.backend,
                base_url=handle.base_url,
                healthy=False,
                status="timeout",
                attempts=2,
                started_at=datetime.now(timezone.utc).isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                error="server did not become ready",
            )

        def stop_server(self, handle, *, timeout_s=30.0):
            self.stop_count += 1
            return super().stop_server(handle, timeout_s=timeout_s)

    adapter = HealthFailAdapter()
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=adapter,
        candidate_provider=lambda: [_config()],
        evidence_write=False,
    )

    run_dir = tmp_path / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lifecycle = [
        json.loads(line)
        for line in (run_dir / "server_lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.status == "failed"
    assert summary.workload_measurement_count == 0
    assert failures[0]["stage"] == "health"
    assert failures[0]["error"] == "server did not become ready"
    assert adapter.stop_count == 1
    assert any(record["event"] == "health" and record["status"] == "failed" for record in lifecycle)
    assert any(record["event"] == "stop" and record["status"] == "stopped" for record in lifecycle)


def test_benchmark_failure_records_reason_and_stops_server(tmp_path, monkeypatch) -> None:
    class TrackingAdapter(_SuccessAdapter):
        def __init__(self) -> None:
            self.stop_count = 0

        def stop_server(self, handle, *, timeout_s=30.0):
            self.stop_count += 1
            return super().stop_server(handle, timeout_s=timeout_s)

    def fail_benchmark(**_kwargs):
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr("serve_optimize.managed.run_endpoint_benchmark", fail_benchmark)
    adapter = TrackingAdapter()
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=adapter,
        candidate_provider=lambda: [_config()],
        evidence_write=False,
    )

    run_dir = tmp_path / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.status == "failed"
    assert failures[0]["stage"] == "benchmark"
    assert "benchmark failed" in failures[0]["error"]
    assert adapter.stop_count == 1


def test_stop_failure_is_recorded_without_losing_measurement(tmp_path) -> None:
    class StopFailAdapter(_SuccessAdapter):
        def stop_server(self, handle, *, timeout_s=30.0):
            del handle, timeout_s
            raise RuntimeError("stop failed")

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=StopFailAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config()],
        evidence_write=False,
    )

    run_dir = tmp_path / summary.run_id
    lifecycle = [
        json.loads(line)
        for line in (run_dir / "server_lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stop = next(record for record in lifecycle if record["event"] == "stop")

    assert summary.status == "success"
    assert summary.workload_measurement_count == 1
    assert stop["status"] == "failed"
    assert "stop failed" in stop["message"]


def test_keyboard_interrupt_records_reason_and_stops_server(tmp_path, monkeypatch) -> None:
    class TrackingAdapter(_SuccessAdapter):
        def __init__(self) -> None:
            self.stop_count = 0

        def stop_server(self, handle, *, timeout_s=30.0):
            self.stop_count += 1
            return super().stop_server(handle, timeout_s=timeout_s)

    def interrupt_benchmark(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("serve_optimize.managed.run_endpoint_benchmark", interrupt_benchmark)
    adapter = TrackingAdapter()

    with pytest.raises(KeyboardInterrupt):
        run_managed_evaluation(
            backend="vllm",
            model="model-path",
            goal=Goal.BALANCED,
            limit=1,
            trials=1,
            startup_timeout_s=1.0,
            cooldown_s=0.0,
            host="127.0.0.1",
            port=None,
            out_dir=tmp_path,
            telemetry="none",
            adapter=adapter,
            candidate_provider=lambda: [_config()],
            evidence_write=False,
        )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lifecycle = [
        json.loads(line)
        for line in (run_dir / "server_lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert failures[0]["stage"] == "interruption"
    assert "KeyboardInterrupt" in failures[0]["error"]
    assert adapter.stop_count == 1
    assert any(record["event"] == "interruption" for record in lifecycle)
    assert any(record["event"] == "stop" and record["status"] == "stopped" for record in lifecycle)


def test_managed_rejects_incompatible_quantization_before_launch(tmp_path) -> None:
    model_dir = tmp_path / "bf16-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8")
    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: [_config(config_id="cfg-awq", quantization="awq")],
        evidence_write=False,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failure_cache = json.loads((run_dir / "optimizer_failure_cache.json").read_text(encoding="utf-8"))
    launch_specs = [
        json.loads(line)
        for line in (run_dir / "launch_specs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.cold_launch_count == 0
    assert summary.candidates[0].status == "rejected"
    assert failures[0]["stage"] == "validation"
    assert "quantization awq requires model config quantization_config.quant_method=awq" in failures[0]["error"]
    assert summary.artifacts["optimizer_failure_cache_json"].endswith("optimizer_failure_cache.json")
    assert failure_cache["summary"]["stage_counts"]["validation"] == 1
    assert failure_cache["entries"][0]["cache_key"]
    assert failure_cache["entries"][0]["config_id"] == "cfg-awq"
    assert launch_specs == []


def test_managed_rejects_unrenderable_engine_field_before_launch(tmp_path) -> None:
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_NoOptionalEngineAdapter(),
        candidate_provider=lambda: [_config(config_id="cfg-cuda", max_cudagraph_capture_size=32)],
        evidence_write=False,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    launch_specs = [
        json.loads(line)
        for line in (run_dir / "launch_specs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.cold_launch_count == 0
    assert summary.failed_candidate_count == 1
    assert failures[0]["stage"] == "validation"
    assert "max_cudagraph_capture_size requires installed vLLM support" in failures[0]["error"]
    assert launch_specs == []


def test_generated_managed_candidates_include_model_native_bf16_baseline(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    candidates = _generate_managed_candidates(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        hardware=_managed_hardware(),
        model_metadata=metadata,
    )

    assert candidates[0].quantization == "none"
    assert candidates[0].dtype == "bf16"
    assert candidates[0].extra["candidate_source"] == "safe_baseline"
    assert candidates[0].extra["baseline"] is True
    assert candidates[0].extra["model_native"] is True
    assert any(config.quantization == "none" for config in candidates)


def test_managed_preflight_renders_without_launching(tmp_path) -> None:
    adapter = _CountingSuccessAdapter()
    run = build_managed_preflight(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=adapter,
        candidate_provider=lambda: [_config(config_id="cfg-preflight")],
        evidence_write=True,
    )

    payload = json.loads((run.run_dir / "preflight.json").read_text(encoding="utf-8"))
    text = (run.run_dir / "preflight.txt").read_text(encoding="utf-8")
    rendered_rows = [
        json.loads(line)
        for line in (run.run_dir / "rendered_launch_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert adapter.launch_count == 0
    assert payload["mode"] == "managed"
    assert payload["safety"]["will_launch_servers"] is False
    assert payload["safety"]["will_call_endpoint"] is False
    assert payload["safety"]["will_write_measured_evidence"] is False
    assert payload["evidence"]["write_enabled"] is True
    assert payload["candidates"]["valid_count"] == 1
    assert payload["budget"]["planned_workload_measurements"] == 1
    assert rendered_rows[0]["command"][:3] == ["vllm", "serve", "model-path"]
    assert "will launch servers: no" in text


def test_capability_generation_for_bf16_model_emits_multiple_none_candidates(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_all_engine_caps(),
        ),
        limit=5,
    )

    assert len(result.candidates) == 5
    assert result.safe_baseline_added is True
    assert result.candidates[0].extra["candidate_source"] == "safe_baseline"
    assert {config.quantization for config in result.candidates} == {"none"}
    assert {config.dtype for config in result.candidates} == {"bf16"}
    assert len({(config.max_batch_size, config.max_context_tokens, config.extra.get("workload_concurrency")) for config in result.candidates}) > 1
    assert result.invalid_quantization_filtered_count > 0
    assert all(
        getattr(result.candidates[0], field_name) is None
        for field_name in (
            "block_size",
            "kv_cache_dtype",
            "enforce_eager",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "max_cudagraph_capture_size",
            "enable_prefix_caching",
        )
    )
    assert any(config.max_num_batched_tokens is not None for config in result.candidates[1:])
    assert any(config.block_size == 16 for config in result.candidates[1:])


def test_sglang_capability_generation_is_bounded_and_excludes_vllm_fields(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="sglang",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            sglang_argument_capabilities=_sglang_caps(),
        ),
        limit=10,
    )

    assert 1 <= len(result.candidates) <= 4
    assert result.candidates[0].extra["candidate_source"] == "safe_baseline"
    assert result.safe_baseline_added is True
    assert {config.backend for config in result.candidates} == {"sglang"}
    assert {normalize_quantization(config.quantization) for config in result.candidates} == {"none"}
    assert {config.dtype for config in result.candidates} == {"bf16"}
    assert {config.gpu_memory_utilization for config in result.candidates} == {0.8, 0.9}
    assert any(config.max_batch_size == 4 for config in result.candidates)
    assert any("chunked_prefill_size" in config.extra for config in result.candidates)
    assert all(config.extra["disable_piecewise_cuda_graph"] is True for config in result.candidates)
    assert all(
        getattr(config, field_name) is None
        for config in result.candidates
        for field_name in (
            "block_size",
            "kv_cache_dtype",
            "enforce_eager",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "max_cudagraph_capture_size",
            "enable_prefix_caching",
        )
    )


def test_sglang_generation_uses_model_compatible_quantization(tmp_path) -> None:
    model_dir = _model_dir(
        tmp_path,
        {
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "awq"},
        },
    )
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="sglang",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            sglang_argument_capabilities=_sglang_caps(),
        ),
        limit=4,
    )

    assert {normalize_quantization(config.quantization) for config in result.candidates} == {"awq"}
    assert all(
        validate_managed_candidate(
            config,
            backend="sglang",
            model_metadata=metadata,
            sglang_argument_capabilities=_sglang_caps(),
        ).valid
        for config in result.candidates
    )


def test_capability_generation_omits_engine_options_when_vllm_help_unavailable(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=VLLMArgumentCapabilities(
                executable="vllm",
                version=None,
                detection_status="failed",
                detection_error="test",
            ),
        ),
        limit=5,
    )

    assert result.candidates[0].extra["candidate_source"] == "safe_baseline"
    assert all(config.block_size is None for config in result.candidates)
    assert all(config.kv_cache_dtype is None for config in result.candidates)
    assert all(config.max_cudagraph_capture_size is None for config in result.candidates)


def test_capability_generation_respects_installed_kv_cache_dtype_choices(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_caps(
                "--kv-cache-dtype",
                "--max-num-batched-tokens",
                "--enable-chunked-prefill",
                kv_cache_dtype_choices=("auto", "fp8", "fp8_e4m3"),
            ),
        ),
        limit=6,
    )

    assert all(config.kv_cache_dtype is None for config in result.candidates)


def test_default_workload_profile_preserves_candidate_behavior(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    default_result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_all_engine_caps(),
        ),
        limit=5,
    )
    explicit_default = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_all_engine_caps(),
            workload_profile=WorkloadProfile(),
        ),
        limit=5,
    )

    assert [config.id for config in explicit_default.candidates] == [config.id for config in default_result.candidates]
    assert all("workload_profile" not in config.extra for config in default_result.candidates)


def test_repeated_prefix_profile_generates_prefix_caching_when_supported(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_all_engine_caps(),
            workload_profile=WorkloadProfile(profile_name="repeated_prefix", prefix_reuse_expected=True, repeated_prefix_ratio=0.75),
        ),
        limit=5,
    )

    assert result.candidates[0].extra["candidate_source"] == "safe_baseline"
    assert result.candidates[0].enable_prefix_caching is None
    assert any(config.enable_prefix_caching is True for config in result.candidates[1:])
    assert any(config.extra["workload_profile"]["profile_name"] == "repeated_prefix" for config in result.candidates)


def test_repeated_prefix_profile_omits_prefix_caching_when_unsupported(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
            vllm_argument_capabilities=_caps("--max-num-batched-tokens"),
            workload_profile=WorkloadProfile(profile_name="repeated_prefix", prefix_reuse_expected=True),
        ),
        limit=5,
    )

    assert all(config.enable_prefix_caching is None for config in result.candidates)


def test_capability_generation_filters_legacy_quantized_candidates_for_bf16_model(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
        ),
        limit=8,
    )

    assert all(normalize_quantization(config.quantization) == "none" for config in result.candidates)
    assert len(result.candidates) == 8
    assert result.capability_filtered_count >= result.invalid_quantization_filtered_count
    assert result.invalid_quantization_filtered_count > 0


def test_capability_generation_allows_awq_when_model_metadata_matches(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "float16", "quantization_config": {"quant_method": "awq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
        ),
        limit=5,
    )

    assert {normalize_quantization(config.quantization) for config in result.candidates} == {"awq"}
    assert result.invalid_quantization_filtered_count > 0


def test_capability_generation_allows_gptq_when_model_metadata_matches(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "float16", "quantization_config": {"quant_method": "gptq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model=str(model_dir),
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
        ),
        limit=5,
    )

    assert {normalize_quantization(config.quantization) for config in result.candidates} == {"gptq"}
    assert result.invalid_quantization_filtered_count > 0


def test_capability_generation_for_unknown_remote_model_is_conservative_none_only() -> None:
    metadata = infer_model_capability_metadata("remote-org/remote-model")

    result = generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend="vllm",
            model="remote-org/remote-model",
            goal=Goal.BALANCED,
            hardware=_managed_hardware(),
            model_metadata=metadata,
        ),
        limit=5,
    )

    assert metadata.metadata_known is False
    assert {config.quantization for config in result.candidates} == {"none"}


def test_generated_baseline_survives_backend_filtering_and_truncation(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    candidates = _generate_managed_candidates(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        hardware=_managed_hardware(),
        model_metadata=metadata,
    )
    valid_candidates = [config for config in candidates if config.backend == "vllm" and config.quantization == "none"]

    assert valid_candidates
    assert candidates[0].id == valid_candidates[0].id


def test_safe_baseline_survives_prior_pruning(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))
    candidates = _generate_managed_candidates(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=4,
        hardware=_managed_hardware(),
        model_metadata=metadata,
    )

    result = apply_managed_prior_policy(
        candidates,
        prior_results=[_RecordingPriorProvider(candidate_id=candidates[-1].id).collect_priors(model=str(model_dir), backend="vllm", goal=Goal.BALANCED, candidates=candidates, out_dir=tmp_path)],
        evidence_priors=[],
        policy=ManagedPriorPolicy(max_prior_candidates=1, preserve_backend_default=False, preserve_low_memory_candidate=False, preserve_diversity=False),
    )

    assert any(config.extra.get("candidate_source") == "safe_baseline" for config in result.candidates)


def test_internal_generation_creates_launch_group_for_bf16_local_model(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    launch_groups = json.loads((run_dir / "launch_groups.json").read_text(encoding="utf-8"))

    assert summary.launch_groups_count >= 1
    assert launch_groups[0]["launch_config"]["quantization"] == "none"
    assert launch_groups[0]["launch_config"]["extra"]["candidate_source"] == "safe_baseline"


def test_internal_prior_provider_uses_resolved_hardware_system(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    systems: list[str] = []

    class RecordingPriorProvider:
        source = PriorSource.AICONFIGURATOR.value

        def __init__(self, *, system: str) -> None:
            systems.append(system)

        def collect_priors(self, *, model, backend, goal, candidates, out_dir):
            del model, backend, goal, candidates, out_dir
            return PriorResult(source=self.source, available=False, used=False)

    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware(name="NVIDIA H200 NVL"))
    monkeypatch.setattr("serve_optimize.managed.AIConfiguratorPriorProvider", RecordingPriorProvider)

    run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_db_path=tmp_path / "evidence.sqlite",
        synthesis_provider=_StaticSynthesisProvider([]),
    )

    assert systems == ["h200_sxm"]


def test_internal_generation_limit_five_keeps_useful_valid_candidates(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=5,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_CountingSuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        prior_policy=ManagedPriorPolicy(max_prior_candidates=5),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    launch_groups = json.loads((run_dir / "launch_groups.json").read_text(encoding="utf-8"))
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.valid_candidate_count_before_prior_pruning == 5
    assert summary.rejected_candidate_count_before_prior_pruning == 0
    assert summary.invalid_quantization_filtered_count > 0
    assert summary.safe_baseline_added is True
    assert summary.candidates_after_prior_pruning >= 2
    assert summary.launch_groups_count >= 2 or summary.average_workloads_per_launch > 1.0
    assert all(failure["stage"] != "validation" for failure in failures)
    assert managed_run["candidate_source_counts"]["safe_baseline"] == 1
    assert managed_run["valid_candidate_count_before_prior_pruning"] == 5
    assert all(group["launch_config"]["quantization"] == "none" for group in launch_groups)


def test_synthesized_candidate_appears_with_source_metadata(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    provider = _StaticSynthesisProvider([_synth_config(str(model_dir), config_id="cfg-synth", max_batch_size=4)])

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
        prior_policy=ManagedPriorPolicy(max_prior_candidates=5),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    synthesis = json.loads((run_dir / "candidate_synthesis.json").read_text(encoding="utf-8"))
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    rendered_rows = [
        json.loads(line)
        for line in (run_dir / "rendered_launch_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_rows = recommendation["recommendation"]["candidate_table"]

    assert provider.call_count == 1
    assert managed_run["candidate_source_counts"][SYNTHESIS_SOURCE] == 1
    assert managed_run["artifacts"]["candidate_synthesis_json"].endswith("candidate_synthesis.json")
    assert synthesis["summary"]["measured_candidate_count"] == 1
    assert synthesis["candidate_records"][0]["candidate_source"] == SYNTHESIS_SOURCE
    assert synthesis["candidate_records"][0]["status"] == "measured"
    assert any(row.get("candidate_source") == SYNTHESIS_SOURCE for row in candidate_rows)
    assert any(row["canonical_config"]["extra"]["candidate_source"] == SYNTHESIS_SOURCE for row in rendered_rows)


def test_synthesis_unavailable_preserves_safe_baseline(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    provider = _StaticSynthesisProvider([], available=False, warning="synthesis unavailable")

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    synthesis = json.loads((run_dir / "candidate_synthesis.json").read_text(encoding="utf-8"))

    assert summary.safe_baseline_added is True
    assert summary.completed_candidate_count == 1
    assert synthesis["provider_results"][0]["available"] is False
    assert synthesis["provider_results"][0]["warnings"] == ["synthesis unavailable"]


def test_invalid_synthesized_candidate_is_rejected_before_launch(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    provider = _StaticSynthesisProvider([_synth_config(str(model_dir), config_id="cfg-synth-bad", block_size=0)])

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synthesis = json.loads((run_dir / "candidate_synthesis.json").read_text(encoding="utf-8"))

    assert any(failure["config_id"] == "cfg-synth-bad" and failure["stage"] == "validation" for failure in failures)
    assert synthesis["summary"]["rejected_candidate_count"] == 1
    assert synthesis["candidate_records"][0]["status"] == "rejected"


def test_duplicate_synthesized_candidate_is_deduped(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    duplicate = _synth_config(str(model_dir), config_id="cfg-synth-duplicate", max_batch_size=1)
    provider = _StaticSynthesisProvider([duplicate])

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    synthesis = json.loads((run_dir / "candidate_synthesis.json").read_text(encoding="utf-8"))

    assert summary.valid_candidate_count_before_prior_pruning == 1
    assert synthesis["summary"]["deduped_candidate_count"] == 1
    assert synthesis["candidate_records"][0]["status"] == "deduped"


def test_exact_fresh_evidence_duplicate_synthesis_is_deduped(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    db_path = tmp_path / "evidence.sqlite"
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())

    run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "first",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=_StaticSynthesisProvider([]),
    )
    duplicate = _synth_config(str(model_dir), config_id="cfg-synth-duplicate", max_batch_size=1)
    second = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=_ShouldNotLaunchAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=_StaticSynthesisProvider([duplicate]),
    )

    run_dir = tmp_path / "second" / second.run_id
    synthesis = json.loads((run_dir / "candidate_synthesis.json").read_text(encoding="utf-8"))

    assert second.cold_launch_count == 0
    assert second.evidence_hit_candidate_count == 1
    assert synthesis["initial_preflight"]["exact_fresh_candidate_ids"]
    assert synthesis["summary"]["deduped_candidate_count"] == 1


def test_synthesized_candidate_with_unsupported_flag_is_rejected(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    provider = _StaticSynthesisProvider([_synth_config(str(model_dir), config_id="cfg-synth-block", max_batch_size=4, block_size=16)])

    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_NoOptionalEngineAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any("block_size requires installed vLLM support" in failure["error"] for failure in failures)


def test_synthesized_candidate_with_unsupported_kv_cache_dtype_is_rejected(tmp_path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    monkeypatch.setattr("serve_optimize.managed.detect_hardware", lambda: _managed_hardware())
    provider = _StaticSynthesisProvider([_synth_config(str(model_dir), config_id="cfg-synth-kv", max_batch_size=4, kv_cache_dtype="fp8_e4m3")])

    run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
        synthesis_provider=provider,
    )

    run_dir = next((tmp_path / "runs").glob("managed-*"))
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any("kv_cache_dtype 'fp8_e4m3' is not listed by installed vLLM" in failure["error"] for failure in failures)


def test_managed_evaluate_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["managed-evaluate", "--help"])
    assert exc.value.code == 0


def test_managed_evaluate_cli_help_does_not_expose_engine_surface_flags(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["managed-evaluate", "--help"])

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())
    assert exc.value.code == 0
    assert "vLLM and SGLang are supported" in normalized_output
    assert "--block-size" not in output
    assert "--kv-cache-dtype" not in output
    assert "--max-num-batched-tokens" not in output


def test_managed_evaluate_cli_sglang_unavailable_writes_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "serve_optimize.backends.sglang.detect_sglang_argument_capabilities",
        lambda: SGLangArgumentCapabilities(
            executable="python",
            detection_status="unavailable",
            detection_error="SGLang is unavailable for this test.",
        ),
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "managed-evaluate",
                "--backend",
                "sglang",
                "--model",
                "model-path",
                "--out",
                str(tmp_path),
                "--no-evidence-write",
            ]
        )

    assert exc.value.code == 1
    run_dir = next(tmp_path.glob("managed-*"))
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    capabilities = json.loads((run_dir / "sglang_argument_capabilities.json").read_text(encoding="utf-8"))
    failures = [
        json.loads(line)
        for line in (run_dir / "candidate_failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert managed_run["backend"] == "sglang"
    assert managed_run["status"] == "failed"
    assert managed_run["cold_launch_count"] == 0
    assert managed_run["workload_measurement_count"] == 0
    assert capabilities["backend"] == "sglang"
    assert capabilities["detection_status"] in {"unavailable", "timeout", "error"}
    assert any(failure["stage"] == "availability" for failure in failures)


def test_managed_evaluation_sglang_mocked_success_writes_backend_artifacts(tmp_path) -> None:
    config = _config(
        backend="sglang",
        config_id="cfg-sglang",
        max_batch_size=1,
        gpu_memory_utilization=0.0,
        kv_cache_policy="backend-default",
        scheduler="backend-default",
        extra={"disable_piecewise_cuda_graph": True},
    )

    summary = run_managed_evaluation(
        backend="sglang",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessSglangAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [config],
        evidence_write=False,
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    capabilities = json.loads((run_dir / "sglang_argument_capabilities.json").read_text(encoding="utf-8"))
    rendered_rows = [
        json.loads(line)
        for line in (run_dir / "rendered_launch_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))

    assert summary.status == "success"
    assert summary.backend == "sglang"
    assert summary.cold_launch_count == 1
    assert summary.workload_measurement_count == 1
    assert capabilities["backend"] == "sglang"
    assert capabilities["detection_status"] == "success"
    assert rendered_rows[0]["canonical_config"]["backend"] == "sglang"
    assert rendered_rows[0]["canonical_config"]["block_size"] is None
    assert rendered_rows[0]["unavailable_fields"] == {}
    assert "--disable-piecewise-cuda-graph" in rendered_rows[0]["command"]
    assert rendered_rows[0]["rendered_launch_command_hash"]
    assert rendered_rows[0]["runtime_environment"]["backend_name"] == "sglang"
    assert recommendation["recommendation"]["selected_serve_command"].startswith("python -m sglang.launch_server")
    assert "--context-length 2048" in recommendation["recommendation"]["selected_serve_command"]
    assert "--disable-piecewise-cuda-graph" in recommendation["recommendation"]["selected_serve_command"]
    assert recommendation["selected_runtime_fingerprint"]
    assert summary_json["selected"]["backend"] == "sglang"
    assert summary_json["recommended_command"].startswith("python -m sglang.launch_server")


def test_repeatability_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["repeatability", "--help"])

    assert exc.value.code == 0


def test_managed_evaluate_cli_prints_recommended_command(tmp_path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "managed-run"
    run_dir.mkdir()
    summary_json_path = run_dir / "recommendation_summary.json"
    summary_txt_path = run_dir / "recommendation_summary.txt"
    summary_json_path.write_text(
        json.dumps(
            {
                "status": "success",
                "confidence": "high",
                "recommended_command": "vllm serve model-path --dtype bf16 --max-model-len 2048",
                "metrics": {
                    "throughput_tokens_per_sec": 9125.02,
                    "p95_latency_ms": 54.84,
                    "average_power_w": 170.93,
                    "joules_per_token": 0.018732,
                    "tokens_per_watt": 53.39,
                },
            }
        ),
        encoding="utf-8",
    )
    summary_txt_path.write_text("summary", encoding="utf-8")

    def fake_run_managed_evaluation(**kwargs):
        del kwargs
        return ManagedRunSummary(
            run_id="managed-test",
            created_at="2026-01-01T00:00:00+00:00",
            backend="vllm",
            model="model-path",
            goal="balanced",
            candidate_count=1,
            completed_candidate_count=1,
            failed_candidate_count=0,
            startup_timeout_s=1.0,
            cooldown_s=0.0,
            trials=1,
            status="success",
            artifacts={
                "run_dir": str(run_dir),
                "recommendation_summary_txt": str(summary_txt_path),
                "recommendation_summary_json": str(summary_json_path),
            },
            recommendation_status="success",
            recommendation_confidence="high",
            recommendation_summary_txt_path=str(summary_txt_path),
            recommendation_summary_json_path=str(summary_json_path),
        )

    monkeypatch.setattr("serve_optimize.cli.run_managed_evaluation", fake_run_managed_evaluation)

    main(["managed-evaluate", "--model", "model-path", "--no-evidence-write"])

    output = capsys.readouterr().out
    assert "Recommended configuration:" in output
    assert "vllm serve model-path --dtype bf16 --max-model-len 2048" in output
    assert "Confidence: HIGH" in output
    assert f"Summary: {summary_txt_path}" in output


def test_managed_evaluate_cli_prints_unavailable_recommendation(monkeypatch, capsys) -> None:
    def fake_run_managed_evaluation(**kwargs):
        del kwargs
        return ManagedRunSummary(
            run_id="managed-test",
            created_at="2026-01-01T00:00:00+00:00",
            backend="vllm",
            model="model-path",
            goal="balanced",
            candidate_count=1,
            completed_candidate_count=0,
            failed_candidate_count=1,
            startup_timeout_s=1.0,
            cooldown_s=0.0,
            trials=1,
            status="failed",
            artifacts={"run_dir": "run-dir"},
            recommendation_status="unavailable",
            recommendation_reason="No measured candidates were available.",
        )

    monkeypatch.setattr("serve_optimize.cli.run_managed_evaluation", fake_run_managed_evaluation)

    with pytest.raises(SystemExit):
        main(["managed-evaluate", "--model", "model-path", "--no-evidence-write"])

    output = capsys.readouterr().out
    assert "Recommendation: unavailable" in output
    assert "No measured candidates were available." in output


def test_launch_config_hash_ignores_workload_only_fields() -> None:
    first = _config(config_id="cfg-a", extra={"workload_concurrency": 1, "num_requests": 8})
    second = _config(config_id="cfg-b", extra={"workload_concurrency": 4, "num_requests": 16})

    assert launch_config_hash(serving_config_to_launch_config(first)) == launch_config_hash(serving_config_to_launch_config(second))


def test_workload_config_hash_changes_when_workload_fields_change() -> None:
    first = serving_config_to_workload_config(_config(extra={"workload_concurrency": 1}), trials=1, request_timeout_s=30.0, telemetry="none")
    second = serving_config_to_workload_config(_config(extra={"workload_concurrency": 2}), trials=1, request_timeout_s=30.0, telemetry="none")

    assert workload_config_hash(first) != workload_config_hash(second)


def test_workload_profile_affects_workload_fingerprint_when_non_default() -> None:
    first = serving_config_to_workload_config(
        _config(extra={"workload_concurrency": 1}),
        trials=1,
        request_timeout_s=30.0,
        telemetry="none",
    )
    second = serving_config_to_workload_config(
        _config(extra={"workload_concurrency": 1, "workload_profile": {"profile_name": "repeated_prefix", "prefix_reuse_expected": True}}),
        trials=1,
        request_timeout_s=30.0,
        telemetry="none",
    )

    assert workload_config_hash(first) != workload_config_hash(second)
    assert second.extra["workload_profile"]["profile_name"] == "repeated_prefix"


def test_workload_profile_preserves_dataset_distribution_and_slos() -> None:
    profile = {
        "profile_name": "mixed",
        "dataset": "synthetic-mixed",
        "token_distribution": {"input_tokens": {"p50": 768, "p95": 4096}},
        "slo_constraints": {"p95_latency_ms": 900, "min_throughput_tokens_per_sec": 100},
    }

    workload = serving_config_to_workload_config(
        _config(extra={"workload_concurrency": 2, "workload_profile": profile}),
        trials=1,
        request_timeout_s=30.0,
        telemetry="none",
    )

    assert workload.dataset == "synthetic-mixed"
    assert workload.extra["workload_profile"]["token_distribution"]["input_tokens"]["p95"] == 4096
    assert workload.extra["workload_profile"]["slo_constraints"]["p95_latency_ms"] == 900
    assert workload_config_hash(workload) != workload_config_hash(
        serving_config_to_workload_config(
            _config(
                extra={
                    "workload_concurrency": 2,
                    "workload_profile": {
                        **profile,
                        "token_distribution": {"input_tokens": {"p50": 768, "p95": 2048}},
                    },
                }
            ),
            trials=1,
            request_timeout_s=30.0,
            telemetry="none",
        )
    )


def test_same_launch_different_workloads_form_one_group() -> None:
    candidates = [
        _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
        _config(config_id="cfg-b", extra={"workload_concurrency": 4}),
    ]

    groups = group_candidates_by_launch_config(candidates, trials=1, request_timeout_s=30.0, telemetry="none")

    assert len(groups) == 1
    assert groups[0].original_config_ids == ["cfg-a", "cfg-b"]
    assert [workload.concurrency for workload in groups[0].workload_configs] == [1, 4]


def test_different_launch_configs_form_separate_groups() -> None:
    candidates = [
        _config(config_id="cfg-a", dtype="fp16"),
        _config(config_id="cfg-b", dtype="bf16"),
    ]

    groups = group_candidates_by_launch_config(candidates, trials=1, request_timeout_s=30.0, telemetry="none")

    assert len(groups) == 2


def test_managed_evaluation_writes_evidence_with_mocked_benchmark(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[
                    PowerSampleRecord(float(index), "measured", 100.0 + index, telemetry, provider=telemetry, gpu_util_percent=70.0)
                    for index in range(5)
                ],
                warnings=[],
            )
        ),
        candidate_provider=lambda: [
            _config(
                block_size=16,
                kv_cache_dtype="bfloat16",
                max_num_batched_tokens=4096,
                enable_chunked_prefill=True,
                max_cudagraph_capture_size=32,
            )
        ],
        evidence_db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM evidence_measurements").fetchone()[0]

    assert summary.status == "success"
    assert summary.evidence_warnings == []
    assert count == 1


def test_managed_trials_write_aggregate_evidence(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=2,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config(extra={"num_requests": 2})],
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    aggregate_summary = json.loads((run_dir / "per_candidate" / "cfg-test-measure-aggregate" / "summary.json").read_text(encoding="utf-8"))
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT raw_json FROM evidence_measurements").fetchall()

    assert summary.workload_measurement_count == 2
    assert len(rows) == 1
    assert aggregate_summary["trial_statistics"]["trial_count"] == 2
    assert aggregate_summary["stability_classification"] in {"stable", "mostly_stable", "unstable"}
    raw_json = json.loads(rows[0][0])
    assert len(raw_json["trial_summaries"]) == 2
    assert raw_json["summary"]["trial_statistics"]["trial_count"] == 2


def test_managed_measured_candidate_writes_recommendation_artifacts(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [
            _config(
                block_size=16,
                kv_cache_dtype="bfloat16",
                max_num_batched_tokens=4096,
                enable_chunked_prefill=True,
                max_cudagraph_capture_size=32,
            )
        ],
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    pareto = json.loads((run_dir / "managed_pareto_frontier.json").read_text(encoding="utf-8"))
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))
    summary_text = (run_dir / "recommendation_summary.txt").read_text(encoding="utf-8")
    argument_caps = json.loads((run_dir / "vllm_argument_capabilities.json").read_text(encoding="utf-8"))
    candidate_table = recommendation["recommendation"]["candidate_table"]

    assert summary.recommendation_status == "success"
    assert summary.selected_config_id == "cfg-test"
    assert summary.selected_measurement_id is not None
    assert summary.recommendation_score is not None
    assert recommendation["recommendation"]["recommended_candidate_id"] == "cfg-test"
    assert recommendation["recommendation"]["evaluated_set_fidelity"]["scope"] == "evaluated_candidates_only"
    assert recommendation["recommendation"]["evaluated_set_fidelity"]["selected_rank"] == 1
    assert recommendation["recommendation"]["evaluated_set_fidelity"]["selected_is_best_evaluated"] is True
    assert recommendation["recommendation_quality_audit"]["scope"] == "evaluated_candidates_only"
    assert recommendation["recommendation_quality_audit"]["selected_is_best_evaluated"] is True
    assert recommendation["recommendation_quality_audit"]["optimizer_quality_present"] is True
    assert recommendation["optimizer_quality"]["scope"] == "evaluated_candidates_only"
    assert recommendation["optimizer_quality"]["search_regret"]["score_gap_to_best"] == pytest.approx(0.0)
    assert recommendation["recommendation_quality_audit"]["wording_policy"] == "evaluated_set_only"
    assert "--block-size 16" in recommendation["recommendation"]["selected_serve_command"]
    assert candidate_table[0]["block_size"] == 16
    assert candidate_table[0]["kv_cache_dtype"] == "bfloat16"
    assert candidate_table[0]["max_num_batched_tokens"] == 4096
    assert candidate_table[0]["max_cudagraph_capture_size"] == 32
    _assert_active_engine_fields_are_rendered(candidate_table[0], recommendation["recommendation"]["selected_serve_command"])
    assert pareto[0]["candidate_id"] == "cfg-test"
    assert managed_run["recommendation_status"] == "success"
    assert managed_run["recommendation_quality_audit"]["selected_candidate_id"] == "cfg-test"
    assert managed_run["optimizer_quality"]["scope"] == "evaluated_candidates_only"
    assert managed_run["artifacts"]["optimizer_quality_json"].endswith("optimizer_quality.json")
    assert (run_dir / "optimizer_quality.json").exists()
    assert managed_run["backend_metadata"]["adapter"] == "vllm"
    assert managed_run["runtime_environment"]["backend_name"] == "vllm"
    assert managed_run["runtime_environment"]["torch_version"]
    assert managed_run["artifacts"]["runtime_environment_json"].endswith("runtime_environment.json")
    assert recommendation["selected_runtime_fingerprint"]
    assert recommendation["runtime_environment"]["backend_name"] == "vllm"
    assert (run_dir / "managed_pareto_frontier.csv").exists()
    assert (run_dir / "managed_report.txt").exists()
    assert (run_dir / "recommendation_summary.txt").exists()
    assert (run_dir / "recommendation_summary.json").exists()
    assert managed_run["artifacts"]["recommendation_summary_txt"].endswith("recommendation_summary.txt")
    assert managed_run["recommendation_summary_txt_path"].endswith("recommendation_summary.txt")
    assert summary_json["recommended_command"].startswith("vllm serve model-path")
    assert summary_json["selected"]["candidate_id"] == "cfg-test"
    assert summary_json["selected"]["backend"] == "vllm"
    assert summary_json["selected"]["block_size"] == 16
    assert summary_json["selected"]["kv_cache_dtype"] == "bfloat16"
    assert summary_json["selected"]["max_num_batched_tokens"] == 4096
    assert summary_json["selected"]["enable_chunked_prefill"] is True
    assert summary_json["selected"]["max_cudagraph_capture_size"] == 32
    assert summary_json["metrics"]["throughput_tokens_per_sec"] is not None
    assert summary_json["metrics"]["failed_requests"] == 0
    assert summary_json["evaluated_set_fidelity"]["scope"] == "evaluated_candidates_only"
    assert summary_json["evaluated_set_fidelity"]["selected_rank"] == 1
    assert summary_json["evaluated_set_fidelity"]["selected_is_best_evaluated"] is True
    assert summary_json["selected_runtime_fingerprint"]
    assert summary_json["runtime_environment"]["backend_name"] == "vllm"
    assert "--block-size 16" in summary_json["recommended_command"]
    assert "--max-num-batched-tokens 4096" in summary_json["recommended_command"]
    assert "Recommended serve command:" in summary_text
    assert "vllm serve model-path" in summary_text
    assert "block_size: 16" in summary_text
    assert "Evaluated-set fidelity:" in summary_text
    assert "best among evaluated candidates only" in summary_text
    assert argument_caps["detection_status"] == "success"
    assert argument_caps["supported_flags"]["--block-size"] is True


def test_managed_candidate_table_uses_canonical_cuda_graph_alias(tmp_path) -> None:
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_CudaGraphAliasAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config(max_cudagraph_capture_size=32)],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))
    launch_specs = [
        json.loads(line)
        for line in (run_dir / "launch_specs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rendered_rows = [
        json.loads(line)
        for line in (run_dir / "rendered_launch_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_table = recommendation["recommendation"]["candidate_table"]

    assert summary.status == "success"
    assert launch_specs[0]["command"][launch_specs[0]["command"].index("--cuda-graph-sizes") + 1] == "32"
    assert "--max-cudagraph-capture-size" not in launch_specs[0]["command"]
    assert rendered_rows[0]["flag_aliases"] == {"max_cudagraph_capture_size": "--cuda-graph-sizes"}
    assert candidate_table[0]["max_cudagraph_capture_size"] == 32
    assert summary_json["selected"]["max_cudagraph_capture_size"] == 32
    assert "--cuda-graph-sizes 32" in summary_json["recommended_command"]


def test_workload_profile_appears_in_managed_artifacts(tmp_path) -> None:
    profile = {"profile_name": "repeated_prefix", "prefix_reuse_expected": True, "repeated_prefix_ratio": 0.75}
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config(enable_prefix_caching=True, extra={"workload_concurrency": 2, "workload_profile": profile})],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "runs" / summary.run_id
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))
    workloads = [
        json.loads(line)
        for line in (run_dir / "workload_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert managed_run["workload_profile"]["profile_name"] == "default"
    assert workloads[0]["extra"]["workload_profile"]["profile_name"] == "repeated_prefix"
    assert recommendation["recommendation"]["candidate_table"][0]["workload_profile"] == "repeated_prefix"
    assert summary_json["selected"]["workload_profile"] == "repeated_prefix"
    assert "--enable-prefix-caching" in summary_json["recommended_command"]


def test_evidence_recommendation_row_is_written(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
    )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT run_id, selected_config_id, selected_measurement_id, score, confidence, recommendation_json FROM evidence_recommendations").fetchall()

    assert len(rows) == 1
    assert rows[0][0] == summary.run_id
    assert rows[0][1] == "cfg-test"
    assert rows[0][2] == summary.selected_measurement_id
    assert rows[0][3] == summary.recommendation_score
    assert rows[0][4] == summary.recommendation_confidence
    assert "prior-only" not in rows[0][5]


def test_exact_fresh_evidence_hit_produces_managed_recommendation_without_launch(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "seed",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
    )

    second = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
        prior_provider=_NoPriorProvider(),
    )

    run_dir = tmp_path / "second" / second.run_id
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    decisions = [
        json.loads(line)
        for line in (run_dir / "evidence_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert second.cold_launch_count == 0
    assert second.workload_measurement_count == 0
    assert second.evidence_hit_candidate_count == 1
    assert second.recommendation_status == "success"
    assert recommendation["selected_source"] == "managed_evidence_hit"
    assert recommendation["selected_measurement_id"] is not None
    assert summary_json["status"] == "success"
    assert summary_json["recommendation_type"] == "exact fresh measured evidence recommendation"
    assert summary_json["recommended_command"].startswith("vllm serve model-path")
    assert decisions
    assert any(row["classification"] == "exact_fresh" and row["used_as_exact"] is True for row in decisions)
    assert managed_run["artifacts"]["evidence_decisions_jsonl"].endswith("evidence_decisions.jsonl")
    assert managed_run["evidence_decision_summary"]["used_as_exact_count"] >= 1


def test_prior_only_and_rejected_candidates_do_not_enter_managed_recommendation(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    prior_provider = _RecordingPriorProvider(candidate_id="prior-only", predicted_throughput=999.0)
    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=2,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-good"),
            _config(config_id="cfg-bad", quantization="awq"),
        ],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=prior_provider,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    candidate_table = recommendation["recommendation"]["candidate_table"]

    assert summary.failed_candidate_count == 1
    assert [row["candidate_id"] for row in candidate_table] == ["cfg-good"]
    assert "prior-only" not in json.dumps(recommendation)
    assert "cfg-bad" not in {row["candidate_id"] for row in candidate_table}


def test_evidence_recommendation_write_failure_records_warning(tmp_path, monkeypatch) -> None:
    def fail_insert_recommendation(self, record):
        del self, record
        raise RuntimeError("recommendation write failed")

    monkeypatch.setattr("serve_optimize.evidence.EvidenceStore.insert_recommendation", fail_insert_recommendation)

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config()],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=_NoPriorProvider(),
    )

    assert summary.status == "success"
    assert summary.recommendation_status == "success"
    assert any("Evidence DB recommendation write failed" in warning for warning in summary.evidence_warnings)


def test_no_successful_managed_candidates_marks_recommendation_unavailable(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    summary = run_managed_evaluation(
        backend="vllm",
        model=str(model_dir),
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: [_config(config_id="cfg-awq", quantization="awq")],
        evidence_write=False,
    )

    run_dir = tmp_path / "runs" / summary.run_id
    recommendation = json.loads((run_dir / "managed_recommendation.json").read_text(encoding="utf-8"))
    summary_json = json.loads((run_dir / "recommendation_summary.json").read_text(encoding="utf-8"))
    summary_text = (run_dir / "recommendation_summary.txt").read_text(encoding="utf-8")

    assert summary.status == "failed"
    assert summary.recommendation_status == "unavailable"
    assert summary.selected_config_id is None
    assert summary.pareto_candidate_count == 0
    assert recommendation["status"] == "unavailable"
    assert summary_json["status"] == "unavailable"
    assert summary_json["recommended_command"] == "n/a"
    assert "Recommendation: unavailable" in summary_text


def test_managed_lifecycle_launches_once_for_group_with_multiple_workloads(tmp_path) -> None:
    adapter = _CountingSuccessAdapter()
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=2,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=adapter,
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 4}),
        ],
        evidence_write=False,
    )

    run_dir = tmp_path / summary.run_id
    managed_run = json.loads((run_dir / "managed_run.json").read_text(encoding="utf-8"))
    launch_groups = json.loads((run_dir / "launch_groups.json").read_text(encoding="utf-8"))
    workloads = [
        json.loads(line)
        for line in (run_dir / "workload_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert adapter.launch_count == 1
    assert summary.cold_launch_count == 1
    assert summary.workload_measurement_count == 2
    assert summary.completed_candidate_count == 2
    assert summary.completed_candidate_count <= summary.candidate_count
    assert summary.launch_groups_count == 1
    assert summary.average_workloads_per_launch == 2.0
    assert managed_run["cold_launch_count"] == summary.cold_launch_count
    assert managed_run["workload_measurement_count"] == summary.workload_measurement_count
    assert managed_run["evidence_hit_candidate_count"] == summary.evidence_hit_candidate_count
    assert managed_run["cold_launches"] == summary.cold_launch_count
    assert managed_run["workload_measurements"] == summary.workload_measurement_count
    assert managed_run["evidence_hits"] == summary.evidence_hit_candidate_count
    assert len(launch_groups) == 1
    assert len(workloads) == 2


def test_managed_evaluation_skips_launch_on_exact_fresh_evidence(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    prior_provider = _RecordingPriorProvider()
    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "first",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[
                    PowerSampleRecord(float(index), "measured", 100.0 + index, telemetry, provider=telemetry, gpu_util_percent=70.0)
                    for index in range(5)
                ],
                warnings=[],
            )
        ),
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
    )

    second = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
        prior_provider=prior_provider,
    )

    assert second.status == "success"
    assert second.evidence_hit_candidate_count == 1
    assert second.candidates[0].status == "evidence_hit"
    assert prior_provider.call_count == 0


def test_all_group_workloads_with_exact_fresh_evidence_skip_launch(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    seed_candidates = [
        _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
        _config(config_id="cfg-b", extra={"workload_concurrency": 4}),
    ]
    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=2,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "seed",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: seed_candidates,
        evidence_db_path=db_path,
    )

    second = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=2,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: seed_candidates,
        evidence_db_path=db_path,
    )

    assert second.cold_launch_count == 0
    assert second.skipped_by_evidence_count == 2
    assert [candidate.status for candidate in second.candidates] == ["evidence_hit", "evidence_hit"]


def test_partial_group_evidence_launches_once_for_missing_workload(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "seed",
        telemetry="nvml",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [_config(config_id="cfg-a", extra={"workload_concurrency": 1})],
        evidence_db_path=db_path,
    )
    adapter = _CountingSuccessAdapter()

    second = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=2,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=adapter,
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 4}),
        ],
        evidence_db_path=db_path,
    )

    assert adapter.launch_count == 1
    assert second.skipped_by_evidence_count == 1
    assert second.workload_measurement_count == 1
    assert {candidate.status for candidate in second.candidates} == {"evidence_hit", "completed"}


def test_db_write_failure_does_not_crash_managed_evaluation(tmp_path) -> None:
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config()],
        evidence_db_path=tmp_path,
    )

    assert summary.status == "success"
    assert any("Evidence DB unavailable" in warning for warning in summary.evidence_warnings)


def test_prior_provider_called_on_evidence_miss(tmp_path) -> None:
    prior_provider = _RecordingPriorProvider()

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config()],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=prior_provider,
    )

    assert prior_provider.call_count == 1
    assert summary.ai_configurator_available is True
    assert summary.ai_configurator_used is True
    assert summary.prior_candidate_count == 1


def test_prior_artifacts_and_workload_metadata_written(tmp_path) -> None:
    prior_provider = _RecordingPriorProvider(candidate_id="cfg-test")

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config()],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=prior_provider,
    )

    run_dir = tmp_path / summary.run_id
    prior_candidates = json.loads((run_dir / "prior_candidates.json").read_text(encoding="utf-8"))
    prior_summary = json.loads((run_dir / "prior_summary.json").read_text(encoding="utf-8"))
    workloads = [
        json.loads(line)
        for line in (run_dir / "workload_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert prior_candidates[0]["source"] == PriorSource.AICONFIGURATOR.value
    assert prior_summary["ai_configurator_used"] is True
    assert workloads[0]["prior_source"] == PriorSource.AICONFIGURATOR.value
    assert summary.candidates[0].prior_source == PriorSource.AICONFIGURATOR.value


def test_prior_only_estimates_not_written_as_measured_evidence(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    prior_provider = _RecordingPriorProvider(candidate_id="cfg-test", predicted_throughput=999.0)

    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=1,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_SuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [_config()],
        evidence_db_path=db_path,
        prior_provider=prior_provider,
    )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT throughput_tokens_per_sec, is_measured, raw_json FROM evidence_measurements").fetchall()

    assert len(rows) == 1
    assert rows[0][0] != 999.0
    assert rows[0][1] == 1
    assert "predicted_throughput_tokens_per_sec" not in rows[0][2]


def test_grouping_runs_after_prior_pruning(tmp_path) -> None:
    prior_provider = _RecordingPriorProvider(candidate_id="cfg-a")

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.BALANCED,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=_CountingSuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
            _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
        ],
        evidence_db_path=tmp_path / "evidence.sqlite",
        prior_provider=prior_provider,
        prior_policy=ManagedPriorPolicy(max_prior_candidates=1, preserve_low_memory_candidate=False, preserve_diversity=False),
    )

    assert summary.candidates_after_prior_pruning == 1
    assert summary.candidates_pruned_by_prior == 2
    assert summary.launch_groups_count == 1
    assert summary.workload_measurement_count == 1


def test_staged_managed_evaluation_writes_rung_and_promotion_artifacts(tmp_path) -> None:
    adapter = _CountingSuccessAdapter()

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=adapter,
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
            _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
        ],
        evidence_db_path=tmp_path / "evidence.sqlite",
    )

    run_dir = tmp_path / summary.run_id
    rungs = json.loads((run_dir / "evaluation_rungs.json").read_text(encoding="utf-8"))
    decisions = [
        json.loads(line)
        for line in (run_dir / "promotion_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    workloads = [
        json.loads(line)
        for line in (run_dir / "workload_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [rung["name"] for rung in rungs] == ["probe", "measure", "validate"]
    assert decisions
    assert summary.budget_policy_name == "pareto_successive_halving"
    assert summary.rung_count == 3
    assert summary.probe_measurement_count == 3
    assert all(workload["rung"] in {"probe", "measure", "validate"} for workload in workloads)
    assert adapter.launch_count == summary.cold_launch_count


def test_candidates_not_promoted_are_not_measured_in_later_rungs(tmp_path) -> None:
    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=_CountingSuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
            _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
        ],
        evidence_db_path=tmp_path / "evidence.sqlite",
    )

    run_dir = tmp_path / summary.run_id
    workloads = [
        json.loads(line)
        for line in (run_dir / "workload_configs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rungs_by_candidate: dict[str, list[str]] = {}
    for workload in workloads:
        rungs_by_candidate.setdefault(workload["candidate_id"], []).append(workload["rung"])

    assert summary.pruned_after_probe_count >= 1
    assert rungs_by_candidate["cfg-c"] == ["probe"]


def test_staged_launch_group_launches_once_per_rung_group(tmp_path) -> None:
    adapter = _CountingSuccessAdapter()

    summary = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path,
        telemetry="none",
        adapter=adapter,
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
            _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
        ],
        evidence_db_path=tmp_path / "evidence.sqlite",
    )

    assert adapter.launch_count == 3
    assert summary.launch_groups_count == 3
    assert summary.probe_measurement_count == 3


def test_exact_fresh_evidence_satisfies_staged_rungs_without_launch(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    candidates = [
        _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
        _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
        _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
    ]
    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "seed",
        telemetry="nvml",
        adapter=_CountingSuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _power_collector(telemetry),
        candidate_provider=lambda: candidates,
        evidence_db_path=db_path,
    )

    second = run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "second",
        telemetry="nvml",
        adapter=_ShouldNotLaunchAdapter(),
        candidate_provider=lambda: candidates,
        evidence_db_path=db_path,
    )

    assert second.cold_launch_count == 0
    assert second.workload_measurement_count == 0
    assert second.evidence_hit_candidate_count >= 3
    assert {candidate.measured_or_evidence_source for candidate in second.candidates} == {"evidence"}


def test_staged_prior_only_estimates_do_not_enter_measured_evidence(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    prior_provider = _RecordingPriorProvider(candidate_id="cfg-c", predicted_throughput=999.0)

    run_managed_evaluation(
        backend="vllm",
        model="model-path",
        goal=Goal.PERFORMANCE,
        limit=3,
        trials=1,
        startup_timeout_s=1.0,
        cooldown_s=0.0,
        host="127.0.0.1",
        port=None,
        out_dir=tmp_path / "runs",
        telemetry="none",
        adapter=_CountingSuccessAdapter(),
        request_fn=_ok_request_with_tokens,
        candidate_provider=lambda: [
            _config(config_id="cfg-a", extra={"workload_concurrency": 1}),
            _config(config_id="cfg-b", extra={"workload_concurrency": 2}),
            _config(config_id="cfg-c", extra={"workload_concurrency": 3}),
        ],
        evidence_db_path=db_path,
        prior_provider=prior_provider,
    )

    with sqlite3.connect(db_path) as connection:
        raw_rows = [row[0] for row in connection.execute("SELECT raw_json FROM evidence_measurements").fetchall()]

    assert raw_rows
    assert all("predicted_throughput_tokens_per_sec" not in row for row in raw_rows)


def test_evidence_list_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["evidence", "list", "--help"])
    assert exc.value.code == 0


def _all_engine_caps() -> VLLMArgumentCapabilities:
    return _caps(
        "--block-size",
        "--kv-cache-dtype",
        "--enforce-eager",
        "--max-num-batched-tokens",
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
        "--max-cudagraph-capture-size",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        kv_cache_dtype_choices=("auto", "float16", "bfloat16"),
    )


def _assert_active_engine_fields_are_rendered(candidate_row: dict[str, object], command: str) -> None:
    if candidate_row.get("block_size") is not None:
        assert f"--block-size {candidate_row['block_size']}" in command
    if candidate_row.get("kv_cache_dtype") is not None:
        assert f"--kv-cache-dtype {candidate_row['kv_cache_dtype']}" in command
    if candidate_row.get("enforce_eager") is True:
        assert "--enforce-eager" in command
    if candidate_row.get("max_num_batched_tokens") is not None:
        assert f"--max-num-batched-tokens {candidate_row['max_num_batched_tokens']}" in command
    if candidate_row.get("enable_chunked_prefill") is True:
        assert "--enable-chunked-prefill" in command
    if candidate_row.get("enable_chunked_prefill") is False:
        assert "--no-enable-chunked-prefill" in command
    if candidate_row.get("max_cudagraph_capture_size") is not None:
        value = candidate_row["max_cudagraph_capture_size"]
        assert f"--max-cudagraph-capture-size {value}" in command or f"--cuda-graph-sizes {value}" in command
    if candidate_row.get("enable_prefix_caching") is True:
        assert "--enable-prefix-caching" in command
    if candidate_row.get("enable_prefix_caching") is False:
        assert "--no-enable-prefix-caching" in command


def _caps(*flags: str, kv_cache_dtype_choices: tuple[str, ...] = ()) -> VLLMArgumentCapabilities:
    option_choices = {"--kv-cache-dtype": frozenset(kv_cache_dtype_choices)} if kv_cache_dtype_choices else {}
    return VLLMArgumentCapabilities(
        executable="vllm",
        version="test",
        supported_flags=frozenset(flags),
        option_choices=option_choices,
        help_hash="test",
        detection_status="success",
    )


def _sglang_help_text() -> str:
    return """
usage: python -m sglang.launch_server [options]
  --model-path MODEL_PATH
  --host HOST
  --port PORT
  --dtype {auto,float16,bfloat16}
  --context-length CONTEXT_LENGTH
  --tp-size TP_SIZE
  --mem-fraction-static MEM_FRACTION_STATIC
  --max-running-requests MAX_RUNNING_REQUESTS
  --quantization {awq,gptq}
  --chunked-prefill-size CHUNKED_PREFILL_SIZE
  --disable-radix-cache
  --disable-cuda-graph
  --cuda-graph-max-bs CUDA_GRAPH_MAX_BS
  --served-model-name SERVED_MODEL_NAME
  --trust-remote-code
  --disable-piecewise-cuda-graph
"""


def _sglang_caps(*flags: str) -> SGLangArgumentCapabilities:
    supported_flags = frozenset(
        flags
        or (
            "--model-path",
            "--host",
            "--port",
            "--dtype",
            "--context-length",
            "--tp-size",
            "--mem-fraction-static",
            "--max-running-requests",
            "--quantization",
            "--chunked-prefill-size",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--cuda-graph-max-bs",
            "--served-model-name",
            "--trust-remote-code",
            "--disable-piecewise-cuda-graph",
        )
    )
    return SGLangArgumentCapabilities(
        executable="python",
        launch_command=("python", "-m", "sglang.launch_server"),
        version="test-sglang",
        supported_flags=supported_flags,
        option_choices={
            "--dtype": frozenset({"auto", "float16", "bfloat16"}),
            "--quantization": frozenset({"awq", "gptq"}),
        },
        help_hash="sglang-help-hash",
        detection_status="success",
    )


def _config(
    dtype: str = "fp16",
    quantization: str = "none",
    backend: str = "vllm",
    config_id: str = "cfg-test",
    max_batch_size: int = 2,
    extra: dict[str, object] | None = None,
    **kwargs,
) -> ServingConfig:
    fields = {
        "max_context_tokens": 2048,
        "kv_cache_policy": "paged",
        "scheduler": "continuous-batching",
        "tensor_parallelism": 1,
        "gpu_memory_utilization": 0.9,
    }
    fields.update(kwargs)
    return ServingConfig(
        id=config_id,
        backend=backend,
        model_id="model-path",
        dtype=dtype,
        quantization=quantization,
        max_batch_size=max_batch_size,
        extra=extra or {},
        **fields,
    )


def _synth_config(
    model: str = "model-path",
    *,
    config_id: str = "cfg-synth",
    dtype: str = "bf16",
    quantization: str = "none",
    max_batch_size: int = 1,
    max_context_tokens: int = 2048,
    extra: dict[str, object] | None = None,
    **kwargs,
) -> ServingConfig:
    synth_extra = {
        "candidate_source": SYNTHESIS_SOURCE,
        "model_native": True,
        "workload_concurrency": max_batch_size,
        "max_new_tokens": 128,
        "synthesis_rationale": "test synthesized candidate",
        "synthesis_confidence": 0.8,
        "synthesis_constraints": {"test": True},
        "synthesis_status": "proposed",
    }
    synth_extra.update(extra or {})
    return ServingConfig(
        id=config_id,
        backend="vllm",
        model_id=model,
        dtype=dtype,
        quantization=quantization,
        max_batch_size=max_batch_size,
        max_context_tokens=max_context_tokens,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
        tensor_parallelism=1,
        gpu_memory_utilization=0.9,
        extra=synth_extra,
        **kwargs,
    )


def _model_dir(tmp_path, payload: dict[str, object]):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def _managed_hardware(name: str = "Generic CUDA GPU") -> HardwareSnapshot:
    return HardwareSnapshot(
        hostname="host",
        platform="linux",
        python_version="3.12",
        detected_at="2026-01-01T00:00:00+00:00",
        gpus=[
            GpuDevice(
                index=0,
                name=name,
                uuid="GPU-1",
                total_memory_mb=80_000,
                compute_capability="9.0",
                driver_version="1",
                cuda_version="12",
            )
        ],
    )


def _handle(pid: int = 123, pgid: int = 123) -> ServerHandle:
    return ServerHandle(
        config_id="cfg-test",
        backend="vllm",
        pid=pid,
        pgid=pgid,
        host="127.0.0.1",
        port=8000,
        base_url="http://127.0.0.1:8000/v1",
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def _ok_request(_config, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=0.0,
        end_time=0.1,
        latency_s=0.1,
        status="ok",
    )


def _ok_request_with_tokens(_config, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=0.0,
        end_time=0.1,
        latency_s=0.1,
        status="ok",
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
    )


def _failed_request(_config, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=0.0,
        end_time=0.1,
        latency_s=0.1,
        status="error",
        error="not ready",
    )


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class _FakeCollector:
    def __init__(self, capture: TelemetryCapture):
        self.capture = capture

    def start(self) -> None:
        return None

    def stop(self) -> TelemetryCapture:
        return self.capture


def _power_collector(telemetry: str) -> _FakeCollector:
    return _FakeCollector(
        TelemetryCapture(
            provider=telemetry,
            samples=[
                PowerSampleRecord(float(index), "measured", 100.0 + index, telemetry, provider=telemetry, gpu_util_percent=70.0)
                for index in range(5)
            ],
            warnings=[],
        )
    )


class _SuccessAdapter:
    name = "vllm"

    def is_available(self) -> bool:
        return True

    def argument_capabilities(self) -> VLLMArgumentCapabilities:
        return _all_engine_caps()

    def build_launch_spec(self, config, *, host, port, log_dir) -> ServerLaunchSpec:
        del port
        return ServerLaunchSpec(
            config_id=config.id,
            backend="vllm",
            model_id=config.model_id,
            host=host,
            port=8000,
            base_url="http://127.0.0.1:8000/v1",
            command=vllm_command(config, host=host, port=8000, capabilities=self.argument_capabilities()),
            stdout_log_path=str(log_dir / config.id / "stdout.log"),
            stderr_log_path=str(log_dir / config.id / "stderr.log"),
        )

    def launch_server(self, spec):
        return ServerHandle(
            config_id=spec.config_id,
            backend=spec.backend,
            pid=123,
            pgid=123,
            host=spec.host,
            port=spec.port,
            base_url=spec.base_url,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def wait_for_health(self, handle, *, model, timeout_s, request_fn=None):
        del model, timeout_s, request_fn
        return HealthCheckResult(
            config_id=handle.config_id,
            backend=handle.backend,
            base_url=handle.base_url,
            healthy=True,
            status="ok",
            attempts=1,
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    def stop_server(self, handle, *, timeout_s=30.0):
        del timeout_s
        return ManagedLifecycleRecord(
            run_id="",
            config_id=handle.config_id,
            backend=handle.backend,
            event="stop",
            status="stopped",
            timestamp=datetime.now(timezone.utc).isoformat(),
            pid=handle.pid,
            pgid=handle.pgid,
        )


class _SuccessSglangAdapter:
    name = "sglang"

    def is_available(self) -> bool:
        return True

    def argument_capabilities(self) -> SGLangArgumentCapabilities:
        return _sglang_caps()

    def backend_metadata(self) -> dict[str, object]:
        return {
            "adapter": "sglang",
            "backend": "sglang",
            "version": "test-sglang",
            "argument_detection_status": "success",
            "argument_capabilities_help_hash": "sglang-help-hash",
        }

    def build_launch_spec(self, config, *, host, port, log_dir) -> ServerLaunchSpec:
        del port
        rendered = render_sglang_launch(config, host=host, port=8001, capabilities=self.argument_capabilities())
        return ServerLaunchSpec(
            config_id=config.id,
            backend="sglang",
            model_id=config.model_id,
            host=host,
            port=8001,
            base_url="http://127.0.0.1:8001/v1",
            command=rendered.command,
            stdout_log_path=str(log_dir / config.id / "stdout.log"),
            stderr_log_path=str(log_dir / config.id / "stderr.log"),
            metadata={"rendered_launch": rendered.to_metadata(), **self.backend_metadata()},
        )

    def launch_server(self, spec):
        return ServerHandle(
            config_id=spec.config_id,
            backend=spec.backend,
            pid=124,
            pgid=124,
            host=spec.host,
            port=spec.port,
            base_url=spec.base_url,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def wait_for_health(self, handle, *, model, timeout_s, request_fn=None):
        del model, timeout_s, request_fn
        return HealthCheckResult(
            config_id=handle.config_id,
            backend=handle.backend,
            base_url=handle.base_url,
            healthy=True,
            status="ok",
            attempts=1,
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    def stop_server(self, handle, *, timeout_s=30.0):
        del timeout_s
        return ManagedLifecycleRecord(
            run_id="",
            config_id=handle.config_id,
            backend=handle.backend,
            event="stop",
            status="stopped",
            timestamp=datetime.now(timezone.utc).isoformat(),
            pid=handle.pid,
            pgid=handle.pgid,
        )


class _CountingSuccessAdapter(_SuccessAdapter):
    def __init__(self) -> None:
        self.launch_count = 0

    def launch_server(self, spec):
        self.launch_count += 1
        return super().launch_server(spec)


class _NoOptionalEngineAdapter(_SuccessAdapter):
    def argument_capabilities(self) -> VLLMArgumentCapabilities:
        return _caps()


class _CudaGraphAliasAdapter(_SuccessAdapter):
    def argument_capabilities(self) -> VLLMArgumentCapabilities:
        return _caps("--cuda-graph-sizes")


class _RecordingPriorProvider:
    source = PriorSource.AICONFIGURATOR.value

    def __init__(self, candidate_id: str = "prior-only", predicted_throughput: float = 123.0) -> None:
        self.candidate_id = candidate_id
        self.predicted_throughput = predicted_throughput
        self.call_count = 0

    def collect_priors(self, *, model, backend, goal, candidates, out_dir):
        del model, backend, goal, candidates, out_dir
        self.call_count += 1
        return PriorResult(
            source=self.source,
            available=True,
            used=True,
            candidates=[
                PriorCandidate(
                    source=self.source,
                    candidate_id=self.candidate_id,
                    config_id=self.candidate_id,
                    confidence=0.8,
                    predicted_throughput_tokens_per_sec=self.predicted_throughput,
                    notes=["fake prior"],
                )
            ],
        )


class _NoPriorProvider:
    source = PriorSource.UNAVAILABLE.value

    def collect_priors(self, *, model, backend, goal, candidates, out_dir):
        del model, backend, goal, candidates, out_dir
        return PriorResult(source=self.source, available=False, used=False)


class _StaticSynthesisProvider:
    source = SYNTHESIS_SOURCE

    def __init__(self, candidates: list[ServingConfig], *, available: bool = True, warning: str | None = None) -> None:
        self.candidates = candidates
        self.available = available
        self.warning = warning
        self.call_count = 0

    def synthesize(self, *, context, out_dir):
        del context, out_dir
        self.call_count += 1
        return CandidateSynthesisResult(
            source=self.source,
            available=self.available,
            used=bool(self.candidates),
            candidates=self.candidates,
            rationale="test synthesis",
            confidence=0.8 if self.candidates else None,
            constraints_used={"test": True},
            warnings=[self.warning] if self.warning else [],
            skipped_reason=None if self.candidates else "test no candidates",
        )


class _ShouldNotLaunchAdapter(_SuccessAdapter):
    def is_available(self) -> bool:
        return False

    def launch_server(self, spec):
        del spec
        raise AssertionError("launch should be skipped on exact fresh evidence")


class _LaunchFailAdapter:
    name = "vllm"

    def is_available(self) -> bool:
        return True

    def build_launch_spec(self, config, *, host, port, log_dir) -> ServerLaunchSpec:
        return ServerLaunchSpec(
            config_id=config.id,
            backend="vllm",
            model_id=config.model_id,
            host=host,
            port=8000,
            base_url="http://127.0.0.1:8000/v1",
            command=["vllm", "serve", config.model_id],
            stdout_log_path=str(log_dir / config.id / "stdout.log"),
            stderr_log_path=str(log_dir / config.id / "stderr.log"),
        )

    def launch_server(self, spec):
        del spec
        raise RuntimeError("boom")

    def wait_for_health(self, handle, *, model, timeout_s, request_fn=None):
        raise AssertionError("health should not run after failed launch")

    def stop_server(self, handle, *, timeout_s=30.0):
        raise AssertionError("stop should not run after failed launch")
