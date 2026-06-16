"""Real tiny-model benchmarking through local inference backends."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .schemas import BenchmarkResult, HardwareSnapshot, ModelSpec, ServingConfig

DEFAULT_PROMPTS = [
    "Energy-aware inference means",
    "The fastest server configuration is",
    "A GPU power profiler should measure",
    "For a small production LLM service,",
]


@dataclass(frozen=True)
class RealBenchmarkOptions:
    prompts: list[str]
    max_new_tokens: int = 16
    trial: int = 0
    device: str = "auto"
    cache_dir: Path | None = None


def run_transformers_benchmark(
    config: ServingConfig,
    hardware: HardwareSnapshot,
    model: ModelSpec,
    options: RealBenchmarkOptions | None = None,
) -> BenchmarkResult:
    options = options or RealBenchmarkOptions(prompts=DEFAULT_PROMPTS)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime
        return _failed(config, f"Missing runtime dependency: {exc}")

    device = _choose_device(torch, options.device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    if config.dtype == "fp32":
        dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=str(options.cache_dir) if options.cache_dir else None)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                cache_dir=str(options.cache_dir) if options.cache_dir else None,
                dtype=dtype,
            )
        except TypeError:
            model_obj = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                cache_dir=str(options.cache_dir) if options.cache_dir else None,
                torch_dtype=dtype,
            )
        model_obj.to(device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = tokenizer.eos_token_id
        if getattr(model_obj.generation_config, "pad_token_id", None) is None or model_obj.generation_config.pad_token_id < 0:
            model_obj.generation_config.pad_token_id = pad_token_id
        if getattr(model_obj.config, "pad_token_id", None) is None or model_obj.config.pad_token_id < 0:
            model_obj.config.pad_token_id = pad_token_id
        model_obj.eval()
    except Exception as exc:
        return _failed(config, f"Model load failed: {exc.__class__.__name__}: {exc}")

    prompts = options.prompts[: max(1, config.max_batch_size)]
    if not prompts:
        prompts = DEFAULT_PROMPTS[:1]

    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_tokens = int(encoded["attention_mask"].sum().item())

        with torch.inference_mode():
            _ = model_obj.generate(**encoded, max_new_tokens=1, do_sample=False)
            if device == "cuda":
                torch.cuda.synchronize()

        sampler = NvidiaSmiPowerTrace(interval_s=0.2)
        sampler.start()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model_obj.generate(
                **encoded,
                max_new_tokens=options.max_new_tokens,
                do_sample=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
        end = time.perf_counter()
        samples = sampler.stop()
    except Exception as exc:
        return _failed(config, f"Inference failed: {exc.__class__.__name__}: {exc}")

    duration_s = max(end - start, 1e-9)
    total_tokens = int(output.numel())
    generated_tokens = max(1, total_tokens - input_tokens)
    throughput = generated_tokens / duration_s
    power_summary = summarize_samples(samples)
    average_power = power_summary["average_watts"] or _fallback_power(hardware)
    energy = average_power * duration_s
    joules_per_token = energy / generated_tokens

    return BenchmarkResult(
        config=config,
        throughput_tok_s=round(throughput, 6),
        average_power_watts=round(average_power, 6),
        joules_per_token=round(joules_per_token, 9),
        tokens_per_watt=round(throughput / max(average_power, 1e-9), 9),
        peak_power_watts=round(power_summary["peak_watts"], 6) if power_summary["peak_watts"] is not None else None,
        total_energy_joules=round(energy, 6),
        generated_tokens=generated_tokens,
        raw={
            "mode": "transformers",
            "energy_method": "tokenpowerbench-style-nvidia-smi-sampling",
            "device": device,
            "duration_s": duration_s,
            "input_tokens": input_tokens,
            "output_tokens_total": total_tokens,
            "trial": options.trial,
            "power_sample_count": power_summary["count"],
            "power_source": "nvidia-smi" if samples else "hardware_snapshot_fallback",
        },
    )


def make_transformers_configs(model_id: str, batch_sizes: list[int] | None = None, max_context_tokens: int = 1024) -> list[ServingConfig]:
    batch_sizes = batch_sizes or [1, 2, 4]
    return [
        ServingConfig(
            id=f"tf-{model_id.replace('/', '--')}-bs{batch}",
            backend="transformers",
            model_id=model_id,
            dtype="fp16",
            quantization="none",
            max_batch_size=batch,
            max_context_tokens=max_context_tokens,
            kv_cache_policy="backend-default",
            scheduler="eager-generate",
            notes=["Reference local backend for functional validation."],
        )
        for batch in batch_sizes
    ]


def make_vllm_configs(model_id: str, batch_sizes: list[int] | None = None, max_context_tokens: int = 512) -> list[ServingConfig]:
    batch_sizes = batch_sizes or [1, 2]
    return [
        ServingConfig(
            id=f"vllm-{model_id.replace('/', '--')}-bs{batch}",
            backend="vllm",
            model_id=model_id,
            dtype="fp16",
            quantization="none",
            max_batch_size=batch,
            max_context_tokens=max_context_tokens,
            kv_cache_policy="paged",
            scheduler="continuous-batching",
            gpu_memory_utilization=0.55,
            notes=["Offline vLLM engine benchmark for functional validation."],
        )
        for batch in batch_sizes
    ]


def run_vllm_benchmark(
    config: ServingConfig,
    hardware: HardwareSnapshot,
    model: ModelSpec,
    options: RealBenchmarkOptions | None = None,
) -> BenchmarkResult:
    options = options or RealBenchmarkOptions(prompts=DEFAULT_PROMPTS)
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - optional runtime
        return _failed(config, f"Missing vLLM dependency: {exc}")

    prompts = options.prompts[: max(1, config.max_batch_size)] or DEFAULT_PROMPTS[:1]
    try:
        model_ref = config.model_id
        if options.cache_dir is not None:
            from .model_store import download_model

            model_ref = download_model(config.model_id, cache_dir=options.cache_dir).path
        llm = LLM(
            model=model_ref,
            dtype="float16",
            max_model_len=config.max_context_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=False,
            disable_log_stats=True,
        )
        warmup_params = SamplingParams(max_tokens=1, temperature=0.0)
        _ = llm.generate(prompts[:1], warmup_params, use_tqdm=False)
        sampling_params = SamplingParams(max_tokens=options.max_new_tokens, temperature=0.0)
        sampler = NvidiaSmiPowerTrace(interval_s=0.2)
        sampler.start()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        end = time.perf_counter()
        samples = sampler.stop()
    except Exception as exc:
        return _failed(config, f"vLLM inference failed: {exc.__class__.__name__}: {exc}")

    duration_s = max(end - start, 1e-9)
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    generated_tokens = max(1, generated_tokens)
    throughput = generated_tokens / duration_s
    power_summary = summarize_samples(samples)
    average_power = power_summary["average_watts"] or _fallback_power(hardware)
    energy = average_power * duration_s
    joules_per_token = energy / generated_tokens

    return BenchmarkResult(
        config=config,
        throughput_tok_s=round(throughput, 6),
        average_power_watts=round(average_power, 6),
        joules_per_token=round(joules_per_token, 9),
        tokens_per_watt=round(throughput / max(average_power, 1e-9), 9),
        peak_power_watts=round(power_summary["peak_watts"], 6) if power_summary["peak_watts"] is not None else None,
        total_energy_joules=round(energy, 6),
        generated_tokens=generated_tokens,
        raw={
            "mode": "vllm-offline",
            "energy_method": "tokenpowerbench-style-nvidia-smi-sampling",
            "duration_s": duration_s,
            "trial": options.trial,
            "power_sample_count": power_summary["count"],
            "power_source": "nvidia-smi" if samples else "hardware_snapshot_fallback",
        },
    )


class NvidiaSmiPowerTrace:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._samples

    def _run(self) -> None:
        while not self._stop.is_set():
            watts = query_nvidia_smi_power()
            if watts is not None:
                self._samples.append(watts)
            time.sleep(self.interval_s)


def query_nvidia_smi_power() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    first = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    try:
        return float(first)
    except ValueError:
        return None


def summarize_samples(samples: list[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "average_watts": None, "peak_watts": None}
    return {"count": len(samples), "average_watts": sum(samples) / len(samples), "peak_watts": max(samples)}


def _choose_device(torch_module: object, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _fallback_power(hardware: HardwareSnapshot) -> float:
    gpu = hardware.best_gpu
    if gpu and gpu.current_power_watts:
        return gpu.current_power_watts
    return 1.0


def _failed(config: ServingConfig, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        config=config,
        throughput_tok_s=0.0,
        average_power_watts=0.0,
        joules_per_token=0.0,
        tokens_per_watt=0.0,
        feasible=False,
        reason=reason,
    )
