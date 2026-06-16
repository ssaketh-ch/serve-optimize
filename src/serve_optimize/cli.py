"""Command line interface for Serve Optimize."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from .aiconfig_parser import parse_aiconfig_prediction_csv, parse_aiconfigurator_best_configs
from .aiconfig_plans import (
    candidate_to_endpoint_benchmark_plan,
    candidate_to_evaluation_plan,
    candidate_to_vllm_serve_plan,
)
from .aiconfigurator_bridge import run_aiconfigurator
from .backend_status import INSTALLATION_PROFILES, check_backend_status, check_installation_profile
from .backends.factory import MANAGED_BACKEND_CHOICES
from .benchmark import run_dry_benchmark
from .candidates import generate_candidates
from .endpoint_benchmark import DEFAULT_ENDPOINT_PROMPT, make_run_id, run_endpoint_benchmark
from .evaluation import run_evaluation_plan_dir
from .evidence import DEFAULT_EVIDENCE_DB_PATH, list_evidence_measurements
from .hardware import detect_hardware
from .io import write_json, write_jsonl
from .landscape import grouped_landscape
from .managed import build_managed_preflight, run_managed_evaluation
from .model_store import TINY_MODEL_IDS, download_model, download_tiny_models
from .modeling import infer_model_spec
from .pareto import pareto_frontier, select_recommendation
from .real_benchmark import (
    DEFAULT_PROMPTS,
    RealBenchmarkOptions,
    make_transformers_configs,
    make_vllm_configs,
    run_transformers_benchmark,
    run_vllm_benchmark,
)
from .recommendation import build_attach_preflight, recommend_attach_mode
from .release_check import write_release_check_artifacts
from .repeatability import write_repeatability_artifacts
from .reporting import RichReporter, RichTelemetryCheckReporter, format_recommendation_report
from .research_package import write_research_package_artifacts
from .schemas import EndpointBenchmarkConfig, Goal, Recommendation, RecommendationGoal, to_dict
from .telemetry_check import run_telemetry_check
from .validation_campaign import write_validation_campaign_artifacts
from .workloads import load_workload_profile, workload_profile_choices


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="serve-optimize", description="Energy-aware LLM inference configuration optimizer.")
    parser.add_argument("--version", action="version", version=f"serve-optimize {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Detect local GPU and MIG hardware.")
    detect.add_argument("--json", action="store_true", help="Emit JSON instead of a readable summary.")
    detect.set_defaults(func=_cmd_detect)

    doctor = subparsers.add_parser("doctor", help="Check installed inference, telemetry, and optimizer dependencies.")
    doctor.add_argument("--json", action="store_true", help="Emit JSON.")
    doctor.add_argument(
        "--profile",
        choices=INSTALLATION_PROFILES,
        default=None,
        help="Validate one reproducible installation profile.",
    )
    doctor.set_defaults(func=_cmd_doctor)

    landscape = subparsers.add_parser("landscape", help="Print the related literature and tooling landscape.")
    landscape.add_argument("--json", action="store_true", help="Emit JSON.")
    landscape.set_defaults(func=_cmd_landscape)

    prepare = subparsers.add_parser("prepare-models", help="Download tiny models for functional smoke tests.")
    prepare.add_argument("--model", action="append", dest="models", help="Model id to download. Can be passed multiple times.")
    prepare.add_argument("--cache-dir", type=Path, default=Path("data/models"), help="Model cache directory.")
    prepare.add_argument("--json", action="store_true", help="Emit JSON.")
    prepare.set_defaults(func=_cmd_prepare_models)

    aiconfig = subparsers.add_parser("aiconfig", help="Run AIConfigurator support/generate/estimate for a target system.")
    aiconfig.add_argument("--mode", choices=["support", "generate", "estimate"], default="support")
    aiconfig.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    aiconfig.add_argument("--system", default="h200_sxm", help="AIConfigurator system name, for example h200_sxm.")
    aiconfig.add_argument("--backend", choices=["vllm", "sglang", "trtllm", "all"], default="vllm")
    aiconfig.add_argument("--isl", type=int, default=1024, help="Input sequence length for estimate mode.")
    aiconfig.add_argument("--osl", type=int, default=128, help="Output sequence length for estimate mode.")
    aiconfig.add_argument("--batch-size", type=int, default=16, help="Batch size for estimate mode.")
    aiconfig.add_argument("--total-gpus", type=int, default=1, help="Total GPUs for generate mode.")
    aiconfig.add_argument("--output-dir", type=Path, default=Path("results/aiconfigurator"), help="Output directory.")
    aiconfig.add_argument("--json", action="store_true", help="Emit JSON.")
    aiconfig.set_defaults(func=_cmd_aiconfig)

    candidates = subparsers.add_parser("candidates", help="Generate candidate serving configurations.")
    _add_model_args(candidates)
    _add_goal_arg(candidates)
    candidates.add_argument("--limit", type=int, default=24, help="Maximum number of candidates to print.")
    candidates.add_argument("--json", action="store_true", help="Emit JSON.")
    candidates.set_defaults(func=_cmd_candidates)

    optimize = subparsers.add_parser("optimize", help="Generate, benchmark, and recommend a configuration.")
    _add_model_args(optimize)
    _add_goal_arg(optimize)
    optimize.add_argument("--limit", type=int, default=36, help="Maximum number of candidates to evaluate.")
    optimize.add_argument(
        "--backend",
        choices=["synthetic", "transformers", "vllm"],
        default="synthetic",
        help="Benchmark backend to use for optimization.",
    )
    optimize.add_argument("--dry-run", action="store_true", help="Use the synthetic benchmark runner.")
    optimize.add_argument("--trials", type=int, default=1, help="Number of repeated trials for real benchmark backends.")
    optimize.add_argument("--max-new-tokens", type=int, default=16, help="Generated tokens per prompt for real benchmark backends.")
    optimize.add_argument("--cache-dir", type=Path, default=Path("data/models"), help="Model cache directory.")
    optimize.add_argument("--output-dir", type=Path, default=Path("results/synthetic"), help="Directory for JSON artifacts.")
    optimize.add_argument("--json", action="store_true", help="Emit JSON.")
    optimize.set_defaults(func=_cmd_optimize)

    benchmark = subparsers.add_parser("benchmark", help="Run the current synthetic benchmark path.")
    _add_model_args(benchmark)
    _add_goal_arg(benchmark)
    benchmark.add_argument("--limit", type=int, default=12, help="Maximum number of candidates to benchmark.")
    benchmark.add_argument("--output", type=Path, default=Path("results/synthetic/results.jsonl"), help="JSONL output path.")
    benchmark.set_defaults(func=_cmd_benchmark)

    plan = subparsers.add_parser("plan-from-aic", help="Generate serve and benchmark plans from AIConfigurator CSV output.")
    plan.add_argument("--best-config-csv", type=Path, required=True, help="Path to AIConfigurator best_config_topn.csv.")
    plan.add_argument("--top-k", type=int, default=None, help="Limit to the first K ranked candidates.")
    plan.add_argument("--host", default="127.0.0.1", help="Host for generated vLLM serve plans.")
    plan.add_argument("--port", type=int, default=8080, help="Port for generated vLLM serve plans.")
    plan.add_argument("--base-url", required=True, help="Base URL for generated endpoint benchmark plans.")
    plan.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory utilization flag.")
    plan.add_argument("--out", type=Path, default=Path("results/plans"), help="Output root directory.")
    plan.set_defaults(func=_cmd_plan_from_aic)

    run_plan = subparsers.add_parser("run-evaluation-plan", help="Run endpoint benchmarks from generated evaluation plans.")
    run_plan.add_argument("--plan-dir", type=Path, required=True, help="Directory produced by plan-from-aic.")
    run_plan.add_argument("--out", type=Path, default=Path("results/evaluations"), help="Evaluation output root directory.")
    run_plan.add_argument("--limit-candidates", type=int, default=None, help="Limit the number of candidates to evaluate.")
    run_plan.add_argument("--override-concurrency", type=int, default=None, help="Override benchmark concurrency.")
    run_plan.add_argument("--override-num-requests", type=int, default=None, help="Override request count per candidate.")
    run_plan.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    run_plan.add_argument(
        "--telemetry",
        choices=["none", "nvml", "nvidia-smi", "auto"],
        default="none",
        help="Optional host-side telemetry provider.",
    )
    run_plan.set_defaults(func=_cmd_run_evaluation_plan)

    recommend = subparsers.add_parser("recommend", help="Attach to a running endpoint, evaluate candidates, and recommend a measured configuration.")
    recommend.add_argument("--base-url", required=True, help="Base URL, for example http://127.0.0.1:8080/v1.")
    recommend.add_argument("--model", required=True, help="Model name or path used for candidate generation and endpoint requests.")
    recommend.add_argument("--backend", required=True, help="Candidate backend label, for example vllm or sglang.")
    recommend.add_argument("--system", required=True, help="Target system label for candidate generation, for example local_gpu.")
    recommend.add_argument("--total-gpus", type=int, required=True, help="Total GPUs for candidate generation inputs.")
    recommend.add_argument("--isl", type=int, required=True, help="Expected input sequence length.")
    recommend.add_argument("--osl", type=int, required=True, help="Expected output sequence length.")
    recommend.add_argument("--ttft", type=float, default=None, help="Target TTFT in milliseconds for AIConfigurator candidate generation.")
    recommend.add_argument("--tpot", type=float, default=None, help="Target TPOT in milliseconds for AIConfigurator candidate generation.")
    recommend.add_argument(
        "--candidate-source",
        choices=["aiconfigurator", "heuristic", "sweep", "auto"],
        default="auto",
        help="Candidate source to use before benchmarking.",
    )
    recommend.add_argument("--top-k", type=int, default=4, help="Maximum number of candidates to evaluate.")
    recommend.add_argument(
        "--concurrency-sweep",
        default="16,32,64,128,256,512",
        help="Comma-separated Attach Mode concurrency sweep values.",
    )
    recommend.add_argument("--disable-sweep", action="store_true", help="Disable sweep candidates when candidate-source=auto.")
    recommend.add_argument(
        "--goal",
        choices=[goal.value for goal in RecommendationGoal],
        default=RecommendationGoal.BALANCED.value,
        help="Attach Mode recommendation goal.",
    )
    recommend.add_argument(
        "--telemetry",
        choices=["none", "nvml", "nvidia-smi", "auto"],
        default="auto",
        help="Optional host-side telemetry provider.",
    )
    recommend.add_argument("--format", choices=["text", "json"], default="text", help="Terminal output format.")
    recommend.add_argument("--quiet", action="store_true", help="Print only the selected candidate and artifact path.")
    recommend.add_argument("--override-concurrency", type=int, default=None, help="Override benchmark concurrency for every candidate.")
    recommend.add_argument("--override-num-requests", type=int, default=None, help="Override request count for every candidate.")
    recommend.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    recommend.add_argument("--allow-efficiency-fallback", action="store_true", help="Allow efficiency goal to fall back to balanced scoring when power telemetry is unavailable.")
    recommend.add_argument("--dry-run", action="store_true", help="Write a preflight plan without endpoint health checks or benchmarks.")
    _add_workload_args(recommend)
    recommend.add_argument("--out", type=Path, default=Path("results/recommendations"), help="Recommendation output root directory.")
    recommend.set_defaults(func=_cmd_recommend)

    managed = subparsers.add_parser("managed-evaluate", help="Launch, benchmark, and stop candidate servers.")
    managed.add_argument(
        "--backend",
        choices=MANAGED_BACKEND_CHOICES,
        default="vllm",
        help="Managed backend to launch. vLLM and SGLang are supported; vLLM remains the default.",
    )
    managed.add_argument("--model", required=True, help="Model name or path to serve and benchmark.")
    managed.add_argument(
        "--goal",
        choices=[goal.value for goal in Goal],
        default=Goal.BALANCED.value,
        help="Candidate generation goal.",
    )
    managed.add_argument("--limit", type=int, default=4, help="Maximum number of managed candidates to evaluate.")
    managed.add_argument("--trials", type=int, default=1, help="Benchmark trials per launched candidate.")
    managed.add_argument("--startup-timeout", type=float, default=300.0, help="Seconds to wait for the launched server to become healthy.")
    managed.add_argument("--cooldown-seconds", type=float, default=5.0, help="Seconds to wait after stopping each candidate server.")
    managed.add_argument("--host", default="127.0.0.1", help="Host passed to the backend server.")
    managed.add_argument("--port", type=int, default=None, help="Port passed to the backend server. Omit to allocate one per candidate.")
    managed.add_argument(
        "--telemetry",
        choices=["none", "nvml", "nvidia-smi", "auto"],
        default="auto",
        help="Optional host-side telemetry provider during endpoint benchmarks.",
    )
    managed.add_argument(
        "--evidence-db",
        type=Path,
        default=DEFAULT_EVIDENCE_DB_PATH,
        help="SQLite evidence database for measured managed evaluation results.",
    )
    managed.add_argument("--no-evidence-write", action="store_true", help="Do not create or write the evidence database.")
    managed.add_argument("--evidence-freshness-hours", type=float, default=168.0, help="Freshness window for exact measured evidence hits.")
    managed.add_argument("--dry-run", action="store_true", help="Write a preflight plan without launching servers, health checks, benchmarks, or evidence writes.")
    _add_workload_args(managed)
    _add_measurement_quality_args(managed)
    managed.add_argument("--out", type=Path, default=Path("results/managed"), help="Managed evaluation output root directory.")
    managed.set_defaults(func=_cmd_managed_evaluate)

    repeatability = subparsers.add_parser("repeatability", help="Compare managed recommendation repeatability across run directories.")
    repeatability.add_argument("run_dirs", nargs="+", type=Path, help="Managed run directories to compare.")
    repeatability.set_defaults(func=_cmd_repeatability)

    validate_campaign = subparsers.add_parser(
        "validate-campaign",
        help="Validate recommendation quality across existing managed run directories.",
    )
    validate_campaign.add_argument("run_dirs", nargs="+", type=Path, help="Managed run directories to analyze.")
    validate_campaign.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/validation-campaign/<timestamp>.",
    )
    validate_campaign.set_defaults(func=_cmd_validate_campaign)

    release_check = subparsers.add_parser("release-check", help="Run local release readiness checks.")
    release_check.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to inspect.")
    release_check.add_argument("--out", type=Path, default=Path("results/release-check"), help="Output directory.")
    release_check.add_argument("--json", action="store_true", help="Emit JSON.")
    release_check.set_defaults(func=_cmd_release_check)

    research_package = subparsers.add_parser(
        "research-package",
        help="Package existing managed run artifacts for research analysis.",
    )
    research_package.add_argument("run_dirs", nargs="+", type=Path, help="Managed run directories to package.")
    research_package.add_argument("--out", type=Path, default=Path("results/research-package"), help="Output directory.")
    research_package.add_argument("--json", action="store_true", help="Emit JSON.")
    research_package.set_defaults(func=_cmd_research_package)

    evidence = subparsers.add_parser("evidence", help="Inspect measured evidence stored by Serve Optimize.")
    evidence_subparsers = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_list = evidence_subparsers.add_parser("list", help="List recent measured evidence rows.")
    evidence_list.add_argument("--db", type=Path, default=DEFAULT_EVIDENCE_DB_PATH, help="SQLite evidence database path.")
    evidence_list.add_argument("--limit", type=int, default=20, help="Maximum number of evidence rows to print.")
    evidence_list.set_defaults(func=_cmd_evidence_list)

    endpoint = subparsers.add_parser("endpoint-bench", help="Benchmark an already-running OpenAI-compatible endpoint.")
    endpoint.add_argument("--base-url", required=True, help="Base URL, for example http://127.0.0.1:8080/v1.")
    endpoint.add_argument("--model", required=True, help="Model name or path served by the endpoint.")
    endpoint.add_argument("--concurrency", type=int, default=1, help="Concurrent in-flight requests.")
    endpoint.add_argument("--num-requests", type=int, default=16, help="Total requests to send.")
    endpoint.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per request.")
    endpoint.add_argument("--prompt", default=None, help="Prompt text. Defaults to a generated validation prompt.")
    endpoint.add_argument("--prompt-file", type=Path, default=None, help="Path to a UTF-8 prompt file.")
    endpoint.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    _add_measurement_quality_args(endpoint)
    endpoint.add_argument("--prediction-csv", type=Path, default=None, help="AIConfigurator best_config_topn.csv or pareto.csv.")
    endpoint.add_argument("--use-aic-concurrency", action="store_true", help="Use concurrency from the prediction CSV top row.")
    endpoint.add_argument(
        "--telemetry",
        choices=["none", "nvml", "nvidia-smi", "auto"],
        default="none",
        help="Optional host-side telemetry provider.",
    )
    endpoint.add_argument("--device-index", type=int, default=0, help="GPU index for telemetry.")
    endpoint.add_argument("--out", type=Path, default=Path("results/endpoint_runs"), help="Output root directory.")
    endpoint.set_defaults(func=_cmd_endpoint_bench)

    telemetry_check = subparsers.add_parser("telemetry-check", help="Collect telemetry samples without running an endpoint benchmark.")
    telemetry_check.add_argument(
        "--telemetry",
        choices=["none", "nvml", "nvidia-smi", "auto"],
        default="auto",
        help="Telemetry provider to validate.",
    )
    telemetry_check.add_argument("--duration", type=float, default=15.0, help="Sampling duration in seconds.")
    telemetry_check.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds.")
    telemetry_check.add_argument("--device-index", type=int, default=0, help="Device index for provider queries.")
    telemetry_check.add_argument(
        "--with-nvidia-smi-loop",
        action="store_true",
        help="Record a TODO note for future provider comparison diagnostics.",
    )
    telemetry_check.add_argument("--out", type=Path, default=Path("results/telemetry_checks"), help="Telemetry check output root directory.")
    telemetry_check.set_defaults(func=_cmd_telemetry_check)

    smoke = subparsers.add_parser("smoke", help="Download tiny models and run repeated functional benchmarks.")
    smoke.add_argument("--model", action="append", dest="models", help="Model id to test. Defaults to two tiny models.")
    smoke.add_argument("--backend", choices=["transformers", "vllm"], default="transformers", help="Functional smoke backend.")
    smoke.add_argument("--trials", type=int, default=2, help="Trials per candidate.")
    smoke.add_argument("--max-new-tokens", type=int, default=16, help="Generated tokens per prompt.")
    smoke.add_argument("--cache-dir", type=Path, default=Path("data/models"), help="Model cache directory.")
    smoke.add_argument("--output-dir", type=Path, default=Path("results/real-smoke"), help="Directory for JSON artifacts.")
    smoke.add_argument("--json", action="store_true", help="Emit JSON.")
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model name or Hugging Face id, for example mistral-7b.")
    parser.add_argument("--max-context", type=int, default=None, help="Override model maximum context tokens.")


def _add_goal_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--goal",
        choices=[goal.value for goal in Goal],
        default=Goal.BALANCED.value,
        help="Optimization goal.",
    )


def _add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workload-profile",
        choices=workload_profile_choices(),
        default="default",
        help="Synthetic workload profile preset.",
    )
    parser.add_argument("--workload-manifest", type=Path, default=None, help="JSON workload manifest path.")
    parser.add_argument("--slo-ttft-ms", type=float, default=None, help="Maximum TTFT in milliseconds for recommendation eligibility.")
    parser.add_argument("--slo-tpot-ms", type=float, default=None, help="Maximum TPOT in milliseconds for recommendation eligibility.")
    parser.add_argument("--slo-p95-latency-ms", type=float, default=None, help="Maximum p95 latency in milliseconds for recommendation eligibility.")
    parser.add_argument(
        "--slo-min-throughput-tokens-per-sec",
        type=float,
        default=None,
        help="Minimum total token throughput for recommendation eligibility.",
    )
    parser.add_argument(
        "--slo-max-failed-request-rate",
        type=float,
        default=None,
        help="Maximum failed request ratio from 0.0 to 1.0 for recommendation eligibility.",
    )


def _add_measurement_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup-requests", type=int, default=0, help="Successful requests to exclude from measured summary metrics.")
    parser.add_argument("--steady-state-seconds", type=float, default=None, help="Measured steady state window duration in seconds.")
    parser.add_argument("--idle-baseline-seconds", type=float, default=0.0, help="Seconds to sample idle power before active requests.")
    parser.add_argument("--idle-power-watts", type=float, default=None, help="Known idle power baseline in watts for idle subtracted energy.")


def _cmd_detect(args: argparse.Namespace) -> None:
    hardware = detect_hardware()
    _emit(hardware, as_json=args.json)


def _cmd_doctor(args: argparse.Namespace) -> None:
    statuses = (
        check_installation_profile(args.profile)
        if args.profile
        else check_backend_status()
    )
    if args.json:
        print(json.dumps(to_dict(statuses), indent=2, sort_keys=True))
        if args.profile and any(not status.available for status in statuses):
            raise SystemExit(1)
        return
    title = f"Serve Optimize {args.profile} profile check" if args.profile else "Serve Optimize dependency check"
    print(title)
    for status in statuses:
        state = "ok" if status.available else "missing"
        detail = status.version or status.command or status.reason or ""
        print(f"  {status.name:<18} {state:<8} {detail}")
    if args.profile and any(not status.available for status in statuses):
        raise SystemExit(1)


def _cmd_landscape(args: argparse.Namespace) -> None:
    payload = grouped_landscape()
    if args.json:
        print(json.dumps(to_dict(payload), indent=2, sort_keys=True))
        return
    print("Serve Optimize landscape")
    for category, items in payload.items():
        print(f"\n{category}")
        for item in items:
            print(f"  - {item.name} [{item.priority}]: {item.relevance}")
            print(f"    {item.url}")


def _cmd_prepare_models(args: argparse.Namespace) -> None:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.models:
        models = [download_model(model_id, cache_dir=args.cache_dir) for model_id in args.models]
    else:
        models = download_tiny_models(cache_dir=args.cache_dir)
    if args.json:
        print(json.dumps(to_dict(models), indent=2, sort_keys=True))
        return
    print("Downloaded models")
    for model in models:
        print(f"  {model.model_id}: {model.path}")


def _cmd_aiconfig(args: argparse.Namespace) -> None:
    run = run_aiconfigurator(
        mode=args.mode,
        model=args.model,
        system=args.system,
        backend=args.backend,
        output_dir=args.output_dir,
        isl=args.isl,
        osl=args.osl,
        batch_size=args.batch_size,
        total_gpus=args.total_gpus,
    )
    if args.json:
        print(json.dumps(to_dict(run), indent=2, sort_keys=True))
        return
    print("AIConfigurator run")
    print(f"  command: {' '.join(run.command)}")
    print(f"  returncode: {run.returncode}")
    if run.output_path:
        print(f"  output: {run.output_path}")
    if run.stdout.strip():
        print(run.stdout.strip())
    if run.stderr.strip():
        print(run.stderr.strip())


def _cmd_candidates(args: argparse.Namespace) -> None:
    hardware = detect_hardware()
    model = infer_model_spec(args.model, max_context_tokens=args.max_context)
    goal = Goal(args.goal)
    candidates = generate_candidates(hardware, model, goal=goal, limit=args.limit)
    _emit({"hardware": hardware, "model": model, "candidates": candidates}, as_json=args.json)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    hardware = detect_hardware()
    model = infer_model_spec(args.model, max_context_tokens=args.max_context)
    goal = Goal(args.goal)
    configs = generate_candidates(hardware, model, goal=goal, limit=args.limit)
    results = [run_dry_benchmark(config, hardware, model) for config in configs]
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} synthetic benchmark results to {args.output}")


def _cmd_endpoint_bench(args: argparse.Namespace) -> None:
    _validate_measurement_quality_args(args)
    prediction = parse_aiconfig_prediction_csv(args.prediction_csv) if args.prediction_csv else None
    concurrency = args.concurrency
    if args.use_aic_concurrency:
        if prediction is None:
            raise SystemExit("--use-aic-concurrency requires --prediction-csv.")
        if prediction.concurrency is None:
            raise SystemExit("Prediction CSV does not include a usable concurrency value.")
        concurrency = prediction.concurrency
    if concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be at least 1.")

    config = EndpointBenchmarkConfig(
        run_id=make_run_id(),
        base_url=args.base_url,
        model=args.model,
        concurrency=concurrency,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        prompt=_endpoint_prompt(args),
        timeout_s=args.timeout,
        telemetry=args.telemetry,
        device_index=args.device_index,
        prediction_csv=str(args.prediction_csv) if args.prediction_csv else None,
        warmup_requests=args.warmup_requests,
        steady_state_duration_s=args.steady_state_seconds,
        idle_baseline_duration_s=args.idle_baseline_seconds,
        idle_power_watts=args.idle_power_watts,
    )
    hardware = detect_hardware()
    run = run_endpoint_benchmark(config=config, out_dir=args.out, prediction=prediction, hardware=hardware)
    _print_endpoint_run(run.run_dir, run.summary, run.comparison)


def _cmd_telemetry_check(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than 0.")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0.")
    run = run_telemetry_check(
        telemetry=args.telemetry,
        duration_s=args.duration,
        interval_s=args.interval,
        out_dir=args.out,
        device_index=args.device_index,
        with_nvidia_smi_loop=args.with_nvidia_smi_loop,
    )
    console = Console()
    RichTelemetryCheckReporter(console=console).render(
        summary=run.summary,
        artifacts={
            "run_dir": str(run.run_dir),
            "samples_jsonl": str(run.run_dir / "samples.jsonl"),
            "telemetry_summary_json": str(run.run_dir / "telemetry_summary.json"),
            "telemetry_capabilities_json": str(run.run_dir / "telemetry_capabilities.json"),
            "report_txt": str(run.run_dir / "report.txt"),
        },
    )


def _cmd_plan_from_aic(args: argparse.Namespace) -> None:
    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be at least 1 when provided.")
    candidates = parse_aiconfigurator_best_configs(str(args.best_config_csv), top_k=args.top_k)
    run_id = make_run_id(prefix="plans")
    run_dir = args.out / run_id
    serve_plans = [
        candidate_to_vllm_serve_plan(
            candidate,
            host=args.host,
            port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for candidate in candidates
    ]
    benchmark_plans = [
        candidate_to_endpoint_benchmark_plan(candidate, base_url=args.base_url)
        for candidate in candidates
    ]
    evaluation_plans = [
        candidate_to_evaluation_plan(
            candidate,
            base_url=args.base_url,
            host=args.host,
            port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for candidate in candidates
    ]
    write_jsonl(run_dir / "candidates.jsonl", candidates)
    write_jsonl(run_dir / "serve_plans.jsonl", serve_plans)
    write_jsonl(run_dir / "benchmark_plans.jsonl", benchmark_plans)
    write_jsonl(run_dir / "evaluation_plans.jsonl", evaluation_plans)
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "source": str(args.best_config_csv),
            "candidate_count": len(candidates),
            "top_k": args.top_k,
            "artifact_files": [
                "candidates.jsonl",
                "serve_plans.jsonl",
                "benchmark_plans.jsonl",
                "evaluation_plans.jsonl",
                "summary.json",
            ],
        },
    )
    _print_aic_plan_summary(run_dir, candidates, serve_plans)


def _cmd_run_evaluation_plan(args: argparse.Namespace) -> None:
    if args.limit_candidates is not None and args.limit_candidates < 1:
        raise SystemExit("--limit-candidates must be at least 1 when provided.")
    if args.override_concurrency is not None and args.override_concurrency < 1:
        raise SystemExit("--override-concurrency must be at least 1 when provided.")
    if args.override_num_requests is not None and args.override_num_requests < 1:
        raise SystemExit("--override-num-requests must be at least 1 when provided.")
    result = run_evaluation_plan_dir(
        plan_dir=args.plan_dir,
        out_dir=args.out,
        limit_candidates=args.limit_candidates,
        override_concurrency=args.override_concurrency,
        override_num_requests=args.override_num_requests,
        timeout_s=args.timeout,
        telemetry=args.telemetry,
    )
    _print_evaluation_summary(result.run_dir, result.summary)
    if result.failed:
        raise SystemExit(1)


def _cmd_recommend(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")
    if args.total_gpus < 1:
        raise SystemExit("--total-gpus must be at least 1.")
    if args.isl < 1 or args.osl < 1:
        raise SystemExit("--isl and --osl must both be at least 1.")
    if args.override_concurrency is not None and args.override_concurrency < 1:
        raise SystemExit("--override-concurrency must be at least 1 when provided.")
    if args.override_num_requests is not None and args.override_num_requests < 1:
        raise SystemExit("--override-num-requests must be at least 1 when provided.")
    concurrency_sweep = _parse_concurrency_sweep(args.concurrency_sweep)
    workload_profile = _resolve_workload_profile(args)

    try:
        if args.dry_run:
            run = build_attach_preflight(
                base_url=args.base_url,
                model=args.model,
                backend=args.backend,
                system=args.system,
                total_gpus=args.total_gpus,
                isl=args.isl,
                osl=args.osl,
                ttft=args.ttft,
                tpot=args.tpot,
                goal=RecommendationGoal(args.goal),
                telemetry=args.telemetry,
                out_dir=args.out,
                candidate_source=args.candidate_source,
                top_k=args.top_k,
                concurrency_sweep=concurrency_sweep,
                disable_sweep=args.disable_sweep,
                override_concurrency=args.override_concurrency,
                override_num_requests=args.override_num_requests,
                timeout_s=args.timeout,
                allow_efficiency_fallback=args.allow_efficiency_fallback,
                workload_profile=workload_profile,
            )
            _print_preflight(run.payload)
            return
        run = recommend_attach_mode(
            base_url=args.base_url,
            model=args.model,
            backend=args.backend,
            system=args.system,
            total_gpus=args.total_gpus,
            isl=args.isl,
            osl=args.osl,
            ttft=args.ttft,
            tpot=args.tpot,
            goal=RecommendationGoal(args.goal),
            telemetry=args.telemetry,
            out_dir=args.out,
            candidate_source=args.candidate_source,
            top_k=args.top_k,
            concurrency_sweep=concurrency_sweep,
            disable_sweep=args.disable_sweep,
            override_concurrency=args.override_concurrency,
            override_num_requests=args.override_num_requests,
            timeout_s=args.timeout,
            allow_efficiency_fallback=args.allow_efficiency_fallback,
            workload_profile=workload_profile,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    report_text = format_recommendation_report(run.result, metadata=run.summary, artifacts=run.result.artifacts)
    report_path = run.run_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    metadata_path = run.run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_files = metadata.get("artifact_files", [])
        if isinstance(artifact_files, list) and "report.txt" not in artifact_files:
            metadata["artifact_files"] = sorted([*(str(item) for item in artifact_files), "report.txt"])
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.quiet:
        print(f"{run.result.recommended_candidate_id or 'none'} {run.run_dir}")
    elif args.format == "json":
        print(json.dumps(to_dict(run.result), indent=2, sort_keys=True))
    else:
        console = Console()
        RichReporter(console=console).render(result=run.result, metadata=run.summary, artifacts=run.result.artifacts)
    if run.failed:
        raise SystemExit(1)


def _cmd_managed_evaluate(args: argparse.Namespace) -> None:
    workload_profile = _resolve_workload_profile(args)
    _validate_measurement_quality_args(args)
    try:
        if args.dry_run:
            run = build_managed_preflight(
                backend=args.backend,
                model=args.model,
                goal=Goal(args.goal),
                limit=args.limit,
                trials=args.trials,
                startup_timeout_s=args.startup_timeout,
                cooldown_s=args.cooldown_seconds,
                host=args.host,
                port=args.port,
                out_dir=args.out,
                telemetry=args.telemetry,
                evidence_db_path=args.evidence_db,
                evidence_write=not args.no_evidence_write,
                evidence_freshness_hours=args.evidence_freshness_hours,
                workload_profile=workload_profile,
                warmup_requests=args.warmup_requests,
                steady_state_duration_s=args.steady_state_seconds,
                idle_baseline_duration_s=args.idle_baseline_seconds,
                idle_power_watts=args.idle_power_watts,
            )
            _print_preflight(run.payload)
            return
        summary = run_managed_evaluation(
            backend=args.backend,
            model=args.model,
            goal=Goal(args.goal),
            limit=args.limit,
            trials=args.trials,
            startup_timeout_s=args.startup_timeout,
            cooldown_s=args.cooldown_seconds,
            host=args.host,
            port=args.port,
            out_dir=args.out,
            telemetry=args.telemetry,
            evidence_db_path=args.evidence_db,
            evidence_write=not args.no_evidence_write,
            evidence_freshness_hours=args.evidence_freshness_hours,
            command=["serve-optimize", "managed-evaluate"],
            workload_profile=workload_profile,
            warmup_requests=args.warmup_requests,
            steady_state_duration_s=args.steady_state_seconds,
            idle_baseline_duration_s=args.idle_baseline_seconds,
            idle_power_watts=args.idle_power_watts,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("Managed evaluation")
    print(f"  status: {summary.status}")
    print(f"  backend: {summary.backend}")
    print(f"  candidates: {summary.completed_candidate_count}/{summary.candidate_count} completed")
    print(f"  launch groups: {summary.launch_groups_count}")
    print(f"  cold launches: {summary.cold_launch_count}")
    print(f"  workload measurements: {summary.workload_measurement_count}")
    print(f"  evidence hits: {summary.evidence_hit_candidate_count}")
    if summary.evidence_db_path:
        print(f"  evidence db: {summary.evidence_db_path}")
    for warning in summary.evidence_warnings:
        print(f"  evidence warning: {warning}")
    print(f"  artifacts: {summary.artifacts['run_dir']}")
    _print_managed_recommendation_summary(summary)
    if summary.status == "failed":
        raise SystemExit(1)


def _validate_measurement_quality_args(args: argparse.Namespace) -> None:
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be at least 0.")
    if args.steady_state_seconds is not None and args.steady_state_seconds <= 0:
        raise SystemExit("--steady-state-seconds must be greater than 0 when provided.")
    if args.idle_baseline_seconds < 0:
        raise SystemExit("--idle-baseline-seconds must be at least 0.")
    if args.idle_power_watts is not None and args.idle_power_watts < 0:
        raise SystemExit("--idle-power-watts must be at least 0 when provided.")


def _resolve_workload_profile(args: argparse.Namespace):
    constraints = {
        "ttft_ms": args.slo_ttft_ms,
        "tpot_ms": args.slo_tpot_ms,
        "p95_latency_ms": args.slo_p95_latency_ms,
        "min_throughput_tokens_per_sec": args.slo_min_throughput_tokens_per_sec,
        "max_failed_request_rate": args.slo_max_failed_request_rate,
    }
    max_failed_rate = constraints["max_failed_request_rate"]
    if max_failed_rate is not None and not 0.0 <= max_failed_rate <= 1.0:
        raise SystemExit("--slo-max-failed-request-rate must be between 0.0 and 1.0.")
    try:
        return load_workload_profile(
            profile_name=args.workload_profile,
            manifest_path=args.workload_manifest,
            slo_constraints=constraints,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _print_preflight(payload: dict[str, Any]) -> None:
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    candidates = payload.get("candidates", {}) if isinstance(payload.get("candidates"), dict) else {}
    budget = payload.get("budget", {}) if isinstance(payload.get("budget"), dict) else {}
    print("Serve Optimize preflight")
    print(f"  mode: {payload.get('mode')}")
    print(f"  backend: {payload.get('backend')}")
    print(f"  model: {payload.get('model')}")
    print(f"  candidates: {candidates.get('valid_count')}/{candidates.get('generated_count')} valid")
    print(f"  launch groups: {budget.get('launch_group_count')}")
    print(f"  planned workload measurements: {budget.get('planned_workload_measurements')}")
    print("  will launch servers: no")
    print("  will call endpoint: no")
    print(f"  artifacts: {artifacts.get('run_dir')}")
    print(f"  summary: {artifacts.get('preflight_txt')}")


def _cmd_repeatability(args: argparse.Namespace) -> None:
    payload = write_repeatability_artifacts(args.run_dirs)
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    evidence_reuse = payload.get("evidence_reuse", {}) if isinstance(payload.get("evidence_reuse"), dict) else {}
    print("Recommendation repeatability")
    print(f"  runs: {payload.get('usable_run_count')}/{payload.get('run_count')} usable")
    print(f"  stability: {payload.get('stability_classification')}")
    print(f"  reuse: {evidence_reuse.get('reuse_classification')}")
    print(f"  json: {artifacts.get('recommendation_repeatability_json')}")
    print(f"  text: {artifacts.get('recommendation_repeatability_txt')}")
    for warning in payload.get("warnings", []):
        print(f"  warning: {warning}")


def _cmd_validate_campaign(args: argparse.Namespace) -> None:
    payload = write_validation_campaign_artifacts(args.run_dirs, output_dir=args.out)
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    print("Validation campaign")
    print(f"  runs: {payload.get('usable_run_count')}/{payload.get('run_count')} usable")
    print(f"  recommendation quality: {summary.get('quality_classification')}")
    print(f"  repeatability: {summary.get('repeatability_classification')}")
    print(f"  telemetry: {summary.get('telemetry_classification')}")
    print(f"  evidence: {summary.get('evidence_reuse_classification')}")
    print(f"  json: {artifacts.get('validation_campaign_json')}")
    print(f"  text: {artifacts.get('validation_campaign_txt')}")
    print(f"  csv: {artifacts.get('validation_campaign_runs_csv')}")
    for warning in payload.get("warnings", []):
        print(f"  warning: {warning}")


def _cmd_release_check(args: argparse.Namespace) -> None:
    payload = write_release_check_artifacts(out_dir=args.out, root=args.root)
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        print("Release check")
        print(f"  status: {payload.get('status')}")
        print(f"  checks: {summary.get('check_count')}")
        print(f"  failed: {summary.get('failed_count')}")
        print(f"  warnings: {summary.get('warning_count')}")
        print(f"  json: {artifacts.get('release_check_json')}")
        print(f"  text: {artifacts.get('release_check_txt')}")
    if payload.get("status") != "pass":
        raise SystemExit(1)


def _cmd_research_package(args: argparse.Namespace) -> None:
    payload = write_research_package_artifacts(args.run_dirs, output_dir=args.out)
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    print("Research package")
    print(f"  runs: {summary.get('usable_run_count')}/{summary.get('run_count')} usable")
    print(f"  backends: {summary.get('backend_count')}")
    print(f"  goals: {summary.get('goal_count')}")
    print(f"  workloads: {summary.get('workload_profile_count')}")
    print(f"  json: {artifacts.get('research_package_json')}")
    print(f"  methodology: {artifacts.get('methodology_md')}")
    print(f"  runs csv: {artifacts.get('runs_csv')}")


def _print_managed_recommendation_summary(summary) -> None:
    payload = _load_recommendation_summary(summary)
    if summary.recommendation_status != "success":
        reason = payload.get("reason") if payload else summary.recommendation_reason
        print("Recommendation: unavailable")
        print(f"  reason: {_display_text(reason)}")
        return

    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    command = payload.get("recommended_command") or "n/a"
    summary_path = summary.artifacts.get("recommendation_summary_txt") or summary.recommendation_summary_txt_path or "recommendation_summary.txt"
    print("")
    print("Recommended configuration:")
    print(f"  {command}")
    print("")
    print("Measured:")
    print(f"  throughput: {_format_metric(metrics.get('throughput_tokens_per_sec'), ' tok/s', decimals=0)}")
    print(f"  p95 latency: {_format_metric(metrics.get('p95_latency_ms'), ' ms', decimals=1)}")
    print(f"  avg power: {_format_metric(metrics.get('average_power_w'), ' W', decimals=1)}")
    print(f"  energy/token: {_format_metric(metrics.get('joules_per_token'), ' J/token', decimals=4)}")
    print(f"  efficiency: {_format_metric(metrics.get('tokens_per_watt'), ' tok/W', decimals=1)}")
    print("")
    print(f"Confidence: {_display_text(payload.get('confidence')).upper()}")
    print(f"Summary: {summary_path}")


def _load_recommendation_summary(summary) -> dict[str, Any]:
    path_text = summary.artifacts.get("recommendation_summary_json") or summary.recommendation_summary_json_path
    if not path_text:
        return {}
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_metric(value: object, suffix: str, *, decimals: int) -> str:
    number = _optional_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.{decimals}f}{suffix}"


def _display_text(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cmd_evidence_list(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    rows = list_evidence_measurements(args.db, limit=args.limit)
    if not rows:
        print(f"No evidence rows found in {args.db}")
        return
    print("created_at backend model goal throughput p95_ms avg_power_w joules_token tokens_watt confidence evidence_key")
    for row in rows:
        print(
            " ".join(
                [
                    _fmt_text(row.get("created_at")),
                    _fmt_text(row.get("backend")),
                    _fmt_text(row.get("model")),
                    _fmt_text(row.get("goal")),
                    _fmt_number(row.get("throughput_tokens_per_sec")),
                    _fmt_number(row.get("p95_latency_ms")),
                    _fmt_number(row.get("average_power_w")),
                    _fmt_number(row.get("joules_per_token")),
                    _fmt_number(row.get("tokens_per_watt")),
                    _fmt_text(row.get("confidence")),
                    str(row.get("evidence_key") or "")[:12],
                ]
            )
        )


def _cmd_optimize(args: argparse.Namespace) -> None:
    hardware = detect_hardware()
    model = infer_model_spec(args.model, max_context_tokens=args.max_context)
    goal = Goal(args.goal)
    if args.backend == "synthetic" or args.dry_run:
        hardware.notes.append("Only synthetic benchmarking is implemented in this scaffold; using synthetic runner.")
        configs = generate_candidates(hardware, model, goal=goal, limit=args.limit)
        results = [run_dry_benchmark(config, hardware, model) for config in configs]
    elif args.backend == "transformers":
        download_model(args.model, cache_dir=args.cache_dir)
        configs = make_transformers_configs(args.model)[: args.limit]
        results = []
        for trial in range(args.trials):
            for config in configs:
                options = RealBenchmarkOptions(
                    prompts=DEFAULT_PROMPTS,
                    max_new_tokens=args.max_new_tokens,
                    trial=trial,
                    cache_dir=args.cache_dir,
                )
                results.append(run_transformers_benchmark(config, hardware, model, options))
    elif args.backend == "vllm":
        download_model(args.model, cache_dir=args.cache_dir)
        configs = make_vllm_configs(args.model)[: args.limit]
        results = []
        for trial in range(args.trials):
            for config in configs:
                options = RealBenchmarkOptions(
                    prompts=DEFAULT_PROMPTS,
                    max_new_tokens=args.max_new_tokens,
                    trial=trial,
                    cache_dir=args.cache_dir,
                )
                results.append(run_vllm_benchmark(config, hardware, model, options))
    else:
        raise SystemExit(f"Unsupported backend: {args.backend}")
    frontier = pareto_frontier(results)
    selected = select_recommendation(results, goal)
    recommendation = Recommendation(
        goal=goal.value,
        selected=selected,
        frontier=frontier,
        evaluated=results,
        hardware=hardware,
        model=model,
        notes=_recommendation_notes(args.backend, args.dry_run),
    )
    write_json(args.output_dir / "recommendation.json", recommendation)
    write_jsonl(args.output_dir / "results.jsonl", results)
    _emit(recommendation, as_json=args.json)


def _cmd_smoke(args: argparse.Namespace) -> None:
    hardware = detect_hardware()
    model_ids = args.models or TINY_MODEL_IDS
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    all_recommendations = []
    all_results = []
    for model_id in model_ids:
        download_model(model_id, cache_dir=args.cache_dir)
        model = infer_model_spec(model_id)
        configs = make_transformers_configs(model_id) if args.backend == "transformers" else make_vllm_configs(model_id)
        results = []
        for trial in range(args.trials):
            for config in configs:
                options = RealBenchmarkOptions(
                    prompts=DEFAULT_PROMPTS,
                    max_new_tokens=args.max_new_tokens,
                    trial=trial,
                    cache_dir=args.cache_dir,
                )
                if args.backend == "transformers":
                    result = run_transformers_benchmark(config, hardware, model, options)
                else:
                    result = run_vllm_benchmark(config, hardware, model, options)
                results.append(result)
                all_results.append(result)
        frontier = pareto_frontier(results)
        selected = select_recommendation(results, Goal.BALANCED)
        recommendation = Recommendation(
            goal=Goal.BALANCED.value,
            selected=selected,
            frontier=frontier,
            evaluated=results,
            hardware=hardware,
            model=model,
            notes=[f"Functional smoke benchmark using the {args.backend} backend and nvidia-smi power sampling."],
        )
        safe_name = model_id.replace("/", "--")
        write_json(args.output_dir / safe_name / "recommendation.json", recommendation)
        write_jsonl(args.output_dir / safe_name / "results.jsonl", results)
        all_recommendations.append(recommendation)

    write_jsonl(args.output_dir / "all_results.jsonl", all_results)
    if args.json:
        print(json.dumps(to_dict(all_recommendations), indent=2, sort_keys=True))
        return
    print(f"Smoke benchmark complete: {len(model_ids)} model(s), {len(all_results)} run(s)")
    for recommendation in all_recommendations:
        selected = recommendation.selected
        if selected is None:
            print(f"  {recommendation.model.model_id}: no feasible result")
        else:
            print(
                f"  {recommendation.model.model_id}: batch={selected.config.max_batch_size}, "
                f"tok/s={selected.throughput_tok_s}, W={selected.average_power_watts}, "
                f"J/token={selected.joules_per_token}"
            )


def _emit(payload: object, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(to_dict(payload), indent=2, sort_keys=True))
        return
    if isinstance(payload, Recommendation):
        _print_recommendation(payload)
        return
    if isinstance(payload, dict) and "candidates" in payload:
        _print_candidates(payload)
        return
    _print_hardware(payload)


def _endpoint_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return DEFAULT_ENDPOINT_PROMPT.strip()


def _parse_concurrency_sweep(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--concurrency-sweep must include at least one integer.")
    parsed: list[int] = []
    for part in parts:
        try:
            number = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid --concurrency-sweep value: {part}") from exc
        if number < 1:
            raise SystemExit("--concurrency-sweep values must be at least 1.")
        parsed.append(number)
    return tuple(parsed)


def _print_endpoint_run(run_dir: Path, summary: object, comparison: object | None) -> None:
    print(f"Endpoint benchmark complete: {run_dir}")
    print("Measured")
    print(f"  requests: {summary.successful_requests}/{summary.total_requests} ok")
    print(f"  request_rate_req_s: {summary.request_rate_req_s:.3f}")
    print(f"  total_tokens_s: {summary.total_tokens_s:.3f}")
    print(f"  output_tokens_s: {summary.output_tokens_s:.3f}")
    print(f"  avg_latency_s: {_format_optional(summary.avg_latency_s)}")
    print(f"  p95_latency_s: {_format_optional(summary.p95_latency_s)}")
    if summary.telemetry_provider is not None:
        print(f"  telemetry_provider: {summary.telemetry_provider}")
        print(f"  average_power_watts: {_format_optional(summary.average_power_watts)}")
        print(f"  peak_power_watts: {_format_optional(summary.peak_power_watts)}")
        print(f"  energy_joules: {_format_optional(summary.energy_joules)}")
        print(f"  joules_per_token: {_format_optional(summary.joules_per_token)}")
        print(f"  tokens_per_second_per_watt: {_format_optional(summary.tokens_per_second_per_watt)}")
    for warning in getattr(summary, "warnings", []):
        print(f"  warning: {warning}")
    if comparison is None:
        return
    metrics = comparison.metrics
    print("Predicted vs measured")
    for name in ("tokens_s", "request_rate", "request_latency_avg_s", "request_latency_p95_s", "concurrency"):
        metric = metrics.get(name)
        if not metric:
            continue
        predicted = _format_optional(metric["predicted"])
        measured = _format_optional(metric["measured"])
        ratio = _format_optional(metric["measured_over_predicted_ratio"])
        print(f"  {name}: predicted={predicted}, measured={measured}, ratio={ratio}")


def _print_aic_plan_summary(run_dir: Path, candidates: list[object], serve_plans: list[object]) -> None:
    print(f"AIConfigurator plan artifacts: {run_dir}")
    print("rank backend model tp pp dp dtype concurrency predicted_tokens_s predicted_latency_ms command")
    for candidate, serve_plan in zip(candidates, serve_plans, strict=False):
        model = candidate.model or "unknown"
        if len(model) > 48:
            model = "..." + model[-45:]
        print(
            f"{candidate.rank} "
            f"{candidate.backend or 'unknown'} "
            f"{model} "
            f"{candidate.tp or 'unknown'} "
            f"{candidate.pp or 'unknown'} "
            f"{candidate.dp or 'unknown'} "
            f"{serve_plan.dtype} "
            f"{candidate.concurrency or 'unknown'} "
            f"{_format_optional(candidate.predicted_tokens_s)} "
            f"{_format_optional(candidate.predicted_request_latency_ms)} "
            f"{serve_plan.shell_command}"
        )


def _print_evaluation_summary(run_dir: Path, summary: dict[str, object]) -> None:
    print(f"Evaluation artifacts: {run_dir}")
    print(
        "candidate_id concurrency predicted_tokens_s measured_total_tokens_s measured/predicted "
        "measured_request_rate avg_latency_ms p95_latency_ms failed_requests"
    )
    for row in summary.get("candidates", []):
        if not isinstance(row, dict):
            continue
        print(
            f"{row.get('candidate_id', 'unknown')} "
            f"{row.get('concurrency', 'unknown')} "
            f"{_format_optional(row.get('predicted_tokens_s'))} "
            f"{_format_optional(row.get('measured_total_tokens_s'))} "
            f"{_format_optional(row.get('measured_over_predicted_tokens_ratio'))} "
            f"{_format_optional(row.get('measured_request_rate'))} "
            f"{_format_optional(row.get('measured_avg_latency_ms'))} "
            f"{_format_optional(row.get('measured_p95_latency_ms'))} "
            f"{row.get('failed_requests', 'unknown')}"
        )


def _format_optional(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_number(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_text(value: object) -> str:
    text = str(value) if value is not None else "unknown"
    return text.replace(" ", "_")


def _print_hardware(payload: object) -> None:
    hardware = payload
    print("Hardware snapshot")
    for note in getattr(hardware, "notes", []):
        print(f"  note: {note}")
    gpus = getattr(hardware, "gpus", [])
    if not gpus:
        print("  GPUs: none detected")
        return
    for gpu in gpus:
        print(f"  GPU {gpu.index}: {gpu.name}")
        print(f"    uuid: {gpu.uuid or 'unknown'}")
        print(f"    memory: {gpu.free_memory_mb or 'unknown'} / {gpu.total_memory_mb or 'unknown'} MB free")
        print(f"    power: {gpu.current_power_watts or 'unknown'} W current, {gpu.power_limit_watts or 'unknown'} W limit")
        print(f"    mig: {gpu.mig_mode or 'unknown'} {gpu.mig_profile or ''}".rstrip())
        print(f"    source: {gpu.source}")


def _print_candidates(payload: dict[str, object]) -> None:
    model = payload["model"]
    candidates = payload["candidates"]
    print(f"Candidate configs for {model.model_id} ({model.parameter_count_b}B params)")
    print("id           backend    dtype  quant       batch  ctx     est_vram_mb  power_w")
    for config in candidates:
        print(
            f"{config.id:<12} {config.backend:<10} {config.dtype:<5} {config.quantization:<11} "
            f"{config.max_batch_size:<6} {config.max_context_tokens:<7} "
            f"{config.estimated_vram_mb or 'unknown':<12} {config.power_limit_watts or 'default'}"
        )


def _print_recommendation(recommendation: Recommendation) -> None:
    selected = recommendation.selected
    print(f"Serve Optimize recommendation for {recommendation.model.model_id}")
    print(f"Goal: {recommendation.goal}")
    for note in recommendation.hardware.notes + recommendation.notes:
        print(f"note: {note}")
    if selected is None:
        print("No feasible configuration found.")
        return
    config = selected.config
    print("")
    print("Recommended configuration")
    print(f"  backend: {config.backend}")
    print(f"  dtype: {config.dtype}")
    print(f"  quantization: {config.quantization}")
    print(f"  max_batch_size: {config.max_batch_size}")
    print(f"  max_context_tokens: {config.max_context_tokens}")
    print(f"  kv_cache_policy: {config.kv_cache_policy}")
    print(f"  scheduler: {config.scheduler}")
    print(f"  power_limit_watts: {config.power_limit_watts or 'default'}")
    print("")
    print("Expected performance")
    print(f"  throughput_tok_s: {selected.throughput_tok_s}")
    print(f"  average_power_watts: {selected.average_power_watts}")
    print(f"  joules_per_token: {selected.joules_per_token}")
    print(f"  tokens_per_watt: {selected.tokens_per_watt}")
    print(f"  ttft_ms: {selected.ttft_ms}")
    print(f"  frontier_size: {len(recommendation.frontier)}")


def _recommendation_notes(backend: str, dry_run: bool) -> list[str]:
    if backend == "transformers" and not dry_run:
        return ["Measured with the transformers backend and local power sampling."]
    if backend == "vllm" and not dry_run:
        return ["Measured with the offline vLLM backend and local power sampling."]
    return ["Dry-run metrics are synthetic and intended for CI and offline optimizer validation only."]
