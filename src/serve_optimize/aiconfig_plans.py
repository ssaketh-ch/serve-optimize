"""Translate AIConfigurator candidates into runnable Serve Optimize plans."""

from __future__ import annotations

import shlex

from .schemas import CandidateEvaluationPlan, EndpointBenchmarkPlan, ServeCandidate, VllmServePlan


def candidate_to_vllm_serve_plan(
    candidate: ServeCandidate,
    host: str = "127.0.0.1",
    port: int = 8080,
    gpu_memory_utilization: float = 0.90,
) -> VllmServePlan:
    model = candidate.model or ""
    dtype = _candidate_dtype(candidate)
    tensor_parallel_size = candidate.tp or 1
    max_model_len = max((candidate.isl or 0) + (candidate.osl or 0), 2048)
    command = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
    ]
    return VllmServePlan(
        candidate_id=candidate.candidate_id,
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=candidate.pp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        command=command,
        shell_command=shlex.join(command),
    )


def candidate_to_endpoint_benchmark_plan(
    candidate: ServeCandidate,
    base_url: str,
    num_requests: int | None = None,
) -> EndpointBenchmarkPlan:
    concurrency = candidate.concurrency or 1
    output_tokens = candidate.osl or 1
    request_count = num_requests if num_requests is not None else max(2 * concurrency, 128)
    return EndpointBenchmarkPlan(
        candidate_id=candidate.candidate_id,
        base_url=base_url,
        model=candidate.model or "",
        concurrency=concurrency,
        num_requests=request_count,
        max_tokens=output_tokens,
        expected_input_tokens=candidate.isl,
        expected_output_tokens=candidate.osl,
    )


def candidate_to_evaluation_plan(
    candidate: ServeCandidate,
    base_url: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    gpu_memory_utilization: float = 0.90,
    num_requests: int | None = None,
) -> CandidateEvaluationPlan:
    notes = []
    if candidate.backend and candidate.backend.lower() != "vllm":
        notes.append("vLLM serve plan generated only because the current translator targets vLLM.")
    return CandidateEvaluationPlan(
        candidate_id=candidate.candidate_id,
        rank=candidate.rank,
        candidate=candidate,
        serve_plan=candidate_to_vllm_serve_plan(candidate, host=host, port=port, gpu_memory_utilization=gpu_memory_utilization),
        benchmark_plan=candidate_to_endpoint_benchmark_plan(candidate, base_url=base_url, num_requests=num_requests),
        notes=notes,
    )


def _candidate_dtype(candidate: ServeCandidate) -> str:
    precision_fields = [candidate.gemm, candidate.kvcache, candidate.fmha]
    if any((value or "").lower() == "bfloat16" for value in precision_fields):
        return "bfloat16"
    return "float16"
