"""Curated landscape catalog for related systems and integration targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LandscapeItem:
    name: str
    category: str
    relevance: str
    priority: str
    url: str


LANDSCAPE: list[LandscapeItem] = [
    LandscapeItem(
        name="AIConfigurator",
        category="configuration optimizer",
        relevance="Framework-agnostic throughput modeling and launch-parameter search.",
        priority="anchor",
        url="https://arxiv.org/abs/2601.06288",
    ),
    LandscapeItem(
        name="TokenPowerBench",
        category="energy benchmark",
        relevance="Declarative LLM inference power measurement with prefill/decode attribution.",
        priority="anchor",
        url="https://arxiv.org/abs/2512.03024",
    ),
    LandscapeItem(
        name="Towards Greener LLMs",
        category="energy-aware LLM serving paper",
        relevance="Shows LLM inference energy/performance tradeoffs under SLOs.",
        priority="high",
        url="https://arxiv.org/abs/2403.20306",
    ),
    LandscapeItem(
        name="Watt Counts",
        category="energy benchmark dataset",
        relevance="Large open-access LLM inference energy dataset across GPUs.",
        priority="high",
        url="https://arxiv.org/abs/2604.09048",
    ),
    LandscapeItem(
        name="throttLL'eM",
        category="DVFS optimizer",
        relevance="SLO-aware GPU frequency scaling for energy-efficient LLM serving.",
        priority="high",
        url="https://arxiv.org/abs/2408.05235",
    ),
    LandscapeItem(
        name="GreenLLM",
        category="DVFS optimizer",
        relevance="Queue-aware clock selection for prefill/decode energy minimization.",
        priority="high",
        url="https://arxiv.org/abs/2508.16449",
    ),
    LandscapeItem(
        name="vLLM / PagedAttention",
        category="serving backend",
        relevance="High-throughput serving with paged KV cache and continuous batching.",
        priority="first backend",
        url="https://arxiv.org/abs/2309.06180",
    ),
    LandscapeItem(
        name="SGLang",
        category="serving backend",
        relevance="High-performance serving runtime with OpenAI-compatible benchmarking support.",
        priority="second backend",
        url="https://arxiv.org/abs/2312.07104",
    ),
    LandscapeItem(
        name="TensorRT-LLM",
        category="serving backend",
        relevance="NVIDIA-optimized LLM inference stack with OpenAI-compatible serving.",
        priority="performance backend",
        url="https://github.com/NVIDIA/TensorRT-LLM",
    ),
    LandscapeItem(
        name="Hugging Face TGI",
        category="serving backend",
        relevance="Production LLM serving toolkit and comparative baseline.",
        priority="baseline",
        url="https://github.com/huggingface/text-generation-inference",
    ),
    LandscapeItem(
        name="GenAI-Perf",
        category="benchmark client",
        relevance="Measures TTFT, ITL, throughput, and request rate for OpenAI-compatible endpoints.",
        priority="high",
        url="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html",
    ),
    LandscapeItem(
        name="NVIDIA DCGM",
        category="telemetry",
        relevance="Production GPU telemetry and profiling fields, including power metrics.",
        priority="high",
        url="https://docs.nvidia.com/datacenter/dcgm/latest/contents.html",
    ),
    LandscapeItem(
        name="NVML",
        category="telemetry",
        relevance="Direct NVIDIA management API for GPU identity, memory, clocks, and power.",
        priority="implemented",
        url="https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html",
    ),
    LandscapeItem(
        name="MIG User Guide",
        category="MIG deployment",
        relevance="Official reference for MIG partitioning, device identity, and operations.",
        priority="high",
        url="https://docs.nvidia.com/datacenter/tesla/mig-user-guide/",
    ),
    LandscapeItem(
        name="MIG power partitioning",
        category="MIG research",
        relevance="Highlights hardware limits and modeling needs for attributing power to MIG instances.",
        priority="high",
        url="https://arxiv.org/abs/2501.17752",
    ),
    LandscapeItem(
        name="Zeus",
        category="energy optimizer",
        relevance="Energy optimization framework for DNN jobs; useful design precedent for power-limit search.",
        priority="medium",
        url="https://arxiv.org/abs/2208.06102",
    ),
    LandscapeItem(
        name="CodeCarbon",
        category="carbon accounting",
        relevance="Tracks CPU/GPU/RAM power and converts energy to emissions estimates.",
        priority="optional",
        url="https://docs.codecarbon.io/latest/",
    ),
    LandscapeItem(
        name="MLPerf Inference",
        category="benchmark standard",
        relevance="Industry benchmark with performance and power methodology precedents.",
        priority="methodology",
        url="https://arxiv.org/abs/1911.02549",
    ),
]


def grouped_landscape() -> dict[str, list[LandscapeItem]]:
    grouped: dict[str, list[LandscapeItem]] = {}
    for item in LANDSCAPE:
        grouped.setdefault(item.category, []).append(item)
    return grouped

