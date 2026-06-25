# Literature And Tooling Landscape

This document maps research and software relevant to Serve Optimize. It is a literature and future integration guide, not a current compatibility statement. Current support is defined in [Compatibility](../compatibility.md).

## Anchor Systems

### AIConfigurator

- Category: throughput-oriented configuration optimization.
- Role: analytical performance modeling, backend-agnostic search, launch parameter recommendation.
- What to reuse conceptually: candidate pruning, framework abstraction, performance model calibration, fast search.
- What Serve Optimize adds: power/energy objective functions, joules/token modeling, MIG energy caveats, power-limit search, and Pareto outputs.
- Source: https://arxiv.org/abs/2601.06288

### TokenPowerBench

- Category: LLM inference energy benchmark.
- Role: declarative benchmark configuration, GPU/node/system power measurement, prefill/decode phase attribution.
- What to reuse conceptually: phase-aligned metrics, power sampling methodology, benchmark artifact structure.
- What Serve Optimize adds: automatic candidate generation, backend launch recommendation, Pareto optimizer, and search-cost reduction.
- Source: https://arxiv.org/abs/2512.03024

## Energy-Aware LLM Inference Papers

### Towards Greener LLMs

Shows that energy should be optimized under performance SLOs rather than treated as a passive measurement. This supports Serve Optimize's `balanced` and `efficient` modes.

Source: https://arxiv.org/abs/2403.20306

### Watt Counts

Provides an open-access LLM inference energy dataset across many GPUs and models. This is important related work and a potential dataset comparison point.

Source: https://arxiv.org/abs/2604.09048

### Where Do The Joules Go?

Diagnoses inference energy across model/task/configuration choices. This supports the case for phase-aware and configuration-aware energy modeling.

Source: https://arxiv.org/abs/2601.22076

### ML.ENERGY Benchmark

Frames inference energy measurement and optimization as a benchmark and leaderboard problem. Useful for methodology, artifact release, and leaderboard design.

Source: https://arxiv.org/abs/2505.06371

## DVFS And Power-Cap Systems

### throttLL'eM

Uses SLO-aware GPU frequency scaling for energy-efficient LLM serving. Serve Optimize should treat clocks and power limits as optional search dimensions when permissions and hardware support exist.

Source: https://arxiv.org/abs/2408.05235

### GreenLLM

Uses prefill/decode-aware latency-power models and queueing-aware optimization to select energy-minimal clocks. This is directly relevant to phase-aware recommendation.

Source: https://arxiv.org/abs/2508.16449

### Zeus

Although focused on DNN training, Zeus is a strong precedent for automatically navigating GPU energy/performance tradeoffs through job-level and GPU-level configuration search.

Source: https://arxiv.org/abs/2208.06102

## LLM Serving Systems

### vLLM

A validated first class Managed Mode backend and a natural baseline for throughput and energy frontier studies.

Sources:

- https://arxiv.org/abs/2309.06180
- https://docs.vllm.ai/en/latest/

### SGLang

A validated first class Managed Mode backend for the supported detected surface. SGLang also provides native benchmark tooling and OpenAI compatible endpoints.

Sources:

- https://arxiv.org/abs/2312.07104
- https://docs.sglang.ai/

### TensorRT-LLM

High performance NVIDIA inference stack. TensorRT LLM is planned only and is not in the current Serve Optimize Managed Mode scope.

Sources:

- https://github.com/NVIDIA/TensorRT-LLM
- https://nvidia.github.io/TensorRT-LLM/

### Hugging Face Text Generation Inference

Important production serving baseline and external Attach Mode baseline. It may be measured through Attach Mode when an OpenAI compatible endpoint is available.

Source: https://github.com/huggingface/text-generation-inference

### LMDeploy And llama.cpp

Useful secondary baselines. LMDeploy matters for server deployments and llama.cpp matters for local/edge baselines where GPU power is not the only resource.

Sources:

- https://github.com/InternLM/lmdeploy
- https://github.com/ggml-org/llama.cpp

## Benchmark Clients

### NVIDIA GenAI-Perf

Production-grade client-side benchmark tool for LLM endpoints. It measures TTFT, inter-token latency, output token throughput, and request throughput, and supports OpenAI-compatible APIs.

Source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html

### vLLM Bench

Native vLLM benchmark tooling for online serving throughput. It should be used both as an integration option and as a baseline for Serve Optimize's own client.

Source: https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/

### SGLang Bench Serving

SGLang's benchmark client supports synthetic or dataset-driven prompts, TTFT, ITL, latency, rate control, concurrency limits, and multiple backend endpoint types.

Source: https://docs.sglang.ai/developer_guide/bench_serving.html

### MLPerf Inference

Important benchmark methodology precedent, especially for performance and power reporting discipline.

Source: https://arxiv.org/abs/1911.02549

## Telemetry And Power Measurement

### NVML

Primary direct telemetry path for GPU identity, memory, clocks, power draw, and power limits.

Source: https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html

### nvidia-smi

Fallback and operator-facing telemetry path. It is also useful for power-limit and clock-control commands where supported.

Source: https://docs.nvidia.com/deploy/nvidia-smi/

### NVIDIA DCGM

Production telemetry path for datacenter GPUs. DCGM and DCGM Exporter should become first-class integrations for long-running measurements and Prometheus/Grafana deployments.

Sources:

- https://docs.nvidia.com/datacenter/dcgm/latest/contents.html
- https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html

### CodeCarbon

Optional emissions accounting integration. It is less precise for phase-level inference analysis but useful for reporting energy and estimated CO2 emissions.

Source: https://docs.codecarbon.io/latest/

## MIG-Specific Work

### NVIDIA MIG Documentation

MIG partitions supported GPUs into isolated instances with dedicated resources. Serve Optimize must report the visible device, profile, UUIDs, and whether power telemetry is board-level or instance-attributed.

Source: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/

### MIGPerf

Benchmarking study of training and inference workloads on MIG. Useful for understanding resource partition behavior.

Source: https://arxiv.org/abs/2301.00407

### On The Partitioning Of GPU Power Among Multi-Instances

Directly relevant to the hardest MIG measurement issue: power apportionment among instances is challenging because of limited hardware support.

Source: https://arxiv.org/abs/2501.17752

### MISO

Shows dynamic exploitation of MIG capability for multi-tenant ML systems. Relevant for future scheduling work.

Source: https://arxiv.org/abs/2207.11428

## Future Design Implications

1. Keep AIConfigurator and TokenPowerBench as anchor comparisons while preserving measured evidence as final truth.
2. Evaluate additional telemetry providers only in explicitly scoped phases.
3. Evaluate native benchmark clients as baselines without replacing the shared OpenAI compatible product boundary casually.
4. Add power caps and clocks only after permissions, safety, and measurement contracts are defined.
5. Continue distinguishing measured, estimated, synthetic, and imported values.
6. Add phase aware metrics only after defensible measurement boundaries exist.
7. Keep MIG power attribution limits explicit.
8. Keep TensorRT LLM planned only. Measure external TGI, LMDeploy, and llama.cpp endpoints through Attach Mode unless lifecycle ownership is separately scoped.
