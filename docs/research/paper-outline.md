# Paper Outline

This is a planned paper structure. It includes evaluation goals and claims that are not yet implemented or proven. Current product support is defined in `docs/compatibility.md`.

Working title:

> Serve Optimize: Energy-Aware Configuration Search for Single-GPU and MIG-Based LLM Inference

## Abstract

State the problem, the missing energy dimension in inference configuration search, the system design, and the main empirical result.

## 1. Introduction

- LLM inference is increasingly the dominant serving cost.
- Current optimization often targets throughput or latency.
- Small deployments need power-performance tradeoff guidance.
- Energy-optimal configs can differ from throughput-optimal configs.
- Contributions: system, optimizer, MIG study, dataset.

## 2. Background And Motivation

- LLM inference phases: prefill and decode.
- Serving configuration knobs.
- Energy metrics.
- MIG deployment constraints.
- Motivating experiment showing different winners for throughput and joules/token.

## 3. System Design

- Hardware detection.
- Candidate generation.
- Benchmarking.
- Telemetry.
- Metrics.
- Pareto optimizer.
- Recommendation engine.

## 4. Energy-Aware Search

- Search space.
- Feasibility filters.
- Analytical priors.
- Measurement-guided refinement.
- Goal functions.

## 5. Evaluation

Questions:

- How much energy can be saved at similar throughput?
- How much search can be avoided?
- How stable are Pareto frontiers across models and workloads?
- How does MIG affect efficiency?
- How do quantization and power caps affect energy?

## 6. Discussion

- Measurement limitations.
- Board-level versus MIG-level power.
- Backend version churn.
- Generalization.
- Practitioner guidance.

## 7. Related Work

- AIConfigurator and inference configuration optimization.
- TokenPowerBench and inference power measurement.
- vLLM, SGLang, TensorRT-LLM.
- DVFS and power capping for GPU workloads.
- Sustainable AI benchmarks.

## 8. Conclusion

Summarize the case for energy-aware inference configuration search and the practical value of exposing Pareto tradeoffs.
