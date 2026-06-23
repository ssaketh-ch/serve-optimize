# Hardware Support

## Current Detection

Serve Optimize detects NVIDIA GPU metadata through NVML or `nvidia-smi` when available.

Recorded fields may include:

* device name
* visible memory
* driver and CUDA information
* MIG mode and visible instance metadata
* power, temperature, utilization, clocks, and memory usage when exposed

Missing telemetry fields are capability limitations, not zero values.

## Backend Runtime Requirements

Managed Mode requires a compatible NVIDIA GPU, a supported backend runtime, and an OpenAI compatible serving endpoint.

The validated backend stacks are listed in [Support Matrix](support_matrix.md). Backend wheel environments do not require a repository specific compiler path.

## MIG

MIG metadata and telemetry caveats are supported.

Power scope may be board level rather than instance level. Recommendations must state the available scope and avoid claiming per instance attribution unless the provider exposes it.

## CPU Or Unsupported GPU Hosts

Schema, CLI, synthetic, and some functional tests can run without supported GPU telemetry.

Managed backend measurements require a compatible installed runtime and hardware.

## New Hardware Setup

For any new NVIDIA host:

1. Install the core profile in a fresh environment.
2. Run `serve-optimize detect`.
3. Run `serve-optimize telemetry-check --telemetry auto --duration 5`.
4. Install backend profiles only for the backends you plan to run.
5. Run `serve-optimize optimize MODEL --dry-run` before measured runs.
6. Collect fresh runtime fingerprinted evidence on the new host.
7. Validate repeatability with `serve-optimize validate-campaign`.

Evidence from another host can be archived and analyzed, but it cannot be reused as exact evidence when the hardware fingerprint changes.

## Broader Hardware Evaluation

Each new system needs fresh runtime fingerprinted evidence before recommendation claims are made for that hardware.
