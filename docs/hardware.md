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

## Validated Host

The current Managed Mode evidence baseline was produced on an NVIDIA H200 NVL host.

Validated backend environments:

* `serve-vllm-baseline`
* `serve-sglang-latest`

SGLang uses GCC Toolset `12.2.1` through `scripts/env_base_runtime.sh`.

## MIG

MIG metadata and telemetry caveats are supported.

Power scope may be board level rather than instance level. Recommendations must state the available scope and avoid claiming per instance attribution unless the provider exposes it.

## CPU Or Unsupported GPU Hosts

Schema, CLI, synthetic, and some functional tests can run without supported GPU telemetry.

Managed backend measurements require a compatible installed runtime and hardware.

## RTX Pro 6000 Migration

RTX Pro 6000 is a supported evaluation target when the server exposes NVIDIA telemetry and the selected backend runtime installs cleanly.

Migration checklist:

1. Install the core profile in a fresh environment.
2. Run `serve-optimize detect`.
3. Run `serve-optimize telemetry-check --telemetry auto --duration 5`.
4. Install the vLLM profile in its own environment if vLLM Managed Mode is needed.
5. Install the SGLang profile in its own environment if SGLang Managed Mode is needed.
6. Run `serve-optimize managed-evaluate --dry-run` before any measured run.
7. Collect fresh runtime fingerprinted evidence on the RTX Pro 6000 host.
8. Validate repeats with `serve-optimize validate-campaign`.

Do not reuse H200 evidence as exact evidence on RTX Pro 6000. The hardware fingerprint must change.

## Broader Hardware Evaluation

Additional evidence can be collected for:

* H200 MIG profiles
* additional NVIDIA architectures

Each new system needs fresh runtime fingerprinted evidence before recommendation claims are made for that hardware.
