# Backend Assumptions

This document records backend specific boundaries for the deployed vLLM and SGLang Managed Mode support.

## Adapter Ownership

`src/serve_optimize/backends/vllm.py` owns:

* vLLM availability and version detection
* vLLM help parsing and canonical capability identity
* vLLM command rendering and canonical metadata
* vLLM managed lifecycle details

`src/serve_optimize/backends/sglang.py` owns:

* SGLang availability and version detection
* SGLang help parsing and canonical capability identity
* SGLang command rendering and canonical metadata
* SGLang managed lifecycle details

Managed orchestration must not reproduce backend command logic.

## Shared Boundary

Both adapters participate in:

* prelaunch validation
* rendered command hashing
* canonical launch configuration
* runtime fingerprinting
* evidence compatibility
* OpenAI compatible health and benchmark behavior
* lifecycle and failure artifacts
* measured recommendations

## vLLM Surface

vLLM supports the common managed fields plus capability detected engine fields such as block size, KV cache dtype, eager mode, maximum batched tokens, chunked prefill, CUDA graph capture sizing, and prefix caching.

Detection failure does not imply optional flag support.

## SGLang Surface

SGLang supports the common managed fields plus capability detected memory fraction, request capacity, compatible AWQ or GPTQ quantization, chunked prefill size, radix cache disable, CUDA graph controls, served model naming, remote code trust, and piecewise CUDA graph disable.

For local SGLang runtimes, options are rendered only when the installed command surface reports support. The conservative generated baseline enables `--disable-piecewise-cuda-graph` when that flag is available. Source builds may use `scripts/env_base_runtime.sh` to expose the local CUDA toolkit, but wheel installs do not require the helper.

## Intentional Non Parity

Fields without a direct semantic match are rejected or marked unsupported. Current examples include vLLM specific block size, KV cache dtype, eager mode, maximum batched tokens, and prefix caching when no direct SGLang mapping exists.

First class support means lifecycle and evidence parity for the supported surface. It does not mean universal command option parity.

## Evidence Separation

Backend name, version, capabilities, rendered command, canonical configuration, runtime environment, model, and workload identities participate in evidence compatibility.

vLLM and SGLang evidence cannot collide as exact matches.

## Remaining Limits

* Local backend installation is not assumed.
* SGLang candidate generation remains conservative and bounded.
* AIConfigurator synthesis currently skips non vLLM backends.
* Attach Mode serve plan generation may remain backend specific.
* Generic recommendation tables retain optional vLLM originated columns when absent values are harmless.

No broad backend abstraction refactor is required for the current deployed product surface.
