# Backend Expansion Scope

Status: accepted on 2026-06-22.

## Decision

TensorRT LLM is planned only and is not in the current Serve Optimize Managed Mode scope.

TGI, LMDeploy, llama.cpp, and NIM remain external Attach Mode targets when they expose an OpenAI compatible endpoint. Serve Optimize does not launch, stop, configure, or claim the runtime identity of those servers.

The supported Managed Mode backends remain vLLM and SGLang.

## Rationale

TensorRT LLM requires an engine build lifecycle in addition to a server lifecycle. The repository does not currently own model conversion, engine build execution, build compatibility, cache invalidation, or engine artifact cleanup. A launch only adapter would produce incomplete lifecycle and evidence claims.

The removed TensorRT LLM placeholder returned an empty command and could not satisfy the backend adapter contract. Keeping it would imply implementation progress that did not exist.

## Future Admission Gate

Before a TensorRT LLM adapter is added, a separate accepted design must define and test:

1. Source model and conversion identity.
2. Builder version, CUDA, TensorRT, hardware, precision, quantization, parallelism, shape limits, and plugin fingerprints.
3. Deterministic engine build commands and retained build logs.
4. Engine artifact layout, manifest, validation, cache lookup, invalidation, and cleanup.
5. Build failure and interruption recovery.
6. Server launch, health, benchmark, telemetry, stop, and log ownership after a valid engine exists.
7. Evidence rules that bind measurements to the exact engine manifest and runtime fingerprint.
8. Isolated installation and real hardware validation.

An adapter must not be written before this lifecycle design is accepted.

## Consequences

Managed Mode CLI choices expose only vLLM and SGLang. Requests for TensorRT LLM explain that it is planned only. Requests for TGI, LMDeploy, llama.cpp, or NIM explain that they are Attach Mode only.

Attach Mode may measure a compatible endpoint from any of these engines, but its existing limitation remains: it cannot prove how that endpoint was launched.
