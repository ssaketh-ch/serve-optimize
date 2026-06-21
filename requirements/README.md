# Requirements

Pinned requirement profiles for reproducible installs.

Profiles:

* `profiles/core.txt`: CLI, schemas, synthetic paths, endpoint client, and artifact tooling.
* `profiles/telemetry.txt`: core plus NVIDIA telemetry bindings.
* `profiles/vllm.txt`: validated vLLM Managed Mode runtime.
* `profiles/sglang.txt`: validated SGLang Managed Mode runtime.

The vLLM and SGLang profiles must be installed in separate environments because they require different Torch and Transformers stacks.

Constraints under `constraints/` pin directly validated backend packages while allowing backend packages to resolve transitive dependencies.
