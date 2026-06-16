# Reproducible Installation

Serve Optimize uses mutually exclusive backend profiles because the validated vLLM and SGLang releases require different Torch and Transformers versions.

## Profiles

| Profile | Requirement file | Purpose |
|---|---|---|
| core | `requirements/profiles/core.txt` | CLI, schemas, synthetic paths, endpoint client, and artifact tooling |
| telemetry | `requirements/profiles/telemetry.txt` | Core plus direct NVML Python bindings |
| vLLM | `requirements/profiles/vllm.txt` | Validated vLLM Managed Mode stack |
| SGLang | `requirements/profiles/sglang.txt` | Validated SGLang Managed Mode stack |

Do not install the vLLM and SGLang profiles into the same environment.

## Core

```bash
python -m venv .venv-core
.venv-core/bin/python -m pip install --upgrade "pip==26.1.2"
.venv-core/bin/python -m pip install -r requirements/profiles/core.txt
.venv-core/bin/serve-optimize doctor --profile core
```

## Telemetry

```bash
python -m venv .venv-telemetry
.venv-telemetry/bin/python -m pip install --upgrade "pip==26.1.2"
.venv-telemetry/bin/python -m pip install -r requirements/profiles/telemetry.txt
.venv-telemetry/bin/serve-optimize doctor --profile telemetry
```

## vLLM

```bash
python -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade "pip==26.1.2"
.venv-vllm/bin/python -m pip install -r requirements/profiles/vllm.txt
.venv-vllm/bin/serve-optimize doctor --profile vllm
```

The profile pins:

* vLLM `0.10.0`
* Torch `2.7.1`
* Transformers `4.57.6`
* Hugging Face Hub `0.36.2`
* NVML Python bindings `13.610.43`

## SGLang

```bash
python -m venv .venv-sglang
.venv-sglang/bin/python -m pip install --upgrade "pip==26.1.2"
.venv-sglang/bin/python -m pip install -r requirements/profiles/sglang.txt
source scripts/env_base_runtime.sh
.venv-sglang/bin/serve-optimize doctor --profile sglang
```

The profile pins:

* SGLang `0.5.10.post1`
* SGLang Kernel `0.4.1`
* Torch `2.9.1`
* Transformers `5.3.0`
* Hugging Face Hub `1.19.0`
* NVML Python bindings `13.610.43`

The runtime helper verifies GCC Toolset 12 and CUDA `nvcc`, then exports `CC`, `CXX`, `CUDAHOSTCXX`, `CUDA_HOME`, and the required `PATH`.

The validated SGLang launch still requires `--disable-piecewise-cuda-graph`.

## Automated Profile Check

Create and verify a fresh profile environment:

```bash
scripts/verify_install_profile.sh core /tmp/serve-optimize-core
scripts/verify_install_profile.sh telemetry /tmp/serve-optimize-telemetry
scripts/verify_install_profile.sh vllm /tmp/serve-optimize-vllm
scripts/verify_install_profile.sh sglang /tmp/serve-optimize-sglang
```

The target directory must not already exist.

## Verification Evidence

Clean profile verification completed on 2026-06-16.

The reproducible vLLM profile includes the direct NVML binding so telemetry availability does not depend on an ambient package.

## Constraints

Backend constraints live under `requirements/constraints`.

They pin the directly validated runtime packages while allowing each backend package to resolve its remaining transitive dependencies.
The profile files perform standard local package installs, not editable installs.

## Safety

* Use pip installation commands.
* Keep backend environments isolated.
* Do not mutate a validated environment to test another profile.
* Do not disable SSL verification.
* Do not treat the failed latest vLLM environment as validated.
