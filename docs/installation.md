# Reproducible Installation

Serve Optimize keeps vLLM and SGLang in separate environments because their tested Transformers versions differ. The installation scripts use `uv`, which follows the current vLLM guidance for selecting a compatible Torch backend.

## Prerequisites

* Linux with an NVIDIA driver compatible with the selected CUDA wheels
* Python 3.10 or newer
* [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
* An NVIDIA GPU supported by the selected backend
* A C compiler and Python development headers for runtime kernel launchers

On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential python3-dev
```

Blackwell GPUs require CUDA 12.8 or newer for vLLM. See the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) and the [SGLang installation guide](https://docs.sglang.io/docs/get-started/install) for upstream platform requirements.

## Automated Install

Create and verify a fresh environment from the repository root:

```bash
scripts/verify_install_profile.sh core /tmp/serve-optimize-core
scripts/verify_install_profile.sh telemetry /tmp/serve-optimize-telemetry
scripts/verify_install_profile.sh vllm /tmp/serve-optimize-vllm
scripts/verify_install_profile.sh sglang /tmp/serve-optimize-sglang
```

The target directory must not already exist. Each command creates an isolated environment, installs the selected profile, runs dependency checks, and runs `serve-optimize doctor --profile PROFILE`.

## vLLM

Manual installation:

```bash
uv venv --python python3 .venv-vllm
uv pip install \
  --python .venv-vllm/bin/python \
  --torch-backend=auto \
  -r requirements/profiles/vllm.txt
.venv-vllm/bin/serve-optimize doctor --profile vllm
```

Pinned and validated stack:

* vLLM `0.23.0`
* Torch `2.11.0`, with the CUDA wheel selected by `uv`
* Transformers `5.9.0`
* Hugging Face Hub `1.17.0`
* NVML Python bindings `13.610.43`

## SGLang

Manual installation:

```bash
uv venv --python python3 .venv-sglang
uv pip install \
  --python .venv-sglang/bin/python \
  -r requirements/profiles/sglang.txt
.venv-sglang/bin/serve-optimize doctor --profile sglang
```

Pinned profile:

* SGLang `0.5.13.post1`
* FlashAttention 4 `4.0.0b18`, the explicit prerelease required by SGLang
* SGLang Kernel `0.4.3`
* Torch `2.11.0`
* Transformers `5.8.1`
* Hugging Face Hub `1.17.0`
* NVML Python bindings `13.610.43`

The SGLang wheel stack targets CUDA 13. It no longer requires the old validation host's GCC Toolset 12 path. `scripts/env_base_runtime.sh` is only an optional helper for source builds that need an explicit local CUDA toolkit.

## Hugging Face Token

Public models do not need a token. Gated models require both approved browser access on the model page and a read token on the server.

Create a read token at <https://huggingface.co/settings/tokens>, then store it in the active backend environment:

```bash
hf auth login
hf auth whoami
```

For noninteractive runs, export the token instead:

```bash
export HF_TOKEN=hf_your_read_token
```

Hugging Face Hub uses the saved token by default, and `HF_TOKEN` overrides the saved token when set. Request gated model access from the model page in a browser before launching a campaign.

## Core And Telemetry

For development without a serving backend:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e ".[dev,telemetry]"
.venv/bin/serve-optimize doctor --profile telemetry
```

## Current Validation Host

Recorded on 2026-06-23:

| Component | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB |
| NVIDIA driver | 595.71.05 |
| Driver CUDA capability | 13.2 |
| CUDA toolkit | 13.2 |
| vLLM | 0.23.0 |
| Torch | 2.11.0+cu130 in the clean vLLM profile environment |
| Transformers | 5.9.0 |
| Hugging Face Hub | 1.17.0 |
| NVML Python bindings | 13.610.43 |

The vLLM profile passes the profile doctor on this host. Runtime launch evidence is recorded in [Verification](verification.md).
Review [Security Notes](security.md) before production deployment. It records audit commands and the current upstream advisory boundary for optional backend stacks.

## Rules

* Keep backend environments isolated.
* Keep SSL verification enabled.
* Use the profile requirement files for reproducible runs.
* Rerun the profile installer and a real managed smoke test before changing a validated version claim.
