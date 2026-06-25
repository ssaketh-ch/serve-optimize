# Reproducible Installation

This page is the clean setup path from a fresh server to managed vLLM and SGLang runs.

Serve Optimize keeps vLLM and SGLang in separate Python environments because their tested backend stacks pin different runtime packages. The command line tools and helper scripts do not switch environments for you. They use whichever `serve-optimize`, `vllm`, and `sglang` commands are active in your current shell.

The practical rule is simple:

1. Use the vLLM environment when running vLLM commands.
2. Use the SGLang environment when running SGLang commands.
3. Run generated campaign scripts from the matching backend environment.

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

## Fresh Server Setup

From the repository root, first install the small core toolchain:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e ".[dev,telemetry]"
.venv/bin/serve-optimize doctor --profile telemetry
```

This environment is good for development, telemetry checks, Attach Mode, documentation, and artifact inspection. It is not the recommended environment for managed vLLM or managed SGLang measurements.

Next create the backend environments.

## vLLM Environment

Create a fresh vLLM environment:

```bash
uv venv --python python3 .venv-vllm
uv pip install \
  --python .venv-vllm/bin/python \
  --torch-backend=auto \
  -r requirements/profiles/vllm.txt
.venv-vllm/bin/serve-optimize doctor --profile vllm
```

Activate it before running vLLM managed work:

```bash
source .venv-vllm/bin/activate
serve-optimize doctor --profile vllm
```

If vLLM is already installed, update the environment to the pinned project profile and recheck it:

```bash
uv pip install \
  --python .venv-vllm/bin/python \
  --upgrade \
  --torch-backend=auto \
  -r requirements/profiles/vllm.txt
.venv-vllm/bin/serve-optimize doctor --profile vllm
```

Run a small managed dry run:

```bash
.venv-vllm/bin/serve-optimize optimize Qwen/Qwen3-0.6B \
  --backend vllm \
  --workload-profile short \
  --dry-run \
  --out results/setup-vllm-dry-run
```

Pinned and validated stack:

* vLLM `0.23.0`
* Torch `2.11.0`, with the CUDA wheel selected by `uv`
* Transformers `5.9.0`
* Hugging Face Hub `1.17.0`
* NVML Python bindings `13.610.43`

## SGLang Environment

Create a fresh SGLang environment:

```bash
uv venv --python python3 .venv-sglang
uv pip install \
  --python .venv-sglang/bin/python \
  -r requirements/profiles/sglang.txt
.venv-sglang/bin/serve-optimize doctor --profile sglang
```

Activate it before running SGLang managed work:

```bash
source .venv-sglang/bin/activate
serve-optimize doctor --profile sglang
```

If SGLang is already installed, update the environment to the pinned project profile and recheck it:

```bash
uv pip install \
  --python .venv-sglang/bin/python \
  --upgrade \
  -r requirements/profiles/sglang.txt
.venv-sglang/bin/serve-optimize doctor --profile sglang
```

Run a small managed dry run:

```bash
.venv-sglang/bin/serve-optimize optimize Qwen/Qwen3-0.6B \
  --backend sglang \
  --workload-profile short \
  --dry-run \
  --out results/setup-sglang-dry-run
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

## How Scripts Use Environments

Repository scripts use the active shell environment. They do not activate `.venv-vllm` or `.venv-sglang` internally.

Campaign plan scripts follow the same rule. Run each generated backend script in the matching backend environment, then run the generated postprocess script after both backend runs finish.

## Automated Install

Create and verify a fresh environment from the repository root:

```bash
scripts/verify_install_profile.sh core /tmp/serve-optimize-core
scripts/verify_install_profile.sh telemetry /tmp/serve-optimize-telemetry
scripts/verify_install_profile.sh vllm /tmp/serve-optimize-vllm
scripts/verify_install_profile.sh sglang /tmp/serve-optimize-sglang
```

The target directory must not already exist. Each command creates an isolated environment, installs the selected profile, runs dependency checks, and runs `serve-optimize doctor --profile PROFILE`.

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

Hugging Face Hub uses the saved token by default, and `HF_TOKEN` overrides the saved token when set. Request gated model access from the model page in a browser before launching a managed run.

Runtime launch evidence and verification notes are recorded in [Verification](docs/development/verification.md). Review [Security Notes](docs/security.md) before production deployment. It records audit commands and the current upstream advisory boundary for optional backend stacks.

## Rules

* Keep backend environments isolated.
* Keep SSL verification enabled.
* Use the profile requirement files for reproducible runs.
* Rerun the profile installer and a real managed smoke test before changing a validated version claim.
