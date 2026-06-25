# Scripts

Repository helper scripts.

* `verify_fast.sh`: local fast release gate.
* `verify_full.sh`: full local release gate.
* `verify_install_profile.sh`: creates and validates a fresh install profile.
* `env_base_runtime.sh`: optional CUDA and compiler setup for source builds.
* `plot_pareto.py`: optional plotting helper for generated Pareto artifacts.

Run scripts from the repository root.

Backend scripts inherit the active shell environment. They do not activate backend environments internally. For managed vLLM, activate `.venv-vllm` first. For managed SGLang, activate `.venv-sglang` first. Generated campaign scripts should be run from the matching backend environment.
