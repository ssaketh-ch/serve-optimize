# Scripts

Repository helper scripts.

* `verify_fast.sh`: local fast release gate.
* `verify_full.sh`: full local release gate.
* `verify_install_profile.sh`: creates and validates a fresh install profile.
* `env_base_runtime.sh`: optional CUDA and compiler setup for source builds.
* `plot_pareto.py`: optional plotting helper for generated Pareto artifacts.
* `audit_guidellm_results.py`: validates GuideLLM 0.7.3 request, metric, percentile, and concurrency coverage artifacts.
* `analyze_recommendation_search.py`: replays equal budget searches and supported recommendation ablations.
* `run_research_validation.py`: executes fresh bounded search baselines and a measured oracle on the representative paper subset.
* `slurm/run_research_closure_h200.sbatch`: resumable research closure campaign with model and backend scopes. Use `SERVE_OPTIMIZE_MODEL_SCOPE=wide` for the multi family matrix, including the Qwen3.8 27B release.
* `slurm/run_research_validation.sbatch`: dependency scheduled validation campaign for executed search baselines, repeated cells, and the held out prompt split.
* Set `SERVE_OPTIMIZE_DRIVER_LIB_ROOT` to the node's matching NVIDIA user libraries when the system NVML library does not match the loaded kernel driver.

Run scripts from the repository root.

Backend scripts inherit the active shell environment. They do not activate backend environments internally. For managed vLLM, activate `.venv-vllm` first. For managed SGLang, activate `.venv-sglang` first. Generated campaign scripts should be run from the matching backend environment.
