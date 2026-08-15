# Experiments

Example experiment configuration files for synthetic and planned evaluation workflows.

Current first class measured workflows are driven through the CLI:

```bash
serve-optimize managed-evaluate --help
serve-optimize campaign-plan --help
serve-optimize benchmark-matrix-plan --help
serve-optimize validate-campaign --help
serve-optimize research-package --help
```

Treat files in this directory as templates. Copy and adapt them outside the repository when running host specific experiments.

The H200 Slurm templates live under `scripts/slurm`. Adjust the partition and GPU resource directives for the target cluster. Set `SERVE_OPTIMIZE_ROOT` when the submission directory is not the repository root. The Stage 1 runner also requires `SERVE_OPTIMIZE_PLAN_ROOT` and `SERVE_OPTIMIZE_RUN_ROOT`; the closure runner requires `SERVE_OPTIMIZE_CLOSURE_ROOT`.

The research closure runner additionally requires `SERVE_OPTIMIZE_SOURCE_TARBALL` and `SERVE_OPTIMIZE_SOURCE_SHA256`. Backend and benchmark environment paths may be overridden with `SERVE_OPTIMIZE_VLLM_ENV`, `SERVE_OPTIMIZE_SGLANG_ENV`, and `SERVE_OPTIMIZE_GUIDELLM_ENV`. Use `SERVE_OPTIMIZE_MODEL_SCOPE` and `SERVE_OPTIMIZE_BACKEND_SCOPE` to split a campaign without changing its measurement protocol.

Generated plans and raw results are external experiment artifacts. Preserve them with the source revision, environment capture, and checksums used for the run.

The closure runner accepts `SERVE_OPTIMIZE_MODEL_SCOPE=wide` for the publication matrix. This scope covers Qwen 0.6B, Qwen 1.5B, DeepSeek distilled 1.5B, Granite 2B, Qwen 7B, Mistral 7B, Qwen 8B, Granite 8B, Qwen 3.5 9B, Qwen 3.6 27B, Qwen 3.8 27B in BF16 and FP8, and Llama 3.1 8B when access is approved. For each model and backend it runs short, medium, and long prefill saturation, mixed, code generation, repeated prefix, permitted real chat, three balanced repetitions, and three GuideLLM cross checks. The full scope is 260 managed runs and 78 GuideLLM reports per hardware host. The Qwen3.8 models are evaluated through their text serving path; this does not claim multimodal performance. A model that cannot load on a backend remains an explicit failure or unsupported result rather than being silently removed from the matrix.
