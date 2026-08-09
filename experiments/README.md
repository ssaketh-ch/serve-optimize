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

The H200 Slurm templates live under `scripts/slurm`. Adjust the partition and GPU resource directives for the target cluster. Set `SERVE_OPTIMIZE_ROOT` when the submission directory is not the repository root. The Stage 1 runner also requires `SERVE_OPTIMIZE_PLAN_ROOT` and `SERVE_OPTIMIZE_RUN_ROOT`; the closure runner requires `SERVE_OPTIMIZE_CLOSURE_ROOT`. Backend environment paths may be overridden with `SERVE_OPTIMIZE_VLLM_ENV` and `SERVE_OPTIMIZE_SGLANG_ENV`.

Generated plans and raw results are external experiment artifacts. Preserve them with the source revision, environment capture, and checksums used for the run.
