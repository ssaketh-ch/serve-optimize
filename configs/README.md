# Configs

Static YAML files used by the legacy synthetic and planning paths.

These files are lightweight defaults and examples. Managed Mode derives most runtime behavior from backend capability detection, model metadata, workload profiles, and CLI options rather than requiring edits here.

`overnight_gated_models.tsv` lists optional gated model ids only. Store Hugging Face tokens in the environment or local Hugging Face auth cache, not in this directory.

Do not put host specific secrets, local paths, or measured evidence in this directory.
