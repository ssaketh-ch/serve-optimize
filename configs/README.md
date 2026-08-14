# Configs

Static configuration files used by the synthetic and planning paths.

These files are lightweight defaults and examples. Managed Mode derives most runtime behavior from backend capability detection, model metadata, workload profiles, and CLI options rather than requiring edits here.

Do not put host specific secrets, local paths, or measured evidence in this directory.

`workloads/` contains versioned JSON manifests for permitted prompt data and fixed synthetic saturation shapes.
