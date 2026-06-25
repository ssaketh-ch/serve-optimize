# Release Engineering

## Current Release Gate

Serve Optimize uses a local release gate in this workspace. The CI workflow file is present for hosted use, but local verification remains the source of truth before publishing or deploying changes.

This page records the Phase Eight release engineering gate for the supported project surface.

## Local Release Gate

Run:

```bash
bash scripts/verify_full.sh
```

The full gate checks:

* Python compilation for `src` and `tests`
* full test suite
* Ruff lint
* source distribution and wheel builds
* packaged wheel import and CLI version smoke
* feature list JSON validity
* required CLI help surfaces
* release readiness artifacts

The lighter development gate is:

```bash
bash scripts/verify_fast.sh
```

## Release Check Command

Run:

```bash
serve-optimize release-check --out results/release-check
```

Artifacts:

* `release_check.json`
* `release_check.txt`

The release check inspects packaging metadata, verification scripts, support documents, schema markers, and required files. It does not run backend measurements.

Hosted CI runs the fast gate on Python 3.10, 3.11, and 3.12. A separate package job builds the source distribution and wheel, installs the wheel, and checks the installed CLI version.

## Build And Installation Policy

Package metadata lives in `pyproject.toml`. Backend runtime installs stay split across the validated profiles:

* core
* telemetry
* vLLM
* SGLang

The vLLM and SGLang runtime profiles remain separate because their Torch and Transformers stacks are incompatible.

## Schema And Migration Policy

Machine readable artifacts carry explicit schema versions. Backward compatibility is preserved by accepting missing optional fields in older artifacts where the product already did so.

No database migration command is required for the current evidence schema. Future migrations must be documented in this file and in `docs/development/verification.md`.

## Cleanup Policy

Generated artifacts belong under `results/` or an explicit user supplied output directory. Tests and release checks must not require deleting user artifacts.

## Research Artifact Policy

Research packaging uses existing managed run artifacts and must not imply coverage for models, hardware, workloads, or backends that were not present in the supplied evidence.
