# Research Package

## Phase Nine Status

Phase Nine indexes existing managed run artifacts into analysis ready research outputs.

It does not run a benchmark campaign, copy raw run directories, or broaden evidence by inference. A package is not a self contained reproducibility archive.

## Command

```bash
serve-optimize research-package RUN_DIR... --out results/research-package
```

Inputs must be managed run directories containing the usual recommendation and campaign artifacts.

Outputs:

* `research_package.json`
* `validation_campaign.json`
* `methodology.md`
* `runs.csv`
* `coverage.csv`

## Methodology

The package records:

* supplied run directories
* usable run count
* backend coverage
* goal coverage
* workload profile coverage
* model coverage
* dtype coverage
* quantization coverage
* telemetry quality coverage
* validation campaign summary
* backend and runtime versions
* runtime fingerprints
* output and total token throughput
* request rate and streaming latency fields
* joules per generated token and tokens per joule
* energy accounting mode
* client and load saturation status

Recommendation claims remain scoped to best among evaluated candidates.

Campaign repeatability is reported only when supplied runs share backend version, model, goal, and workload profile. A heterogeneous matrix is coverage evidence, not repeatability evidence.

## Raw Artifact Archive

For a paper release, archive every referenced managed run directory separately. Preserve request records, power samples, lifecycle events, backend logs, launch specifications, capability reports, failure records, summaries, and runtime metadata. Generate checksums for the archive and verify them before publication.

`research_package.json` records that raw artifacts are not embedded. Its run paths are references to the source directories used during packaging.

## Extending Coverage

To broaden the research package, collect fresh runtime fingerprinted evidence for additional:

* models
* hardware
* quantization modes
* backends
* workload profiles
* SLO constraints

Then rerun `serve-optimize research-package` over the expanded run set.

Use `serve-optimize benchmark-matrix-plan` for the staged paper matrix or `serve-optimize campaign-plan` for a smaller custom matrix before collecting those runs.
