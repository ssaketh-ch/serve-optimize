# Research Package

## Phase Nine Status

Phase Nine packages existing managed run artifacts into reproducible research outputs.

It does not run a benchmark campaign. It does not broaden evidence by inference.

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

Recommendation claims remain scoped to best among evaluated candidates.

## Extending Coverage

To broaden the research package, collect fresh runtime fingerprinted evidence for additional:

* models
* hardware
* quantization modes
* backends
* workload profiles
* SLO constraints

Then rerun `serve-optimize research-package` over the expanded run set.

Use `serve-optimize campaign-plan` to create a reproducible command matrix before collecting those runs.
