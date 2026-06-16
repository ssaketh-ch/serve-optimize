# Architecture Rules

Serve Optimize recommends LLM serving configurations from measured runtime behavior. Keep these rules easy to find and hard to bypass.

## Invariants

- Serve Optimize is the product. AIConfigurator is an input, not the product.
- Measured runtime evidence and written artifacts are the source of truth for recommendations.
- AIConfigurator predictions are candidate sources and priors only.
- Attach Mode evaluates an already-running OpenAI-compatible endpoint and does not prove how that endpoint was launched.
- Managed Evaluation Mode may own server lifecycle only through backend lifecycle adapters.
- Final recommendation quality depends on measured evidence or exact fresh measured evidence, not stale priors.
- Telemetry fields are optional and generic. Missing fields must not be confused with true zero values.
- MIG telemetry caveats are notes unless the benchmark actually fails.

## Module Boundaries

- CLI commands should orchestrate existing layers instead of duplicating benchmark, scoring, or telemetry logic.
- Candidate generation, planning, benchmark execution, telemetry collection, recommendation scoring, and reporting should remain separate enough to test directly.
- Backend-specific launch details belong behind backend adapters.
- Managed backends must keep help parsing, command rendering, unsupported-field handling, and canonical launch metadata behind their backend adapter.
- OpenAI-compatible HTTP is the Attach Mode interface boundary.
- Artifact schemas should stay inspectable and automation-friendly.
- `docs/compatibility.md` is authoritative for current support and exclusion claims.

## Dependency Direction

- User-facing commands may depend on planning, benchmark, telemetry, evidence, and reporting modules.
- Core scoring and reporting should not depend on a specific telemetry provider implementation.
- Telemetry providers should emit generic records and capabilities rather than provider-specific downstream behavior.
- Managed lifecycle code should call backend adapters instead of shelling out from scoring or reporting code.

## Critical Workflows

- Attach Mode: running server -> `serve-optimize recommend` -> candidate generation -> benchmark -> telemetry -> scoring -> recommendation artifacts.
- Managed Evaluation Mode: generate candidates -> check evidence -> apply priors only when evidence is weak or missing -> staged evaluation -> backend launch -> endpoint benchmark -> telemetry -> stop process group -> artifacts.
- Telemetry check: sample provider without inference -> write samples, summary, capabilities, and report.

## Do Not Casually Change

- Do not rewrite Attach Mode while validating Managed Evaluation Mode.
- Do not treat AIConfigurator output as measured evidence.
- Do not add new telemetry providers unless explicitly requested.
- Do not make provider-specific telemetry fields required.
- Do not weaken tests or delete artifact outputs to simplify implementation.
- Do not claim idle-subtracted or phase-attributed energy unless implemented and verified.
- Do not add TensorRT-LLM, Kubernetes, power limits, or parallel candidate execution to Managed Mode unless explicitly scoped.
- Treat SGLang as first class only for the capability detected Managed Mode surface. Do not imply universal option parity or exact evidence reuse without current runtime fingerprints.
