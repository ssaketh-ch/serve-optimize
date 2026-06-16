# Decision: Measured Runtime Evidence Is Source Of Truth

## Status

Accepted

## Context

Serve Optimize can use AIConfigurator output and prior evidence to reduce search cost, but recommendations should explain measured behavior on the user's system.

## Decision

Measured runtime evidence and generated artifacts are the source of truth for final recommendation quality. Predictions, priors, stale evidence, and near-compatible evidence may guide candidate selection but must remain distinguishable from measured results.

## Consequences

- Recommendation artifacts must keep measured and predicted values explicit.
- Managed Evaluation Mode may skip duplicate work only for exact fresh measured evidence.
- Reports should not imply that Attach Mode proves how a live endpoint was launched.
