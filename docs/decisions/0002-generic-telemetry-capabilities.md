# Decision: Telemetry Capabilities Stay Generic

## Status

Accepted

## Context

Telemetry providers expose different fields across GPUs, drivers, and MIG environments. Missing utilization can be a platform limitation even when power and memory fields are present.

## Decision

Telemetry capability reporting uses generic field names and stays additive to the existing scoring logic. Missing provider fields are reported as capability limitations or notes when supported by sample evidence.

## Consequences

- Recommendation confidence can mention missing capability context without changing ranking policy.
- MIG utilization caveats are informational unless the benchmark fails.
- Provider-specific fields remain optional and should not leak into scoring contracts.
