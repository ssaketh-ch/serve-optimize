# Release Notes

## 0.1.0

First public release of Serve Optimize.

Highlights:

* Attach Mode for existing OpenAI compatible endpoints.
* Managed Mode for vLLM and the detected supported SGLang surface.
* Capability aware candidate validation and canonical launch rendering.
* Runtime fingerprinted evidence reuse.
* OpenAI compatible health checks and endpoint benchmarks.
* Optional NVIDIA telemetry.
* Workload profiles, JSON workload manifests, and SLO eligibility guards.
* Measurement quality controls for warmup, steady state windows, and idle baseline energy.
* Optimizer quality artifacts with bounded evaluated candidate regret.
* Validation campaign and repeatability artifacts.
* Release readiness checks.
* Research package generation from existing managed run artifacts.

Validated backend stacks:

* vLLM `0.10.0`
* SGLang `0.5.10.post1`

Known limits:

* TensorRT LLM Managed Mode is not implemented.
* Latest vLLM is not validated.
* vLLM and SGLang require separate environments.
* New hardware requires fresh runtime fingerprinted evidence before exact reuse.
* Prefill and decode energy attribution is not implemented.
* Recommendations are scoped to best among evaluated candidates.
