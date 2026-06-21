# Support Matrix

## Product Modes

| Surface | Status | Notes |
|---|---|---|
| Attach Mode | First class | Existing OpenAI compatible endpoints only. |
| Managed Mode | First class | vLLM and detected SGLang surface only. |
| Evidence reuse | First class | Exact reuse requires matching runtime fingerprints. |
| Validation campaign | First class | Uses existing managed run artifacts. |
| Research package | First class | Packages existing artifacts for analysis. |

## Managed Backends

| Backend | Status | Validated Runtime | Managed Mode |
|---|---|---|---|
| vLLM | First class | `0.10.0` | Supported. |
| SGLang | First class for detected surface | `0.5.10.post1` | Supported with runtime helper. |
| TensorRT LLM | Planned only | none | Managed Mode is not in current scope; Attach Mode may measure an external compatible endpoint. |
| TGI | Attach only | none | Not supported. |
| LMDeploy | Attach only | none | Not supported. |
| llama.cpp | Attach only | none | Not supported. |

## Installation Profiles

| Profile | Status | Purpose |
|---|---|---|
| core | Verified | CLI, docs, schema, synthetic paths. |
| telemetry | Verified | Optional host telemetry. |
| vLLM | Verified | Managed vLLM runtime. |
| SGLang | Verified | Managed SGLang runtime. |

## Evidence And Recommendation Claims

Recommendations are scoped to best among evaluated candidates.

Exact evidence reuse requires compatible hardware, backend, runtime, capability hash, rendered command, canonical config, model, workload, telemetry requirements, and measurement quality policy.

## Research Coverage

Research packages report only the coverage present in supplied managed run artifacts:

* backends
* goals
* workload profiles
* models
* dtypes
* quantization modes
* telemetry quality

Broader claims require additional fresh runtime fingerprinted evidence.
