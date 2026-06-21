# Product Readiness

Serve Optimize is ready for broader measured validation of its current product contract:

* Attach Mode for existing OpenAI compatible endpoints.
* Managed Mode for the validated vLLM and SGLang surfaces.
* Runtime fingerprinted evidence, conservative reuse, resume, workload profiles, SLO guards, measurement quality controls, campaign planning, validation, and research packaging.
* Reproducible backend installation profiles, package builds, Python 3.10 through 3.12 CI, release checks, contributor guidance, and a security reporting policy.

## Remaining Before Broad Production Adoption

These items are not solved by testing more model sizes:

1. Static typing debt. The runtime test and lint gates pass, but the repository does not yet pass a strict full source MyPy run. Dynamic artifact parsing, telemetry provider protocols, and orchestration payloads need a dedicated typing pass.
2. Production trace inputs. Built in workload profiles are controlled synthetic workloads, not anonymized production request traces.
3. Distribution publication. CI builds and installs the wheel, but publishing signed releases to a package index still requires repository owner credentials and release policy.
4. Broader environment coverage. Current first class evidence is scoped to the validated vLLM and SGLang stacks and the recorded GPU host environment.
5. Operational orchestration. Containers, Kubernetes, multi node execution, and parallel managed launches remain explicitly outside the current contract.
6. Phase energy attribution. Prefill and decode attribution remains unavailable until a backend exposes defensible phase markers.

## Campaign Readiness

Campaign plans now generate separate executable scripts for each backend environment. A dispatcher selects the backend runner, each runner continues through failed matrix cells, and a separate postprocessing script targets the timestamped managed run directories.

Before launching a campaign:

1. Run `serve-optimize doctor --profile BACKEND` in each isolated backend environment.
2. Generate the plan and inspect `campaign_matrix.csv`.
3. Run `campaign_commands.sh vllm` and `campaign_commands.sh sglang` in their matching environments.
4. Run `campaign_postprocess.sh` after all backend runners complete.
5. Keep claims scoped to usable measured artifacts reported by validation.
