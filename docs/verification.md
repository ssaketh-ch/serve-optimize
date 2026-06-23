# Verification

## Standard Commands

```bash
python -m compileall -q src tests
pytest -q
ruff check .
python -m build --no-isolation --outdir /tmp/serve-optimize-dist .
python -m json.tool feature_list.json
serve-optimize --help
serve-optimize optimize --help
serve-optimize validate-campaign --help
serve-optimize campaign-plan --help
serve-optimize release-check --help
serve-optimize research-package --help
```

Fast harness:

```bash
bash scripts/verify_fast.sh
```

Full harness:

```bash
bash scripts/verify_full.sh
```

## Verification Rules

* Run focused tests for narrow behavior changes.
* Run the standard commands before declaring a phase complete.
* Do not weaken or delete tests to make verification pass.
* Do not mark hardware, backend, telemetry, or evidence behavior verified without real commands and inspected artifacts.
* Keep failed and unavailable run artifacts.
* Record release verification in `docs/verification.md` or a release note.

## Product Readiness Hardening

Recorded on 2026-06-22:

* corrected package repository metadata for the actual GitHub remote
* added Python 3.10, 3.11, and 3.12 CI coverage
* added source distribution, wheel, packaged import, and CLI version verification
* added contributor scope and private security reporting guidance
* removed an exposed telemetry option that only recorded a future work note
* corrected Attach Mode idle baseline limitation wording
* fixed duplicate decision numbering
* fixed campaign postprocessing discovery for nested managed run directories
* added executable backend specific campaign runners and a dispatcher for isolated backend environments
* made backend runners continue through individual matrix failures
* focused product readiness tests passed
* clean Python 3.12 development environment verification passed
* 315 tests passed
* 1 test skipped
* Ruff passed
* source distribution and wheel builds passed
* packaged CLI smoke passed
* release check passed with 72 checks

## Correctness And Security Audit

Recorded on 2026-06-23:

* aligned request, token, time, failure, and power accounting to the same measured window
* separated full run counters from measured window counters in benchmark summaries
* removed stream chunk token proxies and recorded token count provenance
* added authenticated endpoint support through environment variable names without serializing secrets
* restricted endpoint base URLs to HTTP and HTTPS without embedded credentials
* required complete goal metrics and applied real score penalties for missing optional metrics
* rejected nonfinite recommendation metrics and invalid SLO constraints
* removed fabricated local benchmark power fallback and corrected padded token accounting
* added model revision and GPU index propagation for local real benchmarks
* corrected balanced Pareto normalization for two point tradeoffs
* dependency audit found no known vulnerabilities
* Bandit medium and high severity scan passed
* 336 tests passed
* 1 test skipped
* Ruff passed
* source distribution and wheel builds passed
* packaged CLI smoke passed
* release check passed with 72 checks

## Phase 5 Measurement Depth

Recorded on 2026-06-21:

* streaming endpoint timing tests passed
* TTFT and stream chunk TPOT summary tests passed
* soak duration request extension tests passed
* thermal trend and stability reporting tests passed
* campaign plan measurement option tests passed
* compilation passed
* 304 tests passed
* 1 test skipped
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed
* release check passed with 51 checks

## Phase 6 Backend Expansion

Recorded on 2026-06-22:

* TensorRT LLM was decided out of current Managed Mode scope and remains planned only
* the nonfunctional TensorRT LLM placeholder adapter was removed
* the future engine build lifecycle admission gate was recorded in an accepted decision
* TGI, LMDeploy, llama.cpp, and NIM were fixed as external Attach Mode only targets
* Managed Mode factory rejection tests passed for planned and Attach Mode only engines
* release checks now enforce backend registration, support manifests, decision documentation, and adapter absence
* compilation passed
* 313 tests passed
* 1 test skipped
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed
* release check passed with 60 checks

## Phase One Baseline

Recorded on 2026-06-16:

* fresh runtime fingerprinted vLLM measurement passed
* identical vLLM repeat used exact fresh evidence with zero launches and zero measurements
* fresh runtime fingerprinted SGLang measurement passed
* identical SGLang repeat used exact fresh evidence with zero launches and zero measurements
* SGLang rendered and fingerprinted `--disable-piecewise-cuda-graph`
* four run validation campaign passed with 4 of 4 usable runs
* six runtime drift tests passed
* six persistent lifecycle failure tests passed
* compilation passed
* 264 tests passed
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed

Artifacts:

* `results/phase1-runtime-evidence-v2`
* `results/phase1-failure-lifecycle`

## Phase Five Baseline

Recorded on 2026-06-16:

* workload profile and manifest tests passed
* token distribution fingerprint drift test passed
* SLO recommendation eligibility tests passed
* managed workload profile artifact tests passed
* compilation passed
* 280 tests passed
* Ruff passed
* feature JSON validation passed
* root, Managed Mode, and campaign CLI help checks passed

## Phase Six Baseline

Recorded on 2026-06-16:

* idle subtraction tests passed
* warmup and steady state tests passed
* trial aggregation confidence and stability tests passed
* managed aggregate evidence tests passed
* compilation passed
* 285 tests passed
* Ruff passed
* feature JSON validation passed
* root, Managed Mode, and campaign CLI help checks passed

## Phase Seven Baseline

Recorded on 2026-06-16:

* optimizer quality tests passed
* managed optimizer artifact tests passed
* managed failure cache artifact tests passed
* compilation passed
* 286 tests passed
* Ruff passed
* feature JSON validation passed

## Phase Eight Baseline

Recorded on 2026-06-16:

* release check tests passed
* release readiness command passed
* wheel build passed
* full verification script updated
* CI workflow added
* release documentation passed local checks
* support matrix passed local checks

## Phase Nine Baseline

Recorded on 2026-06-16:

* research package tests passed
* research package CLI help passed
* research package artifacts verified
* coverage table generation verified
* methodology artifact generation verified
* full repository verification passed with 290 tests

## Phase One Roadmap Truth And Polish

Recorded on 2026-06-21:

* stale vLLM and SGLang support statuses were reconciled in config files
* release wording was corrected for local and hosted use
* core profile documentation was aligned with the profile file
* release readiness checks were extended to catch support truth drift
* focused release check tests passed
* full repository verification passed with 291 tests and 1 skipped test
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed
* release check passed with 50 checks

## Phase Two Roadmap Artifact Readability

Recorded on 2026-06-21:

* `load_result_jsonl` now reads current `BenchmarkResult` JSONL artifacts
* result loading supports common legacy aliases for result metrics and serving config fields
* result loading reports invalid JSON rows and missing throughput metrics with `ValueError`
* focused IO tests passed
* full repository verification passed with 295 tests and 1 skipped test
* Ruff passed
* required CLI help checks passed
* release check passed with 50 checks

## Phase Three Roadmap Managed Resume

Recorded on 2026-06-21:

* Managed Mode supports `--resume-from` for previous managed run directories
* resume reuses only completed measured workloads with matching candidate, workload, launch, and workload identities
* resumed workloads write `resume_skip` lifecycle records and avoid backend launch
* workload identity drift falls back to normal measurement
* focused managed resume tests passed
* full repository verification passed with 297 tests and 1 skipped test
* Ruff passed
* required CLI help checks passed
* release check passed with 50 checks

## Phase Four Roadmap Campaign Planning

Recorded on 2026-06-21:

* added `serve-optimize campaign-plan`
* campaign plans write a managed run matrix, shell command script, text summary, and JSON manifest
* campaign planning does not launch servers or create measured evidence
* post run validation and research package commands are included
* focused campaign plan tests passed
* release check tests passed
* full repository verification passed with 300 tests and 1 skipped test
* Ruff passed
* required CLI help checks passed
* release check passed with 51 checks

## Security audit

Latest dependency audit results are recorded in [Security Notes](security.md). The default project environment reported no known vulnerabilities. Optional backend profile advisories remain bounded to upstream vLLM, SGLang, and torch pins with no compatible fixed backend release available at the time of validation.

## Current Validation Environment

Recorded on 2026-06-23:

### vLLM

* profile: requirements/profiles/vllm.txt
* vLLM: 0.23.0
* Torch: 2.11.0+cu130 in the clean profile environment
* Transformers: 5.9.0
* Python: 3.12.3
* GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB
* NVIDIA driver: 595.71.05
* CUDA driver capability and toolkit: 13.2
* profile doctor: passed
* live smoke: Qwen/Qwen3-0.6B, 2 candidates, short streaming workload, selected true backend default
* measured smoke result: 8,306 total tokens/s, p95 latency 137.9 ms, active energy 0.015457 J/token, good telemetry, 307 TTFT and TPOT samples

### SGLang

* profile: requirements/profiles/sglang.txt
* SGLang: 0.5.13.post1
* Torch: 2.11.0
* Transformers: 5.8.1
* Python: 3.12.3
* clean dependency resolution: passed
* profile doctor: passed
* live smoke: Qwen/Qwen3-0.6B, baseline candidate, short streaming workload, selected true backend default
* measured smoke result: 8,561 total tokens/s, p95 latency 127.8 ms, active energy 0.014066 J/token, good telemetry, 159 TTFT and TPOT samples

## Known Verification Limits

* Installation is not yet tested across a multi platform release matrix.
* SGLang runtime validation is a single baseline smoke on Qwen/Qwen3-0.6B, not a broad option matrix.
* TensorRT LLM is not implemented.
* Prefill and decode phase attribution is not verified.
* Broad production workload and SLO coverage is not verified.

## Phase Two Result

Recorded on 2026-06-16:

* documentation contradiction audit passed
* all local Markdown links resolved
* documented commands matched CLI help
* compilation passed
* 264 tests passed
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed

## Phase Three Result

Recorded on 2026-06-16:

* clean standard core installation passed
* clean standard telemetry installation passed
* clean standard vLLM installation passed
* clean standard SGLang installation passed
* all four package installs were noneditable
* all four clean environments passed `pip check`
* all four `doctor --profile` checks passed
* vLLM and SGLang commands resolved inside their active environments
* SGLang runtime help, GCC Toolset 12.2.1, and CUDA 12.8 checks passed
* wheel build and metadata inspection passed
* compilation passed
* 269 tests passed
* Ruff passed
* feature JSON validation passed
* shell syntax and required CLI help checks passed

Artifacts:

* `results/phase3-installation`

## Phase Four Result

Recorded on 2026-06-16:

* Attach Mode dry run wrote `preflight.json`, `preflight.txt`, and candidate plan artifacts without endpoint access
* Managed vLLM dry run wrote preflight, rendered launch, workload, launch group, and validation artifacts without launch
* Managed SGLang dry run passed in the validated SGLang profile
* SGLang dry run rendered `--disable-piecewise-cuda-graph`
* focused dry run tests passed
* compilation passed
* 271 tests passed
* Ruff passed
* feature JSON validation passed
* required CLI help checks passed
