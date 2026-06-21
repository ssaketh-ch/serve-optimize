# Contributing

Serve Optimize is meant to be useful to researchers and practitioners. Contributions are welcome in five areas:

- Hardware profiles and measurement notes.
- Backend improvements for the supported vLLM and SGLang Managed Mode surfaces.
- Attach Mode compatibility notes for external OpenAI compatible endpoints.
- Reproducible benchmark runs with raw telemetry.
- Optimizer, modeling, and plotting improvements.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Benchmark Result Contributions

When contributing benchmark results, include:

- GPU model, VRAM, driver, CUDA version, and power limit.
- MIG profile if applicable.
- Model id and exact revision.
- Backend name and version.
- Full launch command.
- Raw JSONL result file.
- Power sampling method.
- Any known system load or measurement limitations.

Do not submit benchmark results that mix unrelated workloads in the same run artifact.

## Code Style

- Keep backend-specific behavior behind adapter modules.
- Prefer typed dataclasses for artifact records.
- Keep synthetic smoke-test mode working on CPU-only machines.
- Add tests for optimizer logic and parsers.
- Be explicit when a metric is measured, estimated, or synthetic.
- Do not add a Managed Mode backend without explicit lifecycle ownership and an accepted design.
