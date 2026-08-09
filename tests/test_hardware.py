from io import StringIO
from types import SimpleNamespace

from serve_optimize.hardware import _apply_visible_mig_memory, _detect_cpu_model
from serve_optimize.schemas import GpuDevice


def test_cpu_model_prefers_description_over_processor_index(monkeypatch) -> None:
    cpuinfo = StringIO("processor : 0\nmodel name : Example CPU 9000\n")
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: cpuinfo)

    assert _detect_cpu_model() == "Example CPU 9000"


def test_visible_mig_memory_replaces_parent_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "serve_optimize.hardware.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='[{"name": "Example MIG 3g", "total_memory_mb": 71424, "free_memory_mb": 70000}]'
        ),
    )
    notes = []
    parent = GpuDevice(
        index=0,
        name="Example Parent GPU",
        total_memory_mb=143771,
        free_memory_mb=143000,
        mig_mode="enabled",
        source="pynvml",
    )

    detected = _apply_visible_mig_memory([parent], notes)

    assert detected[0].name == "Example MIG 3g"
    assert detected[0].total_memory_mb == 71424
    assert detected[0].free_memory_mb == 70000
    assert detected[0].raw["parent_total_memory_mb"] == 143771
    assert "CUDA-visible" in notes[0]
