import json

from serve_optimize.schemas import PowerSampleRecord
from serve_optimize.telemetry import TelemetryCapture
from serve_optimize.telemetry_check import run_telemetry_check


def test_telemetry_check_writes_artifacts(tmp_path) -> None:
    run = run_telemetry_check(
        telemetry="nvml",
        duration_s=0.01,
        interval_s=0.2,
        out_dir=tmp_path,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[
                    PowerSampleRecord(float(index), "measured", 100.0 + index, telemetry, provider=telemetry, gpu_util_percent=70.0 + index)
                    for index in range(5)
                ],
                warnings=[],
            )
        ),
    )

    assert (run.run_dir / "samples.jsonl").exists()
    assert (run.run_dir / "telemetry_summary.json").exists()
    assert (run.run_dir / "telemetry_capabilities.json").exists()
    assert (run.run_dir / "report.txt").exists()
    summary = json.loads((run.run_dir / "telemetry_summary.json").read_text(encoding="utf-8"))
    assert summary["telemetry_provider"] == "nvml"
    assert summary["telemetry_available"] is True
    assert summary["sample_count"] == 5
    assert summary["power_stats"]["avg"] == 102.0
    assert "Power" in (run.run_dir / "report.txt").read_text(encoding="utf-8")
    capabilities = json.loads((run.run_dir / "telemetry_capabilities.json").read_text(encoding="utf-8"))
    assert capabilities["provider"] == "nvml"
    assert "power" in capabilities["available_fields"]
    assert "Telemetry Capabilities" in (run.run_dir / "report.txt").read_text(encoding="utf-8")


def test_telemetry_check_records_nvidia_smi_loop_note(tmp_path) -> None:
    run = run_telemetry_check(
        telemetry="nvml",
        duration_s=0.01,
        interval_s=0.2,
        out_dir=tmp_path,
        with_nvidia_smi_loop=True,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[PowerSampleRecord(0.0, "measured", 100.0, telemetry, provider=telemetry)],
                warnings=[],
            )
        ),
    )

    summary = json.loads((run.run_dir / "telemetry_summary.json").read_text(encoding="utf-8"))
    assert any("nvidia-smi loop comparison" in note for note in summary["notes"])


class _FakeCollector:
    def __init__(self, capture: TelemetryCapture):
        self.capture = capture

    def start(self) -> None:
        return None

    def stop(self) -> TelemetryCapture:
        return self.capture
