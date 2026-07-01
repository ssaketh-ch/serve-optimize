"""Power telemetry utilities for Attach Mode benchmarking."""

from __future__ import annotations

import csv
import io
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from .schemas import PowerSampleRecord, TelemetryCapabilities, TelemetrySummary

TELEMETRY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "power_watts": ("power_watts", "watts"),
    "gpu_util_percent": ("gpu_util_percent", "gpu_utilization_pct"),
    "memory_util_percent": ("memory_util_percent",),
    "memory_used_mb": ("memory_used_mb", "gpu_memory_used_mb"),
    "memory_total_mb": ("memory_total_mb",),
    "temperature_c": ("temperature_c", "gpu_temperature_c"),
    "graphics_clock_mhz": ("graphics_clock_mhz",),
    "sm_clock_mhz": ("sm_clock_mhz",),
    "memory_clock_mhz": ("memory_clock_mhz",),
    "power_limit_watts": ("power_limit_watts",),
    "enforced_power_limit_watts": ("enforced_power_limit_watts",),
    "throttle_reasons": ("throttle_reasons",),
    "device_name": ("device_name",),
}
MAJOR_TELEMETRY_FIELDS = ("power_watts", "gpu_util_percent")
FLAT_POWER_STDDEV_THRESHOLD_W = 2.0
FLAT_POWER_COV_THRESHOLD = 0.01
THERMAL_SOAK_MIN_DURATION_S = 60.0
THERMAL_STABLE_MAX_RISE_C = 5.0
THERMAL_STABLE_MAX_SLOPE_C_PER_MIN = 1.0
CAPABILITY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "power": ("power_watts", "watts"),
    "temperature": ("temperature_c", "gpu_temperature_c"),
    "memory_usage": ("memory_used_mb", "gpu_memory_used_mb", "memory_total_mb"),
    "gpu_utilization": ("gpu_util_percent", "gpu_utilization_pct"),
    "memory_utilization": ("memory_util_percent",),
    "clocks": ("graphics_clock_mhz", "sm_clock_mhz", "memory_clock_mhz"),
    "power_limit": ("power_limit_watts", "enforced_power_limit_watts"),
    "throttle_reasons": ("throttle_reasons",),
}


@dataclass(frozen=True)
class TelemetryCapture:
    provider: str | None
    samples: list[PowerSampleRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TelemetryCollector:
    def __init__(self, telemetry: str, device_index: int = 0, interval_s: float = 0.2):
        self.telemetry = telemetry
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[PowerSampleRecord] = []
        self._warnings: list[str] = []
        self._provider: str | None = None
        self._last_cpu_times: tuple[int, int] | None = None

    def start(self) -> None:
        if self.telemetry == "none":
            return
        provider = _resolve_provider(self.telemetry)
        if provider is None:
            self._warnings.append(f"Telemetry provider '{self.telemetry}' is unsupported.")
            self._samples.append(self._annotate_sample(_error_sample(self.telemetry, self.device_index, f"Unsupported telemetry provider: {self.telemetry}")))
            return
        self._provider = provider
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetryCapture:
        if self.telemetry == "none":
            return TelemetryCapture(provider=None, samples=[], warnings=[])
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return TelemetryCapture(provider=self._provider, samples=self._samples, warnings=self._warnings)

    def _run(self) -> None:
        assert self._provider is not None
        if self._provider == "nvml":
            self._run_nvml()
            return
        if self._provider == "nvidia-smi":
            self._run_nvidia_smi()
            return
        self._warnings.append(f"Telemetry provider '{self._provider}' is unsupported.")
        self._samples.append(_error_sample(self._provider, self.device_index, f"Unsupported telemetry provider: {self._provider}"))

    def _run_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            self._warnings.append(f"NVML telemetry unavailable: {exc.__class__.__name__}: {exc}")
            self._samples.append(self._annotate_sample(_error_sample("nvml", self.device_index, f"{exc.__class__.__name__}: {exc}")))
            return
        try:
            while not self._stop.is_set():
                self._samples.append(self._annotate_sample(_sample_nvml(pynvml, handle, self.device_index)))
                time.sleep(self.interval_s)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run_nvidia_smi(self) -> None:
        while not self._stop.is_set():
            sample = self._annotate_sample(_sample_nvidia_smi(self.device_index))
            self._samples.append(sample)
            if sample.error:
                self._warnings.append(f"nvidia-smi telemetry sample failed: {sample.error}")
                return
            time.sleep(self.interval_s)

    def _annotate_sample(self, sample: PowerSampleRecord) -> PowerSampleRecord:
        return replace(
            sample,
            sample_interval_s=self.interval_s,
            cpu_util_percent=self._sample_cpu_util_percent(),
        )

    def _sample_cpu_util_percent(self) -> float | None:
        current = _read_proc_stat_cpu_times()
        if current is None:
            return None
        previous = self._last_cpu_times
        self._last_cpu_times = current
        if previous is None:
            return None
        idle_delta = current[1] - previous[1]
        total_delta = current[0] - previous[0]
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))


def make_telemetry_collector(telemetry: str, device_index: int = 0, interval_s: float = 0.2) -> TelemetryCollector:
    return TelemetryCollector(telemetry=telemetry, device_index=device_index, interval_s=interval_s)


def detect_telemetry_capabilities(
    samples: list[PowerSampleRecord],
    *,
    provider: str | None = None,
    warnings: list[str] | None = None,
    mig_mode: str | None = None,
    mig_profile: str | None = None,
    device_name: str | None = None,
) -> TelemetryCapabilities:
    provider = provider or _first_text([sample.provider or sample.source for sample in samples])
    device_name = device_name or _first_text([sample.device_name for sample in samples])
    mig_mode = mig_mode or _first_text([sample.mig_mode for sample in samples])
    mig_profile = mig_profile or _first_text([sample.mig_profile for sample in samples])
    available_fields: list[str] = []
    unavailable_fields: list[str] = []
    for field_name, aliases in CAPABILITY_FIELD_ALIASES.items():
        if any(_sample_has_value(sample, aliases) for sample in samples):
            available_fields.append(field_name)
        else:
            unavailable_fields.append(field_name)

    notes: list[str] = []
    capability_warnings = list(warnings or [])
    if not provider:
        capability_warnings.append("No telemetry provider was selected.")
    if not samples:
        capability_warnings.append("No telemetry samples were collected.")
    if samples and provider and ("gpu_utilization" in unavailable_fields or "memory_utilization" in unavailable_fields):
        notes.append("Platform reports utilization metrics as unavailable.")
        notes.append("Recommendation confidence may be reduced when power telemetry lacks utilization context.")
    if _mig_active(mig_mode, mig_profile, device_name):
        notes.append("MIG environments may not expose utilization counters depending on driver/platform support.")

    return TelemetryCapabilities(
        provider=provider,
        device_name=device_name,
        available_fields=available_fields,
        unavailable_fields=unavailable_fields,
        notes=_unique_preserve_order(notes),
        warnings=sorted(set(capability_warnings)),
    )


def summarize_power_samples(samples: list[PowerSampleRecord], wall_time_s: float, total_tokens: int) -> dict[str, object]:
    summary = summarize_telemetry(samples, wall_time_s, total_tokens)
    return {
        "power_sample_count": summary.power_sample_count,
        "average_power_watts": summary.average_power_watts,
        "min_power_watts": summary.min_power_watts,
        "max_power_watts": summary.max_power_watts,
        "peak_power_watts": summary.peak_power_watts,
        "power_stddev_watts": summary.power_stddev_watts,
        "power_sampling_duration_s": summary.power_sampling_duration_s,
        "power_sampling_rate_hz": summary.power_sampling_rate_hz,
        "energy_joules": summary.energy_joules,
        "joules_per_token": summary.joules_per_token,
        "tokens_per_second_per_watt": summary.tokens_per_second_per_watt,
        "telemetry_provider": summary.telemetry_provider,
        "telemetry_available": summary.telemetry_available,
        "telemetry_quality": summary.telemetry_quality,
        "telemetry_warnings": summary.telemetry_warnings,
        "telemetry_notes": summary.telemetry_notes,
        "sample_count": summary.sample_count,
        "duration_s": summary.duration_s,
        "sampling_rate_hz": summary.sampling_rate_hz,
        "sample_interval_s": summary.sample_interval_s,
        "missing_fields": summary.missing_fields,
        "warnings": summary.warnings,
        "notes": summary.notes,
        "telemetry_capabilities": summary.telemetry_capabilities,
        "power_stats": summary.power_stats,
        "utilization_stats": summary.utilization_stats,
        "thermal_stats": summary.thermal_stats,
        "clock_stats": summary.clock_stats,
        "average_gpu_util_percent": summary.average_gpu_util_percent,
        "max_gpu_util_percent": summary.max_gpu_util_percent,
        "average_memory_util_percent": summary.average_memory_util_percent,
        "max_memory_util_percent": summary.max_memory_util_percent,
        "average_temperature_c": summary.average_temperature_c,
        "max_temperature_c": summary.max_temperature_c,
        "temperature_rise_c": summary.temperature_rise_c,
        "temperature_slope_c_per_min": summary.temperature_slope_c_per_min,
        "thermal_stability_classification": summary.thermal_stability_classification,
        "average_sm_clock_mhz": summary.average_sm_clock_mhz,
        "average_memory_clock_mhz": summary.average_memory_clock_mhz,
        "average_cpu_util_percent": summary.average_cpu_util_percent,
        "average_client_process_cpu_percent": summary.average_client_process_cpu_percent,
        "average_backend_process_cpu_percent": summary.average_backend_process_cpu_percent,
        "observed_memory_mb": summary.max_memory_used_mb,
    }


def summarize_telemetry(
    samples: list[PowerSampleRecord],
    wall_time_s: float,
    total_tokens: int,
    provider: str | None = None,
    warnings: list[str] | None = None,
) -> TelemetrySummary:
    telemetry_warnings = list(warnings or [])
    notes = [GROSS_ENERGY_NOTE]
    error_messages = [sample.error for sample in samples if sample.error]
    telemetry_warnings.extend(str(error) for error in error_messages if error)
    watts = [_power_value(sample) for sample in samples if _power_value(sample) is not None]
    average_power = sum(watts) / len(watts) if watts else None
    min_power = min(watts) if watts else None
    max_power = max(watts) if watts else None
    peak_power = max_power
    power_stddev = statistics.pstdev(watts) if len(watts) > 1 else (0.0 if watts else None)
    coefficient_of_variation = (power_stddev / average_power) if power_stddev is not None and average_power not in {None, 0} else None
    energy = average_power * wall_time_s if average_power is not None else None
    joules_per_token = energy / total_tokens if energy is not None and total_tokens > 0 else None
    tokens_per_second_per_watt = total_tokens / wall_time_s / average_power if average_power not in {None, 0} and wall_time_s > 0 else None
    timestamps = [sample.timestamp_s for sample in samples]
    sampling_duration = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else None
    sampling_rate = (len(samples) - 1) / sampling_duration if sampling_duration and sampling_duration > 0 else None
    sample_intervals = [_optional_float(sample.sample_interval_s) for sample in samples]
    sample_intervals = [value for value in sample_intervals if value is not None]
    sample_interval = _average(sample_intervals)

    gpu_utils = [_first_float(sample.gpu_util_percent, sample.gpu_utilization_pct) for sample in samples]
    gpu_utils = [value for value in gpu_utils if value is not None]
    memory_utils = [_first_float(sample.memory_util_percent) for sample in samples]
    memory_utils = [value for value in memory_utils if value is not None]
    temperature_points = [
        (sample.timestamp_s, value)
        for sample in samples
        for value in [_first_float(sample.temperature_c, sample.gpu_temperature_c)]
        if value is not None
    ]
    temperature_points = sorted(temperature_points)
    temperatures = [value for _, value in temperature_points]
    sm_clocks = [_optional_float(sample.sm_clock_mhz) for sample in samples]
    sm_clocks = [value for value in sm_clocks if value is not None]
    memory_clocks = [_optional_float(sample.memory_clock_mhz) for sample in samples]
    memory_clocks = [value for value in memory_clocks if value is not None]
    graphics_clocks = [_optional_float(sample.graphics_clock_mhz) for sample in samples]
    graphics_clocks = [value for value in graphics_clocks if value is not None]
    memory_used = [_first_int(sample.memory_used_mb, sample.gpu_memory_used_mb) for sample in samples]
    memory_used = [value for value in memory_used if value is not None]
    memory_totals = [_first_int(sample.memory_total_mb) for sample in samples]
    memory_totals = [value for value in memory_totals if value is not None]
    cpu_utils = [_optional_float(sample.cpu_util_percent) for sample in samples]
    cpu_utils = [value for value in cpu_utils if value is not None]
    client_process_cpu = [_optional_float(sample.client_process_cpu_percent) for sample in samples]
    client_process_cpu = [value for value in client_process_cpu if value is not None]
    backend_process_cpu = [_optional_float(sample.backend_process_cpu_percent) for sample in samples]
    backend_process_cpu = [value for value in backend_process_cpu if value is not None]

    provider = provider or _first_text([sample.provider or sample.source for sample in samples])
    power_limit = _first_float(*[sample.power_limit_watts for sample in samples])
    enforced_power_limit = _first_float(*[sample.enforced_power_limit_watts for sample in samples])
    device_name = _first_text([sample.device_name for sample in samples])
    mig_mode = _first_text([sample.mig_mode for sample in samples])
    mig_profile = _first_text([sample.mig_profile for sample in samples])
    missing_fields = _missing_fields(samples)

    telemetry_available = bool(watts)
    if provider is None and samples:
        telemetry_warnings.append("Telemetry samples did not include a resolved provider.")
    if not samples and provider:
        telemetry_warnings.append("Telemetry provider was selected but no samples were collected.")
    if not watts and (samples or provider or telemetry_warnings):
        telemetry_warnings.append("Power telemetry was requested but no valid power samples were collected.")
    if samples and len(samples) < 5:
        telemetry_warnings.append("Telemetry sample count is low; power statistics may be noisy.")
    if sampling_rate is not None and sampling_rate < 1.0:
        telemetry_warnings.append("Telemetry sampling rate is below 1 Hz; short-lived power changes may be missed.")
    if power_stddev is not None and (
        power_stddev < FLAT_POWER_STDDEV_THRESHOLD_W
        or (coefficient_of_variation is not None and coefficient_of_variation < FLAT_POWER_COV_THRESHOLD)
    ):
        notes.append("Power readings are nearly flat. Efficiency differences may mainly reflect throughput differences.")
    if watts and not gpu_utils:
        telemetry_warnings.append("GPU utilization was unavailable, so power interpretation is limited.")
    if _mig_active(mig_mode, mig_profile, device_name):
        notes.append("MIG power readings may be device-level, slice-level, or limited depending on platform support.")
    temperature_duration = _temperature_duration_s(temperature_points)
    temperature_rise = _temperature_rise_c(temperature_points)
    temperature_slope = _temperature_slope_c_per_min(temperature_points)
    thermal_classification = _thermal_stability_classification(
        temperature_duration_s=temperature_duration,
        temperature_rise_c=temperature_rise,
        temperature_slope_c_per_min=temperature_slope,
    )
    if thermal_classification == "limited_window":
        notes.append("Thermal stability is based on a short active window. Use a longer soak duration for stronger evidence.")

    quality = _classify_quality(
        provider=provider,
        telemetry_available=telemetry_available,
        sample_count=len(samples),
        has_gpu_util=bool(gpu_utils),
        sampling_rate=sampling_rate,
        missing_fields=missing_fields,
    )
    unique_warnings = sorted(set(telemetry_warnings))
    unique_notes = _unique_preserve_order(notes)
    rounded_sampling_duration = _round_or_none(sampling_duration)
    rounded_sampling_rate = _round_or_none(sampling_rate)
    power_stats = {
        "avg": _round_or_none(average_power),
        "min": _round_or_none(min_power),
        "max": _round_or_none(max_power),
        "peak": _round_or_none(peak_power),
        "stddev": _round_or_none(power_stddev),
        "coefficient_of_variation": _round_or_none(coefficient_of_variation),
        "valid_sample_count": len(watts),
    }
    utilization_stats = {
        "avg_gpu_util_percent": _round_or_none(_average(gpu_utils)),
        "max_gpu_util_percent": _round_or_none(max(gpu_utils) if gpu_utils else None),
        "avg_memory_util_percent": _round_or_none(_average(memory_utils)),
        "max_memory_util_percent": _round_or_none(max(memory_utils) if memory_utils else None),
    }
    thermal_stats = {
        "avg_temperature_c": _round_or_none(_average(temperatures)),
        "max_temperature_c": _round_or_none(max(temperatures) if temperatures else None),
        "temperature_rise_c": _round_or_none(temperature_rise),
        "temperature_slope_c_per_min": _round_or_none(temperature_slope),
        "observation_duration_s": _round_or_none(temperature_duration),
        "stability_classification": thermal_classification,
    }
    clock_stats = {
        "avg_graphics_clock_mhz": _round_or_none(_average(graphics_clocks)),
        "avg_sm_clock_mhz": _round_or_none(_average(sm_clocks)),
        "avg_memory_clock_mhz": _round_or_none(_average(memory_clocks)),
    }
    cpu_stats = {
        "avg_cpu_util_percent": _round_or_none(_average(cpu_utils)),
        "avg_client_process_cpu_percent": _round_or_none(_average(client_process_cpu)),
        "avg_backend_process_cpu_percent": _round_or_none(_average(backend_process_cpu)),
    }
    provider_info = {
        "provider": provider,
        "device_name": device_name,
        "mig_mode": mig_mode,
        "mig_profile": mig_profile,
        "power_limit_watts": _round_or_none(power_limit),
        "enforced_power_limit_watts": _round_or_none(enforced_power_limit),
    }
    capabilities = detect_telemetry_capabilities(
        samples,
        provider=provider,
        warnings=unique_warnings,
        mig_mode=mig_mode,
        mig_profile=mig_profile,
        device_name=device_name,
    )
    return TelemetrySummary(
        telemetry_provider=provider,
        telemetry_available=telemetry_available,
        telemetry_quality=quality,
        telemetry_warnings=unique_warnings,
        telemetry_notes=unique_notes,
        sample_count=len(samples),
        duration_s=rounded_sampling_duration,
        sampling_rate_hz=rounded_sampling_rate,
        sample_interval_s=_round_or_none(sample_interval),
        missing_fields=missing_fields,
        warnings=unique_warnings,
        notes=unique_notes,
        provider_info=provider_info,
        telemetry_capabilities=capabilities,
        power_stats=power_stats,
        utilization_stats=utilization_stats,
        thermal_stats=thermal_stats,
        clock_stats=clock_stats,
        power_sample_count=len(samples),
        valid_power_sample_count=len(watts),
        power_sampling_duration_s=rounded_sampling_duration,
        power_sampling_rate_hz=rounded_sampling_rate,
        power_sample_interval_s=_round_or_none(sample_interval),
        average_power_watts=_round_or_none(average_power),
        min_power_watts=_round_or_none(min_power),
        max_power_watts=_round_or_none(max_power),
        peak_power_watts=_round_or_none(peak_power),
        power_stddev_watts=_round_or_none(power_stddev),
        energy_joules=_round_or_none(energy),
        joules_per_token=_round_or_none(joules_per_token),
        tokens_per_second_per_watt=_round_or_none(tokens_per_second_per_watt),
        average_gpu_util_percent=utilization_stats["avg_gpu_util_percent"],
        max_gpu_util_percent=utilization_stats["max_gpu_util_percent"],
        average_memory_util_percent=utilization_stats["avg_memory_util_percent"],
        max_memory_util_percent=utilization_stats["max_memory_util_percent"],
        average_temperature_c=thermal_stats["avg_temperature_c"],
        max_temperature_c=thermal_stats["max_temperature_c"],
        temperature_rise_c=thermal_stats["temperature_rise_c"],
        temperature_slope_c_per_min=thermal_stats["temperature_slope_c_per_min"],
        thermal_stability_classification=thermal_classification,
        average_sm_clock_mhz=clock_stats["avg_sm_clock_mhz"],
        average_memory_clock_mhz=clock_stats["avg_memory_clock_mhz"],
        average_cpu_util_percent=cpu_stats["avg_cpu_util_percent"],
        average_client_process_cpu_percent=cpu_stats["avg_client_process_cpu_percent"],
        average_backend_process_cpu_percent=cpu_stats["avg_backend_process_cpu_percent"],
        average_memory_used_mb=_round_or_none(_average(memory_used)),
        max_memory_used_mb=max(memory_used) if memory_used else None,
        memory_total_mb=max(memory_totals) if memory_totals else None,
        power_limit_watts=_round_or_none(power_limit),
        enforced_power_limit_watts=_round_or_none(enforced_power_limit),
        device_name=device_name,
        mig_mode=mig_mode,
        mig_profile=mig_profile,
    )


def _resolve_provider(telemetry: str) -> str | None:
    if telemetry in {"none", "nvml", "nvidia-smi"}:
        return None if telemetry == "none" else telemetry
    if telemetry == "auto":
        if _nvml_available():
            return "nvml"
        if _nvidia_smi_available():
            return "nvidia-smi"
        return "nvidia-smi"
    return None


def _nvml_available() -> bool:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


def _nvidia_smi_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=4,
        )
        return True
    except Exception:
        return False


def _sample_nvml(pynvml: object, handle: object, device_index: int) -> PowerSampleRecord:
    watts = _safe_float(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    memory = _safe_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
    power_limit = _safe_float(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
    utilization = _safe_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
    temperature = _safe_float(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    graphics_clock = _safe_int(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
    sm_clock = _safe_int(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    memory_clock = _safe_int(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    device_name = _safe_call(lambda: pynvml.nvmlDeviceGetName(handle))
    throttle = _safe_call(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
    mig_mode = _safe_call(lambda: pynvml.nvmlDeviceGetMigMode(handle))
    device_name_text = device_name.decode("utf-8") if isinstance(device_name, bytes) else (str(device_name) if device_name is not None else None)
    mig_mode_text = str(mig_mode[0]) if isinstance(mig_mode, tuple) and mig_mode else (str(mig_mode) if mig_mode is not None else None)
    memory_used_mb = int(memory.used / (1024 * 1024)) if memory else None
    memory_total_mb = int(memory.total / (1024 * 1024)) if memory else None
    return PowerSampleRecord(
        timestamp_s=time.time(),
        phase="measured",
        watts=watts,
        source="nvml",
        provider="nvml",
        power_watts=watts,
        device_index=device_index,
        gpu_memory_used_mb=memory_used_mb,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        memory_util_percent=(memory_used_mb / memory_total_mb * 100.0) if memory_used_mb is not None and memory_total_mb else None,
        gpu_utilization_pct=float(utilization.gpu) if utilization else None,
        gpu_util_percent=float(utilization.gpu) if utilization else None,
        gpu_temperature_c=temperature,
        temperature_c=temperature,
        graphics_clock_mhz=graphics_clock,
        sm_clock_mhz=sm_clock,
        memory_clock_mhz=memory_clock,
        power_limit_watts=power_limit,
        enforced_power_limit_watts=power_limit,
        throttle_reasons=hex(int(throttle)) if throttle is not None else None,
        mig_mode=mig_mode_text,
        device_name=device_name_text,
    )


def _sample_nvidia_smi(device_index: int) -> PowerSampleRecord:
    fields = NVIDIA_SMI_QUERY_FIELDS
    try:
        completed = _run_nvidia_smi_query(device_index, fields)
    except Exception:
        try:
            fields = NVIDIA_SMI_MINIMAL_FIELDS
            completed = _run_nvidia_smi_query(device_index, fields)
        except Exception as fallback_exc:
            return _error_sample("nvidia-smi", device_index, f"{fallback_exc.__class__.__name__}: {fallback_exc}")
    line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    return parse_nvidia_smi_sample(fields, line, device_index=device_index)


NVIDIA_SMI_QUERY_FIELDS = [
    "index",
    "name",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "clocks.gr",
    "clocks.sm",
    "clocks.mem",
]
NVIDIA_SMI_MINIMAL_FIELDS = [
    "index",
    "name",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
]


def _run_nvidia_smi_query(device_index: int, fields: list[str]) -> subprocess.CompletedProcess[str]:
    query = ",".join(fields)
    return subprocess.run(
        ["nvidia-smi", "-i", str(device_index), f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=4,
    )


def parse_nvidia_smi_sample(fields: list[str], line: str, device_index: int = 0) -> PowerSampleRecord:
    values = next(csv.reader(io.StringIO(line)), [])
    if not values:
        return _error_sample("nvidia-smi", device_index, "nvidia-smi returned no sample data.")
    if len(values) < len(fields):
        values = values + [""] * (len(fields) - len(values))
    row = dict(zip(fields, values[: len(fields)], strict=True))
    watts = _parse_float(row.get("power.draw", ""))
    power_limit = _parse_float(row.get("power.limit", ""))
    memory_used = _parse_int(row.get("memory.used", ""))
    memory_total = _parse_int(row.get("memory.total", ""))
    if watts is None and power_limit is None and memory_used is None:
        return _error_sample("nvidia-smi", device_index, f"Failed to parse nvidia-smi output: {line!r}")
    parsed_device_index = _parse_int(row.get("index", "")) or device_index
    return PowerSampleRecord(
        timestamp_s=time.time(),
        phase="measured",
        watts=watts,
        source="nvidia-smi",
        provider="nvidia-smi",
        power_watts=watts,
        device_index=parsed_device_index,
        gpu_memory_used_mb=memory_used,
        memory_used_mb=memory_used,
        memory_total_mb=memory_total,
        memory_util_percent=(memory_used / memory_total * 100.0) if memory_used is not None and memory_total else None,
        gpu_utilization_pct=_parse_float(row.get("utilization.gpu", "")),
        gpu_util_percent=_parse_float(row.get("utilization.gpu", "")),
        gpu_temperature_c=_parse_float(row.get("temperature.gpu", "")),
        temperature_c=_parse_float(row.get("temperature.gpu", "")),
        graphics_clock_mhz=_parse_int(row.get("clocks.gr", "")),
        sm_clock_mhz=_parse_int(row.get("clocks.sm", "")),
        memory_clock_mhz=_parse_int(row.get("clocks.mem", "")),
        power_limit_watts=power_limit,
        device_name=(row.get("name") or "").strip() or None,
    )


def _error_sample(source: str, device_index: int, error: str) -> PowerSampleRecord:
    return PowerSampleRecord(
        timestamp_s=time.time(),
        phase="measured",
        watts=None,
        source=source,
        provider=source,
        device_index=device_index,
        error=error,
    )


def _safe_call(fn: Callable[[], object]) -> object | None:
    try:
        return fn()
    except Exception:
        return None


def _safe_float(fn: Callable[[], float]) -> float | None:
    try:
        return float(fn())
    except Exception:
        return None


def _safe_int(fn: Callable[[], int]) -> int | None:
    try:
        return int(fn())
    except Exception:
        return None


def _parse_float(value: str) -> float | None:
    text = value.strip()
    if text in {"", "N/A", "[N/A]", "[Not Supported]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    text = value.strip()
    if text in {"", "N/A", "[N/A]", "[Not Supported]"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _read_proc_stat_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            line = handle.readline()
    except OSError:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


GROSS_ENERGY_NOTE = "Energy uses gross active power unless an idle baseline is supplied or sampled."


def _power_value(sample: PowerSampleRecord) -> float | None:
    return _first_float(sample.power_watts, sample.watts)


def _average(values: list[float] | list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _first_float(*values: object) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_text(values: list[object]) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _temperature_duration_s(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    duration = points[-1][0] - points[0][0]
    return duration if duration > 0 else None


def _temperature_rise_c(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    return points[-1][1] - points[0][1]


def _temperature_slope_c_per_min(points: list[tuple[float, float]]) -> float | None:
    duration = _temperature_duration_s(points)
    rise = _temperature_rise_c(points)
    if duration is None or rise is None:
        return None
    return rise / duration * 60.0


def _thermal_stability_classification(
    *,
    temperature_duration_s: float | None,
    temperature_rise_c: float | None,
    temperature_slope_c_per_min: float | None,
) -> str:
    if temperature_duration_s is None or temperature_rise_c is None or temperature_slope_c_per_min is None:
        return "unavailable"
    if temperature_duration_s < THERMAL_SOAK_MIN_DURATION_S:
        return "limited_window"
    if abs(temperature_rise_c) <= THERMAL_STABLE_MAX_RISE_C and abs(temperature_slope_c_per_min) <= THERMAL_STABLE_MAX_SLOPE_C_PER_MIN:
        return "stable"
    return "warming"


def _missing_fields(samples: list[PowerSampleRecord]) -> list[str]:
    missing: list[str] = []
    for field_name, aliases in TELEMETRY_FIELD_ALIASES.items():
        if not any(_sample_has_value(sample, aliases) for sample in samples):
            missing.append(field_name)
    return missing


def _sample_has_value(sample: PowerSampleRecord, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        value = getattr(sample, alias, None)
        if value is not None and str(value).strip():
            return True
    return False


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _mig_active(mig_mode: str | None, mig_profile: str | None, device_name: str | None) -> bool:
    text = " ".join(value for value in (mig_mode, mig_profile, device_name) if value)
    return "MIG" in text.upper() or mig_mode in {"1", "Enabled", "enabled", "True", "true"}


def _classify_quality(
    *,
    provider: str | None,
    telemetry_available: bool,
    sample_count: int,
    has_gpu_util: bool,
    sampling_rate: float | None,
    missing_fields: list[str],
) -> str:
    if provider is None or sample_count == 0:
        return "unavailable"
    if not telemetry_available or sample_count < 5:
        return "poor"
    major_missing = [field for field in missing_fields if field in MAJOR_TELEMETRY_FIELDS]
    if not has_gpu_util or (sampling_rate is not None and sampling_rate < 1.0) or major_missing:
        return "limited"
    return "good"
