"""Presentation helpers for Attach Mode recommendation output."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .schemas import CheckRecord, RecommendationResult, TelemetrySummary, to_dict

SECTION_WIDTH = 60


def format_recommendation_report(
    result: RecommendationResult,
    metadata: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> str:
    reporter = PlainTextReporter()
    return reporter.render(result=result, metadata=metadata, artifacts=artifacts)


def format_metric_table(rows: Iterable[tuple[str, str, str, str]], headers: tuple[str, str, str, str]) -> str:
    row_list = list(rows)
    columns = list(zip(*([headers] + row_list), strict=True))
    widths = [max(len(str(value)) for value in column) for column in columns]
    output = [
        "  ".join(str(value).ljust(width) for value, width in zip(headers, widths, strict=True)),
    ]
    for row in row_list:
        output.append("  ".join(str(value).ljust(width) for value, width in zip(row, widths, strict=True)))
    return "\n".join(output)


def format_command_block(command: str | None) -> str:
    if not command:
        return "  unavailable"
    if " \\" in command:
        return "\n".join(f"  {line}" for line in command.splitlines())
    return f"  {command}"


def format_checks(checks: list[CheckRecord]) -> str:
    if not checks:
        return "  [SKIP] No checks recorded"
    lines = []
    for check in checks:
        label = check.status.upper()
        lines.append(f"  [{label}] {check.message}")
    return "\n".join(lines)


@dataclass
class PlainTextReporter:
    def render(
        self,
        *,
        result: RecommendationResult,
        metadata: dict[str, object] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        metadata = metadata or {}
        artifact_map = artifacts or result.artifacts
        sections = [
            _banner("Serve Optimize Recommendation"),
            self._summary_section(result, artifact_map),
            self._recommended_configuration(result),
            self._reasons_section(result),
            self._weights_section(result),
            self._candidates_section(result),
            self._pareto_section(result),
            self._prediction_section(result),
            self._power_section(result),
            self._resource_telemetry_section(result),
            self._telemetry_capabilities_section(result),
            self._objective_alternatives_section(result),
            self._alternatives_section(result),
            self._checks_section(result),
            self._metadata_section(result),
            self._warnings_section(result),
            self._limitations_section(result),
            self._artifacts_section(artifact_map),
        ]
        return "\n\n".join(section for section in sections if section).rstrip() + "\n"

    def _summary_section(self, result: RecommendationResult, artifacts: dict[str, str]) -> str:
        status = result.status.upper()
        telemetry_line = _telemetry_path(result)
        lines = [
            f"Status: {status}",
            f"Mode: {result.mode.title()}",
            f"Goal: {result.goal}",
            f"Endpoint: {result.endpoint or 'unknown'}",
            f"Model: {result.model or 'unknown'}",
            f"Backend: {result.backend or 'unknown'}",
            f"Telemetry: {telemetry_line}",
            f"Candidates: {result.valid_candidate_count}/{result.candidate_count} valid",
            f"Recommendation Confidence: {(result.confidence_level or 'unknown').upper()}",
            f"Recommendation type: {'comparative search' if result.was_comparative else 'single-candidate validation'}",
            f"Artifacts: {artifacts.get('run_dir', 'unknown')}",
        ]
        return "\n".join(lines)

    def _recommended_configuration(self, result: RecommendationResult) -> str:
        lines = [_section("Recommended Configuration")]
        if result.recommended_candidate_id is None:
            lines.append("No recommendation could be selected.")
            return "\n".join(lines)
        selected = result.selected_config
        benchmark = result.selected_benchmark_plan
        lines.extend(
            [
                f"Candidate: {result.recommended_candidate_id}",
                f"Source: {_source_label(result.candidate_source)}",
                "Serve command:",
                format_command_block(result.selected_serve_command),
                "",
                "Benchmark plan:",
                f"  concurrency: {benchmark.concurrency if benchmark else 'unknown'}",
                f"  num_requests: {benchmark.num_requests if benchmark else 'unknown'}",
                f"  max_tokens: {benchmark.max_tokens if benchmark else 'unknown'}",
                f"  expected_input_tokens: {benchmark.expected_input_tokens if benchmark else 'unknown'}",
                f"  expected_output_tokens: {benchmark.expected_output_tokens if benchmark else 'unknown'}",
            ]
        )
        if selected is not None and selected.parallel:
            lines.append(f"  topology: {selected.parallel}")
        return "\n".join(lines)

    def _reasons_section(self, result: RecommendationResult) -> str:
        lines = [_section("Selection Rationale")]
        if result.recommended_candidate_id is None:
            lines.append("Serve Optimize could not select a usable candidate.")
            return "\n".join(lines)
        lines.append("Serve Optimize selected this candidate because:")
        for index, reason in enumerate(result.selection_reasons, start=1):
            lines.append(f"  {index}. {reason}")
        return "\n".join(lines)

    def _candidates_section(self, result: RecommendationResult) -> str:
        lines = [_section("Candidates Evaluated")]
        if not result.candidate_table:
            lines.append("No candidate table was recorded.")
            return "\n".join(lines)
        rows = []
        for row in result.candidate_table:
            rows.append(
                (
                    str(row.get("candidate_id") or "unknown"),
                    str(row.get("source") or "unknown"),
                    str(row.get("concurrency") or "n/a"),
                    _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                    _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                    _fmt_metric(row.get("average_power_watts"), "W"),
                    _fmt_metric(row.get("joules_per_token"), "J/tok"),
                    _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                    str(row.get("failed_requests") if row.get("failed_requests") is not None else "n/a"),
                    _fmt_metric(row.get("throughput_score"), None),
                    _fmt_metric(row.get("latency_score"), None),
                    _fmt_metric(row.get("power_score"), None),
                    _fmt_metric(row.get("reliability_score"), None),
                    _fmt_metric(row.get("score"), None),
                    "yes" if row.get("pareto_optimal") else "no",
                )
            )
        lines.append(
            format_metric_table(
                rows=rows,
                headers=(
                    "candidate_id",
                    "source",
                    "conc",
                    "tokens/s",
                    "p95",
                    "watts",
                    "J/tok",
                    "tok/s/W",
                    "fail",
                    "thr",
                    "lat",
                    "power",
                    "rel",
                    "score",
                    "pareto",
                ),
            )
        )
        return "\n".join(lines)

    def _weights_section(self, result: RecommendationResult) -> str:
        if not result.score_weights:
            return ""
        lines = [_section("Scoring Policy")]
        lines.append("Weights used for the selected goal:")
        for key, value in result.score_weights.items():
            lines.append(f"  {key}: {_fmt_metric(value, None)}")
        if result.telemetry_used_in_scoring:
            lines.append("Power telemetry was used in scoring.")
        elif result.power_missing_reason:
            lines.append(f"Power telemetry was not used: {result.power_missing_reason}")
        return "\n".join(lines)

    def _pareto_section(self, result: RecommendationResult) -> str:
        lines = [_section("Pareto Frontier")]
        if not result.pareto_frontier:
            lines.append("No Pareto frontier was recorded.")
            return "\n".join(lines)
        rows = [
            (
                str(row.get("candidate_id") or "unknown"),
                str(row.get("concurrency") or "n/a"),
                _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                _fmt_metric(row.get("joules_per_token"), "J/tok"),
                _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                _fmt_metric(row.get("score"), None),
            )
            for row in result.pareto_frontier
        ]
        lines.append(
            format_metric_table(
                rows=rows,
                headers=("candidate_id", "conc", "tokens/s", "p95", "J/tok", "tok/s/W", "score"),
            )
        )
        return "\n".join(lines)

    def _prediction_section(self, result: RecommendationResult) -> str:
        lines = [_section("Prediction vs Measurement")]
        rows = [
            (
                "tokens/sec",
                _fmt_metric(result.predicted_metrics.get("predicted_tokens_s"), "tokens/s"),
                _fmt_metric(result.measured_metrics.get("total_tokens_s"), "tokens/s"),
                _fmt_ratio(result.comparison_metrics.get("measured_over_predicted_tokens_ratio")),
            ),
            (
                "request rate",
                _fmt_metric(result.predicted_metrics.get("predicted_request_rate"), "req/s"),
                _fmt_metric(result.measured_metrics.get("request_rate_req_s"), "req/s"),
                _fmt_ratio(result.comparison_metrics.get("measured_over_predicted_request_rate_ratio")),
            ),
            (
                "request latency",
                _fmt_metric(result.predicted_metrics.get("predicted_request_latency_ms"), "ms"),
                _fmt_metric(_ms_from_seconds(result.measured_metrics.get("avg_latency_s")), "ms"),
                _fmt_percent_delta(result.comparison_metrics.get("latency_delta_percent")),
            ),
            (
                "p95 latency",
                "n/a",
                _fmt_metric(_ms_from_seconds(result.measured_metrics.get("p95_latency_s")), "ms"),
                "n/a",
            ),
        ]
        lines.append(
            format_metric_table(
                rows=rows,
                headers=("Metric", "Predicted", "Measured", "Ratio/Delta"),
            )
        )
        return "\n".join(lines)

    def _power_section(self, result: RecommendationResult) -> str:
        lines = [_section("Power and Efficiency")]
        provider = result.telemetry_provider
        if provider is None:
            lines.append("Power telemetry: unavailable")
            if result.goal == "efficiency":
                lines.append("Efficiency goal: unavailable unless --allow-efficiency-fallback was used")
            return "\n".join(lines)
        rows = [
            ("average power", _fmt_metric(result.telemetry_metrics.get("average_power_watts"), "W")),
            ("peak power", _fmt_metric(result.telemetry_metrics.get("peak_power_watts"), "W")),
            ("energy", _fmt_metric(result.telemetry_metrics.get("energy_joules"), "J")),
            ("joules/token", _fmt_metric(result.telemetry_metrics.get("joules_per_token"), "J/token")),
            ("tokens/sec/watt", _fmt_metric(result.telemetry_metrics.get("tokens_per_second_per_watt"), "tokens/s/W")),
        ]
        for label, value in rows:
            lines.append(f"{label:<24} {value}")
        return "\n".join(lines)

    def _resource_telemetry_section(self, result: RecommendationResult) -> str:
        lines = [_section("Resource Telemetry")]
        provider = result.telemetry_provider or result.telemetry_metrics.get("telemetry_provider")
        if provider is None:
            lines.append("Resource telemetry: unavailable")
            return "\n".join(lines)
        rows = [
            ("provider", str(provider), "", ""),
            ("quality", str(result.telemetry_metrics.get("telemetry_quality") or "unknown"), "", ""),
            ("quality meaning", _telemetry_quality_explanation(result.telemetry_metrics.get("telemetry_quality")), "", ""),
            ("samples", _fmt_metric(result.telemetry_metrics.get("power_sample_count"), None), "", ""),
            ("sampling rate", _fmt_metric(result.telemetry_metrics.get("power_sampling_rate_hz"), "Hz"), "", ""),
            ("average power", _fmt_metric(result.telemetry_metrics.get("average_power_watts"), "W"), "", ""),
            ("power stddev", _fmt_metric(result.telemetry_metrics.get("power_stddev_watts"), "W"), "", ""),
            ("average GPU util", _fmt_metric(result.telemetry_metrics.get("average_gpu_util_percent"), "%"), "", ""),
            ("max GPU util", _fmt_metric(result.telemetry_metrics.get("max_gpu_util_percent"), "%"), "", ""),
            ("average memory util", _fmt_metric(result.telemetry_metrics.get("average_memory_util_percent"), "%"), "", ""),
            ("average temperature", _fmt_metric(result.telemetry_metrics.get("average_temperature_c"), "C"), "", ""),
            ("power cap", _fmt_metric(result.telemetry_metrics.get("power_limit_watts"), "W"), "", ""),
            ("missing fields", _join_values(result.telemetry_metrics.get("missing_fields")), "", ""),
        ]
        lines.append(format_metric_table(rows=rows, headers=("Metric", "Value", "", "")))
        if result.confidence_level:
            lines.append(f"Recommendation Confidence: {result.confidence_level.upper()}")
        for reason in result.confidence_reasons:
            lines.append(f"- {reason}")
        for warning in _list_values(result.telemetry_metrics.get("telemetry_warnings")):
            lines.append(f"- telemetry warning: {warning}")
        for note in _list_values(result.telemetry_metrics.get("telemetry_notes")):
            lines.append(f"- telemetry note: {note}")
        return "\n".join(lines)

    def _telemetry_capabilities_section(self, result: RecommendationResult) -> str:
        capabilities = _capability_payload(result.telemetry_metrics.get("telemetry_capabilities"))
        if not capabilities:
            return ""
        lines = [_section("Telemetry Capabilities")]
        lines.append("Available:")
        for field_name in capabilities.get("available_fields") or ["none"]:
            lines.append(f"  OK {field_name}")
        lines.append("")
        lines.append("Unavailable:")
        for field_name in capabilities.get("unavailable_fields") or ["none"]:
            lines.append(f"  missing {field_name}")
        notes = capabilities.get("notes") or []
        if notes:
            lines.append("")
            lines.append("Notes:")
            for note in notes:
                lines.append(f"  {note}")
        return "\n".join(lines)

    def _alternatives_section(self, result: RecommendationResult) -> str:
        if not result.alternatives:
            return ""
        lines = [_section("Ranked Alternatives")]
        lines.append("Candidate                  Final score   Status")
        for score in result.alternatives[:5]:
            lines.append(
                f"{score.candidate_id:<25} {_fmt_metric(score.final_score, None):<13} "
                f"{','.join(score.disqualifiers) if score.disqualifiers else 'eligible'}"
            )
        return "\n".join(lines)

    def _objective_alternatives_section(self, result: RecommendationResult) -> str:
        if not result.alternative_recommendations:
            return ""
        lines = [_section("Alternative Recommendations")]
        rows = []
        for objective, row in result.alternative_recommendations.items():
            rows.append(
                (
                    objective,
                    str(row.get("candidate_id") or "unknown"),
                    str(row.get("concurrency") or "n/a"),
                    _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                    _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                    _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                    str(row.get("reason") or ""),
                )
            )
        lines.append(
            format_metric_table(
                rows=rows,
                headers=("objective", "candidate_id", "conc", "tokens/s", "p95", "tok/s/W", "reason"),
            )
        )
        return "\n".join(lines)

    def _checks_section(self, result: RecommendationResult) -> str:
        lines = [_section("Checks Performed"), format_checks(result.checks)]
        return "\n".join(lines)

    def _warnings_section(self, result: RecommendationResult) -> str:
        if not result.warnings:
            return ""
        lines = [_section("Warnings")]
        for warning in result.warnings:
            lines.append(f"- {warning}")
        return "\n".join(lines)

    def _metadata_section(self, result: RecommendationResult) -> str:
        if not result.metadata_notes:
            return ""
        lines = [_section("Metadata Notes")]
        for note in result.metadata_notes:
            lines.append(f"- {note}")
        return "\n".join(lines)

    def _limitations_section(self, result: RecommendationResult) -> str:
        lines = [_section("Limitations")]
        for limitation in result.limitations:
            lines.append(f"- {limitation}")
        return "\n".join(lines)

    def _artifacts_section(self, artifacts: dict[str, str]) -> str:
        lines = [_section("Artifacts")]
        lines.extend(_artifact_lines(artifacts))
        return "\n".join(lines)


@dataclass
class RichReporter:
    console: Console

    def render(
        self,
        *,
        result: RecommendationResult,
        metadata: dict[str, object] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        del metadata
        artifact_map = artifacts or result.artifacts
        self.console.print(Panel(self._summary_table(result, artifact_map), title="Serve Optimize Recommendation", expand=True))
        self.console.print(Panel(self._config_text(result), title="Recommended Configuration", expand=True))
        self.console.print(Panel(self._reason_text(result), title="Selection Rationale", expand=True))
        if result.score_weights:
            self.console.print(Panel(self._weights_table(result), title="Scoring Policy", expand=True))
        self.console.print(Panel(self._candidates_table(result), title="Candidates Evaluated", expand=True))
        self.console.print(Panel(self._pareto_table(result), title="Pareto Frontier", expand=True))
        self.console.print(Panel(self._prediction_table(result), title="Prediction vs Measurement", expand=True))
        self.console.print(Panel(self._power_table(result), title="Power and Efficiency", expand=True))
        self.console.print(Panel(self._resource_telemetry_table(result), title="Resource Telemetry", expand=True))
        capabilities = self._telemetry_capabilities_table(result)
        if capabilities is not None:
            self.console.print(Panel(capabilities, title="Telemetry Capabilities", expand=True))
        if result.alternative_recommendations:
            self.console.print(Panel(self._objective_alternatives_table(result), title="Alternative Recommendations", expand=True))
        self.console.print(Panel(self._checks_table(result.checks), title="Checks Performed", expand=True))
        if result.metadata_notes:
            metadata_text = Text("\n".join(f"- {note}" for note in result.metadata_notes))
            self.console.print(Panel(metadata_text, title="Metadata Notes", expand=True))
        if result.warnings:
            warning_text = Text("\n".join(f"- {warning}" for warning in result.warnings))
            self.console.print(Panel(warning_text, title="Warnings", expand=True))
        self.console.print(Panel(Text("\n".join(f"- {item}" for item in result.limitations)), title="Limitations", expand=True))
        self.console.print(Panel(Text("\n".join(_artifact_lines(artifact_map))), title="Artifacts", expand=True))

    def _summary_table(self, result: RecommendationResult, artifacts: dict[str, str]) -> Table:
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        rows = [
            ("Status", result.status.upper()),
            ("Mode", result.mode.title()),
            ("Goal", result.goal),
            ("Endpoint", result.endpoint or "unknown"),
            ("Model", result.model or "unknown"),
            ("Backend", result.backend or "unknown"),
            ("Telemetry", _telemetry_path(result)),
            ("Candidates", f"{result.valid_candidate_count}/{result.candidate_count} valid"),
            ("Recommendation Confidence", (result.confidence_level or "unknown").upper()),
            ("Recommendation type", "comparative search" if result.was_comparative else "single-candidate validation"),
            ("Artifacts", artifacts.get("run_dir", "unknown")),
        ]
        for label, value in rows:
            table.add_row(label, str(value))
        return table

    def _config_text(self, result: RecommendationResult) -> Text:
        lines = []
        if result.recommended_candidate_id is None:
            return Text("No recommendation could be selected.")
        lines.append(f"Candidate: {result.recommended_candidate_id}")
        lines.append(f"Source: {_source_label(result.candidate_source)}")
        lines.append("Serve command:")
        lines.append(format_command_block(result.selected_serve_command))
        benchmark = result.selected_benchmark_plan
        if benchmark is not None:
            lines.extend(
                [
                    "",
                    "Benchmark plan:",
                    f"  concurrency: {benchmark.concurrency}",
                    f"  num_requests: {benchmark.num_requests}",
                    f"  max_tokens: {benchmark.max_tokens}",
                    f"  expected_input_tokens: {benchmark.expected_input_tokens}",
                ]
            )
        return Text("\n".join(lines))

    def _reason_text(self, result: RecommendationResult) -> Text:
        lines = []
        if result.recommended_candidate_id is None:
            lines.append("No usable candidate could be recommended.")
        else:
            lines.append("Serve Optimize selected this candidate because:")
            for index, reason in enumerate(result.selection_reasons, start=1):
                lines.append(f"  {index}. {reason}")
        return Text("\n".join(lines))

    def _candidates_table(self, result: RecommendationResult) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        for column in ("candidate_id", "source", "conc", "tokens/s", "p95", "watts", "J/tok", "tok/s/W", "fail", "thr", "lat", "power", "rel", "score", "pareto"):
            justify = "right" if column not in {"candidate_id", "source"} else "left"
            table.add_column(column, justify=justify)
        for row in result.candidate_table:
            table.add_row(
                str(row.get("candidate_id") or "unknown"),
                str(row.get("source") or "unknown"),
                str(row.get("concurrency") or "n/a"),
                _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                _fmt_metric(row.get("average_power_watts"), "W"),
                _fmt_metric(row.get("joules_per_token"), "J/tok"),
                _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                str(row.get("failed_requests") if row.get("failed_requests") is not None else "n/a"),
                _fmt_metric(row.get("throughput_score"), None),
                _fmt_metric(row.get("latency_score"), None),
                _fmt_metric(row.get("power_score"), None),
                _fmt_metric(row.get("reliability_score"), None),
                _fmt_metric(row.get("score"), None),
                "yes" if row.get("pareto_optimal") else "no",
            )
        return table

    def _weights_table(self, result: RecommendationResult) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component")
        table.add_column("Weight", justify="right")
        for key, value in result.score_weights.items():
            table.add_row(key, _fmt_metric(value, None))
        if result.telemetry_used_in_scoring:
            table.add_row("power status", "used")
        elif result.power_missing_reason:
            table.add_row("power status", result.power_missing_reason)
        return table

    def _pareto_table(self, result: RecommendationResult) -> Table | Text:
        if not result.pareto_frontier:
            return Text("No Pareto frontier was recorded.")
        table = Table(show_header=True, header_style="bold magenta")
        for column in ("candidate_id", "conc", "tokens/s", "p95", "J/tok", "tok/s/W", "score"):
            justify = "right" if column != "candidate_id" else "left"
            table.add_column(column, justify=justify)
        for row in result.pareto_frontier:
            table.add_row(
                str(row.get("candidate_id") or "unknown"),
                str(row.get("concurrency") or "n/a"),
                _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                _fmt_metric(row.get("joules_per_token"), "J/tok"),
                _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                _fmt_metric(row.get("score"), None),
            )
        return table

    def _prediction_table(self, result: RecommendationResult) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Predicted", justify="right")
        table.add_column("Measured", justify="right")
        table.add_column("Ratio/Delta", justify="right")
        rows = [
            (
                "tokens/sec",
                _fmt_metric(result.predicted_metrics.get("predicted_tokens_s"), "tokens/s"),
                _fmt_metric(result.measured_metrics.get("total_tokens_s"), "tokens/s"),
                _fmt_ratio(result.comparison_metrics.get("measured_over_predicted_tokens_ratio")),
            ),
            (
                "request rate",
                _fmt_metric(result.predicted_metrics.get("predicted_request_rate"), "req/s"),
                _fmt_metric(result.measured_metrics.get("request_rate_req_s"), "req/s"),
                _fmt_ratio(result.comparison_metrics.get("measured_over_predicted_request_rate_ratio")),
            ),
            (
                "request latency",
                _fmt_metric(result.predicted_metrics.get("predicted_request_latency_ms"), "ms"),
                _fmt_metric(_ms_from_seconds(result.measured_metrics.get("avg_latency_s")), "ms"),
                _fmt_percent_delta(result.comparison_metrics.get("latency_delta_percent")),
            ),
            (
                "p95 latency",
                "n/a",
                _fmt_metric(_ms_from_seconds(result.measured_metrics.get("p95_latency_s")), "ms"),
                "n/a",
            ),
        ]
        for row in rows:
            table.add_row(*row)
        return table

    def _power_table(self, result: RecommendationResult) -> Table | Text:
        provider = result.telemetry_provider
        if provider is None:
            lines = ["Power telemetry: unavailable"]
            if result.goal == "efficiency":
                lines.append("Efficiency goal: unavailable unless --allow-efficiency-fallback was used")
            return Text("\n".join(lines))
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        rows = [
            ("average power", _fmt_metric(result.telemetry_metrics.get("average_power_watts"), "W")),
            ("peak power", _fmt_metric(result.telemetry_metrics.get("peak_power_watts"), "W")),
            ("energy", _fmt_metric(result.telemetry_metrics.get("energy_joules"), "J")),
            ("joules/token", _fmt_metric(result.telemetry_metrics.get("joules_per_token"), "J/token")),
            ("tokens/sec/watt", _fmt_metric(result.telemetry_metrics.get("tokens_per_second_per_watt"), "tokens/s/W")),
        ]
        for row in rows:
            table.add_row(*row)
        return table

    def _resource_telemetry_table(self, result: RecommendationResult) -> Table | Text:
        provider = result.telemetry_provider or result.telemetry_metrics.get("telemetry_provider")
        if provider is None:
            return Text("Resource telemetry: unavailable")
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        rows = [
            ("provider", str(provider)),
            ("quality", str(result.telemetry_metrics.get("telemetry_quality") or "unknown")),
            ("quality meaning", _telemetry_quality_explanation(result.telemetry_metrics.get("telemetry_quality"))),
            ("samples", _fmt_metric(result.telemetry_metrics.get("power_sample_count"), None)),
            ("sampling rate", _fmt_metric(result.telemetry_metrics.get("power_sampling_rate_hz"), "Hz")),
            ("average power", _fmt_metric(result.telemetry_metrics.get("average_power_watts"), "W")),
            ("power stddev", _fmt_metric(result.telemetry_metrics.get("power_stddev_watts"), "W")),
            ("average GPU util", _fmt_metric(result.telemetry_metrics.get("average_gpu_util_percent"), "%")),
            ("max GPU util", _fmt_metric(result.telemetry_metrics.get("max_gpu_util_percent"), "%")),
            ("average memory util", _fmt_metric(result.telemetry_metrics.get("average_memory_util_percent"), "%")),
            ("average temperature", _fmt_metric(result.telemetry_metrics.get("average_temperature_c"), "C")),
            ("power cap", _fmt_metric(result.telemetry_metrics.get("power_limit_watts"), "W")),
            ("missing fields", _join_values(result.telemetry_metrics.get("missing_fields"))),
            ("confidence", (result.confidence_level or "unknown").upper()),
        ]
        for row in rows:
            table.add_row(*row)
        for reason in result.confidence_reasons:
            table.add_row("confidence reason", reason)
        for warning in _list_values(result.telemetry_metrics.get("telemetry_warnings")):
            table.add_row("telemetry warning", warning)
        for note in _list_values(result.telemetry_metrics.get("telemetry_notes")):
            table.add_row("telemetry note", note)
        return table

    def _telemetry_capabilities_table(self, result: RecommendationResult) -> Table | None:
        capabilities = _capability_payload(result.telemetry_metrics.get("telemetry_capabilities"))
        if not capabilities:
            return None
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Capability")
        table.add_column("Status")
        available = capabilities.get("available_fields") or []
        unavailable = capabilities.get("unavailable_fields") or []
        for field_name in available:
            table.add_row(str(field_name), "[green]available[/green]")
        for field_name in unavailable:
            table.add_row(str(field_name), "[yellow]unavailable[/yellow]")
        for note in capabilities.get("notes") or []:
            table.add_row("note", str(note))
        return table

    def _checks_table(self, checks: list[CheckRecord]) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status")
        table.add_column("Check")
        table.add_column("Message")
        for check in checks:
            status_style = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}.get(check.status, "white")
            table.add_row(f"[{status_style}]{check.status.upper()}[/{status_style}]", check.name, check.message)
        return table

    def _objective_alternatives_table(self, result: RecommendationResult) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        for column in ("objective", "candidate_id", "conc", "tokens/s", "p95", "tok/s/W", "reason"):
            justify = "right" if column in {"conc", "tokens/s", "p95", "tok/s/W"} else "left"
            table.add_column(column, justify=justify)
        for objective, row in result.alternative_recommendations.items():
            table.add_row(
                objective,
                str(row.get("candidate_id") or "unknown"),
                str(row.get("concurrency") or "n/a"),
                _fmt_metric(row.get("total_tokens_s"), "tok/s"),
                _fmt_metric(_ms_from_seconds(row.get("p95_latency_s")), "ms"),
                _fmt_metric(row.get("tokens_per_second_per_watt"), "tok/s/W"),
                str(row.get("reason") or ""),
            )
        return table


@dataclass
class RichTelemetryCheckReporter:
    console: Console

    def render(self, *, summary: TelemetrySummary, artifacts: dict[str, str]) -> None:
        self.console.print(Panel(self._summary_table(summary, artifacts), title="Serve Optimize Telemetry Check", expand=True))
        self.console.print(Panel(self._power_table(summary), title="Power", expand=True))
        self.console.print(Panel(self._resource_table(summary), title="Resource Fields", expand=True))
        self.console.print(Panel(self._field_table(summary), title="Field Availability", expand=True))
        self.console.print(Panel(self._capabilities_table(summary), title="Telemetry Capabilities", expand=True))
        self.console.print(Panel(_bullet_text(summary.warnings), title="Warnings", expand=True))
        self.console.print(Panel(_bullet_text(summary.notes), title="Notes", expand=True))

    def _summary_table(self, summary: TelemetrySummary, artifacts: dict[str, str]) -> Table:
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        rows = [
            ("Provider", summary.telemetry_provider or "unavailable"),
            ("Device", summary.device_name or "unknown"),
            ("Quality", summary.telemetry_quality),
            ("Samples", str(summary.sample_count)),
            ("Duration", _fmt_metric(summary.duration_s, "s")),
            ("Sampling rate", _fmt_metric(summary.sampling_rate_hz, "Hz")),
            ("Artifacts", artifacts.get("run_dir", "unknown")),
        ]
        for row in rows:
            table.add_row(*row)
        return table

    def _power_table(self, summary: TelemetrySummary) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        rows = [
            ("average power", _fmt_metric(summary.power_stats.get("avg"), "W")),
            ("min power", _fmt_metric(summary.power_stats.get("min"), "W")),
            ("max power", _fmt_metric(summary.power_stats.get("max"), "W")),
            ("stddev", _fmt_metric(summary.power_stats.get("stddev"), "W")),
            ("coefficient of variation", _fmt_metric(summary.power_stats.get("coefficient_of_variation"), None)),
        ]
        for row in rows:
            table.add_row(*row)
        return table

    def _resource_table(self, summary: TelemetrySummary) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        rows = [
            ("average GPU util", _fmt_metric(summary.utilization_stats.get("avg_gpu_util_percent"), "%")),
            ("max GPU util", _fmt_metric(summary.utilization_stats.get("max_gpu_util_percent"), "%")),
            ("average memory util", _fmt_metric(summary.utilization_stats.get("avg_memory_util_percent"), "%")),
            ("max memory util", _fmt_metric(summary.utilization_stats.get("max_memory_util_percent"), "%")),
            ("average temperature", _fmt_metric(summary.thermal_stats.get("avg_temperature_c"), "C")),
            ("max temperature", _fmt_metric(summary.thermal_stats.get("max_temperature_c"), "C")),
            ("average graphics clock", _fmt_metric(summary.clock_stats.get("avg_graphics_clock_mhz"), "MHz")),
            ("average SM clock", _fmt_metric(summary.clock_stats.get("avg_sm_clock_mhz"), "MHz")),
            ("average memory clock", _fmt_metric(summary.clock_stats.get("avg_memory_clock_mhz"), "MHz")),
            ("power limit", _fmt_metric(summary.power_limit_watts, "W")),
        ]
        for row in rows:
            table.add_row(*row)
        return table

    def _field_table(self, summary: TelemetrySummary) -> Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field")
        table.add_column("Status")
        missing = set(summary.missing_fields)
        for field_name in sorted(missing):
            table.add_row(field_name, "[yellow]missing[/yellow]")
        if not missing:
            table.add_row("all tracked fields", "[green]available[/green]")
        return table

    def _capabilities_table(self, summary: TelemetrySummary) -> Table:
        capabilities = _capability_payload(summary.telemetry_capabilities)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Capability")
        table.add_column("Status")
        for field_name in capabilities.get("available_fields") or []:
            table.add_row(str(field_name), "[green]available[/green]")
        for field_name in capabilities.get("unavailable_fields") or []:
            table.add_row(str(field_name), "[yellow]unavailable[/yellow]")
        if not capabilities.get("available_fields") and not capabilities.get("unavailable_fields"):
            table.add_row("none", "[yellow]unavailable[/yellow]")
        for note in capabilities.get("notes") or []:
            table.add_row("note", str(note))
        return table


def _banner(title: str) -> str:
    line = "=" * SECTION_WIDTH
    return f"{line}\n{title}\n{line}"


def _section(title: str) -> str:
    line = "-" * SECTION_WIDTH
    return f"{line}\n{title}\n{line}"


def _fmt_metric(value: object, unit: str | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        rendered = f"{number:,.2f}"
    elif abs(number) >= 1:
        rendered = f"{number:.3f}"
    else:
        rendered = f"{number:.6f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip() if unit else rendered


def _fmt_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return str(value)


def _fmt_percent_delta(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _join_values(value: object) -> str:
    values = _list_values(value)
    return ", ".join(values) if values else "none"


def _list_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _telemetry_quality_explanation(value: object) -> str:
    quality = str(value or "unavailable")
    if quality == "good":
        return "power and utilization coverage are sufficient for power-aware scoring"
    if quality == "limited":
        return "power exists, but missing fields or sample timing limit confidence"
    if quality == "poor":
        return "sample coverage is too weak for strong power conclusions"
    return "no usable telemetry samples were collected"


def _capability_payload(value: object) -> dict[str, object]:
    if value is None:
        return {}
    payload = to_dict(value)
    return payload if isinstance(payload, dict) else {}


def _ms_from_seconds(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return None


def _source_label(source: str | None) -> str:
    if source == "aiconfigurator":
        return "AIConfigurator"
    if source == "aiconfigurator+sweep":
        return "AIConfigurator + concurrency sweep"
    if source == "sweep":
        return "Concurrency sweep"
    if source == "heuristic":
        return "Heuristic"
    return source or "unknown"


def _telemetry_path(result: RecommendationResult) -> str:
    if result.telemetry_requested is None:
        return "unknown"
    if result.telemetry_provider is None:
        return f"{result.telemetry_requested} -> unavailable"
    return f"{result.telemetry_requested} -> {result.telemetry_provider}"


def _artifact_lines(artifacts: dict[str, str]) -> list[str]:
    lines = []
    mapping = [
        ("report_txt", "report.txt"),
        ("recommendation_json", "recommendation.json"),
        ("scores_jsonl", "scores.jsonl"),
        ("pareto_frontier_json", "pareto_frontier.json"),
        ("pareto_frontier_csv", "pareto_frontier.csv"),
        ("summary_json", "summary.json"),
        ("metadata_json", "metadata.json"),
    ]
    for key, label in mapping:
        if key in artifacts:
            lines.append(f"{label}: {artifacts[key]}")
    if "evaluation_run_dir" in artifacts:
        lines.append(f"internal evaluation artifacts: {artifacts['evaluation_run_dir']}")
    if "telemetry_summary_json" in artifacts:
        lines.append(f"selected telemetry summary: {artifacts['telemetry_summary_json']}")
    if "telemetry_capabilities_json" in artifacts:
        lines.append(f"selected telemetry capabilities: {artifacts['telemetry_capabilities_json']}")
    return lines


def _bullet_text(items: list[str]) -> Text:
    if not items:
        return Text("none")
    return Text("\n".join(f"- {item}" for item in items))
