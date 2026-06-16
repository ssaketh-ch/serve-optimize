from serve_optimize.landscape import LANDSCAPE, grouped_landscape


def test_landscape_contains_anchor_systems() -> None:
    names = {item.name for item in LANDSCAPE}
    assert "AIConfigurator" in names
    assert "TokenPowerBench" in names
    assert "vLLM / PagedAttention" in names


def test_grouped_landscape() -> None:
    grouped = grouped_landscape()
    assert "serving backend" in grouped
    assert "telemetry" in grouped

