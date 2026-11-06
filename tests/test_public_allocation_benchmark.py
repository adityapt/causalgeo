import importlib.util
from pathlib import Path


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_allocation_benchmark.py"
    spec = importlib.util.spec_from_file_location("public_allocation_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_meridian_panel_scores_the_bundle_allocation():
    result = _benchmark().run()

    assert result["source"].endswith("geo_all_channels.csv")
    assert "zillow" not in result["source"].lower()
    assert result["n_geos"] == 40
    assert abs(result["reconciliation_error"]) < 1e-6
    assert result["max_abs_informed_share_error"] < 1e-8
    assert result["max_abs_equal_share_error"] > 0.05
    assert abs(result["bundle_effect"] - result["planted_bundle_effect"]) < 5 * result["bundle_std_err"]
    channels = {row["channel"]: row for row in result["channels"]}

    def percent(share: float) -> float:
        return round(share * 100, 1)

    assert percent(channels["Channel2"]["share_equal"]) == 7.8
    assert percent(channels["Channel2"]["planted_share"]) == 2.4
    assert percent(channels["Channel1"]["share_equal"]) == 12.7
    assert percent(channels["Channel1"]["planted_share"]) == 19.2
