import numpy as np
import pandas as pd

from costradar.attribution import attribute_event, estimate_excess, series_baselines


def test_attribution_ranks_the_spiking_series_first():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=90, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [("AWS", "prod", "EC2"), ("AWS", "prod", "S3"), ("GCP", "ml", "Vertex AI"), ("Azure", "dev", "VMs")],
        names=["provider", "account", "service"],
    )
    levels = np.array([4000.0, 1200.0, 900.0, 600.0])
    wide = pd.DataFrame(levels * (1 + rng.normal(0, 0.02, (90, 4))), index=idx, columns=cols)
    spike_days = list(idx[70:72])
    wide.loc[spike_days, ("GCP", "ml", "Vertex AI")] += 3000.0
    wide.loc[spike_days, ("AWS", "prod", "S3")] += 300.0

    result = attribute_event(wide, series_baselines(wide), spike_days, top_n=3)
    top = result["contributors"][0]
    assert (top["provider"], top["account"], top["service"]) == ("GCP", "ml", "Vertex AI")
    assert 5000 < top["delta_usd"] < 7000
    assert result["contributors"][1]["service"] == "S3"
    assert 0.8 < top["share"] <= 1.0
    assert result["by_provider"][0]["name"] == "GCP"


def test_excess_counts_only_days_above_baseline():
    idx = pd.date_range("2025-06-01", periods=4, freq="D")
    total = pd.Series([110.0, 90.0, 150.0, 100.0], index=idx)
    baseline = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    out = estimate_excess(total, baseline, list(idx))
    assert out["excess_usd"] == 60.0  # 10 + 0 + 50 + 0
    assert out["peak_excess_usd"] == 50.0
    assert out["peak_date"] == "2025-06-03"
    assert out["excess_pct_of_baseline"] == 0.15
