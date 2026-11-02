import numpy as np
import pandas as pd

from costradar.detectors import (
    MAD_TO_SIGMA,
    isolation_forest_score,
    mad,
    robust_sigma,
    rolling_mad_zscore,
    seasonal_baseline,
)


def test_mad_and_rolling_robust_zscore_flags_spike():
    assert mad(np.array([1.0, 2.0, 3.0, 4.0, 100.0])) == 1.0
    assert np.isclose(robust_sigma(np.array([1.0, 2.0, 3.0, 4.0, 100.0])), MAD_TO_SIGMA)

    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    x = pd.Series(1000 + rng.normal(0, 10, len(idx)), index=idx)
    x.iloc[60] += 200  # a 20-sigma spike
    out = rolling_mad_zscore(x, window=28, min_periods=14)
    assert out["score"].idxmax() == idx[60]
    assert out["score"].iloc[60] > 10
    # The baseline uses only past values, so the spike does not move its own baseline.
    assert abs(out["baseline"].iloc[60] - 1000) < 15
    # A flat history does not explode thanks to the scale floor.
    flat = pd.Series([500.0] * 30 + [520.0], index=pd.date_range("2025-01-01", periods=31, freq="D"))
    assert np.isfinite(rolling_mad_zscore(flat)["score"].iloc[-1])


def test_seasonal_baseline_recovers_weekday_profile():
    idx = pd.date_range("2025-01-06", periods=140, freq="D")  # starts on a Monday
    weekday_true = np.array([1.05, 1.05, 1.05, 1.05, 1.05, 0.875, 0.875])
    x = pd.Series(2000 * weekday_true[idx.dayofweek], index=idx)
    base = seasonal_baseline(x, window=21)
    season = base["season"].groupby(idx.dayofweek).first().to_numpy()
    # Month-end days carry their own factor; compare on a non month-end Saturday/Monday.
    assert season[5] < season[0]
    assert np.allclose(base["baseline"].iloc[30:], x.iloc[30:], rtol=0.01)


def test_isolation_forest_scores_injected_outlier_highest():
    rng = np.random.default_rng(1)
    feats = pd.DataFrame(rng.normal(0, 1, size=(200, 4)), columns=list("abcd"))
    feats.iloc[150] = [8.0, -7.0, 9.0, 6.0]
    out = isolation_forest_score(feats, contamination=0.05, warmup_days=0, seed=42)
    assert out["norm"].idxmax() == 150
    assert out["norm"].iloc[150] >= 1.0
    again = isolation_forest_score(feats, contamination=0.05, warmup_days=0, seed=42)
    pd.testing.assert_frame_equal(out, again)
