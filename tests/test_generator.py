import numpy as np
import pandas as pd

from costradar.generator import (
    ANOMALIES,
    PROVIDERS,
    anomaly_multiplier,
    generate_billing,
    month_end_mask,
)


def test_generator_is_deterministic_and_complete():
    b1, t1 = generate_billing(seed=42)
    b2, t2 = generate_billing(seed=42)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(t1, t2)
    assert b1["date"].nunique() == 365
    assert b1.groupby(["provider", "account", "service"]).ngroups == 54
    assert len(b1) == 365 * 54
    assert (b1["cost_usd"] > 0).all()
    assert len(t1) == 12
    assert set(t1["type"]) == {"spike", "level_shift", "slow_leak"}
    b3, _ = generate_billing(seed=7)
    assert not np.allclose(b1["cost_usd"], b3["cost_usd"])


def test_injected_excess_matches_ground_truth():
    clean, _ = generate_billing(seed=42, inject=False)
    dirty, truth = generate_billing(seed=42, inject=True)
    diff = dirty.assign(delta=dirty["cost_usd"] - clean["cost_usd"])
    for row in truth.itertuples(index=False):
        mask = (
            (diff["provider"] == row.provider)
            & (diff["account"] == row.account)
            & (diff["service"] == row.service)
        )
        added = diff.loc[mask, "delta"].sum()
        # Rounding to cents on each of <= 21 days bounds the error well below $1.
        assert abs(added - row.injected_excess_usd) < 1.0
        window = diff.loc[mask & (diff["delta"].abs() > 0.02), "date"]
        assert window.min() == pd.Timestamp(row.start_date)
        assert window.max() == pd.Timestamp(row.end_date)
    untouched = ~diff.set_index(["provider", "account", "service"]).index.isin(
        list(zip(truth["provider"], truth["account"], truth["service"]))
    )
    assert diff.loc[untouched, "delta"].abs().max() < 1e-9


def test_anomaly_shapes_and_calendar_helpers():
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    for spec in ANOMALIES:
        assert spec.account in PROVIDERS[spec.provider]["accounts"]
    leak = next(s for s in ANOMALIES if s.kind == "slow_leak")
    leak_dates = pd.date_range(leak.start, periods=leak.days + 2, freq="D")
    ramp = anomaly_multiplier(leak, leak_dates)
    assert np.all(np.diff(ramp[: leak.days]) > 0)
    assert np.isclose(ramp[leak.days - 1], leak.magnitude)
    assert ramp[leak.days] == 0
    me = month_end_mask(dates)
    assert list(dates[me].strftime("%m-%d")) == ["01-30", "01-31"]
