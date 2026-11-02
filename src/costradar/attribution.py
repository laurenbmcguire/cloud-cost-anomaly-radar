"""Root-cause attribution and savings estimates for detected anomaly events.

Each of the 54 provider/account/service series gets its own causal seasonal
baseline. For an event, the per-series delta is

    delta_s = sum over event days of (actual_s - baseline_s)

Series are ranked by delta and also rolled up by provider, service and account.
The savings estimate is the dollar excess of total spend over the total
baseline on the event's flagged days, counting only days above baseline.
"""

from __future__ import annotations

import pandas as pd

from .detectors import seasonal_baseline
from .evaluation import Event


def to_wide(billing: pd.DataFrame) -> pd.DataFrame:
    """Pivot long billing to a date x (provider, account, service) cost matrix."""
    wide = billing.pivot_table(
        index="date",
        columns=["provider", "account", "service"],
        values="cost_usd",
        aggfunc="sum",
    ).sort_index(axis=1)
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.fillna(0.0)


def series_baselines(wide: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Seasonal baseline for every series (same shape as ``wide``)."""
    cols = {}
    for col in wide.columns:
        cols[col] = seasonal_baseline(wide[col], window=window)["baseline"]
    out = pd.DataFrame(cols, index=wide.index)
    out.columns = wide.columns
    return out


def attribute_event(
    wide: pd.DataFrame,
    baselines: pd.DataFrame,
    days: list[pd.Timestamp],
    top_n: int = 5,
) -> dict:
    """Rank series and dimensions by their contribution to the event's delta."""
    actual = wide.loc[days].sum()
    expected = baselines.loc[days].sum()
    delta = (actual - expected).sort_values(ascending=False)
    positive_total = float(delta[delta > 0].sum())

    contributors = []
    for (provider, account, service), value in delta.head(top_n).items():
        contributors.append(
            {
                "provider": provider,
                "account": account,
                "service": service,
                "delta_usd": round(float(value), 2),
                "share": round(float(value) / positive_total, 4) if positive_total > 0 else 0.0,
            }
        )

    def rollup(level: str) -> list[dict]:
        grouped = delta.groupby(level=level).sum().sort_values(ascending=False)
        return [
            {"name": str(k), "delta_usd": round(float(v), 2)}
            for k, v in grouped.items()
        ]

    return {
        "contributors": contributors,
        "by_provider": rollup("provider"),
        "by_service": rollup("service"),
        "by_account": rollup("account"),
        "net_delta_usd": round(float(delta.sum()), 2),
    }


def estimate_excess(total: pd.Series, baseline: pd.Series, days: list[pd.Timestamp]) -> dict:
    """Dollar excess over baseline on the event days (negative days count as 0)."""
    diff = (total.loc[days] - baseline.loc[days]).clip(lower=0.0)
    peak_day = diff.idxmax() if len(diff) else None
    return {
        "excess_usd": round(float(diff.sum()), 2),
        "peak_excess_usd": round(float(diff.max()) if len(diff) else 0.0, 2),
        "peak_date": peak_day.date().isoformat() if peak_day is not None else None,
        "excess_pct_of_baseline": round(
            float(diff.sum() / baseline.loc[days].sum()) if len(diff) else 0.0, 4
        ),
    }


def pattern_label(event: Event) -> str:
    """Heuristic label from the shape of the flagged run."""
    return "spike" if event.n_days <= 3 else "sustained"
