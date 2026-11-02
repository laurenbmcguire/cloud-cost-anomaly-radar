"""Anomaly detectors for a daily spend series and their ensemble.

Three detectors look at the same total-spend series from different angles:

1. ``rolling_mad``: robust z-score against a trailing rolling median and MAD.
   No seasonality model, so it is simple but noisy around weekends and
   month-end batch runs.
2. ``seasonal_residual``: removes a weekday profile and a month-end factor,
   tracks the de-seasonalised level with a trailing median and scores the
   relative residual with a global MAD scale.
3. ``isolation_forest``: scikit-learn IsolationForest on engineered features
   (seasonal residual, ratio to trailing median, day-over-day and week-over-week
   change, provider and service-category mix shifts).

Only cost *increases* are flagged; drops are usually data latency, not waste.
The ensemble flags a day when at least ``min_votes`` detectors agree and ranks
days by the mean of the normalised scores (each detector's threshold maps to 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .generator import month_end_mask

MAD_TO_SIGMA = 1.4826
DETECTORS = ("rolling_mad", "seasonal_residual", "isolation_forest")
DETECTOR_LABELS = {
    "rolling_mad": "Rolling median + MAD",
    "seasonal_residual": "Seasonal residual",
    "isolation_forest": "Isolation Forest",
    "ensemble": "Ensemble (2 of 3 vote)",
}


@dataclass(frozen=True)
class DetectorConfig:
    mad_window: int = 28
    mad_threshold: float = 4.5
    seasonal_window: int = 21
    seasonal_threshold: float = 4.5
    if_contamination: float = 0.08
    if_estimators: int = 300
    min_votes: int = 2
    warmup_days: int = 21
    seed: int = 42


def mad(values: np.ndarray) -> float:
    """Median absolute deviation (unscaled)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def robust_sigma(values: np.ndarray) -> float:
    """MAD scaled to be a consistent estimator of the normal sigma."""
    return MAD_TO_SIGMA * mad(values)


# --------------------------------------------------------------------------- #
# Detector 1: rolling median + MAD robust z-score
# --------------------------------------------------------------------------- #
def rolling_mad_zscore(x: pd.Series, window: int = 28, min_periods: int = 14) -> pd.DataFrame:
    """Robust z-score of each day against the *previous* ``window`` days.

    z_t = (x_t - median(x_{t-w..t-1})) / (1.4826 * MAD(x_{t-w..t-1}))

    The scale is floored at 0.5% of the median so a perfectly flat history does
    not produce infinite scores.
    """
    hist = x.shift(1).rolling(window, min_periods=min_periods)
    med = hist.median()
    mad_vals = hist.apply(mad, raw=True)
    scale = (MAD_TO_SIGMA * mad_vals).clip(lower=0.005 * med.abs())
    z = (x - med) / scale
    return pd.DataFrame({"baseline": med, "scale": scale, "score": z})


# --------------------------------------------------------------------------- #
# Detector 2: seasonal residual
# --------------------------------------------------------------------------- #
def fit_seasonality(x: pd.Series) -> tuple[np.ndarray, float]:
    """Estimate weekday factors (mean 1) and a month-end factor robustly.

    Ratios to a centred 15-day rolling median are pooled by weekday and the
    median ratio is used, which keeps short spikes from leaking into the
    seasonal profile.
    """
    idx = pd.DatetimeIndex(x.index)
    trend = x.rolling(15, center=True, min_periods=5).median()
    ratio = (x / trend).to_numpy()
    me = month_end_mask(idx)
    wd = np.asarray(idx.dayofweek)
    raw = np.array([np.nanmedian(ratio[(wd == d) & ~me]) for d in range(7)])
    # The month-end factor is relative to the same weekday, so divide by the
    # un-normalised weekday ratios (both are measured against the same trend).
    month_end = float(np.nanmedian(ratio[me] / raw[wd[me]])) if me.any() else 1.0
    return raw / raw.mean(), month_end


def seasonal_baseline(x: pd.Series, window: int = 21, min_periods: int = 7) -> pd.DataFrame:
    """Causal seasonal baseline: trailing median level x weekday x month-end factor."""
    idx = pd.DatetimeIndex(x.index)
    weekday, month_end = fit_seasonality(x)
    season = weekday[np.asarray(idx.dayofweek)] * np.where(month_end_mask(idx), month_end, 1.0)
    season = pd.Series(season, index=x.index)
    deseason = x / season
    level = deseason.shift(1).rolling(window, min_periods=min_periods).median()
    level = level.fillna(float(deseason.iloc[:min_periods].median()))
    baseline = level * season
    return pd.DataFrame({"baseline": baseline, "level": level, "season": season})


def seasonal_residual_score(x: pd.Series, window: int = 21, warmup_days: int = 21) -> pd.DataFrame:
    """Relative residual vs the seasonal baseline, scaled by its global MAD."""
    base = seasonal_baseline(x, window=window)
    rel = (x - base["baseline"]) / base["baseline"]
    ref = rel.iloc[warmup_days:]
    center = float(np.median(ref))
    sigma = max(robust_sigma(ref.to_numpy()), 1e-6)
    score = (rel - center) / sigma
    out = base.copy()
    out["rel_resid"] = rel
    out["sigma"] = sigma
    out["score"] = score
    return out


# --------------------------------------------------------------------------- #
# Detector 3: Isolation Forest on engineered features
# --------------------------------------------------------------------------- #
def _share_deviation(part: pd.DataFrame, total: pd.Series, window: int = 28) -> pd.DataFrame:
    share = part.div(total, axis=0)
    ref = share.shift(1).rolling(window, min_periods=7).median()
    return (share - ref).fillna(0.0)


def build_features(
    total: pd.Series,
    by_provider: pd.DataFrame,
    by_category: pd.DataFrame,
    rel_resid: pd.Series,
    window: int = 28,
) -> pd.DataFrame:
    """Engineer per-day features for the Isolation Forest."""
    trailing_med = total.shift(1).rolling(window, min_periods=7).median()
    feats = pd.DataFrame(index=total.index)
    feats["seasonal_rel_resid"] = rel_resid
    feats["ratio_to_trailing_median"] = total / trailing_med - 1.0
    feats["dod_change"] = total / total.shift(1) - 1.0
    feats["wow_change"] = total / total.shift(7) - 1.0
    prov = _share_deviation(by_provider, total, window).add_prefix("share_dev_provider_")
    cat = _share_deviation(by_category, total, window).add_prefix("share_dev_category_")
    feats = pd.concat([feats, prov, cat], axis=1)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def isolation_forest_score(
    features: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 300,
    warmup_days: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """Anomaly score = -score_samples; threshold = (1 - contamination) quantile."""
    fit_rows = features.iloc[warmup_days:]
    scaler = RobustScaler().fit(fit_rows)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(scaler.transform(fit_rows))
    raw = -model.score_samples(scaler.transform(features))
    raw = pd.Series(raw, index=features.index)
    threshold = float(np.quantile(raw.iloc[warmup_days:], 1.0 - contamination))
    center = float(np.median(raw.iloc[warmup_days:]))
    norm = (raw - center) / max(threshold - center, 1e-9)
    return pd.DataFrame({"raw": raw, "threshold": threshold, "norm": norm})


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def build_monitors(
    total: pd.Series, by_provider: pd.DataFrame, by_category: pd.DataFrame
) -> dict[str, pd.Series]:
    """The cost views the statistical detectors watch: total, each provider, each category."""
    monitors = {"Total": total}
    for col in by_provider.columns:
        monitors[f"Provider: {col}"] = by_provider[col]
    for col in by_category.columns:
        monitors[f"Category: {col}"] = by_category[col]
    return monitors


def _max_over_monitors(scores: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame(scores).fillna(0.0)
    return frame.max(axis=1), frame.idxmax(axis=1)


def run_detectors(
    total: pd.Series,
    by_provider: pd.DataFrame,
    by_category: pd.DataFrame,
    config: DetectorConfig | None = None,
) -> pd.DataFrame:
    """Run all detectors and return one row per day.

    The two statistical detectors score every monitor (total, 3 providers,
    6 service categories) and keep the maximum upward score per day, so a spike
    that is small in the grand total but large inside one category still
    registers. Their thresholds are set higher than a single-series test would
    use to absorb the multiple comparisons.
    """
    cfg = config or DetectorConfig()
    eligible = np.arange(len(total)) >= cfg.warmup_days
    monitors = build_monitors(total, by_provider, by_category)

    mad_scores = {k: rolling_mad_zscore(v, window=cfg.mad_window)["score"] for k, v in monitors.items()}
    seasonal = {
        k: seasonal_residual_score(v, window=cfg.seasonal_window, warmup_days=cfg.warmup_days)
        for k, v in monitors.items()
    }
    d1_score, d1_monitor = _max_over_monitors(mad_scores)
    d2_score, d2_monitor = _max_over_monitors({k: v["score"] for k, v in seasonal.items()})
    d2_total = seasonal["Total"]

    feats = build_features(total, by_provider, by_category, d2_total["rel_resid"], window=cfg.mad_window)
    d3 = isolation_forest_score(
        feats,
        contamination=cfg.if_contamination,
        n_estimators=cfg.if_estimators,
        warmup_days=cfg.warmup_days,
        seed=cfg.seed,
    )

    out = pd.DataFrame(index=total.index)
    out["total"] = total
    out["baseline"] = d2_total["baseline"]
    out["band_sigma_usd"] = d2_total["sigma"] * d2_total["baseline"]
    out["rolling_median"] = rolling_mad_zscore(total, window=cfg.mad_window)["baseline"]

    out["rolling_mad_score"] = d1_score
    out["rolling_mad_monitor"] = d1_monitor
    out["seasonal_residual_score"] = d2_score
    out["seasonal_residual_monitor"] = d2_monitor
    out["isolation_forest_score"] = d3["norm"]

    out["rolling_mad_norm"] = out["rolling_mad_score"] / cfg.mad_threshold
    out["seasonal_residual_norm"] = out["seasonal_residual_score"] / cfg.seasonal_threshold
    out["isolation_forest_norm"] = out["isolation_forest_score"]

    # Isolation Forest is direction-agnostic; only keep days where spend is up.
    upward_total = (total > d2_total["baseline"]).to_numpy()
    out["rolling_mad_flag"] = (out["rolling_mad_norm"] >= 1.0).to_numpy() & eligible
    out["seasonal_residual_flag"] = (out["seasonal_residual_norm"] >= 1.0).to_numpy() & eligible
    out["isolation_forest_flag"] = (out["isolation_forest_norm"] >= 1.0).to_numpy() & eligible & upward_total

    votes = out[[f"{n}_flag" for n in DETECTORS]].sum(axis=1)
    out["votes"] = votes
    out["ensemble_score"] = out[[f"{n}_norm" for n in DETECTORS]].clip(-2, 5).mean(axis=1)
    out["ensemble_flag"] = (votes >= cfg.min_votes).to_numpy() & eligible
    return out
