"""End-to-end pipeline: data -> detection -> evaluation -> attribution -> outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import charts
from .attribution import attribute_event, estimate_excess, pattern_label, series_baselines, to_wide
from .detectors import DETECTORS, DetectorConfig, run_detectors
from .evaluation import evaluate_events, group_events, match_events
from .generator import (
    DEFAULT_SEED,
    generate_billing,
    load_dataset,
    planned_events_table,
    save_dataset,
    series_catalog,
    write_examples,
)
from .report import write_report

TOLERANCE_DAYS = 2
ROBUSTNESS_SEEDS = (3, 7, 11, 99)
ALL_DETECTORS = (*DETECTORS, "ensemble")
HERO_BAND_SIGMA = 3.0


def severity(excess_usd: float) -> str:
    if excess_usd >= 10_000:
        return "High"
    if excess_usd >= 3_000:
        return "Medium"
    return "Low"


def aggregate_views(billing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    wide = to_wide(billing)
    total = wide.sum(axis=1)
    by_provider = wide.T.groupby(level="provider").sum().T
    by_category = billing.pivot_table(index="date", columns="category", values="cost_usd", aggfunc="sum")
    by_category.index = pd.DatetimeIndex(by_category.index)
    return wide, total, by_provider, by_category


def evaluate_all(scores: pd.DataFrame, truth: pd.DataFrame) -> dict:
    out = {}
    for name in ALL_DETECTORS:
        events = group_events(scores[f"{name}_flag"])
        metrics = evaluate_events(events, truth, tolerance_days=TOLERANCE_DAYS)
        metrics["flagged_days"] = int(scores[f"{name}_flag"].sum())
        out[name] = metrics
    return out


def robustness_check(seeds=ROBUSTNESS_SEEDS, config: DetectorConfig | None = None) -> dict:
    """Re-run detection on other generator seeds to show the result is not seed luck."""
    per_seed = {}
    for seed in seeds:
        billing, truth = generate_billing(seed=seed)
        _, total, by_provider, by_category = aggregate_views(billing)
        scores = run_detectors(total, by_provider, by_category, config)
        metrics = evaluate_all(scores, truth)
        per_seed[str(seed)] = {n: {k: metrics[n][k] for k in ("precision", "recall", "f1")} for n in ALL_DETECTORS}
    mean_f1 = {
        n: round(sum(per_seed[s][n]["f1"] for s in per_seed) / len(per_seed), 4) for n in ALL_DETECTORS
    }
    return {"seeds": list(seeds), "per_seed": per_seed, "mean_f1": mean_f1}


def _planned_overlap(start: pd.Timestamp, end: pd.Timestamp, planned: pd.DataFrame) -> str | None:
    tol = pd.Timedelta(days=TOLERANCE_DAYS)
    for row in planned.itertuples(index=False):
        if start <= pd.Timestamp(row.end_date) + tol and end >= pd.Timestamp(row.start_date) - tol:
            return row.event_id
    return None


def run_pipeline(
    root: Path | str = ".",
    seed: int = DEFAULT_SEED,
    regenerate: bool = False,
    robustness: bool = True,
    make_charts: bool = True,
) -> dict:
    root = Path(root)
    data_dir, reports_dir, img_dir = root / "data", root / "reports", root / "docs" / "img"
    examples_dir = root / "examples"

    if regenerate or not (data_dir / "billing_daily.csv").exists():
        billing, truth = generate_billing(seed=seed)
        save_dataset(billing, truth, data_dir)
        write_examples(billing, truth, examples_dir)
    billing, truth = load_dataset(data_dir)
    planned = planned_events_table()

    cfg = DetectorConfig(seed=seed)
    wide, total, by_provider, by_category = aggregate_views(billing)
    scores = run_detectors(total, by_provider, by_category, cfg)
    metrics = evaluate_all(scores, truth)

    # ---- per-event attribution and savings (ensemble events) -------------- #
    baselines = series_baselines(wide)
    events = group_events(scores["ensemble_flag"])
    matches = match_events(events, truth, tolerance_days=TOLERANCE_DAYS)
    catalog = series_catalog().set_index("service")["category"].to_dict()
    truth_by_id = truth.set_index("anomaly_id")

    anomalies = []
    for i, (ev, match) in enumerate(zip(events, matches), start=1):
        excess = estimate_excess(total, scores["baseline"], ev.days)
        attribution = attribute_event(wide, baselines, ev.days)
        driver = dict(attribution["contributors"][0])
        driver["category"] = catalog.get(driver["service"], "unknown")
        peak = pd.Timestamp(excess["peak_date"])
        record = {
            "event_id": f"E{i:02d}",
            "start": ev.start.date().isoformat(),
            "end": ev.end.date().isoformat(),
            "flagged_days": ev.n_days,
            "pattern": pattern_label(ev),
            "peak_date": excess["peak_date"],
            "ensemble_score_peak": round(float(scores.loc[ev.days, "ensemble_score"].max()), 3),
            "top_monitor": str(scores.loc[peak, "seasonal_residual_monitor"]),
            "excess_usd": excess["excess_usd"],
            "peak_excess_usd": excess["peak_excess_usd"],
            "excess_pct_of_baseline": excess["excess_pct_of_baseline"],
            "severity": severity(excess["excess_usd"]),
            "driver": driver,
            "attribution": attribution,
            "matched_truth": match,
            "planned_event": None if match else _planned_overlap(ev.start, ev.end, planned),
        }
        if match:
            t = truth_by_id.loc[match]
            record["truth_type"] = t["type"]
            record["truth_driver"] = {"provider": t["provider"], "account": t["account"], "service": t["service"]}
            record["attribution_correct"] = all(
                driver[k] == record["truth_driver"][k] for k in ("provider", "account", "service")
            )
            top3 = attribution["contributors"][:3]
            record["truth_in_top3"] = any(
                all(c[k] == record["truth_driver"][k] for k in ("provider", "account", "service")) for c in top3
            )
        anomalies.append(record)

    # ---- attribution accuracy: one (largest) event per detected anomaly --- #
    best_event: dict[str, dict] = {}
    for a in anomalies:
        if a["matched_truth"] and (
            a["matched_truth"] not in best_event or a["excess_usd"] > best_event[a["matched_truth"]]["excess_usd"]
        ):
            best_event[a["matched_truth"]] = a
    n_detected = len(best_event)
    top1 = sum(1 for a in best_event.values() if a["attribution_correct"])
    top3 = sum(1 for a in best_event.values() if a["truth_in_top3"])

    detected_ids = sorted(best_event)
    est_matched = sum(a["excess_usd"] for a in anomalies if a["matched_truth"])
    injected_detected = float(truth.set_index("anomaly_id").loc[detected_ids, "injected_excess_usd"].sum()) if detected_ids else 0.0
    est_total = sum(a["excess_usd"] for a in anomalies)

    ens = metrics["ensemble"]
    best_single = max(DETECTORS, key=lambda n: (metrics[n]["f1"], metrics[n]["precision"]))
    largest = max(anomalies, key=lambda a: a["excess_usd"]) if anomalies else None
    fp_events = [a for a in anomalies if not a["matched_truth"]]

    summary = {
        "project": "cloud-cost-anomaly-radar",
        "seed": seed,
        "data": {
            "start": total.index[0].date().isoformat(),
            "end": total.index[-1].date().isoformat(),
            "days": int(len(total)),
            "providers": int(billing["provider"].nunique()),
            "services": int(billing.groupby("provider")["service"].nunique().sum()),
            "accounts": int(billing["account"].nunique()),
            "series": int(wide.shape[1]),
            "line_items": int(len(billing)),
            "total_spend_usd": round(float(total.sum()), 2),
            "avg_daily_spend_usd": round(float(total.mean()), 2),
            "spend_by_provider_usd": {k: round(float(v), 2) for k, v in by_provider.sum().items()},
            "injected_anomalies": int(len(truth)),
            "injected_excess_usd": round(float(truth["injected_excess_usd"].sum()), 2),
            "planned_events": int(len(planned)),
        },
        "config": {**asdict(cfg), "tolerance_days": TOLERANCE_DAYS, "event_merge_gap_days": 1},
        "detectors": metrics,
        "best_single_detector": best_single,
        "ensemble": {
            "precision": ens["precision"],
            "recall": ens["recall"],
            "f1": ens["f1"],
            "events_flagged": ens["predicted_events"],
            "false_positive_events": ens["false_positive_events"],
            "false_positives_on_planned_events": sum(1 for a in fp_events if a["planned_event"]),
            "missed_anomalies": ens["missed_anomalies"],
        },
        "attribution": {
            "detected_anomalies": n_detected,
            "top1_driver_correct": top1,
            "top1_accuracy": round(top1 / n_detected, 4) if n_detected else 0.0,
            "top3_accuracy": round(top3 / n_detected, 4) if n_detected else 0.0,
        },
        "savings": {
            "estimated_excess_usd_all_events": round(est_total, 2),
            "estimated_excess_usd_matched_events": round(est_matched, 2),
            "injected_excess_usd_detected_anomalies": round(injected_detected, 2),
            "capture_ratio": round(est_matched / injected_detected, 4) if injected_detected else 0.0,
        },
        "largest_anomaly": None if largest is None else {
            k: largest[k] for k in ("event_id", "start", "end", "excess_usd", "driver", "matched_truth")
        },
        "anomalies": anomalies,
    }
    if robustness:
        summary["robustness"] = robustness_check(config=DetectorConfig())

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_report(summary, truth, planned, reports_dir / "finops_report.md")

    if make_charts:
        charts.hero_chart(scores, anomalies, img_dir / "hero_spend_anomalies.png", band_k=HERO_BAND_SIGMA)
        charts.provider_stacked_area(by_provider, img_dir / "spend_by_provider.png")
        charts.attribution_chart(anomalies, img_dir / "attribution_top_anomalies.png")
        charts.detector_comparison_chart(metrics, img_dir / "detector_comparison.png")
    return summary
