"""Event grouping and event-level precision / recall / F1 with a date tolerance."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Event:
    """A contiguous run of flagged days (short gaps allowed)."""

    start: pd.Timestamp
    end: pd.Timestamp
    days: list[pd.Timestamp] = field(default_factory=list)

    @property
    def n_days(self) -> int:
        return len(self.days)


def group_events(flags: pd.Series, max_gap: int = 1) -> list[Event]:
    """Group flagged dates into events.

    Two flagged days belong to the same event when at most ``max_gap``
    unflagged days separate them.
    """
    flagged = [pd.Timestamp(d) for d, f in flags.items() if bool(f)]
    events: list[Event] = []
    for day in sorted(flagged):
        if events and (day - events[-1].end).days <= max_gap + 1:
            events[-1].end = day
            events[-1].days.append(day)
        else:
            events.append(Event(start=day, end=day, days=[day]))
    return events


def _overlaps(event: Event, start: pd.Timestamp, end: pd.Timestamp, tolerance_days: int) -> bool:
    tol = pd.Timedelta(days=tolerance_days)
    return event.start <= end + tol and event.end >= start - tol


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def match_events(events: list[Event], truth: pd.DataFrame, tolerance_days: int = 2) -> list[str | None]:
    """For each predicted event, the id of the best-overlapping true anomaly (or None)."""
    matches: list[str | None] = []
    for ev in events:
        best_id, best_overlap = None, -1
        for row in truth.itertuples(index=False):
            start, end = pd.Timestamp(row.start_date), pd.Timestamp(row.end_date)
            if _overlaps(ev, start, end, tolerance_days):
                overlap = sum(1 for d in ev.days if start <= d <= end)
                if overlap > best_overlap:
                    best_id, best_overlap = row.anomaly_id, overlap
        matches.append(best_id)
    return matches


def evaluate_events(events: list[Event], truth: pd.DataFrame, tolerance_days: int = 2) -> dict:
    """Event-level metrics.

    * precision = predicted events that overlap any true window (+/- tolerance)
      divided by all predicted events
    * recall = true anomalies hit by at least one predicted event divided by all
      true anomalies
    """
    n_pred, n_true = len(events), len(truth)
    hit_pred = 0
    detected: set[str] = set()
    for ev in events:
        hit = False
        for row in truth.itertuples(index=False):
            if _overlaps(ev, pd.Timestamp(row.start_date), pd.Timestamp(row.end_date), tolerance_days):
                detected.add(row.anomaly_id)
                hit = True
        hit_pred += int(hit)
    precision = hit_pred / n_pred if n_pred else 0.0
    recall = len(detected) / n_true if n_true else 0.0
    return {
        "predicted_events": n_pred,
        "true_events": n_true,
        "true_positive_events": hit_pred,
        "false_positive_events": n_pred - hit_pred,
        "detected_anomalies": sorted(detected),
        "missed_anomalies": sorted(set(truth["anomaly_id"]) - detected),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
    }
