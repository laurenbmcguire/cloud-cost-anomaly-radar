import pandas as pd

from costradar.evaluation import evaluate_events, group_events, match_events


def _flags(days_on, start="2025-03-01", n=30):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series([i in days_on for i in range(n)], index=idx)


def test_group_events_merges_single_day_gaps_only():
    events = group_events(_flags({2, 3, 5, 10, 11, 20}), max_gap=1)
    spans = [(e.start.day, e.end.day, e.n_days) for e in events]
    assert spans == [(3, 6, 3), (11, 12, 2), (21, 21, 1)]
    assert group_events(_flags(set())) == []


def test_event_level_metrics_with_two_day_tolerance():
    truth = pd.DataFrame(
        {
            "anomaly_id": ["A", "B", "C"],
            "start_date": ["2025-03-05", "2025-03-15", "2025-03-25"],
            "end_date": ["2025-03-06", "2025-03-15", "2025-03-26"],
        }
    )
    # Day index 0 = 2025-03-01. Event on 03-08 is 2 days after A ends -> hit.
    # Event on 03-18 is 3 days after B -> miss (false positive). C is never hit.
    events = group_events(_flags({7, 17}))
    m = evaluate_events(events, truth, tolerance_days=2)
    assert m["predicted_events"] == 2
    assert m["true_positive_events"] == 1
    assert m["detected_anomalies"] == ["A"]
    assert m["missed_anomalies"] == ["B", "C"]
    assert m["precision"] == 0.5
    assert round(m["recall"], 4) == round(1 / 3, 4)
    assert round(m["f1"], 4) == round(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3), 4)
    assert match_events(events, truth) == ["A", None]
    empty = evaluate_events([], truth)
    assert (empty["precision"], empty["recall"], empty["f1"]) == (0.0, 0.0, 0.0)
