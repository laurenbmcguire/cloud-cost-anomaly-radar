import io
import json
from datetime import date

import pytest

from costradar import ai_pm

TODAY = date(2026, 11, 2)

MILESTONES = [
    {"id": "M1", "name": "Generator", "due": "2026-10-01", "status": "done"},
    {"id": "M2", "name": "Detectors", "due": "2026-12-01", "status": "in_progress"},
    {"id": "M3", "name": "Slack", "due": "2027-01-01", "status": "planned"},
]
BACKLOG = [
    {"id": "B1", "title": "gen", "priority": "P0", "status": "done", "owner": "Lauren", "depends_on": [], "milestone": "M1"},
    {"id": "B2", "title": "mad", "priority": "P1", "status": "done", "owner": "AI PM", "depends_on": ["B1"], "milestone": "M2"},
    {"id": "B3", "title": "iforest", "priority": "P1", "status": "todo", "owner": "Lauren", "depends_on": ["B1", "B2"], "milestone": "M2"},
    {"id": "B4", "title": "eval", "priority": "P1", "status": "todo", "owner": "AI PM", "depends_on": ["B1"], "milestone": "M2"},
    {"id": "B5", "title": "slack", "priority": "P0", "status": "todo", "owner": "Lauren", "depends_on": ["B3"], "milestone": "M3"},
    {"id": "B6", "title": "docs", "priority": "P3", "status": "todo", "owner": "AI PM", "depends_on": []},
    {"id": "B7", "title": "lint", "priority": "P2", "status": "doing", "owner": "AI PM", "depends_on": []},
]
RISKS = [
    {"id": "R1", "risk": "a", "probability": 2, "impact": 3, "owner": "x", "mitigation": "m", "status": "open"},
    {"id": "R2", "risk": "b", "probability": 4, "impact": 4, "owner": "x", "mitigation": "m", "status": "open"},
    {"id": "R3", "risk": "c", "probability": 5, "impact": 5, "owner": "x", "mitigation": "m", "status": "closed"},
]


def test_health_rules_red_amber_green():
    low_risks = [dict(RISKS[0])]
    no_p0 = [b for b in BACKLOG if b["id"] != "B5"]
    assert ai_pm.compute_health(MILESTONES, no_p0, low_risks, TODAY)["rag"] == "GREEN"
    # One open P0 -> AMBER; a 16-point open risk alone -> AMBER; closed 25-point risk is ignored.
    assert ai_pm.compute_health(MILESTONES, BACKLOG, low_risks, TODAY)["rag"] == "AMBER"
    amber = ai_pm.compute_health(MILESTONES, no_p0, RISKS, TODAY)
    assert amber["rag"] == "AMBER" and amber["high_risks"] == ["R2"]
    assert "R2=16" in amber["reason"]
    # Overdue milestone -> RED.
    late = MILESTONES + [{"id": "M0", "name": "late", "due": "2026-10-15", "status": "in_progress"}]
    red = ai_pm.compute_health(late, no_p0, low_risks, TODAY)
    assert red["rag"] == "RED" and red["overdue_milestones"] == ["M0"]
    # Two open P0 items -> RED.
    two_p0 = BACKLOG + [{"id": "B9", "title": "x", "priority": "P0", "status": "todo", "owner": "AI PM", "depends_on": []}]
    assert ai_pm.compute_health(MILESTONES, two_p0, low_risks, TODAY)["rag"] == "RED"


def test_next_actions_skip_blocked_and_rank_by_priority_then_deps():
    actions = ai_pm.next_actions(BACKLOG)
    # B5 (P0) is blocked by B3; B7 is "doing", so not a todo.
    assert [a["id"] for a in actions] == ["B3", "B4", "B6"]
    assert actions[0]["deps_satisfied"] == 2
    assert ai_pm.blocked_items(BACKLOG) == ["B5"]


def test_risk_register_sorted_with_heat_and_milestone_progress():
    reg = ai_pm.risk_register(RISKS)
    assert [r["id"] for r in reg] == ["R3", "R2", "R1"]
    assert [r["heat"] for r in reg] == ["Critical", "High", "Low"]
    assert ai_pm.heat_label(8) == "Medium"
    prog = {m["id"]: m for m in ai_pm.milestone_progress(MILESTONES, BACKLOG, TODAY)}
    assert prog["M1"]["pct_done"] == 100
    assert prog["M2"]["pct_done"] == 33  # 1 of 3 linked items done
    assert prog["M3"]["pct_done"] == 0


def _write_project(root, summary=None):
    import yaml

    (root / "project").mkdir()
    (root / "project" / "ROADMAP.yml").write_text(
        yaml.safe_dump({"project": "Test Project", "milestones": MILESTONES, "backlog": BACKLOG})
    )
    (root / "project" / "RISKS.yml").write_text(yaml.safe_dump({"risks": RISKS}))
    if summary is not None:
        (root / "reports").mkdir()
        (root / "reports" / "summary.json").write_text(json.dumps(summary))


SUMMARY = {
    "data": {"days": 365, "providers": 3, "series": 54, "total_spend_usd": 1.5e7, "avg_daily_spend_usd": 41000,
             "injected_anomalies": 12},
    "ensemble": {"events_flagged": 14, "precision": 0.93, "recall": 0.92, "f1": 0.92, "missed_anomalies": ["A09"],
                 "false_positive_events": 1, "false_positives_on_planned_events": 1},
    "attribution": {"top1_driver_correct": 11, "detected_anomalies": 11, "top1_accuracy": 1.0},
    "savings": {"capture_ratio": 0.72, "estimated_excess_usd_matched_events": 91000,
                "injected_excess_usd_detected_anomalies": 127000},
    "largest_anomaly": {"event_id": "E03", "start": "2025-03-24", "end": "2025-03-31", "excess_usd": 14627,
                        "driver": {"provider": "Azure", "service": "Virtual Machines", "account": "az-prod-apps"}},
}


def test_run_pm_is_deterministic_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _write_project(tmp_path, SUMMARY)

    def no_network(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("network used")

    r1 = ai_pm.run_pm(tmp_path, today="2026-11-02", opener=no_network)
    md1 = (tmp_path / "project" / "STATUS.md").read_text()
    r2 = ai_pm.run_pm(tmp_path, today="2026-11-02", opener=no_network)
    assert md1 == (tmp_path / "project" / "STATUS.md").read_text()
    assert r1 == r2
    assert r1["executive_summary_source"] == "deterministic template"
    assert "2026-11-02" in md1 and "## Next 3 actions" in md1 and "## Risk register" in md1
    assert 3 <= len(r1["data_insights"]) <= 5
    assert "precision 0.93" in r1["data_insights"][1]
    status = json.loads((tmp_path / "project" / "status.json").read_text())
    assert status["health"]["rag"] == "AMBER"
    assert status["change_log"].startswith("2026-11-02: status run. Health AMBER")


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_llm_summary_used_when_valid_and_falls_back_on_error(tmp_path, monkeypatch):
    _write_project(tmp_path, SUMMARY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AI_PM_MODEL", "claude-test")
    seen = {}
    text = " ".join(["word"] * 100)

    def fake_opener(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data)
        return _FakeResponse(json.dumps({"content": [{"type": "text", "text": text}]}).encode())

    result = ai_pm.run_pm(tmp_path, today="2026-11-02", opener=fake_opener)
    assert result["executive_summary"] == text
    assert result["executive_summary_source"] == "LLM (claude-test)"
    assert seen["url"] == ai_pm.API_URL
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-test"

    def broken_opener(req, timeout):
        raise OSError("offline")

    fallback = ai_pm.run_pm(tmp_path, today="2026-11-02", opener=broken_opener)
    assert fallback["executive_summary_source"] == "deterministic template"
    assert fallback["executive_summary"].startswith("Test Project is AMBER")


def test_missing_summary_still_produces_status(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _write_project(tmp_path, summary=None)
    result = ai_pm.run_pm(tmp_path, today=date(2026, 11, 2))
    assert "No pipeline results found" in result["data_insights"][0]
    assert (tmp_path / "project" / "STATUS.md").exists()


@pytest.mark.parametrize("score,label", [(25, "Critical"), (15, "High"), (12, "Medium"), (4, "Low")])
def test_heat_labels(score, label):
    assert ai_pm.heat_label(score) == label
