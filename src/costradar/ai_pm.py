"""AI Project Manager: manages this repo as a project.

Inputs
    project/ROADMAP.yml   milestones and backlog
    project/RISKS.yml     risk register
    reports/summary.json  latest pipeline results

Deterministic engine (always runs; tests target it)
    * overall health RAG with a one-line reason (explicit rules)
    * milestone table with % done
    * risk register sorted by score (probability x impact) with a heat label
    * "Next 3 actions": unblocked todo items by priority, then dependencies satisfied
    * "What the data says": bullets built from summary.json
    * a short decision / change log entry

Optional LLM narrative
    If ANTHROPIC_API_KEY is set, the Anthropic Messages API is asked for a
    90-120 word executive summary grounded only in the computed facts. Any
    error (or no key) falls back to a deterministic template. The run never
    fails because of the LLM.

Outputs
    project/STATUS.md and project/status.json
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable

import yaml

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-5"
PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
STATUS_PCT_FALLBACK = {"done": 100, "in_progress": 50, "planned": 0}
HIGH_RISK_SCORE = 15


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_summary(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# --------------------------------------------------------------------------- #
# Deterministic engine
# --------------------------------------------------------------------------- #
def milestone_progress(milestones: list[dict], backlog: list[dict], today: date) -> list[dict]:
    """Percent done per milestone from its linked backlog items.

    Milestones without linked items fall back to their status
    (done 100, in_progress 50, planned 0).
    """
    rows = []
    for m in milestones:
        items = [b for b in backlog if b.get("milestone") == m["id"]]
        if items:
            pct = round(100 * sum(1 for b in items if b["status"] == "done") / len(items))
        else:
            pct = STATUS_PCT_FALLBACK.get(m["status"], 0)
        due = _as_date(m["due"])
        rows.append(
            {
                "id": m["id"],
                "name": m["name"],
                "due": due.isoformat(),
                "status": m["status"],
                "pct_done": pct,
                "items": len(items),
                "overdue": m["status"] != "done" and due < today,
                "done_criteria": m.get("done_criteria", ""),
            }
        )
    return rows


def heat_label(score: int) -> str:
    if score >= 20:
        return "Critical"
    if score >= HIGH_RISK_SCORE:
        return "High"
    if score >= 8:
        return "Medium"
    return "Low"


def risk_register(risks: list[dict]) -> list[dict]:
    """Risks with score = probability x impact, sorted by score desc then id."""
    rows = []
    for r in risks:
        score = int(r["probability"]) * int(r["impact"])
        rows.append({**r, "score": score, "heat": heat_label(score)})
    return sorted(rows, key=lambda r: (-r["score"], r["id"]))


def compute_health(milestones: list[dict], backlog: list[dict], risks: list[dict], today: date) -> dict:
    """Explicit RAG rules.

    RED   : any milestone overdue, or more than one P0 item open
    AMBER : exactly one P0 item open, or any open risk scoring >= 15
    GREEN : otherwise
    """
    progress = milestone_progress(milestones, backlog, today)
    overdue = [m["id"] for m in progress if m["overdue"]]
    open_p0 = [b["id"] for b in backlog if b["priority"] == "P0" and b["status"] != "done"]
    high_risks = [r for r in risk_register(risks) if r.get("status", "open") != "closed" and r["score"] >= HIGH_RISK_SCORE]

    facts = (
        f"{len(overdue)} overdue milestone(s)"
        + (f" ({', '.join(overdue)})" if overdue else "")
        + f", {len(open_p0)} open P0 item(s)"
        + (f" ({', '.join(open_p0)})" if open_p0 else "")
        + f", {len(high_risks)} open risk(s) scored >= {HIGH_RISK_SCORE}"
        + (f" ({', '.join(f'{r['id']}={r['score']}' for r in high_risks)})" if high_risks else "")
    )
    if overdue or len(open_p0) > 1:
        rag = "RED"
    elif open_p0 or high_risks:
        rag = "AMBER"
    else:
        rag = "GREEN"
    return {
        "rag": rag,
        "reason": f"{rag}: {facts}.",
        "overdue_milestones": overdue,
        "open_p0": open_p0,
        "high_risks": [r["id"] for r in high_risks],
    }


def next_actions(backlog: list[dict], n: int = 3) -> list[dict]:
    """Unblocked todo items ranked by priority, then dependencies satisfied, then id."""
    done = {b["id"] for b in backlog if b["status"] == "done"}
    candidates = []
    for b in backlog:
        deps = list(b.get("depends_on") or [])
        if b["status"] != "todo" or any(d not in done for d in deps):
            continue
        candidates.append({**b, "depends_on": deps, "deps_satisfied": len(deps)})
    candidates.sort(key=lambda b: (PRIORITY_RANK.get(b["priority"], 9), -b["deps_satisfied"], b["id"]))
    return candidates[:n]


def blocked_items(backlog: list[dict]) -> list[str]:
    done = {b["id"] for b in backlog if b["status"] == "done"}
    return [
        b["id"] for b in backlog
        if b["status"] == "todo" and any(d not in done for d in (b.get("depends_on") or []))
    ]


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def data_insights(summary: dict | None) -> list[str]:
    """3-5 bullets grounded in reports/summary.json."""
    if not summary:
        return ["No pipeline results found. Run `python -m costradar run` to produce reports/summary.json."]
    d, ens, att, sav = summary["data"], summary["ensemble"], summary["attribution"], summary["savings"]
    bullets = [
        f"Analysed {d['days']} days of synthetic spend across {d['providers']} providers and {d['series']} "
        f"series ({_usd(d['total_spend_usd'])} total, {_usd(d['avg_daily_spend_usd'])} per day).",
        f"The ensemble flagged {ens['events_flagged']} events: precision {ens['precision']:.2f}, recall "
        f"{ens['recall']:.2f}, F1 {ens['f1']:.2f} against {d['injected_anomalies']} injected anomalies"
        + (f"; missed {', '.join(ens['missed_anomalies'])}." if ens["missed_anomalies"] else "; none missed."),
    ]
    largest = summary.get("largest_anomaly")
    if largest:
        dr = largest["driver"]
        svc = dr["service"] if dr["service"].startswith(dr["provider"]) else f"{dr['provider']} {dr['service']}"
        bullets.append(
            f"Largest event {largest['event_id']} ({largest['start']} to {largest['end']}) carried "
            f"{_usd(largest['excess_usd'])} excess, driven by {svc} in {dr['account']}."
        )
    bullets.append(
        f"Root-cause attribution ranked the injected series first for {att['top1_driver_correct']} of "
        f"{att['detected_anomalies']} detected anomalies ({att['top1_accuracy']:.0%})."
    )
    bullets.append(
        f"Savings estimate covers {sav['capture_ratio']:.0%} of the injected excess on detected anomalies "
        f"({_usd(sav['estimated_excess_usd_matched_events'])} of {_usd(sav['injected_excess_usd_detected_anomalies'])}); "
        f"{ens['false_positive_events']} false-positive event(s), "
        f"{ens['false_positives_on_planned_events']} of them on planned events."
    )
    return bullets[:5]


def change_log_entry(today: date, health: dict, actions: list[dict], summary: dict | None) -> str:
    focus = actions[0]["id"] + " " + actions[0]["title"] if actions else "no unblocked items"
    data = (
        f"ensemble F1 {summary['ensemble']['f1']:.2f} on {summary['ensemble']['events_flagged']} events"
        if summary else "no pipeline results"
    )
    return f"{today.isoformat()}: status run. Health {health['rag']}; {data}. Next focus: {focus}."


def build_facts(roadmap: dict, risks_doc: dict, summary: dict | None, today: date) -> dict:
    milestones = roadmap.get("milestones", [])
    backlog = roadmap.get("backlog", [])
    risks = risks_doc.get("risks", [])
    health = compute_health(milestones, backlog, risks, today)
    actions = next_actions(backlog)
    progress = milestone_progress(milestones, backlog, today)
    register = risk_register(risks)
    return {
        "project": roadmap.get("project", "Cloud Cost Anomaly Radar"),
        "date": today.isoformat(),
        "health": health,
        "milestones": progress,
        "milestones_done": sum(1 for m in progress if m["status"] == "done"),
        "milestones_total": len(progress),
        "next_actions": [
            {k: a[k] for k in ("id", "title", "priority", "owner", "depends_on", "milestone") if k in a}
            for a in actions
        ],
        "blocked_items": blocked_items(backlog),
        "risks": register,
        "data_insights": data_insights(summary),
        "change_log": change_log_entry(today, health, actions, summary),
    }


# --------------------------------------------------------------------------- #
# Executive summary: deterministic template or LLM
# --------------------------------------------------------------------------- #
def deterministic_summary(facts: dict) -> str:
    h = facts["health"]
    top_risk = facts["risks"][0] if facts["risks"] else None
    actions = facts["next_actions"]
    drivers = []
    if h["overdue_milestones"]:
        drivers.append(f"overdue milestones {', '.join(h['overdue_milestones'])}")
    if h["open_p0"]:
        drivers.append(f"open P0 items {', '.join(h['open_p0'])}")
    if h["high_risks"]:
        drivers.append(f"open risks scoring {HIGH_RISK_SCORE}+: {', '.join(h['high_risks'])}")
    why = "; ".join(drivers) if drivers else "no overdue milestones, open P0 items or high risks"
    remaining = [m["name"] for m in facts["milestones"] if m["status"] != "done"]
    parts = [
        f"{facts['project']} is {h['rag']} as of {facts['date']} ({why}).",
        f"{facts['milestones_done']} of {facts['milestones_total']} milestones are done"
        + (f"; still open: {', '.join(remaining)}." if remaining else "."),
    ]
    if facts["data_insights"]:
        parts.append(facts["data_insights"][1] if len(facts["data_insights"]) > 1 else facts["data_insights"][0])
    if top_risk:
        parts.append(f"Top risk is {top_risk['id']} ({top_risk['risk']}), score {top_risk['score']} ({top_risk['heat']}).")
    if actions:
        parts.append("Next actions: " + "; ".join(f"{a['id']} {a['title']}" for a in actions) + ".")
    return " ".join(parts)


def llm_summary(
    facts: dict,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    opener: Callable = urllib.request.urlopen,
) -> str | None:
    """Ask the Anthropic Messages API for an executive summary. Returns None on any failure."""
    prompt = (
        "You are the AI project manager for a portfolio software repo. Write a 90-120 word executive "
        "summary of project status for a hiring manager. Use ONLY the facts in the JSON below; do not "
        "invent numbers, dates, users or results. Plain prose, one paragraph, no headings or bullet "
        "points.\n\nFACTS:\n" + json.dumps(facts, indent=2, default=str)
    )
    body = json.dumps(
        {"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with opener(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        ).strip()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    words = len(text.split())
    if not text or words < 40 or words > 200:
        return None
    return text


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v) -> str:
        return str(v).replace("|", "/").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_status_md(facts: dict, exec_summary: str, source: str) -> str:
    h = facts["health"]
    lines = [
        f"# Project status: {facts['project']}",
        "",
        f"_Status date: {facts['date']}. Generated by the AI PM (`python -m costradar pm`). "
        f"Executive summary source: {source}._",
        "",
        f"**Health: {h['rag']}**: {h['reason'].split(': ', 1)[1]}",
        "",
        "## Executive summary",
        "",
        exec_summary,
        "",
        "## Milestones",
        "",
        _md_table(
            ["ID", "Milestone", "Due", "Status", "% done", "Overdue"],
            [[m["id"], m["name"], m["due"], m["status"], f"{m['pct_done']}%", "yes" if m["overdue"] else "no"]
             for m in facts["milestones"]],
        ),
        "",
        "## Next 3 actions",
        "",
        _md_table(
            ["#", "ID", "Action", "Priority", "Owner", "Depends on"],
            [[i + 1, a["id"], a["title"], a["priority"], a["owner"], ", ".join(a["depends_on"]) or "none"]
             for i, a in enumerate(facts["next_actions"])],
        ) if facts["next_actions"] else "_No unblocked todo items._",
        "",
        "## Risk register",
        "",
        _md_table(
            ["ID", "Risk", "P", "I", "Score", "Heat", "Owner", "Mitigation", "Status"],
            [[r["id"], r["risk"], r["probability"], r["impact"], r["score"], r["heat"], r["owner"],
              r["mitigation"], r.get("status", "open")] for r in facts["risks"]],
        ),
        "",
        "## What the data says",
        "",
        *[f"- {b}" for b in facts["data_insights"]],
        "",
        "## Decision / change log",
        "",
        f"- {facts['change_log']}",
        "",
    ]
    return "\n".join(lines)


def run_pm(
    root: Path | str = ".",
    today: date | str | None = None,
    use_llm: bool = True,
    opener: Callable = urllib.request.urlopen,
) -> dict:
    root = Path(root)
    if today is None:
        today = os.environ.get("AI_PM_TODAY") or date.today()
    today = _as_date(today)

    roadmap = load_yaml(root / "project" / "ROADMAP.yml")
    risks_doc = load_yaml(root / "project" / "RISKS.yml")
    summary = load_summary(root / "reports" / "summary.json")
    facts = build_facts(roadmap, risks_doc, summary, today)

    exec_summary, source = None, "deterministic template"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if use_llm and api_key:
        model = os.environ.get("AI_PM_MODEL", DEFAULT_MODEL)
        exec_summary = llm_summary(facts, api_key, model=model, opener=opener)
        if exec_summary:
            source = f"LLM ({model})"
    if not exec_summary:
        exec_summary = deterministic_summary(facts)

    result = {**facts, "executive_summary": exec_summary, "executive_summary_source": source}
    out_dir = root / "project"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.md").write_text(render_status_md(facts, exec_summary, source), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result
