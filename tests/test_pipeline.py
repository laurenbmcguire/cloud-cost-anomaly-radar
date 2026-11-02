import json

from costradar.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path):
    summary = run_pipeline(tmp_path, seed=42, robustness=False, make_charts=True)
    for rel in [
        "data/billing_daily.csv",
        "data/anomalies_truth.csv",
        "examples/sample_billing_first_month.csv",
        "reports/finops_report.md",
        "reports/summary.json",
        "docs/img/hero_spend_anomalies.png",
        "docs/img/spend_by_provider.png",
        "docs/img/attribution_top_anomalies.png",
        "docs/img/detector_comparison.png",
    ]:
        assert (tmp_path / rel).exists(), rel

    on_disk = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert on_disk["ensemble"] == summary["ensemble"]
    assert summary["data"]["series"] == 54 and summary["data"]["days"] == 365

    ens = summary["ensemble"]
    assert ens["recall"] >= 0.8 and ens["precision"] >= 0.8 and ens["f1"] >= 0.85
    # The 2-of-3 vote should beat the naive rolling baseline it includes.
    assert ens["f1"] >= summary["detectors"]["rolling_mad"]["f1"]
    assert summary["attribution"]["top1_accuracy"] >= 0.8
    for a in summary["anomalies"]:
        assert a["excess_usd"] >= 0
        assert a["driver"]["delta_usd"] > 0

    report = (tmp_path / "reports" / "finops_report.md").read_text()
    assert "synthetic" in report.lower()
    assert "## Anomaly register" in report

    # Second run reuses the generated data and is deterministic.
    again = run_pipeline(tmp_path, seed=42, robustness=False, make_charts=False)
    assert again["ensemble"] == summary["ensemble"]
    assert again["savings"] == summary["savings"]
