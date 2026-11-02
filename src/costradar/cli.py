"""Command line interface: ``python -m costradar run`` and ``python -m costradar pm``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .generator import DEFAULT_SEED


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    t0 = time.perf_counter()
    summary = run_pipeline(
        root=Path(args.root),
        seed=args.seed,
        regenerate=args.regenerate,
        robustness=not args.no_robustness,
    )
    ens = summary["ensemble"]
    sav = summary["savings"]
    att = summary["attribution"]
    print(f"Analysed {summary['data']['days']} days, {summary['data']['series']} series, "
          f"${summary['data']['total_spend_usd']:,.0f} total spend.")
    print(f"Ensemble: {ens['events_flagged']} events | precision {ens['precision']:.2f} "
          f"recall {ens['recall']:.2f} F1 {ens['f1']:.2f}")
    print(f"Attribution top-1 accuracy: {att['top1_accuracy']:.0%} | "
          f"estimated excess ${sav['estimated_excess_usd_all_events']:,.0f}")
    print(f"Wrote reports/finops_report.md, reports/summary.json and docs/img/*.png "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0


def _cmd_pm(args: argparse.Namespace) -> int:
    from .ai_pm import run_pm

    result = run_pm(root=Path(args.root), today=args.today, use_llm=not args.no_llm)
    print(f"AI PM health: {result['health']['rag']} - {result['health']['reason']}")
    print(f"Executive summary source: {result['executive_summary_source']}")
    print("Wrote project/STATUS.md and project/status.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="costradar", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate data if missing, detect, attribute, chart and report")
    run.add_argument("--root", default=".", help="project root (default: current directory)")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--regenerate", action="store_true", help="regenerate synthetic data even if present")
    run.add_argument("--no-robustness", action="store_true", help="skip the multi-seed robustness check")
    run.set_defaults(func=_cmd_run)

    pm = sub.add_parser("pm", help="run the AI project manager and write project/STATUS.md")
    pm.add_argument("--root", default=".")
    pm.add_argument("--today", default=None, help="override the status date (YYYY-MM-DD)")
    pm.add_argument("--no-llm", action="store_true", help="never call the LLM, even if a key is set")
    pm.set_defaults(func=_cmd_pm)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
