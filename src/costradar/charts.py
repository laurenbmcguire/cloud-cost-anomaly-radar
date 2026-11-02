"""Matplotlib charts written to docs/img by the pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from . import style
from .detectors import DETECTOR_LABELS

TITLE_PAD = 26


def _usd_k(x, _pos=None) -> str:
    return f"${x / 1000:,.0f}k"


def _short_usd(v: float) -> str:
    return f"${v / 1000:,.1f}k" if abs(v) >= 1000 else f"${v:,.0f}"


def hero_chart(scores: pd.DataFrame, anomalies: list[dict], path: Path, band_k: float) -> Path:
    """Total daily spend with the seasonal baseline band and flagged anomalies."""
    fig, ax = style.new_figure()
    idx = scores.index
    upper = scores["baseline"] + band_k * scores["band_sigma_usd"]
    lower = scores["baseline"] - band_k * scores["band_sigma_usd"]
    ax.fill_between(idx, lower, upper, color=style.SAND, alpha=0.55, linewidth=0,
                    label=f"Expected range (baseline +/- {band_k:g} sigma)")
    ax.plot(idx, scores["baseline"], color=style.GOLD, linewidth=1.4, linestyle="--", label="Seasonal baseline")
    ax.plot(idx, scores["total"], color=style.INK, linewidth=1.1, label="Actual daily spend")
    flagged = scores[scores["ensemble_flag"]]
    ax.scatter(flagged.index, flagged["total"], s=34, color=style.CRIMSON, zorder=5,
               edgecolor=style.BACKGROUND, linewidth=0.8, label="Flagged anomaly day")

    ymin = float(min(scores["total"].min(), lower.min())) * 0.97
    ymax = float(max(scores["total"].max(), upper.max()))
    label_levels = (ymax * 1.04, ymax * 1.13)
    top = sorted(anomalies, key=lambda a: a["excess_usd"], reverse=True)[:4]
    for i, a in enumerate(sorted(top, key=lambda a: a["peak_date"])):
        peak = pd.Timestamp(a["peak_date"])
        y = scores.loc[peak, "total"]
        d = a["driver"]
        label = f"{a['event_id']}  {d['service']}\n{d['account']}  +{_short_usd(a['excess_usd'])}"
        ax.annotate(label, xy=(peak, y), xytext=(peak, label_levels[i % 2]), textcoords="data",
                    ha="center", va="bottom", fontsize=8.5, color=style.DEEP_CRIMSON,
                    arrowprops={"arrowstyle": "-", "color": style.DEEP_CRIMSON, "linewidth": 0.7,
                                "shrinkA": 1, "shrinkB": 4})

    ax.set_title("Total daily cloud spend vs. expected baseline", pad=TITLE_PAD)
    style.subtitle(ax, f"AWS + Azure + GCP, 54 series, synthetic {idx[0].year}. {len(anomalies)} events flagged; "
                       "labels: 4 largest by excess, with top driver.")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_k))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylim(ymin, ymax * 1.24)
    ax.set_xlim(idx[0], idx[-1])
    ax.set_ylabel("Daily spend (USD)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9)
    fig.tight_layout()
    return style.save(fig, path)


def provider_stacked_area(by_provider: pd.DataFrame, path: Path) -> Path:
    fig, ax = style.new_figure()
    cols = [c for c in ("AWS", "Azure", "GCP") if c in by_provider.columns]
    weekly = by_provider[cols].rolling(7, min_periods=1).mean()
    ax.stackplot(weekly.index, *[weekly[c] for c in cols], labels=cols,
                 colors=[style.PROVIDER_COLORS[c] for c in cols], alpha=0.92,
                 edgecolor=style.BACKGROUND, linewidth=0.6)
    total = by_provider[cols].sum()
    share = total / total.sum()
    ax.set_title("Daily spend by cloud provider", pad=TITLE_PAD)
    style.subtitle(ax, "7-day rolling mean, stacked. Annual share: "
                   + ", ".join(f"{c} {share[c]:.0%}" for c in cols) + ".")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_k))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(weekly.index[0], weekly.index[-1])
    ax.set_ylabel("Daily spend (USD)")
    ax.set_ylim(0, float(weekly.sum(axis=1).max()) * 1.22)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=9.5)
    fig.tight_layout()
    return style.save(fig, path)


def attribution_chart(anomalies: list[dict], path: Path, n: int = 4) -> Path:
    """Small multiples: top contributors (delta vs baseline) for the largest anomalies."""
    top = sorted(anomalies, key=lambda a: a["excess_usd"], reverse=True)[:n]
    ncols = 2
    nrows = int(np.ceil(len(top) / ncols)) or 1
    fig, axes = style.new_figure(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()
    for ax, a in zip(axes, top):
        contrib = a["attribution"]["contributors"][:4][::-1]
        labels = [f"{c['service']} | {c['account']}" for c in contrib]
        values = [c["delta_usd"] for c in contrib]
        colors = [style.CRIMSON if i == len(contrib) - 1 else style.SAND for i in range(len(contrib))]
        ax.barh(labels, values, color=colors, height=0.62)
        for y, v in enumerate(values):
            ax.text(v, y, f"  {_short_usd(v)}", va="center", fontsize=8.5, color=style.INK)
        ax.set_title(f"{a['event_id']}  {a['start']} to {a['end']}", fontsize=11, pad=6)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: _short_usd(x)))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", visible=False)
        xmax = max(values) if values else 1.0
        ax.set_xlim(min(0.0, min(values) if values else 0.0), xmax * 1.35)
    for ax in axes[len(top):]:
        ax.set_visible(False)
    fig.suptitle("Root-cause attribution for the largest anomalies", x=0.012, ha="left",
                 fontsize=style.TITLE_SIZE, fontweight="semibold", color=style.INK)
    fig.text(0.012, 0.915, "Excess vs. each series' own seasonal baseline over the event days; top driver in crimson.",
             color=style.MUTED, fontsize=10, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0)
    return style.save(fig, path)


def detector_comparison_chart(metrics: dict, path: Path) -> Path:
    names = ["rolling_mad", "seasonal_residual", "isolation_forest", "ensemble"]
    fig, ax = style.new_figure()
    x = np.arange(len(names))
    width = 0.26
    series = [("precision", style.DEEP_CRIMSON), ("recall", style.GOLD), ("f1", style.CRIMSON)]
    for i, (key, color) in enumerate(series):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=color, label=key.upper() if key == "f1" else key.title())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=style.INK)
    ax.set_xticks(x, [DETECTOR_LABELS[n] for n in names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Event-level score")
    ax.set_title("Detector comparison against injected anomalies", pad=TITLE_PAD)
    style.subtitle(ax, "Event-level precision, recall and F1 with a +/-2 day tolerance on 12 labelled anomalies.")
    ax.legend(loc="upper left", ncol=3, fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return style.save(fig, path)
