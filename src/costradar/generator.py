"""Seeded synthetic multi-cloud billing generator with labelled anomalies.

The generator produces one year of daily cost for 3 providers x 3 accounts x
6 services (54 series). Each series combines:

* a base level that depends on provider size, account type and service category
* compound annual growth (AI/ML grows fastest)
* weekly seasonality (non-production accounts idle at weekends)
* month-end batch effects (analytics and data workloads)
* AR(1) multiplicative log-normal noise

Twelve anomalies are then injected (spikes, level shifts and slow leaks) and
their exact dollar impact is recorded as ground truth. Everything is driven by
one ``numpy.random.Generator`` seeded once, so the output is reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2025-01-01"
N_DAYS = 365
DEFAULT_SEED = 42

CATEGORIES = ("compute", "storage", "database", "networking", "ai_ml", "analytics")

# Daily USD for a production account on the reference (largest) provider.
CATEGORY_BASE = {
    "compute": 4200.0,
    "storage": 1300.0,
    "database": 1800.0,
    "networking": 700.0,
    "ai_ml": 1100.0,
    "analytics": 600.0,
}

PROVIDERS: dict[str, dict] = {
    "AWS": {
        "scale": 1.00,
        "services": {
            "compute": "EC2",
            "storage": "S3",
            "database": "RDS",
            "networking": "Data Transfer",
            "ai_ml": "SageMaker Inference",
            "analytics": "Athena",
        },
        "accounts": {"aws-prod-core": "prod", "aws-data-platform": "data", "aws-dev-sandbox": "dev"},
    },
    "Azure": {
        "scale": 0.65,
        "services": {
            "compute": "Virtual Machines",
            "storage": "Blob Storage",
            "database": "Azure SQL Database",
            "networking": "Bandwidth",
            "ai_ml": "Azure OpenAI",
            "analytics": "Synapse Analytics",
        },
        "accounts": {"az-prod-apps": "prod", "az-analytics": "data", "az-dev-test": "dev"},
    },
    "GCP": {
        "scale": 0.45,
        "services": {
            "compute": "Compute Engine",
            "storage": "Cloud Storage",
            "database": "Cloud SQL",
            "networking": "Network Egress",
            "ai_ml": "Vertex AI Prediction",
            "analytics": "BigQuery",
        },
        "accounts": {"gcp-prod-web": "prod", "gcp-ml-research": "ml", "gcp-sandbox": "dev"},
    },
}

ACCOUNT_KINDS = {
    "prod": {"mult": 1.00, "tilt": {}},
    "data": {"mult": 0.55, "tilt": {"analytics": 2.0, "storage": 1.4}},
    "ml": {"mult": 0.55, "tilt": {"ai_ml": 2.5, "compute": 1.2}},
    "dev": {"mult": 0.25, "tilt": {}},
}

# Weekend multiplier as (production, non-production).
WEEKEND_FACTOR = {
    "compute": (0.93, 0.70),
    "storage": (1.00, 1.00),
    "database": (0.95, 0.75),
    "networking": (0.86, 0.80),
    "ai_ml": (0.85, 0.65),
    "analytics": (0.80, 0.60),
}
# Multiplier on the last two calendar days of each month (batch close jobs).
MONTH_END_FACTOR = {
    "compute": 1.06,
    "storage": 1.00,
    "database": 1.05,
    "networking": 1.03,
    "ai_ml": 1.00,
    "analytics": 1.55,
}
NOISE_SIGMA = {
    "compute": 0.035,
    "storage": 0.015,
    "database": 0.030,
    "networking": 0.060,
    "ai_ml": 0.070,
    "analytics": 0.080,
}
GROWTH_RANGE = {
    "compute": (0.05, 0.30),
    "storage": (0.15, 0.35),
    "database": (0.05, 0.25),
    "networking": (0.05, 0.30),
    "ai_ml": (0.45, 0.85),
    "analytics": (0.05, 0.30),
}
AR_PHI = 0.5
MONTH_END_DAYS = 2


@dataclass(frozen=True)
class AnomalySpec:
    """One injected anomaly. ``magnitude`` is the fractional uplift at peak."""

    anomaly_id: str
    kind: str  # spike | level_shift | slow_leak
    provider: str
    account: str
    category: str
    start: str
    days: int
    magnitude: float
    description: str


ANOMALIES: tuple[AnomalySpec, ...] = (
    AnomalySpec("A01", "spike", "AWS", "aws-dev-sandbox", "compute", "2025-02-11", 2, 4.0,
                "GPU instance fleet left running in the sandbox account"),
    AnomalySpec("A02", "spike", "GCP", "gcp-prod-web", "networking", "2025-03-04", 1, 9.0,
                "CDN cache-miss storm drives internet egress"),
    AnomalySpec("A03", "level_shift", "Azure", "az-prod-apps", "compute", "2025-03-24", 14, 0.45,
                "VM scale-set minimum capacity raised and not reverted"),
    AnomalySpec("A04", "spike", "AWS", "aws-data-platform", "analytics", "2025-04-17", 3, 5.0,
                "Unpartitioned Athena query running in a retry loop"),
    AnomalySpec("A05", "slow_leak", "AWS", "aws-prod-core", "storage", "2025-05-06", 20, 1.5,
                "S3 lifecycle rule disabled; object versions accumulate"),
    AnomalySpec("A06", "spike", "GCP", "gcp-ml-research", "ai_ml", "2025-06-10", 2, 4.0,
                "Vertex AI endpoint autoscaled to max replicas"),
    AnomalySpec("A07", "level_shift", "AWS", "aws-prod-core", "database", "2025-07-08", 21, 0.60,
                "RDS moved to provisioned IOPS without rightsizing"),
    AnomalySpec("A08", "spike", "Azure", "az-analytics", "analytics", "2025-08-05", 1, 7.0,
                "Synapse dedicated pool resumed for an ad-hoc backfill"),
    AnomalySpec("A09", "slow_leak", "GCP", "gcp-prod-web", "compute", "2025-08-25", 18, 0.90,
                "GKE node-pool autoscaler fails to scale down"),
    AnomalySpec("A10", "spike", "Azure", "az-prod-apps", "ai_ml", "2025-09-23", 2, 3.0,
                "Azure OpenAI prompt retry storm after a client bug"),
    AnomalySpec("A11", "level_shift", "Azure", "az-dev-test", "compute", "2025-10-21", 10, 1.50,
                "Dev/test VMs excluded from the nightly shutdown schedule"),
    AnomalySpec("A12", "spike", "AWS", "aws-prod-core", "networking", "2025-11-18", 3, 3.5,
                "NAT gateway cross-AZ traffic surge after a deploy"),
)


@dataclass(frozen=True)
class PlannedEvent:
    """An expected, approved cost increase. Not an anomaly in the ground truth."""

    event_id: str
    start: str
    days: int
    categories: tuple[str, ...]
    account_kinds: tuple[str, ...]
    uplift: float
    description: str


PLANNED_EVENTS: tuple[PlannedEvent, ...] = (
    PlannedEvent("P01", "2025-06-24", 1, ("compute", "database"), ("prod",), 0.25,
                 "Quarterly disaster-recovery failover test"),
    PlannedEvent("P02", "2025-11-26", 7, ("compute", "networking"), ("prod",), 0.15,
                 "Holiday-season traffic scale-up (approved capacity plan)"),
)


def series_catalog() -> pd.DataFrame:
    """Return one row per (provider, account, service) series in a fixed order."""
    rows = []
    for provider, spec in PROVIDERS.items():
        for account, kind in spec["accounts"].items():
            for category in CATEGORIES:
                rows.append(
                    {
                        "provider": provider,
                        "account": account,
                        "account_kind": kind,
                        "category": category,
                        "service": spec["services"][category],
                    }
                )
    return pd.DataFrame(rows)


def month_end_mask(dates: pd.DatetimeIndex, n_days: int = MONTH_END_DAYS) -> np.ndarray:
    """True on the last ``n_days`` calendar days of each month."""
    dates = pd.DatetimeIndex(dates)
    days_left = dates.days_in_month - dates.day
    return np.asarray(days_left < n_days)


def anomaly_multiplier(spec: AnomalySpec, dates: pd.DatetimeIndex) -> np.ndarray:
    """Fractional uplift applied to the target series on each date."""
    start = pd.Timestamp(spec.start)
    offset = np.asarray((dates - start).days)
    active = (offset >= 0) & (offset < spec.days)
    mult = np.zeros(len(dates))
    if spec.kind in ("spike", "level_shift"):
        mult[active] = spec.magnitude
    elif spec.kind == "slow_leak":
        mult[active] = spec.magnitude * (offset[active] + 1) / spec.days
    else:
        raise ValueError(f"unknown anomaly kind: {spec.kind}")
    return mult


def generate_billing(
    seed: int = DEFAULT_SEED,
    n_days: int = N_DAYS,
    start: str = START_DATE,
    inject: bool = True,
    anomalies: tuple[AnomalySpec, ...] = ANOMALIES,
    planned: tuple[PlannedEvent, ...] = PLANNED_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate long-format daily billing and the anomaly ground truth.

    Returns ``(billing, truth)``. ``billing`` has columns date, provider,
    account, service, category, cost_usd. ``truth`` has one row per injected
    anomaly including ``injected_excess_usd`` (exact dollars added).
    Planned events are always applied (they are normal business activity) and
    are listed by :func:`planned_events_table`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)
    weekend = np.asarray(dates.dayofweek >= 5)
    month_end = month_end_mask(dates)
    catalog = series_catalog()

    frames = []
    excess_by_id: dict[str, float] = {}
    for row in catalog.itertuples(index=False):
        provider, account, kind, category, service = (
            row.provider, row.account, row.account_kind, row.category, row.service,
        )
        kind_spec = ACCOUNT_KINDS[kind]
        base = (
            CATEGORY_BASE[category]
            * PROVIDERS[provider]["scale"]
            * kind_spec["mult"]
            * kind_spec["tilt"].get(category, 1.0)
        )
        base *= rng.uniform(0.85, 1.15)
        growth = rng.uniform(*GROWTH_RANGE[category])
        trend = (1.0 + growth) ** (t / 365.0)

        prod_wk, nonprod_wk = WEEKEND_FACTOR[category]
        wk = prod_wk if kind == "prod" else nonprod_wk
        weekly = np.where(weekend, wk, 1.0)
        monthly = np.where(month_end, MONTH_END_FACTOR[category], 1.0)

        sigma = NOISE_SIGMA[category]
        eps = rng.standard_normal(n_days)
        ar = np.empty(n_days)
        ar[0] = sigma * eps[0]
        innov = sigma * np.sqrt(1.0 - AR_PHI**2)
        for i in range(1, n_days):
            ar[i] = AR_PHI * ar[i - 1] + innov * eps[i]
        noise = np.exp(ar - 0.5 * sigma**2)

        cost = base * trend * weekly * monthly * noise

        for ev in planned:
            if category in ev.categories and kind in ev.account_kinds:
                offset = np.asarray((dates - pd.Timestamp(ev.start)).days)
                active = (offset >= 0) & (offset < ev.days)
                cost = cost * np.where(active, 1.0 + ev.uplift, 1.0)

        if inject:
            for spec in anomalies:
                if (spec.provider, spec.account, spec.category) == (provider, account, category):
                    mult = anomaly_multiplier(spec, dates)
                    added = cost * mult
                    excess_by_id[spec.anomaly_id] = float(added.sum())
                    cost = cost + added

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "provider": provider,
                    "account": account,
                    "service": service,
                    "category": category,
                    "cost_usd": np.round(cost, 2),
                }
            )
        )

    billing = pd.concat(frames, ignore_index=True)
    truth = build_truth(anomalies, excess_by_id if inject else {})
    return billing, truth


def build_truth(anomalies: tuple[AnomalySpec, ...], excess_by_id: dict[str, float]) -> pd.DataFrame:
    rows = []
    for spec in anomalies:
        d = asdict(spec)
        start = pd.Timestamp(spec.start)
        rows.append(
            {
                "anomaly_id": spec.anomaly_id,
                "type": spec.kind,
                "provider": spec.provider,
                "account": spec.account,
                "service": PROVIDERS[spec.provider]["services"][spec.category],
                "category": spec.category,
                "start_date": start.date().isoformat(),
                "end_date": (start + pd.Timedelta(days=spec.days - 1)).date().isoformat(),
                "duration_days": spec.days,
                "magnitude_pct": round(100.0 * d["magnitude"], 1),
                "injected_excess_usd": round(excess_by_id.get(spec.anomaly_id, 0.0), 2),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def planned_events_table(planned: tuple[PlannedEvent, ...] = PLANNED_EVENTS) -> pd.DataFrame:
    rows = []
    for ev in planned:
        start = pd.Timestamp(ev.start)
        rows.append(
            {
                "event_id": ev.event_id,
                "start_date": start.date().isoformat(),
                "end_date": (start + pd.Timedelta(days=ev.days - 1)).date().isoformat(),
                "categories": "+".join(ev.categories),
                "account_kinds": "+".join(ev.account_kinds),
                "uplift_pct": round(100.0 * ev.uplift, 1),
                "description": ev.description,
            }
        )
    return pd.DataFrame(rows)


BILLING_FILE = "billing_daily.csv"
TRUTH_FILE = "anomalies_truth.csv"


def save_dataset(billing: pd.DataFrame, truth: pd.DataFrame, data_dir: Path) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    billing_path = data_dir / BILLING_FILE
    truth_path = data_dir / TRUTH_FILE
    out = billing.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(billing_path, index=False)
    truth.to_csv(truth_path, index=False)
    return billing_path, truth_path


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    billing = pd.read_csv(data_dir / BILLING_FILE, parse_dates=["date"])
    truth = pd.read_csv(data_dir / TRUTH_FILE)
    return billing, truth


def write_examples(billing: pd.DataFrame, truth: pd.DataFrame, examples_dir: Path, head_days: int = 31) -> None:
    """Write small committed samples: first month of line items, daily totals, truth."""
    examples_dir = Path(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    b = billing.copy()
    b["date"] = pd.to_datetime(b["date"])
    first = b["date"].min()
    head = b[b["date"] < first + pd.Timedelta(days=head_days)].copy()
    head["date"] = head["date"].dt.strftime("%Y-%m-%d")
    head.to_csv(examples_dir / "sample_billing_first_month.csv", index=False)

    daily = b.pivot_table(index="date", columns="provider", values="cost_usd", aggfunc="sum")
    daily["total"] = daily.sum(axis=1)
    daily = daily.round(2)
    daily.index = daily.index.strftime("%Y-%m-%d")
    daily.index.name = "date"
    daily.to_csv(examples_dir / "daily_spend_by_provider.csv")
    truth.to_csv(examples_dir / "anomalies_truth.csv", index=False)
    planned_events_table().to_csv(examples_dir / "planned_events.csv", index=False)
