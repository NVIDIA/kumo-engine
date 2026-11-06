# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An H&M-shaped retail dataset with a planted, exactly known purchase signal.

Three tables shaped like RelBench ``rel-hm``: ``customers`` and ``articles``
(static attributes; each article also has a ``listed_at`` time before
:data:`START`) and ``transactions`` (``customer_id`` -> customers,
``article_id`` -> articles, ``t_dat`` whole-second timestamps). The task is
"does the customer buy anything in the next 7 days", per anchor
``(customer, cutoff)``; train, val and test anchors are split by cutoff time.

**The planted signal.** Time runs in days from :data:`START`. On day ``d`` a
customer makes one shopping session with probability
``h = sigmoid(intercept + z(d) + recency + frequency + affinity)``:

- ``z(d)``: a latent per-customer log-rate, drifting slowly (a sinusoid),
  shifted by static attributes (club status, news frequency, age);
- ``recency = recency_weight * exp(-(days since last session) / recency_scale_days)``;
- ``frequency = frequency_weight * log1p(sessions in the 30 days before d)``;
- ``affinity = affinity_weight * a``, where ``a`` is an exponential average of
  the replenishment score of the product types bought in past sessions.

Ten point-lookup customers are planted: five active ones with a high constant
rate, and five dormant ones that buy often until ``dormant_churn_day`` (day 60,
1 March 2024), when their rate drops by ``churn_offset``.

Recency, frequency and affinity are functions of rows dated before ``d``. If
there is no session, they evolve deterministically, so the probability of at
least one session in ``[t, t + 7 days)`` given everything known at ``t`` is
``1 - prod(1 - h)`` along the no-purchase path. The generator records it per
anchor (``true_prob``). It conditions on the latent rate too, so the AUROC of
``true_prob`` (the **ceiling**) bounds any model that sees only past rows; a
model clearly above it is using future information. ``ceiling_auroc_se`` is the
ceiling's sampling standard error (Hanley and McNeil), the noise a leak band
must allow for.

Sessions happen between 08:00 and 22:00, never at a midnight cutoff, so the
label is the same whether its window is read as ``[t, t + 7d)`` or
``(t, t + 7d]``.

Output layout under ``root``::

    tables/customers.parquet      graph tables
    tables/articles.parquet
    tables/transactions.parquet
    anchors.parquet               sample_id, customer_id, cutoff, split
    oracle/labels.parquet         sample_id, label, true_prob
    oracle/latent.parquet         per-customer latent parameters
    metadata.json                 sizes, cutoffs, ceilings, point lookups, hashes

The oracle files are kept apart from the inputs so a flow can't read them by
accident. Per-customer event sequences are the transactions ordered by
``(t_dat, transaction_id)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch.datasets.metrics import auroc, auroc_standard_error

__all__ = [
    "CLUB_STATUSES",
    "COLOURS",
    "DAY",
    "FLOAT_SUM_REL_TOL",
    "FREQUENCY_WINDOW_DAYS",
    "GENERATOR_VERSION",
    "HORIZON_DAYS",
    "INDEX_GROUPS",
    "NEWS_FREQUENCIES",
    "PRODUCT_TYPES",
    "SCALES",
    "SESSION_HOURS",
    "SPLITS",
    "START",
    "START_EPOCH",
    "HMDataset",
    "Metadata",
    "PointLookup",
    "Scale",
    "SignalParams",
    "fingerprint_mismatches",
    "generate",
    "latent_rate",
    "load",
    "main",
    "sample_anchors",
    "session_hazard",
]

GENERATOR_VERSION = 2
START = datetime(2024, 1, 1)
START_EPOCH = 1_704_067_200
DAY = 86_400
HORIZON_DAYS = 7
FREQUENCY_WINDOW_DAYS = 30
SESSION_HOURS = (8, 22)
SPLITS = ("train", "val", "test")

INDEX_GROUPS = ("Ladieswear", "Menswear", "Baby/Children", "Divided", "Sport")
COLOURS = (
    "Black",
    "White",
    "Dark Blue",
    "Grey",
    "Beige",
    "Pink",
    "Light Blue",
    "Red",
    "Green",
    "Yellow",
)
# (product_type_name, product_group_name, replenishment score in [0, 1])
PRODUCT_TYPES: tuple[tuple[str, str, float], ...] = (
    ("Trousers", "Garment Lower body", 0.3),
    ("Dress", "Garment Full body", 0.2),
    ("Sweater", "Garment Upper body", 0.3),
    ("T-shirt", "Garment Upper body", 0.6),
    ("Top", "Garment Upper body", 0.5),
    ("Blouse", "Garment Upper body", 0.3),
    ("Shorts", "Garment Lower body", 0.3),
    ("Skirt", "Garment Lower body", 0.2),
    ("Jacket", "Garment Upper body", 0.1),
    ("Coat", "Garment Upper body", 0.05),
    ("Socks", "Socks & Tights", 0.9),
    ("Leggings/Tights", "Garment Lower body", 0.6),
    ("Underwear bottom", "Underwear", 0.9),
    ("Bra", "Underwear", 0.7),
    ("Swimwear bottom", "Swimwear", 0.2),
    ("Sneakers", "Shoes", 0.1),
    ("Boots", "Shoes", 0.05),
    ("Bag", "Accessories", 0.1),
    ("Hat/beanie", "Accessories", 0.2),
    ("Earring", "Accessories", 0.4),
)
CLUB_STATUSES = ("ACTIVE", "PRE-CREATE", "LEFT CLUB")
NEWS_FREQUENCIES = ("NONE", "Regularly", "Monthly")


@dataclass(frozen=True)
class Scale:
    """Sizes and the cutoff schedule (days after :data:`START`) per split."""

    customers: int
    articles: int
    days: int
    train_cutoffs: tuple[int, ...]
    val_cutoffs: tuple[int, ...]
    test_cutoffs: tuple[int, ...]

    def cutoffs(self, split: str) -> tuple[int, ...]:
        """The cutoff days of ``split`` (``"train"``, ``"val"`` or ``"test"``)."""
        return {"train": self.train_cutoffs, "val": self.val_cutoffs, "test": self.test_cutoffs}[
            split
        ]


_VAL = (301, 308)
_TEST = (315, 322, 329, 336)
SCALES: dict[str, Scale] = {
    "XS": Scale(200, 60, 364, tuple(range(273, 295, 7)), _VAL, _TEST),
    "S": Scale(2_000, 500, 364, tuple(range(189, 295, 7)), _VAL, _TEST),
    "M": Scale(50_000, 5_000, 364, tuple(range(245, 295, 7)), _VAL, _TEST),
}


@dataclass(frozen=True)
class SignalParams:
    """The known function behind every purchase; recorded in the metadata."""

    intercept: float = -5.5
    latent_sd: float = 0.9
    drift_amplitude_max: float = 0.8
    drift_period_days: tuple[float, float] = (90.0, 365.0)
    club_effect: tuple[float, float, float] = (0.3, -0.3, -0.6)
    news_effect: tuple[float, float, float] = (0.0, 0.2, 0.1)
    young_age: int = 30
    young_effect: float = 0.2
    old_age: int = 60
    old_effect: float = -0.2
    recency_weight: float = 1.5
    recency_scale_days: float = 10.0
    frequency_weight: float = 0.35
    affinity_weight: float = 0.8
    affinity_decay: float = 0.6
    basket_extra_mean: float = 1.5
    preferred_group_share: float = 0.6
    discount_share: float = 0.2
    discount_factor: float = 0.7
    online_share: float = 0.7
    point_lookups_per_kind: int = 5
    active_mu: float = 4.0
    dormant_mu: float = 5.0
    dormant_churn_day: int = 60
    churn_offset: float = -12.0


@dataclass(frozen=True)
class PointLookup:
    """A deliberately extreme customer whose prediction direction is known."""

    customer_id: int
    kind: str
    expected: str
    test_true_prob: tuple[float, ...]


@dataclass(frozen=True)
class Metadata:
    """What ``metadata.json`` records: sizes, cutoffs, ceilings, point lookups and hashes."""

    generator_version: int
    scale: str
    seed: int
    start: str
    horizon_days: int
    cutoffs: dict[str, tuple[str, ...]]
    sizes: dict[str, int]
    ceiling_auroc: dict[str, float]
    ceiling_auroc_se: dict[str, float]
    positive_rate: dict[str, float]
    mean_true_prob: dict[str, float]
    point_lookups: tuple[PointLookup, ...]
    signal: SignalParams
    table_sha256: dict[str, str] = field(default_factory=dict)
    float_sums: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> str:
        """The JSON text written to ``metadata.json``."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Metadata:
        """Read the text :meth:`to_json` wrote."""
        raw = json.loads(text)
        signal = SignalParams(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in raw["signal"].items()
            }
        )
        return cls(
            generator_version=raw["generator_version"],
            scale=raw["scale"],
            seed=raw["seed"],
            start=raw["start"],
            horizon_days=raw["horizon_days"],
            cutoffs={split: tuple(values) for split, values in raw["cutoffs"].items()},
            sizes=raw["sizes"],
            ceiling_auroc=raw["ceiling_auroc"],
            ceiling_auroc_se=raw["ceiling_auroc_se"],
            positive_rate=raw["positive_rate"],
            mean_true_prob=raw["mean_true_prob"],
            point_lookups=tuple(
                PointLookup(
                    customer_id=entry["customer_id"],
                    kind=entry["kind"],
                    expected=entry["expected"],
                    test_true_prob=tuple(entry["test_true_prob"]),
                )
                for entry in raw["point_lookups"]
            ),
            signal=signal,
            table_sha256=raw["table_sha256"],
            float_sums=raw["float_sums"],
        )


@dataclass(frozen=True)
class HMDataset:
    """Paths of one generated dataset and its metadata."""

    root: Path
    metadata: Metadata

    @property
    def tables_dir(self) -> Path:
        """The directory holding the three graph tables."""
        return self.root / "tables"

    def table(self, name: str) -> Path:
        """The Parquet file of table ``name``."""
        return self.tables_dir / f"{name}.parquet"

    @property
    def anchors(self) -> Path:
        """``sample_id, customer_id, cutoff, split``: one row per customer per cutoff."""
        return self.root / "anchors.parquet"

    @property
    def labels(self) -> Path:
        """The oracle ``sample_id, label, true_prob``; never a model input."""
        return self.root / "oracle" / "labels.parquet"

    @property
    def latent(self) -> Path:
        """The oracle per-customer latent parameters."""
        return self.root / "oracle" / "latent.parquet"


@dataclass(frozen=True)
class _Customers:
    age: npt.NDArray[np.int64]
    club: npt.NDArray[np.int64]
    news: npt.NDArray[np.int64]
    postal_area: npt.NDArray[np.int64]
    preferred_group: npt.NDArray[np.int64]
    mu: npt.NDArray[np.float64]
    amplitude: npt.NDArray[np.float64]
    period: npt.NDArray[np.float64]
    phase: npt.NDArray[np.float64]
    churn_day: npt.NDArray[np.int64]
    active: npt.NDArray[np.int64]
    dormant: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _Articles:
    product_type: npt.NDArray[np.int64]
    index_group: npt.NDArray[np.int64]
    colour: npt.NDArray[np.int64]
    base_price: npt.NDArray[np.float64]
    popularity: npt.NDArray[np.float64]

    @property
    def replenish(self) -> npt.NDArray[np.float64]:
        scores = np.array([score for _, _, score in PRODUCT_TYPES], dtype=np.float64)
        return scores[self.product_type]


def latent_rate(
    day: int,
    mu: npt.NDArray[np.float64],
    amplitude: npt.NDArray[np.float64],
    period: npt.NDArray[np.float64],
    phase: npt.NDArray[np.float64],
    churn_day: npt.NDArray[np.int64],
    signal: SignalParams,
) -> npt.NDArray[np.float64]:
    """Each customer's latent log-rate on ``day``: drift plus the churn offset once churned."""
    drift = amplitude * np.sin(2.0 * np.pi * day / period + phase)
    return mu + drift + np.where(day >= churn_day, signal.churn_offset, 0.0)


def session_hazard(
    day: int,
    latent: npt.NDArray[np.float64],
    last_day: npt.NDArray[np.int64],
    sessions_30d: npt.NDArray[np.int64],
    affinity: npt.NDArray[np.float64],
    signal: SignalParams,
) -> npt.NDArray[np.float64]:
    """Probability of a session on ``day``; ``last_day`` is -1 before the first one."""
    since = (day - last_day).astype(np.float64)
    recency = np.where(last_day >= 0, np.exp(-since / signal.recency_scale_days), 0.0)
    logit = (
        signal.intercept
        + latent
        + signal.recency_weight * recency
        + signal.frequency_weight * np.log1p(sessions_30d.astype(np.float64))
        + signal.affinity_weight * affinity
    )
    return 1.0 / (1.0 + np.exp(-logit))


def _articles(rng: np.random.Generator, count: int) -> _Articles:
    return _Articles(
        product_type=rng.integers(0, len(PRODUCT_TYPES), count),
        index_group=rng.integers(0, len(INDEX_GROUPS), count),
        colour=rng.integers(0, len(COLOURS), count),
        base_price=np.round(rng.lognormal(math.log(0.025), 0.5, count), 6),
        popularity=rng.lognormal(0.0, 1.0, count),
    )


def _customers(rng: np.random.Generator, count: int, days: int, signal: SignalParams) -> _Customers:
    age = 16 + np.minimum(np.round(rng.gamma(2.0, 8.0, count)), 64).astype(np.int64)
    club = rng.choice(len(CLUB_STATUSES), count, p=(0.85, 0.10, 0.05))
    news = rng.choice(len(NEWS_FREQUENCIES), count, p=(0.60, 0.38, 0.02))
    postal_area = rng.integers(0, max(10, count // 20), count)
    preferred_group = rng.integers(0, len(INDEX_GROUPS), count)
    mu = (
        np.asarray(signal.club_effect)[club]
        + np.asarray(signal.news_effect)[news]
        + np.where(age < signal.young_age, signal.young_effect, 0.0)
        + np.where(age >= signal.old_age, signal.old_effect, 0.0)
        + rng.normal(0.0, signal.latent_sd, count)
    )
    amplitude = rng.uniform(0.0, signal.drift_amplitude_max, count)
    period = rng.uniform(*signal.drift_period_days, count)
    phase = rng.uniform(0.0, 2.0 * np.pi, count)
    churn_day = np.full(count, days, dtype=np.int64)
    extremes = rng.choice(count, 2 * signal.point_lookups_per_kind, replace=False)
    active = np.sort(extremes[: signal.point_lookups_per_kind])
    dormant = np.sort(extremes[signal.point_lookups_per_kind :])
    mu[active] = signal.active_mu
    mu[dormant] = signal.dormant_mu
    amplitude[extremes] = 0.0
    churn_day[dormant] = signal.dormant_churn_day
    return _Customers(
        age=age,
        club=club,
        news=news,
        postal_area=postal_area,
        preferred_group=preferred_group,
        mu=mu,
        amplitude=amplitude,
        period=period,
        phase=phase,
        churn_day=churn_day,
        active=active,
        dormant=dormant,
    )


@dataclass
class _Simulation:
    sessions: npt.NDArray[np.bool_]
    true_prob: dict[int, npt.NDArray[np.float64]]
    tx_second: list[npt.NDArray[np.int64]] = field(default_factory=list)
    tx_customer: list[npt.NDArray[np.int64]] = field(default_factory=list)
    tx_article: list[npt.NDArray[np.int64]] = field(default_factory=list)
    tx_price: list[npt.NDArray[np.float64]] = field(default_factory=list)
    tx_channel: list[npt.NDArray[np.int64]] = field(default_factory=list)


def _simulate(
    rng: np.random.Generator,
    scale: Scale,
    customers: _Customers,
    articles: _Articles,
    signal: SignalParams,
) -> _Simulation:
    count = scale.customers
    cutoffs = set(scale.train_cutoffs + scale.val_cutoffs + scale.test_cutoffs)
    sim = _Simulation(sessions=np.zeros((scale.days, count), dtype=np.bool_), true_prob={})
    last_day = np.full(count, -1, dtype=np.int64)
    sessions_30d = np.zeros(count, dtype=np.int64)
    affinity = np.zeros(count, dtype=np.float64)
    replenish = articles.replenish
    global_cdf = np.cumsum(articles.popularity) / articles.popularity.sum()
    group_members = [np.flatnonzero(articles.index_group == g) for g in range(len(INDEX_GROUPS))]
    group_cdfs = [
        np.cumsum(articles.popularity[members]) / articles.popularity[members].sum()
        for members in group_members
    ]

    def latent(day: int) -> npt.NDArray[np.float64]:
        return latent_rate(
            day,
            customers.mu,
            customers.amplitude,
            customers.period,
            customers.phase,
            customers.churn_day,
            signal,
        )

    for day in range(scale.days):
        if day in cutoffs:
            survival = np.ones(count, dtype=np.float64)
            dropped = np.zeros(count, dtype=np.int64)
            for offset in range(HORIZON_DAYS):
                hazard = session_hazard(
                    day + offset, latent(day + offset), last_day, sessions_30d - dropped,
                    affinity, signal,
                )  # fmt: skip
                survival *= 1.0 - hazard
                leaving = day - FREQUENCY_WINDOW_DAYS + offset
                if leaving >= 0:
                    dropped += sim.sessions[leaving]
            sim.true_prob[day] = 1.0 - survival

        hazard = session_hazard(day, latent(day), last_day, sessions_30d, affinity, signal)
        buyers = np.flatnonzero(rng.random(count) < hazard)
        sizes = 1 + rng.poisson(signal.basket_extra_mean, buyers.size)
        item_customer = np.repeat(buyers, sizes)
        item_session = np.repeat(np.arange(buyers.size), sizes)
        from_preferred = rng.random(item_customer.size) < signal.preferred_group_share
        draws = rng.random(item_customer.size)
        item_article = np.searchsorted(global_cdf, draws, side="right")
        for group, members in enumerate(group_members):
            chosen = from_preferred & (customers.preferred_group[item_customer] == group)
            picks = np.searchsorted(group_cdfs[group], draws[chosen], side="right")
            item_article[chosen] = members[np.minimum(picks, members.size - 1)]
        item_article = np.minimum(item_article, articles.popularity.size - 1)
        session_second = day * DAY + rng.integers(
            SESSION_HOURS[0] * 3600, SESSION_HOURS[1] * 3600, buyers.size
        )
        channel = np.where(rng.random(buyers.size) < signal.online_share, 2, 1)
        discount = np.where(
            rng.random(item_customer.size) < signal.discount_share, signal.discount_factor, 1.0
        )
        sim.tx_second.append(session_second[item_session])
        sim.tx_customer.append(item_customer)
        sim.tx_article.append(item_article)
        sim.tx_price.append(np.round(articles.base_price[item_article] * discount, 6))
        sim.tx_channel.append(channel[item_session])

        session_replenish = np.bincount(
            item_session, weights=replenish[item_article], minlength=buyers.size
        ) / np.maximum(sizes, 1)
        affinity[buyers] = (
            signal.affinity_decay * affinity[buyers]
            + (1.0 - signal.affinity_decay) * session_replenish
        )
        last_day[buyers] = day
        sim.sessions[day, buyers] = True
        sessions_30d += sim.sessions[day]
        if day >= FREQUENCY_WINDOW_DAYS:
            sessions_30d -= sim.sessions[day - FREQUENCY_WINDOW_DAYS]
    return sim


def _timestamps(seconds: npt.NDArray[np.int64]) -> pa.Array:
    return pa.array(START_EPOCH + seconds, type=pa.timestamp("s"))


def _strings(names: Sequence[str], index: npt.NDArray[np.int64]) -> pa.Array:
    return pa.array(np.asarray(names, dtype=object)[index], type=pa.string())


def _sha256(table: pa.Table) -> str:
    """Hash of the non-float columns; floats are fingerprinted by :func:`_float_sums`."""
    exact = table.select(
        [
            n
            for n, c in zip(table.column_names, table.columns, strict=True)
            if not pa.types.is_floating(c.type)
        ]
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, exact.schema) as writer:
        writer.write_table(exact)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _float_sums(table: pa.Table) -> dict[str, float]:
    # NumPy's exp/sin may differ in the last bit by CPU, so floats are compared within a tolerance.
    return {
        name: float(np.sum(column.to_numpy()))
        for name, column in zip(table.column_names, table.columns, strict=True)
        if pa.types.is_floating(column.type)
    }


FLOAT_SUM_REL_TOL = 1e-9


def fingerprint_mismatches(
    metadata: Metadata, table_sha256: dict[str, str], float_sums: dict[str, dict[str, float]]
) -> list[str]:
    """Tables whose recorded fingerprint differs from ``metadata``'s; empty for the same data."""
    mismatched = {
        name
        for name in table_sha256.keys() | metadata.table_sha256.keys()
        if table_sha256.get(name) != metadata.table_sha256.get(name)
    }
    for name in float_sums.keys() | metadata.float_sums.keys():
        recorded, actual = float_sums.get(name, {}), metadata.float_sums.get(name, {})
        if recorded.keys() != actual.keys() or any(
            not math.isclose(recorded[c], actual[c], rel_tol=FLOAT_SUM_REL_TOL, abs_tol=1e-9)
            for c in recorded
        ):
            mismatched.add(name)
    return sorted(mismatched)


def generate(scale: str, seed: int, root: Path, signal: SignalParams | None = None) -> HMDataset:
    """Generate the dataset deterministically from ``seed`` and write it under ``root``."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose one of {sorted(SCALES)}")
    signal = signal or SignalParams()
    size = SCALES[scale]
    article_rng, customer_rng, simulation_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    )
    articles = _articles(article_rng, size.articles)
    customers = _customers(customer_rng, size.customers, size.days, signal)
    sim = _simulate(simulation_rng, size, customers, articles, signal)

    customer_ids = np.arange(size.customers, dtype=np.int64)
    price_band = np.digitize(articles.base_price, (0.02, 0.04))
    tables: dict[str, pa.Table] = {
        "customers": pa.table(
            {
                "customer_id": customer_ids,
                "age": customers.age,
                "club_member_status": _strings(CLUB_STATUSES, customers.club),
                "fashion_news_frequency": _strings(NEWS_FREQUENCIES, customers.news),
                "postal_area": pa.array(
                    [f"area-{area:04d}" for area in customers.postal_area], type=pa.string()
                ),
            }
        ),
        "articles": pa.table(
            {
                "article_id": np.arange(size.articles, dtype=np.int64),
                "product_type_name": _strings(
                    [name for name, _, _ in PRODUCT_TYPES], articles.product_type
                ),
                "product_group_name": _strings(
                    [group for _, group, _ in PRODUCT_TYPES], articles.product_type
                ),
                "colour_group_name": _strings(COLOURS, articles.colour),
                "index_group_name": _strings(INDEX_GROUPS, articles.index_group),
                "department_name": pa.array(
                    [
                        f"{INDEX_GROUPS[group]} {PRODUCT_TYPES[kind][1]}"
                        for group, kind in zip(
                            articles.index_group, articles.product_type, strict=True
                        )
                    ],
                    type=pa.string(),
                ),
                "price_band": _strings(("low", "mid", "high"), price_band),
                "base_price": articles.base_price,
                # Every article is listed before START (no RNG draw), so the
                # table has a clock and graph sampling can reach it.
                "listed_at": _timestamps(-DAY * (1 + np.arange(size.articles) % 365)),
            }
        ),
    }
    tx_second = np.concatenate(sim.tx_second)
    tables["transactions"] = pa.table(
        {
            "transaction_id": np.arange(tx_second.size, dtype=np.int64),
            "t_dat": _timestamps(tx_second),
            "customer_id": np.concatenate(sim.tx_customer).astype(np.int64),
            "article_id": np.concatenate(sim.tx_article).astype(np.int64),
            "price": np.concatenate(sim.tx_price),
            "sales_channel_id": np.concatenate(sim.tx_channel).astype(np.int64),
        }
    )

    anchor_customer: list[npt.NDArray[np.int64]] = []
    anchor_day: list[npt.NDArray[np.int64]] = []
    anchor_split: list[str] = []
    labels: list[npt.NDArray[np.bool_]] = []
    true_prob: list[npt.NDArray[np.float64]] = []
    ceiling: dict[str, float] = {}
    ceiling_se: dict[str, float] = {}
    positive_rate: dict[str, float] = {}
    mean_true_prob: dict[str, float] = {}
    for split in SPLITS:
        split_labels = [
            sim.sessions[day : day + HORIZON_DAYS].any(axis=0) for day in size.cutoffs(split)
        ]
        split_prob = [sim.true_prob[day] for day in size.cutoffs(split)]
        for day in size.cutoffs(split):
            anchor_customer.append(customer_ids)
            anchor_day.append(np.full(size.customers, day, dtype=np.int64))
            anchor_split.extend([split] * size.customers)
        labels.extend(split_labels)
        true_prob.extend(split_prob)
        flat_labels = np.concatenate(split_labels)
        flat_prob = np.concatenate(split_prob)
        ceiling[split] = auroc(flat_labels, flat_prob)
        positives = int(flat_labels.sum())
        ceiling_se[split] = auroc_standard_error(
            ceiling[split], positives, flat_labels.size - positives
        )
        positive_rate[split] = float(flat_labels.mean())
        mean_true_prob[split] = float(flat_prob.mean())

    days = np.concatenate(anchor_day)
    sample_ids = np.arange(days.size, dtype=np.int64)
    anchors = pa.table(
        {
            "sample_id": sample_ids,
            "customer_id": np.concatenate(anchor_customer),
            "cutoff": _timestamps(days * DAY),
            "split": pa.array(anchor_split, type=pa.string()),
        }
    )
    oracle_labels = pa.table(
        {
            "sample_id": sample_ids,
            "label": np.concatenate(labels),
            "true_prob": np.concatenate(true_prob),
        }
    )
    latent = pa.table(
        {
            "customer_id": customer_ids,
            "mu": customers.mu,
            "drift_amplitude": customers.amplitude,
            "drift_period_days": customers.period,
            "drift_phase": customers.phase,
            "churn_day": customers.churn_day,
            "preferred_index_group": _strings(INDEX_GROUPS, customers.preferred_group),
        }
    )

    point_lookups = tuple(
        PointLookup(
            customer_id=int(index),
            kind=kind,
            expected=expected,
            test_true_prob=tuple(float(sim.true_prob[day][index]) for day in size.test_cutoffs),
        )
        for kind, expected, members in (
            ("active", "high", customers.active),
            ("dormant", "low", customers.dormant),
        )
        for index in members
    )
    for lookup in point_lookups:
        probs = lookup.test_true_prob
        if (lookup.expected == "high" and min(probs) < 0.9) or (
            lookup.expected == "low" and max(probs) > 0.05
        ):
            raise RuntimeError(f"point lookup is not extreme: {lookup}")
    for index in customers.dormant:
        if not sim.sessions[: signal.dormant_churn_day, index].any():
            raise RuntimeError(f"dormant customer {index} never bought before churning")

    root.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(exist_ok=True)
    (root / "oracle").mkdir(exist_ok=True)
    written: dict[str, tuple[pa.Table, Path]] = {
        **{name: (table, root / "tables" / f"{name}.parquet") for name, table in tables.items()},
        "anchors": (anchors, root / "anchors.parquet"),
        "oracle/labels": (oracle_labels, root / "oracle" / "labels.parquet"),
        "oracle/latent": (latent, root / "oracle" / "latent.parquet"),
    }
    for table, path in written.values():
        pq.write_table(table, path)

    metadata = Metadata(
        generator_version=GENERATOR_VERSION,
        scale=scale,
        seed=seed,
        start=START.isoformat(),
        horizon_days=HORIZON_DAYS,
        cutoffs={
            split: tuple((START + timedelta(days=day)).isoformat() for day in size.cutoffs(split))
            for split in SPLITS
        },
        sizes={name: table.num_rows for name, (table, _) in written.items()},
        ceiling_auroc=ceiling,
        ceiling_auroc_se=ceiling_se,
        positive_rate=positive_rate,
        mean_true_prob=mean_true_prob,
        point_lookups=point_lookups,
        signal=signal,
        table_sha256={name: _sha256(table) for name, (table, _) in written.items()},
        float_sums={name: _float_sums(table) for name, (table, _) in written.items()},
    )
    (root / "metadata.json").write_text(metadata.to_json())
    return HMDataset(root, metadata)


def load(root: Path) -> HMDataset:
    """Open a dataset written by :func:`generate`."""
    return HMDataset(root, Metadata.from_json((root / "metadata.json").read_text()))


def sample_anchors(
    dataset: HMDataset, fractions: Mapping[str, float], seed: int, out: Path
) -> HMDataset:
    """The same tables with a fixed random sample of the anchors of the named splits.

    Other splits keep every anchor. ``tables/``, ``oracle/`` and ``metadata.json``
    link to ``dataset``'s, so the metadata still describes the full dataset.
    """
    anchors = pq.read_table(dataset.anchors)
    split = anchors["split"].to_numpy(zero_copy_only=False)
    keep = np.ones(len(anchors), dtype=bool)
    rng = np.random.default_rng(seed)
    for name, fraction in sorted(fractions.items()):
        if not 0 < fraction <= 1:
            raise ValueError(f"the {name} fraction must be in (0, 1], not {fraction}")
        rows = np.flatnonzero(split == name)
        if rows.size == 0:
            raise ValueError(f"{dataset.anchors} has no {name!r} anchors to sample")
        keep[rows] = False
        keep[rng.choice(rows, size=round(fraction * rows.size), replace=False)] = True
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "oracle", "metadata.json"):
        link = out / name
        link.unlink(missing_ok=True)
        link.symlink_to((dataset.root / name).resolve())
    pq.write_table(anchors.filter(pa.array(keep)), out / "anchors.parquet")
    return load(out)


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: ``--scale``, ``--seed`` and ``--out``; prints sizes and ceilings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", choices=sorted(SCALES), default="S")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    began = time.perf_counter()
    dataset = generate(args.scale, args.seed, args.out)
    elapsed = time.perf_counter() - began
    meta = dataset.metadata
    print(f"wrote {args.scale} seed {args.seed} to {args.out} in {elapsed:.1f}s")
    print("sizes:", json.dumps(meta.sizes))
    print("ceiling AUROC:", json.dumps({k: round(v, 4) for k, v in meta.ceiling_auroc.items()}))
    print("positive rate:", json.dumps({k: round(v, 4) for k, v in meta.positive_rate.items()}))


if __name__ == "__main__":
    main()
