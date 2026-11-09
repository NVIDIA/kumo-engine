# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small banking dataset with planted account takeovers.

Five tables: ``users``, ``accounts``, ``merchants`` and ``devices`` (static)
and ``transactions`` (``occurred_at`` whole-second timestamps). Each user buys
from merchants in a few preferred categories, on their own device, in their home
currency, during the day.

**The planted signal.** A fraction of users is compromised on a random day.
That day brings a few tiny "card testing" payments at high-risk merchants from a
new device; over the following days come larger payments at high-risk merchants,
at night, often in a foreign currency, from the same new device. Every one of
these rows has ``is_fraud = 1``. The task is "does the user have a fraudulent
transaction in the next 7 days", per anchor ``(user, cutoff)``: the label is a
future-window sum of ``is_fraud``, so a contract computes it from the rows, and
the precursors before a cutoff are what make it predictable.

Output layout under ``root``::

    tables/users.parquet, accounts.parquet, merchants.parquet,
           devices.parquet, transactions.parquet
    anchors.parquet               sample_id, user_id, cutoff, split
    metadata.json                 sizes, cutoffs, positive rates, parameters
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "BANKING_TABLES",
    "BankingDataset",
    "BankingMetadata",
    "FraudParams",
    "generate",
    "load",
]

GENERATOR_VERSION = 1
START = datetime(2024, 1, 1)
START_EPOCH = 1_704_067_200
DAY = 86_400
HORIZON_DAYS = 7
SPLITS = ("train", "val", "test")
BANKING_TABLES = ("users", "accounts", "merchants", "devices", "transactions")
"""The table names, in the order they are written."""

COUNTRIES = (("US", "USD"), ("GB", "GBP"), ("DE", "EUR"), ("FR", "EUR"), ("CA", "CAD"))
FOREIGN_CURRENCIES = ("USD", "GBP", "EUR", "CAD", "BRL", "NGN", "RUB")
RISK_SEGMENTS = ("low", "medium", "high")
ACCOUNT_TYPES = ("checking", "savings", "credit")
DEVICE_OS = ("ios", "android", "web")
CHANNELS = ("card_present", "online", "mobile_app")
# (category, typical amount, high risk)
CATEGORIES: tuple[tuple[str, float, bool], ...] = (
    ("grocery", 45.0, False),
    ("restaurant", 30.0, False),
    ("fuel", 50.0, False),
    ("pharmacy", 20.0, False),
    ("clothing", 70.0, False),
    ("utilities", 90.0, False),
    ("travel", 300.0, False),
    ("electronics", 250.0, False),
    ("gift_cards", 100.0, True),
    ("crypto_exchange", 400.0, True),
    ("money_transfer", 350.0, True),
)


@dataclass(frozen=True)
class FraudParams:
    """The parameters of the simulation; recorded in the metadata."""

    users: int = 400
    merchants_per_category: int = 6
    days: int = 126
    daily_rate_mean: float = 0.8
    compromised_share: float = 0.3
    first_compromise_day: int = 28
    card_tests: tuple[int, int] = (1, 3)
    fraud_days: int = 10
    fraud_daily_probability: float = 0.6
    fraud_foreign_share: float = 0.6
    traveller_foreign_share: float = 0.03
    train_cutoffs: tuple[int, ...] = tuple(range(35, 92, 7))
    val_cutoffs: tuple[int, ...] = (98, 105)
    test_cutoffs: tuple[int, ...] = (112, 119)

    def cutoffs(self, split: str) -> tuple[int, ...]:
        """The cutoff days of ``split`` (``"train"``, ``"val"`` or ``"test"``)."""
        return {"train": self.train_cutoffs, "val": self.val_cutoffs, "test": self.test_cutoffs}[
            split
        ]


@dataclass(frozen=True)
class BankingMetadata:
    """What ``metadata.json`` records."""

    generator_version: int
    seed: int
    start: str
    horizon_days: int
    cutoffs: dict[str, tuple[str, ...]]
    sizes: dict[str, int]
    positive_rate: dict[str, float]
    compromised_users: int
    params: FraudParams

    def to_json(self) -> str:
        """The JSON text written to ``metadata.json``."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> BankingMetadata:
        """Read the text :meth:`to_json` wrote."""
        raw = json.loads(text)
        params = FraudParams(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in raw["params"].items()
            }
        )
        return cls(
            generator_version=raw["generator_version"],
            seed=raw["seed"],
            start=raw["start"],
            horizon_days=raw["horizon_days"],
            cutoffs={split: tuple(values) for split, values in raw["cutoffs"].items()},
            sizes=raw["sizes"],
            positive_rate=raw["positive_rate"],
            compromised_users=raw["compromised_users"],
            params=params,
        )


@dataclass(frozen=True)
class BankingDataset:
    """Paths of one generated banking dataset and its metadata."""

    root: Path
    metadata: BankingMetadata

    @property
    def tables_dir(self) -> Path:
        """The directory holding the five tables."""
        return self.root / "tables"

    def table(self, name: str) -> Path:
        """The Parquet file of table ``name``."""
        return self.tables_dir / f"{name}.parquet"

    @property
    def anchors(self) -> Path:
        """``sample_id, user_id, cutoff, split``: one row per user per cutoff."""
        return self.root / "anchors.parquet"


def _strings(names: tuple[str, ...], index: npt.NDArray[np.int64]) -> pa.Array:
    return pa.array(np.asarray(names, dtype=object)[index], type=pa.string())


def generate(root: Path, seed: int = 0, params: FraudParams | None = None) -> BankingDataset:
    """Generate the dataset deterministically from ``seed`` and write it under ``root``."""
    params = params or FraudParams()
    rng = np.random.default_rng(seed)
    users = params.users

    country = rng.integers(0, len(COUNTRIES), users)
    home_currency = np.array([COUNTRIES[c][1] for c in country], dtype=object)
    risk = rng.choice(len(RISK_SEGMENTS), users, p=(0.6, 0.3, 0.1))
    birth_year = rng.integers(1950, 2005, users)

    account_owner = np.concatenate([np.arange(users), rng.choice(users, users // 2, False)])
    account_type = rng.integers(0, len(ACCOUNT_TYPES), account_owner.size)
    primary_account = np.arange(users)

    categories = len(CATEGORIES)
    merchant_category = np.repeat(np.arange(categories), params.merchants_per_category)
    high_risk_category = np.array([risky for _, _, risky in CATEGORIES])
    merchant_risk = np.round(
        np.where(
            high_risk_category[merchant_category],
            rng.uniform(0.6, 0.95, merchant_category.size),
            rng.uniform(0.0, 0.3, merchant_category.size),
        ),
        3,
    )
    risky = np.flatnonzero(high_risk_category[merchant_category])

    device_owner = list(range(users))
    device_os = list(rng.integers(0, len(DEVICE_OS), users))

    rate = rng.lognormal(np.log(params.daily_rate_mean) - 0.125, 0.5, users)
    everyday_categories = np.flatnonzero(~high_risk_category)
    preferred = np.stack([rng.choice(everyday_categories, 3, replace=False) for _ in range(users)])
    preferred_channel = rng.integers(0, len(CHANNELS), users)

    compromised = np.flatnonzero(rng.random(users) < params.compromised_share)
    compromise_day = rng.integers(
        params.first_compromise_day, params.days - 2, compromised.size
    ).astype(np.int64)

    rows: dict[str, list[object]] = {
        name: []
        for name in (
            "user_id",
            "account_id",
            "merchant_id",
            "device_id",
            "second",
            "amount",
            "currency",
            "channel",
            "is_fraud",
        )
    }

    def add(
        user: int,
        merchant: int,
        device: int,
        second: int,
        amount: float,
        currency: str,
        channel: int,
        fraud: int,
    ) -> None:
        rows["user_id"].append(user)
        rows["account_id"].append(int(primary_account[user]))
        rows["merchant_id"].append(merchant)
        rows["device_id"].append(device)
        rows["second"].append(second)
        rows["amount"].append(round(amount, 2))
        rows["currency"].append(currency)
        rows["channel"].append(CHANNELS[channel])
        rows["is_fraud"].append(fraud)

    for user in range(users):
        counts = rng.poisson(rate[user], params.days)
        for day in np.flatnonzero(counts):
            for _ in range(int(counts[day])):
                category = int(rng.choice(preferred[user]))
                merchant = int(
                    category * params.merchants_per_category
                    + rng.integers(0, params.merchants_per_category)
                )
                traveller = rng.random() < params.traveller_foreign_share
                add(
                    user,
                    merchant,
                    user,
                    int(day * DAY + rng.integers(8 * 3600, 22 * 3600)),
                    float(rng.lognormal(np.log(CATEGORIES[category][1]), 0.6)),
                    str(rng.choice(FOREIGN_CURRENCIES)) if traveller else str(home_currency[user]),
                    int(preferred_channel[user] if rng.random() < 0.8 else rng.integers(0, 3)),
                    0,
                )

    for user, day in zip(compromised.tolist(), compromise_day.tolist(), strict=True):
        device = len(device_owner)
        device_owner.append(user)
        device_os.append(int(rng.integers(0, len(DEVICE_OS))))
        for _ in range(int(rng.integers(params.card_tests[0], params.card_tests[1] + 1))):
            add(
                user,
                int(rng.choice(risky)),
                device,
                int(day * DAY + rng.integers(0, DAY)),
                float(rng.uniform(0.5, 2.0)),
                str(home_currency[user]),
                1,
                1,
            )
        for offset in range(1, params.fraud_days + 1):
            if day + offset >= params.days or rng.random() >= params.fraud_daily_probability:
                continue
            for _ in range(int(rng.integers(1, 4))):
                category = int(merchant_category[int(rng.choice(risky))])
                foreign = rng.random() < params.fraud_foreign_share
                add(
                    user,
                    int(rng.choice(risky)),
                    device,
                    int((day + offset) * DAY + rng.integers(0, 6 * 3600)),
                    float(rng.lognormal(np.log(CATEGORIES[category][1]), 0.4)),
                    str(rng.choice(FOREIGN_CURRENCIES)) if foreign else str(home_currency[user]),
                    int(rng.integers(1, 3)),
                    1,
                )

    order = np.lexsort((np.array(rows["user_id"]), np.array(rows["second"])))
    second = np.array(rows["second"], dtype=np.int64)[order]
    tables: dict[str, pa.Table] = {
        "users": pa.table(
            {
                "user_id": np.arange(users, dtype=np.int64),
                "country": _strings(tuple(c for c, _ in COUNTRIES), country),
                "home_currency": pa.array(home_currency, type=pa.string()),
                "risk_segment": _strings(RISK_SEGMENTS, risk),
                "birth_year": birth_year.astype(np.int64),
            }
        ),
        "accounts": pa.table(
            {
                "account_id": np.arange(account_owner.size, dtype=np.int64),
                "user_id": account_owner.astype(np.int64),
                "account_type": _strings(ACCOUNT_TYPES, account_type),
            }
        ),
        "merchants": pa.table(
            {
                "merchant_id": np.arange(merchant_category.size, dtype=np.int64),
                "category": _strings(tuple(c for c, _, _ in CATEGORIES), merchant_category),
                "risk_score": merchant_risk,
            }
        ),
        "devices": pa.table(
            {
                "device_id": np.arange(len(device_owner), dtype=np.int64),
                "user_id": np.array(device_owner, dtype=np.int64),
                "os": _strings(DEVICE_OS, np.array(device_os, dtype=np.int64)),
            }
        ),
        "transactions": pa.table(
            {
                "transaction_id": np.arange(second.size, dtype=np.int64),
                "user_id": np.array(rows["user_id"], dtype=np.int64)[order],
                "account_id": np.array(rows["account_id"], dtype=np.int64)[order],
                "merchant_id": np.array(rows["merchant_id"], dtype=np.int64)[order],
                "device_id": np.array(rows["device_id"], dtype=np.int64)[order],
                "occurred_at": pa.array(START_EPOCH + second, type=pa.timestamp("s")),
                "amount": np.array(rows["amount"], dtype=np.float64)[order],
                "currency": pa.array(np.array(rows["currency"], dtype=object)[order], pa.string()),
                "channel": pa.array(np.array(rows["channel"], dtype=object)[order], pa.string()),
                "is_fraud": np.array(rows["is_fraud"], dtype=np.int64)[order],
            }
        ),
    }

    fraud_seconds = second[tables["transactions"]["is_fraud"].to_numpy() == 1]
    fraud_users = tables["transactions"]["user_id"].to_numpy()[
        tables["transactions"]["is_fraud"].to_numpy() == 1
    ]
    sample_id: list[int] = []
    anchor_user: list[int] = []
    anchor_day: list[int] = []
    anchor_split: list[str] = []
    positive_rate: dict[str, float] = {}
    for split in SPLITS:
        positives = 0
        for day in params.cutoffs(split):
            start, stop = day * DAY, (day + HORIZON_DAYS) * DAY
            labelled = set(fraud_users[(fraud_seconds > start) & (fraud_seconds <= stop)].tolist())
            positives += len(labelled)
            for user in range(users):
                sample_id.append(len(sample_id))
                anchor_user.append(user)
                anchor_day.append(day)
                anchor_split.append(split)
        positive_rate[split] = positives / (users * len(params.cutoffs(split)))
    anchors = pa.table(
        {
            "sample_id": np.array(sample_id, dtype=np.int64),
            "user_id": np.array(anchor_user, dtype=np.int64),
            "cutoff": pa.array(
                START_EPOCH + np.array(anchor_day, dtype=np.int64) * DAY, pa.timestamp("s")
            ),
            "split": pa.array(anchor_split, type=pa.string()),
        }
    )

    root = Path(root)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        pq.write_table(table, root / "tables" / f"{name}.parquet")
    pq.write_table(anchors, root / "anchors.parquet")
    metadata = BankingMetadata(
        generator_version=GENERATOR_VERSION,
        seed=seed,
        start=START.isoformat(),
        horizon_days=HORIZON_DAYS,
        cutoffs={
            split: tuple((START + timedelta(days=day)).isoformat() for day in params.cutoffs(split))
            for split in SPLITS
        },
        sizes={**{name: table.num_rows for name, table in tables.items()}, "anchors": len(anchors)},
        positive_rate=positive_rate,
        compromised_users=int(compromised.size),
        params=params,
    )
    (root / "metadata.json").write_text(metadata.to_json())
    return BankingDataset(root, metadata)


def load(root: Path) -> BankingDataset:
    """Open a dataset written by :func:`generate`."""
    root = Path(root)
    return BankingDataset(root, BankingMetadata.from_json((root / "metadata.json").read_text()))
