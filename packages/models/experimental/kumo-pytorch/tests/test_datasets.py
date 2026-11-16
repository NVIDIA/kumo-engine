# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import datasets
from kumo.pytorch.datasets.bank import BANKING_TABLES, DAY, START_EPOCH


@pytest.fixture(scope="module")
def banking(tmp_path_factory: pytest.TempPathFactory) -> datasets.BankingDataset:
    return datasets.banking(tmp_path_factory.mktemp("banking"), seed=3)


def test_banking_is_deterministic(banking: datasets.BankingDataset, tmp_path: Path) -> None:
    again = datasets.banking(tmp_path, seed=3)
    for name in BANKING_TABLES:
        assert pq.read_table(again.table(name)).equals(pq.read_table(banking.table(name)))
    assert pq.read_table(again.anchors).equals(pq.read_table(banking.anchors))
    assert again.metadata == banking.metadata
    assert datasets.bank.load(banking.root).metadata == banking.metadata
    other = datasets.banking(tmp_path / "other", seed=4)
    assert not pq.read_table(other.table("transactions")).equals(
        pq.read_table(banking.table("transactions"))
    )


def test_banking_keys_join(banking: datasets.BankingDataset) -> None:
    tables = {name: pq.read_table(banking.table(name)) for name in BANKING_TABLES}
    transactions = tables["transactions"]
    for column, table, key in (
        ("user_id", "users", "user_id"),
        ("account_id", "accounts", "account_id"),
        ("merchant_id", "merchants", "merchant_id"),
        ("device_id", "devices", "device_id"),
    ):
        assert pc.all(pc.is_in(transactions[column], tables[table][key])).as_py(), column
    assert transactions["transaction_id"].to_pylist() == list(range(len(transactions)))
    assert banking.metadata.sizes["anchors"] == pq.read_table(banking.anchors).num_rows


def test_banking_plants_takeovers(banking: datasets.BankingDataset) -> None:
    transactions = pq.read_table(banking.table("transactions"))
    merchants = pq.read_table(banking.table("merchants"))
    risky = set(
        merchants.filter(pc.greater(merchants["risk_score"], 0.5))["merchant_id"].to_pylist()
    )
    fraud = np.array(transactions["is_fraud"].to_pylist(), dtype=bool)
    merchant = np.array(transactions["merchant_id"].to_pylist())
    seconds = transactions["occurred_at"].cast(pa.timestamp("s")).cast(pa.int64()).to_numpy()
    hour = (seconds - START_EPOCH) % DAY // 3600
    assert 0 < fraud.mean() < 0.1
    assert np.isin(merchant[fraud], list(risky)).all()
    assert not np.isin(merchant[~fraud], list(risky)).any()
    assert (hour[~fraud] >= 8).all()
    assert (hour[fraud] < 8).mean() > 0.8
    for split, rate in banking.metadata.positive_rate.items():
        assert 0.01 < rate < 0.2, split


def test_hm_accepts_lowercase_scale(tmp_path: Path) -> None:
    data = datasets.hm(tmp_path, scale="xs", seed=1)
    assert data.metadata.scale == "XS"
    assert data.metadata.sizes["customers"] == 200
    with pytest.raises(ValueError, match="unknown scale"):
        datasets.hm(tmp_path / "bad", scale="xl")


def test_auroc_is_exported() -> None:
    assert datasets.auroc(np.array([False, True]), np.array([0.1, 0.9])) == 1.0
