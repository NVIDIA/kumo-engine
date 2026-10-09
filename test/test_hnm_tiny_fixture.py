# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import datetime
from collections import Counter
from pathlib import Path

import torch

FIXTURE_ROOT = Path(__file__).parent / 'rest' / 'csv_data' / 'HNM_TINY'


def _read_rows(name: str) -> list[dict[str, str]]:
    with (FIXTURE_ROOT / name).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def test_hnm_tiny_spans_multiple_temporal_training_windows() -> None:
    timestamps = [
        datetime.date.fromisoformat(row['t_dat'])
        for row in _read_rows('transactions.csv')
    ]

    assert max(timestamps) - min(timestamps) >= datetime.timedelta(days=120)


def test_hnm_tiny_has_both_age_classes() -> None:
    ages = [int(row['age']) for row in _read_rows('customers.csv')]

    assert sum(age <= 30 for age in ages) >= 14
    assert sum(age > 30 for age in ages) >= 14


def test_hnm_tiny_seeded_warm_start_split_has_both_classes() -> None:
    ages = [int(row['age']) for row in _read_rows('customers.csv')]
    permutation = torch.randperm(
        len(ages),
        generator=torch.Generator().manual_seed(0),
    ).tolist()
    train_size = round(0.6 * len(ages))
    validation_size = round(0.2 * len(ages))
    validation_labels = {
        ages[index] > 30
        for index in permutation[train_size:train_size + validation_size]
    }

    assert validation_labels == {False, True}


def test_hnm_tiny_transactions_preserve_foreign_keys() -> None:
    customer_ids = {row['customer_id'] for row in _read_rows('customers.csv')}
    article_ids = {row['article_id'] for row in _read_rows('articles.csv')}
    transactions = _read_rows('transactions.csv')
    transactions_by_customer = Counter(row['customer_id']
                                       for row in transactions)

    assert {row['customer_id'] for row in transactions} <= customer_ids
    assert {row['article_id'] for row in transactions} <= article_ids
    assert customer_ids <= {row['customer_id'] for row in transactions}
    assert len(customer_ids) >= 33
    assert len(transactions) >= len(customer_ids)
    assert min(transactions_by_customer.values()) >= 1


def test_hnm_tiny_primary_key_variant_matches_transactions() -> None:
    transactions = _read_rows('transactions.csv')
    keyed_transactions = _read_rows('transactions_w_pkey.csv')
    transaction_ids = [row.pop('transaction_id') for row in keyed_transactions]

    assert len(transaction_ids) == len(set(transaction_ids))
    assert keyed_transactions == transactions
