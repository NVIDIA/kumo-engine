# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small seeded datasets with planted signals, for examples, notebooks and tests.

Each generator writes Parquet tables and an anchor table under a directory you
choose and returns the paths and metadata. The planted signal is known exactly,
so a model can be checked against it rather than against intuition.

- `hm`: an H&M-shaped retail dataset (customers, articles, transactions) whose
  purchases follow a known function of past rows; the metadata records the
  best AUROC any model that reads only past rows can reach.
- `banking`: users, accounts, merchants, devices and transactions, with planted
  account takeovers labelled ``is_fraud``.
"""

from __future__ import annotations

from pathlib import Path

from kumo.pytorch.datasets import bank, retail
from kumo.pytorch.datasets.bank import BankingDataset
from kumo.pytorch.datasets.metrics import auroc
from kumo.pytorch.datasets.retail import HMDataset

__all__ = ["BankingDataset", "HMDataset", "auroc", "banking", "hm"]


def hm(root: str | Path, scale: str = "xs", seed: int = 0) -> HMDataset:
    """Generate the H&M-shaped retail dataset under ``root``.

    :param root: directory to write; created if missing.
    :param scale: ``"xs"`` (200 customers, 60 articles, about 8k transactions,
        under a second), ``"s"`` or ``"m"``.
    :param seed: the same seed writes the same tables.
    :returns: the paths and the metadata, including ``ceiling_auroc`` per split.
    :raises ValueError: for an unknown scale.
    """
    return retail.generate(scale.upper(), seed, Path(root))


def banking(root: str | Path, seed: int = 0) -> BankingDataset:
    """Generate the banking dataset with planted account takeovers under ``root``.

    About 400 users and 40k transactions over 126 days, written in under a
    second. Anchors are one row per user per weekly cutoff, split by time.

    :param root: directory to write; created if missing.
    :param seed: the same seed writes the same tables.
    :returns: the paths and the metadata, including the positive rate per split.
    """
    return bank.generate(Path(root), seed)
