# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`write_tabformer_splits` follows the Blueprint's notebook-01 split."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import pytest

from tfm.cli.main import main
from tfm.data.adapters.tabformer_splits import CSV_NAME, write_tabformer_splits


def _book(path: Path, days: int = 20, per_day: int = 10) -> Path:
    rows = []
    for day in range(days):
        for i in range(per_day):
            rows.append(
                {
                    "User": i % 4,
                    "Card": 0,
                    "Year": 2019,
                    "Month": 1 + day // 28,
                    "Day": 1 + day % 28,
                    "Time": f"{i:02d}:00",
                    "Amount": f"${i + 1}.00",
                    "Use Chip": "Chip Transaction",
                    "Merchant Name": 100 + i,
                    "Merchant City": "X",
                    "Merchant State": "CA",
                    "Zip": 90000.0,
                    "MCC": 5411,
                    "Errors?": "",
                    "Is Fraud?": "Yes" if i == 0 else "No",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_split_cuts_by_cumulative_rows_and_keeps_days_whole(tmp_path: Path) -> None:
    csv = _book(tmp_path / CSV_NAME)
    manifest = write_tabformer_splits(csv, tmp_path / "out", eval_samples=10, seed=42)
    # 200 rows over 20 days. The cumulative count reaches 80% on day 16 and 90%
    # on day 18; as in the Blueprint, each partition ends before its cutoff day.
    assert manifest.train_cutoff == "2019-01-16"
    assert manifest.test_cutoff == "2019-01-18"
    assert manifest.rows == {"train": 150, "val": 20, "test": 30, "val_eval": 10, "test_eval": 10}
    train = pd.read_parquet(tmp_path / "out" / "train.parquet")
    val = pd.read_parquet(tmp_path / "out" / "val.parquet")
    assert train["Day"].max() < val["Day"].min()
    evaluation = pd.read_parquet(tmp_path / "out" / "test_eval.parquet")
    assert (evaluation["Is Fraud?"] == "Yes").sum() == manifest.fraud["test_eval"] == 1


def test_cli_extracts_the_archive_and_writes_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = _book(tmp_path / CSV_NAME)
    root = tmp_path / "TabFormer"
    root.mkdir()
    with tarfile.open(root / "transactions.tgz", "w:gz") as tar:
        tar.add(csv, arcname=CSV_NAME)
    assert main(["prepare-tabformer", "--root", str(root)]) == 0
    assert (root / "raw" / CSV_NAME).exists()
    for name in ("train", "val", "test", "val_eval", "test_eval"):
        assert (root / "temporal_split" / f"{name}.parquet").exists()
    assert "TFM_TABFORMER_ROOT" in capsys.readouterr().out
