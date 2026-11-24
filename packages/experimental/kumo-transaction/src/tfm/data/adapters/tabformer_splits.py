# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the TabFormer `temporal_split/` parquet files the `tabformer` adapter reads.

This follows the NVIDIA TFM Blueprint's notebook 01 on CPU, with pandas in
place of cuDF:

- `train`, `val`, `test`: the whole book cut by date where the cumulative row
  count first reaches 80% and 90%. Every row of a day lands on one side.
- `val_eval`, `test_eval`: 100,000-row samples of `val` and `test`, stratified
  on the fraud label with seed 42, in the order scikit-learn draws them.

The split is by transaction date across all users, not by user, so a user's
history can span all three partitions. A recipe reads these partitions as given
with `data.split_protocol: precomputed`.
"""

from __future__ import annotations

import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_NAME = "card_transaction.v1.csv"


@dataclass(frozen=True)
class TabFormerSplitManifest:
    """What `write_tabformer_splits` wrote."""

    train_cutoff: str
    test_cutoff: str
    rows: dict[str, int]
    fraud: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_tabformer(archive: Path, root: Path) -> Path:
    """Extract IBM's `transactions.tgz` to `root/raw/` and return the CSV path."""
    raw = root / "raw"
    csv_path = raw / CSV_NAME
    if csv_path.exists():
        return csv_path
    raw.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=raw, filter="data")
    if not csv_path.exists():
        raise FileNotFoundError(f"{archive} did not contain {CSV_NAME}.")
    return csv_path


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy()


def _cutoff(day_keys: np.ndarray, ratio: float) -> int:
    days, counts = np.unique(day_keys, return_counts=True)
    running = np.cumsum(counts)
    return int(days[int(np.searchsorted(running, running[-1] * ratio))])


def _iso(day_key: int) -> str:
    return f"{day_key // 10000:04d}-{day_key // 100 % 100:02d}-{day_key % 100:02d}"


def write_tabformer_splits(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    eval_samples: int = 100_000,
    seed: int = 42,
) -> TabFormerSplitManifest:
    """Write `train`, `val`, `test`, `val_eval`, and `test_eval` parquet to `out_dir`."""
    frame = pd.read_csv(csv_path)
    frame.columns = pd.Index([str(c).strip() for c in frame.columns])
    day_keys = (
        _column(frame, "Year").astype(np.int64) * 10000
        + _column(frame, "Month").astype(np.int64) * 100
        + _column(frame, "Day").astype(np.int64)
    )
    is_fraud = np.isin(np.char.strip(_column(frame, "Is Fraud?").astype(str)), ["Yes", "1"])
    train_cutoff = _cutoff(day_keys, train_ratio)
    test_cutoff = _cutoff(day_keys, train_ratio + val_ratio)
    masks = {
        "train": day_keys < train_cutoff,
        "val": (day_keys >= train_cutoff) & (day_keys < test_cutoff),
        "test": day_keys >= test_cutoff,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, pd.DataFrame] = {}
    rows: dict[str, int] = {}
    fraud: dict[str, int] = {}
    for name, mask in masks.items():
        part = frame.loc[mask].reset_index(drop=True)
        parts[name] = part
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        rows[name], fraud[name] = len(part), int(is_fraud[mask].sum())
    for name in ("val", "test"):
        part = parts[name]
        labels = is_fraud[masks[name]]
        if eval_samples >= len(part):
            chosen = np.arange(len(part))
        else:
            _, chosen = train_test_split(
                np.arange(len(part)), test_size=eval_samples, stratify=labels, random_state=seed
            )
        sample = part.iloc[chosen].reset_index(drop=True)
        sample.to_parquet(out_dir / f"{name}_eval.parquet", index=False)
        rows[f"{name}_eval"] = len(sample)
        fraud[f"{name}_eval"] = int(labels[chosen].sum())
    return TabFormerSplitManifest(
        train_cutoff=_iso(train_cutoff),
        test_cutoff=_iso(test_cutoff),
        rows=rows,
        fraud=fraud,
    )


__all__ = ["CSV_NAME", "TabFormerSplitManifest", "extract_tabformer", "write_tabformer_splits"]
