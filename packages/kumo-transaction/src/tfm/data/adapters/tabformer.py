# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map IBM TabFormer (public synthetic card transactions) into TransactionEvent.

Never fetched by CI. Resolve the local root via ``data.path`` or
``TFM_TABFORMER_ROOT``. Do not commit a symlink into this repository.

Precomputed ``temporal_split/*.parquet`` files use v1's 80/10/10 cumulative-count
cutoffs, not v2 ``train_end`` / ``label_delay``. Set ``split_protocol: precomputed``
to consume those partitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.config import DataConfig
from tfm.contracts.errors import SchemaError
from tfm.contracts.schema import EventLabel, TransactionEvent
from tfm.data.adapters.paths import entity_cap, sample_seed
from tfm.data.adapters.result import AdapterResult
from tfm.data.sampling import stratified_sample_ids
from tfm.data.splits.temporal import SplitAssignment

TABFORMER_COLUMNS = (
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Is Fraud?",
)


def _parse_amount(raw: object) -> float:
    text = str(raw).replace("$", "").replace(",", "").strip()
    if not text or text.lower() in {"none", "nan"}:
        return 0.0
    return float(text)


def _parse_hour(raw: object) -> int:
    text = str(raw).strip()
    if ":" in text:
        return max(0, min(23, int(text.split(":", 1)[0])))
    if not text or text.lower() in {"none", "nan"}:
        return 0
    return max(0, min(23, int(float(text))))


def _as_datetime(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(int(year), int(month), int(day), int(hour), tzinfo=UTC)


def _row_to_event(row: dict[str, Any], *, index: int, label_delay: timedelta) -> TransactionEvent:
    user = str(row["User"])
    hour = _parse_hour(row.get("Time", "00:00"))
    ts = _as_datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]), hour)
    amount = _parse_amount(row.get("Amount", 0.0))
    zip_raw = row.get("Zip")
    zip_val = 0.0 if zip_raw is None else float(zip_raw)
    state = row.get("Merchant State")
    # Online rows carry no merchant state; they share the tokenizer's unknown
    # state, STATE_XX, while CHIP_ONLINE already says the row was online.
    state_text = "XX" if state is None or str(state).strip() == "" else str(state).strip()
    fraud_raw = str(row.get("Is Fraud?", "No")).strip().lower()
    is_fraud = 1.0 if fraud_raw in {"yes", "1", "true"} else 0.0
    event_id = f"{user}-{ts.strftime('%Y%m%d')}-{hour:02d}-{index}"
    return TransactionEvent(
        entity_id=user,
        event_id=event_id,
        event_timestamp=ts,
        timezone="UTC",
        categorical={
            "merchant_name": str(row.get("Merchant Name", "UNK")),
            "merchant_state": state_text,
            "use_chip": str(row.get("Use Chip", "CHIP TRANSACTION")),
            "mcc": str(int(row.get("MCC", -1))),
            "merchant_city": str(row.get("Merchant City", "")),
        },
        numeric={
            "amount": amount,
            "card": float(row.get("Card", 0) or 0),
            "zip": zip_val,
            "user": float(user) if str(user).isdigit() else float(abs(hash(user)) % 3000),
        },
        labels={
            "fraud": EventLabel(
                name="fraud",
                value=is_fraud,
                available_at=ts + label_delay,
                level="event",
            )
        },
    )


def _present_columns(schema: pa.Schema) -> list[str]:
    names = set(schema.names)
    return [col for col in TABFORMER_COLUMNS if col in names]


def _is_fraud_value(raw: object) -> bool:
    return str(raw).strip().lower() in {"yes", "1", "true"}


def _users_with_fraud(path: Path) -> list[object]:
    if not path.exists():
        return []
    try:
        table = pq.read_table(path, columns=["User", "Is Fraud?"])
    except (OSError, pa.ArrowInvalid):
        return []
    if "User" not in table.column_names or "Is Fraud?" not in table.column_names:
        return []
    flags = [_is_fraud_value(v) for v in table.column("Is Fraud?").to_pylist()]
    users = table.column("User").to_pylist()
    return list(dict.fromkeys(user for user, flag in zip(users, flags, strict=True) if flag))


def _fraud_positive_ids(*paths: Path) -> list[object]:
    ordered: list[object] = []
    seen: set[object] = set()
    for path in paths:
        for user in _users_with_fraud(path):
            if user not in seen:
                seen.add(user)
                ordered.append(user)
    return ordered


def _sample_ids(
    unique: Sequence[object],
    *,
    cap: int,
    seed: int,
    positives: Sequence[object] = (),
) -> list[object]:
    if cap <= 0:
        raise SchemaError("max_entities / n_entities must be positive for TabFormer sampling.")
    if positives:
        return stratified_sample_ids(unique, positives, cap=cap, seed=seed)
    ids = list(unique)
    rng = Random(seed)
    rng.shuffle(ids)
    return ids[: min(cap, len(ids))]


def _read_user_ids(path: Path) -> list[object]:
    table = pq.read_table(path, columns=["User"])
    if "User" not in table.column_names:
        raise SchemaError("TabFormer table is missing column 'User'.")
    return table.column("User").unique().to_pylist()


def _filter_parquet_file(path: Path, users: Sequence[object]) -> pa.Table:
    return pq.read_table(path, filters=[("User", "in", list(users))])


def _union_user_ids(paths: Sequence[Path]) -> list[object]:
    seen: set[object] = set()
    ordered: list[object] = []
    for path in paths:
        for user in _read_user_ids(path):
            if user not in seen:
                seen.add(user)
                ordered.append(user)
    return ordered


def _table_to_events(table: pa.Table, *, label_delay: timedelta) -> list[TransactionEvent]:
    cols = _present_columns(table.schema)
    missing = [
        col for col in ("User", "Year", "Month", "Day", "Amount", "Is Fraud?") if col not in cols
    ]
    if missing:
        raise SchemaError(f"TabFormer table missing required columns: {missing}.")
    rows = table.select(cols).to_pylist()
    return [_row_to_event(row, index=i, label_delay=label_delay) for i, row in enumerate(rows)]


def write_tabformer_fixture(
    path: Path | str,
    *,
    n_users: int = 6,
    events_per_user: int = 6,
    seed: int = 0,
) -> Path:
    """Write a tiny TabFormer-shaped parquet book for tests (no real PII)."""

    rng = Random(seed)
    root = Path(path)
    split_dir = root / "temporal_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)

    def rows_for_year(year: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for user in range(n_users):
            for event_i in range(events_per_user):
                amount = round(rng.uniform(5.0, 80.0), 2)
                fraud = rng.random() < 0.2
                if fraud:
                    amount *= 4.0
                out.append(
                    {
                        "User": user,
                        "Card": user % 3,
                        "Year": year,
                        "Month": 1,
                        "Day": 1 + event_i,
                        "Time": f"{8 + event_i:02d}:15",
                        "Amount": f"${amount:.2f}",
                        "Use Chip": "Chip Transaction",
                        "Merchant Name": 3527213246127876953 + user,
                        "Merchant City": "La Verne",
                        "Merchant State": "CA",
                        "Zip": 91750.0,
                        "MCC": 5411,
                        "Is Fraud?": "Yes" if fraud else "No",
                    }
                )
        return out

    schema = pa.schema(
        [
            pa.field("User", pa.int64(), nullable=False),
            pa.field("Card", pa.int64(), nullable=False),
            pa.field("Year", pa.int64(), nullable=False),
            pa.field("Month", pa.int64(), nullable=False),
            pa.field("Day", pa.int64(), nullable=False),
            pa.field("Time", pa.string(), nullable=False),
            pa.field("Amount", pa.string(), nullable=False),
            pa.field("Use Chip", pa.string(), nullable=False),
            pa.field("Merchant Name", pa.int64(), nullable=False),
            pa.field("Merchant City", pa.string(), nullable=False),
            pa.field("Merchant State", pa.string()),
            pa.field("Zip", pa.float64()),
            pa.field("MCC", pa.int64(), nullable=False),
            pa.field("Is Fraud?", pa.string(), nullable=False),
        ]
    )

    def write(name: str, year: int) -> None:
        rows = rows_for_year(year)
        arrays = {
            field.name: pa.array([row[field.name] for row in rows], type=field.type)
            for field in schema
        }
        pq.write_table(pa.table(arrays, schema=schema), split_dir / name)

    write("train.parquet", 2002)
    write("val.parquet", 2016)
    write("test.parquet", 2018)
    return root


def _load_tabformer_csv(
    csv_path: Path, *, cap: int, seed: int, delay: timedelta
) -> tuple[list[TransactionEvent], list[object]]:
    import pandas as pd

    # pandas' own annotations cannot solve `usecols`' type variable for a list.
    unique = pd.read_csv(csv_path, usecols=["User"])["User"].drop_duplicates().tolist()  # type: ignore[call-overload]
    users = _sample_ids(unique, cap=cap, seed=seed)
    selected = set(users)
    chunks: list[Any] = []
    for chunk in pd.read_csv(csv_path, chunksize=250_000):
        filtered = chunk[chunk["User"].isin(selected)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise SchemaError("TabFormer CSV filter produced no rows for the sampled users.")
    table = pa.Table.from_pandas(pd.concat(chunks, ignore_index=True), preserve_index=False)
    return _table_to_events(table, label_delay=delay), users


def precomputed_split_paths(root: Path) -> tuple[Path, Path, Path]:
    """`temporal_split/{train,val,test}.parquet` under `root`; raise unless all exist."""
    split_dir = root / "temporal_split"
    paths = (
        split_dir / "train.parquet",
        split_dir / "val.parquet",
        split_dir / "test.parquet",
    )
    if not all(path.exists() for path in paths):
        raise SchemaError(
            f"precomputed TabFormer splits require train/val/test parquet under {split_dir}."
        )
    return paths


def precomputed_sample(root: Path, *, cap: int, seed: int) -> list[object]:
    """The users `load_tabformer_adapter` samples from precomputed splits.

    `cap` and `seed` are `entity_cap(config)` and `sample_seed(config, seed)`.
    """
    train_path, _val_path, test_path = precomputed_split_paths(root)
    return _sample_ids(
        _read_user_ids(train_path),
        cap=cap,
        seed=seed,
        positives=_fraud_positive_ids(test_path, train_path),
    )


def precomputed_train_table(root: Path, users: Sequence[object]) -> pa.Table:
    """`users`' rows of `train.parquet`, in the order the adapter reads them.

    Row `i` becomes the train-side event whose `event_id` ends in `-{i}`
    (`tabformer_row_event(row, index=i, ...)`), so a reader that keeps this
    order reproduces the adapter's event ids and their tie-breaking order.
    """
    return _filter_parquet_file(precomputed_split_paths(root)[0], users)


def tabformer_row_event(
    row: dict[str, Any], *, index: int, label_delay: timedelta
) -> TransactionEvent:
    """One TabFormer row as the adapter maps it."""
    return _row_to_event(row, index=index, label_delay=label_delay)


def tabformer_label_delay(config: DataConfig) -> timedelta:
    return _label_delay(config)


def load_tabformer_adapter(root: Path, config: DataConfig, *, seed: int) -> AdapterResult:
    delay = _label_delay(config)
    cap = entity_cap(config)
    rng_seed = sample_seed(config, seed)
    split_dir = root / "temporal_split"
    protocol = config.split_protocol.strip().lower()
    train_path = split_dir / "train.parquet"
    val_path = split_dir / "val.parquet"
    test_path = split_dir / "test.parquet"
    if protocol == "precomputed" and train_path.exists():
        users = precomputed_sample(root, cap=cap, seed=rng_seed)
        train_events = _table_to_events(precomputed_train_table(root, users), label_delay=delay)
        val_events = _table_to_events(_filter_parquet_file(val_path, users), label_delay=delay)
        test_events = _table_to_events(_filter_parquet_file(test_path, users), label_delay=delay)
        splits = SplitAssignment(tuple(train_events), tuple(val_events), tuple(test_events))
        return AdapterResult(
            events=tuple([*train_events, *val_events, *test_events]),
            splits=splits,
            metadata={
                "adapter": "tabformer",
                "split_protocol": "precomputed",
                "n_entities_sampled": len(users),
                "root": str(root),
            },
        )
    parquet_files = [
        path
        for path in (train_path, val_path, test_path, root / "raw" / "transactions.parquet")
        if path.exists()
    ]
    if parquet_files:
        users = _sample_ids(
            _union_user_ids(parquet_files),
            cap=cap,
            seed=rng_seed,
            positives=_fraud_positive_ids(*parquet_files),
        )
        table = pa.concat_tables([_filter_parquet_file(path, users) for path in parquet_files])
        events = _table_to_events(table, label_delay=delay)
        return AdapterResult(
            events=tuple(events),
            splits=None,
            metadata={
                "adapter": "tabformer",
                "split_protocol": "timestamp",
                "n_entities_sampled": len(users),
                "root": str(root),
            },
        )
    csv_path = root / "raw" / "card_transaction.v1.csv"
    if not csv_path.exists():
        raise SchemaError(
            f"No TabFormer parquet or CSV under {root}. Expected temporal_split/*.parquet "
            "or raw/card_transaction.v1.csv."
        )
    events, users = _load_tabformer_csv(csv_path, cap=cap, seed=rng_seed, delay=delay)
    return AdapterResult(
        events=tuple(events),
        splits=None,
        metadata={
            "adapter": "tabformer",
            "split_protocol": "timestamp",
            "n_entities_sampled": len(users),
            "root": str(root),
        },
    )


def _label_delay(config: DataConfig) -> timedelta:
    text = config.temporal_split.label_delay.strip().lower()
    if text.endswith("d"):
        return timedelta(days=int(text[:-1]))
    if text.endswith("h"):
        return timedelta(hours=int(text[:-1]))
    raise SchemaError(f"Unsupported label_delay {config.temporal_split.label_delay!r}.")
