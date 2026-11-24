# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and write the pragmatiq four-file parquet contract.

On-disk layout (Arrow dtypes from dynamiq-ai/pragmatiq):

- ``events.parquet``: user_id, ts timestamp[us], source, fields map<str,str>
- ``profiles.parquet``: user_id, as_of, attributes map<str,str>, lifelong
- ``transfers.parquet``: from_user, to_user, ts, amount (optional)
- ``labels/*.parquet``: task tables keyed by user_id plus a time column

Adapters map into TFM domain types. They do not replace ``TransactionEvent``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError
from tfm.contracts.schema import (
    EntityProfile,
    EventLabel,
    LifelongItem,
    TransactionEvent,
    TransferEdge,
    ensure_utc,
)
from tfm.data.sampling import stratified_sample_ids

SOURCES: tuple[str, ...] = ("transaction", "app", "trading", "communication")

EVENTS_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("fields", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)

PROFILES_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("as_of", pa.timestamp("us"), nullable=False),
        pa.field("attributes", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field(
            "lifelong",
            pa.list_(pa.struct([pa.field("key", pa.string()), pa.field("ts", pa.timestamp("us"))])),
            nullable=False,
        ),
    ]
)

TRANSFERS_SCHEMA = pa.schema(
    [
        pa.field("from_user", pa.string(), nullable=False),
        pa.field("to_user", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us"), nullable=False),
        pa.field("amount", pa.float64(), nullable=False),
    ]
)

# Keys the 12-field tokenizer reads, plus common pragmatiq aliases.
_FLAT_CATEGORICAL_ALIASES = {
    "merchant": "merchant_name",
    "merchant_name": "merchant_name",
    "merchant_state": "merchant_state",
    "state": "merchant_state",
    "use_chip": "use_chip",
    "mcc": "mcc",
}
_FLAT_NUMERIC_ALIASES = {
    "amount": "amount",
    "card": "card",
    "zip": "zip",
    "user": "user",
}

FORECAST_TASKS = frozenset({"default_12m", "churn_6m", "ltv_positive"})
EVENT_LABEL_TASKS = frozenset({"fraud", "recurring"})
# Skip comm_uplift (no `label` column) and unused giant event tables by default.
LOAD_LABEL_TASKS = FORECAST_TASKS | {"fraud", "aml"}


@dataclass(frozen=True)
class LabelRow:
    """One row from a pragmatiq labels/*.parquet table."""

    entity_id: str
    timestamp: datetime
    label: int
    task: str
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PragmatiqDataset:
    events: tuple[TransactionEvent, ...]
    profiles: tuple[EntityProfile, ...]
    transfers: tuple[TransferEdge, ...]
    labels: Mapping[str, tuple[LabelRow, ...]]
    dropped_sources: Mapping[str, int] = field(default_factory=dict)


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    raise SchemaError(f"Expected a timestamp, got {type(value).__name__}.")


def _map_to_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v is not None}
    # pyarrow MapScalar / list of key-value pairs
    items: Sequence[object]
    if hasattr(value, "items"):
        items = list(value.items())  # type: ignore[arg-type]
    else:
        items = list(value)  # type: ignore[arg-type]
    out: dict[str, str] = {}
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            key, val = item
        elif hasattr(item, "get"):
            key, val = item.get("key"), item.get("value")  # type: ignore[union-attr]
        else:
            continue
        if key is None or val is None:
            continue
        out[str(key)] = str(val)
    return out


def parse_float(text: str) -> float | None:
    """A finite float, or `None` for text that is not one (pragmatiq stores numbers as text)."""
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def split_fields(
    fields: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    """Split a string map into categorical / numeric / text views."""
    categorical: dict[str, str] = {}
    numeric: dict[str, float] = {}
    text: dict[str, str] = {}
    for key, raw in fields.items():
        parsed = parse_float(raw)
        if parsed is not None:
            numeric[key] = parsed
        elif len(raw) > 64:
            text[key] = raw
        else:
            categorical[key] = raw
    return categorical, numeric, text


def _event_id(user_id: str, ts: datetime, index: int) -> str:
    ts_us = int(ts.timestamp() * 1_000_000)
    return f"{user_id}:{ts_us}:{index}"


def events_table_from_records(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    user_ids = [str(r["user_id"]) for r in rows]
    ts = [_as_utc_datetime(r["ts"]) for r in rows]
    sources = [str(r["source"]) for r in rows]
    fields = [list(dict(r["fields"]).items()) for r in rows]
    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.string()),
            "ts": pa.array(ts, type=pa.timestamp("us")),
            "source": pa.array(sources, type=pa.string()),
            "fields": pa.array(fields, type=pa.map_(pa.string(), pa.string())),
        },
        schema=EVENTS_SCHEMA,
    )


def profiles_table_from_records(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    user_ids = [str(r["user_id"]) for r in rows]
    as_of = [_as_utc_datetime(r["as_of"]) for r in rows]
    attributes = [list(dict(r.get("attributes", {})).items()) for r in rows]
    lifelong = []
    for row in rows:
        items = []
        for item in row.get("lifelong", []):
            if isinstance(item, Mapping):
                items.append({"key": str(item["key"]), "ts": _as_utc_datetime(item["ts"])})
            else:
                items.append({"key": str(item[0]), "ts": _as_utc_datetime(item[1])})
        lifelong.append(items)
    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.string()),
            "as_of": pa.array(as_of, type=pa.timestamp("us")),
            "attributes": pa.array(attributes, type=pa.map_(pa.string(), pa.string())),
            "lifelong": pa.array(
                lifelong,
                type=pa.list_(
                    pa.struct([pa.field("key", pa.string()), pa.field("ts", pa.timestamp("us"))])
                ),
            ),
        },
        schema=PROFILES_SCHEMA,
    )


def transfers_table_from_records(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.table(
        {
            "from_user": pa.array([str(r["from_user"]) for r in rows], type=pa.string()),
            "to_user": pa.array([str(r["to_user"]) for r in rows], type=pa.string()),
            "ts": pa.array([_as_utc_datetime(r["ts"]) for r in rows], type=pa.timestamp("us")),
            "amount": pa.array([float(r["amount"]) for r in rows], type=pa.float64()),
        },
        schema=TRANSFERS_SCHEMA,
    )


def label_schema_for(task: str) -> pa.Schema:
    base = [pa.field("user_id", pa.string(), nullable=False)]
    if task in FORECAST_TASKS:
        cols = base + [
            pa.field("eval_ts", pa.timestamp("us"), nullable=False),
            pa.field("label", pa.int8(), nullable=False),
        ]
        if task == "ltv_positive":
            cols.append(pa.field("profit_6m", pa.float64(), nullable=False))
        return pa.schema(cols)
    if task == "aml":
        return pa.schema(
            base
            + [
                pa.field("observed_through", pa.timestamp("us"), nullable=False),
                pa.field("label", pa.int8(), nullable=False),
            ]
        )
    if task == "fraud":
        return pa.schema(
            base
            + [
                pa.field("ts", pa.timestamp("us"), nullable=False),
                pa.field("label", pa.int8(), nullable=False),
            ]
        )
    if task == "recurring":
        return pa.schema(
            base
            + [
                pa.field("ts", pa.timestamp("us"), nullable=False),
                pa.field("series_id", pa.string(), nullable=False),
                pa.field("label", pa.int8(), nullable=False),
            ]
        )
    raise SchemaError(f"Unknown pragmatiq label task {task!r}.")


def write_pragmatiq_dataset(path: Path | str, dataset: Mapping[str, Any]) -> Path:
    """Write a dict of tables/records to the four-file contract."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    events = dataset["events"]
    if isinstance(events, pa.Table):
        pq.write_table(events, root / "events.parquet")
    else:
        pq.write_table(events_table_from_records(events), root / "events.parquet")
    profiles = dataset.get("profiles", [])
    if isinstance(profiles, pa.Table):
        pq.write_table(profiles, root / "profiles.parquet")
    else:
        pq.write_table(profiles_table_from_records(profiles), root / "profiles.parquet")
    transfers = dataset.get("transfers", [])
    if isinstance(transfers, pa.Table):
        pq.write_table(transfers, root / "transfers.parquet")
    elif transfers:
        pq.write_table(transfers_table_from_records(transfers), root / "transfers.parquet")
    labels = dataset.get("labels", {})
    if labels:
        label_dir = root / "labels"
        label_dir.mkdir(exist_ok=True)
        for task, rows in labels.items():
            schema = label_schema_for(task)
            if isinstance(rows, pa.Table):
                pq.write_table(rows, label_dir / f"{task}.parquet")
                continue
            arrays: dict[str, pa.Array] = {}
            for field in schema:
                values = [row[field.name] for row in rows]
                if pa.types.is_timestamp(field.type):
                    values = [_as_utc_datetime(v) for v in values]
                arrays[field.name] = pa.array(values, type=field.type)
            pq.write_table(pa.table(arrays, schema=schema), label_dir / f"{task}.parquet")
    return root


def generate_pragmatiq_fixture(
    path: Path | str,
    *,
    n_users: int = 4,
    events_per_user: int = 6,
    seed: int = 0,
) -> Path:
    """Write a tiny synthetic pragmatiq book (no real customer data)."""
    from random import Random

    rng = Random(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    fraud_rows: list[dict[str, Any]] = []
    default_rows: list[dict[str, Any]] = []
    merchants = ("TESCO", "ACME MART", "FUEL STOP")
    for user_i in range(n_users):
        user_id = f"u{user_i:04d}"
        opened = start
        profiles.append(
            {
                "user_id": user_id,
                "as_of": datetime(2024, 6, 1, tzinfo=UTC),
                "attributes": {"country": "GB", "plan": "plus"},
                "lifelong": [("account_opened", opened)],
            }
        )
        default_rows.append(
            {
                "user_id": user_id,
                "eval_ts": datetime(2024, 3, 1, tzinfo=UTC),
                "label": int(user_i % 5 == 0),
            }
        )
        for event_i in range(events_per_user):
            ts = datetime(2024, 1, 1 + event_i, 12, tzinfo=UTC)
            if event_i % 3 == 2:
                source = "app"
                fields = {"screen": "home", "action": "open", "os": "ios"}
            else:
                source = "transaction"
                amount = round(rng.uniform(5.0, 80.0), 2)
                is_fraud = int(rng.random() < 0.2)
                if is_fraud:
                    amount = round(amount * 4.0, 2)
                fields = {
                    "amount": str(amount),
                    "currency": "GBP",
                    "mcc": "5411",
                    "merchant": rng.choice(merchants),
                    "channel": "CHIP TRANSACTION",
                }
                fraud_rows.append({"user_id": user_id, "ts": ts, "label": is_fraud})
            events.append({"user_id": user_id, "ts": ts, "source": source, "fields": fields})
        if user_i > 0:
            transfers.append(
                {
                    "from_user": f"u{user_i - 1:04d}",
                    "to_user": user_id,
                    "ts": datetime(2024, 2, 1, tzinfo=UTC),
                    "amount": 25.0,
                }
            )
    return write_pragmatiq_dataset(
        path,
        {
            "events": events,
            "profiles": profiles,
            "transfers": transfers,
            "labels": {"fraud": fraud_rows, "default_12m": default_rows},
        },
    )


def _validate_events_table(table: pa.Table) -> None:
    missing = [name for name in EVENTS_SCHEMA.names if name not in table.schema.names]
    if missing:
        raise SchemaError(f"events.parquet missing columns {missing}.")
    sources = set(table.column("source").to_pylist())
    unknown = sources - set(SOURCES)
    if unknown:
        raise SchemaError(
            f"Unknown event sources {sorted(unknown)}; expected one of {list(SOURCES)}."
        )
    user_ids = table.column("user_id").to_pylist()
    if any(uid is None or str(uid).strip() == "" for uid in user_ids):
        raise SchemaError("events.parquet has null or empty user_id values.")
    timestamps = table.column("ts").to_pylist()
    if any(ts is None for ts in timestamps):
        raise SchemaError("events.parquet has null timestamps.")


def _load_events(table: pa.Table) -> tuple[list[TransactionEvent], dict[str, int]]:
    _validate_events_table(table)
    rows = table.to_pylist()
    rows.sort(key=lambda r: (str(r["user_id"]), _as_utc_datetime(r["ts"])))
    events: list[TransactionEvent] = []
    per_user_index: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        user_id = str(row["user_id"])
        ts = ensure_utc(_as_utc_datetime(row["ts"]))
        source = str(row["source"])
        idx = per_user_index.get(user_id, 0)
        per_user_index[user_id] = idx + 1
        fields = _map_to_dict(row.get("fields"))
        categorical, numeric, text = split_fields(fields)
        categorical["source"] = source
        events.append(
            TransactionEvent(
                entity_id=user_id,
                event_id=_event_id(user_id, ts, idx),
                event_timestamp=ts,
                timezone="UTC",
                categorical=categorical,
                numeric=numeric,
                text=text,
            )
        )
        source_counts[source] = source_counts.get(source, 0) + 1
    return events, {k: v for k, v in source_counts.items() if k != "transaction"}


def load_profiles(table: pa.Table) -> list[EntityProfile]:
    """`EntityProfile`s from a pragmatiq `profiles` table, one per row."""
    profiles: list[EntityProfile] = []
    for row in table.to_pylist():
        lifelong_items: list[LifelongItem] = []
        for item in row.get("lifelong") or []:
            if isinstance(item, Mapping):
                lifelong_items.append(
                    LifelongItem(key=str(item["key"]), timestamp=_as_utc_datetime(item["ts"]))
                )
            else:
                lifelong_items.append(
                    LifelongItem(key=str(item[0]), timestamp=_as_utc_datetime(item[1]))
                )
        profiles.append(
            EntityProfile(
                entity_id=str(row["user_id"]),
                as_of=_as_utc_datetime(row["as_of"]),
                attributes=_map_to_dict(row.get("attributes")),
                lifelong=tuple(lifelong_items),
            )
        )
    return profiles


def _load_transfers(table: pa.Table) -> list[TransferEdge]:
    return [
        TransferEdge(
            from_entity=str(row["from_user"]),
            to_entity=str(row["to_user"]),
            timestamp=_as_utc_datetime(row["ts"]),
            amount=float(row["amount"]),
        )
        for row in table.to_pylist()
    ]


def _load_label_table(task: str, table: pa.Table) -> list[LabelRow]:
    if "label" not in table.column_names:
        warnings.warn(
            f"Skipping labels/{task}.parquet: no 'label' column (columns={table.column_names}).",
            stacklevel=2,
        )
        return []
    rows: list[LabelRow] = []
    for raw in table.to_pylist():
        if "eval_ts" in raw:
            ts = _as_utc_datetime(raw["eval_ts"])
        elif "observed_through" in raw:
            ts = _as_utc_datetime(raw["observed_through"])
        elif "ts" in raw:
            ts = _as_utc_datetime(raw["ts"])
        else:
            raise SchemaError(f"Label table {task!r} has no time column.")
        extra = {
            k: v
            for k, v in raw.items()
            if k not in {"user_id", "ts", "eval_ts", "observed_through", "label"}
        }
        rows.append(
            LabelRow(
                entity_id=str(raw["user_id"]),
                timestamp=ts,
                label=int(raw["label"]),
                task=task,
                extra=extra,
            )
        )
    return rows


def attach_event_labels(
    events: Sequence[TransactionEvent],
    rows: Sequence[LabelRow],
    *,
    name: str = "fraud",
    label_delay_hours: int = 0,
) -> list[TransactionEvent]:
    """Join event-level label rows onto matching (entity_id, timestamp) events."""
    from datetime import timedelta

    delay = timedelta(hours=label_delay_hours)
    lookup: dict[tuple[str, datetime], LabelRow] = {
        (row.entity_id, ensure_utc(row.timestamp)): row for row in rows
    }
    out: list[TransactionEvent] = []
    for event in events:
        row = lookup.get((event.entity_id, event.event_timestamp))
        if row is None:
            out.append(event)
            continue
        labels = dict(event.labels)
        labels[name] = EventLabel(
            name=name,
            value=float(row.label),
            available_at=event.event_timestamp + delay,
            level="event",
        )
        out.append(
            TransactionEvent(
                entity_id=event.entity_id,
                event_id=event.event_id,
                event_timestamp=event.event_timestamp,
                timezone=event.timezone,
                categorical=event.categorical,
                numeric=event.numeric,
                text=event.text,
                labels=labels,
                group=event.group,
            )
        )
    return out


def events_for_flat_fields(events: Sequence[TransactionEvent]) -> list[TransactionEvent]:
    """Keep transaction-source events and alias fields for the 12-field tokenizer.

    App / trading / communication events are dropped with a warning: the
    ``flat_fields`` tokenizer cannot represent heterogeneous keys.
    """
    kept: list[TransactionEvent] = []
    dropped: dict[str, int] = {}
    for event in events:
        source = event.categorical.get("source", "transaction")
        if source != "transaction":
            dropped[source] = dropped.get(source, 0) + 1
            continue
        categorical = dict(event.categorical)
        numeric = dict(event.numeric)
        for src, dest in _FLAT_CATEGORICAL_ALIASES.items():
            if src in categorical and dest not in categorical:
                categorical[dest] = categorical[src]
            if src in event.text and dest not in categorical:
                categorical[dest] = event.text[src]
        if "channel" in categorical and "use_chip" not in categorical:
            categorical["use_chip"] = categorical["channel"]
        for src, dest in _FLAT_NUMERIC_ALIASES.items():
            if src in numeric and dest not in numeric:
                numeric[dest] = numeric[src]
        if "user" not in numeric:
            numeric["user"] = float(abs(int.from_bytes(event.entity_id.encode(), "little")) % 3000)
        kept.append(
            TransactionEvent(
                entity_id=event.entity_id,
                event_id=event.event_id,
                event_timestamp=event.event_timestamp,
                timezone=event.timezone,
                categorical=categorical,
                numeric=numeric,
                text=event.text,
                labels=event.labels,
                group=event.group,
            )
        )
    if dropped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        warnings.warn(
            "Dropped non-transaction events that the 12-field tokenizer cannot "
            f"represent ({detail}). Use the key-value-time tokenizer for mixed sources.",
            stacklevel=2,
        )
    return kept


def load_pragmatiq_dataset(
    path: Path | str,
    *,
    max_entities: int | None = None,
    sample_seed: int = 0,
    attach_fraud_labels: bool = True,
    load_labels: bool = True,
    stratify_by_fraud: bool = True,
) -> PragmatiqDataset:
    """Read a pragmatiq directory into TFM domain objects.

    When ``max_entities`` is set, user ids are sampled first so a large book
    (e.g. synth_1m) is not fully materialized; ``stratify_by_fraud`` fills up
    to half the sample with users who have the book's own fraud. With
    ``attach_fraud_labels=False`` the fraud table is still returned in
    ``labels`` but no event carries it.
    """
    root = Path(path)
    events_path = root / "events.parquet"
    if not events_path.exists():
        raise SchemaError(f"Missing {events_path}.")
    selected: list[str] | None = None
    profiles_path = root / "profiles.parquet"
    if max_entities is not None:
        selected = _sample_pragmatiq_users(
            root,
            cap=int(max_entities),
            seed=int(sample_seed),
            stratify=stratify_by_fraud and load_labels,
        )
    events_table = _read_events_table(events_path, selected)
    events, dropped = _load_events(events_table)
    profiles = (
        load_profiles(_read_user_table(profiles_path, selected, "user_id"))
        if profiles_path.exists()
        else []
    )
    transfers_path = root / "transfers.parquet"
    transfers = (
        _load_transfers(_read_user_table(transfers_path, selected, "from_user"))
        if transfers_path.exists()
        else []
    )
    labels: dict[str, tuple[LabelRow, ...]] = {}
    label_dir = root / "labels"
    if load_labels and label_dir.is_dir():
        for parquet in sorted(label_dir.glob("*.parquet")):
            task = parquet.stem
            if task not in LOAD_LABEL_TASKS:
                continue
            table = _read_user_table(parquet, selected, "user_id")
            loaded = tuple(_load_label_table(task, table))
            if loaded:
                labels[task] = loaded
    if attach_fraud_labels and "fraud" in labels:
        events = attach_event_labels(events, labels["fraud"], name="fraud")
    return PragmatiqDataset(
        events=tuple(events),
        profiles=tuple(profiles),
        transfers=tuple(transfers),
        labels=labels,
        dropped_sources=dropped,
    )


def _fraud_positive_users(root: Path) -> list[str]:
    path = root / "labels" / "fraud.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["user_id", "label"])
    users = table.column("user_id").to_pylist()
    flags = table.column("label").to_pylist()
    return list(
        dict.fromkeys(str(user) for user, flag in zip(users, flags, strict=True) if int(flag) == 1)
    )


def fraud_positive_users(labels: Mapping[str, Sequence[LabelRow]]) -> frozenset[str]:
    """Users with at least one positive row in the book's own fraud table."""
    return frozenset(row.entity_id for row in labels.get("fraud", ()) if int(row.label) == 1)


def _sample_pragmatiq_users(root: Path, *, cap: int, seed: int, stratify: bool = True) -> list[str]:
    from random import Random

    profiles_path = root / "profiles.parquet"
    if profiles_path.exists():
        users = pq.read_table(profiles_path, columns=["user_id"]).column("user_id").to_pylist()
        unique = list(dict.fromkeys(str(u) for u in users))
    else:
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(root / "events.parquet"), format="parquet")
        unique = [
            str(u)
            for u in dataset.to_table(columns=["user_id"]).column("user_id").unique().to_pylist()
        ]
    if cap <= 0:
        raise SchemaError("max_entities must be positive when sampling pragmatiq users.")
    positives = _fraud_positive_users(root) if stratify else []
    if positives:
        return [str(u) for u in stratified_sample_ids(unique, positives, cap=cap, seed=seed)]
    rng = Random(seed)
    rng.shuffle(unique)
    return unique[: min(cap, len(unique))]


def _filter_user_table(table: pa.Table, selected: list[str] | None, column: str) -> pa.Table:
    if selected is None or column not in table.column_names:
        return table
    values = pa.array(selected, type=pa.string())
    mask = pc.is_in(  # type: ignore[attr-defined]
        table.column(column).cast(pa.string()), value_set=values
    )
    return table.filter(mask)


def _read_user_table(path: Path, selected: list[str] | None, column: str) -> pa.Table:
    if selected is None:
        return pq.read_table(path)
    try:
        return pq.read_table(path, filters=[(column, "in", selected)])
    except (NotImplementedError, OSError, TypeError, pa.ArrowInvalid):
        return _filter_user_table(pq.read_table(path), selected, column)


def _read_events_table(events_path: Path, selected: list[str] | None) -> pa.Table:
    return _read_user_table(events_path, selected, "user_id")
