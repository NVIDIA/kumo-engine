# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project transaction books into scalar source tables in bounded blocks.

This is ingestion, not window construction: a source event is stored once,
regardless of the number of prediction anchors. Static profile attributes and
separately dated milestones retain the existing EntityProfile contract.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tfm.config import DataConfig
from tfm.contracts.errors import SchemaError
from tfm.contracts.graph import utc_us
from tfm.contracts.relational import RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource
from tfm.data.adapters.paths import ADAPTER_ENV, entity_cap, resolve_data_path, sample_seed
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
CLOCK = pa.timestamp("us", tz="UTC")
NUMERIC_KEYS = frozenset({"amount", "quantity", "price", "card", "zip", "user"})
TEXT_KEYS = frozenset({"text", "body", "message", "description", "note"})
SOURCES = frozenset({"transaction", "app", "trading", "communication"})


def _safe(keys: Iterator[str] | Mapping[str, Any]) -> None:
    for key in keys:
        if key.strip().casefold() in PROHIBITED_INPUT_KEYS:
            raise SchemaError(f"Input field {key!r} looks like a label and cannot be a model input")


def _stamp(value: Any) -> int:
    if not isinstance(value, datetime):
        raise SchemaError("Transaction source clocks must be datetime values")
    return utc_us(value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC))


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    for batch in ds.dataset(path, format="parquet").scanner(batch_size=4096).to_batches():
        yield from batch.to_pylist()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class _ScalarBook(AbstractContextManager["_ScalarBook"]):
    def __init__(self, directory: Path, amount_key: str) -> None:
        _safe(iter((amount_key,)))
        self.directory, self.amount_key = directory, amount_key
        directory.mkdir(parents=True, exist_ok=True)
        self.resources = ExitStack()
        temporary = Path(
            self.resources.enter_context(TemporaryDirectory(prefix="projection-", dir=directory))
        )
        self.db = sqlite3.connect(temporary / "records.sqlite")
        self.resources.callback(self.db.close)
        self.db.executescript("""
            CREATE TABLE users (id TEXT PRIMARY KEY,attributes TEXT);
            CREATE TABLE events (id TEXT PRIMARY KEY,owner TEXT,at INTEGER,source TEXT,numbers TEXT,categories TEXT,texts TEXT,fraud REAL,known INTEGER,partition TEXT);
            CREATE TABLE facts (id TEXT PRIMARY KEY,owner TEXT,at INTEGER,fact TEXT);
            CREATE TABLE transfers (id TEXT PRIMARY KEY,sender TEXT,receiver TEXT,at INTEGER,amount REAL);
            CREATE TABLE fraud (owner TEXT,at INTEGER,value REAL,PRIMARY KEY(owner,at));
        """)
        self.numeric: set[str] = set()
        self.categorical: set[str] = set()
        self.text: set[str] = set()
        self.profile: set[str] = set()
        self.buffer: list[tuple[Any, ...]] = []
        self.count = 0

    def event(
        self,
        owner: str,
        event_id: str,
        at: int,
        source: str,
        numbers: Mapping[str, Any],
        categories: Mapping[str, Any],
        texts: Mapping[str, Any],
        fraud: float | None,
        known: int | None,
        partition: str | None = None,
    ) -> None:
        _safe(numbers)
        _safe(categories)
        _safe(texts)
        if not owner.strip() or not event_id.strip():
            raise SchemaError("Source events require nonempty entity and event IDs")
        if (fraud is None) != (known is None):
            raise SchemaError("Fraud availability must be supplied with its label value")
        if known is not None and known < at:
            raise SchemaError("An event fraud label cannot be available before its event")
        if fraud is not None and (fraud not in (0, 1) or known is None):
            raise SchemaError("Event fraud labels require binary values and an availability clock")
        numeric = {key: _finite(value) for key, value in numbers.items()}
        cats = {key: str(value) for key, value in categories.items() if value is not None}
        cats["source"] = source
        text = {key: str(value) for key, value in texts.items() if value is not None}
        self.numeric.update(numeric)
        self.categorical.update(cats)
        self.text.update(text)
        self.buffer.append(
            (
                event_id,
                owner,
                at,
                source,
                json.dumps(numeric),
                json.dumps(cats),
                json.dumps(text),
                fraud,
                known,
                partition,
            )
        )
        self.count += 1
        if len(self.buffer) == 4096:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", self.buffer)
            self.db.executemany(
                "INSERT OR IGNORE INTO users(id) VALUES (?)", ((row[1],) for row in self.buffer)
            )
            self.buffer.clear()
        self.db.commit()

    def profile_row(self, row: Mapping[str, Any], selected: set[str] | None) -> None:
        owner = str(row["user_id"])
        if selected is not None and owner not in selected:
            return
        attrs = dict(row.get("attributes") or ())
        _safe(attrs)
        attrs = {str(key): str(value) for key, value in attrs.items() if value is not None}
        self.profile.update(attrs)
        encoded = json.dumps(attrs, sort_keys=True)
        previous = self.db.execute("SELECT attributes FROM users WHERE id=?", (owner,)).fetchone()
        if previous is not None and previous[0] is not None:
            raise SchemaError(
                "Native EntityProfile projection requires one static snapshot per entity"
            )
        self.db.execute(
            "INSERT INTO users VALUES (?,?) ON CONFLICT(id) DO UPDATE SET attributes=excluded.attributes",
            (owner, encoded),
        )
        for index, item in enumerate(row.get("lifelong") or ()):
            key, at = (
                (str(item["key"]), _stamp(item["ts"]))
                if isinstance(item, Mapping)
                else (str(item[0]), _stamp(item[1]))
            )
            _safe(iter((key,)))
            self.db.execute(
                "INSERT INTO facts VALUES (?,?,?,?)", (f"{owner}/{index}", owner, at, key)
            )

    def tables(self, config: DataConfig) -> TableSource:
        self.flush()
        if not self.count:
            raise SchemaError("Native source projection contains no selected events")
        if self.amount_key not in self.numeric:
            raise SchemaError(
                f"Configured amount_field {self.amount_key!r} is absent from source numeric fields"
            )
        if (
            self.numeric & self.categorical
            or self.numeric & self.text
            or self.categorical & self.text
        ):
            raise SchemaError(
                "Native source feature types must be consistent; declare mixed fields in a custom TableSource adapter"
            )
        # Drop profile-only roots, matching the transaction book owner population.
        self.db.execute("DELETE FROM users WHERE id NOT IN (SELECT DISTINCT owner FROM events)")
        self.db.execute("DELETE FROM facts WHERE owner NOT IN (SELECT id FROM users)")
        self.db.execute(
            "DELETE FROM transfers WHERE sender NOT IN (SELECT id FROM users) OR receiver NOT IN (SELECT id FROM users)"
        )
        self.db.commit()
        numeric = {name: f"n{i}" for i, name in enumerate(sorted(self.numeric))}
        categorical = {name: f"c{i}" for i, name in enumerate(sorted(self.categorical))}
        texts = {name: f"x{i}" for i, name in enumerate(sorted(self.text))}
        profiles = {name: f"p{i}" for i, name in enumerate(sorted(self.profile))}
        schemas = {
            "users": pa.schema(
                [
                    ("id", pa.string()),
                    ("bias", pa.float32()),
                    *((column, pa.string()) for column in profiles.values()),
                ]
            ),
            "events": pa.schema(
                [
                    ("id", pa.string()),
                    ("owner", pa.string()),
                    ("at", CLOCK),
                    *((column, pa.float64()) for column in numeric.values()),
                    *((column, pa.string()) for column in (*categorical.values(), *texts.values())),
                    ("amount_log", pa.float32()),
                    ("amount_present", pa.float32()),
                    ("fraud_value", pa.float32()),
                    ("fraud_available_us", pa.int64()),
                    ("partition", pa.string()),
                ]
            ),
            "transfers": pa.schema(
                [
                    ("id", pa.string()),
                    ("sender", pa.string()),
                    ("receiver", pa.string()),
                    ("at", CLOCK),
                    ("amount", pa.float64()),
                    ("amount_log", pa.float32()),
                ]
            ),
        }
        if self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]:
            schemas["facts"] = pa.schema(
                [("id", pa.string()), ("owner", pa.string()), ("at", CLOCK), ("fact", pa.string())]
            )
        files = {name: self.directory / f"{name}.parquet" for name in schemas}

        def output(name: str) -> Iterator[dict[str, Any]]:
            if name == "users":
                for owner, raw in self.db.execute("SELECT id,attributes FROM users ORDER BY id"):
                    attrs = {} if raw is None else json.loads(raw)
                    yield {
                        "id": owner,
                        "bias": 1.0,
                        **{column: attrs.get(key) for key, column in profiles.items()},
                    }
            elif name == "events":
                for (
                    event_id,
                    owner,
                    at,
                    _,
                    raw_num,
                    raw_cat,
                    raw_txt,
                    fraud,
                    known,
                    partition,
                ) in self.db.execute("SELECT * FROM events ORDER BY rowid"):
                    nums, cats, txt = json.loads(raw_num), json.loads(raw_cat), json.loads(raw_txt)
                    amount = nums.get(self.amount_key)
                    yield {
                        "id": event_id,
                        "owner": owner,
                        "at": EPOCH + timedelta(microseconds=at),
                        **{column: nums.get(key) for key, column in numeric.items()},
                        **{column: cats.get(key) for key, column in categorical.items()},
                        **{column: txt.get(key) for key, column in texts.items()},
                        "amount_log": 0.0
                        if amount is None
                        else math.copysign(math.log1p(abs(amount)), amount),
                        "amount_present": float(amount is not None),
                        "fraud_value": fraud,
                        "fraud_available_us": known,
                        "partition": partition,
                    }
            elif name == "transfers":
                for event_id, sender, receiver, at, amount in self.db.execute(
                    "SELECT * FROM transfers ORDER BY at,sender,receiver,amount,id"
                ):
                    if amount is None or not math.isfinite(amount):
                        raise SchemaError("Transfer input amounts must be finite")
                    yield {
                        "id": event_id,
                        "sender": sender,
                        "receiver": receiver,
                        "at": EPOCH + timedelta(microseconds=at),
                        "amount": amount,
                        "amount_log": math.copysign(math.log1p(abs(amount)), amount),
                    }
            else:
                for event_id, owner, at, fact in self.db.execute(
                    "SELECT * FROM facts ORDER BY rowid"
                ):
                    yield {
                        "id": event_id,
                        "owner": owner,
                        "at": EPOCH + timedelta(microseconds=at),
                        "fact": fact,
                    }

        for name, schema in schemas.items():
            with pq.ParquetWriter(files[name], schema) as writer:
                buffer = []
                for row in output(name):
                    buffer.append(row)
                    if len(buffer) == 4096:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                        buffer.clear()
                if buffer:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        tables = [
            RelationalTable("users", ("id",), categorical=profiles, immutable=True),
            RelationalTable(
                "events",
                ("id",),
                numeric=numeric,
                categorical=categorical,
                text=texts,
                event_time="at",
                available_at="at",
            ),
            RelationalTable(
                "transfers",
                ("id",),
                numeric={"amount": "amount"},
                event_time="at",
                available_at="at",
            ),
        ]
        roles = [
            RelationalForeignKey("events", "users", "events", ("owner",)),
            RelationalForeignKey("sent", "users", "transfers", ("sender",)),
            RelationalForeignKey("received", "users", "transfers", ("receiver",)),
        ]
        paths = {}
        if "facts" in schemas:
            tables.append(
                RelationalTable(
                    "facts",
                    ("id",),
                    categorical={"milestone": "fact"},
                    event_time="at",
                    available_at="at",
                )
            )
            roles.append(RelationalForeignKey("facts", "users", "facts", ("owner",)))
            paths["milestones"] = ("facts",)
        return TableSource(
            tuple(tables),
            tuple(roles),
            files,
            graph_numeric={
                "users": {"bias": "bias"},
                "events": {"amount_log": "amount_log", "present": "amount_present"},
                "transfers": {"amount_log": "amount_log"},
            },
            profile_paths=paths,
            event_labels={"events": ("fraud_value", "fraud_available_us")},
            future_numeric={
                "events": {"amount": self.amount_key},
                "transfers": {"amount": "amount"},
            },
            observed_through=parse_timestamp(config.observed_through),
        )

    def __exit__(self, *args: Any) -> None:
        self.resources.close()


def synthetic_tables(config: DataConfig, *, seed: int, fixture_dir: Path) -> TableSource:
    """Generate one source event at a time; no Python history corpus is retained."""
    from tfm.data.synthetic.generator import iter_synthetic_events

    with _ScalarBook(fixture_dir, config.amount_field) as book:
        for event in iter_synthetic_events(
            seed=seed,
            n_entities=config.n_entities,
            events_per_entity=config.events_per_entity,
            step_hours=config.event_step_hours,
            label_delay=parse_duration(config.temporal_split.label_delay),
        ):
            label = event.labels["fraud"]
            book.event(
                event.entity_id,
                event.event_id,
                utc_us(event.event_timestamp),
                "transaction",
                event.numeric,
                event.categorical,
                event.text,
                label.value if config.load_labels else None,
                utc_us(label.available_at) if config.load_labels else None,
            )
        book.flush()
        book.db.execute("""
            INSERT INTO transfers SELECT 'transfer/'||e.rowid,e.owner,u.next,e.at,json_extract(e.numbers,'$.amount')
            FROM events e JOIN (SELECT id,COALESCE(LEAD(id) OVER(ORDER BY id),(SELECT MIN(id) FROM users)) AS next FROM users) u ON u.id=e.owner
            WHERE CAST(strftime('%d',e.at/1000000,'unixepoch') AS INTEGER)%7=0
        """)
        return book.tables(config)


def _sample_book_users(root: Path, *, cap: int, seed: int, stratify: bool) -> list[str]:
    """Read bounded key/label columns; retain unique owners, not all label rows."""
    from random import Random

    from tfm.data.sampling import stratified_sample_ids

    profiles = root / "profiles.parquet"
    source = profiles if profiles.exists() else root / "events.parquet"
    population: dict[str, None] = {}
    for block in (
        ds.dataset(source, format="parquet")
        .scanner(columns=["user_id"], batch_size=4096)
        .to_batches()
    ):
        for owner in block.column(0).to_pylist():
            population.setdefault(str(owner), None)
    positives: dict[str, None] = {}
    labels = root / "labels/fraud.parquet"
    if stratify and labels.exists():
        for block in pq.ParquetFile(labels).iter_batches(
            batch_size=4096, columns=["user_id", "label"]
        ):
            for owner, label in zip(
                block.column(0).to_pylist(), block.column(1).to_pylist(), strict=True
            ):
                if int(label) == 1:
                    positives.setdefault(str(owner), None)
    if cap <= 0:
        raise SchemaError("max_entities must be positive when sampling book owners")
    if positives:
        return [
            str(owner)
            for owner in stratified_sample_ids(
                tuple(population), tuple(positives), cap=cap, seed=seed
            )
        ]
    owners = list(population)
    Random(seed).shuffle(owners)
    return owners[: min(cap, len(owners))]


def pragmatiq_tables(config: DataConfig, *, seed: int, fixture_dir: Path) -> TableSource:
    """Scan map-valued Parquet in bounded blocks and expose typed scalar columns."""
    from tfm.data.adapters.pragmatiq import (
        _map_to_dict,
        generate_pragmatiq_fixture,
    )

    if config.path or os.environ.get(ADAPTER_ENV["pragmatiq"]):
        root = resolve_data_path(config)
        selected = set(
            _sample_book_users(
                root,
                cap=entity_cap(config),
                seed=sample_seed(config, seed),
                stratify=config.stratify_by_fraud and config.load_labels,
            )
        )
    else:
        root = generate_pragmatiq_fixture(
            fixture_dir / "book",
            n_users=max(config.n_entities, 8),
            events_per_user=max(config.events_per_entity, 6),
            seed=seed,
        )
        selected = None
    with _ScalarBook(fixture_dir, config.amount_field) as book:
        fraud = root / "labels/fraud.parquet"
        if config.load_labels and fraud.exists():
            for row in _rows(fraud):
                owner = str(row["user_id"])
                if selected is None or owner in selected:
                    value = float(row["label"])
                    if value not in (0, 1):
                        raise SchemaError("Pragmatiq fraud labels must be binary")
                    book.db.execute(
                        "INSERT OR REPLACE INTO fraud VALUES (?,?,?)",
                        (owner, _stamp(row["ts"]), value),
                    )
        for index, row in enumerate(_rows(root / "events.parquet")):
            owner = str(row["user_id"])
            if selected is not None and owner not in selected:
                continue
            source = str(row["source"])
            if source not in SOURCES:
                continue
            fields = _map_to_dict(row["fields"])
            _safe(fields)
            numeric = {
                key: _finite(value)
                for key, value in fields.items()
                if key in NUMERIC_KEYS or key == config.amount_field
            }
            texts = {key: value for key, value in fields.items() if key in TEXT_KEYS}
            cats = {
                key: value
                for key, value in fields.items()
                if key not in numeric and key not in texts
            }
            at = _stamp(row["ts"])
            label = book.db.execute(
                "SELECT value FROM fraud WHERE owner=? AND at=?", (owner, at)
            ).fetchone()
            book.event(
                owner,
                f"{owner}/{at}/{index:020d}",
                at,
                source,
                numeric,
                cats,
                texts,
                None if label is None else label[0],
                None if label is None else at,
            )
        book.flush()
        profiles = root / "profiles.parquet"
        if profiles.exists():
            for row in _rows(profiles):
                book.profile_row(row, selected)
        transfers = root / "transfers.parquet"
        if transfers.exists():
            for index, row in enumerate(_rows(transfers)):
                sender, receiver = str(row["from_user"]), str(row["to_user"])
                if selected is None or sender in selected and receiver in selected:
                    book.db.execute(
                        "INSERT INTO transfers VALUES (?,?,?,?,?)",
                        (
                            f"transfer/{index}",
                            sender,
                            receiver,
                            _stamp(row["ts"]),
                            _finite(row["amount"]),
                        ),
                    )
        result = book.tables(config)
        if config.load_labels:
            from tfm.data.adapters.pragmatiq import LOAD_LABEL_TASKS

            tables, files = list(result.tables), dict(result.files)
            for task in sorted(LOAD_LABEL_TASKS - {"fraud"}):
                path = root / "labels" / f"{task}.parquet"
                if not path.exists():
                    continue
                schema = pq.read_schema(path)
                if "label" not in schema.names:
                    continue
                clock = next(
                    (
                        name
                        for name in ("eval_ts", "observed_through", "ts")
                        if name in schema.names
                    ),
                    None,
                )
                if clock is None:
                    raise SchemaError(f"Label table {task!r} has no prediction time column")
                name = f"labels_{task}"
                tables.append(
                    RelationalTable(name, ("user_id", clock), immutable=True, role="labels")
                )
                files[name] = path
            result = replace(result, tables=tuple(tables), files=files)
        return result


def tabformer_tables(config: DataConfig, *, seed: int, fixture_dir: Path) -> TableSource:
    """Scan a local card book into one scalar row per original transaction.

    Owner selection retains the object adapter's sampling order. No history or
    model minibatch is constructed; the partition column is withheld metadata.
    """
    from hashlib import sha256

    import pyarrow.csv as csv

    from tfm.data.adapters.tabformer import (
        _as_datetime,
        _is_fraud_value,
        _parse_amount,
        _parse_hour,
        _sample_ids,
        precomputed_split_paths,
        tabformer_label_delay,
    )

    root = resolve_data_path(config)
    paths: tuple[Path, ...]
    if config.split_protocol == "precomputed":
        paths = precomputed_split_paths(root)
    else:
        paths = tuple(
            path
            for path in (
                root / "temporal_split/train.parquet",
                root / "temporal_split/val.parquet",
                root / "temporal_split/test.parquet",
                root / "raw/transactions.parquet",
            )
            if path.exists()
        )
        if not paths:
            paths = (root / "raw/card_transaction.v1.csv",)
    required = {"User", "Year", "Month", "Day", "Amount"}
    if config.load_labels:
        required.add("Is Fraud?")

    def rows(path: Path, columns: tuple[str, ...] | None = None) -> Iterator[dict[str, Any]]:
        if path.suffix == ".csv":
            with csv.open_csv(path) as reader:
                if missing := required - set(reader.schema.names):
                    raise SchemaError(f"TabFormer source is missing columns: {sorted(missing)}")
                for block in reader:
                    yield from (block if columns is None else block.select(columns)).to_pylist()
        else:
            source = pq.ParquetFile(path)
            if missing := required - set(source.schema_arrow.names):
                raise SchemaError(f"TabFormer source is missing columns: {sorted(missing)}")
            for block in source.iter_batches(
                batch_size=4096, columns=None if columns is None else list(columns)
            ):
                yield from block.to_pylist()

    unique: dict[object, None] = {}
    unique_paths = paths[:1] if config.split_protocol == "precomputed" else paths
    for path in unique_paths:
        for row in rows(path, ("User",)):
            unique.setdefault(row["User"], None)
    positive: dict[object, None] = {}
    positive_paths = (
        (precomputed_split_paths(root)[2], paths[0])
        if config.split_protocol == "precomputed"
        else paths
    )
    # The CSV reference samples owners without looking at outcomes.
    if config.load_labels and config.stratify_by_fraud and paths[0].suffix != ".csv":
        for path in positive_paths:
            for row in rows(path, ("User", "Is Fraud?")):
                if _is_fraud_value(row["Is Fraud?"]):
                    positive.setdefault(row["User"], None)
    selected = set(
        _sample_ids(
            tuple(unique),
            cap=entity_cap(config),
            seed=sample_seed(config, seed),
            positives=tuple(positive),
        )
    )
    delay = tabformer_label_delay(config)
    with _ScalarBook(fixture_dir, config.amount_field) as book:
        ordinal = 0
        for part, path in enumerate(paths):
            partition = (
                ("train", "validation", "test")[part]
                if config.split_protocol == "precomputed"
                else None
            )
            for row in rows(path):
                if row["User"] not in selected:
                    continue
                owner = str(row["User"])
                hour = _parse_hour(row.get("Time", "00:00"))
                at = _as_datetime(row["Year"], row["Month"], row["Day"], hour)
                state = row.get("Merchant State")
                state = "XX" if state is None or not str(state).strip() else str(state).strip()
                book.event(
                    owner,
                    f"{owner}-{at:%Y%m%d}-{hour:02d}-{ordinal}",
                    utc_us(at),
                    "transaction",
                    {
                        "amount": _parse_amount(row["Amount"]),
                        "card": float(row.get("Card", 0) or 0),
                        "zip": 0.0 if row.get("Zip") is None else float(row["Zip"]),
                        "user": float(owner)
                        if owner.isdigit()
                        else float(int.from_bytes(sha256(owner.encode()).digest()[:8]) % 3000),
                    },
                    {
                        "merchant_name": str(row.get("Merchant Name", "UNK")),
                        "merchant_state": state,
                        "use_chip": str(row.get("Use Chip", "CHIP TRANSACTION")),
                        "mcc": str(int(row.get("MCC", -1))),
                        "merchant_city": str(row.get("Merchant City", "")),
                    },
                    {},
                    float(_is_fraud_value(row["Is Fraud?"])) if config.load_labels else None,
                    utc_us(at + delay) if config.load_labels else None,
                    partition,
                )
                ordinal += 1
        return book.tables(config)
