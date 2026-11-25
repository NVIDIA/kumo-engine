# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy transaction task anchors over disk-backed source identities and labels."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import utc_us
from tfm.data.native_source import NativeSourceFiles, source_entity_id
from tfm.data.splits.temporal import parse_timestamp

SSL_TRAIN = ("future_train_0", "future_train_1", "future_train_2")
SSL_VALIDATION = "future_validation"


def graph_anchors(source: NativeSourceFiles, config: RecipeConfig, path: Path) -> Path:
    """Preserve owner splits and available labels on the retained input history.

    This writes only scalar prediction requests. SQL does not construct Python
    events, histories, tokenized rows, or sampled input graphs. A boundary tie
    that could change a truncated label is refused until the native API exposes
    a stable secondary event ordering.
    """
    db, data = source.connection, config.data
    owners = [
        (source_entity_id(tuple(json.loads(key))), code)
        for key, code in db.execute(
            "SELECT source_key,code FROM identities WHERE table_name='users'"
        )
    ]
    owners.sort()
    if len(owners) < 12:
        raise SchemaError("Graph recipe needs at least 12 observed owners")
    owners = [owners[i] for i in np.random.default_rng(config.seed).permutation(len(owners))]
    ntrain, nval = int(0.6 * len(owners)), int(0.2 * len(owners))
    groups = {
        "train": owners[:ntrain],
        "validation": owners[ntrain : ntrain + nval],
        "test": owners[ntrain + nval :],
    }
    cutoff = parse_timestamp(data.probe_cutoff)
    if cutoff > parse_timestamp(data.observed_through):
        raise SchemaError("Probe cutoff exceeds source observation coverage")
    supervised = config.runtime.stage != "pretrain" or config.training.objective == "binary"
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("sample_id", pa.string()),
            ("cutoff", pa.timestamp("us", tz="UTC")),
            ("split", pa.string()),
            ("target", pa.float32()),
            ("label_available_us", pa.int64()),
            ("label_cutoff_us", pa.int64()),
        ]
    )
    buffer: list[dict] = []
    query = """
        SELECT i.event_us,e.value,e.available_us FROM links l
        JOIN identities i ON i.table_name='events' AND i.code=l.child
        LEFT JOIN event_labels e ON e.table_name='events' AND e.code=i.code
        WHERE l.role='events' AND l.parent=? AND i.event_us<=? AND i.available_us<=?
        ORDER BY i.event_us DESC,i.code DESC LIMIT ?
    """
    with pq.ParquetWriter(path, schema) as writer:

        def append(owner: str, code: int, at, split: str, labels: bool) -> None:
            value = None
            if labels:
                now = utc_us(at)
                covered = db.execute(
                    """
                    SELECT 1 FROM links l JOIN identities i ON i.table_name='events' AND i.code=l.child
                    JOIN event_labels e ON e.table_name='events' AND e.code=i.code
                    WHERE l.role='events' AND l.parent=? AND i.event_us<=? AND i.available_us<=?
                    AND e.available_us<=? LIMIT 1
                """,
                    (code, now, now, now),
                ).fetchone()
                if covered is None:
                    raise SchemaError(
                        f"Missing available fraud label coverage for supervised owner {owner!r}"
                    )
                retained = db.execute(query, (code, now, now, data.sequence_length + 1)).fetchall()
                if not retained:
                    raise SchemaError(
                        f"Probe root {owner!r} has no history at the prediction cutoff"
                    )
                if len(retained) > data.sequence_length and retained[-1][0] == retained[-2][0]:
                    raise UnsupportedCombinationError(
                        f"history_paths.events: truncated label boundary for owner {owner!r} has tied event clocks; native secondary event ordering is unsupported"
                    )
                value = float(
                    any(row[1] == 1 and row[2] <= now for row in retained[: data.sequence_length])
                )
            buffer.append(
                {
                    "id": owner,
                    "sample_id": f"{owner}:{at.isoformat()}",
                    "cutoff": at,
                    "split": split,
                    "target": value,
                    "label_available_us": utc_us(at) if labels else None,
                    "label_cutoff_us": utc_us(at) if labels else None,
                }
            )
            if len(buffer) == 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()

        if config.training.objective == "future_subgraph" and config.runtime.stage != "downstream":
            boundary = parse_timestamp(data.pretrain_end)
            if supervised and cutoff <= boundary:
                raise SchemaError("Future pretraining requires a later probe cutoff")
            horizon, delay = (
                timedelta(days=data.horizon_days),
                timedelta(days=data.label_delay_days),
            )
            nssl = max(2, len(groups["train"]) // 4)
            ssl_val, ssl_train = groups["train"][:nssl], groups["train"][nssl:]
            for split, k in zip(SSL_TRAIN, (4, 3, 2), strict=True):
                for owner, code in ssl_train:
                    append(owner, code, boundary - delay - k * horizon, split, False)
            for owner, code in ssl_val:
                append(owner, code, boundary - delay - horizon, SSL_VALIDATION, False)
        if supervised:
            for split, members in groups.items():
                for owner, code in members:
                    append(owner, code, cutoff, split, True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    return path


def sequence_anchors(source: NativeSourceFiles, config: RecipeConfig, path: Path) -> Path:
    """Write temporal entity requests and labels from their own retained windows."""
    from datetime import UTC, datetime

    from tfm.data.labels.windows import side_label_windows
    from tfm.data.splits.temporal import parse_duration

    db, data = source.connection, config.data
    precomputed = data.split_protocol == "precomputed"
    db.execute("CREATE TABLE event_partitions(code INTEGER PRIMARY KEY,partition TEXT)")
    for block in pq.ParquetFile(source.source.files["events"]).iter_batches(
        batch_size=4096, columns=["__tfm_row_id", "partition"]
    ):
        db.executemany(
            "INSERT INTO event_partitions VALUES (?,?)",
            zip(block.column(0).to_pylist(), block.column(1).to_pylist(), strict=True),
        )
    if precomputed:
        cutoffs = {}
        for side, parts in (
            ("train", ("train",)),
            ("validation", ("train", "validation")),
            ("test", ("train", "validation", "test")),
        ):
            placeholders = ",".join("?" for _ in parts)
            maximum = db.execute(
                f"SELECT MAX(i.event_us) FROM identities i JOIN event_partitions p ON i.code=p.code WHERE i.table_name='events' AND p.partition IN ({placeholders})",
                parts,
            ).fetchone()[0]
            if maximum is None:
                raise SchemaError(f"Precomputed source has no {side} history")
            cutoffs[side] = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=maximum)
    else:
        maximum = db.execute(
            "SELECT MAX(event_us) FROM identities WHERE table_name='events'"
        ).fetchone()[0]
        if maximum is None:
            raise SchemaError("Source has no events for temporal anchors")
        cutoffs = {
            "train": parse_timestamp(data.temporal_split.train_end),
            "validation": parse_timestamp(data.temporal_split.validation_end),
            "test": datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=maximum),
        }
        if cutoffs["test"] < cutoffs["validation"]:
            raise SchemaError("Temporal source has no test partition; adjust cutoffs")
    window = parse_duration(data.observation_window)
    windows = side_label_windows(cutoffs["train"], cutoffs["validation"], cutoffs["test"], window)
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("sample_id", pa.string()),
            ("cutoff", pa.timestamp("us", tz="UTC")),
            ("split", pa.string()),
            ("target", pa.float32()),
            ("label_available_us", pa.int64()),
            ("label_cutoff_us", pa.int64()),
        ]
    )
    with pq.ParquetWriter(path, schema) as writer:
        for side, at in cutoffs.items():
            now = utc_us(at)
            parts = (
                ("train",)
                if side == "train"
                else ("train", "validation")
                if side == "validation"
                else ("train", "validation", "test")
            )
            partitions = (
                " AND p.partition IN (" + ",".join("?" for _ in parts) + ")" if precomputed else ""
            )
            # Training is exclusive at train_end; validation includes its
            # cutoff but excludes test-partition rows at the same timestamp.
            upper = "<" if not precomputed and side == "train" else "<="
            if not precomputed and side == "validation":
                upper = "<"
            query = f"""
                SELECT i.event_us,e.value,e.available_us FROM links l
                JOIN identities i ON i.table_name='events' AND i.code=l.child
                JOIN event_partitions p ON p.code=i.code
                LEFT JOIN event_labels e ON e.table_name='events' AND e.code=i.code
                WHERE l.role='events' AND l.parent=? AND i.event_us{upper}? AND i.event_us>=?
                  AND i.available_us<=? {partitions}
                ORDER BY i.event_us DESC,i.code DESC LIMIT ?
            """
            buffer = []
            for key, code in db.execute(
                "SELECT source_key,code FROM identities WHERE table_name='users' ORDER BY source_key"
            ):
                retained = db.execute(
                    query,
                    (
                        code,
                        now,
                        now - int(window.total_seconds()) * 1_000_000,
                        now,
                        *(parts if precomputed else ()),
                        data.sequence_length + 1,
                    ),
                ).fetchall()
                if not retained:
                    continue
                if len(retained) > data.sequence_length and retained[-1][0] == retained[-2][0]:
                    raise UnsupportedCombinationError(
                        "history_paths.events: truncated sequence label boundary has tied event clocks; stable native secondary ordering is unsupported"
                    )
                retained = retained[: data.sequence_length]
                owner = source_entity_id(tuple(json.loads(key)))
                # Absence of a label source must never become an all-negative
                # scored dataset. Unsupervised training can omit it explicitly.
                available = [row for row in retained if row[1] is not None and row[2] <= now]
                target = None
                if data.load_labels:
                    if not available:
                        # Coverage is independent of truncation: app-only final
                        # histories are valid when earlier labels are known.
                        covered = db.execute(
                            """SELECT 1 FROM links l JOIN identities i ON i.table_name='events' AND i.code=l.child JOIN event_labels e ON e.table_name='events' AND e.code=i.code WHERE l.role='events' AND l.parent=? AND i.event_us<=? AND e.available_us<=? LIMIT 1""",
                            (code, now, now),
                        ).fetchone()
                        if covered is None:
                            raise SchemaError(
                                f"Missing available fraud label coverage for supervised owner {owner!r}"
                            )
                    start = windows[side].start
                    low = utc_us(start) if start is not None else -(2**63)
                    target = float(
                        any(
                            row[1] == 1 and row[2] <= now and low < row[0] <= now
                            for row in retained
                        )
                    )
                buffer.append(
                    {
                        "id": owner,
                        "sample_id": f"{owner}:{at.isoformat()}",
                        "cutoff": at,
                        "split": side,
                        "target": target,
                        "label_available_us": now if target is not None else None,
                        "label_cutoff_us": now if target is not None else None,
                    }
                )
                if len(buffer) == 4096:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        if data.pretrain_windows is not None:
            from tfm.data.native_windows import PREFIX

            for (key,) in db.execute(
                "SELECT source_key FROM identities WHERE table_name='users' ORDER BY source_key"
            ):
                owner = str(json.loads(key)[0])
                if owner.startswith(PREFIX) and not owner.endswith("/unused"):
                    writer.write_table(
                        pa.Table.from_pylist(
                            [
                                {
                                    "id": owner,
                                    "sample_id": f"pretrain/{owner}",
                                    "cutoff": cutoffs["train"],
                                    "split": "pretrain",
                                    "target": None,
                                    "label_available_us": None,
                                    "label_cutoff_us": None,
                                }
                            ],
                            schema=schema,
                        )
                    )
        db.execute("CREATE TABLE task_skips(task TEXT PRIMARY KEY,reason TEXT)")
        if config.evaluation.history_window.enabled and data.load_labels:
            _event_anchors(source, config, cutoffs, windows, writer, schema)
        _forecast_anchors(source, config, cutoffs["train"], writer, schema)
    return path


def _event_anchors(
    source: NativeSourceFiles, config: RecipeConfig, cutoffs, windows, writer, schema
) -> None:
    """Sample scalar evaluation requests on disk, independently of model inputs."""
    from datetime import UTC, datetime
    from random import Random

    from tfm.data.splits.entity import split_entities

    db, settings = source.connection, config.evaluation.history_window
    classes = list(
        db.execute(
            """
        SELECT u.source_key,MAX(e.value) FROM event_labels e
        JOIN identities i ON i.table_name='events' AND i.code=e.code
        JOIN links l ON l.role='events' AND l.child=i.code
        JOIN identities u ON u.table_name='users' AND u.code=l.parent
        WHERE e.table_name='events' AND i.event_us>? AND i.event_us<=? AND e.available_us<=?
        GROUP BY l.parent ORDER BY u.source_key
    """,
            (utc_us(windows["test"].start), utc_us(cutoffs["test"]), utc_us(cutoffs["test"])),
        )
    )
    from tfm.contracts.errors import InsufficientLabelSupportError

    try:
        split = split_entities(
            [(str(json.loads(key)[0]), int(label)) for key, label in classes], seed=config.seed
        )
    except InsufficientLabelSupportError as error:
        db.execute("INSERT INTO task_skips VALUES (?,?)", ("history_window", str(error)))
        return
    heldout = set(split.evaluation)
    db.execute(
        "CREATE TABLE history_candidates(owner TEXT,event_key TEXT,at INTEGER,value REAL,known INTEGER)"
    )
    for side in ("train", "test"):
        db.execute("DELETE FROM history_candidates")
        start = windows[side].start
        query = """
            SELECT u.source_key,i.source_key,i.event_us,e.value,e.available_us FROM event_labels e
            JOIN identities i ON i.table_name='events' AND i.code=e.code
            JOIN links l ON l.role='events' AND l.child=i.code
            JOIN identities u ON u.table_name='users' AND u.code=l.parent
            WHERE e.table_name='events' AND i.event_us>? AND i.event_us<=? AND e.available_us<=?
            ORDER BY u.source_key,i.event_us,i.source_key
        """
        cursor = db.execute(
            query,
            (
                utc_us(start) if start is not None else -(2**63),
                utc_us(cutoffs[side]),
                utc_us(cutoffs[side]),
            ),
        )
        while block := cursor.fetchmany(4096):
            db.executemany(
                "INSERT INTO history_candidates VALUES (?,?,?,?,?)",
                (
                    (str(json.loads(owner)[0]), str(json.loads(key)[0]), at, value, known)
                    for owner, key, at, value, known in block
                    if (str(json.loads(owner)[0]) in heldout) == (side == "test")
                ),
            )
        n = db.execute("SELECT COUNT(*) FROM history_candidates").fetchone()[0]
        rng = Random(config.seed)
        ratio = settings.train_negatives_per_positive if side == "train" else None
        selected, positives = None, None
        negatives: set[int] = set()
        if ratio is not None:
            np = db.execute("SELECT COUNT(*) FROM history_candidates WHERE value=1").fetchone()[0]
            keep = np if settings.max_rows is None else min(np, settings.max_rows)
            positives = None if keep == np else set(rng.sample(range(np), keep))
            nn = min(n - np, round(ratio * keep))
            if settings.max_rows is not None:
                nn = min(nn, settings.max_rows - keep)
            negatives = set(rng.sample(range(n - np), nn))
        elif settings.max_rows is not None and n > settings.max_rows:
            selected = set(rng.sample(range(n), settings.max_rows))
        counters, buffer = [0, 0], []
        for index, (owner, key, at, value, known) in enumerate(
            db.execute("SELECT * FROM history_candidates ORDER BY rowid")
        ):
            group = int(value)
            position = counters[group]
            counters[group] += 1
            if selected is not None and index not in selected:
                continue
            if ratio is not None and (
                (group == 1 and positives is not None and position not in positives)
                or (group == 0 and position not in negatives)
            ):
                continue
            code = source.code("users", (owner,))
            tied = db.execute(
                "SELECT COUNT(*) FROM links l JOIN identities i ON i.table_name='events' AND i.code=l.child WHERE l.role='events' AND l.parent=? AND i.event_us=?",
                (code, at),
            ).fetchone()[0]
            if tied != 1:
                raise UnsupportedCombinationError(
                    "history_window.events: native event histories cannot stop at one row within a tied timestamp; select data.loader: python"
                )
            cutoff = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=at)
            buffer.append(
                {
                    "id": owner,
                    "sample_id": f"history/{owner}/{key}",
                    "cutoff": cutoff,
                    "split": f"history_{side}",
                    "target": value,
                    "label_available_us": known,
                    "label_cutoff_us": utc_us(cutoffs[side]),
                }
            )
            if len(buffer) == 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))


def _forecast_anchors(
    source: NativeSourceFiles, config: RecipeConfig, cutoff, writer, schema
) -> None:
    """Preserve legacy frozen LabelRow probes without inventing availability clocks.

    These are label-free model requests. Their external outcomes are joined
    after embedding extraction, never passed as native training targets.
    """
    from tfm.contracts.errors import InsufficientLabelSupportError
    from tfm.data.splits.entity import split_entities
    from tfm.data.splits.temporal import parse_duration

    db = source.connection
    db.execute(
        "CREATE TABLE forecast_labels(sample_id TEXT PRIMARY KEY,task TEXT,owner TEXT,at INTEGER,value REAL)"
    )
    db.execute(
        "CREATE TABLE forecast_candidates(sample_id TEXT PRIMARY KEY,owner TEXT,at INTEGER,value REAL)"
    )
    owners = {
        str(json.loads(row[0])[0])
        for row in db.execute("SELECT source_key FROM identities WHERE table_name='users'")
    }
    lookback = int(parse_duration(config.data.observation_window).total_seconds()) * 1_000_000
    for table in source.original.tables:
        if table.role != "labels" or not table.name.startswith("labels_"):
            continue
        task = table.name.removeprefix("labels_")
        db.execute("DELETE FROM forecast_candidates")
        ordinal = 0
        for block in pq.ParquetFile(source.original.files[table.name]).iter_batches(
            batch_size=4096
        ):
            records = []
            for row in block.to_pylist():
                owner = str(row["user_id"])
                if owner not in owners:
                    continue
                from tfm.data.adapters.pragmatiq import _as_utc_datetime

                at = _as_utc_datetime(row[table.key[1]])
                value = float(row["label"])
                if value not in (0, 1):
                    raise SchemaError(f"Forecast task {task!r} requires binary labels")
                # LabelRow is a legacy timestamped outcome, with no label
                # availability assertion. Keep this distinct from target.
                records.append((f"forecast/{task}/{ordinal}", owner, utc_us(at), value))
                ordinal += 1
            db.executemany("INSERT INTO forecast_candidates VALUES (?,?,?,?)", records)
        early = db.execute(
            "SELECT 1 FROM forecast_candidates WHERE at<? LIMIT 1", (utc_us(cutoff),)
        ).fetchone()
        if early is not None:
            db.execute(
                "INSERT INTO task_skips VALUES (?,?)",
                (task, "label rows predate pretraining cutoff"),
            )
            continue
        # Remove requests without any eligible history before stratification.
        for sid, owner, at in db.execute("SELECT sample_id,owner,at FROM forecast_candidates"):
            code = source.code("users", (owner,))
            if (
                db.execute(
                    "SELECT 1 FROM links l JOIN identities i ON i.table_name='events' AND i.code=l.child WHERE l.role='events' AND l.parent=? AND i.event_us<=? AND i.event_us>=? LIMIT 1",
                    (code, at, at - lookback),
                ).fetchone()
                is None
            ):
                db.execute("DELETE FROM forecast_candidates WHERE sample_id=?", (sid,))
        classes = list(
            db.execute(
                "SELECT owner,MAX(value) FROM forecast_candidates GROUP BY owner ORDER BY owner"
            )
        )
        try:
            split = split_entities(
                [(owner, int(value)) for owner, value in classes], seed=config.seed
            )
        except InsufficientLabelSupportError as error:
            db.execute("INSERT INTO task_skips VALUES (?,?)", (task, str(error)))
            continue
        heldout = set(split.evaluation)
        buffer = []
        from datetime import UTC, datetime

        for sid, owner, at, value in db.execute("SELECT * FROM forecast_candidates ORDER BY rowid"):
            when = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=at)
            side = "test" if owner in heldout else "train"
            buffer.append(
                {
                    "id": owner,
                    "sample_id": sid,
                    "cutoff": when,
                    "split": f"forecast_{task}_{side}",
                    "target": None,
                    "label_available_us": None,
                    "label_cutoff_us": None,
                }
            )
            db.execute(
                "INSERT INTO forecast_labels VALUES (?,?,?,?,?)", (sid, task, owner, at, value)
            )
            if len(buffer) == 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    db.commit()
