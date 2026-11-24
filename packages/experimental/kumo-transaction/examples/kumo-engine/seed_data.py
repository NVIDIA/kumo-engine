# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the default Gen2 book, then write scalar tables and repeated anchors."""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq
import yaml

from contract import NUMERIC_FIELDS, SOURCES, STATIC_PROFILE_FIELDS
from tfm.config import load_recipe_config

# PyArrow populates these compute exports dynamically.
pc: Any = arrow_compute
ROOT = Path(__file__).resolve().parents[2]
TIMESTAMP = pa.timestamp("s", tz="UTC")
ANCHORS = pa.schema(
    [
        ("sample_id", pa.string()),
        ("user_id", pa.string()),
        ("as_of_time", TIMESTAMP),
        ("split", pa.string()),
    ]
)


def table_schemas() -> dict[str, pa.Schema]:
    return {
        "users": pa.schema(
            [
                ("user_id", pa.string()),
                *((f, pa.string()) for f in STATIC_PROFILE_FIELDS),
            ]
        ),
        **{
            table: pa.schema(
                [
                    ("event_id", pa.string()),
                    ("user_id", pa.string()),
                    ("ts", TIMESTAMP),
                    ("source", pa.string()),
                    *((f, pa.float64() if f in NUMERIC_FIELDS else pa.string()) for f in fields),
                ]
            )
            for table, fields in SOURCES.values()
        },
    }


def anchors_for(
    user: str,
    times: list[int],
    train_end: int,
    validation_end: int,
    test_end: int,
    window: int,
) -> list[dict[str, Any]]:
    times.sort()
    eligible = times[bisect_left(times, train_end - 730 * 86400) : bisect_left(times, train_end)]
    endpoints = sorted({*range(window, len(eligible) + 1, window), len(eligible)} - {0})
    rows = []
    previous = None
    for index, endpoint in enumerate(endpoints):
        timestamp = eligible[endpoint - 1]
        # A time-only anchor includes the whole same-second tie. Keep one anchor
        # when several stride endpoints fall in that same group.
        if timestamp != previous:
            rows.append(
                dict(
                    sample_id=f"{user}/train/{index:06d}",
                    user_id=user,
                    as_of_time=timestamp,
                    split="train",
                )
            )
        previous = timestamp
    # The final endpoint intentionally gets the same latest-N history as every
    # other anchor; a partial tail therefore overlaps the preceding window.
    for split, cutoff in (("validation", validation_end), ("test", test_end)):
        rows.append(
            dict(
                sample_id=f"{user}/{split}",
                user_id=user,
                as_of_time=cutoff - 1,
                split=split,
            )
        )
    return rows


def prepare(source: Path, output: Path, config: Path) -> dict[str, Any]:
    """Stream a user-grouped Pragmatiq book into relational tables."""
    recipe = load_recipe_config(config)
    train_end = int(datetime.fromisoformat(recipe.data.temporal_split.train_end).timestamp())
    validation_end = int(
        datetime.fromisoformat(recipe.data.temporal_split.validation_end).timestamp()
    )
    test_end = int(datetime(2025, 8, 1, tzinfo=UTC).timestamp())
    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    tables.mkdir()
    schemas = table_schemas()
    counts: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    users: set[str] = set()
    event_users: set[str] = set()
    current: str | None = None
    times: list[int] = []
    buffer: list[dict[str, Any]] = []
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                pq.ParquetWriter(tables / f"{name}.parquet", schema, compression="zstd")
            )
            for name, schema in schemas.items()
        }
        anchor_writer = stack.enter_context(
            pq.ParquetWriter(output / "anchors.parquet", ANCHORS, compression="zstd")
        )

        def flush(user: str, timestamps: list[int]) -> None:
            rows = anchors_for(
                user,
                timestamps,
                train_end,
                validation_end,
                test_end,
                recipe.data.sequence_length,
            )
            buffer.extend(rows)
            splits.update(row["split"] for row in rows)
            if len(buffer) >= 4096:
                anchor_writer.write_table(pa.Table.from_pylist(buffer, schema=ANCHORS))
                buffer.clear()

        for batch in pq.ParquetFile(source / "profiles.parquet").iter_batches(batch_size=4096):
            rows = [
                {"user_id": row["user_id"], **dict(row["attributes"])} for row in batch.to_pylist()
            ]
            users.update(row["user_id"] for row in rows)
            writers["users"].write_table(pa.Table.from_pylist(rows, schema=schemas["users"]))
        for batch in pq.ParquetFile(source / "events.parquet").iter_batches(batch_size=16384):
            raw = pa.Table.from_batches([batch])
            sources = raw["source"].combine_chunks()
            if set(pc.unique(sources).to_pylist()) - set(SOURCES):
                raise ValueError("this example supports the four Pragmatiq Gen2 event sources")
            # Ceil microseconds to the engine's second clock, so an event is
            # never visible before it happened. Stored calendar features use UTC.
            micros = raw["ts"].cast(pa.timestamp("us")).cast(pa.int64()).to_numpy()
            seconds = (micros + 999999) // 1000000
            indices = np.arange(counts["events"], counts["events"] + len(raw), dtype=np.int64)
            ids = pc.binary_join_element_wise(
                "e",
                pc.utf8_lpad(pa.array(indices).cast(pa.string()), width=12, padding="0"),
                "",
            )
            raw = raw.append_column("event_id", ids)
            raw = raw.set_column(
                raw.schema.get_field_index("ts"),
                "ts",
                pa.array(seconds).cast(TIMESTAMP),
            )
            user_ids = raw["user_id"].to_numpy()
            boundaries = [
                0,
                *(np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1).tolist(),
                len(raw),
            ]
            for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
                user = user_ids[lo]
                if current != user:
                    if current is not None:
                        flush(current, times)
                    current, times = user, []
                    event_users.add(user)
                times.extend(seconds[lo:hi].tolist())
            for name, (table, fields) in SOURCES.items():
                selected = raw.filter(pc.equal(sources, name))
                if not len(selected):
                    continue
                field_map = selected["fields"].combine_chunks()
                columns = {name: selected[name] for name in ("event_id", "user_id", "ts", "source")}
                for field in fields:
                    values = pc.map_lookup(field_map, field, "first")
                    columns[field] = (
                        values.cast(pa.float64()) if field in NUMERIC_FIELDS else values
                    )
                writers[table].write_table(pa.Table.from_pydict(columns, schema=schemas[table]))
                counts[table] += len(selected)
            counts["events"] += len(raw)
            if counts["events"] % 1_000_000 < len(raw):
                print(f"Prepared {counts['events']:,} events", flush=True)
        if current is not None:
            flush(current, times)
        for user in sorted(users - event_users):
            flush(user, [])
        if buffer:
            anchor_writer.write_table(pa.Table.from_pylist(buffer, schema=ANCHORS))
    manifest = {
        "source": str(source.resolve()),
        "counts": {"users": len(users), **counts},
        "anchors_by_split": dict(splits),
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
        "history_events": recipe.data.sequence_length,
        "observation_days": 730,
        "clock": "UTC seconds; raw event times rounded upward",
        "windows": "stride endpoints plus overlapping final tail",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/kumo-engine/source")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kumo-engine/data")
    parser.add_argument("--config", type=Path, default=ROOT / "recipes/blueprint_v2/config.yaml")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if not (args.source / "events.parquet").exists():
        # Reuse an existing book with --source to skip generation.
        from pragmatiq.data.synthetic.config import WorldConfig
        from pragmatiq.data.synthetic.generate import generate

        world = yaml.safe_load((ROOT / "configs/data/pragmatiq/world.yaml").read_text())
        generate(
            WorldConfig.from_dict(world),
            args.source,
            n_workers=args.threads,
            write_report=False,
        )
    print(json.dumps(prepare(args.source, args.output, args.config), indent=2))


if __name__ == "__main__":
    main()
