# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate declared count anchors from partitioned Parquet with bounded memory."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .data import safe_path


def table_files(data_dir, entry, files):
    """Resolve only inventory members; reject extra files in a native table directory."""
    if "path" in entry:
        keys = [entry["path"]]
    else:
        prefix = entry["prefix"]
        keys = sorted(
            k for k in files if k.startswith(prefix) and k.endswith(".parquet")
        )
        actual = {
            str(p.relative_to(data_dir))
            for p in safe_path(data_dir, prefix.rstrip("/")).rglob("*.parquet")
        }
        if actual != set(keys):
            raise ValueError("Parquet directory differs from the pinned file inventory")
    if not keys:
        raise ValueError("No pinned Parquet source files")
    return [{**files[key], "path": str(safe_path(data_dir, key))} for key in keys]


def clock(value):
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(UTC)
        .replace(tzinfo=None)
    )


def generate(data_dir, directory, definition, files):
    """Keep integer source IDs; select a fixed cohort and count half-open windows."""
    import polars as pl

    spec = definition["anchor_generation"]
    cohort = spec["cohort_size"]
    if type(cohort) is not int or cohort < 1:
        raise ValueError("cohort_size must be a positive integer")
    windows = {split: clock(value) for split, value in spec["prediction_times"].items()}
    if set(windows) != {"train", "valid", "test"}:
        raise ValueError("Generated anchors require train, valid and test windows")
    horizon = spec["window_seconds"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("window_seconds must be a positive integer")
    start, stop = map(clock, spec["coverage"])
    if any(
        not start <= t
        or t.replace(tzinfo=UTC).timestamp() + horizon
        > stop.replace(tzinfo=UTC).timestamp()
        for t in windows.values()
    ):
        raise ValueError("Prediction window exceeds declared source coverage")

    def scan(table):
        return pl.scan_parquet(
            [
                f["path"]
                for f in table_files(data_dir, definition["tables"][table], files)
            ]
        )

    key = definition["tables"][spec["root_table"]]["key"]
    roots = scan(spec["root_table"]).select(key, spec["root_time"])
    roots = roots.filter(pl.col(spec["root_time"]) < min(windows.values()))
    chosen = (
        roots.select(key)
        .sort(key)
        .head(cohort)
        .collect(engine="streaming")[key]
        .to_list()
    )
    if (
        len(chosen) != cohort
        or len(set(chosen)) != cohort
        or any(type(k) is not int or not 0 <= k < 2**63 for k in chosen)
    ):
        raise ValueError(
            "Cohort requires enough unique nonnegative int64 root identities"
        )
    events = scan(spec["event_table"]).filter(pl.col(spec["event_root"]).is_in(chosen))
    event_clock = pl.col(spec["event_time"])
    # Predicate pushdown excludes row groups outside all target windows.
    from datetime import timedelta

    events = events.filter(
        (event_clock >= min(windows.values()))
        & (event_clock < max(windows.values()) + timedelta(seconds=horizon))
    )
    counts = (
        events.group_by(spec["event_root"])
        .agg(
            [
                ((event_clock >= t) & (event_clock < t + timedelta(seconds=horizon)))
                .sum()
                .cast(pl.Int64)
                .alias(split)
                for split, t in windows.items()
            ]
        )
        .collect(engine="streaming")
    )
    counts = {r[spec["event_root"]]: r for r in counts.iter_rows(named=True)}
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "sample_id": f"{split}:{entity}",
            "entity_id": str(entity),
            "as_of_time": int(t.replace(tzinfo=UTC).timestamp()) * 10**9,
            "split": split,
            "future_count": counts.get(entity, {}).get(split, 0),
        }
        for split, t in windows.items()
        for entity in chosen
    ]
    path = Path(directory) / "source-anchors.parquet"
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("entity_id", pa.string()),
            ("as_of_time", pa.timestamp("ns")),
            ("split", pa.string()),
            ("future_count", pa.int64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    identity = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generation": spec,
        "cohort_size": cohort,
        "rows": len(rows),
    }
    return path, {str(k): k for k in chosen}, identity
