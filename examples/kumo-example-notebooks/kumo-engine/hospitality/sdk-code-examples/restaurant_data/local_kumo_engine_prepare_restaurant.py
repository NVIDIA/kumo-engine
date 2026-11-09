# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and copy the four restaurant tables with declared relationships.

The context table is excluded because interactions has no interaction_id column.
This script does not infer a relationship from CSV row order.
"""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, destination):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError(
            "Source and destination must be separate, non-overlapping directories"
        )
    specs = {
        "users": ["user_id", "age", "gender", "preferred_cuisine"],
        "restaurants": [
            "restaurant_id", "name", "cuisine", "rating", "price_range",
            "location"
        ],
        "interactions":
        ["user_id", "restaurant_id", "interaction_type", "timestamp"],
        "social_graph": ["user_a", "user_b"],
    }
    rows, metadata = {}, {}
    for name, columns in specs.items():
        path = source / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise ValueError(
                    f"{path.name}: expected columns {columns}, got {reader.fieldnames}"
                )
            rows[name] = list(reader)
        if not rows[name] or any(None in row or any(v is None
                                                    for v in row.values())
                                 for row in rows[name]):
            raise ValueError(f"{path.name}: empty table or malformed CSV row")
        metadata[name] = {
            "file": path.name,
            "rows": len(rows[name]),
            "columns": columns,
            "sha256": digest(path)
        }
    keys = {}
    for name, key in (("users", "user_id"), ("restaurants", "restaurant_id")):
        values = [row[key] for row in rows[name]]
        if any(not v for v in values) or len(set(values)) != len(values):
            raise ValueError(
                f"{name}.{key}: primary key must be nonempty and unique")
        keys[name] = set(values)
    relationships = [
        ("interactions", "user_id", "users"),
        ("interactions", "restaurant_id", "restaurants"),
        ("social_graph", "user_a", "users"),
        ("social_graph", "user_b", "users"),
    ]
    for table, column, target in relationships:
        missing = {row[column] for row in rows[table]} - keys[target]
        if missing:
            raise ValueError(
                f"{table}.{column}: {len(missing)} values are absent from {target}'s primary key"
            )
    times = [
        datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        for row in rows["interactions"]
    ]
    # Refuse all conflicts before writing anything; an identical rerun is safe.
    for item in metadata.values():
        target = destination / item["file"]
        if target.is_symlink() or (target.exists() and
                                   (not target.is_file()
                                    or digest(target) != item["sha256"])):
            raise ValueError(f"Destination conflict: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in metadata.values():
        original, copied = source / item["file"], destination / item["file"]
        if not copied.exists():
            shutil.copyfile(original, copied)
        if copied.samefile(original) or digest(copied) != item["sha256"]:
            raise RuntimeError(f"Copy verification failed: {copied}")
    return {
        "tables":
        metadata,
        "relationships": [{
            "src_table": t,
            "fkey": c,
            "dst_table": d
        } for t, c, d in relationships],
        "timestamp_format":
        "%Y-%m-%d %H:%M:%S",
        "timestamp_min":
        min(times).isoformat(),
        "timestamp_max":
        max(times).isoformat(),
        "excluded": {
            "context.csv":
            "No shared interaction_id column in interactions.csv; row-order alignment is unverified"
        },
        "independent_copies_verified":
        True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
