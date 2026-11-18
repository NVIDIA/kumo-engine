# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream partitioned string-root events into shared native table layouts."""

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import identity_map, verify_file
from .relational_sources import source_paths


def partition(path, name, allowed):
    values = [
        part.split("=", 1)[1]
        for part in Path(path).parts
        if part.startswith(name + "=")
    ]
    if len(values) != 1 or values[0] not in allowed:
        raise ValueError(f"Unexpected source partition: {path}")
    return values[0]


def numeric_category(value):
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("Category codes must be numeric or null")
    if not math.isfinite(value):
        return None
    return str(int(value)) if int(value) == value else repr(value)


def prepare_sources(data_dir, output, definition, files, roots):
    from kumo.pytorch.api import Dataset, Parquet

    options = definition["source_options"]
    paths = {
        name: source_paths(data_dir, entry, files)
        for name, entry in options["tables"].items()
    }
    for group in paths.values():
        for path in group:
            verify_file(path, files[path.relative_to(data_dir).as_posix()])
    identities = identity_map(data_dir, definition, files)
    names = pa.array(sorted(identities, key=lambda v: v.encode("utf-8")), pa.string())
    ids = pa.array([identities[name] for name in names.to_pylist()], pa.int64())
    membership = {}
    root = options["root"]
    for path in paths[root["table"]]:
        group = partition(path, options["partition"], options["splits"])
        for name in (
            pq.ParquetFile(path).read(columns=[root["key"]]).column(0).to_pylist()
        ):
            if name in membership:
                raise ValueError(f"Duplicate root membership: {name}")
            membership[name] = group
    groups = pa.array([int(membership[name]) for name in names.to_pylist()], pa.int64())
    selected = names if roots is None else pa.array(sorted(roots), pa.string())
    if pc.any(pc.invert(pc.is_in(selected, value_set=names))).as_py():
        raise ValueError("Unknown selected source root")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(
        pa.table({"root_id": pc.take(ids, pc.index_in(selected, value_set=names))}),
        output / "roots.parquet",
    )
    spans = {}
    for family, spec in options["events"].items():
        schema = pa.schema(
            [
                ("event_id", pa.int64()),
                ("root_id", pa.int64()),
                ("event_time", pa.timestamp("ms")),
                ("event_time_seconds", pa.int64()),
                ("event_time_ns", pa.int64()),
                *[(name, pa.float64()) for name in spec["numerical"]],
                *[(name, pa.string()) for name in spec["categorical"]],
            ]
        )
        offset, minimum, maximum = 0, None, None
        with pq.ParquetWriter(
            output / f"{family}.parquet", schema, compression="zstd"
        ) as writer:
            for path in paths[family]:
                fold = int(partition(path, options["partition"], options["splits"]))
                columns = [
                    spec["root"],
                    spec["time"],
                    options["partition"],
                    *spec["numerical"],
                    *spec["categorical"],
                ]
                for batch in pq.ParquetFile(path).iter_batches(columns=columns):
                    data = pa.Table.from_batches([batch])
                    indices = pc.index_in(data[spec["root"]], value_set=names)
                    if indices.null_count or data[spec["root"]].null_count:
                        raise ValueError(f"{path}: event has unknown root")
                    actual = data[options["partition"]]
                    if (
                        actual.null_count
                        or pc.any(pc.not_equal(actual, fold)).as_py()
                        or pc.any(pc.not_equal(pc.take(groups, indices), fold)).as_py()
                    ):
                        raise ValueError(
                            f"{path}: event partition disagrees with root membership"
                        )
                    clock = data[spec["time"]]
                    if not pa.types.is_timestamp(clock.type) or clock.null_count:
                        raise ValueError(
                            f"{path}: event time must be a non-null timestamp"
                        )
                    stamp = pc.cast(pc.cast(clock, pa.timestamp("ns")), pa.int64())
                    lo, hi = pc.min(stamp).as_py(), pc.max(stamp).as_py()
                    if lo is not None:
                        minimum = lo if minimum is None else min(minimum, lo)
                        maximum = hi if maximum is None else max(maximum, hi)
                    # Stable file/physical-row identity; independent of selected roots.
                    data = data.append_column(
                        "__id", pa.array(range(offset, offset + len(data)), pa.int64())
                    )
                    data = data.append_column(
                        "__root", pc.take(ids, indices)
                    ).append_column("__ns", stamp)
                    offset += len(data)
                    data = data.filter(pc.is_in(data[spec["root"]], value_set=selected))
                    stamp = data["__ns"]
                    # Arrow integer division truncates; explicitly implement floor for pre-epoch rows.
                    quotient = pc.divide(stamp, 10**9)
                    remainder = pc.subtract(stamp, pc.multiply(quotient, 10**9))
                    seconds = pc.subtract(
                        quotient, pc.cast(pc.less(remainder, 0), pa.int64())
                    )
                    values = {
                        "event_id": data["__id"],
                        "root_id": data["__root"],
                        "event_time": pc.cast(
                            pc.multiply(seconds, 1000), pa.timestamp("ms")
                        ),
                        "event_time_seconds": seconds,
                        "event_time_ns": stamp,
                    }
                    for name in spec["numerical"]:
                        value = pc.cast(data[name], pa.float64())
                        values[name] = pc.if_else(pc.is_finite(value), value, None)
                    for name in spec["categorical"]:
                        value = data[name]
                        # Normalize unique codes, rather than a Python loop over all events.
                        unique = pc.unique(value)
                        codes = pa.array(
                            [numeric_category(v) for v in unique.to_pylist()],
                            pa.string(),
                        )
                        values[name] = pc.take(
                            codes, pc.index_in(value, value_set=unique)
                        )
                    writer.write_table(pa.table(values, schema=schema))
        spans[family] = {"min_ns": minimum, "max_ns": maximum, "source_rows": offset}
    dataset = Dataset(
        {
            name: Parquet(str(output / f"{name}.parquet"))
            for name in definition["tables"]
        }
    )
    mapping = {
        "scope": {
            "kind": "full_source" if roots is None else "selected_roots",
            "complete_selected_root_histories": True,
            "time_filter": None,
        },
        "sources": {
            name: [
                {**files[path.relative_to(data_dir).as_posix()], "path": str(path)}
                for path in group
            ]
            for name, group in paths.items()
        },
        "recorded_spans": spans,
        "tables": {
            name: {
                "rows": pq.read_metadata(source.path).num_rows,
                "bytes": Path(source.path).stat().st_size,
            }
            for name, source in dataset.tables.items()
        },
    }
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return dataset, mapping
