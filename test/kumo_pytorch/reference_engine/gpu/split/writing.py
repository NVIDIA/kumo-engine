# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`write_partitions`: writing a dataset's split, its files and `layout.json`.

Split a dataset by root entity, once, on the GPU, so that it can stream through GPUs.
A dataset too big for any GPU (or for the host's memory) is streamed through a
GPU one partition at a time (`test.kumo_pytorch.reference_engine.gpu.split.streaming`). A partition holds a share of
the root entities and every row that belongs to them. This module makes those
partitions: it reads the dataset once, in pieces that fit the GPU, assigns each
row to its partition(s) on the GPU, and writes a split Parquet copy plus
``layout.json``.

Which rows belong to a root entity follows the schema's relations
(`plan_partitions`). A relation says two columns are equal. Where one side is
its table's primary key, the other table's rows point at a parent row:

- A root row goes to partition ``murmurhash3_x86_32(primary key) mod K`` (an
  integer's value, a text's bytes).
- A table that reaches the root through a chain of such pointers (an
  "owned" table, such as ``transactions -> accounts -> users``) follows its
  owner: each row goes to the partition of the root row its chain reaches.
- A row with more than one owner, such as a transfer between two users'
  accounts, is copied into each owner's partition, once per partition.
- A row whose chains reach no root row (a null or dangling key) belongs to no
  partition. It is not written, and ``layout.json`` counts it.
- Every other table is "shared": it is not split or copied. Each GPU loads it
  once from its original source and keeps it while partitions stream past.

Given the contract's sample, a partition also holds every row its root rows
reach along the routes the sample reads (`test.kumo_pytorch.reference_engine.gpu.sampling.paths`): a neighborhood's
layers, and the relations its events, values and targets follow. These rows
are copied in from wherever they belong, after the partition's own rows
(``halo-*.parquet`` files, which sort after its ``chunk-*.parquet`` files), so
that a sample built from one partition reads what it would read from the whole
dataset. Rows of shared tables are on every GPU already and are not copied.
A route that only walks down the chains from the root (``users -> accounts ->
cards``) reaches the partition's own rows alone and copies nothing. When the
split chooses K itself, it first counts each partition's own and copied rows
(reading key columns only) and raises K until the largest fits, or refuses.
Working out the copies holds, on the GPU, the keys each partition's routes
reach. When that would take more than a quarter of the GPU's free memory, the
partitions are taken in groups, each group's copies written in a pass of its
own; a single partition that does not fit is refused before anything is written.

Splitting is never the default. A dataset is loaded directly
(`test.kumo_pytorch.reference_engine.gpu.data.tables.load_tables`) unless it does not fit, as
`test.kumo_pytorch.reference_engine.gpu.split.routing.choose_route` decides. `write_partitions` refuses to split a
dataset that fits, unless a partition count is given explicitly. When a split is
needed, K is chosen so that two partitions (the one in use and the next, read
ahead) fit in the room left for data, with one partition's peak while it is
loaded or an index is built on it, each counted as `test.kumo_pytorch.reference_engine.gpu.split.routing` counts
what the data takes on the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import test.kumo_pytorch.reference_engine.gpu.split.routing as footprint
from collections.abc import Mapping, Sequence
from glob import has_magic
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset, Parquet

from test.kumo_pytorch.reference_engine.gpu.split import owners as owner_maps
from test.kumo_pytorch.reference_engine.gpu.split import sizing, walk
from test.kumo_pytorch.reference_engine.gpu.split.owners import _Owners
from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import (
    _check_keys,
    _sample_routes,
    plan_partitions,
)

__all__ = ['LAYOUT', 'source_fingerprint', 'write_partitions']


LAYOUT = "layout.json"


def write_partitions(schema: DataSchema, dataset: Dataset, root: str, directory: str | Path, *,
                     sample: SampleSpec | None = None, num_partitions: int | None = None,
                     model_bytes: int | None = None, chunk_bytes: int | None = None) -> dict[str, Any]:
    """Write `dataset` split by `root` entity into `directory`; return its layout.

    :param sample: the contract's sample, whose entity must be `root`. Each
        partition then also holds the rows its root rows reach along the
        routes the sample reads. Without it, only the rows that belong to the
        partition's root rows are written.
    :param num_partitions: K. When omitted, `choose_route` decides, with
        `model_bytes`: a dataset that fits on the GPU is refused (load it
        directly instead), and one that does not gets the K it computes. Given
        explicitly, the split is made whatever the dataset's size (for tests and
        benchmarks), and `model_bytes` is not used.
    :param model_bytes: the model's measured peak memory while training, for `choose_route`.
    :param chunk_bytes: how much of a table is read onto the GPU at once;
        one sixteenth of the GPU's memory when omitted.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.sources import (
        resolve_sources,
    )

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty; partitions are written into a new or empty folder")
    if num_partitions is None:
        route = footprint.choose_route(schema, dataset, root, sample=sample, model_bytes=model_bytes)
        if route.route == "direct":
            beside = "" if model_bytes is None else f", beside a model peaking at {model_bytes / 1e9:.2f} GB"
            raise ValueError(
                f"the dataset fits on the GPU (about {route.dataset_bytes / 1e9:.2f} GB with its indexes, "
                f"{route.peak_bytes / 1e9:.2f} GB at its peak, of {route.gpu_bytes / 1e9:.0f} GB{beside}), so it "
                "is not split: load it directly with test.kumo_pytorch.reference_engine.gpu.data.tables.load_tables, or pass num_partitions to split "
                "it anyway"
            )
        num_partitions, room, pair = route.num_partitions, route.partition_bytes, route.partition_peak_bytes
    else:
        room = pair = None
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be at least 1, not {num_partitions}")
    plan = plan_partitions(schema, root)
    sources = resolve_sources(schema, dataset)
    fingerprints = {name: source_fingerprint(source) for name, (source, _) in sources.items()}
    _check_keys(plan, sources)
    routes = _sample_routes(schema, plan, sources, sample)
    loads = footprint._loads(schema, sample, sources)
    total = footprint._gpu_memory()
    chunk_bytes = chunk_bytes or total // 16
    budget = walk._walk_budget()
    num_partitions, owners, groups, first, tried = sizing._fit(plan, sources, routes, loads, num_partitions, room, pair,
                                                               chunk_bytes, budget)
    directory.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Any] = {}
    per_partition: list[dict[str, int]] = [{} for _ in range(num_partitions)]
    halo_per_partition: list[dict[str, int]] = [{} for _ in range(num_partitions)]
    for name in plan.owned:
        source, columns = sources[name]
        stats = _write_table(name, source, columns, owners, first.get(name, ()), directory, num_partitions,
                             chunk_bytes)
        for k, rows in enumerate(stats.pop("rows_per_partition")):
            per_partition[k][name] = rows
        for k, rows in enumerate(stats.pop("halo_rows_per_partition")):
            halo_per_partition[k][name] = rows
        tables[name] = {
            "kind": "partitioned",
            "columns": _columns_json(columns),
            "fingerprint": fingerprints[name],
            "chains": [[dataclasses.asdict(step) for step in chain] for chain in plan.owned[name]],
            **stats,
        }
    for group, partitions in enumerate(groups[1:], start=1):
        # Each further group of partitions: walk again, then one pass over each table it copies from.
        reached = walk._reach(plan, sources, owners, routes, chunk_bytes, partitions=partitions, budget=budget)
        for name, selectors in reached.items():
            source, columns = sources[name]
            stats = _write_table(name, source, columns, owners, selectors, directory, num_partitions, chunk_bytes,
                                 group=group, home=False)
            for k, rows in enumerate(stats["halo_rows_per_partition"]):
                halo_per_partition[k][name] += rows
            tables[name]["halo_rows"] += stats["halo_rows"]
        del reached
    for name in plan.shared:
        source, columns = sources[name]
        path = source.path if has_magic(source.path) else str(Path(source.path).expanduser().resolve())
        tables[name] = {
            "kind": "shared",
            "columns": _columns_json(columns),
            "fingerprint": fingerprints[name],
            "source": {"path": path, "columns": dict(source.columns)},
        }
    changed = sorted(name for name, (source, _) in sources.items() if source_fingerprint(source) != fingerprints[name])
    if changed:
        raise ValueError(f"the Parquet files of {', '.join(changed)} changed while they were split; split them again")
    import pylibcudf as plc

    layout = {
        "schema_version": 1,
        "floats": "exact",  # float columns kept bit for bit (-0.0 included); see `_write`
        "root": root,
        "num_partitions": num_partitions,
        "assignment": {
            "hash": "murmurhash3_x86_32", "seed": int(plc.hashing.LIBCUDF_DEFAULT_HASH_SEED),
            "of": f"{root}.{plan.primary_keys[root]}",
            "partition": "hash modulo num_partitions",
        },
        "routes": [[dataclasses.asdict(hop) for hop in route] for route in routes],
        "room_bytes": room,
        "peak_room_bytes": pair,
        "sizing": tried,
        "walk": {"budget_bytes": budget, "groups": groups},
        "tables": tables,
        "partitions": [
            {"id": k, "rows": per_partition[k], "halo_rows": halo_per_partition[k],
             **footprint._partition_bytes(loads, {name: per_partition[k][name] + halo_per_partition[k][name]
                                                  for name in plan.owned}),
             "bytes": sum(path.stat().st_size for path in (directory / _part(k)).rglob("*.parquet"))}
            for k in range(num_partitions)
        ],
    }
    (directory / LAYOUT).write_text(json.dumps(layout, indent=2) + "\n")
    return layout


def source_fingerprint(source: Parquet) -> str:
    """A table's Parquet source as upstream's DiskGraph engine identifies the files it indexed: their paths,
    sizes and modification times, with the binding's own settings (``kumo.pytorch.materialization``)."""

    from kumo.pytorch.dataset import Dataset
    from kumo.pytorch.materialization import _dataset_payload, _fingerprint

    return _fingerprint(_dataset_payload(Dataset({"table": source})))


def _part(k: int) -> str:
    return f"part-{k:05d}"


def _columns_json(columns: Mapping[str, DType]) -> dict[str, dict[str, Any]]:
    return {name: dataclasses.asdict(dtype) for name, dtype in columns.items()}


def _write_table(name: str, source: Parquet, columns: Mapping[str, DType], owners: _Owners,
                 selectors: Sequence[tuple[str, Any, Any, DType]], directory: Path, num_partitions: int,
                 chunk_bytes: int, *, group: int = 0, home: bool = True) -> dict[str, Any]:
    """Write one owned table's rows into their partitions, one chunk at a time.

    A partition's own rows go to ``chunk-*.parquet``; the rows its routes reach
    (`selectors`, from `_reach` for one `group` of partitions) and it does not
    own go to ``halo-<group>-*.parquet``. Without `home`, only the copies are written.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.tables import read_chunks

    import pylibcudf as plc

    rows_per_partition = [0] * num_partitions
    halo_per_partition = [0] * num_partitions
    read = unreachable = copies = 0
    template = None
    for index, chunk in enumerate(read_chunks(name, source, columns, chunk_bytes=chunk_bytes)):
        read += chunk.num_rows
        table = plc.Table(list(chunk.columns.values()))
        template = plc.copying.slice(table, [0, 0])[0]
        if chunk.num_rows == 0:
            continue
        rows, parts, halo_rows, halo_parts = owner_maps._places(name, chunk, owners, selectors)
        reached = plc.stream_compaction.distinct(
            plc.Table([rows]), [0], plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
            plc.types.NullEquality.EQUAL, plc.types.NanEquality.ALL_EQUAL).num_rows()
        unreachable += chunk.num_rows - reached
        copies += rows.size() - reached
        if home:
            selected = plc.copying.gather(table, rows, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
            grouped, offsets = plc.partitioning.partition(selected, parts, num_partitions)
            for k in range(num_partitions):
                if offsets[k + 1] > offsets[k]:
                    piece = plc.copying.slice(grouped, [offsets[k], offsets[k + 1]])[0]
                    _write(piece, columns, directory / _part(k) / name / f"chunk-{index:06d}.parquet")
                    rows_per_partition[k] += offsets[k + 1] - offsets[k]
        if halo_rows is None or halo_rows.size() == 0:
            continue
        selected = plc.copying.gather(table, halo_rows, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
        grouped, offsets = plc.partitioning.partition(selected, halo_parts, num_partitions)
        for k in range(num_partitions):
            if offsets[k + 1] > offsets[k]:
                piece = plc.copying.slice(grouped, [offsets[k], offsets[k + 1]])[0]
                _write(piece, columns, directory / _part(k) / name / f"halo-{group:04d}-{index:06d}.parquet")
                halo_per_partition[k] += offsets[k + 1] - offsets[k]
    for k in range(num_partitions if home else 0):
        if rows_per_partition[k] == 0:
            # A partition holds every table, if only as an empty file of its types.
            _write(template, columns, directory / _part(k) / name / "empty.parquet")
    return {"rows_read": read, "rows_written": sum(rows_per_partition), "unreachable_rows": unreachable,
            "copied_rows": copies, "halo_rows": sum(halo_per_partition), "rows_per_partition": rows_per_partition,
            "halo_rows_per_partition": halo_per_partition}


def _write(table: Any, columns: Mapping[str, DType], path: Path) -> None:
    """Write one piece of a partition, with ZSTD: every pass over the split reads these files again.

    The PRAGMA medium release (20 GiB of ZSTD Parquet) split into 8 partitions takes 27 GiB
    with ZSTD against 35 GiB with libcudf's default, Snappy.

    A bytes column is written as binary, as it was read (libcudf holds it as strings). libcudf writes no 256-bit
    decimal: a decimal256 column is written as a 128-bit one (precision up to 38) or, wider, as binary (its 32
    bytes a value), which a split's read takes back (`test.kumo_pytorch.reference_engine.gpu.data.sources.split_form`). A float column holding
    -0.0 is written without a dictionary: libcudf's dictionary takes -0.0 and 0.0
    as one value (they compare equal) and writes whichever it met first for both, so a sign upstream reads
    (a max or a first value of -0.0) would be lost. Other columns keep the dictionary's smaller files.
    """

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view

    import cupy
    import pylibcudf as plc

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = plc.io.types.TableInputMetadata(table)
    for i, (name, dtype) in enumerate(columns.items()):
        metadata.column_metadata[i].set_name(name)
        wide = dtype.name == "decimal256" and table.columns()[i].type().id() == plc.TypeId.STRING
        if dtype.precision is not None and not wide:
            metadata.column_metadata[i].set_decimal_precision(dtype.precision)
        if dtype.name == "bytes" or wide:  # libcudf holds both as strings: written as binary, as they were read
            metadata.column_metadata[i].set_output_as_binary(True)
        if dtype.name.startswith("float") and table.num_rows():
            values = _view(table.columns()[i])
            if bool(((values == 0) & cupy.signbit(values)).any()):
                metadata.column_metadata[i].set_encoding(plc.io.types.ColumnEncoding.PLAIN)
    options = (plc.io.parquet.ParquetWriterOptions.builder(plc.io.SinkInfo([str(path)]), table)
               .metadata(metadata).compression(plc.io.types.CompressionType.ZSTD).build())
    plc.io.parquet.write_parquet(options)
