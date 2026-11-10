# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recorder in place of ``diskgraph.ingest``, and what this package's ingest config must say.

The recorder replaces the one native call kumo-pytorch makes while ensuring an index
(`materialization._import_diskgraph`), so the config and the index lifecycle are
checked on the CPU job, with no DiskGraph wheel. It cannot check native semantics.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from kumo.pytorch.api import DataSchema, Dataset, Parquet, RelationSpec, TableSpec
from kumo.pytorch.contract.schema import table_clock

IngestOption = bool | int | str | None
#: Arrow types the conformance Parquet writer uses, as DiskGraph names them.
INGESTED_DTYPES: Mapping[str, str] = {
    "int64": "int64",
    "double": "float64",
    "string": "string",
}
#: Fixed ingest options: no column statistics or category counts (kumo-pytorch fits its own).
INGEST_OPTIONS: Mapping[str, IngestOption] = {
    "col_stats": False,
    "cat_count_strategy": "none",
    "output_dataplane_dir": None,
    "success_marker_content": None,
    "resume_from_phase": None,
    "parent_index_path": None,
}
THREAD_OPTIONS = (
    "context_threads",
    "encode_threads",
    "pk_map_threads",
    "edges_map_threads",
    "index_threads",
)


class TimeRef(TypedDict):
    table: str
    column: str
    unit: str


class Direction(TypedDict):
    time: TimeRef


class FKey(TypedDict, total=False):
    src_column: str
    dst_table: str
    bidirectional: bool
    forward: Direction
    reverse: Direction


class ColumnConfig(TypedDict, total=False):
    name: str
    ingested_dtype: str
    precision: int
    scale: int


class SourceConfig(TypedDict):
    type: str
    paths: list[str]


class TableConfig(TypedDict, total=False):
    name: str
    columns: list[ColumnConfig]
    primary_key: str
    source: SourceConfig
    fkeys: list[FKey]
    create_time: str


class IngestConfig(TypedDict):
    dataset_name: str
    tables: list[TableConfig]
    num_shards: int
    edge_format: str


class EdgeRelation(TypedDict):
    name: str
    src_type: str
    dst_type: str


class RelationMapping(TypedDict):
    left_table: str
    right_table: str
    forward: str
    reverse: str


@dataclass(frozen=True)
class IngestCall:
    config: IngestConfig
    output_dir: Path
    options: Mapping[str, IngestOption]


@dataclass
class FakeDiskGraph:
    """Stands in for the ``diskgraph`` module at kumo-pytorch's ingest call.

    Each call is recorded, then the output directory gets what kumo-pytorch and a sampler
    read back: ``index_metadata.json`` (with DiskGraph's edge naming),
    ``ingest_stats.json`` and ``diskgraph_config.json``. ``during`` runs inside
    the call once the output exists; it may raise, change a source file or create
    the destination.
    """

    during: Callable[[IngestCall], None] | None = None
    dropped: Mapping[str, int] = field(default_factory=dict)
    calls: list[IngestCall] = field(default_factory=list)

    def ingest(
        self,
        config_json: str,
        output_dir: str,
        col_stats: bool = True,
        cat_count_strategy: str = "hybrid",
        output_dataplane_dir: str | None = None,
        success_marker_content: str | None = None,
        resume_from_phase: str | None = None,
        context_threads: int | None = None,
        encode_threads: int | None = None,
        pk_map_threads: int | None = None,
        edges_map_threads: int | None = None,
        index_threads: int | None = None,
        parent_index_path: str | None = None,
    ) -> None:
        """The parameters of the pinned ``diskgraph.ingest``; an unknown keyword fails."""

        config: IngestConfig = json.loads(config_json)
        options: dict[str, IngestOption] = {
            "col_stats": col_stats,
            "cat_count_strategy": cat_count_strategy,
            "output_dataplane_dir": output_dataplane_dir,
            "success_marker_content": success_marker_content,
            "resume_from_phase": resume_from_phase,
            "context_threads": context_threads,
            "encode_threads": encode_threads,
            "pk_map_threads": pk_map_threads,
            "edges_map_threads": edges_map_threads,
            "index_threads": index_threads,
            "parent_index_path": parent_index_path,
        }
        call = IngestCall(config, Path(output_dir), options)
        self.calls.append(call)
        call.output_dir.mkdir()
        (call.output_dir / "diskgraph_config.json").write_text(config_json)
        if self.during is not None:
            self.during(call)
        metadata = {
            "dataset_name": config["dataset_name"],
            "timestamp_unit": "seconds",
            "edge_relations": edge_relations(config),
        }
        (call.output_dir / "index_metadata.json").write_text(json.dumps(metadata))
        stats = {"stats": {"dropped_rows_counts": dict(self.dropped)}}
        (call.output_dir / "ingest_stats.json").write_text(json.dumps(stats))


@contextmanager
def recording(fake: FakeDiskGraph) -> Iterator[FakeDiskGraph]:
    with patch("kumo.pytorch.materialization._import_diskgraph", return_value=fake):
        yield fake


def edge_relations(config: IngestConfig) -> list[EdgeRelation]:
    """DiskGraph's naming: FK table → PK table, and ``_reverse`` back (checked on 55d179a)."""

    keys = {table["name"]: table["primary_key"] for table in config["tables"]}
    edges: list[EdgeRelation] = []
    for table in config["tables"]:
        for fkey in table["fkeys"]:
            dst = fkey["dst_table"]
            name = f"{table['name']}_{fkey['src_column']}_to_{dst}_{keys[dst]}"
            edges.append({"name": name, "src_type": table["name"], "dst_type": dst})
            edges.append({"name": f"{name}_reverse", "src_type": dst, "dst_type": table["name"]})
    return edges


# -- What the config must say ------------------------------------------------


def parquet_files(source: Parquet) -> list[str]:
    """The conformance layouts: one file, or a directory of part files."""

    path = Path(source.path).resolve()
    return (
        sorted(str(p.resolve()) for p in path.rglob("*.parquet")) if path.is_dir() else [str(path)]
    )


def foreign_key(relation: RelationSpec, tables: Mapping[str, TableSpec]) -> tuple[str, str, str]:
    """(FK table, FK column, PK table) of a relation with one primary-key side."""

    if relation.right_key == tables[relation.right_table].primary_key:
        return relation.left_table, relation.left_key, relation.right_table
    return relation.right_table, relation.right_key, relation.left_table


def _direction(table: TableSpec) -> Direction | None:
    clock = table_clock(table)
    return None if clock is None else {"time": {"table": table.name, "column": clock, "unit": "s"}}


def ingested_dtype(arrow_type: str) -> str | None:
    """DiskGraph's name for an Arrow type; a timestamp of any unit (Parquet reads
    ``timestamp[s]`` back as ``timestamp[ms]``). None for a type outside the table.
    """

    if arrow_type.startswith("timestamp["):
        return "timestamp"
    return INGESTED_DTYPES.get(arrow_type)


def option_violations(options: Mapping[str, IngestOption], *, num_threads: int | None) -> list[str]:
    """Every keyword of the ingest call that departs from the fixed options and threads."""

    expected = {**INGEST_OPTIONS, **{name: num_threads for name in THREAD_OPTIONS}}
    return [
        f"{name}={options.get(name)!r} != {value!r}"
        for name, value in expected.items()
        if options.get(name) != value or type(options.get(name)) is not type(value)
    ]


def config_violations(
    schema: DataSchema,
    dataset: Dataset,
    config: IngestConfig,
    arrow_types: Mapping[str, Mapping[str, str]],
    *,
    num_shards: int,
    dataset_name: str = "default",
) -> list[str]:
    """Every way ``config`` departs from the schema; ``arrow_types`` per table and column.

    Columns are the declared ones, by name, each with a known Arrow type; the clock
    is ``create_time``; each FK column has one fkey, both directions timed by their
    destination's clock; edges are written in DiskGraph's binary format.
    """

    problems: list[str] = []
    tables = {table.name: table for table in schema.tables}
    if config["num_shards"] != num_shards:
        problems.append(f"num_shards {config['num_shards']} != {num_shards}")
    if config["dataset_name"] != dataset_name:
        problems.append(f"dataset_name {config['dataset_name']!r} != {dataset_name!r}")
    if config["edge_format"] != "binary":
        problems.append(f"edge_format {config['edge_format']!r} != 'binary'")
    names = [table["name"] for table in config["tables"]]
    if names != list(tables):
        problems.append(f"tables {names} != {list(tables)}")
    for table in config["tables"]:
        spec = tables.get(table["name"])
        if spec is None:
            continue
        label = spec.name
        columns = [column["name"] for column in table["columns"]]
        if columns != sorted(spec.columns):
            problems.append(f"{label}: columns {columns} != declared {sorted(spec.columns)}")
        for column in table["columns"]:
            arrow = arrow_types[label].get(column["name"])
            expected = None if arrow is None else ingested_dtype(arrow)
            if expected is None:
                problems.append(f"{label}.{column['name']}: Arrow type {arrow} is not in the table")
            elif column.get("ingested_dtype") != expected:
                problems.append(f"{label}.{column['name']}: dtype {column.get('ingested_dtype')}")
        if table["primary_key"] != spec.primary_key:
            problems.append(f"{label}: primary key {table['primary_key']}")
        if table.get("create_time") != table_clock(spec):
            problems.append(f"{label}: create_time {table.get('create_time')}")
        source = dataset.tables[label]
        assert isinstance(source, Parquet)
        if table["source"] != {"type": "parquet", "paths": parquet_files(source)}:
            problems.append(f"{label}: source {table['source']}")
    declared = {foreign_key(relation, tables)[:2] for relation in schema.relations}
    emitted = Counter(
        (table["name"], fkey["src_column"]) for table in config["tables"] for fkey in table["fkeys"]
    )
    for (fk_table, column), count in sorted(emitted.items()):
        if count != 1:
            problems.append(f"{fk_table}.{column}: {count} fkeys")
    if set(emitted) != declared:
        problems.append(f"fkeys on {sorted(emitted)} != FK columns {sorted(declared)}")
    for relation in schema.relations:
        fk_table, column, pk_table = foreign_key(relation, tables)
        expected_fkey: FKey = {"src_column": column, "dst_table": pk_table, "bidirectional": True}
        for direction, destination in (("forward", pk_table), ("reverse", fk_table)):
            timed = _direction(tables[destination])
            if timed is not None:
                expected_fkey[direction] = timed
        fkeys = next((t["fkeys"] for t in config["tables"] if t["name"] == fk_table), [])
        for fkey in fkeys:
            if fkey["src_column"] == column and fkey != expected_fkey:
                problems.append(f"{fk_table}.{column}: fkey {fkey} != {expected_fkey}")
    return problems


def mapping_violations(
    schema: DataSchema,
    mapping: Mapping[str, RelationMapping],
    edges: Sequence[EdgeRelation],
) -> list[str]:
    """Each logical direction names a physical edge with the same endpoints."""

    problems: list[str] = []
    by_name = {edge["name"]: edge for edge in edges}
    if set(mapping) != {relation.name for relation in schema.relations}:
        problems.append(f"mapped relations {sorted(mapping)}")
    for relation in schema.relations:
        entry = mapping.get(relation.name)
        if entry is None:
            continue
        if (entry["left_table"], entry["right_table"]) != (
            relation.left_table,
            relation.right_table,
        ):
            problems.append(
                f"{relation.name}: tables {entry['left_table']}, {entry['right_table']}"
            )
        for direction, src, dst in (
            ("forward", relation.left_table, relation.right_table),
            ("reverse", relation.right_table, relation.left_table),
        ):
            edge = by_name.get(entry[direction])
            if edge is None or (edge["src_type"], edge["dst_type"]) != (src, dst):
                problems.append(
                    f"{relation.name}.{direction}: {entry[direction]} is not {src}→{dst}"
                )
    return problems
