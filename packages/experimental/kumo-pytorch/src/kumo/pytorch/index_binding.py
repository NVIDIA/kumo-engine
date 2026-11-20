# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read existing DiskGraph artifacts without their original table sources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec
from kumo.pytorch.materialization import _DiskGraphIndex, _fingerprint, _foreign_key

_CLOCKS = {"seconds": "s", "milliseconds": "ms", "microseconds": "us", "nanoseconds": "ns"}


def _read(root: Path, name: str) -> dict:
    try:
        value = json.loads((root / name).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot open DiskGraph metadata {root / name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"DiskGraph metadata {root / name} must be an object")
    return value


def _dtype(column: dict, clock: str, table: str) -> DType:
    name = column["ingested_dtype"]
    name = {"date": "date32", "utf8": "string"}.get(name, name.replace("u_int", "uint"))
    try:
        if name == "timestamp":
            return DType(name, unit=_CLOCKS[clock], timezone="UTC")
        if name.startswith("decimal"):
            return DType(name, precision=column["precision"], scale=column["scale"])
        if column.get("semantic_type") == "multicategorical":
            raise ValueError("multicategorical storage requires list support")
        return DType(name)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"unsupported stored type for {table}.{column['name']}: {column}"
        ) from error


@dataclass(frozen=True)
class IndexBinding:
    artifact: _DiskGraphIndex
    schema: DataSchema
    identity: str
    metadata_fingerprint: str
    physical_relations: tuple[tuple[RelationSpec, dict], ...]

    def resolve(self, schema: DataSchema) -> tuple[DataSchema, _DiskGraphIndex]:
        """Validate a projection and map consumer relation aliases to physical edges."""
        stored = {table.name: table for table in self.schema.tables}
        tables = []
        for table in schema.tables:
            actual = stored.get(table.name)
            if actual is None:
                raise ValueError(f"index has no node table {table.name!r}")
            if (table.primary_key, table.time_column) != (actual.primary_key, actual.time_column):
                raise ValueError(f"index keys/clock differ for table {table.name!r}")
            columns = actual.columns if table.columns is None else table.columns
            resolved = {}
            for name, dtype in columns.items():
                if name not in actual.columns:
                    raise ValueError(f"index has no column {table.name}.{name}")
                expected = actual.columns[name]
                if dtype is not None and dtype != expected:
                    raise ValueError(
                        f"index stored type differs for {table.name}.{name}: {expected}"
                    )
                resolved[name] = expected
            tables.append(replace(table, columns=resolved))
        mapping = {}
        definitions = {table.name: table for table in tables}
        for relation in schema.relations:
            try:
                child, fk, parent, forward = _foreign_key(relation, definitions)
            except KeyError as error:
                raise ValueError(
                    f"relation {relation.name!r} refers to an undeclared table"
                ) from error
            matches = [
                native
                for physical, native in self.physical_relations
                if (physical.left_table, physical.left_key, physical.right_table)
                == (child, fk, parent)
            ]
            if len(matches) != 1:
                raise ValueError(f"index has no unambiguous stored relation for {relation.name!r}")
            native = matches[0]
            mapping[relation.name] = {
                "left_table": relation.left_table,
                "right_table": relation.right_table,
                "forward": native["forward" if forward else "reverse"],
                "reverse": native["reverse" if forward else "forward"],
            }
        return DataSchema(tuple(tables), schema.relations), _DiskGraphIndex(
            self.artifact.path, {"relation_mapping": mapping}
        )

    def validate_files(self) -> None:
        root = Path(self.artifact.path)
        if _metadata_fingerprint(root) != self.metadata_fingerprint:
            raise ValueError("DiskGraph index metadata changed after binding; open it again")
        _check_complete(root)


def _check_complete(root: Path) -> None:
    import diskgraph

    if (root / "_MANIFEST.json").is_file():
        diskgraph.verify_snapshot_integrity(str(root))
    elif (root / "index_manifest.json").is_file():
        # PyTorch publishes this manifest by atomic rename after native ingest;
        # that producer does not request engine _MANIFEST/_SUCCESS markers.
        _DiskGraphIndex.open(root)
    else:
        raise ValueError(f"incomplete DiskGraph index: {root} has no completion manifest")


def _metadata_fingerprint(root: Path) -> str:
    return _fingerprint(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "diskgraph_config.json",
                "index_metadata.json",
                "_MANIFEST.json",
                "index_manifest.json",
                "_provenance.json",
            )
            if (root / name).is_file()
        }
    )


def open_index(path: str | Path, *, snapshot_id: str | None = None) -> IndexBinding:
    """Build an in-memory binding; write nothing and never follow stored source paths."""
    root = Path(path).resolve()
    if not (root / "diskgraph_config.json").is_file():
        raise ValueError(
            f"{root} has no diskgraph_config.json; automatic opening of this legacy "
            "index is unsupported (no schema is inferred from unavailable sources)"
        )
    _check_complete(root)
    config, metadata = _read(root, "diskgraph_config.json"), _read(root, "index_metadata.json")
    clock = metadata.get("timestamp_unit", "seconds")
    if clock not in _CLOCKS or config.get("timestamp_unit", "seconds") != clock:
        raise ValueError(f"unsupported or inconsistent DiskGraph clock: {clock!r}")
    nodes = {node["name"]: node for node in metadata["node_types"]}
    tables = []
    for table in config["tables"]:
        name = table["name"]
        if table.get("edge_table") is not None:
            raise ValueError(f"native edge table {name!r} is not supported by the PyTorch binding")
        node = nodes.get(name)
        if node is None:
            raise ValueError(f"config table {name!r} is absent from native node metadata")
        key, time = table.get("primary_key"), table.get("create_time")
        if key != (node.get("primary_key_map") or {}).get("column"):
            raise ValueError(f"inconsistent native primary key for {name!r}")
        if time != node.get("create_time_col") or table.get("end_time"):
            raise ValueError(f"inconsistent or unsupported native clock/end_time for {name!r}")
        tables.append(
            TableSpec(
                name,
                key,
                {col["name"]: _dtype(col, clock, name) for col in table["columns"]},
                time_column=time or NON_TEMPORAL,
            )
        )
    if set(nodes) != {table.name for table in tables}:
        raise ValueError("native node types differ from the resolved config")
    by_name = {table.name: table for table in tables}
    edges = {edge["name"]: edge for edge in metadata["edge_relations"]}
    physical = []
    for table in config["tables"]:
        for fk in table.get("fkeys", []):
            child, column, parent = table["name"], fk["src_column"], fk["dst_table"]
            if parent not in by_name or column not in by_name[child].columns:
                raise ValueError(f"invalid native foreign key {child}.{column} -> {parent}")
            pk = by_name[parent].primary_key
            # Native FK naming is defined by ingest; verify the name AND endpoints/pair.
            name = f"{child}_{column}_to_{parent}_{pk}"
            edge = edges.get(name)
            reverse = edges.get((edge or {}).get("reverse_of"))
            if (
                edge is None
                or reverse is None
                or (edge["src_type"], edge["dst_type"]) != (child, parent)
                or (reverse["src_type"], reverse["dst_type"]) != (parent, child)
                or reverse.get("reverse_of") != name
            ):
                raise ValueError(f"missing or inconsistent native FK edge pair for {name!r}")
            for direction in (edge, reverse):
                target = by_name[direction["dst_type"]]
                expected = (
                    None
                    if target.time_column == NON_TEMPORAL
                    else (target.name, target.time_column)
                )
                timestamp = (direction.get("dest_sort_key") or {}).get("timestamp")
                actual = None if timestamp is None else (timestamp["table"], timestamp["column"])
                if actual != expected:
                    raise ValueError(f"unsupported stored edge clock for {direction['name']!r}")
            physical.append(
                (
                    RelationSpec(name, child, column, parent, pk),
                    {
                        "left_table": child,
                        "right_table": parent,
                        "forward": name,
                        "reverse": reverse["name"],
                    },
                )
            )
    legacy = _DiskGraphIndex.open(root).manifest if (root / "index_manifest.json").is_file() else {}
    relations = []
    for physical_relation, mapping in physical:
        aliases = [
            (alias, entry)
            for alias, entry in legacy.get("relation_mapping", {}).items()
            if entry["forward"] in (mapping["forward"], mapping["reverse"])
        ]
        if not aliases:
            relations.append(physical_relation)
        for alias, entry in aliases:
            if entry["forward"] == mapping["forward"]:
                relations.append(replace(physical_relation, name=alias))
            else:
                relations.append(
                    RelationSpec(
                        alias,
                        physical_relation.right_table,
                        physical_relation.right_key,
                        physical_relation.left_table,
                        physical_relation.left_key,
                    )
                )
    schema = DataSchema(tuple(tables), tuple(relations))
    provenance = _read(root, "_provenance.json") if (root / "_provenance.json").is_file() else {}
    recorded_id = provenance.get("id")
    if snapshot_id and recorded_id and snapshot_id != recorded_id:
        raise ValueError("snapshot identity differs from persisted provenance")
    identity = metadata.get("index_build_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            "snapshot has no physical index_build_id; legacy indexes cannot be adopted"
        )
    binding = IndexBinding(
        _DiskGraphIndex(str(root), {}),
        schema,
        str(identity),
        _metadata_fingerprint(root),
        tuple(physical),
    )
    _, artifact = binding.resolve(schema)
    return replace(binding, artifact=artifact)
