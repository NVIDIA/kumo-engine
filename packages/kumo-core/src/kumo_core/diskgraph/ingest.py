# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict, service-independent DiskGraph ingest boundary."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types.schema import TS_FORMAT_METADATA
from ._progress import log_progress
from .runner import (
    IngestExecutionConfig,
    IngestRunResult,
    run_ingest_configured,
)
from .snapshot import DiskgraphIngestResult

logger = logging.getLogger(__name__)

_MULTICATEGORICAL_INGESTED_LIST_DTYPES = frozenset({
    "string_array",
    "int64_array",
})


class DiskgraphInputError(ValueError):
    """A caller crossed the core boundary with a non-canonical input."""


def _require_nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiskgraphInputError(f"{location} must be a non-empty string")
    return value


def validate_canonical_diskgraph_graph(
        graph_config_json: Mapping[str, Any]) -> None:
    """Reject legacy graph *shape* aliases; column aliases are resolved.

    Table-level aliases stay rejected, because a caller spelling a source
    two ways is ambiguous and core cannot pick for them. Column-level
    aliases are different: ``normalize_legacy_column_aliases`` resolves
    them, since ``dtype``/``metadata`` describe the source and the
    converter wants storage, which is a translation with one right answer
    rather than a choice.
    """
    if not isinstance(graph_config_json, Mapping):
        raise DiskgraphInputError("graph must be a mapping")
    _require_nonempty_string(graph_config_json.get("name"), "graph.name")
    tables = graph_config_json.get("tables")
    if not isinstance(tables, list) or not tables:
        raise DiskgraphInputError("graph.tables must be a non-empty list")
    table_names: set[str] = set()
    destinations: list[tuple[str, str]] = []
    for index, table in enumerate(tables):
        location = f"graph.tables[{index}]"
        if not isinstance(table, Mapping):
            raise DiskgraphInputError(f"{location} must be a mapping")
        for alias in ("source_name", "data_source", "source_table"):
            if alias in table:
                raise DiskgraphInputError(
                    f"{location}.{alias} is a legacy alias; provide exact "
                    "top-level source_table_name and connector_id fields")
        table_name = _require_nonempty_string(table.get("table_name"),
                                              f"{location}.table_name")
        _require_nonempty_string(table.get("source_table_name"),
                                 f"{location}.source_table_name")
        if table_name in table_names:
            raise DiskgraphInputError(
                f"graph contains duplicate table_name {table_name!r}")
        table_names.add(table_name)

        primary_key = table.get("pkey")
        if primary_key is not None:
            if not isinstance(primary_key, Mapping):
                raise DiskgraphInputError(f"{location}.pkey must be a mapping")
            for field_name in ("name", "dtype", "stype"):
                _require_nonempty_string(primary_key.get(field_name),
                                         f"{location}.pkey.{field_name}")

        columns = table.get("cols")
        if not isinstance(columns, list):
            raise DiskgraphInputError(f"{location}.cols must be a list")
        for column_index, column in enumerate(columns):
            column_location = f"{location}.cols[{column_index}]"
            if not isinstance(column, Mapping):
                raise DiskgraphInputError(
                    f"{column_location} must be a mapping")
            for field_name in ("name", "dtype", "stype"):
                _require_nonempty_string(column.get(field_name),
                                         f"{column_location}.{field_name}")

        foreign_keys = table.get("fkeys")
        if not isinstance(foreign_keys, list):
            raise DiskgraphInputError(f"{location}.fkeys must be a list")
        for fkey_index, foreign_key in enumerate(foreign_keys):
            fkey_location = f"{location}.fkeys[{fkey_index}]"
            if not isinstance(foreign_key, Mapping):
                raise DiskgraphInputError(f"{fkey_location} must be a mapping")
            for field_name in ("name", "dtype", "stype", "dst_table",
                               "dst_key"):
                _require_nonempty_string(foreign_key.get(field_name),
                                         f"{fkey_location}.{field_name}")
            if "." in foreign_key["dst_table"]:
                raise DiskgraphInputError(
                    f"{fkey_location}.dst_table must be the exact graph "
                    "table_name, not a qualified legacy alias")
            destinations.append((fkey_location, foreign_key["dst_table"]))
    for location, destination in destinations:
        if destination not in table_names:
            raise DiskgraphInputError(
                f"{location}.dst_table references unknown graph table_name "
                f"{destination!r}")


def validate_resolved_diskgraph_inputs(
    graph_config_json: Mapping[str, Any],
    table_data_versions: Mapping[str, str],
    source_paths: Mapping[str, str],
) -> None:
    """Validate the exact execution handoff accepted by DiskGraph core."""
    validate_canonical_diskgraph_graph(graph_config_json)
    table_names: set[str] = set()
    for index, table in enumerate(graph_config_json["tables"]):
        table_name = str(table["table_name"])
        table_names.add(table_name)
        connector_id = _require_nonempty_string(
            table.get("connector_id"), f"graph.tables[{index}].connector_id")
        if connector_id == "default":
            raise DiskgraphInputError(
                f"graph.tables[{index}].connector_id must be resolved; "
                "'default' is not canonical")
    for label, values in (("table_data_versions", table_data_versions),
                          ("source_paths", source_paths)):
        if not isinstance(values, Mapping):
            raise DiskgraphInputError(f"{label} must be a mapping")
        keys = set(values)
        if keys != table_names:
            raise DiskgraphInputError(
                f"{label} keys must exactly match graph table_name values; "
                f"expected {sorted(table_names)!r}, got {sorted(keys)!r}")
        for table_name, value in values.items():
            _require_nonempty_string(value, f"{label}[{table_name!r}]")


def normalize_legacy_column_aliases(graph_config_json: dict[str, Any]) -> None:
    """Rewrite source-side column aliases into converter fields in place.

    Kumo describes a column by how the customer's warehouse spells it;
    DiskGraph's converter wants how the column is stored once ingested. Two
    aliases differ between those views and every caller has to translate them
    before handing a graph to ``DiskgraphConfigConverter``:

    ``dtype=string`` + ``stype=sequence`` is a float array serialized as text.
    DiskGraph parses it into floats at ingest time, so the converter must be
    told the storage dtype ``floatlist``.

    ``metadata[TS_FORMAT]`` is where kumo records a user-declared chrono
    format; the converter reads ``timestamp_format``. The alias is consumed
    rather than copied so a prepared config carries one spelling of the fact.
    Injection is write-once: an explicit ``timestamp_format`` wins. A
    ``TS_FORMAT`` that merely echoes the column's ``timestamp_unit`` (e.g.
    ``"s"`` for a seconds-epoch column) is a UNIT, not a strftime format --
    the unit already rides ``timestamp_unit`` and DiskGraph rejects a bare
    unit token as an invalid chrono format -- so it is dropped rather than
    injected, mirroring kumo's own ``TS_FORMAT != timestamp_unit`` guard.

    Only ``cols`` participate: pkey/fkey are ``stype=ID`` by product
    constraint and cannot carry a timestamp format.
    """
    for table in (graph_config_json.get("tables") or []):
        for column in (table.get("cols") or []):
            if not isinstance(column, dict):
                continue
            if (column.get("dtype") == "string"
                    and column.get("stype") == "sequence"):
                column["dtype"] = "floatlist"
            metadata = column.get("metadata")
            if not isinstance(metadata, dict):
                continue
            timestamp_format = metadata.pop(TS_FORMAT_METADATA, None)
            if not metadata:
                column.pop("metadata", None)
            if not isinstance(timestamp_format, str):
                continue
            timestamp_format = timestamp_format.strip()
            unit = str(column.get("timestamp_unit") or "").strip()
            if timestamp_format and timestamp_format != unit:
                column.setdefault("timestamp_format", timestamp_format)


def prepare_diskgraph_converter_config(
    graph_config_json: dict[str, Any], ) -> dict[str, Any]:
    """Translate the canonical graph to the private wheel input contract."""
    validate_canonical_diskgraph_graph(graph_config_json)
    converter_config = copy.deepcopy(graph_config_json)
    normalize_legacy_column_aliases(converter_config)
    for table in converter_config["tables"]:
        table["data_source"] = table.pop("connector_id")
    return converter_config


def validate_diskgraph_ingest_config(ingest_config: dict[str, Any]) -> None:
    for table in ingest_config.get("tables", []):
        for column in table.get("columns", []):
            if column.get("semantic_type") != "multicategorical":
                continue
            ingested_dtype = column.get("ingested_dtype")
            if ingested_dtype not in _MULTICATEGORICAL_INGESTED_LIST_DTYPES:
                raise ValueError(
                    "DiskGraph multicategorical columns require a supported "
                    "string/int list-shaped ingested dtype after conversion; "
                    f"table '{table.get('name', '<unknown>')}', column "
                    f"'{column.get('name', '<unknown>')}' remained "
                    f"'{ingested_dtype}'. Use categorical for one token per "
                    "row, configure a supported comma/pipe/semicolon/tab "
                    "delimiter, or use a native stringlist/intlist dtype.")


@dataclass(frozen=True)
class DiskgraphIngestSpec:
    """Fully resolved inputs for one DiskGraph snapshot ingest."""

    data_persist_dir: str
    table_data_versions: Mapping[str, str]
    output_dir: str
    source_paths: Mapping[str, str]
    execution: IngestExecutionConfig = field(
        default_factory=IngestExecutionConfig)


IngestRunner = Callable[
    [dict[str, Any], Path, IngestExecutionConfig],
    IngestRunResult,
]


def execute_diskgraph_ingest(
    graph_config_json: dict[str, Any],
    spec: DiskgraphIngestSpec,
    *,
    runner: IngestRunner = run_ingest_configured,
) -> DiskgraphIngestResult:
    """Convert an exact resolved graph and execute the ingest binary."""
    validate_resolved_diskgraph_inputs(graph_config_json,
                                       spec.table_data_versions,
                                       spec.source_paths)
    from diskgraph import DiskgraphConfigConverter

    output_path = Path(spec.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    converter_config = prepare_diskgraph_converter_config(graph_config_json)
    ingest_config = DiskgraphConfigConverter(
        converter_config,
        spec.data_persist_dir,
        dict(spec.table_data_versions),
        source_paths=dict(spec.source_paths),
    ).convert()
    validate_diskgraph_ingest_config(ingest_config)

    run_result = runner(ingest_config, output_path, spec.execution)
    log_progress(
        logger,
        "diskgraph ingest succeeded after %d attempt(s) at %d threads",
        run_result.attempts,
        run_result.final_thread_count,
    )
    stats = run_result.stats.get("stats") or {}
    log_progress(
        logger,
        "diskgraph ingest completed: %d node tables, %d edge relations, "
        "%d nodes, %d edges",
        stats.get("node_tables", 0),
        stats.get("edge_relations", 0),
        stats.get("total_nodes", 0),
        stats.get("total_edges", 0),
    )

    result = DiskgraphIngestResult.for_index(output_path)
    missing = result.missing_paths()
    if missing:
        raise RuntimeError(
            "diskgraph ingest completed but missing expected output paths: " +
            ", ".join(missing))
    return result
