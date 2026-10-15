# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable predictive-query validation and node/link task planning.

The public inputs are plain mappings and strings.  ``kumo-pql`` and
``kumo-api`` remain the authoritative parser and validator, but the platform
``Graph``, ``PredictiveQueryV2``, ``Task``, StateDB, and service caches are not
part of this boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from typing import Any, Mapping, Sequence

from diskgraph import read_col_stats
from kumoapi.common import ValidationResponse
from kumoapi.encoder import EncoderType
from kumoapi.graph import (
    ColumnKey,
    GraphDefinition,
    group_column_relationships,
)
from kumoapi.pquery import QueryType
from kumoapi.pquery.AST import DateOffsetRange
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import Column, TableDefinition
from kumoapi.typing import ColStatType, Dtype, Stype
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator
from pandas import DateOffset, Timedelta, Timestamp
from pydantic import TypeAdapter

from .encoder_selection import (
    _EncoderStat,
    feature_column_names,
    select_encoder,
)
from .model_plan import (
    TrainingTableGenerationPlan,
    inferred_or_missing_type_to_false,
    inferred_or_missing_type_to_none,
)
from .task_dataset import TaskDatasetMaterializeSpec
from .train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
    ONE_YEAR_AGO_OFFSET_DAYS,
    PADDING_SIZE,
    YOY_COL,
)
from .training.splits import (
    SplitArguments,
    TrainTableTime,
    default_split_for_query,
    normalize_static_link_split,
    train_table_time,
)
from .training.table import (
    TrainTableRequest,
    split_config_from_descriptor,
    split_from_repr,
)
from .training.timestamps import (
    default_target_time_range,
    generate_legacy_timestamps,
)
from .training.validation import validate_train_offsets
from .types import TaskType


class PredictiveQueryPlanError(ValueError):
    """A notebook graph/query cannot be converted to a native task plan."""


_NATIVE_TRAINING_TASK_TYPES = frozenset({
    "node_prediction",
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
    "regression",
    "forecasting",
    "link_prediction",
    "static_link_prediction",
    "temporal_link_prediction",
})


@dataclass(frozen=True)
class NativeGraphColumn:
    name: str
    dtype: str
    stype: str
    encoder: Mapping[str, Any] | None = None
    timestamp_format: str | None = None
    multicategorical_delimiter: str | None = None


@dataclass(frozen=True)
class NativeGraphForeignKey:
    name: str
    destination_table: str
    destination_column: str


@dataclass(frozen=True)
class NativeGraphTable:
    name: str
    source_table_name: str
    columns: tuple[NativeGraphColumn, ...]
    primary_key: str | None
    time_column: str | None
    end_time_column: str | None
    foreign_keys: tuple[NativeGraphForeignKey, ...]

    def column(self, name: str) -> NativeGraphColumn:
        for column in self.columns:
            if column.name == name:
                return column
        raise PredictiveQueryPlanError(
            f"graph table {self.name!r} has no column {name!r}")


@dataclass(frozen=True)
class NativeGraph:
    tables: tuple[NativeGraphTable, ...]

    def table(self, name: str) -> NativeGraphTable:
        bare_name = name.rsplit(".", 1)[-1]
        matches = [table for table in self.tables if table.name == bare_name]
        if len(matches) != 1:
            raise PredictiveQueryPlanError(
                f"graph has no unique table named {name!r}")
        return matches[0]

    def foreign_key(self, column_fqn: str) -> NativeGraphForeignKey | None:
        table_name, column_name = split_column_fqn(column_fqn)
        for foreign_key in self.table(table_name).foreign_keys:
            if foreign_key.name == column_name:
                return foreign_key
        return None


@dataclass(frozen=True)
class ValidatedNativeQuery:
    """Validated PQL plus the portable metadata needed by later stages."""

    query: Any
    task_type: str
    task_family: str
    entity_table: str
    entity_column: str
    target_stype: str
    query_type: str
    rhs_node_type: str | None
    rhs_is_temporal: bool
    list_distinct_target_is_array: bool
    forecast_length: int
    validation_response: ValidationResponse = dataclass_field(
        default_factory=ValidationResponse, compare=False)


@dataclass(frozen=True)
class NativePqueryTaskPlan:
    train_table: TrainTableRequest
    task_dataset: TaskDatasetMaterializeSpec
    graph_index_id: str
    graph_index_path: str


def stable_fingerprint(value: Any) -> str:
    """Return the shared deterministic fingerprint for notebook inputs."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _field_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        raw = value.get("name")
        return str(raw) if raw else None
    raise PredictiveQueryPlanError(
        "portable graph fields must be strings or resolved mappings")


def _bare_table_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def split_column_fqn(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) < 2:
        raise PredictiveQueryPlanError(
            f"expected a table-qualified column, got {value!r}")
    return parts[-2], parts[-1]


def _column_from_payload(value: Any, ) -> NativeGraphColumn:
    if not isinstance(value, Mapping):
        raise PredictiveQueryPlanError(
            "portable graph columns must be fully resolved mappings")
    name = _field_name(value)
    if not name:
        raise PredictiveQueryPlanError("graph column is missing its name")
    if "dtype" not in value or "stype" not in value:
        raise PredictiveQueryPlanError(
            f"portable graph column {name!r} requires dtype and stype")
    return NativeGraphColumn(
        name=name,
        dtype=_enum_value(value["dtype"]).rsplit(".", 1)[-1],
        stype=_enum_value(value["stype"]).rsplit(".", 1)[-1],
        encoder=(dict(value["encoder"])
                 if isinstance(value.get("encoder"), Mapping) else None),
        timestamp_format=(_enum_value(value["timestamp_format"]) if
                          value.get("timestamp_format") is not None else None),
        multicategorical_delimiter=(
            None if value.get("multicategorical_delimiter") is None else
            _enum_value(value["multicategorical_delimiter"])),
    )


def normalize_graph(graph: Mapping[str, Any]) -> NativeGraph:
    """Validate the one fully resolved portable graph JSON contract."""

    raw_tables = graph.get("tables")
    if (not isinstance(raw_tables, Sequence)
            or isinstance(raw_tables, (str, bytes)) or not raw_tables):
        raise PredictiveQueryPlanError(
            "portable graph requires a non-empty tables list")

    tables: list[NativeGraphTable] = []
    for raw_table in raw_tables:
        if not isinstance(raw_table, Mapping):
            raise PredictiveQueryPlanError("graph tables must be mappings")
        forbidden = {
            "name",
            "source_name",
            "data_source",
            "source_table",
            "columns",
            "primary_key",
            "foreign_keys",
            "time_column",
            "end_time_column",
        } & set(raw_table)
        if forbidden:
            raise PredictiveQueryPlanError(
                "portable graph table contains legacy aliases: " +
                ", ".join(sorted(forbidden)))
        if not raw_table.get("table_name"):
            raise PredictiveQueryPlanError(
                "portable graph table requires table_name")
        name = str(raw_table["table_name"])
        name = _bare_table_name(name)
        source_table_name = raw_table.get("source_table_name")
        if (not isinstance(source_table_name, str)
                or len(source_table_name.split(".")) != 3):
            raise PredictiveQueryPlanError(
                f"portable graph table {name!r} requires canonical "
                "source_table_name 'catalog.schema.table'")
        primary_key = _field_name(raw_table.get("pkey"))
        time_column = _field_name(raw_table.get("time_col"))
        end_time_column = _field_name(raw_table.get("end_time_col"))

        columns: dict[str, NativeGraphColumn] = {}
        if "cols" not in raw_table:
            raise PredictiveQueryPlanError(
                f"portable graph table {name!r} requires an explicit cols list")
        raw_columns = raw_table["cols"]
        if not isinstance(raw_columns, Sequence) or isinstance(
                raw_columns, (str, bytes)):
            raise PredictiveQueryPlanError(
                f"portable graph table {name!r} requires a cols list")
        for raw_column in raw_columns:
            column = _column_from_payload(raw_column)
            columns[column.name] = column

        def add_special(field: Any) -> None:
            column_name = _field_name(field)
            if not column_name:
                return
            column = _column_from_payload(field)
            columns.setdefault(column_name, column)

        add_special(raw_table.get("pkey"))
        add_special(raw_table.get("time_col"))
        add_special(raw_table.get("end_time_col"))

        foreign_keys: list[NativeGraphForeignKey] = []
        if "fkeys" not in raw_table:
            raise PredictiveQueryPlanError(
                f"portable graph table {name!r} requires an explicit fkeys list"
            )
        raw_foreign_keys = raw_table["fkeys"]
        if not isinstance(raw_foreign_keys, Sequence) or isinstance(
                raw_foreign_keys, (str, bytes)):
            raise PredictiveQueryPlanError(
                f"portable graph table {name!r} requires an fkeys list")
        for raw_foreign_key in raw_foreign_keys:
            if not isinstance(raw_foreign_key, Mapping):
                raise PredictiveQueryPlanError(
                    "graph foreign keys must be mappings")
            forbidden_foreign_key = {
                "column_name", "destination_table", "destination_column"
            } & set(raw_foreign_key)
            if forbidden_foreign_key:
                raise PredictiveQueryPlanError(
                    "portable foreign key contains legacy aliases: " +
                    ", ".join(sorted(forbidden_foreign_key)))
            source_name = _field_name(raw_foreign_key)
            destination = raw_foreign_key.get("dst_table")
            destination_column = raw_foreign_key.get("dst_key")
            if not source_name or not destination or not destination_column:
                raise PredictiveQueryPlanError(
                    f"portable foreign key in {name!r} requires name, "
                    "dst_table, and dst_key")
            if "." in str(destination):
                raise PredictiveQueryPlanError(
                    "portable foreign-key dst_table must be an unqualified "
                    "graph table_name")
            destination_name = _bare_table_name(destination)
            foreign_keys.append(
                NativeGraphForeignKey(
                    source_name,
                    destination_name,
                    str(destination_column),
                ))
            add_special(raw_foreign_key)

        tables.append(
            NativeGraphTable(
                name=name,
                source_table_name=source_table_name,
                columns=tuple(columns.values()),
                primary_key=primary_key,
                time_column=time_column,
                end_time_column=end_time_column,
                foreign_keys=tuple(foreign_keys),
            ))
    if not tables:
        raise PredictiveQueryPlanError("graph must contain at least one table")

    by_name = {table.name: table for table in tables}
    if len(by_name) != len(tables):
        raise PredictiveQueryPlanError("graph contains duplicate table names")
    for table in tables:
        key_names = ([table.primary_key] if table.primary_key else []) + [
            key.name for key in table.foreign_keys
        ]
        for name in key_names:
            if Dtype(table.column(name).dtype).is_float():
                raise PredictiveQueryPlanError(
                    f"key {table.name}.{name} has a floating-point dtype; "
                    "use an integer or string key before snapshot ingestion")
        for foreign_key in table.foreign_keys:
            destination = by_name.get(foreign_key.destination_table)
            if destination is None:
                raise PredictiveQueryPlanError(
                    f"foreign key {table.name}.{foreign_key.name} references "
                    f"missing table {foreign_key.destination_table!r}")
            if foreign_key.destination_column != destination.primary_key:
                raise PredictiveQueryPlanError(
                    f"foreign key {table.name}.{foreign_key.name} must resolve "
                    f"to primary key {destination.name}.{destination.primary_key}"
                )
            source_dtype = Dtype(table.column(foreign_key.name).dtype)
            destination_dtype = Dtype(
                destination.column(foreign_key.destination_column).dtype)
            if not source_dtype.can_link_to_dtype(destination_dtype):
                raise PredictiveQueryPlanError(
                    f"foreign key {table.name}.{foreign_key.name} "
                    f"(dtype={source_dtype}) cannot link to primary key "
                    f"{destination.name}.{destination.primary_key} "
                    f"(dtype={destination_dtype}); use matching datatypes "
                    "or lossless integer widening")
    return NativeGraph(tuple(tables))


def to_api_graph(graph: NativeGraph) -> Any:
    """Project the portable graph onto the authoritative validator model."""

    tables = {}
    relationships = []
    for table in graph.tables:
        columns = [
            Column(
                name=column.name,
                stype=Stype(column.stype),
                dtype=Dtype(column.dtype),
                timestamp_format=column.timestamp_format,
                multicategorical_delimiter=(column.multicategorical_delimiter),
            ) for column in table.columns
        ]
        tables[table.name] = TableDefinition(
            cols=columns,
            source_table=UnavailableSourceTable(table=table.name),
            pkey=table.primary_key,
            time_col=table.time_column,
            end_time_col=table.end_time_column,
        )
        for foreign_key in table.foreign_keys:
            relationships.append((
                ColumnKey(table.name, foreign_key.name),
                ColumnKey(foreign_key.destination_table,
                          foreign_key.destination_column),
            ))
    return GraphDefinition(
        tables=tables,
        col_groups=group_column_relationships(relationships),
    )


@dataclass(frozen=True)
class PlanGraphColumn:
    """One column in the shape `ModelPlan.validate` reads off a graph."""

    name: str
    stype: Stype


@dataclass(frozen=True)
class PlanGraphKey:
    """One primary or foreign key in the plan-validation graph shape."""

    name: str
    dst_table: str | None = None
    dst_key: str | None = None


@dataclass(frozen=True)
class PlanGraphTable:
    table_name: str
    cols: tuple[PlanGraphColumn, ...]
    pkey: PlanGraphKey | None
    fkeys: tuple[PlanGraphKey, ...]

    def get_col(self, col_name: str) -> PlanGraphColumn | None:
        for col in self.cols:
            if col.name == col_name:
                return col
        return None


@dataclass(frozen=True)
class PlanGraph:
    """The `GraphConfig`-shaped view `ModelPlan.validate` duck-types.

    `ModelPlan.validate` and `NeighborSamplingPlan.from_default` read a
    platform `GraphConfig`: `graph[name]`, `name in graph`, `graph.tables`,
    and per table `table_name` / `cols` / `pkey` / `fkeys`. That type lives
    in the monolith, so portable callers have so far passed no graph at all
    -- which silently skips the per-target-edge hop-1 `inferred` entry and
    the column stypes behind metric validation. This projection supplies
    exactly the attributes those two read, and nothing else.

    `cols` deliberately excludes the primary and foreign keys, matching
    `TableConfig.cols` (whose `all_cols` adds them back). The time column
    stays, as it does on the platform.
    """

    tables: tuple[PlanGraphTable, ...]

    def __getitem__(self, table_name: str) -> PlanGraphTable:
        for table in self.tables:
            if table.table_name == table_name:
                return table
        raise KeyError(f"Table {table_name} not in PlanGraph")

    def __contains__(self, table_name: object) -> bool:
        return any(table.table_name == table_name for table in self.tables)


def to_plan_graph(graph: NativeGraph) -> PlanGraph:
    """Project the portable graph onto the shape model plans validate against."""

    tables = []
    for table in graph.tables:
        key_names = {foreign_key.name for foreign_key in table.foreign_keys}
        if table.primary_key is not None:
            key_names.add(table.primary_key)
        tables.append(
            PlanGraphTable(
                table_name=table.name,
                cols=tuple(
                    PlanGraphColumn(column.name, Stype(column.stype))
                    for column in table.columns
                    if column.name not in key_names),
                pkey=(None if table.primary_key is None else PlanGraphKey(
                    table.primary_key)),
                fkeys=tuple(
                    PlanGraphKey(
                        foreign_key.name,
                        foreign_key.destination_table,
                        foreign_key.destination_column,
                    ) for foreign_key in table.foreign_keys),
            ))
    return PlanGraph(tuple(tables))


def resolve_target_to_entity_paths(
    *,
    target_table: str | None,
    target_key: str | None,
    entity_table: str,
    target_primary_key: str | None,
    entity_key: str | None = None,
    rhs_path: tuple[str, str, str] | None = None,
) -> list[list[tuple[str, str, str]]]:
    """Resolve ordered target-to-entity paths from adapter-provided names."""

    if target_table is None or target_key is None:
        return []
    if target_primary_key != target_key:
        paths = [[(target_table, target_key, entity_table)]]
    else:
        assert entity_key is not None
        paths = [[(entity_table, entity_key, target_table)]]

    if rhs_path is not None:
        paths.append([rhs_path])
    return paths


def target_to_entity_paths(
    query: ValidatedNativeQuery,
    graph: NativeGraph,
) -> list[list[tuple[str, str, str]]]:
    target_fkey = query.query.get_single_target_fkey()
    if target_fkey is None:
        return []
    target_table, target_key = split_column_fqn(str(target_fkey))
    target_table = _bare_table_name(target_table)
    entity_table = _bare_table_name(query.entity_table)
    target_primary_key = graph.table(target_table).primary_key
    entity_fkey = query.query.get_single_target_fkey(lhs=True)
    _, entity_key = split_column_fqn(str(entity_fkey))

    rhs_path = None
    if query.task_family == "link_prediction":
        aggregation = query.query.get_final_target_aggregation()
        rhs_fkey = graph.foreign_key(str(aggregation._get_target_column_name()))
        if rhs_fkey is not None:
            rhs_path = (target_table, rhs_fkey.name, rhs_fkey.destination_table)
    return resolve_target_to_entity_paths(
        target_table=target_table,
        target_key=target_key,
        entity_table=entity_table,
        target_primary_key=target_primary_key,
        entity_key=entity_key,
        rhs_path=rhs_path,
    )


def resolve_task_type(
    *,
    target_is_column: bool,
    target_dtype: Any,
    target_stype: Any,
    problem_type: Any,
    query_type: Any,
    forecast_length: int = 1,
    aggregation_type: Any = None,
    aggregation_target_dtype: Any = None,
    aggregation_is_wildcard: bool = False,
    aggregation_target_is_foreign_key: bool = False,
    rhs_node_type: str | None = None,
) -> tuple[TaskType, str | None]:
    """Resolve a task from validated query facts supplied by an adapter."""

    dtype = _enum_value(target_dtype).lower()
    stype = _enum_value(target_stype)
    problem = _enum_value(problem_type).upper()
    query_kind = _enum_value(query_type).lower()
    aggregation = (_enum_value(aggregation_type).upper()
                   if aggregation_type is not None else None)
    aggregation_dtype = (_enum_value(aggregation_target_dtype).lower()
                         if aggregation_target_dtype is not None else None)
    is_forecasting = problem == "FORECAST" or forecast_length > 1

    if dtype == "bool" and stype in {"categorical", "ID"}:
        return TaskType.BINARY_CLASSIFICATION, None
    if target_is_column:
        if dtype == "bool":
            return TaskType.BINARY_CLASSIFICATION, None
        if stype == "numerical":
            return TaskType.REGRESSION, None
        if stype == "multicategorical":
            task_type = (TaskType.MULTILABEL_RANKING if problem == "RANK" else
                         TaskType.MULTILABEL_CLASSIFICATION)
            return task_type, None
        return TaskType.MULTICLASS_CLASSIFICATION, None

    assert aggregation is not None
    if aggregation_target_is_foreign_key and aggregation == "LIST_DISTINCT":
        if problem == "CLASSIFY":
            return TaskType.MULTILABEL_CLASSIFICATION, None
        if query_kind == "temporal":
            return TaskType.TEMPORAL_LINK_PREDICTION, rhs_node_type
        return TaskType.STATIC_LINK_PREDICTION, rhs_node_type
    if stype in {"categorical", "ID"}:
        return TaskType.MULTICLASS_CLASSIFICATION, None
    if stype == "multicategorical":
        task_type = (TaskType.MULTILABEL_RANKING if problem == "RANK" else
                     TaskType.MULTILABEL_CLASSIFICATION)
        return task_type, None

    assert stype == "numerical"
    if aggregation_is_wildcard:
        return (TaskType.FORECASTING
                if is_forecasting else TaskType.REGRESSION), None
    if aggregation_dtype == "bool":
        if aggregation in {"COUNT", "COUNT_DISTINCT", "SUM", "AVG"}:
            return (TaskType.FORECASTING
                    if is_forecasting else TaskType.REGRESSION), None
        return TaskType.BINARY_CLASSIFICATION, None
    if is_forecasting:
        return TaskType.FORECASTING, None
    return TaskType.REGRESSION, None


def _task_type(query: Any, graph: NativeGraph,
               forecast_length: int) -> tuple[str, str | None]:
    aggregation = query.get_final_target_aggregation()
    target_column = query.get_final_target_column()
    target_fqn = (str(aggregation._get_target_column_name())
                  if aggregation is not None else None)
    target_name = (target_fqn.rsplit(".", 1)[-1]
                   if target_fqn is not None else None)
    foreign_key = (graph.foreign_key(target_fqn)
                   if target_fqn is not None and target_name != "*" else None)
    aggregation_target_dtype = None
    if target_fqn is not None and target_name != "*":
        target_table, target_name = split_column_fqn(target_fqn)
        aggregation_target_dtype = graph.table(target_table).column(
            target_name).dtype
    task_type, rhs_node_type = resolve_task_type(
        target_is_column=target_column is not None,
        target_dtype=query.target_dtype,
        target_stype=query.target_stype,
        problem_type=query.problem_type,
        query_type=query.query_type,
        forecast_length=forecast_length,
        aggregation_type=(aggregation.aggr
                          if aggregation is not None else None),
        aggregation_target_dtype=aggregation_target_dtype,
        aggregation_is_wildcard=target_name == "*",
        aggregation_target_is_foreign_key=foreign_key is not None,
        rhs_node_type=(foreign_key.destination_table
                       if foreign_key is not None else None),
    )
    return task_type.value, rhs_node_type


def parse_and_validate_query(
    query_string: str,
    graph: Mapping[str, Any] | NativeGraph,
) -> ValidatedNativeQuery:
    """Parse and validate PQL using the same first-party implementation."""

    if not query_string.strip():
        raise PredictiveQueryPlanError("predictive query must not be empty")
    native_graph = graph if isinstance(graph,
                                       NativeGraph) else normalize_graph(graph)
    try:
        parsed = PQLParser().to_parsed_predictive_query(query_string)
    except ValueError as error:
        raise PredictiveQueryPlanError(str(error)) from None
    problem_type = _enum_value(parsed.problem_type).upper()
    validator = PredictiveQueryValidator(
        graph=to_api_graph(native_graph),
        allow_array_targets=problem_type == "RANK",
        allow_timestamp_arrays=False,
    )
    query, response = validator.validate_predictive_query(parsed)
    if query is None or not response.ok:
        raise PredictiveQueryPlanError(response.message())

    forecast_length = int(query.num_forecasts or 1)
    task_type, rhs_node_type = _task_type(query, native_graph, forecast_length)
    entity_table, entity_column = split_column_fqn(query.entity_column)
    rhs_table = (native_graph.table(rhs_node_type)
                 if rhs_node_type is not None else None)
    return ValidatedNativeQuery(
        query=query,
        task_type=task_type,
        task_family=(
            "link_prediction" if task_type.endswith("link_prediction") else
            "forecasting" if task_type == "forecasting" else "node_prediction"),
        entity_table=entity_table,
        entity_column=entity_column,
        target_stype=_enum_value(query.target_stype),
        query_type=_enum_value(query.query_type).lower(),
        rhs_node_type=rhs_node_type,
        rhs_is_temporal=bool(
            rhs_table and (rhs_table.time_column or rhs_table.end_time_column)),
        list_distinct_target_is_array=rhs_node_type is not None,
        forecast_length=forecast_length,
        validation_response=response,
    )


def resolve_query_forecast(
    query: ValidatedNativeQuery,
    graph: NativeGraph,
    generation_plan: TrainingTableGenerationPlan,
) -> ValidatedNativeQuery:
    """Apply the generation plan's horizon before resolving task metadata.

    Hosted PredictiveQueryV2 gives a supplied forecast length precedence over
    the PQL horizon. Both generation and reuse of its rows need that same task.
    """
    horizon = inferred_or_missing_type_to_none(generation_plan.forecast_length)
    if horizon is None:
        return query
    task_type, _ = _task_type(query.query, graph, int(horizon))
    return replace(
        query,
        forecast_length=int(horizon),
        task_type=task_type,
        task_family=(
            "link_prediction" if task_type.endswith("link_prediction") else
            "forecasting" if task_type == "forecasting" else "node_prediction"),
    )


def _forecast_options(
    query: ValidatedNativeQuery,
    options: TrainingTableGenerationPlan,
) -> tuple[int | None, Mapping[str, Any] | None]:
    if query.task_type != "forecasting":
        return None, None
    lag_timesteps = inferred_or_missing_type_to_none(options.lag_timesteps)
    if lag_timesteps is None:
        lag_timesteps = 1
    if (isinstance(lag_timesteps, bool)
            or not isinstance(lag_timesteps, Integral) or lag_timesteps < 1):
        raise PredictiveQueryPlanError(
            "training_table_generation_plan.lag_timesteps must be a "
            "positive integer for forecasting")
    timeframe = query.query.get_combined_date_offset_range().timeframe
    step_seconds = int(timeframe.total_seconds())
    lag_offsets = tuple(-step_seconds * index
                        for index in range(int(lag_timesteps), 0, -1))
    future_offsets = tuple(step_seconds * index
                           for index in range(query.forecast_length))
    compiler_options = {
        "future_offsets_secs": future_offsets,
        "lag_offsets_secs": lag_offsets,
    }
    if inferred_or_missing_type_to_false(options.year_over_year):
        year_offset = ONE_YEAR_AGO_OFFSET_DAYS * 86_400
        compiler_options["yoy_offsets_secs"] = tuple(
            -year_offset + step_seconds * index
            for index in range(-PADDING_SIZE, query.forecast_length +
                               PADDING_SIZE))
    return int(lag_timesteps), compiler_options


def _default_split(
    query: ValidatedNativeQuery,
    graph: NativeGraph,
    options: TrainingTableGenerationPlan | None = None,
) -> tuple[dict[str, Any], str]:
    offset = (None if query.query_type == "static" else
              query.query.get_combined_date_offset_range())
    timeframe_step = (None if options is None else
                      inferred_or_missing_type_to_none(options.timeframe_step))
    descriptor = default_split_for_query(
        TaskType(query.task_type),
        query_is_static=query.query_type == "static",
        entity_has_time_column=(graph.table(query.entity_table).time_column
                                is not None),
        list_distinct_target_is_array=query.list_distinct_target_is_array,
        offset_start=(None if offset is None else int(offset.start or 0)),
        offset_end=(None if offset is None else int(offset.end)),
        unit=("days" if offset is None else _enum_value(offset.unit).lower()),
        forecast_length=query.forecast_length,
        timeframe_step=(None
                        if timeframe_step is None else int(timeframe_step)),
    )
    return ({
        key: value
        for key, value in descriptor.to_split_config().items()
        if value is not None
    }, descriptor.__class__.__name__)


def query_train_table_time(query: ValidatedNativeQuery,
                           graph: NativeGraph) -> TrainTableTime:
    return train_table_time(
        TaskType(query.task_type),
        QueryType(query.query_type),
        list_distinct_target_is_array=query.list_distinct_target_is_array,
        entity_table_has_time_col=(graph.table(query.entity_table).time_column
                                   is not None),
    )


def query_is_temporal(query: ValidatedNativeQuery, graph: NativeGraph) -> bool:
    return query_train_table_time(query, graph) == TrainTableTime.REAL


def static_link_fact(query: Any, rhs_node_type: str | None) -> dict[str, str]:
    """The fact table whose rows are a static link query's links."""
    fact, entity_col = split_column_fqn(str(query.get_single_target_fkey()))
    target = query.get_final_target_aggregation()._get_target_column_name()
    target_fact, target_col = split_column_fqn(str(target))
    if fact != target_fact:
        raise ValueError(
            f"static link prediction needs its targets on {fact!r}, the table "
            f"linking to the entity; got {target!r}")
    if fact == rhs_node_type:
        raise ValueError(
            "static link prediction requires an item table distinct from the "
            f"fact table {fact!r}; item seeds cannot hide their own fact rows")
    return {"table": fact, "entity_col": entity_col, "target_col": target_col}


def resolve_query_split(query: ValidatedNativeQuery,
                        value: str) -> SplitArguments:
    return normalize_static_link_split(
        split_from_repr(value),
        default_is_random_link_split=(query.query_type == "static"
                                      and query.task_type
                                      == "static_link_prediction"
                                      and query.list_distinct_target_is_array),
    )


def _split_options(
    query: ValidatedNativeQuery,
    graph: NativeGraph,
    options: TrainingTableGenerationPlan,
) -> tuple[Mapping[str, Any], str]:
    value = inferred_or_missing_type_to_none(options.split)
    if value is None:
        return _default_split(
            query,
            graph,
            options,
        )
    try:
        descriptor = resolve_query_split(query, str(value))
        split, family = split_config_from_descriptor(descriptor)
    except ValueError as error:
        raise PredictiveQueryPlanError(
            f"training-table generation plan {error}") from error
    if query.query_type != "static" and family == "TimeRangeSplit":
        offset = query.query.get_combined_date_offset_range()
        split = {
            **split,
            "unit": _enum_value(offset.unit).lower(),
            "timeframe": int(offset.end) - int(offset.start or 0),
            "anchor_start_offset": int(offset.start or 0),
            "flexible_boundaries": False,
        }
        timeframe_step = inferred_or_missing_type_to_none(
            options.timeframe_step)
        if timeframe_step is not None:
            split["timeframe_step"] = int(timeframe_step)
    return split, family


def _timestamp_ns(value: Any) -> int:
    if isinstance(value, bool):
        raise PredictiveQueryPlanError("boolean timestamp statistic is invalid")
    if isinstance(value, Integral):
        integer = int(value)
        magnitude = abs(integer)
        if magnitude < 10_000_000_000:
            return integer * 1_000_000_000
        if magnitude < 10_000_000_000_000:
            return integer * 1_000_000
        if magnitude < 10_000_000_000_000_000:
            return integer * 1_000
        return integer
    rendered = str(value).removeprefix("kumo_timestamp_").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as error:
        raise PredictiveQueryPlanError(
            f"invalid timestamp statistic {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _stat_value(stats: Mapping[str, Any], column: str, kind: str) -> Any:
    suffixes = (f"__{kind}__False", f"__{kind}__True")
    for key, value in stats.items():
        if (key == kind
                or key.startswith(column + "__")) and key.endswith(suffixes):
            if isinstance(value, Mapping) and "value" in value:
                return value["value"]
            return value
    return None


def _target_time_bound(aggregation: Any,
                       index_path: str) -> tuple[Timestamp, Timestamp] | None:
    time_column = aggregation.time_col
    if not time_column:
        return None
    table_name, column_name = split_column_fqn(str(time_column))
    raw = read_col_stats(index_path, table_name, "full")
    if not raw:
        raise PredictiveQueryPlanError(
            f"DiskGraph index has no column stats for {table_name!r}")
    stats = json.loads(raw)
    minimum = _stat_value(stats, column_name, "MIN")
    maximum = _stat_value(stats, column_name, "MAX")
    if minimum is None or maximum is None:
        raise PredictiveQueryPlanError(
            f"DiskGraph stats lack MIN/MAX for {time_column!r}")
    return (Timestamp(_timestamp_ns(minimum)),
            Timestamp(_timestamp_ns(maximum)))


def _target_time_bounds(
    query: ValidatedNativeQuery,
    index_path: str,
) -> list[tuple[Timestamp, Timestamp]]:
    bounds = [
        bound for aggregation in query.query.get_all_target_aggregations()
        if (bound := _target_time_bound(aggregation, index_path)) is not None
    ]
    if not bounds:
        raise PredictiveQueryPlanError(
            "temporal query has no target time-column statistics")
    return bounds


def _temporal_target_time_range(
    query: ValidatedNativeQuery,
    index_path: str,
) -> tuple[Timestamp, Timestamp, Timedelta | None]:
    bounds = _target_time_bounds(query, index_path)

    offset = query.query.get_combined_date_offset_range()
    entity_range = query.query.entity_ast.non_inf_date_offset_range()
    target_range = query.query.target_ast.non_inf_date_offset_range()
    return default_target_time_range(
        bounds,
        target_range=offset,
        entity_filter_range=entity_range,
        target_filter_range=target_range,
    )


def latest_temporal_timestamp(query: ValidatedNativeQuery,
                              index_path: str) -> int:
    """Return the latest source maximum across target aggregations."""
    if query.query_type != "temporal":
        raise PredictiveQueryPlanError(
            "a source-derived timestamp requires a temporal query")
    return max(int(maximum.value)
               for _, maximum in _target_time_bounds(query, index_path))


def _temporal_timestamps(
    query: ValidatedNativeQuery,
    index_path: str,
    options: TrainingTableGenerationPlan,
    split: Mapping[str, Any],
    split_family: str,
    explicit: tuple[int, ...],
) -> tuple[tuple[int, ...], bool]:
    offset = query.query.get_combined_date_offset_range()

    def date_offset(value: int) -> DateOffset:
        # Preserve calendar units just as the hosted generator does.
        return DateOffsetRange._to_dateOffset(offset.unit, value)

    train_start = inferred_or_missing_type_to_none(options.train_start_offset)
    train_end = inferred_or_missing_type_to_none(options.train_end_offset)
    error = validate_train_offsets(
        None if train_start is None else date_offset(int(train_start)),
        None if train_end is None else date_offset(int(train_end)),
    )
    if error is not None:
        raise PredictiveQueryPlanError(f"{error.title}. {error.message}")
    if explicit:
        return tuple(_timestamp_ns(value) for value in explicit), False
    minimum, maximum, filter_buffer = _temporal_target_time_range(
        query, index_path)

    def relative(value: Any) -> DateOffset | Timestamp | None:
        if value is None or value == "-INF":
            return None
        if isinstance(value, Integral):
            return date_offset(int(value))
        return Timestamp(Timestamp(value).value)

    step_count = int(
        inferred_or_missing_type_to_none(options.timeframe_step)
        or (int(offset.end) - int(offset.start or 0)))
    if step_count <= 0:
        raise PredictiveQueryPlanError(
            "predictive-query timeframe step must be positive")
    normalized_split = dict(split)
    if split_family == "TimeRangeSplit":
        normalized_split["ranges"] = [
            [Timestamp(_timestamp_ns(start)), Timestamp(_timestamp_ns(end))]
            for start, end in split["ranges"]
        ]
    timestamps, fixed_boundaries, _ = generate_legacy_timestamps(
        original_start=minimum,
        original_end=maximum,
        target_timeframe=DateOffsetRange.date_offset_subtract(
            offset.end_date_offset, date_offset(int(offset.start or 0))),
        target_start_offset=date_offset(int(offset.start or 0)),
        target_end_offset=offset.end_date_offset,
        step=date_offset(step_count),
        split_family=split_family,
        split=normalized_split,
        filter_buffer=filter_buffer,
        train_start_offset=(None if train_start is None else date_offset(
            int(train_start))),
        train_end_offset=(None if train_end is None else date_offset(
            int(train_end))),
        start_time=relative(
            inferred_or_missing_type_to_none(options.start_time)),
        end_time=relative(inferred_or_missing_type_to_none(options.end_time)),
        drop_earliest=True,
    )
    return tuple(timestamp.value for timestamp in timestamps), fixed_boundaries


def _column_encoder_stats(
    stats: Mapping[str, Any],
    column_name: str,
) -> dict[ColStatType, _EncoderStat]:
    nested = stats.get(column_name)
    if isinstance(nested, Mapping):
        entries = nested.items()
    else:
        entries = []
    by_value = {stat_type.value: stat_type for stat_type in ColStatType}
    result: dict[ColStatType, _EncoderStat] = {}

    def add(stat_type: ColStatType, value: Any, approx: bool) -> None:
        if isinstance(value, Mapping) and "value" in value:
            stat = _EncoderStat(
                value=value["value"],
                approx=bool(value.get("approx", approx)),
                num_entries=value.get("num_entries"),
            )
        else:
            stat = _EncoderStat(value=value, approx=approx)
        current = result.get(stat_type)
        if current is None or current.approx or not stat.approx:
            result[stat_type] = stat

    for key, value in entries:
        name = str(getattr(key, "value", key))
        stat_type = by_value.get(name)
        if stat_type is not None:
            add(stat_type, value, False)

    if isinstance(nested, Mapping):
        return result

    for key, value in stats.items():
        exact_prefix = column_name + "__"
        legacy_prefix = column_name + "_"
        if key.startswith(exact_prefix):
            segments = key[len(exact_prefix):].split("__")
        elif key.startswith(legacy_prefix):
            # A single-underscore legacy key has the statistic immediately
            # after the column. Do not scan later segments: that would let a
            # column such as ``account`` consume ``account_id__...`` stats.
            segments = key[len(legacy_prefix):].split("__", 1)[:1]
        else:
            continue
        # The statistic is one whole key segment, never a substring of the
        # column name. The optional following boolean is the approximation
        # flag in DiskGraph's flat layout.
        for index, segment in enumerate(segments):
            stat_type = by_value.get(segment)
            if stat_type is None:
                continue
            approx = False
            if index + 1 < len(segments):
                approx = segments[index + 1].lower() == "true"
            add(stat_type, value, approx)
            break
    return result


def _column_stats(stats: Mapping[str, Any], column_name: str) -> dict[str, Any]:
    return {
        stat_type.value: stat.value
        for stat_type, stat in _column_encoder_stats(stats,
                                                     column_name).items()
    }


def _table_stats(index_path: str, table_name: str) -> dict[str, Any]:
    raw = read_col_stats(index_path, table_name, "full")
    if not raw:
        raise PredictiveQueryPlanError(
            f"DiskGraph index has no column stats for {table_name!r}")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise PredictiveQueryPlanError(
            f"DiskGraph column stats for {table_name!r} are malformed")
    return dict(value)


def table_row_count(index_path: str, table_name: str) -> int | None:
    """Read one ingested table's row count out of the DiskGraph statistics.

    DiskGraph writes a per-column ``COUNT`` that counts nulls and invalid
    values alongside the valid ones, so every column of a table reports the
    same number: the table's row count, exactly what the platform reads as
    ``TableStatType.NUM_ROWS``. Returns ``None`` when the index carries no
    statistics for the table, leaving the caller to fall back rather than
    infer from a number it does not have.
    """

    raw = read_col_stats(index_path, table_name, "full")
    if not raw:
        return None
    stats = json.loads(raw)
    if not isinstance(stats, Mapping):
        return None
    counts = []
    for key, value in stats.items():
        # `_column_stats` reads two layouts off this artifact: flat
        # `col__stype__STAT__bool` keys, and a nested per-column mapping.
        # Reading only the flat one would report "no statistics" for a
        # nested payload and silently drop the fanout to the hop default.
        if isinstance(value, Mapping) and "COUNT" in value:
            count = value["COUNT"]
        elif key.endswith(("__COUNT__False", "__COUNT__True")):
            count = (value.get("value")
                     if isinstance(value, Mapping) else value)
        else:
            continue
        if isinstance(count, Integral) and not isinstance(count, bool):
            counts.append(int(count))
    return max(counts) if counts else None


def _encoder_input(column: NativeGraphColumn,
                   stats: Mapping[str, Any]) -> dict[str, Any]:
    encoder_stats = _column_encoder_stats(stats, column.name)
    encoder = (dict(column.encoder) if column.encoder is not None else
               _infer_encoder(column, encoder_stats))
    return {
        "column_name": column.name,
        "dtype": column.dtype,
        "stype": column.stype,
        "encoder": encoder,
        "stats": {
            stat_type.value: stat.value
            for stat_type, stat in encoder_stats.items()
        },
    }


def _infer_encoder(column: NativeGraphColumn,
                   stats: Mapping[Any, Any]) -> dict[str, Any]:
    normalized_stats = {
        ColStatType(getattr(stat_type, "value", stat_type)):
        (stat if isinstance(stat, _EncoderStat) else _EncoderStat(
            value=stat,
            approx=False,
        ))
        for stat_type, stat in stats.items()
    }
    try:
        selected = select_encoder(
            column.name,
            Dtype(column.dtype),
            Stype(column.stype),
            normalized_stats,
        )
    except ValueError as error:
        raise PredictiveQueryPlanError(
            f"encoder inference for {column.name!r} failed: {error}") from error
    return TypeAdapter(EncoderType).dump_python(selected, mode="json")


def _sequence_encoder_input(column_name: str, length: int) -> dict[str, Any]:
    return {
        "column_name": column_name,
        "dtype": "floatlist",
        "stype": "sequence",
        "encoder": {
            "name": "NumericalList"
        },
        "stats": {
            "SEQUENCE_MIN_LENGTH": length,
            "SEQUENCE_MAX_LENGTH": length,
        },
    }


def _feature_schema(
    graph: NativeGraph,
    query: ValidatedNativeQuery,
    index_path: str,
    *,
    lag_timesteps: int | None,
    year_over_year: bool,
) -> dict[str, Any]:
    excluded_targets = query.query.get_exclude_cols_dict()
    result = {}
    for table in graph.tables:
        feature_columns = feature_column_names(
            (column.name for column in table.columns
             if column.stype.lower() != "unsupported"),
            primary_key=table.primary_key,
            foreign_keys=(key.name for key in table.foreign_keys),
            end_time_column=table.end_time_column,
            excluded=excluded_targets.get(table.name, ()),
        )
        stats = (_table_stats(index_path, table.name)
                 if feature_columns else {})
        result[table.name] = {
            "feature_columns":
            feature_columns,
            "mapper_key":
            table.name,
            "time_column":
            table.time_column,
            "column_specs": [
                _encoder_input(table.column(column_name), stats)
                for column_name in feature_columns
            ],
        }
    # TARGET is produced by the predictive-query aggregation. Its encoder and
    # statistics must therefore be resolved from the materialized train-table
    # sidecar, never from the source graph column referenced by the query.
    if query.task_family == "forecasting":
        sequence_inputs = [
            _sequence_encoder_input(
                FUTURE_LABELS_COL,
                query.forecast_length,
            ),
        ]
        if lag_timesteps:
            sequence_inputs.append(
                _sequence_encoder_input(
                    AUTOREGRESSION_COL,
                    lag_timesteps,
                ))
        if year_over_year:
            sequence_inputs.append(
                _sequence_encoder_input(
                    YOY_COL,
                    query.forecast_length + 2 * PADDING_SIZE,
                ))
        result["train_table"] = {
            "feature_columns":
            [value["column_name"] for value in sequence_inputs],
            "mapper_key": "train_table",
            "time_column": None,
            "column_specs": sequence_inputs,
        }
    return {"node_types": result, "target_encoder_input": None}


def _graph_task_dataset_fields(
    *,
    graph: NativeGraph,
    query: ValidatedNativeQuery,
    graph_index_path: str,
    lag_timesteps: int | None,
    year_over_year: bool,
) -> dict[str, Any]:
    """Resolve the task fields owned by one selected graph and query."""

    return {
        "task_family": query.task_family,
        "task_type": query.task_type,
        "is_temporal": query_is_temporal(query, graph),
        "seed_node_type": query.entity_table,
        "entity_table_name": query.entity_table,
        "rhs_node_type": query.rhs_node_type,
        "feature_schema": _feature_schema(
            graph,
            query,
            graph_index_path,
            lag_timesteps=lag_timesteps,
            year_over_year=year_over_year,
        ),
        "graph_node_types": tuple(table.name for table in graph.tables),
        "graph_edge_types": tuple(
            (table.name, foreign_key.name, foreign_key.destination_table)
            for table in graph.tables for foreign_key in table.foreign_keys),
        "target_to_entity_paths": tuple(
            tuple(path) for path in target_to_entity_paths(query, graph)),
    }


def build_generated_training_task_dataset(
    *,
    query: ValidatedNativeQuery,
    graph: NativeGraph,
    graph_index_path: str,
    run_id: str,
    artifact_dir: str,
    tt_id: str,
    required_stages: tuple[str, ...],
    lag_timesteps: int | None,
    year_over_year: bool,
) -> TaskDatasetMaterializeSpec:
    """Build selected-graph task metadata for existing generated rows."""

    if not isinstance(query, ValidatedNativeQuery):
        raise TypeError("query must be a ValidatedNativeQuery")
    if not isinstance(graph, NativeGraph):
        raise TypeError("graph must be a NativeGraph")
    if not graph_index_path:
        raise PredictiveQueryPlanError("ingest output is missing index_path")
    if not tt_id:
        raise PredictiveQueryPlanError(
            "generated training table is missing its id")

    extension_path = os.path.join(graph_index_path, "train_tables", tt_id)
    return TaskDatasetMaterializeSpec(
        tt_manifest_path=os.path.join(extension_path, "manifest.json"),
        job_id=run_id,
        artifact_root=os.path.join(artifact_dir, "task_dataset"),
        train_table_name="train_table",
        required_stages=required_stages,
        **_graph_task_dataset_fields(
            graph=graph,
            query=query,
            graph_index_path=graph_index_path,
            lag_timesteps=lag_timesteps,
            year_over_year=year_over_year,
        ),
    )


def build_native_pquery_task_plan(
        *,
        query: ValidatedNativeQuery,
        graph: NativeGraph,
        training_table_generation_plan: TrainingTableGenerationPlan,
        run_id: str,
        graph_index_path: str,
        scratch_dir: str,
        artifact_dir: str,
        graph_index_id: str | None = None,
        timestamps: tuple[int, ...] = (),
        supplied_rows: bool = False,
) -> NativePqueryTaskPlan:
    """Resolve task inputs for generated or already-supplied rows.

    ``supplied_rows`` skips graph-derived anchor generation because those
    rows carry their own ``TIMESTAMP`` values.
    """

    query = resolve_query_forecast(query, graph, training_table_generation_plan)
    if query.task_type not in _NATIVE_TRAINING_TASK_TYPES:
        raise PredictiveQueryPlanError(
            "Portable training does not support task type "
            f"{query.task_type!r}")
    if not graph_index_path:
        raise PredictiveQueryPlanError("ingest output is missing index_path")
    options = training_table_generation_plan
    split, split_family = _split_options(query, graph, options)
    lag_timesteps, compiler_options = _forecast_options(query, options)
    resolved_timestamps: tuple[int, ...] = ()
    if not supplied_rows:
        if query.query_type == "static":
            if timestamps:
                raise PredictiveQueryPlanError(
                    "explicit timestamps are only meaningful for a temporal "
                    f"query; {query.task_type!r} resolved to a "
                    f"{query.query_type} query")
        else:
            resolved_timestamps, fixed_boundaries = _temporal_timestamps(
                query,
                graph_index_path,
                options,
                split,
                split_family,
                timestamps,
            )
            if fixed_boundaries:
                split = {**split, "flexible_boundaries": False}
    entity_table = graph.table(query.entity_table)
    tt_id = f"tt-{run_id}"
    output_path = os.path.join(scratch_dir, "train_table", "tt.parquet")
    train_table = TrainTableRequest(
        query=query.query,
        timestamps=resolved_timestamps,
        task_type=query.task_type,
        forecast_length=query.forecast_length,
        lag_timesteps=lag_timesteps,
        list_distinct_target_is_array=query.list_distinct_target_is_array,
        include_frequency=query.task_family == "link_prediction",
        entity_time_col=entity_table.time_column,
        index_path=graph_index_path,
        output_path=output_path,
        tt_id=tt_id,
        split_config=split,
        split_family=split_family,
        input_timestamp_unit="nanoseconds",
        target_stype=query.target_stype,
        entity_node_type=query.entity_table,
        entity_primary_key=entity_table.primary_key or "",
        rhs_node_type=query.rhs_node_type,
        rhs_is_temporal=query.rhs_is_temporal,
        compiler_options=compiler_options,
        index_path_verified=True,
    )
    extension_path = os.path.join(graph_index_path, "train_tables", tt_id)
    task_dataset = TaskDatasetMaterializeSpec(
        tt_manifest_path=os.path.join(extension_path, "manifest.json"),
        job_id=run_id,
        artifact_root=os.path.join(artifact_dir, "task_dataset"),
        train_table_name="train_table",
        required_stages=(("train", "val",
                          "test") if split_family == "TimeRangeSplit"
                         and len(split["ranges"]) == 3 else ("train", "val")),
        **_graph_task_dataset_fields(
            graph=graph,
            query=query,
            graph_index_path=graph_index_path,
            lag_timesteps=lag_timesteps,
            year_over_year="yoy_offsets_secs" in (compiler_options or {}),
        ),
    )
    return NativePqueryTaskPlan(
        train_table=train_table,
        task_dataset=task_dataset,
        graph_index_id=(graph_index_id
                        or "graph/" + stable_fingerprint(graph)[:16]),
        graph_index_path=graph_index_path,
    )


__all__ = [
    "NativeGraph",
    "NativeGraphColumn",
    "NativeGraphForeignKey",
    "NativeGraphTable",
    "NativePqueryTaskPlan",
    "PredictiveQueryPlanError",
    "ValidatedNativeQuery",
    "build_generated_training_task_dataset",
    "build_native_pquery_task_plan",
    "latest_temporal_timestamp",
    "normalize_graph",
    "parse_and_validate_query",
    "resolve_query_forecast",
    "stable_fingerprint",
    "to_api_graph",
]
