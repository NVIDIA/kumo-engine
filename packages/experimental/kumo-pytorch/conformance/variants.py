# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refusal-timing strategies that need more than a contract change.

`generate.STRATEGIES` alters only the contract. A strategy here may also add a
table, change how the rows are bound (Parquet types, renames, paths)
and say which call must then succeed: an accepted request runs every path that
could refuse it later.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from conformance.generate import CLOCK, STRATEGIES, GeneratedCase, Tables
from conformance.outputs import packing_contract
from kumo.pytorch.api import (
    ActiveInterval,
    Aggregate,
    Batch,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DatedEntityValue,
    DiskGraphEngine,
    DType,
    Encode,
    EventSpec,
    Field,
    Graph,
    GraphLayer,
    GraphSample,
    InMemoryEngine,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    Packed,
    Parquet,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    TableSpec,
    Task,
    TimeWindow,
)

Engine = InMemoryEngine | DiskGraphEngine
Binding = Callable[[Dataset, Engine], tuple[Dataset, Engine]]
#: What a strategy runs after the preview: the batches it must deliver, or refuse.
Run = Callable[[Task, tuple[SampleRequest, ...]], Iterable[Batch]]


def _plain(dataset: Dataset, engine: Engine) -> tuple[Dataset, Engine]:
    return dataset, engine


def _preview_only(task: Task, requests: tuple[SampleRequest, ...]) -> Iterable[Batch]:
    """The property always previews every request; nothing else to run."""

    return ()


@dataclass(frozen=True)
class Variant:
    contract: DataLoaderSpec
    tables: Tables
    bind: Binding = _plain
    run: Run = _preview_only


# -- A side table the graph, dated, current and interval strategies read -----

DEVICES = "devices"
DEVICE_COLUMNS = {
    "device_id": DType("int64"),
    "user_id": DType("int64"),
    "label": DType("string"),
    **{
        name: DType("timestamp", unit="s", timezone="UTC")
        for name in ("seen_at", "synced_at", "retired_at")
    },
}


def _with_devices(case: GeneratedCase) -> tuple[DataSchema, Tables]:
    schema = case.contract.schema
    devices = TableSpec(DEVICES, "device_id", DEVICE_COLUMNS, time_column="seen_at")
    relation = RelationSpec("user_devices", "users", "user_id", DEVICES, "user_id")
    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    tables[DEVICES] = [
        {
            "device_id": index,
            "user_id": int(request.entity_id),
            "label": "abc"[index % 3],
            "seen_at": int(request.as_of_time) - 1 - index % 2,
            "synced_at": int(request.as_of_time) - 2,
            "retired_at": int(request.as_of_time) + index % 3 - 1,
        }
        for index, request in enumerate(case.requests, 1)
    ]
    extended = replace(
        schema, tables=(*schema.tables, devices), relations=(*schema.relations, relation)
    )
    return extended, tables


def _device_variant(case: GeneratedCase, **sample: object) -> Variant:
    schema, tables = _with_devices(case)
    contract = case.contract
    batch = contract.batch
    if "neighborhoods" in sample:
        graph = Input(NeighborhoodSource("devices"), Graph(), None)
        batch = replace(batch, inputs={**batch.inputs, "devices": graph})
    return Variant(
        replace(contract, schema=schema, sample=replace(contract.sample, **sample), batch=batch),
        tables,
    )


def _graph(*samples: GraphSample) -> dict[str, NeighborhoodSpec]:
    return {"devices": NeighborhoodSpec((GraphLayer(samples),))}


DEVICE_WINDOW = TimeWindow("seen_at", None, 0)
LABEL = Field("label")


def _active_interval(case: GeneratedCase) -> Variant:
    rows = Rows(("user_devices",), active=ActiveInterval("seen_at", "retired_at"))
    variant = _device_variant(case)
    sample = variant.contract.sample
    values = {**sample.entity_values, "active_devices": Aggregate(rows, "count")}
    contract = replace(variant.contract, sample=replace(sample, entity_values=values))
    return replace(variant, contract=contract)


# -- Bindings for the late refusals ----------------------------------------


def _rewrite(table: str, change: Callable[[pa.Table], pa.Table]) -> Binding:
    def bind(dataset: Dataset, engine: Engine) -> tuple[Dataset, Engine]:
        path = Path(dataset.tables[table].path)
        pq.write_table(change(pq.read_table(path)), path)
        return dataset, engine

    return bind


def _set_column(
    name: str, values: Callable[[pa.ChunkedArray], pa.Array]
) -> Callable[..., pa.Table]:
    def change(data: pa.Table) -> pa.Table:
        index = data.column_names.index(name)
        return data.set_column(index, name, values(data.column(name)))

    return change


def _renamed(dataset: Dataset, engine: Engine) -> tuple[Dataset, Engine]:
    events = Parquet(dataset.tables["events"].path, columns={"amount": "amount"})
    return Dataset({**dataset.tables, "events": events}), engine


def _remote(dataset: Dataset, engine: Engine) -> tuple[Dataset, Engine]:
    return Dataset({**dataset.tables, "users": Parquet("s3://conformance/users")}), engine


def _two_ranks(task: Task, requests: tuple[SampleRequest, ...]) -> Iterable[Batch]:
    return task.batches(requests, rank=0, world_size=2)


def _model(task: Task, requests: tuple[SampleRequest, ...]) -> Iterable[Batch]:
    return task.batches(requests, batch_size=8, output_format="model")


def _raw_string(contract: DataLoaderSpec) -> DataLoaderSpec:
    """The entity input also selects the string ``mode`` value, encoded Raw."""

    entity = contract.batch.inputs["entity"]
    fields = replace(
        entity.representation, fields=(*entity.representation.fields, Encode("mode", Raw()))
    )
    inputs = {**contract.batch.inputs, "entity": replace(entity, representation=fields)}
    return replace(contract, batch=replace(contract.batch, inputs=inputs))


def _bound(case: GeneratedCase, bind: Binding) -> Variant:
    return Variant(case.contract, case.tables, bind)


VARIANTS: Mapping[str, Callable[[GeneratedCase], Variant]] = {
    # Refused by DiskGraph at check(); the reference runs them.
    "active_interval": _active_interval,
    "clock_override": lambda case: _device_variant(
        case,
        neighborhoods=_graph(
            GraphSample("user_devices", 2, strategy="recent", time_column="synced_at")
        ),
    ),
    "current_events": lambda case: _device_variant(
        case,
        current_events={
            "now": EventSpec(
                Rows(("user_devices",), TimeWindow("seen_at", -1, 0)), {"l": LABEL}, limit=1
            )
        },
    ),
    "dated_values": lambda case: _device_variant(
        case,
        dated_entity_values={
            "seen": DatedEntityValue(Rows(("user_devices",), DEVICE_WINDOW), LABEL)
        },
    ),
    "non_direct_field": lambda case: _device_variant(
        case,
        events={
            "seen": EventSpec(
                Rows(("user_devices",), DEVICE_WINDOW),
                {"owner": Field("user_id", path=("user_devices",))},
            )
        },
    ),
    "static_traversal": lambda case: _device_variant(
        case,
        neighborhoods=_graph(GraphSample("user_devices", 2, strategy="recent", temporal="static")),
    ),
    "uniform_graph": lambda case: _device_variant(
        case, neighborhoods=_graph(GraphSample("user_devices", 2, strategy="uniform"))
    ),
    # Accepted at check(), refused later: known gaps, or batches() arguments (INAPPLICABLE).
    "column_rename": lambda case: _bound(case, _renamed),
    "duplicate_layer_relation": lambda case: _device_variant(
        case,
        neighborhoods=_graph(
            GraphSample("user_devices", 1, strategy="recent"),
            GraphSample("user_devices", 2, strategy="recent"),
        ),
    ),
    "int64_clock": lambda case: _bound(
        case, _rewrite("events", _set_column(CLOCK, lambda column: column.cast(pa.int64())))
    ),
    "max_tokens_world_size": lambda case: Variant(
        packing_contract(case, Packed(), max_samples=4, max_tokens=64, keep="last", max_events=4),
        case.tables,
        run=_two_ranks,
    ),
    "missing_column": lambda case: _bound(
        case, _rewrite("events", lambda data: data.drop_columns(["amount"]))
    ),
    "model_format_limit": lambda case: Variant(_raw_string(case.contract), case.tables, run=_model),
    "object_store": lambda case: _bound(case, _remote),
    "unsupported_type": lambda case: _bound(
        case,
        _rewrite(
            "events",
            _set_column(
                "amount", lambda column: pa.array([[value] for value in column.to_pylist()])
            ),
        ),
    ),
}


def variant_for(case: GeneratedCase, strategy: str) -> Variant:
    if strategy in VARIANTS:
        return VARIANTS[strategy](case)
    return Variant(STRATEGIES[strategy](case), case.tables)
