# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile file-backed table declarations into native fitted tensor inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import AVAILABLE_US, EVENT_US, KEY_ORDER, validate_feature_bindings
from tfm.data.relational import key_token
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS
from tfm.extras.kumo_pytorch import native_api


class _AnchorFitSource:
    """Use the native API's sampled fitting path, not whole-column statistics.

    Column-statistics fitting reads every row before a cutoff, including rows
    not reached by the fitting anchors. TxFM fits on selected train histories.
    These are optional source capabilities; withholding them leaves fitting to
    the same native API over its fitting-anchor selection. Tensor delivery and
    native anchor batching still delegate directly to DiskGraph.
    """

    column_stats = None
    column_text = None

    def __init__(self, source: Any) -> None:
        self.source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def build(self, *args: Any, **kwargs: Any) -> Any:
        return self.source.build(*args, **kwargs)

    def build_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.source.build_model(*args, **kwargs)

    def fit_graph_fields(self, *args: Any, **kwargs: Any) -> Any:
        return self.source.fit_graph_fields(*args, **kwargs)

    def sample_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self.source.sample_graph(*args, **kwargs)

    @property
    def artifact_identity(self) -> Any:
        return self.source.artifact_identity

    def close(self) -> None:
        self.source.close()


class _AnchorFittedEngineBase:
    """Delegate native execution while fitting only from selected train anchors."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    @property
    def name(self) -> str:
        return str(self.engine.name)

    def open(self, *args: Any, **kwargs: Any) -> _AnchorFitSource:
        return _AnchorFitSource(self.engine.open(*args, **kwargs))

    def materialize_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine.materialize_anchors(*args, **kwargs)


class AnchorFittedEngine(_AnchorFittedEngineBase):
    """Keep explicit snapshot-binding hooks visible to runtime protocol checks."""

    def _check_binding(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine._check_binding(*args, **kwargs)

    def _resolve_index_schema(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine._resolve_index_schema(*args, **kwargs)


def anchor_fitted_engine(engine: Any) -> _AnchorFittedEngineBase:
    """Expose optional index reuse only when the selected backend supplies it."""
    if all(
        callable(getattr(engine, method, None))
        for method in ("_check_binding", "_resolve_index_schema")
    ):
        return AnchorFittedEngine(engine)
    return _AnchorFittedEngineBase(engine)


def _dtype(dtype: pa.DataType, path: str, api: Any) -> Any:
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return api.DType("string")
    if pa.types.is_boolean(dtype):
        return api.DType("bool")
    if pa.types.is_integer(dtype):
        return api.DType(str(dtype))
    if pa.types.is_floating(dtype):
        return api.DType(f"float{dtype.bit_width}")
    if pa.types.is_timestamp(dtype):
        return api.DType("timestamp", unit=dtype.unit, timezone=dtype.tz or "UTC")
    if pa.types.is_date32(dtype):
        return api.DType("date32")
    raise UnsupportedCombinationError(f"{path}: unsupported native scalar dtype {dtype}")


def native_schema(source: TableSource) -> Any:
    """Validate declared feature exposure, then bind original table files."""
    validate_feature_bindings(source)
    api = native_api()
    tables = []
    for table in source.tables:
        if table.role == "labels":
            continue
        if len(table.key) != 1:
            raise UnsupportedCombinationError(
                f"{table.name}.key: normalize composite keys in the table adapter before native ingestion"
            )
        schema = pq.read_schema(source.files[table.name])
        required = {
            *table.key,
            *table.numeric.values(),
            *table.categorical.values(),
            *table.text.values(),
        }
        graph_fields = (
            {} if source.graph_numeric is None else source.graph_numeric.get(table.name, {})
        )
        required.update(graph_fields.values())
        for relation in source.relationships:
            if relation.child == table.name:
                required.update(relation.columns)
        required.update(c for c in (table.available_at, table.event_time) if c)
        required.update(c for c in (AVAILABLE_US, EVENT_US, KEY_ORDER) if c in schema.names)
        missing = required - set(schema.names)
        if missing:
            raise SchemaError(f"{table.name}: missing declared source columns {sorted(missing)}")
        for mapping in (table.numeric, table.categorical, table.text, graph_fields):
            for alias, column in mapping.items():
                if {alias.strip().casefold(), column.strip().casefold()} & PROHIBITED_INPUT_KEYS:
                    raise SchemaError(
                        f"{table.name}.{column}: label-derived fields cannot be model inputs"
                    )
                if column in {
                    *table.key,
                    table.event_time,
                    table.available_at,
                    *source.event_labels.get(table.name, ()),
                }:
                    raise SchemaError(
                        f"{table.name}.{column}: keys/clocks are metadata, not feature fields"
                    )
        if not table.immutable and table.available_at is None:
            raise SchemaError(f"{table.name}: mutable native tables require an availability clock")
        if pq.ParquetFile(source.files[table.name]).metadata.num_rows == 0:
            continue
        tables.append(
            api.TableSpec(
                table.name,
                table.key[0],
                {
                    # Normalization proved exact whole seconds. Retain UTC
                    # source values; declare second units for native sort keys.
                    name: _dtype(
                        pa.timestamp("s", tz="UTC")
                        if KEY_ORDER in schema.names
                        and name in {table.event_time, table.available_at}
                        else schema.field(name).type,
                        f"{table.name}.{name}",
                        api,
                    )
                    for name in required
                },
                time_column=api.NON_TEMPORAL if table.immutable else table.available_at,
            )
        )
    active = {table.name for table in tables}
    return api.DataSchema(
        tuple(tables),
        tuple(
            api.RelationSpec(
                r.name,
                r.parent,
                next(t.key[0] for t in source.tables if t.name == r.parent),
                r.child,
                r.columns[0],
            )
            for r in source.relationships
            if r.parent in active and r.child in active
        ),
    )


def history_contract(
    source: TableSource,
    *,
    root: str,
    paths: Mapping[str, tuple[str, ...]],
    max_events: int,
    batch_size: int,
    observation_seconds: int | None = None,
    amount_key: str = "amount",
    equivalent_clocks: frozenset[str] = frozenset(),
    categorical_keys: frozenset[str] = frozenset(),
    excluded_keys: frozenset[str] = frozenset(),
    fixed_vocabularies: Mapping[str, tuple[str, ...]] | None = None,
    exclusive_terminal_paths: bool = False,
    event_names: Mapping[str, str] | None = None,
) -> Any:
    """Encode selected historical fields natively; expose no identifier features.

    This contract emits model-ready token grids and aligned raw clock/calendar
    columns. A tensor adapter can build either KVT or flat batches without
    reconstructing events or invoking a Python tokenizer.
    """
    api = native_api()
    schema = native_schema(source)
    tables = {t.name: t for t in source.tables}
    relations = {r.name: r for r in source.relationships}
    if root not in tables or tables[root].role == "labels":
        raise SchemaError(f"{root}: native root must be a declared input table")
    events, encoded = {}, []
    terminals = set()
    ordered_paths = list(paths.items())
    if event_names is not None:
        ordered_paths = sorted(
            paths.items(), key=lambda item: key_token((event_names.get(item[0], item[0]), ""))
        )
    for name, path in ordered_paths:
        current = root
        empty = False
        for role in path:
            relation = relations.get(role)
            if relation is None or current not in {relation.parent, relation.child}:
                raise SchemaError(
                    f"history_paths.{name}.{role}: relationship does not join {current}"
                )
            current = relation.child if current == relation.parent else relation.parent
            empty |= pq.ParquetFile(source.files[current]).metadata.num_rows == 0
            if tables[current].role == "labels":
                raise SchemaError(f"history_paths.{name}: label tables cannot supply inputs")
        if empty:
            continue
        if current in terminals and not exclusive_terminal_paths:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native streams cannot deduplicate multiple paths to {current}"
            )
        terminals.add(current)
        table = tables[current]
        if not table.event_time:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: terminal event clock is required"
            )
        if table.event_time != table.available_at and table.name not in equivalent_clocks:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native history selection cannot apply independent event and availability clocks; supply a validated fixed-ownership projection or select data.loader: python"
            )
        component = name if event_names is None else event_names.get(name, name)
        fields = {}
        for kind, mapping in (
            ("numeric", table.numeric),
            ("categorical", table.categorical),
            ("text", table.text),
        ):
            for alias, column in mapping.items():
                if alias in excluded_keys or (
                    fixed_vocabularies is not None and alias not in fixed_vocabularies
                ):
                    continue
                fields[alias] = api.Field(column=column)
                encoding = (
                    api.Buckets(bins=17, separate_zero=True)
                    if kind == "numeric" and alias not in categorical_keys
                    else api.Text()
                    if kind == "text" and alias not in categorical_keys
                    else api.Vocabulary(max_size=4096)
                )
                if fixed_vocabularies is not None:
                    encoding = api.Vocabulary(
                        values=tuple(
                            (value, i + 2) for i, value in enumerate(fixed_vocabularies[alias])
                        )
                    )
                field = alias if fixed_vocabularies is not None else f"events/{component}/{alias}"
                if not any(item.field == field for item in encoded):
                    encoded.append(api.Encode(field, encoding, id_space="model"))
        fields.update(
            {
                "__clock": api.Field(column=table.event_time),
                "__hour": api.Derived(api.CalendarPart("hour"), ("__clock",)),
                "__dow": api.Derived(api.CalendarPart("weekday"), ("__clock",)),
                "__dom": api.Derived(api.CalendarPart("day"), ("__clock",)),
                "__age": api.EventTimeDelta(unit="seconds"),
                "__amount_raw": (
                    api.Field(column=table.numeric[amount_key])
                    if amount_key in table.numeric
                    else api.Constant(0.0, api.DType("float64"))
                ),
                "__amount_zero": api.Constant(0.0, api.DType("float64")),
                "__amount": api.Derived("add", ("__amount_raw", "__amount_zero")),
            }
        )
        component = name if event_names is None else event_names.get(name, name)
        if component in events:
            raise SchemaError(f"history_paths.{name}: duplicate physical history component")
        events[component] = api.EventSpec(
            api.Rows(
                path,
                api.TimeWindow(
                    table.event_time,
                    start=None if observation_seconds is None else -observation_seconds - 1,
                    stop=0,
                ),
            ),
            fields,
        )
    if not encoded:
        raise SchemaError("Native history inputs need at least one declared feature field")
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec(root), events=events),
        api.BatchSpec(
            id_spaces={
                "model": api.IdSpace(
                    specials={
                        "pad": "[PAD]",
                        "mask": "[MASK]",
                        "unknown": "[UNK]",
                        "entity": "[USR]",
                        "event": "[EVT]",
                        "missing": "[MISSING]",
                    },
                    special_ids={
                        "pad": 0,
                        "mask": 1,
                        "unknown": 2,
                        "entity": 3,
                        "event": 4,
                        "missing": 5,
                    },
                )
            },
            field_names={"fields": api.FieldNames()},
            inputs={
                "history": api.Input(
                    api.EventSource(
                        tuple(events), max_events=max_events, keep="last", order="ascending"
                    ),
                    api.Fields(
                        tuple(encoded),
                        row=api.Tokens(missing="omit"),
                        field_names="fields",
                        field_space="model",
                    ),
                    api.Nested(fields=api.Padded(), events=api.Padded()),
                ),
                "history_aux": api.Input(
                    api.EventsOf("history"),
                    api.Fields(
                        tuple(
                            api.Encode(field, api.Raw())
                            for field in ("__hour", "__dow", "__dom", "__age", "__amount")
                        ),
                        row=api.Columns(),
                    ),
                    api.Padded(),
                ),
            },
            batching=api.Batching(max_samples=batch_size, drop_last=False),
        ),
    )

    if fixed_vocabularies is not None:
        space = api.IdSpace(
            specials={
                "pad": "[PAD]",
                "bos": "[BOS]",
                "eos": "[EOS]",
                "separator": "[SEP]",
                "unknown": "[UNK]",
                "mask": "[MASK]",
                "missing": "[MISSING]",
            },
            special_ids={
                "pad": 0,
                "bos": 1,
                "eos": 2,
                "separator": 3,
                "unknown": 4,
                "mask": 5,
                "missing": 6,
            },
        )
        return replace(contract, batch=replace(contract.batch, id_spaces={"model": space}))
    profile_fields = {}
    profile_encodings = []
    for kind, mapping in (
        ("numeric", tables[root].numeric),
        ("categorical", tables[root].categorical),
        ("text", tables[root].text),
    ):
        for alias, column in mapping.items():
            if alias in excluded_keys:
                continue
            profile_fields[alias] = api.Field(column=column)
            encoding = (
                api.Buckets(bins=17, separate_zero=True)
                if kind == "numeric" and alias not in categorical_keys
                else api.Text()
                if kind == "text" and alias not in categorical_keys
                else api.Vocabulary(max_size=4096)
            )
            profile_encodings.append(api.Encode(alias, encoding, id_space="model"))
    if profile_encodings:
        contract = replace(
            contract,
            sample=replace(contract.sample, entity_values=profile_fields),
            batch=replace(
                contract.batch,
                inputs={
                    **contract.batch.inputs,
                    "profile": api.Input(
                        api.EntitySource(),
                        api.Fields(
                            tuple(profile_encodings),
                            row=api.Tokens(missing="omit"),
                            field_names="fields",
                            field_space="model",
                        ),
                        api.Padded(),
                    ),
                },
            ),
        )
    # Dated profile facts form a separate stream: history truncation and its
    # observation window must not erase older, still-valid milestones.
    profile_events, profile_encoded = {}, []
    for name, path in source.profile_paths.items():
        current = root
        for role_name in path:
            relation = relations.get(role_name)
            if relation is None or current not in {relation.parent, relation.child}:
                raise SchemaError(f"profile_paths.{name}.{role_name}: disconnected relationship")
            current = relation.child if current == relation.parent else relation.parent
        table = tables[current]
        if table.role != "input" or table.event_time is None:
            raise SchemaError(f"profile_paths.{name}: dated input facts are required")
        if table.event_time != table.available_at and table.name not in equivalent_clocks:
            raise UnsupportedCombinationError(
                f"profile_paths.{name}: independent event and availability clocks are unsupported"
            )
        if pq.ParquetFile(source.files[current]).metadata.num_rows == 0:
            continue
        event_name = f"__profile_{name}"
        fields = {"__age": api.EventTimeDelta(unit="seconds")}
        for kind, mapping in (
            ("numeric", table.numeric),
            ("categorical", table.categorical),
            ("text", table.text),
        ):
            for alias, column in mapping.items():
                if alias in excluded_keys:
                    continue
                fields[alias] = api.Field(column=column)
                encoding = (
                    api.Buckets(bins=17, separate_zero=True)
                    if kind == "numeric" and alias not in categorical_keys
                    else api.Text()
                    if kind == "text" and alias not in categorical_keys
                    else api.Vocabulary(max_size=4096)
                )
                profile_encoded.append(
                    api.Encode(f"events/{event_name}/{alias}", encoding, id_space="model")
                )
        profile_events[event_name] = api.EventSpec(
            api.Rows(path, api.TimeWindow(table.event_time, stop=0)), fields
        )
    if profile_encoded:
        contract = replace(
            contract,
            sample=replace(contract.sample, events={**contract.sample.events, **profile_events}),
            batch=replace(
                contract.batch,
                inputs={
                    **contract.batch.inputs,
                    "profile_facts": api.Input(
                        api.EventSource(
                            tuple(profile_events), max_events=64, keep="first", order="ascending"
                        ),
                        api.Fields(
                            tuple(profile_encoded),
                            row=api.Tokens(missing="omit"),
                            field_names="fields",
                            field_space="model",
                        ),
                        api.Nested(fields=api.Padded(), events=api.Padded()),
                    ),
                    "profile_facts_age": api.Input(
                        api.EventsOf("profile_facts"),
                        api.Fields((api.Encode("__age", api.Raw()),), row=api.Columns()),
                        api.Padded(),
                    ),
                },
            ),
        )
    return contract


def with_graph(
    contract: Any,
    source: TableSource,
    *,
    root: str,
    depth: int,
    fanout: int,
    forward_degrees: Mapping[str, int] | None = None,
) -> Any:
    """Add source-table neighborhoods, selecting each layer from reached types.

    The pinned backend selects recent neighbors using one destination clock.
    Distinct availability and event clocks therefore need a different backend
    plan; rejecting them preserves eligibility before fanout selection.
    """
    api = native_api()
    tables = {
        table.name: table
        for table in source.tables
        if table.role == "input"
        and (source.graph_numeric is None or table.name in source.graph_numeric)
        and pq.ParquetFile(source.files[table.name]).metadata.num_rows > 0
    }
    if depth < 1 or fanout < 1 or root not in tables:
        raise SchemaError("Native graph inputs require a declared root and positive depth/fanout")
    frontier, layers = {root}, []
    projected = {root}
    for hop in range(depth):
        samples, reached = [], set()
        for role in source.relationships:
            if role.parent not in tables or role.child not in tables:
                continue
            directions = []
            if role.parent in frontier:
                directions.append(("forward", role.child))
            if role.child in frontier:
                directions.append(("reverse", role.parent))
            for direction, destination in directions:
                table = tables[destination]
                complete = direction == "reverse" or (
                    forward_degrees is not None and forward_degrees.get(role.name, 0) <= fanout
                )
                if not table.immutable and table.event_time != table.available_at and not complete:
                    raise UnsupportedCombinationError(
                        f"graph.hop[{hop + 1}].{role.name}: recent selection requires event_time == available_at; independent availability-before-fanout is unsupported"
                    )
                samples.append(
                    api.GraphSample(
                        role.name,
                        fanout,
                        direction=direction,
                        strategy="uniform" if table.immutable else "recent",
                    )
                )
                reached.add(destination)
        if not samples:
            # A terminal frontier cannot grow further; extra empty model layers
            # are legal, but the native API does not accept an empty GraphLayer.
            break
        layers.append(api.GraphLayer(tuple(samples)))
        frontier = reached
        projected.update(reached)
    if not layers:
        raise UnsupportedCombinationError(f"graph.root.{root}: no traversable input relationships")
    graph_fields = {}
    for name, table in tables.items():
        if name not in projected:
            continue
        schema = pq.read_schema(source.files[name])
        numeric = table.numeric if source.graph_numeric is None else source.graph_numeric[name]
        fields = [*table.key, *numeric.values()]
        fields += [c for c in (AVAILABLE_US, EVENT_US) if c in schema.names]
        if not table.immutable and AVAILABLE_US not in fields:
            raise SchemaError(f"{name}: normalize source clocks before graph tensor delivery")
        graph_fields[name] = api.Fields(
            tuple(api.Encode(column, api.Raw()) for column in dict.fromkeys(fields))
        )
    neighborhoods = {"neighbors": api.NeighborhoodSpec(tuple(layers))}
    graph = api.Input(api.NeighborhoodSource("neighbors"), api.Graph(graph_fields), None)
    if contract is None:
        return api.DataLoaderSpec(
            native_schema(source),
            api.SampleSpec(api.EntitySpec(root), neighborhoods=neighborhoods),
            api.BatchSpec(inputs={"graph": graph}),
        )
    return replace(
        contract,
        sample=replace(contract.sample, neighborhoods=neighborhoods),
        batch=replace(contract.batch, inputs={**contract.batch.inputs, "graph": graph}),
    )
