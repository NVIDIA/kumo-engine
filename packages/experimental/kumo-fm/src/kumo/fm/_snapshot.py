# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FM orchestration over an already indexed public PyTorch engine."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, timedelta
from itertools import islice
from numbers import Complex
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ._assembly import GraphAssembler
from ._columnar import adapt_columnar_batch
from ._contract import _Budgets, _PayloadSpec, _SchemaView, _TargetSpec
from ._types import TYPES, timestamp, validate_scalar
from .batch import PayloadBatch, _SplitPayload


@dataclass(frozen=True)
class Context:
    """Historical examples: entities × times, or ordered explicit pairs.

    A pair may include a third, explicit int64 occurrence ID. Repeated pairs
    remain distinct; omitted IDs are their zero-based positions in the split.
    """

    entities: Iterable[int | str] = ()
    times: Iterable[date | datetime | str] = ()
    pairs: Iterable[tuple[int | str, date | datetime | str] | tuple[int | str, date | datetime | str, int]] = ()


@dataclass(frozen=True)
class Query:
    """Prediction examples: entities at one time, or ordered explicit pairs."""

    entities: Iterable[int | str] = ()
    at: date | datetime | str | None = None
    pairs: Iterable[tuple[int | str, date | datetime | str] | tuple[int | str, date | datetime | str, int]] = ()


def _rows(population, split, key_dtype, max_instances=10000):
    if split == "context" and not isinstance(population, Context):
        raise TypeError("context requires fm.Context")
    if split == "predict" and not isinstance(population, Query):
        raise TypeError("query requires fm.Query")

    def bounded(values):
        result = list(islice(values, max_instances + 1))
        if len(result) > max_instances:
            raise ValueError("FM instance budget exceeded")
        return result

    pairs = bounded(population.pairs)
    entities = bounded(population.entities)
    times = bounded(population.times) if split == "context" else [population.at]
    if pairs and (entities or (split == "context" and times) or
                  (split == "predict" and population.at is not None)):
        raise ValueError("Choose explicit pairs or entities with times")
    if not pairs:
        if len(entities) * len(times) > max_instances:
            raise ValueError("FM instance budget exceeded")
        if not entities or not times:
            raise ValueError(f"{split} needs at least one occurrence")
        pairs = ((entity, at) for at in times for entity in entities)
    result = []
    seen = set()
    for position, item in enumerate(pairs):
        if len(item) not in (2, 3):
            raise ValueError("Occurrences require (entity, time[, instance_id])")
        entity, at = item[:2]
        validate_scalar(entity, key_dtype)
        instant = timestamp(at)
        identifier = item[2] if len(item) == 3 else position
        validate_scalar(identifier, "int64")
        if identifier in seen:
            raise ValueError("Duplicate occurrence ID within a split")
        seen.add(identifier)
        result.append({"instance_id": identifier, "entity_key": entity,
                       "anchor_time": instant})
    return result


def _schema_view(schema, sample, features):
    """Derive FM scalar types only for graph inputs retained by the request."""
    from kumo.pytorch.contract.schema import table_clock

    selected_relations = {step.relation for neighborhood in sample.neighborhoods.values()
                          for layer in neighborhood.layers for step in layer.samples}
    selected_tables = {sample.entity.table}
    required = {name: set(columns) for name, columns in features.items()}
    for relation in schema.relations:
        if relation.name in selected_relations:
            selected_tables.update((relation.left_table, relation.right_table))
            required.setdefault(relation.left_table, set()).add(relation.left_key)
            required.setdefault(relation.right_table, set()).add(relation.right_key)
    tables = {}
    for table in schema.tables:
        if table.name not in selected_tables:
            continue
        if table.primary_key is None:
            raise ValueError(f"FM requires a primary key: {table.name}")
        if table.primary_key == "instance_id":
            raise ValueError(f"FM reserves instance_id for occurrence IDs: {table.name}")
        clock = table_clock(table)
        retained = required.get(table.name, set()) | {table.primary_key, clock}
        columns = {}
        for name, dtype in table.columns.items():
            if name not in retained:
                continue
            arrow = dtype.to_arrow()
            if pa.types.is_timestamp(arrow) and arrow.unit != "ns":
                physical = "timestamp[us]"
            elif pa.types.is_date32(arrow):
                physical = "date"
            elif pa.types.is_float64(arrow) or pa.types.is_float32(arrow):
                physical = "float64"
            elif pa.types.is_boolean(arrow):
                physical = "bool"
            elif pa.types.is_int64(arrow) or pa.types.is_int32(arrow):
                physical = "int64"
            elif pa.types.is_string(arrow):
                physical = "string"
            else:
                physical = str(arrow)
            if physical not in TYPES:
                raise ValueError(f"FM cannot encode {table.name}.{name}: {arrow}")
            columns[name] = {"dtype": physical,
                             "nullable": name != table.primary_key}
        if columns[table.primary_key]["dtype"] not in ("int64", "string"):
            raise ValueError(f"FM needs an int64/string key: {table.name}")
        tables[table.name] = {"primary_key": table.primary_key,
                              "time_column": clock, "columns": columns}
    relations = {}
    for relation in schema.relations:
        if relation.name not in selected_relations:
            continue
        left, right = tables[relation.left_table], tables[relation.right_table]
        if relation.right_key == right["primary_key"]:
            child, parent, fk = relation.left_table, relation.right_table, relation.left_key
            parent_is_source = False
        elif relation.left_key == left["primary_key"]:
            child, parent, fk = relation.right_table, relation.left_table, relation.right_key
            parent_is_source = True
        else:
            raise ValueError(f"FM requires one primary-key side: {relation.name}")
        relations[relation.name] = {"child": child, "parent": parent, "foreign_key": fk,
                                    "parent_is_source": parent_is_source}
    return _SchemaView(tables=tables, relations=relations)


def _target_horizon(value):
    from kumo.pytorch import api

    aggregate = value.left if isinstance(value, api.Compare) else value
    if not isinstance(aggregate, api.Aggregate) or not isinstance(aggregate.source, api.Rows):
        raise ValueError("FM targets require a bounded Rows aggregate or comparison")
    window = aggregate.source.time
    if window is None or window.start != 0 or window.stop is None or window.stop <= 0:
        raise ValueError("FM target needs a future window (0, positive stop]")
    return window.stop


def _selected_graph(schema, sample, features, bundle):
    from kumo.pytorch import api

    if len(sample.neighborhoods) != 1:
        raise ValueError("FM requires exactly one declared NeighborhoodSpec")
    graph_name, neighborhood = next(iter(sample.neighborhoods.items()))
    frontier = {sample.entity.table}
    tables, relations = set(frontier), set()
    if not neighborhood.layers:
        raise ValueError("FM requires at least one graph layer")
    for layer in neighborhood.layers:
        if not layer.samples:
            raise ValueError("FM cannot sample an empty graph layer")
        following = set()
        for step in layer.samples:
            if step.strategy != "recent" or not step.fanout or step.fanout < 0:
                raise ValueError("FM supports bounded recent graph sampling only")
            if step.relation not in bundle.relations:
                raise ValueError(f"Unknown relation: {step.relation}")
            rel = bundle.relations[step.relation]
            left, right = ((rel["parent"], rel["child"]) if rel["parent_is_source"]
                           else (rel["child"], rel["parent"]))
            source, dest = (left, right) if step.direction == "forward" else (right, left)
            if source not in frontier:
                raise ValueError(f"Unreachable graph layer: {source}")
            if bundle.tables[dest]["time_column"] is None:
                raise ValueError(f"FM cannot sample uniformly into timeless table {dest}")
            following.add(dest)
            tables.update((source, dest))
            relations.add(step.relation)
        frontier = following
    if set(features) != tables:
        raise ValueError(f"features must name reachable tables: {sorted(tables)}")
    projections = {}
    fields = {}
    for name in sorted(tables):
        declared = bundle.tables[name]["columns"]
        requested = list(features[name])
        if len(requested) != len(set(requested)) or not set(requested) <= set(declared):
            raise ValueError(f"Invalid FM feature projection: {name}")
        if "instance_id" in requested:
            raise ValueError(f"FM reserves instance_id for occurrence IDs: {name}.instance_id")
        pk = bundle.tables[name]["primary_key"]
        projections[name] = list(dict.fromkeys((pk, *requested)))
        internal = list(projections[name])
        clock = bundle.tables[name]["time_column"]
        if clock:
            internal.append(clock)
        internal.extend(bundle.relations[r]["foreign_key"] for r in relations
                        if bundle.relations[r]["child"] == name)
        fields[name] = api.Fields(tuple(api.Encode(column, api.Raw())
                                        for column in dict.fromkeys(internal)))
    return graph_name, fields, tables, relations, projections


def _label_array(values):
    return values if isinstance(values, (pa.Array, pa.ChunkedArray)) else pa.array(np.asarray(values))


def _class_labels(values, classes):
    # Only the two declared class keys/names cross Python; labels stay columnar.
    keys, names = [], []
    dtype = values.type
    for key in classes:
        if pa.types.is_string(dtype):
            if not isinstance(key, str):
                continue
            normalized = key
        else:
            try:
                numeric = key.real if isinstance(key, Complex) and key.imag == 0 else key
                if pa.types.is_boolean(dtype):
                    if key not in (0, 1):
                        continue
                    normalized = bool(key)
                elif pa.types.is_integer(dtype):
                    normalized = int(numeric)
                elif pa.types.is_floating(dtype):
                    normalized = float(numeric)
                else:
                    continue
                normalized = pa.scalar(normalized, type=dtype).as_py()
            except (ValueError, TypeError, OverflowError):
                continue
        if normalized in classes:
            keys.append(normalized)
            names.append(str(classes[normalized]))
    indices = pc.index_in(values, value_set=pa.array(keys, type=dtype))
    if indices.null_count:
        raise ValueError("Context label is outside declared classes")
    return pc.take(pa.array(names, type=pa.string()), indices)


class PreparedRequest:
    """Resolved anchors; every load call samples exactly once per included split."""

    def __init__(self, loader, splits):
        self._loader = loader
        self._splits = splits
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._closed = True

    def load(self):
        if self._closed or self._loader._closed:
            raise RuntimeError("Prepared request or loader is closed")
        return self._loader._load(self._splits)


class _FMLoader:
    """Prepare FM labels and columnar graph plans over one adopted snapshot."""

    def __init__(self, *, engine, sample, target, task, features, classes=None,
                 batch_size=1024, budgets=None, schema=None):
        from kumo.pytorch import api

        if not isinstance(engine, api.DiskGraphEngine) or not isinstance(sample, api.SampleSpec):
            raise TypeError("Provide an adopted DiskGraphEngine and SampleSpec")
        schema = engine.schema if schema is None else schema
        if not isinstance(schema, api.DataSchema):
            raise TypeError("schema must be a public DataSchema projection or alias")
        if target not in sample.targets:
            raise ValueError(f"Unknown FM target: {target}")
        if task not in ("regression", "binary_classification"):
            raise ValueError("task must be 'regression' or 'binary_classification'")
        self._closed = False
        self._engine, self._target = engine, target
        features = {name: tuple(columns) for name, columns in features.items()}
        self._bundle = _schema_view(schema, sample, features)
        self._entity = sample.entity.table
        self._key_type = self._bundle.tables[self._entity]["columns"][
            self._bundle.tables[self._entity]["primary_key"]]["dtype"]
        self._horizon = _target_horizon(sample.targets[target])
        self._kind = task
        if self._kind == "regression":
            if classes is not None:
                raise ValueError("Regression has no classes")
            self._classes = None
            target_spec = _TargetSpec("float64")
        else:
            if not isinstance(classes, dict) or len(classes) != 2 or len(set(classes.values())) != 2:
                raise ValueError("Binary classification requires a two-entry raw-value to class-name map")
            self._classes = dict(classes)
            target_spec = _TargetSpec("string", tuple(str(value) for value in classes.values()))
        graph_name, fields, tables, relations, projections = _selected_graph(
            schema, sample, features, self._bundle)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if budgets is not None and not isinstance(budgets, Mapping):
            raise TypeError("budgets must be a mapping of documented FM limits")
        unknown = set(budgets or ()) - {field.name for field in dataclass_fields(_Budgets)}
        if unknown:
            raise ValueError(f"Unknown FM budget keys: {sorted(unknown, key=str)}")
        limits = _Budgets(**(budgets or {}))
        self._batch_size = min(batch_size, limits.max_instances)
        self._payload = _PayloadSpec(bundle=self._bundle, entity=self._entity,
                                     kind=self._kind, target=target_spec, table_names=frozenset(tables),
                                     relation_names=frozenset(relations), projections=projections, budgets=limits)
        self._adapter = SimpleNamespace(bundle=self._bundle,
                                        node_types=list(self._bundle.tables), assembler=GraphAssembler(self._bundle),
                                        metrics={"native_candidate_rows": 0})
        self._stack = ExitStack()
        pk = self._bundle.tables[self._entity]["primary_key"]
        label_sample = api.SampleSpec(entity=sample.entity,
                                      entity_values={pk: api.Field(pk)}, targets={target: sample.targets[target]})
        label_batch = api.BatchSpec(inputs={"entity": api.Input(
            api.EntitySource(), api.Fields((api.Encode(pk, api.Raw()),)), api.Separate())},
            targets={"label": api.LabelTarget(target, api.Raw())},
            batching=api.Batching(max_samples=self._batch_size))
        graph_sample = api.SampleSpec(entity=sample.entity,
                                      neighborhoods={graph_name: sample.neighborhoods[graph_name]})
        graph_batch = api.BatchSpec(inputs={"graph": api.Input(
            api.NeighborhoodSource(graph_name), api.Graph(node_fields=fields), None)},
            batching=api.Batching(max_samples=self._batch_size))
        try:
            self._labels = self._stack.enter_context(api.prepare(
                contract=api.DataLoaderSpec(schema, label_sample, label_batch), engine=engine))
            self._graphs = self._stack.enter_context(api.prepare(
                contract=api.DataLoaderSpec(schema, graph_sample, graph_batch), engine=engine))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()

    def materialize(self, *, context=None, query=None):
        if self._closed:
            raise RuntimeError("Loader is closed")
        if context is None and query is None:
            raise ValueError("Provide context or query occurrences")
        splits = {}
        for split, population in (("context", context), ("predict", query)):
            if population is not None:
                seeds = _rows(population, split, self._key_type,
                              self._payload.budgets.max_instances)
                splits[split] = seeds
        if "context" in splits:
            if "predict" not in splits:
                raise ValueError("Context labels require a query cutoff")
            cutoff = min(row["anchor_time"] for row in splits["predict"])
            if any(row["anchor_time"] + timedelta(seconds=self._horizon) > cutoff
                   for row in splits["context"]):
                raise ValueError("Context label window is not observable before the earliest query")
        return PreparedRequest(self, splits)

    def load(self, *, context=None, query=None):
        with self.materialize(context=context, query=query) as request:
            return request.load()

    def _anchor(self, seeds):
        from kumo.pytorch import api

        return api.AnchorTable(api.InMemoryRows([
            {"entity": seed["entity_key"], "sample": seed["instance_id"], "at": seed["anchor_time"]}
            for seed in seeds
        ]), entity_id="entity", sample_id="sample", as_of_time="at")

    def _labels_for(self, anchor, seeds):
        expected = pa.array([seed["instance_id"] for seed in seeds], pa.int64())
        parts = []
        with self._labels.materialize(anchor) as task:
            for batch in task.batches(output_format="inspectable", shuffle=False,
                                      batch_size=self._batch_size):
                label = batch.targets["label"]
                identifiers = pa.array(batch.metadata["sample_ids"], pa.int64())
                values, valid = _label_array(label["values"]), _label_array(label["valid"])
                if len(identifiers) != len(values) or len(values) != len(valid):
                    raise ValueError("FM label batch is misaligned")
                parts.append(pa.table({"id": identifiers, "value": values, "valid": valid}))
        if not parts:
            raise ValueError(f"Missing FM context labels for occurrence IDs {expected.to_pylist()}")
        labels = pa.concat_tables(parts)
        identifiers = labels["id"]
        unexpected = pc.is_null(pc.index_in(identifiers, value_set=expected))
        if pc.any(unexpected).as_py():
            raise ValueError(f"Unexpected FM context label for occurrence IDs {identifiers.filter(unexpected).to_pylist()}")
        counts = pc.value_counts(identifiers)
        duplicated = pc.greater(counts.field("counts"), 1)
        if pc.any(duplicated).as_py():
            repeated = counts.field("values").filter(duplicated).to_pylist()
            raise ValueError(f"FM context label duplicated for occurrence IDs {repeated}")
        positions = pc.index_in(expected, value_set=identifiers)
        if positions.null_count:
            missing = expected.filter(pc.is_null(positions)).to_pylist()
            raise ValueError(f"Missing FM context labels for occurrence IDs {missing}")
        valid = pc.take(labels["valid"], positions)
        nulls = pc.invert(pc.fill_null(valid, False))
        if pc.any(nulls).as_py():
            missing = expected.filter(nulls).to_pylist()
            raise ValueError(f"FM context target {self._target!r} is null for occurrence IDs {missing}; "
                             "context targets must be non-null for every occurrence")
        values = pc.take(labels["value"], positions)
        return pc.cast(values, pa.float64()) if self._classes is None else _class_labels(values, self._classes)

    def _graphs_for(self, anchor, seeds):
        from ._contract import schema_for

        expected = {s["instance_id"]: s for s in seeds}
        seen = set()
        parts = {name: [] for name in schema_for(self._payload)["related_tables"]}
        with self._graphs.materialize(anchor) as task:
            for batch in task.batches(output_format="columnar", shuffle=False,
                                      batch_size=self._batch_size):
                identifiers = [int(value) for value in batch.metadata["sample_ids"]]
                if (len(identifiers) != len(set(identifiers)) or
                        any(value not in expected or value in seen for value in identifiers)):
                    raise ValueError("Duplicate or unexpected FM graph occurrence")
                seen.update(identifiers)
                chunk = [expected[value] for value in identifiers]
                local = SimpleNamespace(inputs=batch.inputs,
                                        metadata={"sample_ids": list(range(len(chunk)))})
                related = adapt_columnar_batch(
                    self._adapter, self._payload, chunk, local)
                for name, table in related.items():
                    parts[name].append(table)
        if seen != set(expected):
            raise ValueError("Missing FM graph occurrences")
        order = pa.array([seed["instance_id"] for seed in seeds], pa.int64())
        result = {}
        for name, chunks in parts.items():
            table = pa.concat_tables(chunks) if chunks else pa.table({})
            if len(table):
                rank = pc.index_in(table["instance_id"], value_set=order)
                index = pc.sort_indices(pa.table({"rank": rank,
                                                  "key": table[self._bundle.tables[name]["primary_key"]]}),
                                        sort_keys=[("rank", "ascending"), ("key", "ascending")])
                table = table.take(index)
            result[name] = table
        return result

    def _load(self, splits):
        if self._closed:
            raise RuntimeError("Loader is closed")
        ready = {}
        for split, seeds in splits.items():
            anchor = self._anchor(seeds)
            labels = self._labels_for(anchor, seeds) if split == "context" else None
            ready[split] = _SplitPayload(seeds, self._graphs_for(anchor, seeds), labels)
        return PayloadBatch(self._payload, ready)
