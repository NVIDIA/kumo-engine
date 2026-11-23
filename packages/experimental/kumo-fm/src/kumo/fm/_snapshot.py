# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FM orchestration over an already indexed public PyTorch engine."""
from __future__ import annotations

import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._assembly import GraphAssembler
from ._columnar import adapt_columnar_batch
from ._types import TYPES, timestamp, validate_scalar
from .batch import PayloadBatch


@dataclass(frozen=True)
class Context:
    """Historical examples: entities × times, or ordered explicit pairs.

    A pair may include a third, explicit int64 occurrence ID. Repeated pairs
    remain distinct; omitted IDs are their zero-based positions in the split.
    """

    entities: object = ()
    times: object = ()
    pairs: object = ()


@dataclass(frozen=True)
class Query:
    """Prediction examples: entities at one time, or ordered explicit pairs."""

    entities: object = ()
    at: object = None
    pairs: object = ()


def _rows(population, split, key_dtype):
    if split == "context" and not isinstance(population, Context):
        raise TypeError("context requires fm.Context")
    if split == "predict" and not isinstance(population, Query):
        raise TypeError("query requires fm.Query")
    pairs = list(population.pairs)
    entities = list(population.entities)
    if pairs and (entities or (split == "context" and population.times) or
                  (split == "predict" and population.at is not None)):
        raise ValueError("Choose explicit pairs or entities with times")
    if not pairs:
        if split == "context":
            pairs = [(entity, at) for at in population.times for entity in entities]
        else:
            pairs = [(entity, population.at) for entity in entities]
    if not pairs:
        raise ValueError(f"{split} needs at least one occurrence")
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


def _schema_view(schema):
    """Derive FM's supported scalar schema without touching the base index."""
    from kumo.pytorch.contract.schema import table_clock

    tables = {}
    for table in schema.tables:
        if table.primary_key is None:
            raise ValueError(f"FM requires a primary key: {table.name}")
        columns = {}
        for name, dtype in table.columns.items():
            arrow = dtype.to_arrow()
            if pa.types.is_timestamp(arrow) and arrow.unit != "ns":
                physical = "timestamp[us]"
            elif pa.types.is_date32(arrow):
                physical = "date"
            elif pa.types.is_float64(arrow):
                physical = "float64"
            elif pa.types.is_boolean(arrow):
                physical = "bool"
            elif pa.types.is_int64(arrow):
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
                              "time_column": table_clock(table), "columns": columns}
    relations = {}
    for relation in schema.relations:
        left, right = tables[relation.left_table], tables[relation.right_table]
        if relation.right_key == right["primary_key"]:
            child, parent, fk = relation.left_table, relation.right_table, relation.left_key
        elif relation.left_key == left["primary_key"]:
            child, parent, fk = relation.right_table, relation.left_table, relation.right_key
        else:
            raise ValueError(f"FM requires one primary-key side: {relation.name}")
        relations[relation.name] = {"child": child, "parent": parent, "foreign_key": fk}
    return SimpleNamespace(tables=tables, relations=relations)


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
            source, dest = ((rel["child"], rel["parent"]) if step.direction == "forward"
                            else (rel["parent"], rel["child"]))
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


class SnapshotFMLoader:
    """Prepare FM labels and columnar graph plans over one adopted snapshot."""

    def __init__(self, *, engine, sample, target, task, features, classes=None,
                 batch_size=1024, budgets=None, schema=None):
        from kumo.pytorch import api

        from .loader import PreparedRequest

        if not isinstance(engine, api.DiskGraphEngine) or not isinstance(sample, api.SampleSpec):
            raise TypeError("Provide an adopted DiskGraphEngine and SampleSpec")
        schema = engine.schema if schema is None else schema
        if not isinstance(schema, api.DataSchema):
            raise TypeError("schema must be a public DataSchema projection or alias")
        if target not in sample.targets or task not in ("regression", "classification", "binary_classification"):
            raise ValueError("Choose a declared target and supported FM task")
        self._prepared_request_type = PreparedRequest
        self._closed = False
        self._engine, self._target = engine, target
        self._bundle = _schema_view(schema)
        self._entity = sample.entity.table
        self._key_type = self._bundle.tables[self._entity]["columns"][
            self._bundle.tables[self._entity]["primary_key"]]["dtype"]
        self._horizon = _target_horizon(sample.targets[target])
        self._kind = "regression" if task == "regression" else "binary_classification"
        if self._kind == "regression":
            if classes is not None:
                raise ValueError("Regression has no classes")
            self._classes = None
            target_spec = {"dtype": "float64"}
        else:
            if not isinstance(classes, dict) or len(classes) != 2 or len(set(classes.values())) != 2:
                raise ValueError("Binary classification requires a two-entry raw-value to class-name map")
            self._classes = dict(classes)
            target_spec = {"dtype": "string", "classes": list(classes.values())}
        graph_name, fields, tables, relations, projections = _selected_graph(
            schema, sample, features, self._bundle)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        default_budgets = {"max_instances": 10000, "max_rows_per_table": 1000000,
                           "max_payload_bytes": 30 * 1024**2,
                           "max_decoded_arrow_bytes": 30 * 1024**2,
                           "max_columns": 500, "max_candidate_rows": 1000000}
        default_budgets.update(budgets or {})
        if any(type(value) is not int or value < 1 for value in default_budgets.values()):
            raise ValueError("FM budgets must be positive integers")
        self._batch_size = min(batch_size, default_budgets["max_instances"])
        self.workload = SimpleNamespace(bundle=self._bundle, entity=self._entity,
                                        kind=self._kind, target=target_spec, table_names=tables,
                                        relation_names=relations, projections=projections, budgets=default_budgets)
        self._adapter = SimpleNamespace(bundle=self._bundle,
                                        node_types=list(self._bundle.tables), assembler=GraphAssembler(self._bundle),
                                        metrics={"native_candidate_rows": 0})
        self._stack = ExitStack()
        self._temporary = tempfile.TemporaryDirectory(prefix="kumo-fm-anchors-")
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

    @property
    def preparation_mode(self):
        return "existing_snapshot"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()
            self._temporary.cleanup()

    def materialize(self, *, context=None, query=None):
        if self._closed:
            raise RuntimeError("Loader is closed")
        if context is None and query is None:
            raise ValueError("Provide context or query occurrences")
        splits = {}
        for split, population in (("context", context), ("predict", query)):
            if population is not None:
                seeds = _rows(population, split, self._key_type)
                if len(seeds) > self.workload.budgets["max_instances"]:
                    raise ValueError("FM instance budget exceeded")
                splits[split] = seeds
        if "context" in splits:
            if "predict" not in splits:
                raise ValueError("Context labels require a query cutoff")
            cutoff = min(row["anchor_time"] for row in splits["predict"])
            if any(row["anchor_time"] + timedelta(seconds=self._horizon) > cutoff
                   for row in splits["context"]):
                raise ValueError("Context label window is not observable before the earliest query")
        return self._prepared_request_type(self, splits)

    def load(self, *, context=None, query=None):
        with self.materialize(context=context, query=query) as request:
            return request.load()

    def _anchor(self, seeds):
        from kumo.pytorch import api

        path = Path(self._temporary.name) / f"{uuid.uuid4().hex}.parquet"
        pq.write_table(pa.table({
            "entity": pa.array([s["entity_key"] for s in seeds], TYPES[self._key_type]),
            "sample": pa.array([s["instance_id"] for s in seeds], pa.int64()),
            "at": pa.array([s["anchor_time"] for s in seeds], pa.timestamp("us", tz="UTC")),
        }), path)
        return api.AnchorTable(api.Parquet(path=str(path)), entity_id="entity",
                               sample_id="sample", as_of_time="at")

    def _labels_for(self, anchor, seeds):
        result = {}
        expected = {s["instance_id"] for s in seeds}
        with self._labels.materialize(anchor) as task:
            for batch in task.batches(output_format="inspectable", shuffle=False,
                                      batch_size=self._batch_size):
                label = batch.targets["label"]
                sample_ids = list(batch.metadata["sample_ids"])
                values, valid = label["values"].tolist(), label["valid"].tolist()
                if len(sample_ids) != len(values) or len(values) != len(valid):
                    raise ValueError("FM label batch is misaligned")
                for identifier, value, present in zip(sample_ids, values, valid, strict=True):
                    identifier = int(identifier)
                    if identifier not in expected or identifier in result or not present:
                        raise ValueError("Missing, duplicated, or unexpected FM context label")
                    if self._classes is not None:
                        if value not in self._classes:
                            raise ValueError("Context label is outside declared classes")
                        value = self._classes[value]
                    else:
                        value = float(value)
                    result[identifier] = value
        if set(result) != expected:
            raise ValueError("Missing FM context labels")
        return result

    def _graphs_for(self, anchor, seeds):
        from ._contract import schema_for

        expected = {s["instance_id"]: s for s in seeds}
        seen = set()
        parts = {name: [] for name in schema_for(self.workload)["related_tables"]}
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
                    self._adapter, self.workload, chunk, local)
                for name, table in related.items():
                    parts[name].append(table)
        if seen != set(expected):
            raise ValueError("Missing FM graph occurrences")
        order = {seed["instance_id"]: i for i, seed in enumerate(seeds)}
        result = {}
        for name, chunks in parts.items():
            table = pa.concat_tables(chunks) if chunks else pa.table({})
            if len(table):
                rank = pa.array([order[int(value)] for value in table["instance_id"].to_pylist()])
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
            if split == "context":
                labels = self._labels_for(anchor, seeds)
                seeds = [dict(seed, target=labels[seed["instance_id"]]) for seed in seeds]
            ready[split] = seeds, self._graphs_for(anchor, seeds)
        return PayloadBatch(self.workload, ready)
