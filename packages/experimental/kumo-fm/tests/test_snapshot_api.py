# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot input validation before any native graph is sampled."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow as pa
import pytest

from kumo.fm import Context, Query
from kumo.fm._assembly import GraphAssembler
from kumo.fm._columnar import adapt_columnar_batch
from kumo.fm._snapshot import (
    SnapshotFMLoader,
    _rows,
    _schema_view,
    _selected_graph,
    _target_horizon,
)
from kumo.pytorch import api


def schema():
    return api.DataSchema(
        tables=(
            api.TableSpec("users", "user_id", {"user_id": api.DType("int64")},
                          time_column=api.NON_TEMPORAL),
            api.TableSpec("orders", "order_id", {
                "order_id": api.DType("int64"), "user_id": api.DType("int64"),
                "at": api.DType("timestamp", unit="s", timezone="UTC"),
                "amount": api.DType("float64")}, time_column="at"),
        ),
        relations=(api.RelationSpec("alias", "users", "user_id", "orders", "user_id"),),
    )


def test_population_order_and_repeated_occurrences():
    context = _rows(Context(entities=[7, 8], times=["2025-01-01T00:00:00Z",
                                                    "2025-02-01T00:00:00Z"]),
                    "context", "int64")
    assert [(row["instance_id"], row["entity_key"]) for row in context] == [
        (0, 7), (1, 8), (2, 7), (3, 8)]
    query = _rows(Query(pairs=[(7, "2025-03-01T00:00:00Z", 9007199254740993),
                               (7, "2025-03-01T00:00:00Z", -1)]), "predict", "int64")
    assert [row["instance_id"] for row in query] == [9007199254740993, -1]
    assert query[0]["anchor_time"] == datetime(2025, 3, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Duplicate occurrence"):
        _rows(Query(pairs=[(7, "2025-03-01T00:00:00Z", 1),
                           (8, "2025-03-01T00:00:00Z", 1)]), "predict", "int64")


def test_schema_alias_projection_and_target_window():
    bundle = _schema_view(schema())
    assert bundle.relations["alias"] == {"child": "orders", "parent": "users",
                                         "foreign_key": "user_id", "parent_is_source": True}
    target = api.Aggregate(api.Rows(path=("alias",),
                                    time=api.TimeWindow("at", start=0, stop=30)), "count")
    assert _target_horizon(target) == 30
    layer = api.GraphLayer((api.GraphSample(
        "alias", direction="forward", fanout=2, strategy="recent"),))
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((layer,))})
    name, fields, tables, relations, projections = _selected_graph(
        schema(), sample, {"users": [], "orders": ["amount"]}, bundle)
    assert (name, tables, relations) == ("history", {"users", "orders"}, {"alias"})
    assert projections == {"orders": ["order_id", "amount"], "users": ["user_id"]}
    assert {item.field for item in fields["orders"].fields} == {
        "order_id", "amount", "at", "user_id"}
    with pytest.raises(ValueError, match="uniformly into timeless"):
        reverse_layer = api.GraphLayer((api.GraphSample(
            "alias", direction="reverse", fanout=2, strategy="recent"),))
        reverse_sample = api.SampleSpec(
            entity=api.EntitySpec("orders"),
            neighborhoods={"history": api.NeighborhoodSpec((reverse_layer,))})
        _selected_graph(schema(), reverse_sample, {"orders": [], "users": []}, bundle)


def test_context_labels_join_by_occurrence_id_across_batches():
    class Task:
        def __init__(self, values):
            self.values = values

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def batches(self, **_):
            for identifier, value in self.values:
                yield SimpleNamespace(metadata={"sample_ids": [identifier]},
                                      targets={"label": {"values": SimpleNamespace(tolist=lambda: [value]),
                                                         "valid": SimpleNamespace(tolist=lambda: [True])}})

    loader = object.__new__(SnapshotFMLoader)
    loader._labels = SimpleNamespace(materialize=lambda _: Task([(22, 3), (11, 9)]))
    loader._batch_size = 1
    loader._classes = None
    assert loader._labels_for(None, [{"instance_id": 11}, {"instance_id": 22}]) == {
        22: 3.0, 11: 9.0}
    loader._labels = SimpleNamespace(materialize=lambda _: Task([(22, 3)]))
    with pytest.raises(ValueError, match="Missing FM context labels"):
        loader._labels_for(None, [{"instance_id": 11}, {"instance_id": 22}])
    loader._labels = SimpleNamespace(materialize=lambda _: Task([(22, 3), (22, 9)]))
    with pytest.raises(ValueError, match="duplicated"):
        loader._labels_for(None, [{"instance_id": 11}, {"instance_id": 22}])


@pytest.mark.parametrize("population,split", [
    (Context(entities=range(10), times=range(10)), "context"),
    (Query(entities=range(11), at=None), "predict"),
    (Context(pairs=[(1, None)] * 11), "context"),
    (Query(pairs=[(1, None)] * 11), "predict"),
])
def test_population_budget_precedes_row_validation(population, split):
    # Invalid times would fail conversion if expansion/validation happened first.
    with pytest.raises(ValueError, match="instance budget"):
        _rows(population, split, "int64", max_instances=10)


def test_population_budget_bounds_iterable_consumption():
    def oversized():
        for _ in range(11):
            yield (1, None)
        raise AssertionError("consumed past the budget sentinel")

    with pytest.raises(ValueError, match="instance budget"):
        _rows(Query(pairs=oversized()), "predict", "int64", max_instances=10)
    rows = _rows(Context(entities=[7, 8], times=["2025-01-01T00:00:00Z"] * 5),
                 "context", "int64", max_instances=10)
    assert [row["entity_key"] for row in rows] == [7, 8] * 5
    assert len(_rows(Query(pairs=[(7, "2025-01-01T00:00:00Z", i)
                                  for i in range(10)]),
                     "predict", "int64", max_instances=10)) == 10


@pytest.mark.parametrize("split", ["context", "predict"])
def test_materialize_passes_configured_population_budget(split):
    loader = object.__new__(SnapshotFMLoader)
    loader._closed = False
    loader._key_type = "int64"
    loader.workload = SimpleNamespace(budgets={"max_instances": 2})
    population = (Context(entities=range(2), times=[None, None]) if split == "context"
                  else Query(pairs=[(1, None)] * 3))
    with pytest.raises(ValueError, match="instance budget"):
        loader.materialize(**{"context" if split == "context" else "query": population})


@pytest.mark.parametrize("key_type", ["int64", "string"])
def test_source_instance_id_primary_key_rejected_during_schema_preparation(key_type):
    source = api.DataSchema(tables=(api.TableSpec(
        "employees", "instance_id", {"instance_id": api.DType(key_type)},
        time_column=api.NON_TEMPORAL),))
    with pytest.raises(ValueError, match="reserves instance_id.*employees"):
        _schema_view(source)


@pytest.mark.parametrize("parent_is_source", [True, False])
def test_self_relation_assembly_uses_declared_primary_key_endpoint(parent_is_source):
    left, right = ("id", "manager_id") if parent_is_source else ("manager_id", "id")
    source = api.DataSchema(tables=(api.TableSpec(
        "employees", "id", {"id": api.DType("int64"), "manager_id": api.DType("int64")},
        time_column=api.NON_TEMPORAL),), relations=(api.RelationSpec(
            "manager", "employees", left, "employees", right),))
    bundle = _schema_view(source)
    work = SimpleNamespace(bundle=bundle, entity="employees", table_names={"employees"},
                           relation_names={"manager"}, projections={"employees": ["id"]},
                           kind="regression", target={"dtype": "float64"},
                           budgets={"max_candidate_rows": 10, "max_rows_per_table": 10})
    adapter = SimpleNamespace(bundle=bundle, node_types=["employees"],
                              assembler=GraphAssembler(bundle), metrics={"native_candidate_rows": 0})
    parent, child = [0, 0], [1, 2]
    a, b = (parent, child) if parent_is_source else (child, parent)
    nodes = api.ColumnarNodes(
        ids=pa.array([10, 20, 30]), owners=pa.array([0, 0, 0], pa.uint64()),
        features=pa.table({"id": [10, 20, 30]}))
    edges = api.ColumnarEdges("employees", "employees", pa.table({
        "source": pa.array(a, pa.uint64()), "destination": pa.array(b, pa.uint64())}))
    graph = api.ColumnarGraph(root_type="employees", nodes={"employees": nodes},
                              edges={"manager": edges})
    batch = SimpleNamespace(inputs={"graph": graph}, metadata={"sample_ids": [0]})
    related = adapt_columnar_batch(adapter, work, [{"instance_id": 99, "entity_key": 10}], batch)
    assert related["employees"].to_pydict() == {
        "instance_id": [99, 99, 99], "id": [10, 20, 30],
        "__fm_edge_manager": [None, 10, 10]}


@pytest.mark.parametrize("parent_is_source", [True, False])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_graph_reachability_follows_declared_endpoints(parent_is_source, direction):
    tables = tuple(api.TableSpec(name, key, {
        key: api.DType("int64"), "user_id": api.DType("int64"),
        "at": api.DType("timestamp", unit="s", timezone="UTC")}, time_column="at")
        for name, key in (("users", "user_id"), ("orders", "order_id")))
    left, right = ("users", "orders") if parent_is_source else ("orders", "users")
    source = api.DataSchema(tables=tables, relations=(
        api.RelationSpec("link", left, "user_id", right, "user_id"),))
    root = left if direction == "forward" else right
    layer = api.GraphLayer((api.GraphSample("link", direction=direction, fanout=2,
                                            strategy="recent"),))
    neighborhood = {"history": api.NeighborhoodSpec((layer,))}
    sample = api.SampleSpec(entity=api.EntitySpec(root), neighborhoods=neighborhood)
    result = _selected_graph(source, sample, {"users": [], "orders": []}, _schema_view(source))
    assert result[2] == {"users", "orders"}
    wrong_root = right if direction == "forward" else left
    sample = api.SampleSpec(entity=api.EntitySpec(wrong_root), neighborhoods=neighborhood)
    with pytest.raises(ValueError, match="Unreachable graph layer"):
        _selected_graph(source, sample, {"users": [], "orders": []}, _schema_view(source))


def test_non_key_instance_id_feature_requires_projection_rejection():
    original = schema()
    orders = next(table for table in original.tables if table.name == "orders")
    users = next(table for table in original.tables if table.name == "users")
    source = api.DataSchema(tables=(users, api.TableSpec(
        orders.name, orders.primary_key, {**orders.columns, "instance_id": api.DType("int64")},
        time_column=orders.time_column)), relations=original.relations)
    layer = api.GraphLayer((api.GraphSample("alias", direction="forward", fanout=2,
                                            strategy="recent"),))
    sample = api.SampleSpec(entity=api.EntitySpec("users"), neighborhoods={
        "history": api.NeighborhoodSpec((layer,))})
    bundle = _schema_view(source)
    with pytest.raises(ValueError, match="reserves instance_id.*orders.instance_id"):
        _selected_graph(source, sample, {"users": [], "orders": ["instance_id"]}, bundle)
    result = _selected_graph(source, sample, {"users": [], "orders": ["amount"]}, bundle)
    assert result[4]["orders"] == ["order_id", "amount"]
    assert "instance_id" not in {item.field for item in result[1]["orders"].fields}
