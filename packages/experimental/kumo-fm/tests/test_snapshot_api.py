# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot input validation before any native graph is sampled."""
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pytest

from kumo.fm import Context, Query
from kumo.fm._assembly import assemble_graph
from kumo.fm._contract import _Budgets, _PayloadSpec, _TargetSpec
from kumo.fm._snapshot import (
    _align_context_labels,
    _class_labels,
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
    target = api.Aggregate(api.Rows(path=("alias",),
                                    time=api.TimeWindow("at", start=0, stop=30)), "count")
    assert _target_horizon(target) == 30
    layer = api.GraphLayer((api.GraphSample(
        "alias", direction="forward", fanout=2, strategy="recent"),))
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((layer,))})
    bundle = _schema_view(schema(), sample, {"users": [], "orders": ["amount"]})
    assert bundle.relations["alias"] == {"child": "orders", "parent": "users",
                                         "foreign_key": "user_id", "parent_is_source": True}
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
    ids = pa.array([9007199254740993, -9007199254740993], pa.int64())
    parts = [pa.table({"id": [identifier], "value": [value], "valid": [True]})
             for identifier, value in [(-9007199254740993, 3.0), (9007199254740993, 9.0)]]
    aligned = _align_context_labels(parts, ids, target="future", classes=None)
    assert aligned.to_pylist() == [9.0, 3.0]


@pytest.mark.parametrize("parts,message", [
    ([], "Missing FM context labels"),
    ([pa.table({"id": [22], "value": [3.0], "valid": [True]})], "Missing FM context labels.*11"),
    ([pa.table({"id": [22, 22], "value": [3.0, 9.0], "valid": [True, True]})], "duplicated.*22"),
    ([pa.table({"id": [22], "value": [3.0], "valid": [True]}),
      pa.table({"id": [22], "value": [9.0], "valid": [True]})], "duplicated.*22"),
    ([pa.table({"id": [99], "value": [3.0], "valid": [True]})], "Unexpected FM context label.*99"),
    ([pa.table({"id": pa.array([None], pa.int64()), "value": [3.0], "valid": [True]})],
     "Unexpected FM context label.*None"),
])
def test_arrow_label_alignment_refuses_missing_duplicated_or_unexpected_ids(parts, message):
    with pytest.raises(ValueError, match=message):
        _align_context_labels(parts, pa.array([11, 22], pa.int64()), target="future", classes=None)


def test_null_context_label_ids_are_reported_in_request_order():
    parts = [pa.table({"id": [22, 11], "value": [0.0, 0.0], "valid": [False, False]})]
    with pytest.raises(ValueError) as error:
        _align_context_labels(parts, pa.array([11, 22], pa.int64()), target="future", classes=None)
    assert str(error.value) == (
        "FM context target 'future' is null for occurrence IDs [11, 22]; "
        "context targets must be non-null for every occurrence")


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


@pytest.mark.parametrize("key_type", ["int64", "string"])
def test_source_instance_id_primary_key_rejected_during_schema_preparation(key_type):
    source = api.DataSchema(tables=(api.TableSpec(
        "employees", "instance_id", {"instance_id": api.DType(key_type)},
        time_column=api.NON_TEMPORAL),))
    with pytest.raises(ValueError, match="reserves instance_id.*employees"):
        _schema_view(source, api.SampleSpec(entity=api.EntitySpec("employees")), {"employees": []})


@pytest.mark.parametrize("parent_is_source", [True, False])
@pytest.mark.parametrize("repeated_edge", [True, False])
def test_self_relation_assembly_uses_declared_primary_key_endpoint(parent_is_source, repeated_edge):
    left, right = ("id", "manager_id") if parent_is_source else ("manager_id", "id")
    source = api.DataSchema(tables=(api.TableSpec(
        "employees", "id", {"id": api.DType("int64"), "manager_id": api.DType("int64")},
        time_column=api.NON_TEMPORAL),), relations=(api.RelationSpec(
            "manager", "employees", left, "employees", right),))
    declaration = api.SampleSpec(entity=api.EntitySpec("employees"), neighborhoods={
        "history": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("manager", fanout=2),)),))})
    bundle = _schema_view(source, declaration, {"employees": ["id"]})
    work = _PayloadSpec(schema_view=bundle, entity="employees", table_names=frozenset({"employees"}),
                        relation_names=frozenset({"manager"}), projections={"employees": ["id"]},
                        kind="regression", target=_TargetSpec("float64"),
                        budgets=_Budgets(max_candidate_rows=10, max_rows_per_table=10))
    parent, child = ([0, 0, 0], [1, 2, 1]) if repeated_edge else ([0, 0], [1, 2])
    a, b = (parent, child) if parent_is_source else (child, parent)
    nodes = api.ColumnarNodes(
        ids=pa.array([10, 20, 30]), owners=pa.array([0, 0, 0], pa.uint64()),
        features=pa.table({"id": [10, 20, 30]}))
    edges = api.ColumnarEdges("employees", "employees", pa.table({
        "source": pa.array(a, pa.uint64()), "destination": pa.array(b, pa.uint64())}))
    graph = api.ColumnarGraph(root_type="employees", nodes={"employees": nodes},
                              edges={"manager": edges})
    related = assemble_graph(work, [{"instance_id": 99, "entity_key": 10}], graph)
    assert related["employees"].to_pydict() == {
        "instance_id": [99, 99, 99], "id": [10, 20, 30],
        "__fm_edge_manager": [None, 10, 10]}
    parent, child = [0, 2], [1, 1]
    a, b = (parent, child) if parent_is_source else (child, parent)
    conflicting = api.ColumnarGraph(root_type="employees", nodes={"employees": nodes}, edges={
        "manager": api.ColumnarEdges("employees", "employees", pa.table({
            "source": pa.array(a, pa.uint64()), "destination": pa.array(b, pa.uint64())}))})
    with pytest.raises(ValueError, match="Selected child has conflicting parents"):
        assemble_graph(work, [{"instance_id": 99, "entity_key": 10}], conflicting)
    limited = replace(work, budgets=replace(work.budgets, max_rows_per_table=2))
    # Refuse oversized node columns before attempting the conflicting parent map.
    with pytest.raises(ValueError, match="Expanded related table exceeds row budget"):
        assemble_graph(limited, [{"instance_id": 99, "entity_key": 10}], conflicting)


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
    result = _selected_graph(source, sample, {"users": [], "orders": []}, _schema_view(source, sample, {"users": [], "orders": []}))
    assert result[2] == {"users", "orders"}
    wrong_root = right if direction == "forward" else left
    sample = api.SampleSpec(entity=api.EntitySpec(wrong_root), neighborhoods=neighborhood)
    with pytest.raises(ValueError, match="Unreachable graph layer"):
        _selected_graph(source, sample, {"users": [], "orders": []}, _schema_view(source, sample, {"users": [], "orders": []}))


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
    bundle = _schema_view(source, sample, {"users": [], "orders": ["instance_id", "amount"]})
    with pytest.raises(ValueError, match="reserves instance_id.*orders.instance_id"):
        _selected_graph(source, sample, {"users": [], "orders": ["instance_id"]}, bundle)
    result = _selected_graph(source, sample, {"users": [], "orders": ["amount"]}, bundle)
    assert result[4]["orders"] == ["order_id", "amount"]
    assert "instance_id" not in {item.field for item in result[1]["orders"].fields}


def test_schema_view_ignores_unrequested_types_and_unrelated_tables():
    original = schema()
    users = next(table for table in original.tables if table.name == "users")
    orders = next(table for table in original.tables if table.name == "orders")
    source = api.DataSchema(tables=(
        api.TableSpec(users.name, users.primary_key,
                      {**users.columns, "opaque": api.DType("bytes")}, time_column=api.NON_TEMPORAL),
        orders,
        api.TableSpec("unrelated", None, {"opaque": api.DType("bytes")}, time_column=api.NON_TEMPORAL),
    ), relations=original.relations)
    layer = api.GraphLayer((api.GraphSample("alias", direction="forward", fanout=2,
                                            strategy="recent"),))
    sample = api.SampleSpec(entity=api.EntitySpec("users"), neighborhoods={
        "history": api.NeighborhoodSpec((layer,))})
    view = _schema_view(source, sample, {"users": [], "orders": []})
    assert set(view.tables) == {"users", "orders"}
    assert set(view.tables["users"]["columns"]) == {"user_id"}
    assert set(view.tables["orders"]["columns"]) == {"order_id", "user_id", "at"}
    with pytest.raises(ValueError, match="FM cannot encode users.opaque"):
        _schema_view(source, sample, {"users": ["opaque"], "orders": []})


@pytest.mark.parametrize("values,classes,expected", [
    ([False, True, False], {0: False, 1: True}, ["False", "True", "False"]),
    ([0, 1, 0], {False: "no", True: "yes"}, ["no", "yes", "no"]),
    ([False, True], {np.bool_(False): "no", np.bool_(True): "yes"}, ["no", "yes"]),
    ([0, 1], {np.bool_(False): "no", np.bool_(True): "yes"}, ["no", "yes"]),
    ([0, 1], {0j: "no", 1+0j: "yes"}, ["no", "yes"]),
    ([9007199254740992.0], {np.int64(9007199254740992): "exact", -1: "other"}, ["exact"]),
    ([-9007199254740992.0], {np.int64(-9007199254740992): "exact", 1: "other"}, ["exact"]),
    ([0.0, 1.0], {False: 10, True: 20}, ["10", "20"]),
    ([9007199254740993, -9007199254740993],
     {9007199254740993: "large", -9007199254740993: "negative"}, ["large", "negative"]),
])
def test_class_mapping_preserves_numeric_equality_and_python_class_names(values, classes, expected):
    assert _class_labels(pa.array(values), classes).to_pylist() == expected


@pytest.mark.parametrize("values,classes", [
    ([False], {"false": "no", "true": "yes"}),
    ([1.25], {1: "one", 2: "two"}),
    ([9007199254740993], {float(9007199254740992): "rounded", -1: "other"}),
    ([9007199254740992.0], {np.int64(9007199254740993): "wrong", -1: "other"}),
    ([-9007199254740992.0], {np.int64(-9007199254740993): "wrong", 1: "other"}),
])
def test_class_mapping_rejects_values_outside_exact_declared_classes(values, classes):
    with pytest.raises(ValueError, match="outside declared classes"):
        _class_labels(pa.array(values), classes)
