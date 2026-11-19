"""Snapshot input validation before any native graph is sampled."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kumo.pytorch import api
from kumo.fm import Context, Query
from kumo.fm._snapshot import (SnapshotFMLoader, _rows, _schema_view,
                               _selected_graph, _target_horizon)


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
                                          "foreign_key": "user_id"}
    target = api.Aggregate(api.Rows(path=("alias",),
        time=api.TimeWindow("at", start=0, stop=30)), "count")
    assert _target_horizon(target) == 30
    sample = api.SampleSpec(entity=api.EntitySpec("users"),
        targets={"next": target}, neighborhoods={"history": api.NeighborhoodSpec((
            api.GraphLayer((api.GraphSample("alias", direction="reverse",
                                            fanout=2, strategy="recent"),)),))})
    name, fields, tables, relations, projections = _selected_graph(
        schema(), sample, {"users": [], "orders": ["amount"]}, bundle)
    assert (name, tables, relations) == ("history", {"users", "orders"}, {"alias"})
    assert projections == {"orders": ["order_id", "amount"], "users": ["user_id"]}
    assert {item.field for item in fields["orders"].fields} == {
        "order_id", "amount", "at", "user_id"}
    with pytest.raises(ValueError, match="uniformly into timeless"):
        _selected_graph(schema(), api.SampleSpec(entity=api.EntitySpec("orders"),
            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                api.GraphSample("alias", direction="forward", fanout=2,
                                strategy="recent"),)),))}),
            {"orders": [], "users": []}, bundle)


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
