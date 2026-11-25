# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained payloads over an Arrow-to-Parquet fixture, with explicit expected rows."""
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _fixture import LARGE, build_index, when
from test_native_json import assert_payload_matches_tables

from kumo import fm
from kumo.pytorch import api


def test_snapshot_batch_survives_close_without_resampling(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        targets={"next": api.Aggregate(api.Rows(
            path=("order_user",), time=api.TimeWindow("at", start=0, stop=7 * 86400)), "count")},
        neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample(
            "order_user", direction="reverse", strategy="recent", fanout=1),)),))})
    with fm.prepare(engine=engine, sample=sample, target="next", task="regression",
                    features={"users": [], "orders": ["at"]}) as loader:
        with loader.materialize(context=fm.Context(pairs=[(LARGE, when(10), 10), (-5, when(10), 11)]),
                                query=fm.Query(pairs=[(LARGE, when(25), 20), (-5, when(25), 21)])) as request:
            batch = request.load()
            arrays, ipc = batch.encode("arrays"), batch.encode("arrow_ipc")
        with pytest.raises(RuntimeError, match="closed"):
            request.load()
    # Encoding after loader closure must use retained data; compare both formats to the row oracle too.
    assert assert_payload_matches_tables(batch) == (arrays, ipc)
    assert batch.tables("context")["instance_table"]["__fm_target"].to_pylist() == [1.0, 0.0]
    assert batch.tables("context")["related_tables"]["orders"]["order_id"].to_pylist() == [12, 13]
    assert batch.tables("predict")["related_tables"]["orders"]["order_id"].to_pylist() == [14, 13]
    assert "__fm_target" not in batch.tables("predict")["instance_table"].column_names
    with pytest.raises(RuntimeError, match="closed"):
        loader.load(query=fm.Query(pairs=[(LARGE, when(25))]))


@pytest.mark.parametrize("operation", ["mean", "count"])
def test_context_targets_report_null_occurrences_without_dropping_rows(tmp_path, operation):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    target = api.Aggregate(api.Rows(path=("order_user",),
                                    time=api.TimeWindow("at", start=0, stop=7 * 86400)),
                           operation, "order_id" if operation == "mean" else None)
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"future": target},
                            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                                api.GraphSample("order_user", direction="reverse", strategy="recent", fanout=1),
                            )),))})
    context = fm.Context(pairs=[(-5, when(10), 202), (LARGE, when(10), 101)])
    query = fm.Query(pairs=[(LARGE, when(25), 909)])
    with fm.prepare(engine=engine, sample=sample, target="future", task="regression",
                    features={"users": [], "orders": []}, batch_size=1) as loader:
        if operation == "mean":
            with pytest.raises(ValueError) as error:
                loader.load(context=context, query=query)
            assert str(error.value) == (
                "FM context target 'future' is null for occurrence IDs [202]; "
                "context targets must be non-null for every occurrence")
        else:
            batch = loader.load(context=context, query=query)
            instance = batch.tables("context")["instance_table"]
            assert instance["instance_id"].to_pylist() == [202, 101]
            assert instance["__fm_target"].to_pylist() == [0.0, 1.0]


def test_binary_labels_stay_aligned_and_map_numeric_keys_across_batches(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    target = api.Compare(api.Aggregate(api.Rows(path=("order_user",),
                                                time=api.TimeWindow("at", start=0, stop=7 * 86400)),
                                       "count"), "greater_than", 0)
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                                api.GraphSample("order_user", direction="reverse", strategy="recent", fanout=1),
                            )),))})
    ids = [-9007199254740993, 9007199254740993, -9]
    context = fm.Context(pairs=[(-5, when(10), ids[0]), (LARGE, when(10), ids[1]), (-5, when(10), ids[2])])
    query = fm.Query(pairs=[(LARGE, when(25), 0)])
    with fm.prepare(engine=engine, sample=sample, target="next", task="binary_classification",
                    classes={0: False, 1: True}, features={"users": [], "orders": []},
                    batch_size=1) as loader:
        batch = loader.load(context=context, query=query)
    instance = batch.tables("context")["instance_table"]
    assert instance["instance_id"].to_pylist() == ids
    assert instance["__fm_target"].to_pylist() == ["False", "True", "False"]
    orders = batch.tables("context")["related_tables"]["orders"]
    assert orders["instance_id"].to_pylist() == ids
    assert orders["order_id"].to_pylist() == [13, 12, 13]


@pytest.fixture
def fm_inputs(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="fixture"))
    target = api.Aggregate(api.Rows(path=("order_user",),
                                    time=api.TimeWindow("at", start=0, stop=7 * 86400)), "count")
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                                api.GraphSample("order_user", direction="reverse", strategy="recent", fanout=1),
                            )),))})
    return {"engine": engine, "sample": sample, "target": "next", "task": "regression",
            "features": {"users": [], "orders": []}}


@pytest.mark.parametrize("task", ["classification", "multiclass_classification", "unknown"])
def test_prepare_refuses_ambiguous_or_unsupported_task_names(fm_inputs, task):
    with pytest.raises(ValueError, match="task must be 'regression' or 'binary_classification'"):
        fm.prepare(**{**fm_inputs, "task": task})


@pytest.mark.parametrize("budgets,match", [
    ({"max_payload_byte": 1000}, "Unknown FM budget keys.*max_payload_byte"),
    ({"max_instances": 0}, "max_instances must be a positive integer"),
    ({"max_payload_bytes": -1}, "max_payload_bytes must be a positive integer"),
    ({"max_columns": True}, "max_columns must be a positive integer"),
    ({"max_rows_per_table": 1.5}, "max_rows_per_table must be a positive integer"),
    ({"max_candidate_rows": "10"}, "max_candidate_rows must be a positive integer"),
    ({"max_decoded_arrow_bytes": None}, "max_decoded_arrow_bytes must be a positive integer"),
])
def test_prepare_refuses_unknown_or_invalid_budget_values(fm_inputs, budgets, match):
    with pytest.raises(ValueError, match=match):
        fm.prepare(**fm_inputs, budgets=budgets)


def test_prepare_is_the_public_factory_and_captures_budget_values(fm_inputs):
    assert not hasattr(fm, "FMLoader")
    limits = {"max_instances": 1}
    with fm.prepare(**fm_inputs, budgets=limits) as loader:
        assert not hasattr(loader, "workload")
        assert not hasattr(loader, "preparation_mode")
        limits["max_instances"] = 2
        with pytest.raises(ValueError, match="instance budget exceeded"):
            loader.load(query=fm.Query(pairs=[(LARGE, when(25)), (-5, when(25))]))
        with loader.materialize(query=fm.Query(pairs=[(LARGE, when(25))])) as request:
            assert isinstance(request, fm.PreparedRequest)
            batch = request.load()
    assert batch.tables("predict")["instance_table"]["__fm_entity_key"].to_pylist() == [LARGE]


@pytest.mark.parametrize("split", ["context", "predict"])
def test_materialize_enforces_configured_population_budget(fm_inputs, split):
    population = (fm.Context(entities=range(2), times=[None, None]) if split == "context"
                  else fm.Query(pairs=[(1, None)] * 3))
    with fm.prepare(**fm_inputs, budgets={"max_instances": 2}) as loader:
        with pytest.raises(ValueError, match="instance budget"):
            loader.materialize(**{"context" if split == "context" else "query": population})


def test_prepare_binary_classification_refuses_three_class_map(fm_inputs):
    with pytest.raises(ValueError, match="Binary classification requires a two-entry"):
        fm.prepare(**{**fm_inputs, "task": "binary_classification"},
                   classes={0: "none", 1: "one", 2: "many"})


@pytest.mark.parametrize("parent_is_source", [True, False])
def test_self_relation_payload_retains_selected_parents_and_isolated_occurrences(tmp_path, parent_is_source):
    rows = pa.table({"id": [10, 20, 30, 99],
                     "manager_id": pa.array([None, 10, 10, None], pa.int64()),
                     "at": [when(1), when(2), when(3), when(1)]})
    path = tmp_path / "employees.parquet"
    pq.write_table(rows, path)
    left, right = ("id", "manager_id") if parent_is_source else ("manager_id", "id")
    schema = api.DataSchema(tables=(api.TableSpec(
        "employees", "id", {field.name: api.DType.from_arrow(field.type) for field in rows.schema},
        time_column="at"),), relations=(
            api.RelationSpec("manager", "employees", left, "employees", right),))
    layer = api.GraphLayer((api.GraphSample("manager", fanout=2, strategy="recent",
                                            direction="forward" if parent_is_source else "reverse"),))
    sample = api.SampleSpec(entity=api.EntitySpec("employees"),
                            targets={"next": api.Aggregate(api.Rows(
                                path=("manager",), time=api.TimeWindow("at", start=0, stop=86400)), "count")},
                            neighborhoods={"history": api.NeighborhoodSpec((layer,))})
    source_contract = api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs={
        "graph": api.Input(api.NeighborhoodSource("history"), api.Graph(node_fields={
            "employees": api.Fields(tuple(api.Encode(name, api.Raw()) for name in rows.column_names)),
        }), None)}))
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=source_contract, dataset=api.Dataset({"employees": api.Parquet(str(path))}),
                     engine=engine):
        pass
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=engine.index, id="self-relation"))
    ids = [9007199254740993, -7, -9007199254740993]
    with fm.prepare(engine=engine, sample=sample, target="next", task="regression",
                    features={"employees": ["manager_id", "at"]}, batch_size=2) as loader:
        first = loader.load(query=fm.Query(pairs=[(10, when(10), ids[0]), (99, when(10), ids[1]),
                                                  (10, when(10), ids[2])]))
        # Mutating exported metadata must not poison this loader's cached schema.
        exposed = first.as_arrow_request()
        exposed["schema"]["related_tables"]["employees"]["columns"].clear()
        second = loader.load(query=fm.Query(pairs=[(99, when(10), 8)]))
        with pytest.raises(ValueError, match="Task entity is not eligible at its anchor"):
            loader.load(query=fm.Query(pairs=[(20, when(1), 9)]))
    related = first.tables("predict")["related_tables"]["employees"]
    assert related["instance_id"].to_pylist() == [ids[0]] * 3 + [ids[1]] + [ids[2]] * 3
    assert related["id"].to_pylist() == [10, 20, 30, 99, 10, 20, 30]
    assert related["__fm_edge_manager"].to_pylist() == [None, 10, 10, None, None, 10, 10]
    assert related["at"].type == pa.timestamp("us", tz="UTC")
    assert second.tables("predict")["related_tables"]["employees"]["id"].to_pylist() == [99]
    assert list(second.metadata["schema"]["related_tables"]["employees"]["columns"]) == [
        "instance_id", "id", "manager_id", "at", "__fm_edge_manager"]
    arrays = json.loads(first.encode("arrays"))
    assert arrays["predict"]["related_tables"]["employees"]["rows"][1][:3] == [str(ids[0]), 20, 10]
    ipc = json.loads(second.encode("arrow_ipc"))["predict"]["related_tables"]["employees"]
    assert pa.ipc.open_stream(base64.b64decode(ipc["payload_base64"])).read_all().equals(
        second.tables("predict")["related_tables"]["employees"])


def test_sampled_candidate_budget_is_enforced_before_payload_assembly(fm_inputs):
    with fm.prepare(**fm_inputs, budgets={"max_candidate_rows": 2}) as loader:
        with pytest.raises(ValueError, match="sampled node-plus-edge budget exceeded"):
            loader.load(query=fm.Query(pairs=[(LARGE, when(25))]))


def test_internal_related_row_budget_is_enforced_before_payload_assembly(fm_inputs):
    with fm.prepare(**fm_inputs, budgets={"max_rows_per_table": 1}) as loader:
        with pytest.raises(ValueError, match="Expanded related table exceeds row budget"):
            loader.load(query=fm.Query(pairs=[(LARGE, when(25)), (-5, when(25))]))


def test_default_graph_strategy_matches_explicit_recent_for_timed_destinations(fm_inputs):
    original = fm_inputs["sample"]
    graph_name, neighborhood = next(iter(original.neighborhoods.items()))
    step = neighborhood.layers[0].samples[0]
    default = replace(original, neighborhoods={graph_name: api.NeighborhoodSpec((
        api.GraphLayer((replace(step, strategy=None),)),))})
    query = fm.Query(pairs=[(LARGE, when(25), -9007199254740993), (-5, when(25), 9007199254740993)])
    batches = []
    for sample in (original, default):
        with fm.prepare(**{**fm_inputs, "sample": sample}) as loader:
            batches.append(loader.load(query=query))
    for name, table in batches[0].tables("predict")["related_tables"].items():
        assert batches[1].tables("predict")["related_tables"][name].equals(table)
    assert batches[0].encode("arrays") == batches[1].encode("arrays")
    assert batches[1].tables("predict")["related_tables"]["orders"]["order_id"].to_pylist() == [14, 13]


@pytest.mark.parametrize("strategy", [None, "uniform", "recent"])
def test_fm_rejects_timeless_destinations_after_default_strategy_resolution(fm_inputs, strategy):
    target = api.Aggregate(api.Rows(path=(), time=api.TimeWindow("at", start=0, stop=86400)), "count")
    layer = api.GraphLayer((api.GraphSample("order_user", direction="forward", fanout=1,
                                            strategy=strategy),))
    sample = api.SampleSpec(entity=api.EntitySpec("orders"), targets={"next": target},
                            neighborhoods={"history": api.NeighborhoodSpec((layer,))})
    with pytest.raises(ValueError, match="timeless table users"):
        fm.prepare(**{**fm_inputs, "sample": sample})


def test_resolved_uniform_strategy_is_refused_for_timed_destinations(fm_inputs):
    original = fm_inputs["sample"]
    name, neighborhood = next(iter(original.neighborhoods.items()))
    step = replace(neighborhood.layers[0].samples[0], strategy="uniform")
    sample = replace(original, neighborhoods={name: api.NeighborhoodSpec((api.GraphLayer((step,)),))})
    with pytest.raises(ValueError, match="bounded recent graph sampling only"):
        fm.prepare(**{**fm_inputs, "sample": sample})


def test_default_strategy_preserves_diskgraph_clock_override_refusal(fm_inputs):
    original = fm_inputs["sample"]
    name, neighborhood = next(iter(original.neighborhoods.items()))
    step = replace(neighborhood.layers[0].samples[0], strategy=None, time_column="order_id")
    sample = replace(original, neighborhoods={name: api.NeighborhoodSpec((api.GraphLayer((step,)),))})
    with pytest.raises(api.ValidationError) as error:
        fm.prepare(**{**fm_inputs, "sample": sample})
    assert any(issue.code == "DG-GRAPH-CLOCK-OVERRIDE" for issue in error.value.issues)


@pytest.fixture
def collision_inputs(tmp_path, request):
    primary_key = getattr(request, "param", "event_id")
    events = {"__fm_edge_activity": [700], "__fm_edge_unselected": [800],
              primary_key: [11], "user": [7], "at": [when(1)]}
    rows = {"users": pa.table({"id": [7], "__fm_edge_activity": [900]}), "events": pa.table(events)}
    tables, sources = [], {}
    for name, table in rows.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(table, path)
        sources[name] = api.Parquet(str(path))
        tables.append(api.TableSpec(name, "id" if name == "users" else primary_key,
                                    {field.name: api.DType.from_arrow(field.type) for field in table.schema},
                                    time_column=api.NON_TEMPORAL if name == "users" else "at"))
    schema = api.DataSchema(tuple(tables), (api.RelationSpec("activity", "events", "user", "users", "id"),))
    layer = api.GraphLayer((api.GraphSample("activity", fanout=1, direction="reverse"),))
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"next": api.Aggregate(
        api.Rows(path=("activity",), time=api.TimeWindow("at", start=0, stop=86400)), "count")},
        neighborhoods={"history": api.NeighborhoodSpec((layer,))})
    contract = api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs={"graph": api.Input(
        api.NeighborhoodSource("history"), api.Graph(node_fields={
            name: api.Fields(tuple(api.Encode(column, api.Raw()) for column in table.column_names))
            for name, table in rows.items()}), None)}))
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=contract, dataset=api.Dataset(sources), engine=engine):
        pass
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=engine.index, id="collision"))
    return {"engine": engine, "sample": sample, "target": "next", "task": "regression",
            "features": {"users": [], "events": []}}


def test_projected_source_feature_cannot_replace_its_generated_relationship_column(collision_inputs):
    with pytest.raises(ValueError, match="collide with generated relationship columns.*events.__fm_edge_activity"):
        fm.prepare(**{**collision_inputs, "features": {"users": [], "events": ["__fm_edge_activity"]}})


@pytest.mark.parametrize("collision_inputs", ["__fm_edge_activity"], indirect=True)
def test_retained_primary_key_cannot_replace_its_generated_relationship_column(collision_inputs):
    with pytest.raises(ValueError, match="collide with generated relationship columns.*events.__fm_edge_activity"):
        fm.prepare(**collision_inputs)


def test_same_named_feature_on_other_table_and_unprojected_collision_are_preserved(collision_inputs):
    features = {"users": ["__fm_edge_activity"], "events": ["__fm_edge_unselected"]}
    with fm.prepare(**{**collision_inputs, "features": features}) as loader:
        batch = loader.load(query=fm.Query(pairs=[(7, when(10), 99)]))
    related = batch.tables("predict")["related_tables"]
    assert related["users"]["__fm_edge_activity"].to_pylist() == [900]
    assert related["events"]["__fm_edge_activity"].to_pylist() == [7]
    assert related["events"]["__fm_edge_unselected"].to_pylist() == [800]
    assert_payload_matches_tables(batch)


@pytest.mark.parametrize("collision_inputs", ["__fm_edge_unselected"], indirect=True)
def test_primary_key_with_unrelated_generated_looking_name_remains_valid(collision_inputs):
    with fm.prepare(**collision_inputs) as loader:
        batch = loader.load(query=fm.Query(pairs=[(7, when(10), 99)]))
    table = batch.tables("predict")["related_tables"]["events"]
    assert table["__fm_edge_unselected"].to_pylist() == [11]
    assert table["__fm_edge_activity"].to_pylist() == [7]


@pytest.mark.parametrize("batch_size", [1, 8])
def test_candidate_budget_spans_internal_batches_and_resets_after_failure(fm_inputs, batch_size):
    engine = fm_inputs["engine"]
    task_root = engine.index.parent.parent / "tasks" / engine.index_build_id
    with fm.prepare(**fm_inputs, batch_size=batch_size, budgets={"max_candidate_rows": 5}) as loader:
        with pytest.raises(ValueError, match="node-plus-edge budget exceeded for the load request"):
            loader.load(query=fm.Query(pairs=[(LARGE, when(25), 1), (-5, when(25), 2)]))
        assert not task_root.exists() or not list(task_root.iterdir())
        # A failed load must not consume a later call's three-row allowance.
        batch = loader.load(query=fm.Query(pairs=[(LARGE, when(25), 3)]))
        assert batch.tables("predict")["related_tables"]["orders"]["order_id"].to_pylist() == [14]
        with loader.materialize(query=fm.Query(pairs=[(-5, when(25), 4)])) as request:
            for _ in range(2):
                repeated = request.load()
                assert repeated.tables("predict")["related_tables"]["orders"]["order_id"].to_pylist() == [13]
    assert not task_root.exists() or not list(task_root.iterdir())


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("limit", [5, 6])
def test_candidate_budget_sums_context_and_query_and_resets_each_load(fm_inputs, batch_size, limit):
    engine = fm_inputs["engine"]
    task_root = engine.index.parent.parent / "tasks" / engine.index_build_id
    context = fm.Context(pairs=[(LARGE, when(10), -9007199254740993)])
    query = fm.Query(pairs=[(LARGE, when(25), 9007199254740993)])
    with fm.prepare(**fm_inputs, batch_size=batch_size, budgets={"max_candidate_rows": limit}) as loader:
        if limit == 5:
            with pytest.raises(ValueError, match="node-plus-edge budget exceeded for the load request"):
                loader.load(context=context, query=query)
        else:
            for _ in range(2):
                batch = loader.load(context=context, query=query)
                assert batch.tables("context")["instance_table"]["__fm_target"].to_pylist() == [1.0]
                assert batch.tables("predict")["instance_table"]["instance_id"].to_pylist() == [9007199254740993]

    assert not task_root.exists() or not list(task_root.iterdir())


@pytest.mark.parametrize("limit", [6, 7])
def test_candidate_budget_counts_repeated_selected_edges_before_assembly(tmp_path, limit):
    rows = pa.table({"id": [10, 20, 30], "manager_id": pa.array([None, 10, 10], pa.int64()),
                     "at": [when(1), when(2), when(3)]})
    path = tmp_path / "employees.parquet"
    pq.write_table(rows, path)
    schema = api.DataSchema(tables=(api.TableSpec(
        "employees", "id", {field.name: api.DType.from_arrow(field.type) for field in rows.schema},
        time_column="at"),), relations=(
            api.RelationSpec("manager", "employees", "id", "employees", "manager_id"),))
    sample = api.SampleSpec(entity=api.EntitySpec("employees"), targets={"next": api.Aggregate(
        api.Rows(path=("manager",), time=api.TimeWindow("at", start=0, stop=86400)), "count")},
        neighborhoods={"history": api.NeighborhoodSpec((
            api.GraphLayer((api.GraphSample("manager", direction="forward", fanout=2),)),
            api.GraphLayer((api.GraphSample("manager", direction="reverse", fanout=1),)),
        ))})
    graph_sample = replace(sample, targets={})
    contract = api.DataLoaderSpec(schema, graph_sample, api.BatchSpec(inputs={"graph": api.Input(
        api.NeighborhoodSource("history"), api.Graph(node_fields={"employees": api.Fields(tuple(
            api.Encode(column, api.Raw()) for column in rows.column_names))}), None)}))
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=contract, dataset=api.Dataset({"employees": api.Parquet(str(path))}),
                     engine=engine) as plan:
        anchors = api.AnchorTable(api.InMemoryRows([{"id": 10, "at": when(10)}]), "id", "at")
        with plan.materialize(anchors) as task:
            graph = next(task.batches(output_format="columnar", batch_size=1)).inputs["graph"]
            assert len(graph.nodes["employees"].ids) == 3
            assert len(graph.edges["manager"].index) == 4  # Two pairs selected in both directions.
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=engine.index, id="repeated-edges"))
    with fm.prepare(engine=engine, sample=sample, target="next", task="regression",
                    features={"employees": []}, budgets={"max_candidate_rows": limit}) as loader:
        if limit == 6:
            with pytest.raises(ValueError, match="node-plus-edge budget exceeded for the load request"):
                loader.load(query=fm.Query(pairs=[(10, when(10))]))
        else:
            batch = loader.load(query=fm.Query(pairs=[(10, when(10))]))
            assert batch.tables("predict")["related_tables"]["employees"]["id"].to_pylist() == [10, 20, 30]
