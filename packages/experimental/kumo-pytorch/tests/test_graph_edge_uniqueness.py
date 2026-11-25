# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G-GRAPH-001 rule 3: an edge appears at most once per sample."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api

USERS = [{"user_id": 1, "at": 0}, {"user_id": 2, "at": 0}]
EVENTS = [
    {"event_id": 10, "user_id": 1, "at": 5},
    {"event_id": 11, "user_id": 1, "at": 7},
    {"event_id": 20, "user_id": 2, "at": 6},
]


def _contract(first_fanout=4, topology="relations") -> api.DataLoaderSpec:
    def hop(direction: str, fanout=4) -> api.GraphLayer:
        return api.GraphLayer(
            (api.GraphSample("user_events", fanout, direction=direction, strategy="recent"),)
        )

    timestamp = api.DType("timestamp", unit="s", timezone="UTC")
    return api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "users",
                    "user_id",
                    {"user_id": api.DType("int64"), "at": timestamp},
                    time_column="at",
                ),
                api.TableSpec(
                    "events",
                    "event_id",
                    {
                        "event_id": api.DType("int64"),
                        "user_id": api.DType("int64"),
                        "at": timestamp,
                    },
                    time_column="at",
                ),
            ),
            (api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
        ),
        # Layers 2 and 3 reach every layer-1 edge again, once from each end.
        api.SampleSpec(
            api.EntitySpec("users"),
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (hop("forward", first_fanout), hop("reverse"), hop("forward"))
                )
            },
        ),
        api.BatchSpec(
            {
                "graph": api.Input(
                    api.NeighborhoodSource("history"), api.Graph(topology=topology), None
                )
            }
        ),
    )


@pytest.mark.parametrize("engine_name", ["in_memory", "diskgraph"])
def test_revisited_edges_appear_once_per_sample(engine_name):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    contract = api.DataLoaderSpec.from_json(_contract().to_json())
    # Two samples of user 1 share the stored edges; each keeps its own copy.
    requests = [
        api.SampleRequest("early", 1, 6),
        api.SampleRequest("late", 1, 100),
        api.SampleRequest("other", 2, 100),
    ]
    expected = {
        "early": [(1, 10)],
        "late": [(1, 10), (1, 11)],
        "other": [(2, 20)],
    }
    with (
        bound_tables(contract, {"users": USERS, "events": EVENTS}, engine_name) as (
            dataset,
            engine,
        ),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(requests) as task,
    ):
        batches = list(
            task.request_batches(
                requests,
                batch_size=len(requests),
                output_format="model" if engine_name == "diskgraph" else "inspectable",
            )
        )
    assert len(batches) == 1
    batch = batches[0]
    assert list(batch.metadata["sample_ids"]) == [request.sample_id for request in requests]
    graph = batch.inputs["graph"]
    users, events = graph["nodes"]["users"], graph["nodes"]["events"]
    edge = graph["edges"]["user_events"]
    assert (edge["source_type"], edge["destination_type"]) == ("users", "events")
    pairs = [tuple(pair) for pair in edge["index"].t().tolist()]
    assert len(pairs) == len(set(pairs))
    actual: dict[str, list[tuple[int, int]]] = {request.sample_id: [] for request in requests}
    for source, destination in pairs:
        owner = users["batch"][source].item()
        assert events["batch"][destination].item() == owner
        actual[requests[owner].sample_id].append(
            (users["ids"][source].item(), events["ids"][destination].item())
        )
    assert {key: sorted(value) for key, value in actual.items()} == expected
    owners = Counter(users["batch"].tolist())
    assert owners == Counter({0: 1, 1: 1, 2: 1})


@pytest.mark.parametrize(
    "options",
    [{}, {"revisit_nodes": False}, {"revisit_nodes": True}],
    ids=["default", "disabled", "enabled"],
)
@pytest.mark.parametrize("topology", ["relations", "sampled"])
@pytest.mark.parametrize("from_snapshot", [False, True], ids=["dataset", "snapshot"])
def test_native_revisit_with_larger_fanout(tmp_path, options, topology, from_snapshot):
    contract = _contract(first_fanout=2, topology=topology)
    tables = {"users": USERS, "events": [*EVENTS, {"event_id": 12, "user_id": 1, "at": 9}]}
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"id": [1], "at": [100]}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "id", "at")
    with bound_tables(contract, tables, "diskgraph") as (dataset, builder):
        if from_snapshot:
            with api.prepare(contract, dataset, builder):
                pass
            snapshot = SimpleNamespace(index_path=builder.index, id="test@1", graph_config={})
            engine = api.DiskGraphEngine.from_snapshot(snapshot, num_threads=1, **options)
            dataset = None
        else:
            engine = api.DiskGraphEngine(num_threads=1, **options)
        expected = [10, 11, 12] if options.get("revisit_nodes") else [11, 12]
        with api.prepare(contract, dataset, engine) as plan:
            with (
                plan.materialize(anchors) as task,
                patch.object(
                    task._assembler, "batches", side_effect=AssertionError("Python fallback")
                ),
            ):
                graph = next(task.batches(batch_size=1, output_format="model")).inputs["graph"]
            assert sorted(graph["nodes"]["events"]["ids"].tolist()) == expected
            if topology == "sampled":
                # Two second-hop edges reach the same user. Re-expand it only once.
                edges_per_hop = [
                    sum(counts) for counts in zip(*graph["num_sampled_edges"].values(), strict=True)
                ]
                assert edges_per_hop == [2, 2, 3 if options.get("revisit_nodes") else 0]
            else:
                requests = [api.SampleRequest("late", 1, 100)]
                with plan.materialize(requests) as task:
                    direct = next(
                        task.request_batches(requests, batch_size=1, output_format="model")
                    )
                for result in (graph, direct.inputs["graph"]):
                    src, dst = result["edges"]["user_events"]["index"]
                    assert result["nodes"]["users"]["ids"][src].tolist() == [1] * len(expected)
                    assert sorted(result["nodes"]["events"]["ids"][dst].tolist()) == expected


@pytest.mark.parametrize("trained_revisit", [False, True], ids=["default", "revisit"])
def test_graph_fitted_state_reuse_checks_revisit_policy(tmp_path, trained_revisit):
    contract = _contract(first_fanout=1)
    requests = [api.SampleRequest("late", 1, 100), api.SampleRequest("early", 1, 6)]
    tables = {"users": USERS, "events": EVENTS}
    artifact_path = tmp_path / "artifact"

    with bound_tables(contract, tables, "diskgraph") as (dataset, _):
        options = {"revisit_nodes": True} if trained_revisit else {}
        with (
            api.prepare(contract, dataset, api.DiskGraphEngine(**options)) as plan,
            plan.materialize(requests) as task,
        ):
            task.to_artifact().save(artifact_path)

        artifact = api.Artifact.load(artifact_path)
        fitted = artifact.fitted_state
        assert fitted["revisit_nodes"] is trained_revisit
        engine = api.DiskGraphEngine(revisit_nodes=trained_revisit)
        with (
            api.prepare(artifact.contract, dataset, engine, inference=True) as plan,
            plan.materialize(requests, fitted=fitted) as task,
        ):
            scored = next(task.request_batches(requests, batch_size=2, output_format="model"))
        (served,) = api.ServingAssembler(artifact).assemble(tables, requests)
        expected = [(0, 10), (0, 11), (1, 10)] if trained_revisit else [(0, 11), (1, 10)]
        for batch in (scored, served):
            graph = batch.inputs["graph"]
            users, events = graph["nodes"]["users"], graph["nodes"]["events"]
            src, dst = graph["edges"]["user_events"]["index"]
            assert users["batch"][src].tolist() == events["batch"][dst].tolist()
            assert (
                sorted(zip(events["batch"][dst].tolist(), events["ids"][dst].tolist())) == expected
            )

        engine = api.DiskGraphEngine(revisit_nodes=not trained_revisit)
        with api.prepare(artifact.contract, dataset, engine) as plan:
            with plan.materialize(requests) as task:
                assert task.fitted_state.fingerprint != fitted.fingerprint
            with pytest.raises(ValueError, match="revisit_nodes"):
                plan.materialize(requests, fitted=fitted)
            with pytest.raises(ValueError, match="revisit_nodes"):
                plan.materialize(requests, fitted=fitted.inputs, inference=True)


def test_graph_fitted_state_without_revisit_policy_keeps_legacy_fingerprint():
    from kumo.pytorch.artifact import _restored

    contract = _contract(first_fanout=1)
    requests = [api.SampleRequest("late", 1, 100)]
    with bound_tables(contract, {"users": USERS, "events": EVENTS}, "diskgraph") as (dataset, _):
        with api.prepare(contract, dataset, api.DiskGraphEngine()) as plan:
            with plan.materialize(requests) as task:
                serialized = task.fitted_state.to_dict()
            del serialized["revisit_nodes"]
            # The state a pre-policy artifact restores, with its original weight fingerprint.
            legacy, _, _ = _restored(plan.contract, serialized, {})
            with plan.materialize(requests, fitted=legacy, inference=True) as task:
                assert task.fitted_state.to_dict() == serialized
                assert task.fitted_state.fingerprint == legacy.fingerprint
        engine = api.DiskGraphEngine(revisit_nodes=True)
        with (
            api.prepare(contract, dataset, engine) as plan,
            pytest.raises(ValueError, match="revisit_nodes"),
        ):
            plan.materialize(requests, fitted=legacy, inference=True)


@pytest.mark.parametrize(("hops", "recorded_policy"), [(1, None), (2, True)])
def test_self_relation_records_policy_only_when_it_can_reexpand_a_node(hops, recorded_policy):
    base = _contract()
    schema = api.DataSchema(
        base.schema.tables, (api.RelationSpec("parent", "events", "event_id", "events", "user_id"),)
    )
    layer = api.GraphLayer((api.GraphSample("parent", 2, direction="forward", strategy="recent"),))
    sample = api.SampleSpec(
        api.EntitySpec("events"), neighborhoods={"history": api.NeighborhoodSpec((layer,) * hops)}
    )
    contract = api.DataLoaderSpec(schema, sample, base.batch)
    tables = {"users": USERS, "events": [{"event_id": 1, "user_id": 1, "at": 5}]}
    with (
        bound_tables(contract, tables, "in_memory") as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize([api.SampleRequest("root", 1, 100)]) as task,
    ):
        assert task.fitted_state.get("revisit_nodes") is recorded_policy
