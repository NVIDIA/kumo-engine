# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampled graphs retain the directed native buffers and hop accounting."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors
from tests.test_native_anchor_handoff import native_task as native_task


@pytest.fixture
def sampled_plan(tmp_path):
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "people",
                "id",
                {
                    "id": api.DType("int64"),
                    "created": api.DType("timestamp", unit="s", timezone="UTC"),
                    "value": api.DType("float64"),
                    "manager": api.DType("int64"),
                },
                time_column="created",
            ),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": api.DType("int64"),
                    "owner": api.DType("int64"),
                    "time": api.DType("timestamp", unit="s", timezone="UTC"),
                    "value": api.DType("float64"),
                },
                time_column="time",
            ),
        ),
        relations=(
            api.RelationSpec("owner", "events", "owner", "people", "id"),
            api.RelationSpec("manager", "people", "manager", "people", "id"),
        ),
    )
    timestamp = pa.timestamp("s", tz="UTC")
    paths = {name: tmp_path / f"{name}.parquet" for name in ("people", "events", "anchors")}
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "created": pa.array([0, 0, 0], type=timestamp),
                "value": [1.0, 2.0, 3.0],
                "manager": [2, 3, 3],
            }
        ),
        paths["people"],
    )
    pq.write_table(
        pa.table(
            {
                "id": [10, 11, 12],
                "owner": [1, 1, 2],
                "time": pa.array([10, 20, 30], type=timestamp),
                "value": [10.0, 11.0, 12.0],
            }
        ),
        paths["events"],
    )
    pq.write_table(
        pa.table(
            {
                "entity": [1, 1, 3, 2],
                "cutoff": pa.array([25, 15, 35, 35], type=timestamp),
                "split": ["train"] * 4,
            }
        ),
        paths["anchors"],
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("people"),
        neighborhoods={
            "g": api.NeighborhoodSpec(
                (
                    api.GraphLayer((api.GraphSample("owner", 2, direction="reverse"),)),
                    api.GraphLayer((api.GraphSample("owner", 1),)),
                )
            )
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("g"),
                    api.Graph(
                        node_fields={
                            name: api.Fields((api.Encode("value", api.Raw()),))
                            for name in ("people", "events")
                        },
                        topology="sampled",
                    ),
                    None,
                )
            },
            batching=api.Batching(max_samples=2),
        ),
    )
    dataset = api.Dataset(
        tables={name: api.Parquet(str(paths[name])) for name in ("people", "events")}
    )
    anchors = api.AnchorTable(api.Parquet(str(paths["anchors"])), "entity", "cutoff", split="split")
    with api.prepare(
        contract, dataset, api.DiskGraphEngine(num_threads=2)
    ) as plan:
        yield plan, anchors


@pytest.mark.parametrize("output_format", ["model", "inspectable"])
def test_sampled_counts_consumed_by_heterognn(sampled_plan, output_format):
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    HeteroGNN = pytest.importorskip("kumo_core.modeling.gnn.hetero").HeteroGNN
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        graph = next(task.batches(split="train", output_format=output_format)).inputs["graph"]
        data = HeteroData()
        for table, node in graph["nodes"].items():
            data[table].x = node["features"]["value"]["values"].float().reshape(-1, 1)
            data[table].num_sampled_nodes = graph["num_sampled_nodes"][table]
        for key, edge in graph["edges"].items():
            # Incoming-message convention: reverse endpoints, keeping native
            # occurrence order and hop counts without reconstructing topology.
            edge_type = (edge["destination_type"], key, edge["source_type"])
            data[edge_type].edge_index = edge["index"].flip(0)
            data[edge_type].num_sampled_edges = graph["num_sampled_edges"][key]
        model = HeteroGNN(1, 2, metadata=data.metadata(), norm=None).eval()
        with torch.no_grad():
            output = model(
                data.x_dict,
                data.edge_index_dict,
                data.num_sampled_nodes_dict,
                data.num_sampled_edges_dict,
            )
        for table, values in output.items():
            assert values.shape == (graph["num_sampled_nodes"][table][0], 1)
            assert torch.isfinite(values).all()


@pytest.mark.parametrize("output_format", ["model", "inspectable", "columnar"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sampled_buffers_match_native_without_reconstruction(
    sampled_plan, monkeypatch, output_format, batch_size
):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        native = []
        iterate = task._anchors._iter

        def capture(*args, **kwargs):
            for batch in iterate(*args, **kwargs):
                native.append(batch)
                yield batch

        def forbidden(*args, **kwargs):
            pytest.fail("sampled output reconstructed or deduplicated native topology")

        monkeypatch.setattr(task._anchors, "_iter", capture)
        monkeypatch.setattr(NativeAnchors, "_requests", forbidden)
        monkeypatch.setattr(plan._source, "_runtime_graph_events", forbidden)
        monkeypatch.setattr(plan._source, "_graph_output", forbidden)
        monkeypatch.setattr("kumo.pytorch.engines.diskgraph._first_occurrences", forbidden)
        actual = list(
            task.batches(split="train", batch_size=batch_size, output_format=output_format)
        )
        assert len(actual) == len(native) == 4 // batch_size
        for result, raw in zip(actual, native, strict=True):
            graph = result.inputs["graph"]
            original = raw["graphs"]["main"]
            edges = graph.edges if output_format == "columnar" else graph["edges"]
            counts = (
                graph.num_sampled_edges
                if output_format == "columnar"
                else graph["num_sampled_edges"]
            )
            nodes = (
                graph.num_sampled_nodes
                if output_format == "columnar"
                else graph["num_sampled_nodes"]
            )
            count_values = tuple if output_format == "columnar" else list
            for table, values in original["num_sampled_nodes_dict"].items():
                if table in {"people", "events"}:
                    assert type(nodes[table]) is count_values
                    assert nodes[table] == count_values(values.to_pylist())
            physical = plan._source._sampling_plan(plan.contract.sample.neighborhoods["g"])[1]
            assert dict(counts) == {
                kind["rel"]: count_values(entry["counts"].to_pylist())
                for entry in original["num_sampled_edges_dict"]
                if (kind := entry["edge_type"])["rel"] in physical
            }
            assert all(type(values) is count_values for values in counts.values())
            selected = {
                entry["edge_type"]["rel"]: entry
                for entry in original["edge_index_dict"]
                if entry["edge_type"]["rel"] in physical
            }
            kinds = {
                entry["edge_type"]["rel"]: entry["edge_type"]
                for entry in (*original["edge_index_dict"], *original["num_sampled_edges_dict"])
                if entry["edge_type"]["rel"] in physical
            }
            assert edges.keys() == kinds.keys()
            for key, kind in kinds.items():
                assert isinstance(key, str)
                edge = edges[key]
                entry = selected.get(key)
                left = [] if entry is None else entry["src"].to_pylist()
                right = [] if entry is None else entry["dst"].to_pylist()
                if output_format == "columnar":
                    assert edge.index["source"].to_pylist() == left
                    assert edge.index["destination"].to_pylist() == right
                    metadata = vars(edge)
                else:
                    assert edge["index"].tolist() == [left, right]
                    metadata = edge
                assert (metadata["source_type"], key, metadata["destination_type"]) == (
                    kind["src"],
                    key,
                    kind["dst"],
                )
                logical = physical[key]
                assert metadata["relation"] == logical
                assert metadata["direction"] == (
                    "forward"
                    if key == plan._source._relation_mapping[logical]["forward"]
                    else "reverse"
                )
                assert sum(counts[key]) == len(left)
            # Both directions survive as separate native relation keys.
            assert any(
                kind["src"] == "people" and kind["dst"] == "events" for kind in kinds.values()
            )
            assert any(
                kind["src"] == "events" and kind["dst"] == "people" for kind in kinds.values()
            )
            if output_format != "columnar":
                assert graph["nodes"]["people"]["batch"].dtype == torch.long


def test_sampled_rejects_requests_and_deduplication(sampled_plan):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        requests = task._anchors.fetch(api.ALL_ANCHORS, [0, 1])
        with pytest.raises(ValueError, match="native AnchorTable"):
            list(task.batches(requests, output_format="model"))
        with pytest.raises(ValueError, match="preserves edge occurrences"):
            list(task.batches(output_format="columnar", deduplicate_edges=True))


def test_topology_validation_and_fingerprint(sampled_plan):
    plan, _ = sampled_plan
    with pytest.raises(ValueError, match="graph topology"):
        api.Graph(topology="unknown")
    graph = plan.contract.batch.inputs["graph"]
    relations = replace(graph, representation=replace(graph.representation, topology="relations"))
    changed = replace(
        plan.contract, batch=replace(plan.contract.batch, inputs={"graph": relations})
    )
    assert changed.fingerprint != plan.contract.fingerprint
    assert api.Graph().topology == "relations"


def test_multiple_sampled_neighborhoods_share_schedule(sampled_plan, monkeypatch):
    plan, anchors = sampled_plan
    contract = plan.contract
    neighborhood = contract.sample.neighborhoods["g"]
    contract = replace(
        contract,
        sample=replace(contract.sample, neighborhoods={"g": neighborhood, "h": neighborhood}),
        batch=replace(
            contract.batch,
            inputs={
                name: replace(contract.batch.inputs["graph"], source=api.NeighborhoodSource(source))
                for name, source in (("graph", "g"), ("other", "h"))
            },
        ),
    )
    with api.prepare(contract, plan.dataset, plan.engine) as bound:
        with bound.materialize(anchors, fit_split="train") as task:

            def forbidden(*args, **kwargs):
                pytest.fail("multiple sampled neighborhoods used per-request sampling")

            monkeypatch.setattr(bound._source, "sample_graph", forbidden)
            monkeypatch.setattr(task._anchors, "_requests", forbidden)
            for batch in task.batches(split="train", output_format="model"):
                left, right = batch.inputs["graph"], batch.inputs["other"]
                assert left["num_sampled_nodes"] == right["num_sampled_nodes"]
                assert left["num_sampled_edges"] == right["num_sampled_edges"]
                for key in left["edges"]:
                    torch.testing.assert_close(
                        left["edges"][key]["index"], right["edges"][key]["index"]
                    )


def test_sampled_sharding_padding_and_resume(sampled_plan):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        seen = []
        for rank in range(3):
            options = dict(
                split="train",
                output_format="model",
                rank=rank,
                world_size=3,
                shuffle=True,
                seed=7,
                epoch=1,
            )
            batches = task.batches(**options)
            first = next(batches)
            state = batches.state_dict()
            resumed = task.batches(**options)
            resumed.load_state_dict(state)
            assert list(resumed) == list(batches) == []
            seen.extend(
                sid
                for sid, pad in zip(
                    first.metadata["sample_ids"], first.metadata["is_padding"], strict=True
                )
                if not pad
            )
            graph = first.inputs["graph"]
            for key, edge in graph["edges"].items():
                assert edge["index"].shape[1] == sum(graph["num_sampled_edges"][key])
        assert sorted(seen) == list(range(4))


def test_sampled_self_relation_keeps_both_native_directions(sampled_plan, monkeypatch):
    plan, anchors = sampled_plan
    neighborhood = api.NeighborhoodSpec(
        (
            api.GraphLayer((api.GraphSample("manager", 1),)),
            api.GraphLayer((api.GraphSample("manager", 2, direction="reverse"),)),
        )
    )
    contract = replace(
        plan.contract,
        sample=replace(plan.contract.sample, neighborhoods={"g": neighborhood}),
        batch=replace(
            plan.contract.batch,
            inputs={
                "graph": replace(
                    plan.contract.batch.inputs["graph"],
                    representation=api.Graph(
                        {
                            "people": plan.contract.batch.inputs[
                                "graph"
                            ].representation.node_fields["people"]
                        },
                        topology="sampled",
                    ),
                )
            },
        ),
    )
    with api.prepare(contract, plan.dataset, plan.engine) as bound:
        with bound.materialize(anchors, fit_split="train") as task:

            def forbidden(*args, **kwargs):
                pytest.fail("sampled self-relationship was deduplicated")

            monkeypatch.setattr("kumo.pytorch.engines.diskgraph._first_occurrences", forbidden)
            batch = next(task.batches(split="train", output_format="model"))
            graph = batch.inputs["graph"]
            assert len(graph["edges"]) == 2
            assert all(isinstance(key, str) for key in graph["edges"])
            assert all(
                edge["source_type"] == edge["destination_type"] == "people"
                for edge in graph["edges"].values()
            )
            assert {edge["direction"] for edge in graph["edges"].values()} == {"forward", "reverse"}
            for key, edge in graph["edges"].items():
                assert sum(graph["num_sampled_edges"][key]) == edge["index"].shape[1]


def test_sampled_artifact_retains_mode_and_buffers(sampled_plan, tmp_path):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        artifact = task.to_artifact()
        artifact.save(tmp_path / "artifact")
        loaded = api.Artifact.load(tmp_path / "artifact")
        assert loaded.contract.batch.inputs["graph"].representation.topology == "sampled"
        assert loaded.contract.fingerprint == artifact.contract.fingerprint
        held = next(task.batches(split="train", output_format="columnar"))
        pairs = {key: edge.index.to_pydict() for key, edge in held.inputs["graph"].edges.items()}
    assert pairs == {
        key: edge.index.to_pydict() for key, edge in held.inputs["graph"].edges.items()
    }


def test_sampled_resume_retains_unconsumed_topology(sampled_plan):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        options = dict(split="train", batch_size=1, output_format="model", shuffle=True, seed=7)
        batches = task.batches(**options)
        next(batches)
        state = batches.state_dict()
        expected = list(batches)
        resumed = task.batches(**options)
        resumed.load_state_dict(state)
        actual = list(resumed)
        assert len(actual) == len(expected) == 3
        for left, right in zip(actual, expected, strict=True):
            assert left.metadata["sample_ids"] == right.metadata["sample_ids"]
            left_graph, right_graph = left.inputs["graph"], right.inputs["graph"]
            torch.testing.assert_close(left_graph["nodes"], right_graph["nodes"])
            assert left_graph["num_sampled_nodes"] == right_graph["num_sampled_nodes"]
            assert left_graph["num_sampled_edges"] == right_graph["num_sampled_edges"]
            assert left_graph["edges"].keys() == right_graph["edges"].keys()
            for key, edge in left_graph["edges"].items():
                other = right_graph["edges"][key]
                assert (edge["source_type"], edge["destination_type"]) == (
                    other["source_type"],
                    other["destination_type"],
                )
                torch.testing.assert_close(edge["index"], other["index"])


@pytest.mark.parametrize("encoding", [api.ZScore(), api.Buckets(bins=3)])
def test_sampled_reuses_relation_field_encodings(sampled_plan, encoding):
    plan, anchors = sampled_plan
    outputs = []
    for topology in ("relations", "sampled"):
        use = plan.contract.batch.inputs["graph"]
        use = replace(
            use,
            representation=api.Graph(
                {
                    table: api.Fields((api.Encode("value", encoding),))
                    for table in ("people", "events")
                },
                topology=topology,
            ),
        )
        contract = replace(plan.contract, batch=replace(plan.contract.batch, inputs={"graph": use}))
        with api.prepare(contract, plan.dataset, plan.engine) as bound:
            with bound.materialize(anchors, fit_split="train") as task:
                outputs.append(
                    [
                        batch.inputs["graph"]["nodes"]
                        for batch in task.batches(split="train", output_format="model")
                    ]
                )
    torch.testing.assert_close(outputs[0], outputs[1])


@pytest.mark.parametrize("output_format", ["model", "inspectable"])
@pytest.mark.parametrize("cutoff", [-1, 0, 1])
@pytest.mark.parametrize(
    "topology,requests", [("sampled", False), ("relations", False), ("relations", True)]
)
def test_sampled_enforces_entity_availability(
    sampled_plan, tmp_path, output_format, cutoff, topology, requests
):
    plan, _ = sampled_plan
    contract = replace(
        plan.contract,
        sample=replace(
            plan.contract.sample, entity=api.EntitySpec("people", available_from="created")
        ),
        batch=replace(
            plan.contract.batch,
            inputs={
                "graph": replace(
                    plan.contract.batch.inputs["graph"],
                    representation=replace(
                        plan.contract.batch.inputs["graph"].representation, topology=topology
                    ),
                )
            },
        ),
    )
    path = tmp_path / "availability.parquet"
    pq.write_table(
        pa.table({"entity": [1], "cutoff": pa.array([cutoff], type=pa.timestamp("s", tz="UTC"))}),
        path,
    )
    anchors = api.AnchorTable(api.Parquet(str(path)), "entity", "cutoff")
    with api.prepare(contract, plan.dataset, plan.engine) as bound:
        with bound.materialize(anchors) as task:
            arguments = {}
            if requests:
                arguments["requests"] = task._anchors.fetch(api.ALL_ANCHORS, [0])
            if cutoff <= 0:
                with pytest.raises(ValueError, match="did not exist"):
                    next(task.batches(output_format=output_format, **arguments))
            else:
                assert next(task.batches(output_format=output_format, **arguments)).inputs["graph"]


def _refit_anchors(tmp_path, counts):
    splits = [name for name, count in zip(("train", "valid", "test"), counts) for _ in range(count)]
    path = tmp_path / "refit.parquet"
    pq.write_table(
        pa.table(
            {
                "sid": list(range(len(splits))),
                "entity": [1] * len(splits),
                "cutoff": pa.array([35] * len(splits), type=pa.timestamp("s", tz="UTC")),
                "split": splits,
            }
        ),
        path,
    )
    return api.AnchorTable(
        api.Parquet(str(path)), "entity", "cutoff", sample_id="sid", split="split"
    ), splits


@pytest.mark.parametrize("output_format", ["model", "inspectable", "columnar"])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("counts", [(1, 2), (5, 3, 1)])
def test_sampled_native_refit_coverage_padding_and_resume(
    sampled_plan, tmp_path, monkeypatch, output_format, world_size, counts
):
    plan, _ = sampled_plan
    anchors, splits = _refit_anchors(tmp_path, counts)
    with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:

        def forbidden(*args, **kwargs):
            pytest.fail("native refit rebuilt per-anchor requests or resampled in Python")

        monkeypatch.setattr(task._anchors, "_requests", forbidden)
        monkeypatch.setattr(plan._source, "sample_graph", forbidden)
        orders = []
        for epoch in (0, 1, 0):
            real = []
            epoch_order = []
            step_counts = []
            for rank in range(world_size):
                options = dict(
                    output_format=output_format,
                    batch_size=2,
                    shuffle=True,
                    seed=19,
                    epoch=epoch,
                    rank=rank,
                    world_size=world_size,
                    drop_last=False,
                )
                iterator = task.batches(**options)
                first = next(iterator)
                state = iterator.state_dict()
                suffix = list(iterator)
                resumed = task.batches(**options)
                resumed.load_state_dict(state)
                restored = list(resumed)
                assert [b.metadata["sample_ids"] for b in restored] == [
                    b.metadata["sample_ids"] for b in suffix
                ]
                batches = [first, *suffix]
                step_counts.append(len(batches))
                owned, padded = set(), []
                for batch in batches:
                    row_splits = []
                    for sid, pad, meta in zip(
                        batch.metadata["sample_ids"],
                        batch.metadata["is_padding"].tolist(),
                        batch.metadata["request_metadata"],
                        strict=True,
                    ):
                        original = int(sid.split("#padding")[0]) if pad else sid
                        assert meta["split"] == splits[original]
                        row_splits.append(meta["split"])
                        (padded if pad else real).append(original)
                        if not pad:
                            owned.add(original)
                        epoch_order.append((rank, sid))
                    assert len(set(row_splits)) == 1
                if owned:
                    assert set(padded) <= owned
                else:
                    assert all(b.metadata["is_padding"].all() for b in batches)
            assert len(set(step_counts)) == 1
            assert sorted(real) == list(range(len(splits)))
            orders.append(epoch_order)
        assert orders[0] == orders[2]


@pytest.mark.parametrize("output_format", ["model", "inspectable", "columnar"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_sampled_refit_drop_last_is_stage_local(sampled_plan, tmp_path, output_format, world_size):
    plan, _ = sampled_plan
    anchors, splits = _refit_anchors(tmp_path, (5, 4))
    with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
        steps = []
        for rank in range(world_size):
            batches = list(
                task.batches(
                    output_format=output_format,
                    batch_size=2,
                    world_size=world_size,
                    rank=rank,
                    drop_last=True,
                )
            )
            steps.append(len(batches))
            assert all(len(b.metadata["sample_ids"]) == 2 for b in batches)
            assert all(len({splits[sid] for sid in b.metadata["sample_ids"]}) == 1 for b in batches)
        assert len(set(steps)) == 1
        assert steps[0] == sum(((count + world_size - 1) // world_size) // 2 for count in (5, 4))


@pytest.mark.parametrize("output_format", ["model", "inspectable", "columnar"])
@pytest.mark.parametrize("counts", [(1, 2), (3, 3)])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("drop_last", [False, True])
def test_topologies_share_native_refit_schedule(
    sampled_plan, tmp_path, monkeypatch, output_format, counts, world_size, drop_last
):
    plan, _ = sampled_plan
    anchors, splits = _refit_anchors(tmp_path, counts)
    schedules = []
    for topology in ("relations", "sampled"):
        graph = plan.contract.batch.inputs["graph"]
        contract = replace(
            plan.contract,
            batch=replace(
                plan.contract.batch,
                inputs={
                    "graph": replace(
                        graph, representation=replace(graph.representation, topology=topology)
                    )
                },
            ),
        )
        with api.prepare(contract, plan.dataset, plan.engine) as bound:
            with bound.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:

                def forbidden(*args, **kwargs):
                    pytest.fail("native graph batches reconstructed anchor requests")

                monkeypatch.setattr(task._anchors, "_requests", forbidden)
                ranks = []
                for rank in range(world_size):
                    batches = list(
                        task.batches(
                            output_format=output_format,
                            batch_size=2,
                            shuffle=True,
                            seed=7,
                            epoch=1,
                            rank=rank,
                            world_size=world_size,
                            drop_last=drop_last,
                        )
                    )
                    ranks.append(
                        [
                            (
                                b.metadata["sample_ids"],
                                b.metadata["splits"],
                                b.metadata["is_padding"].tolist(),
                                b.metadata["request_metadata"],
                            )
                            for b in batches
                        ]
                    )
                    if drop_last:
                        assert len(batches) == sum(
                            ((n + world_size - 1) // world_size) // 2 for n in counts
                        )
                    else:
                        owned, padded = set(), []
                        for batch in batches:
                            for sid, pad, meta in zip(
                                batch.metadata["sample_ids"],
                                batch.metadata["is_padding"].tolist(),
                                batch.metadata["request_metadata"],
                                strict=True,
                            ):
                                original = int(sid.split("#padding")[0]) if pad else sid
                                assert meta["split"] == splits[original]
                                if pad:
                                    padded.append(original)
                                else:
                                    owned.add(original)
                        if owned:
                            assert set(padded) <= owned
                assert len({len(rank) for rank in ranks}) == 1
                schedules.append(ranks)
    assert schedules[0] == schedules[1]


def test_sampled_hops_preserve_discovery_in_a_cycle(tmp_path):
    timestamp = api.DType("timestamp", unit="s", timezone="UTC")
    schema = api.DataSchema(
        (
            api.TableSpec(
                "people",
                "id",
                {"id": api.DType("int64"), "manager": api.DType("int64"), "created": timestamp},
                time_column="created",
            ),
        ),
        (api.RelationSpec("manager", "people", "manager", "people", "id"),),
    )
    path = tmp_path / "cycle.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "manager": [2, 3, 4, 1],
                "created": pa.array([0] * 4, type=timestamp.to_arrow()),
            }
        ),
        path,
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("people"),
            neighborhoods={
                "g": api.NeighborhoodSpec(
                    tuple(api.GraphLayer((api.GraphSample("manager", 1),)) for _ in range(4))
                )
            },
        ),
        api.BatchSpec(
            inputs={
                "graph": api.Input(api.NeighborhoodSource("g"), api.Graph(topology="sampled"), None)
            },
            batching=api.Batching(max_samples=1),
        ),
    )
    anchor_path = tmp_path / "anchor.parquet"
    pq.write_table(pa.table({"entity": [1], "cutoff": [1]}), anchor_path)
    with api.prepare(
        contract,
        api.Dataset({"people": api.Parquet(str(path))}),
        api.DiskGraphEngine(num_threads=2),
    ) as plan:
        with plan.materialize(
            api.AnchorTable(api.Parquet(str(anchor_path)), "entity", "cutoff")
        ) as task:
            graph = next(task.batches(output_format="model")).inputs["graph"]
    assert graph["nodes"]["people"]["ids"].tolist() == [1, 2, 3, 4]
    assert graph["num_sampled_nodes"]["people"] == [1, 1, 1, 1, 0]
    edge = next(edge for edge in graph["edges"].values() if edge["index"].numel())
    assert edge["index"].tolist() == [[0, 1, 2, 3], [1, 2, 3, 0]]
    # Adding reverse edges creates a shortcut from root A to D. Recovering BFS
    # depths would move D from discovery hop three to hop one.
    distances = {0: 0}
    pairs = edge["index"].T.tolist()
    for hop in range(4):
        frontier = {node for node, distance in distances.items() if distance == hop}
        for left, right in pairs + [pair[::-1] for pair in pairs]:
            if left in frontier:
                distances.setdefault(right, hop + 1)
    assert distances[3] == 1


@pytest.mark.parametrize("topology", ["relations", "sampled"])
def test_native_graph_inspectable_keeps_raw_string_inputs_and_targets(
    native_task, tmp_path, topology
):
    original, _, anchors = native_task
    inputs = dict(original.contract.batch.inputs)
    graph = inputs["graph"]
    inputs["graph"] = replace(
        graph, representation=replace(graph.representation, topology=topology)
    )
    inputs["entity"] = api.Input(
        api.EntitySource(), api.Fields((api.Encode("raw_id", api.Raw()),)), api.Separate()
    )
    contract = replace(
        original.contract,
        sample=replace(
            original.contract.sample,
            entity_values={"raw_id": api.Field("id")},
            targets={"label": api.ExternalTarget(api.DType("string"))},
        ),
        batch=replace(original.contract.batch, inputs=inputs),
    )
    rows = pq.read_table(anchors.source.path)
    rows = rows.set_column(
        rows.schema.get_field_index("label"),
        "label",
        pa.array([f"label{i}" for i in range(len(rows))]),
    )
    path = tmp_path / "string-labels.parquet"
    pq.write_table(rows, path)
    anchors = replace(anchors, source=api.Parquet(str(path)))
    with api.prepare(
        contract,
        original.dataset,
        api.DiskGraphEngine(num_threads=1),
    ) as plan:
        with plan.materialize(anchors, fit_split="train") as task:
            batches = list(task.batches(split="train", output_format="inspectable"))
            assert [
                x for b in batches for x in b.inputs["entity"]["raw_id"]["values"].to_pylist()
            ] == ["u1", "u1", "u2", "u1"]
            assert [x for b in batches for x in b.targets["label"]["values"].to_pylist()] == [
                "label0",
                "label1",
                "label3",
                "label5",
            ]
            with pytest.raises(ValueError, match="Raw string"):
                next(task.batches(split="train", output_format="model"))


@pytest.mark.parametrize("topology", ["relations", "sampled"])
@pytest.mark.parametrize("output_format", ["model", "inspectable"])
def test_native_graph_iterators_restore_runtime_before_yield(sampled_plan, topology, output_format):
    original, anchors = sampled_plan
    graph = original.contract.batch.inputs["graph"]
    contract = replace(
        original.contract,
        batch=replace(
            original.contract.batch,
            inputs={
                "graph": replace(
                    graph, representation=replace(graph.representation, topology=topology)
                )
            },
        ),
    )
    with api.prepare(contract, original.dataset, original.engine) as plan:
        with plan.materialize(anchors, fit_split="train") as task:
            source = task._assembler._source
            first = task.batches(split="train", output_format=output_format)
            second = task.batches(split="train", output_format=output_format)
            next(first)
            assert getattr(source, "_runtime_batch", None) is None
            next(second)
            assert getattr(source, "_runtime_batch", None) is None
            list(first)
            list(second)
            assert getattr(source, "_runtime_batch", None) is None


def test_sampled_count_only_edge_type_has_empty_buffer(sampled_plan, monkeypatch):
    plan, anchors = sampled_plan
    with plan.materialize(anchors, fit_split="train") as task:
        source = task._assembler._source
        original = source.columnar_graph_batch
        removed = []

        def count_only(sample, spec, runtime, **kwargs):
            graph = dict(runtime["graphs"]["main"])
            _, selected = source._sampling_plan(sample.neighborhoods[spec.source])
            entry = next(
                item
                for item in graph["num_sampled_edges_dict"]
                if item["edge_type"]["rel"] in selected
            )
            physical = entry["edge_type"]["rel"]
            removed.append(physical)
            graph["num_sampled_edges_dict"] = [
                {**item, "counts": pa.array([0] * len(item["counts"]), type=item["counts"].type)}
                if item["edge_type"]["rel"] == physical
                else item
                for item in graph["num_sampled_edges_dict"]
            ]
            graph["edge_index_dict"] = [
                item for item in graph["edge_index_dict"] if item["edge_type"]["rel"] != physical
            ]
            return original(sample, spec, {**runtime, "graphs": {"main": graph}}, **kwargs)

        monkeypatch.setattr(source, "columnar_graph_batch", count_only)
        graph = next(task.batches(output_format="columnar")).inputs["graph"]
        assert removed[0] in graph.edges
        assert graph.edges[removed[0]].index.num_rows == 0
        assert sum(graph.num_sampled_edges[removed[0]]) == 0
