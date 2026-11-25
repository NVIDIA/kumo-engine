# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph batches checked against the exact stationary scores of a star."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api


def star():
    integer = api.DType("int64")
    clock = api.DType("timestamp", unit="s", timezone="UTC")
    schema = api.DataSchema(
        (
            api.TableSpec("pins", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec("boards", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "membership",
                "id",
                {"id": integer, "pin": integer, "board": integer, "at": clock},
                time_column="at",
            ),
        ),
        (
            api.RelationSpec("pin_links", "pins", "id", "membership", "pin"),
            api.RelationSpec("board_links", "boards", "id", "membership", "board"),
        ),
    )
    rows = {
        "pins": [{"id": i} for i in range(1, 5)],
        "boards": [{"id": 10}],
        "membership": [
            {"id": i, "pin": pin, "board": 10, "at": at}
            for i, (pin, at) in enumerate(((1, 10), (2, 10), (3, 20), (3, 20)), 1)
        ],
    }
    return schema, rows


def walk_contract(schema, *, after_hops=False, budget=100_000):
    walk = api.WalkSample("walk", ("membership",), {"pins": 2, "boards": 1}, 0.5, budget)
    layers = (
        (
            api.GraphLayer((api.GraphSample("board_links", 10, strategy="recent"),)),
            api.GraphLayer((api.GraphSample("pin_links", 1, direction="reverse"),)),
        )
        if after_hops
        else ()
    )
    return api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("boards" if after_hops else "pins"),
            neighborhoods={"graph": api.NeighborhoodSpec(layers + (api.GraphLayer((walk,)),))},
        ),
        api.BatchSpec(
            inputs={"graph": api.Input(api.NeighborhoodSource("graph"), api.Graph(), None)},
            batching=api.Batching(max_samples=3),
        ),
    )


def write_parquet(schema, rows, requests, directory):
    sources = {}
    for table in schema.tables:
        path = directory / f"{table.name}.parquet"
        fields = pa.schema([(name, dtype.to_arrow()) for name, dtype in table.columns.items()])
        pq.write_table(pa.Table.from_pylist(rows[table.name], schema=fields), path)
        sources[table.name] = api.Parquet(str(path))
    path = directory / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sid": [request.sample_id for request in requests],
                "entity": [request.entity_id for request in requests],
                "at": pa.array([r.as_of_time for r in requests], type=pa.timestamp("s", tz="UTC")),
            }
        ),
        path,
    )
    return api.Dataset(sources), api.AnchorTable(
        api.Parquet(str(path)), "entity", "at", sample_id="sid"
    )


def walk_answer(batches):
    answer = {}
    for batch in batches:
        graph = batch.inputs["graph"]
        samples = batch.metadata["sample_ids"]
        for sid in samples:
            answer[sid] = {}
        for table in ("pins", "boards"):
            if isinstance(graph, api.ColumnarGraph):
                edge = graph.edges[f"walk.{table}"]
                assert edge.score.type == pa.float32() and edge.rank.type == pa.int64()
                src, dst = (arr.to_pylist() for arr in edge.index.columns)
                scores, ranks = edge.score.to_pylist(), edge.rank.to_pylist()
                roots = graph.nodes["pins"]
                owners, root_ids = roots.owners.to_pylist(), roots.ids.to_pylist()
                dest = graph.nodes[table]
                dest_owners, ids = dest.owners.to_pylist(), dest.ids.to_pylist()
            else:
                edge = graph["edges"][f"walk.{table}"]
                assert edge["score"].dtype == torch.float32 and edge["rank"].dtype == torch.long
                src, dst = edge["index"].tolist()
                scores, ranks = edge["score"].tolist(), edge["rank"].tolist()
                roots = graph["nodes"]["pins"]
                owners, root_ids = roots["batch"].tolist(), roots["ids"].tolist()
                dest = graph["nodes"][table]
                dest_owners, ids = dest["batch"].tolist(), dest["ids"].tolist()
            for i, j, score, rank in zip(src, dst, scores, ranks, strict=True):
                assert owners[i] == dest_owners[j]
                key = root_ids[i], table, rank
                record = answer[samples[owners[i]]]
                assert key not in record
                record[key] = ids[j], score
    return answer


def assert_oracle(answer):
    # On the bipartite star at restart=1/2, board mass is 1/3. Half that
    # mass returns to Pins in proportion to their visible edge multiplicity.
    expected = {
        "early": {(1, "boards", 0): (10, 1 / 3), (1, "pins", 0): (2, 1 / 12)},
        "late": {
            (1, "boards", 0): (10, 1 / 3),
            (1, "pins", 0): (3, 1 / 12),
            (1, "pins", 1): (2, 1 / 24),
        },
        "isolated": {},
    }
    assert answer.keys() == expected.keys()
    for sid, rows in expected.items():
        assert answer[sid].keys() == rows.keys()
        for key, (node, score) in rows.items():
            actual_id, actual_score = answer[sid][key]
            assert actual_id == node
            assert actual_score == pytest.approx(score, abs=0.0001)


def test_walk_scores_at_two_cutoffs_and_an_isolate(tmp_path):
    schema, rows = star()
    requests = (
        api.SampleRequest("early", 1, 10),
        api.SampleRequest("late", 1, 20),
        api.SampleRequest("isolated", 4, 20),
    )
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    engine = api.DiskGraphEngine(num_threads=1)
    with (
        api.prepare(walk_contract(schema), dataset, engine) as plan,
        plan.materialize(anchors) as task,
    ):
        for fmt in ("inspectable", "model"):
            assert_oracle(walk_answer(task.request_batches(requests, output_format=fmt)))
        for fmt in ("inspectable", "model", "columnar"):
            assert_oracle(walk_answer(task.batches(output_format=fmt)))


def test_walk_starts_from_each_history_pin(tmp_path):
    schema, rows = star()
    requests = [api.SampleRequest("history", 10, 10)]
    engine = api.DiskGraphEngine(num_threads=1)
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    with (
        api.prepare(walk_contract(schema, after_hops=True), dataset, engine) as plan,
        plan.materialize(anchors) as task,
    ):
        result = walk_answer(task.request_batches(requests))["history"]
        assert set(result) == {(root, kind, 0) for root in (1, 2) for kind in ("pins", "boards")}
        for root, other in ((1, 2), (2, 1)):
            assert result[root, "pins", 0] == pytest.approx((other, 1 / 12), abs=0.0001)
        assert walk_answer(task.batches(output_format="columnar"))["history"] == result


def test_pin_board_and_pin_pin_walks_in_one_mixed_layer(tmp_path):
    """One Pin seed produces both OmniSage walks and an ordinary relation hop."""
    from dataclasses import replace

    schema, rows = star()
    integer = api.DType("int64")
    engaged = api.TableSpec(
        "engaged", None, {"src": integer, "dst": integer}, time_column=api.NON_TEMPORAL
    )
    schema = replace(
        schema,
        tables=(*schema.tables, engaged),
        relations=(
            *schema.relations,
            api.RelationSpec("engaged_from", "pins", "id", "engaged", "src"),
            api.RelationSpec("engaged_to", "pins", "id", "engaged", "dst"),
        ),
    )
    rows["engaged"] = [{"src": 1, "dst": 4}]
    requests = [api.SampleRequest("pin", 1, 10)]
    declarations = (
        api.GraphSample("pin_links", 1, strategy="recent"),
        api.WalkSample("pin_board", ("membership",), {"pins": 25, "boards": 75}, 0.5, 100_000),
        api.WalkSample("pin_pin", ("engaged",), {"pins": 50}, 0.5, 100_000),
    )
    contract = walk_contract(schema)
    contract = replace(
        contract,
        sample=replace(
            contract.sample,
            neighborhoods={"graph": api.NeighborhoodSpec((api.GraphLayer(declarations),))},
        ),
    )
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    engine = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract, dataset, engine) as plan, plan.materialize(anchors) as task:
        batch = next(task.batches(output_format="model"))
        graph = batch.inputs["graph"]
        assert set(graph["edges"]) == {
            "pin_links",
            "pin_board.pins",
            "pin_board.boards",
            "pin_pin.pins",
        }
        assert "score" not in graph["edges"]["pin_links"]
        for group, table, want_id, score in (
            ("pin_board.pins", "pins", 2, 1 / 12),
            ("pin_board.boards", "boards", 10, 1 / 3),
            ("pin_pin.pins", "pins", 4, 1 / 3),
        ):
            edge = graph["edges"][group]
            src, dst = edge["index"]
            assert graph["nodes"]["pins"]["ids"][src].tolist() == [1]
            assert graph["nodes"][table]["ids"][dst].tolist() == [want_id]
            assert edge["rank"].tolist() == [0]
            assert edge["score"].tolist() == pytest.approx([score], abs=0.0001)


@pytest.mark.parametrize("budget, expected", [(1, {}), (2, {1: 0.25}), (100_000, {1: 1 / 3})])
def test_direct_relation_uses_residual_degree_threshold(tmp_path, budget, expected):
    schema, rows = star()
    walk = api.WalkSample("direct", ("pin_links",), {"membership": 1}, 0.5, budget)
    spec = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("pins"),
            neighborhoods={"g": api.NeighborhoodSpec((api.GraphLayer((walk,)),))},
        ),
        api.BatchSpec(inputs={"g": api.Input(api.NeighborhoodSource("g"), api.Graph(), None)}),
    )
    requests = [api.SampleRequest("one_edge", 1, 10)]
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    with (
        api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(anchors) as task,
    ):
        graph = next(task.request_batches(requests, batch_size=1)).inputs["g"]
    edge = graph["edges"]["direct.membership"]
    ids = graph["nodes"]["membership"]["ids"][edge["index"][1]].tolist()
    assert dict(zip(ids, edge["score"].tolist(), strict=True)) == pytest.approx(
        expected, abs=0.0001
    )
    assert edge["rank"].tolist() == list(range(len(expected)))


def test_next_hop_expands_a_walk_destination_already_seen(tmp_path):
    """Returning to an existing root places it on the next frontier for each owner."""
    schema, rows = star()
    requests = [api.SampleRequest(sid, 10, 20) for sid in ("a", "b")]
    layers = (
        api.GraphLayer((api.GraphSample("board_links", 1, strategy="recent"),)),
        api.GraphLayer((api.GraphSample("pin_links", 1, direction="reverse"),)),
        api.GraphLayer((api.WalkSample("back", ("membership",), {"boards": 1}, 0.5, 100),)),
        api.GraphLayer((api.GraphSample("board_links", 4, strategy="recent"),)),
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec("boards"), neighborhoods={"g": api.NeighborhoodSpec(layers)}),
        api.BatchSpec(
            inputs={"g": api.Input(api.NeighborhoodSource("g"), api.Graph(), None)},
            batching=api.Batching(max_samples=2),
        ),
    )
    dataset, anchors = write_parquet(schema, rows, requests, tmp_path)
    engine = api.DiskGraphEngine(num_threads=1)

    def check(batch):
        g = batch.inputs["g"]
        nodes = g["nodes"]["membership"]
        for owner in (0, 1):
            assert sorted(nodes["ids"][nodes["batch"] == owner].tolist()) == [1, 2, 3, 4]
        # Each walk starts from its own Pin frontier and returns its owning board.
        back = g["edges"]["back.boards"]
        assert back["rank"].tolist() == [0, 0]
        src, dst = back["index"]
        assert sorted(g["nodes"]["pins"]["batch"][src].tolist()) == [0, 1]
        assert torch.equal(g["nodes"]["pins"]["batch"][src], g["nodes"]["boards"]["batch"][dst])

    with api.prepare(contract, dataset, engine) as plan, plan.materialize(anchors) as task:
        check(next(task.request_batches(requests)))
        check(next(task.batches(output_format="model")))
