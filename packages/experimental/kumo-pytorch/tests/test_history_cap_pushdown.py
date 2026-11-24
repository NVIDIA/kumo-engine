# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global history caps bound every native component before payload decoding."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors

pytest.importorskip("diskgraph")


def _history_case(tmp_path, keep, order, layout, same_rows, *, budgeted_graph=False):
    sources = {}
    users = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), users)
    sources["users"] = api.Parquet(str(users))
    for component, name in enumerate(("events", "web")):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": list(range(64)),
                    "user": [1] * 64,
                    "time": pa.array(list(range(64)), type=pa.timestamp("s", tz="UTC")),
                    "value": [float(component * 1000 + i) for i in range(64)],
                    "label": [None if i == 63 else i for i in range(64)],
                }
            ),
            path,
        )
        sources[name] = api.Parquet(str(path))
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
            *(
                api.TableSpec(
                    name,
                    "id",
                    {
                        "id": api.DType("int64"),
                        "user": api.DType("int64"),
                        "time": api.DType("timestamp", unit="s", timezone="UTC"),
                        "value": api.DType("float64"),
                        "label": api.DType("int64"),
                    },
                    time_column="time",
                )
                for name in ("events", "web")
            ),
        ),
        relations=tuple(
            api.RelationSpec(name, "users", "id", name, "user") for name in ("events", "web")
        ),
    )
    fields = {"value": api.Field("value")}
    inputs = {
        "history": api.Input(
            api.EventSource(("events", "web"), max_events=3, keep=keep, order=order),
            api.Fields((api.Encode("value", api.Raw()),)),
            layout,
        )
    }
    neighborhoods, targets = {}, {}
    batching = api.Batching(max_samples=2)
    if budgeted_graph:
        batching = api.Batching(max_samples=2, max_tokens=3, length_input="history")
        fields.update(age=api.EventTimeDelta(), label=api.Field("label"))
        neighborhoods["graph"] = api.NeighborhoodSpec(
            (api.GraphLayer(tuple(api.GraphSample(name, 4) for name in ("events", "web"))),)
        )
        inputs["graph"] = api.Input(api.NeighborhoodSource("graph"), api.Graph(), None)
        inputs["ages"] = api.Input(
            api.EventsOf("history"), api.Fields((api.Encode("age", api.Raw()),)), layout
        )
        targets["labels"] = api.AlignedEventTarget("history", (api.Encode("label", api.Raw()),))
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            name: api.EventSpec(
                api.Rows(("events" if same_rows else name,), time=api.TimeWindow("time", None, 0)),
                fields,
            )
            for name in ("events", "web")
        },
        neighborhoods=neighborhoods,
    )
    contract = api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs, targets, batching=batching))
    return contract, api.Dataset(sources)


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
@pytest.mark.parametrize("same_rows", [False, True])
def test_component_caps_preserve_global_ties_and_empty_histories(
    tmp_path, monkeypatch, keep, order, layout, same_rows
):
    contract, dataset = _history_case(tmp_path, keep, order, layout, same_rows)
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"id": [1, 2], "at": [100, 100]}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "id", "at")
    observed = []
    original = NativeAnchors._iter

    def checked_iter(self, stage, **options):
        history = options.get("history")
        if history:
            plan, _ = map(json.loads, history)
            assert not plan["aggregates"]
            assert [item["limit"] for item in plan["sequences"]] == [3, 3]
        for runtime in original(self, stage, **options):
            if history:
                for sequence in runtime["history"]["sequences"]:
                    assert sequence["batch"].num_rows <= 3
                observed.append(runtime["history"]["stats"])
            yield runtime

    with api.prepare(contract, dataset, api.DiskGraphEngine(num_threads=2)) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            monkeypatch.setattr(NativeAnchors, "_iter", checked_iter)
            monkeypatch.setattr(
                plan._require_open(),
                "merge_event_streams",
                lambda *args: pytest.fail("native iteration must merge before returning to Python"),
            )
            batch = next(task.batches(output_format="model"))
    expected = [0.0, 1000.0, 1.0] if keep == "first" else [1062.0, 63.0, 1063.0]
    if same_rows:
        expected = [0.0, 0.0, 1.0] if keep == "first" else [62.0, 63.0, 63.0]
    if order == "descending":
        expected.reverse()
    history = batch.inputs["history"]
    actual = history["value"]["values"].tolist()
    assert actual == ([expected, [0.0] * 3] if isinstance(layout, api.Padded) else expected)
    assert observed
    # PK + clock + one payload field; only three rows per component survive.
    assert observed[0]["values_decoded"] <= 2 * 3 * 3


def test_graph_and_event_history_share_a_token_budget(tmp_path, monkeypatch):
    contract, dataset = _history_case(
        tmp_path, "last", "ascending", api.Packed(), False, budgeted_graph=True
    )
    cutoffs = [100, 61]
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"user": [1, 1], "at": cutoffs}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "user", "at")
    engine = api.DiskGraphEngine(num_threads=1)
    with (
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task,
    ):

        def forbidden(*args, **kwargs):
            pytest.fail("graph and history delivery fell back to Python request batching")

        monkeypatch.setattr(task._anchors, "_requests", forbidden)
        monkeypatch.setattr(task._assembler, "batches", forbidden)
        monkeypatch.setattr(plan._require_open(), "_sample_graph_events", forbidden)
        batches = list(task.batches(output_format="model"))

    assert [list(batch.metadata["sample_ids"]) for batch in batches] == [[0], [1]]
    for cutoff, batch in zip(cutoffs, batches, strict=True):
        assert batch.metadata["entity_ids"].tolist() == [1]
        assert not any(batch.metadata["is_padding"])
        assert batch.metadata["as_of_time"].tolist() == [cutoff]
        end = min(cutoff, 63)
        times = [end - 1, end, end]
        history, ages = batch.inputs["history"], batch.inputs["ages"]
        labels = batch.targets["labels"]
        assert history["valid"].tolist() == ages["valid"].tolist() == [True] * 3
        assert history["value"]["valid"].all() and ages["age"]["valid"].all()
        for view in (history, ages, labels):
            assert view["offsets"].tolist() == [0, 3]
        assert history["value"]["values"].tolist() == [1000 + end - 1, end, 1000 + end]
        assert ages["age"]["values"].tolist() == [cutoff - t for t in times]
        valid = labels["label"]["valid"]
        assert valid.tolist() == [t != 63 for t in times]
        assert labels["label"]["values"][valid].tolist() == [t for t in times if t != 63]
        graph = batch.inputs["graph"]
        roots = graph["nodes"]["users"]
        assert roots["ids"].tolist() == [1] and roots["batch"].tolist() == [0]
        for table in ("events", "web"):
            nodes = graph["nodes"][table]
            src, dst = graph["edges"][table]["index"]
            neighbors = list(range(end - 3, end + 1))
            assert sorted(nodes["ids"].tolist()) == sorted(nodes["ids"][dst].tolist()) == neighbors
            assert roots["ids"][src].tolist() == [1] * 4
            assert nodes["batch"].tolist() == roots["batch"][src].tolist() == [0] * 4


@pytest.mark.parametrize("with_graph", [False, True], ids=["history", "graph_and_history"])
@pytest.mark.parametrize(
    "splits, max_tokens, expected",
    [
        (["train", "val"], 6, [[0, 1]]),
        (["train", "val", "val"], 6, [[0, 1]]),
        (["train", "val"], 3, [[0]]),
    ],
    ids=["full_batch_across_splits", "incomplete_final_batch", "budget_closed_batch"],
)
def test_budgeted_drop_last_across_splits(tmp_path, with_graph, splits, max_tokens, expected):
    contract, dataset = _history_case(
        tmp_path, "last", "ascending", api.Packed(), False, budgeted_graph=with_graph
    )
    batching = api.Batching(max_samples=2, max_tokens=max_tokens, length_input="history")
    contract = replace(contract, batch=replace(contract.batch, batching=batching))
    path = tmp_path / "anchors.parquet"
    cutoffs = [100, 61, 50][: len(splits)]  # Each anchor has three selected history events.
    pq.write_table(pa.table({"user": [1] * len(splits), "at": cutoffs, "split": splits}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "user", "at", split="split")
    engine = api.DiskGraphEngine(num_threads=1)
    with (
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task,
    ):
        requests = task._anchors.fetch(api.ALL_ANCHORS, range(len(splits)))
        for drop_last in (False, True):
            reference = task.batches(requests, drop_last=drop_last, output_format="model")
            reference_ids = [list(batch.metadata["sample_ids"]) for batch in reference]
            if drop_last:
                assert reference_ids == expected
            else:
                assert [i for batch in reference_ids for i in batch] == list(range(len(splits)))
            actual = task.batches(
                split=api.ALL_ANCHORS, drop_last=drop_last, chunk_batches=1, output_format="model"
            )
            assert [list(batch.metadata["sample_ids"]) for batch in actual] == reference_ids
