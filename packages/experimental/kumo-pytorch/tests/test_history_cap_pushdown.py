# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global history caps bound every native component before payload decoding."""

import json
import re
from contextlib import ExitStack
from itertools import chain

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors
from tests.helpers import parquet_anchors


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
@pytest.mark.parametrize("same_rows", [False, True])
def test_component_caps_preserve_global_ties_and_empty_histories(
    tmp_path, monkeypatch, keep, order, layout, same_rows
):
    pytest.importorskip("diskgraph")
    _check_history(tmp_path, monkeypatch, keep, order, layout, same_rows)


@pytest.mark.usefixtures("native_anchor_delivery")
@pytest.mark.parametrize(
    "keep,order,layout,same_rows,budget",
    [
        (None, "ascending", api.Packed(), False, False),
        (None, "descending", api.Padded(), True, False),
        ("last", "ascending", api.Packed(), False, True),
        ("last", "descending", api.Padded(), True, True),
    ],
)
def test_history_membership_and_aligned_supervision(
    tmp_path, monkeypatch, keep, order, layout, same_rows, budget
):
    pytest.importorskip("diskgraph")
    _check_history(
        tmp_path, monkeypatch, keep, order, layout, same_rows, extended=True, budget=budget
    )


def _history_case(tmp_path, keep, order, layout, same_rows, *, extended, budget, graph):
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
                    **({"label": [None if i == 63 else i for i in range(64)]} if budget else {}),
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
                        **({"label": api.DType("int64")} if budget else {}),
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
    windows = (
        {"events": (-42, -38), "web": (-41, -37)}
        if keep is None
        else dict.fromkeys(("events", "web"), (None, 0))
    )
    limit = None if keep is None else 3
    fields = {"value": api.Field("value")}
    inputs = {
        "history": api.Input(
            api.EventSource(("events", "web"), max_events=limit, keep=keep or "first", order=order),
            api.Fields((api.Encode("value", api.Raw()),)),
            layout,
        ),
    }
    targets = {}
    neighborhoods = {}
    if extended:
        fields.update(age=api.EventTimeDelta(), label=api.Field("label" if budget else "id"))
        inputs["ages"] = api.Input(
            api.EventsOf("history"), api.Fields((api.Encode("age", api.Raw()),)), layout
        )
        targets["labels"] = api.AlignedEventTarget("history", (api.Encode("label", api.Raw()),))
        if graph:
            neighborhoods["graph"] = api.NeighborhoodSpec(
                (api.GraphLayer(tuple(api.GraphSample(name, 4) for name in ("events", "web"))),)
            )
            inputs["graph"] = api.Input(api.NeighborhoodSource("graph"), api.Graph(), None)
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            name: api.EventSpec(
                api.Rows(
                    ("events" if same_rows else name,), time=api.TimeWindow("time", *windows[name])
                ),
                fields,
            )
            for name in ("events", "web")
        },
        neighborhoods=neighborhoods,
    )
    batching = (
        api.Batching(max_samples=2, max_tokens=budget, length_input="history")
        if budget
        else api.Batching(max_samples=2)
    )
    contract = api.DataLoaderSpec(
        schema, sample, api.BatchSpec(inputs, targets=targets, batching=batching)
    )
    return contract, api.Dataset(sources)


def _check_history(
    tmp_path, monkeypatch, keep, order, layout, same_rows, *, extended=False, budget=False
):
    contract, dataset = _history_case(
        tmp_path,
        keep,
        order,
        layout,
        same_rows,
        extended=extended,
        budget=6 if budget else None,
        graph=not budget,
    )
    limit = contract.batch.inputs["history"].source.max_events
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
            if limit is not None:
                assert [item["limit"] for item in plan["sequences"]] == [limit, limit]
        for runtime in original(self, stage, **options):
            if history:
                for sequence in runtime["history"]["sequences"]:
                    if limit is not None:
                        assert sequence["batch"].num_rows <= limit
                observed.append(runtime["history"]["stats"])
            yield runtime

    with api.prepare(
        contract, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            if not budget:
                monkeypatch.setattr(NativeAnchors, "_iter", checked_iter)
                monkeypatch.setattr(
                    plan._require_open(),
                    "merge_event_streams",
                    lambda *args: pytest.fail(
                        "native iteration must merge before returning to Python"
                    ),
                )
            batch = next(task.batches(output_format="model"))
    expected = [0.0, 1000.0, 1.0] if keep == "first" else [1062.0, 63.0, 1063.0]
    if same_rows:
        expected = [0.0, 0.0, 1.0] if keep == "first" else [62.0, 63.0, 63.0]
    if keep is None:
        # At cutoff 100, events has times (58, 62] and web has (59, 63].
        web = 0 if same_rows else 1000
        expected = [59, 60, web + 60, 61, web + 61, 62, web + 62, web + 63]
    if order == "descending":
        expected.reverse()
    history = batch.inputs["history"]
    actual = history["value"]["values"].tolist()
    assert actual == (
        [expected, [0.0] * len(expected)] if isinstance(layout, api.Padded) else expected
    )
    if not budget:
        assert observed
        if limit is not None:
            assert observed[0]["values_decoded"] <= 2 * limit * 3
    else:
        assert history["valid"].numel() <= 6
    if not extended:
        return
    if isinstance(layout, api.Padded):
        assert history["valid"].tolist() == [[True] * len(expected), [False] * len(expected)]
    ages, labels = batch.inputs["ages"], batch.targets["labels"]
    valid = history["valid"]
    assert "label" not in history
    assert ages["valid"].equal(valid)
    times = [int(v) % 1000 for v in expected]
    assert ages["age"]["values"][valid].tolist() == [100 - t for t in times]
    supervised = labels["label"]["valid"]
    assert not supervised[~valid].any()
    assert supervised[valid].tolist() == [not budget or t != 63 for t in times]
    assert labels["label"]["values"][supervised].tolist() == [
        t for t in times if not budget or t != 63
    ]
    if isinstance(layout, api.Packed):
        assert (
            history["offsets"].tolist()
            == ages["offsets"].tolist()
            == labels["offsets"].tolist()
            == [0, len(expected), len(expected)]
        )
    if budget:
        return
    graph = batch.inputs["graph"]
    for table in ("events", "web"):
        nodes = graph["nodes"][table]
        assert sorted(nodes["ids"].tolist()) == [60, 61, 62, 63]
        assert nodes["batch"].tolist() == [0] * 4
        src, dst = graph["edges"][table]["index"]
        roots = graph["nodes"]["users"]
        assert roots["ids"][src].tolist() == [1] * 4
        assert nodes["batch"][dst].tolist() == roots["batch"][src].tolist()


def test_graph_and_event_history_share_a_token_budget(tmp_path, native_anchor_delivery):
    from diskgraph import DiskgraphConfigError

    contract, dataset = _history_case(
        tmp_path, "last", "ascending", api.Packed(), False, extended=True, budget=3, graph=True
    )
    cutoffs = [100, 61]
    anchors = parquet_anchors(
        tmp_path / "anchors.parquet", {"user": [1, 1], "at": cutoffs}, "user", "at"
    )
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with ExitStack() as stack:
        try:
            plan = stack.enter_context(api.prepare(contract, dataset, engine))
            task = stack.enter_context(plan.materialize(anchors, fit_split=api.ALL_ANCHORS))
            batches = iter(native_anchor_delivery(task))
            first = next(batches)
        except (ValueError, NotImplementedError, DiskgraphConfigError) as exc:
            messages = (
                [i.message for i in exc.issues]
                if isinstance(exc, api.ValidationError)
                else [str(exc)]
            )
            graph = r"(?:native )?graphs?(?: (?:inputs?|outputs?|batches|sampling))?"
            budget = r"(?:token budgets?|token budgeting|max_tokens)"
            both = rf"(?:{graph} (?:and|with) {budget}|{budget} (?:and|with) {graph})"
            refusals = (
                rf"(?:mixed )?{both}(?: delivery| batching)? (?:is|are) "
                r"(?:unsupported|not supported)(?: together)?",
                rf"{both} cannot be combined",
                rf"{graph} (?:cannot be combined with|do(?:es)? not support) {budget}",
                rf"{budget} (?:cannot be combined with|(?:is |are )?not supported with) {graph}",
                rf"diskgraph does not support {both}",
            )
            assert messages
            for message in messages:
                reason = message.lower().replace("-", " ").rstrip(".")
                assert any(re.fullmatch(pattern, reason) for pattern in refusals), message
            return

        delivered = []
        for batch in chain((first,), batches):
            # Each anchor fills the three-slot budget. Ignoring it would put both in one batch.
            (sid,) = batch.metadata["sample_ids"]
            delivered.append(sid)
            assert batch.metadata["entity_ids"].tolist() == [1] and not any(
                batch.metadata["is_padding"]
            )
            cutoff = cutoffs[sid]
            assert batch.metadata["as_of_time"].tolist() == [cutoff]
            end = min(cutoff, 63)
            times = [end - 1, end, end]
            history, ages, labels = (
                batch.inputs["history"],
                batch.inputs["ages"],
                batch.targets["labels"],
            )
            assert history["valid"].numel() == contract.batch.batching.max_tokens
            assert history["valid"].all() and ages["valid"].all()
            assert history["value"]["valid"].all() and ages["age"]["valid"].all()
            assert (
                history["offsets"].tolist()
                == ages["offsets"].tolist()
                == labels["offsets"].tolist()
                == [0, 3]
            )
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
                assert (
                    sorted(nodes["ids"].tolist()) == sorted(nodes["ids"][dst].tolist()) == neighbors
                )
                assert roots["ids"][src].tolist() == [1] * 4
                assert nodes["batch"].tolist() == roots["batch"][src].tolist() == [0] * 4
        assert delivered == [0, 1]
