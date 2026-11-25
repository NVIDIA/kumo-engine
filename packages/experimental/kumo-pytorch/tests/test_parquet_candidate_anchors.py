# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-supplied Parquet candidates retain order across native anchor scheduling."""

from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.parquet_anchors import ParquetAnchors, check_parquet_anchors
from tests.test_candidate_neighborhoods import REQUESTS, assert_graphs, contract, parquet_inputs
from tests.test_scored_walks import star


def test_nested_candidates_follow_split_shard_padding_and_resume(tmp_path):
    schema, rows = star()
    requests = (*REQUESTS, replace(REQUESTS[-1], sample_id="held_out"))
    dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    path = tmp_path / "anchors.parquet"
    data = pq.read_table(path).append_column("split", pa.array(["train"] * 3 + ["test"]))
    pq.write_table(data, path)
    anchors = replace(
        anchors,
        source=api.Parquet(str(path), columns={"positives": "peers"}),
        candidates={"positive": api.CandidateColumn("peers", "pin")},
        split="split",
    )
    with (
        api.prepare(contract(schema), dataset, api.DiskGraphEngine(num_threads=2)) as plan,
        plan.materialize(anchors, fit_split="train") as task,
        patch.object(task._anchors, "_requests", side_effect=AssertionError("Python fallback")),
    ):
        seen = []
        padded_count = 0
        for rank in range(2):
            options = dict(
                split="train",
                batch_size=1,
                world_size=2,
                rank=rank,
                seed=19,
                shuffle=True,
                output_format="columnar",
            )
            stream = task.batches(**options)
            first = next(stream)
            cursor = stream.state_dict()
            resumed = task.batches(**options)
            resumed.load_state_dict(cursor)
            (second,) = stream
            (replay,) = resumed
            assert second.metadata["sample_ids"] == replay.metadata["sample_ids"]
            for batch in (first, second, replay):
                ids = tuple(str(s).split("#padding")[0] for s in batch.metadata["sample_ids"])
                assert_graphs(replace(batch, metadata={**batch.metadata, "sample_ids": ids}))
            for batch in (first, second):
                if batch.metadata["is_padding"].item():
                    padded_count += 1
                else:
                    seen.extend(batch.metadata["sample_ids"])
        assert sorted(seen) == ["early", "empty", "late"]
        assert padded_count == 1


def test_parquet_candidates_require_list_of_structs_and_bound_fields(tmp_path):
    path = tmp_path / "bad.parquet"
    table = api.AnchorTable(
        api.Parquet(str(path)),
        "owner",
        candidates={"positive": api.CandidateColumn("peers", "pin", targets={"clicked": "label"})},
    )
    for entries in (pa.array([[1]]), pa.array([[{"pin": 1}]])):
        pq.write_table(pa.table({"owner": [1], "peers": entries}), path)
        issues = check_parquet_anchors(table)
        assert len(issues) == 1 and issues[0].path == "anchors.candidates.positive"
        with pytest.raises(ValueError, match="list of structs"):
            ParquetAnchors(table)


def test_parquet_candidate_order_nulls_and_submicrosecond_labels(tmp_path):
    path = tmp_path / "anchors.parquet"
    dtype = pa.list_(pa.struct([("pin", pa.int32()), ("at", pa.timestamp("ns", tz="UTC"))]))
    nested = pa.array([None, [], [{"pin": 3, "at": 3}, {"pin": 1, "at": 1}]], type=dtype)
    pq.write_table(pa.table({"id": [0, 1, 2], "positives": nested}), path)
    table = api.AnchorTable(
        api.Parquet(str(path)),
        "id",
        candidates={"positive": api.CandidateColumn("positives", "pin", targets={"label": "at"})},
    )
    anchors = ParquetAnchors(table)
    selected = anchors.fetch(api.ALL_ANCHORS, [2, 1, 0, 2])
    groups = [request.candidates["positive"] for request in selected]
    assert [tuple(entry.entity_id for entry in group) for group in groups] == [
        (3, 1),
        (),
        (),
        (3, 1),
    ]
    assert [entry.targets[0].value.value for entry in groups[0]] == [3, 1]


def test_native_walk_counters_include_candidate_work(tmp_path):
    schema, rows = star()
    dataset, anchors = parquet_inputs(schema, rows, REQUESTS, tmp_path)
    mixed = contract(schema)
    keys = ("walk_pushes", "walk_decoded_entries")
    observed = {}
    for name, inputs in (
        ("root", {"graph": mixed.batch.inputs["graph"]}),
        ("candidates", {"positive_graph": mixed.batch.inputs["positive_graph"]}),
        ("both", mixed.batch.inputs),
    ):
        scope = mixed.sample if name != "candidates" else replace(mixed.sample, neighborhoods={})
        decl = replace(mixed, sample=scope, batch=replace(mixed.batch, inputs=inputs))
        with (
            api.prepare(decl, dataset, api.DiskGraphEngine(num_threads=2)) as plan,
            plan.materialize(anchors) as task,
        ):
            for fmt in ("model", "columnar", "inspectable"):
                batch = next(task.batches(output_format=fmt))
                observed[name, fmt] = tuple(batch.metadata[k] for k in keys)
    for fmt in ("model", "columnar", "inspectable"):
        root, candidates = observed["root", fmt], observed["candidates", fmt]
        assert min(*root, *candidates) > 0
        assert observed["both", fmt] == tuple(a + b for a, b in zip(root, candidates, strict=True))
