# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import torch

from kumo.pytorch import api
from kumo.pytorch.engines import candidate_walks
from tests.e2e.omnisage_fixture import make_fixture, spec


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key, value in expected.items():
            _assert_same(actual[key], value)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


def test_grouped_candidate_graphs_match_separate_walks(tmp_path, monkeypatch):
    data = make_fixture(tmp_path / "source", pins=32, anchors=48, width=16)
    task_spec = spec(data, 48)
    sets = dict(task_spec.sample.candidate_sets)
    sets["empty"] = replace(sets["supplied"], draw=None)
    inputs = dict(task_spec.batch.inputs)
    # A second input for identical roots, a permanently empty group, and an incompatible
    # projection which takes the ordinary path alongside the shared walks.
    inputs["duplicate"] = inputs["supplied"]
    inputs["empty"] = replace(inputs["supplied"], source=api.NeighborhoodSource("graph", "empty"))
    inputs["different"] = replace(inputs["future"], representation=api.Graph({}))
    task_spec = replace(
        task_spec,
        sample=replace(task_spec.sample, candidate_sets=sets),
        batch=replace(task_spec.batch, inputs=inputs),
    )
    with (
        api.prepare(task_spec, data.dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(data.anchors) as task,
    ):
        cache = task._anchors._graph_feature_cache
        read = cache.read
        calls = []

        def traced(key, *args, **kwargs):
            calls.append(key)
            return read(key, *args, **kwargs)

        monkeypatch.setattr(cache, "read", traced)
        actual = next(task.batches(output_format="model", shuffle=False))
        # The anchor and every compatible candidate input share one physical read per table.
        assert [table for table, _ in calls] == ["boards", "pins"]
        monkeypatch.setattr(candidate_walks, "compatible_walk_groups", lambda *_: ())
        separate = next(task.batches(output_format="model", shuffle=False))

    _assert_same(actual.inputs, separate.inputs)
    assert actual.metadata["candidate_entity_ids"] == separate.metadata["candidate_entity_ids"]
    # The native counters measure pushes and decodes actually performed on distinct ID/time pairs.
    for name in ("walk_pushes", "walk_decoded_entries"):
        assert 0 < actual.metadata[name] < separate.metadata[name]
    assert actual.inputs["empty"]["nodes"]["pins"]["ids"].numel() == 0
    images = actual.inputs["supplied"]["nodes"]["pins"]["features"]["image"]["values"]
    images.fill_(-77)
    _assert_same(actual.inputs["duplicate"], separate.inputs["duplicate"])
    _assert_same(actual.inputs["query"], separate.inputs["query"])
