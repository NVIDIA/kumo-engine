# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent answers must reject matching-but-wrong native outputs and state."""

from copy import deepcopy

import pytest

from conformance.generate import generate_batch
from conformance.properties import PropertyFailure, fitted_state
from kumo.pytorch.artifact import FittedState


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "kind",
        "ids",
        "moments",
        "entity_cuts",
        "spaces",
        "field_names",
        "inputs",
        "targets",
    ],
)
def test_fitted_oracle_rejects_corrupt_serialized_state(monkeypatch, defect):
    original = FittedState.to_dict

    def corrupt(self):
        state = deepcopy(original(self))
        if defect in ("spaces", "field_names", "inputs", "targets"):
            state[defect] = {}
        encoders = state["encoders"]
        if defect == "missing":
            encoders.clear()
        for identity, encoder in encoders.items():
            if defect == "kind" and encoder["kind"] == "zscore":
                encoder["kind"] = "raw"
            if defect == "ids" and encoder["kind"] == "vocabulary":
                encoder["entries"] = [[value, code + 1000] for value, code in encoder["entries"]]
            if defect == "moments" and encoder["kind"] == "zscore":
                encoder["mean"] += 1
            if defect == "entity_cuts" and encoder["kind"] == "buckets" and '"entity"' in identity:
                encoder["boundaries"] = [cut + 0.125 for cut in encoder["boundaries"]]
        return state

    case = generate_batch(0, "diskgraph")[0]
    fitted_state(case, "diskgraph", 0)
    monkeypatch.setattr(FittedState, "to_dict", corrupt)
    with pytest.raises(PropertyFailure, match="fitted encoder inventory|fitted state differs"):
        fitted_state(case, "diskgraph", 0)


@pytest.mark.parametrize("defect", ["features", "edges"])
def test_matching_corrupt_graph_features_do_not_pass_format_comparison(
    tmp_path, monkeypatch, defect
):
    from kumo.pytorch.engines.diskgraph import DiskGraphSource
    from tests.test_diskgraph_backend import (
        test_diskgraph_temporal_sequences_aggregates_and_labels_match_fixture as check,
    )

    original = DiskGraphSource._graph_output

    def corrupt(self, *args, **kwargs):
        graph = original(self, *args, **kwargs)
        if defect == "edges":
            for edge in graph["edges"].values():
                edge["index"] = edge["index"][:, :0]
        else:
            for node in graph["nodes"].values():
                for field in node["features"].values():
                    field["values"].fill_(12345)
        return graph

    monkeypatch.setattr(DiskGraphSource, "_graph_output", corrupt)
    with pytest.raises(AssertionError):
        check(tmp_path)


@pytest.mark.parametrize(
    "defect", ["entity", "history_calendar", "history_timing", "tokens", "no_supervision"]
)
def test_pragma_fixture_rejects_corrupt_delivered_values(tmp_path, monkeypatch, defect):
    from kumo.pytorch.task import TaskBatches
    from tests.test_diskgraph_backend import (
        test_diskgraph_pragma_batches_match_fixture_without_record_adaptation as check,
    )

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    check(baseline)
    original = TaskBatches.__next__
    changed = []

    def corrupt(self):
        batch = original(self)
        if "masked_values" not in batch.targets:
            return batch
        changed.append(True)
        if defect == "no_supervision":
            batch.targets["masked_values"]["valid"].fill_(False)
            batch.targets["masked_values"]["values"].fill_(-100)
        elif defect == "tokens":
            batch.inputs["history"]["value_ids"].fill_(12345)
        elif defect == "entity":
            batch.inputs["entity"]["value_ids"].fill_(12345)
        else:
            for field in batch.inputs[defect].values():
                if isinstance(field, dict) and "values" in field:
                    field["values"].fill_(12345)
        return batch

    monkeypatch.setattr(TaskBatches, "__next__", corrupt)
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    with pytest.raises(AssertionError):
        check(damaged)
    assert changed
