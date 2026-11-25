# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_fm_benchmark._pytorch import compatibility_issues, contract_for
from kumo_fm_benchmark.loader import _select


def test_unsupported_sampling_is_explicit():
    bundle = SimpleNamespace(
        tables={"events": {"time_column": None}},
        endpoint=lambda step: ("users", "events"))
    workload = SimpleNamespace(
        bundle=bundle, selection="native_multihop",
        layers=[[{"relation": "event_user", "direction": "reverse", "fanout": 3}]])
    assert "uniform sampling" in compatibility_issues(workload)[0]
    with pytest.raises(ValueError, match="Cannot preserve FM sampling"):
        contract_for(workload)
    workload.layers = [[{"relation": "event_user", "direction": "reverse", "fanout": 0}]]
    assert "empty graph layer" in compatibility_issues(workload)[0]
    workload.selection = "temporal_query"
    assert "native_multihop" in compatibility_issues(workload)[0]


def test_anchor_selection_keeps_order_and_rejects_unknown_ids():
    available = [{"instance_id": 1}, {"instance_id": 2}]
    assert _select(available, [2, 1], "predict") == [available[1], available[0]]
    with pytest.raises(ValueError, match="duplicate or unknown"):
        _select(available, [1, 1], "predict")
    with pytest.raises(ValueError, match="duplicate or unknown"):
        _select(available, [3], "predict")
