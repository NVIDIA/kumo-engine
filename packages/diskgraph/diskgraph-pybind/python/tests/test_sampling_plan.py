# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed sampling plans preserve the native schema and runtime behavior."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from dataclasses import is_dataclass
from typing import TypedDict

import diskgraph
import pytest
from diskgraph import DiskgraphConfigError, RuntimeIndex
from diskgraph.runtime_batch_iterator import _sampling_plan_json
from diskgraph.sampling_plan import (
    CustomExpression,
    ExponentialDecay,
    FanoutSpec,
    LayerSpec,
    LinearDecay,
    SamplingMode,
    SamplingPlan,
    WeightedSampling,
    Weighting,
)
from test_runtime_batch_iterator_e2e import (
    seed_table_fixture,  # noqa: F401 -- shared native ingest fixture
)


class SeedTableFixture(TypedDict):
    """Paths and counts provided by the shared native ingest fixture."""

    manifest_path: str
    parent_root: str
    n_users: int
    n_seeds: int


def _plan(mode: SamplingMode) -> SamplingPlan:
    return SamplingPlan(
        [
            LayerSpec(default_fanout=FanoutSpec(4, mode, "uniform_fallback")),
            LayerSpec(
                fanouts={"users__orders": FanoutSpec(0, "uniform", "reject_plan")},
                default_fanout=FanoutSpec(2, mode, "uniform_fallback"),
            ),
        ]
    )


def test_empty_plan_defaults_and_public_exports() -> None:
    """Public dataclasses construct without mutable defaults shared between plans."""
    classes = (
        SamplingPlan,
        LayerSpec,
        FanoutSpec,
        WeightedSampling,
        ExponentialDecay,
        LinearDecay,
        CustomExpression,
    )
    for cls in classes:
        assert is_dataclass(cls)
        assert getattr(diskgraph, cls.__name__) is cls
        assert cls.__name__ in diskgraph.__all__
    first, second = SamplingPlan(), SamplingPlan()
    first.layers.append(LayerSpec())
    assert second.layers == []
    first_layer, second_layer = LayerSpec(), LayerSpec()
    first_layer.fanouts["relation"] = FanoutSpec(0, "uniform", "reject_plan")
    assert second_layer.fanouts == {}
    assert SamplingPlan().to_dict() == {"layers": []}
    assert SamplingPlan([LayerSpec()]).to_dict() == {
        "layers": [{"fanouts": {}, "default_fanout": None}]
    }


@pytest.mark.parametrize(
    "mode", ["uniform", "deterministic_uniform", "temporal_most_recent"]
)
def test_basic_modes_and_relation_overrides_keep_native_shape(
    mode: SamplingMode,
) -> None:
    """Hop order, relation names, defaults, and missing-key policies are explicit."""
    plan = _plan(mode)
    expected = {
        "layers": [
            {
                "fanouts": {},
                "default_fanout": {
                    "fanout": 4,
                    "mode": mode,
                    "absent_key_policy": "uniform_fallback",
                },
            },
            {
                "fanouts": {
                    "users__orders": {
                        "fanout": 0,
                        "mode": "uniform",
                        "absent_key_policy": "reject_plan",
                    }
                },
                "default_fanout": {
                    "fanout": 2,
                    "mode": mode,
                    "absent_key_policy": "uniform_fallback",
                },
            },
        ]
    }
    assert plan.to_dict() == expected
    assert json.loads(_sampling_plan_json(plan)) == expected


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize(
    "weighting, expected",
    [
        (ExponentialDecay(0.125), {"exponential_decay": {"lambda": 0.125}}),
        (LinearDecay(3600), {"linear_decay": {"half_life_seconds": 3600}}),
        (CustomExpression("score"), {"custom_expression": {"id": "score"}}),
    ],
)
def test_weighted_modes_keep_rust_variant_tags(
    temporal: bool,
    weighting: Weighting,
    expected: dict[str, dict[str, float | int | str]],
) -> None:
    """Both weighted modes preserve all three native weighting representations."""
    mode = WeightedSampling(weighting, temporal=temporal)
    kind = "temporal_weighted" if temporal else "weighted"
    assert SamplingPlan(
        [LayerSpec(default_fanout=FanoutSpec(3, mode, "reject_plan"))]
    ).to_dict() == {
        "layers": [
            {
                "fanouts": {},
                "default_fanout": {
                    "fanout": 3,
                    "mode": {kind: {"weighting": expected}},
                    "absent_key_policy": "reject_plan",
                },
            }
        ]
    }


def test_plan_remains_editable_and_export_is_independent() -> None:
    """Ordinary dataclass/list/dict edits affect future exports without stale state."""
    plan = _plan("uniform")
    exported = plan.to_dict()
    default = plan.layers[0].default_fanout
    assert default is not None
    default.fanout = 8
    plan.layers[1].fanouts.clear()
    assert plan.to_dict() != exported
    restored = pickle.loads(pickle.dumps(plan))
    assert isinstance(restored, SamplingPlan)
    assert restored == plan
    assert restored.to_dict() == plan.to_dict()


@pytest.mark.parametrize(
    "invalid_plan", [{"layers": []}, {}, '{"layers":[]}', None, [], 1, object()]
)
def test_iter_rejects_untyped_plans_before_accessing_the_manifest(
    invalid_plan: object,
) -> None:
    """Reject all legacy input forms before opening manifests or native indexes."""
    index = RuntimeIndex("manifest-must-not-be-opened.json")
    with pytest.raises(
        TypeError, match="sampling_plan must be a diskgraph.SamplingPlan"
    ):
        index.iter(
            stage="train",
            sampling_plan=invalid_plan,  # type: ignore[arg-type]
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=2,
        )


def test_public_plan_construction_does_not_import_training_or_open_graphs() -> None:
    """Plan construction loads neither graph data nor the Torch/training stack."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import diskgraph
from unittest.mock import patch
with patch.object(
    diskgraph, "SamplerEngine", side_effect=AssertionError("graph opened")
):
    from diskgraph import SamplingPlan, LayerSpec, FanoutSpec
    plan = SamplingPlan([
        LayerSpec(default_fanout=FanoutSpec(4, "uniform", "reject_plan"))
    ])
    assert plan.to_dict()["layers"]
assert "diskgraph.runtime_batch_iterator" not in sys.modules
assert "torch" not in sys.modules
assert "pytorch_lightning" not in sys.modules
assert "kumo_core" not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "uniform",
        "deterministic_uniform",
        "temporal_most_recent",
        WeightedSampling(ExponentialDecay(0.125)),
        WeightedSampling(LinearDecay(3600)),
        WeightedSampling(ExponentialDecay(0.125), temporal=True),
        WeightedSampling(LinearDecay(3600), temporal=True),
        WeightedSampling(CustomExpression("score")),
        WeightedSampling(CustomExpression("score"), temporal=True),
    ],
)
def test_typed_plans_produce_native_batches_and_unsupported_mode_errors(
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
    mode: SamplingMode,
) -> None:
    """Typed plans execute supported modes and preserve native unsupported errors."""
    plan = _plan(mode)
    index = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    typed, typed_count = index.iter(
        stage="train",
        entity_node_type="users",
        sampling_plan=plan,
        num_workers=1,
        worker_id=0,
        shuffle=False,
        epoch=0,
        batch_size=2,
    )
    assert typed_count == seed_table_fixture["n_seeds"]
    if isinstance(mode, WeightedSampling):
        with pytest.raises(DiskgraphConfigError, match="not supported"):
            next(typed)
    else:
        batches = list(typed)
        assert sum(len(batch["task"]["input_ids"]) for batch in batches) == typed_count
        assert all("main" in batch["graphs"] for batch in batches)


def test_native_validation_rejects_invalid_dataclass_fanout(
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
) -> None:
    """Native validation remains authoritative for invalid sampling requests."""
    plan = SamplingPlan(
        [LayerSpec(default_fanout=FanoutSpec(-1, "uniform", "reject_plan"))]
    )
    index = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    with pytest.raises(DiskgraphConfigError, match="invalid value"):
        index.iter(
            stage="train",
            sampling_plan=plan,
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=2,
        )
