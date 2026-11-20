# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed sampling plans preserve the native schema and runtime behavior."""

from __future__ import annotations

import multiprocessing
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import TypedDict

import diskgraph
import pytest
from diskgraph import DiskgraphConfigError, RuntimeIndex
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
    """Public exports refer to native classes with immutable empty defaults."""
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
        assert cls.__module__ == "diskgraph._c"
        assert getattr(diskgraph._c, cls.__name__) is cls
        assert getattr(diskgraph, cls.__name__) is cls
        assert cls.__name__ in diskgraph.__all__
    assert SamplingPlan().layers == ()
    assert SamplingPlan([LayerSpec()]).layers == (LayerSpec(),)


def test_native_plans_copy_inputs_and_expose_immutable_values() -> None:
    """Native plans own their inputs and reject misleading in-place mutations."""
    spec = FanoutSpec(4, "uniform", "reject_plan")
    overrides = {"edge": spec}
    layer = LayerSpec(fanouts=overrides)
    layers = [layer]
    plan = SamplingPlan(layers)

    overrides.clear()
    layers.clear()

    assert plan.layers == (layer,)
    assert layer.fanouts == {"edge": spec}
    assert LayerSpec(layer.fanouts) == layer
    with pytest.raises(AttributeError):
        plan.layers.append(layer)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        layer.fanouts["edge"] = spec  # type: ignore[index]
    with pytest.raises(AttributeError):
        spec.fanout = 8  # type: ignore[misc]


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize(
    "value",
    [
        ExponentialDecay(0.125),
        LinearDecay(3600),
        CustomExpression("score"),
        WeightedSampling(ExponentialDecay(0.125)),
        WeightedSampling(LinearDecay(3600), temporal=True),
        WeightedSampling(CustomExpression("score")),
        FanoutSpec(4, "uniform", "reject_plan"),
        LayerSpec(),
        SamplingPlan(),
        _plan("uniform"),
        _plan(WeightedSampling(ExponentialDecay(0.125), temporal=True)),
    ],
)
def test_native_values_survive_pickle(value: object, protocol: int) -> None:
    """Every native value reconstructs its Rust state through pickle arguments."""
    restored = pickle.loads(pickle.dumps(value, protocol=protocol))

    assert type(restored) is type(value)
    assert restored == value


def _worker_plan(plan: SamplingPlan) -> SamplingPlan:
    return plan


def test_native_plans_survive_spawned_worker_round_trip() -> None:
    """Spawned workers can receive and return a complete native weighted plan."""
    plan = _plan(WeightedSampling(CustomExpression("score"), temporal=True))

    with multiprocessing.get_context("spawn").Pool(1) as workers:
        restored = workers.apply(_worker_plan, (plan,))

    assert restored == plan


@pytest.mark.parametrize(
    "invalid_plan",
    [
        {"layers": []},
        {},
        '{"layers":[]}',
        None,
        [],
        1,
        object(),
        SimpleNamespace(layers=[]),
    ],
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

    with pytest.raises(TypeError, match="SamplingPlan"):
        diskgraph.RuntimeBatchIterator(
            parent_root="index-must-not-be-opened",
            tt_id="unused",
            stage_table="train",
            sampling_plan=invalid_plan,
            num_workers=1,
            worker_id=0,
            shuffle=False,
            rnd_seed=None,
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
    assert plan.layers
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


@pytest.mark.parametrize("fanout", [-1, 2**32])
def test_native_validation_rejects_invalid_fanout(fanout: int) -> None:
    """Invalid fanouts fail when constructing the native value."""
    with pytest.raises(DiskgraphConfigError, match="invalid fanout"):
        FanoutSpec(fanout, "uniform", "reject_plan")


def test_native_iterators_keep_values_when_the_plan_is_replaced(
    seed_table_fixture: SeedTableFixture,  # noqa: F811 -- imported pytest fixture
) -> None:
    """Each iterator owns the native values copied when it was constructed."""
    index = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=1
    )
    plan = _plan("uniform")
    before, _ = index.iter(
        stage="train",
        sampling_plan=plan,
        num_workers=1,
        worker_id=0,
        shuffle=False,
        epoch=0,
        batch_size=2,
        entity_node_type="users",
    )
    plan = SamplingPlan()
    after, _ = index.iter(
        stage="train",
        sampling_plan=plan,
        num_workers=1,
        worker_id=0,
        shuffle=False,
        epoch=0,
        batch_size=2,
        entity_node_type="users",
    )

    before_nodes = next(before)["graphs"]["main"]["node_ids"]
    after_nodes = next(after)["graphs"]["main"]["node_ids"]
    assert len(before_nodes["orders"]) > 0
    assert "orders" not in after_nodes or len(after_nodes["orders"]) == 0


@pytest.mark.parametrize(
    "mode", ["uniform", "deterministic_uniform", "temporal_most_recent"]
)
def test_native_fanout_properties(mode: SamplingMode) -> None:
    """Native fanouts expose the count, mode, and policy passed to construction."""
    spec = FanoutSpec(7, mode, "uniform_fallback")

    assert spec.fanout == 7
    assert spec.mode == mode
    assert spec.absent_key_policy == "uniform_fallback"


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize(
    "weighting", [ExponentialDecay(0.125), LinearDecay(3600), CustomExpression("score")]
)
def test_native_weighting_properties(weighting: Weighting, temporal: bool) -> None:
    """Native enum wrappers preserve every weighting variant and temporal flag."""
    mode = WeightedSampling(weighting, temporal=temporal)

    assert mode.weighting == weighting
    assert mode.temporal is temporal
    assert FanoutSpec(3, mode, "reject_plan").mode == mode


@pytest.mark.parametrize("invalid", [{}, object(), None])
def test_native_constructors_reject_untyped_nested_values(invalid: object) -> None:
    """Native constructors require native layers, fanouts, and weighting objects."""
    with pytest.raises(TypeError):
        SamplingPlan([invalid])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        LayerSpec({"edge": invalid})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        WeightedSampling(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mode, policy", [("unknown", "reject_plan"), ("uniform", "unknown")]
)
def test_native_constructors_reject_invalid_modes_and_policies(
    mode: str, policy: str
) -> None:
    """Invalid native enum names produce configuration errors at construction."""
    with pytest.raises(DiskgraphConfigError, match="invalid"):
        FanoutSpec(1, mode, policy)  # type: ignore[arg-type]
