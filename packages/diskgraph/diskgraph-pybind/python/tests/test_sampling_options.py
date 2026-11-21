# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed sampling options preserve native iterator behavior."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest
from diskgraph import (
    DiskgraphConfigError,
    InheritEngineSeed,
    RuntimeIndex,
    SamplingOptions,
    SamplingPlan,
    WeightedSeedSampling,
)


def test_sampling_options_defaults_and_pickle() -> None:
    """Options are frozen, pickleable and keep the native traversal plan typed."""
    options = SamplingOptions(stage="train", batch_size=8)
    assert options.sampling_plan == SamplingPlan()
    assert options.rnd_seed is InheritEngineSeed.DEFAULT
    assert options.num_workers == 1 and options.worker_id == 0
    assert options.epoch == 0 and not options.shuffle
    assert pickle.loads(pickle.dumps(options)) == options
    with pytest.raises(FrozenInstanceError):
        options.batch_size = 16  # type: ignore[misc]
    assert replace(options, epoch=2).epoch == 2
    assert options.epoch == 0


def test_weighted_sampling_retains_omitted_class_weights() -> None:
    """Per-row weighting does not implicitly request binary-class sidecars."""
    weights = WeightedSeedSampling(num_samples=5, use_sample_weights=True)
    assert weights.false_weight is None and weights.true_weight is None
    assert pickle.loads(pickle.dumps(weights)) == weights
    assert replace(weights, false_weight=0.0).false_weight == 0.0
    with pytest.raises(FrozenInstanceError):
        weights.num_samples = 7  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [None, {}, "{}", b"{}"])
def test_untyped_options_rejected_before_opening_manifest(invalid: object) -> None:
    """Legacy dictionaries and JSON cannot accidentally open a graph."""
    index = RuntimeIndex("manifest-must-not-be-opened.json")
    with pytest.raises(TypeError, match="diskgraph.SamplingOptions"):
        index.iter(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [InheritEngineSeed.DEFAULT, None, 0, 91])
def test_native_iterator_seed_inheritance(
    monkeypatch: pytest.MonkeyPatch, seed: int | None | InheritEngineSeed
) -> None:
    """Omitted seed inherits the index seed; explicit None and zero are retained."""
    index = RuntimeIndex("metadata.json", rnd_seed=17)
    monkeypatch.setattr(index, "_resolve", Mock())
    monkeypatch.setattr(index, "stage_size", Mock(return_value=3))
    index._tables = {
        "train": {"table_name": "seeds", "index_path": "/graph/train_tables/task"}
    }
    index._parent_root = "/graph"
    index._manifest = {"task": {"is_temporal": False}}
    native_index = Mock()
    iterator = native_index.iter.return_value
    iterator.num_samples.return_value = 3
    monkeypatch.setattr(index, "_index_for", Mock(return_value=native_index))
    options = SamplingOptions(stage="train", batch_size=2, rnd_seed=seed)
    result, count = index.iter(options)
    assert result is iterator and count == 3
    assert native_index.iter.call_args.args[5] == (
        17 if seed is InheritEngineSeed.DEFAULT else seed
    )
    assert native_index.iter.call_args.args[1] is options.sampling_plan


def test_explicit_null_seed_cannot_shuffle() -> None:
    """An explicitly disabled seed is distinct from inheriting the engine seed."""
    index = RuntimeIndex("manifest-must-not-be-opened.json", rnd_seed=17)
    with pytest.raises(DiskgraphConfigError, match="rnd_seed is required"):
        index.iter(
            SamplingOptions(stage="train", batch_size=2, shuffle=True, rnd_seed=None)
        )
