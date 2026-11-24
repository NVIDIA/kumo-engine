# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed sampling options preserve native iterator behavior."""

from __future__ import annotations

import json
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
from diskgraph.runtime_batch_iterator import _weighted_seed_sampling_json


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
@pytest.mark.parametrize("is_temporal", [False, True])
def test_native_iterator_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None | InheritEngineSeed,
    is_temporal: bool,
) -> None:
    """Match the pre-options native call, including every optional default.

    Args:
        monkeypatch: Fixture replacing manifest access and the native index.
        seed: Inherited, explicitly disabled, zero, or nonzero shuffle seed.
        is_temporal: Whether the manifest enables seed timestamp anchoring.
    """
    index = RuntimeIndex("metadata.json", rnd_seed=17)
    monkeypatch.setattr(index, "_resolve", Mock())
    monkeypatch.setattr(index, "stage_size", Mock(return_value=3))
    index._tables = {
        "train": {"table_name": "seeds", "index_path": "/graph/train_tables/task"}
    }
    index._parent_root = "/graph"
    index._manifest = {"task": {"is_temporal": is_temporal}}
    native_index = Mock()
    iterator = native_index.iter.return_value
    iterator.num_samples.return_value = 3
    monkeypatch.setattr(index, "_index_for", Mock(return_value=native_index))
    options = SamplingOptions(stage="train", batch_size=2, rnd_seed=seed)
    result, count = index.iter(options)
    assert result is iterator and count == 3
    # Positional contract of RuntimeIndex.iter before SamplingOptions, using
    # an explicit empty plan and the single-worker, unshuffled epoch-zero case.
    native_index.iter.assert_called_once_with(
        "seeds",
        SamplingPlan(),
        1,
        0,
        False,
        17 if seed is InheritEngineSeed.DEFAULT else seed,
        0,
        2,
        False,
        None,
        True,
        None,
        None,
        None,
        False,
        True,
        False,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        is_temporal,
        None,
        False,
        None,
    )


def test_explicit_null_seed_cannot_shuffle() -> None:
    """An explicitly disabled seed is distinct from inheriting the engine seed."""
    index = RuntimeIndex("manifest-must-not-be-opened.json", rnd_seed=17)
    with pytest.raises(DiskgraphConfigError, match="rnd_seed is required"):
        index.iter(
            SamplingOptions(stage="train", batch_size=2, shuffle=True, rnd_seed=None)
        )


@pytest.mark.parametrize("use_sample_weights", [False, True])
@pytest.mark.parametrize("class_weight", [None, 1.0])
def test_weighted_seed_defaults_preserve_legacy_buckets(
    use_sample_weights: bool,
    class_weight: float | None,
) -> None:
    """Preserve unit class multipliers and the omitted-class sidecar choice.

    Args:
        use_sample_weights: Whether per-row weighted sidecars are requested.
        class_weight: An omitted or explicitly unit false-class multiplier.
    """
    ref = {
        "index_path": "/graph/train_tables/task",
        "binary_label_index": {
            "false_node_ids": "false.ids",
            "false_count": 3,
            "true_node_ids": "true.ids",
            "true_count": 2,
        },
        "sample_weight_index": {
            "all": {
                "node_ids": "all.ids",
                "cdf": "all.cdf",
                "count": 5,
                "total_weight": 9.0,
            },
            "false_bucket": {
                "node_ids": "false.ids",
                "cdf": "false.cdf",
                "count": 3,
                "total_weight": 7.0,
            },
            "true_bucket": {
                "node_ids": "true.ids",
                "cdf": "true.cdf",
                "count": 2,
                "total_weight": 2.0,
            },
        },
    }
    weights = WeightedSeedSampling(
        num_samples=5,
        false_weight=class_weight,
        use_sample_weights=use_sample_weights,
    )

    encoded = _weighted_seed_sampling_json(ref, "train", weights)

    assert encoded is not None
    payload = json.loads(encoded)
    assert payload["num_samples"] == 5
    buckets = payload["buckets"]
    if use_sample_weights:
        expected = (
            [("all", 5, 9.0)]
            if class_weight is None
            else [("false_bucket", 3, 7.0), ("true_bucket", 2, 2.0)]
        )
        assert all("cdf_path" in bucket for bucket in buckets)
    else:
        expected = [("false", 3, 3.0), ("true", 2, 2.0)]
        assert all("cdf_path" not in bucket for bucket in buckets)
    assert [
        (bucket["name"], bucket["count"], bucket["mass"]) for bucket in buckets
    ] == expected
