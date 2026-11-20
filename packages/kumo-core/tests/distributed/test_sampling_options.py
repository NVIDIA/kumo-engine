# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling configuration carries the native plan without serialization adapters."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, replace

import pytest
from diskgraph.sampling_plan import FanoutSpec, LayerSpec, SamplingPlan
from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchRequest,
    ConsumerToken,
    SamplingContractError,
    SamplingOptions,
    WeightedSeedSampling,
    WeightedSeedSamplingArguments,
)


def test_sampling_options_pass_the_native_plan_directly() -> None:
    """The immutable plan is shared directly while mutable keyword inputs are copied."""
    plan = SamplingPlan(
        [LayerSpec(default_fanout=FanoutSpec(4, "uniform", "reject_plan"))]
    )
    projections = {"users": ["age"]}
    selected = [3, 1, 3]
    weights: WeightedSeedSamplingArguments = {"num_samples": 3, "false_weight": 2.0}
    options = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        sampling_plan=plan,
        projections=projections,
        selected_stage_node_ids=selected,
        weighted_seed_sampling=weights,
    )
    projections["users"].append("name")
    selected.append(8)
    weights["num_samples"] = 20
    assert options.sampling_plan is plan
    assert options.to_native_kwargs()["sampling_plan"] is plan
    assert options.selected_stage_node_ids == (3, 1, 3)
    assert options.weighted_seed_sampling == WeightedSeedSampling(
        num_samples=3, false_weight=2.0
    )
    assert options.to_native_kwargs()["projections"] == {"users": ["age"]}


def test_frozen_options_survive_pickle_with_native_plan_equality() -> None:
    """Native plans retain their value identity after options cross a process boundary."""
    options = SamplingOptions(stage="train", batch_size=2)
    with pytest.raises(FrozenInstanceError):
        options.batch_size = 3  # type: ignore[misc]
    restored = pickle.loads(pickle.dumps(options))
    assert isinstance(restored, SamplingOptions)
    assert restored == options
    assert restored.to_native_kwargs() == options.to_native_kwargs()


def test_sampling_only_removes_schedule_without_copying_native_plan() -> None:
    """Trainer-owned selection is excluded without rebuilding the immutable plan."""
    original = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        num_workers=2,
        worker_id=1,
        epoch=3,
        shuffle=True,
        drop_last=True,
        group_by_timestamp=True,
        selected_stage_node_ids=[1, 3],
        weighted_seed_sampling={"num_samples": 4},
        projections={"users": ["age"]},
        rnd_seed=42,
    )
    sampling = original.sampling_only()
    assert original.has_seed_schedule and not sampling.has_seed_schedule
    assert sampling.sampling_only() is sampling
    assert sampling.sampling_plan is original.sampling_plan
    assert original.selected_stage_node_ids == (1, 3)
    assert sampling.projections == original.projections
    assert sampling.rnd_seed == 42


def test_omitted_class_weights_preserve_native_sample_weight_mode() -> None:
    """Omitted class weights must not require binary sidecars."""
    settings = WeightedSeedSampling(num_samples=5, use_sample_weights=True)
    assert settings.to_native_kwargs() == {"num_samples": 5, "use_sample_weights": True}
    assert replace(settings, false_weight=0.0).to_native_kwargs() == {
        "num_samples": 5,
        "use_sample_weights": True,
        "false_weight": 0.0,
    }


def test_request_compares_typed_options_and_rejects_wrong_plan() -> None:
    """Request identity includes structural options instead of a serialized plan digest."""
    options = SamplingOptions(stage="train", batch_size=2)
    request = BatchRequest(
        ArtifactIdentity("snapshot", "task", "mapping"),
        ConsumerToken("job", 0, "train", 0),
        0,
        0,
        1,
        0,
        (1, 2),
        options,
        42,
        1,
    )
    assert request.validate() is options
    restored = pickle.loads(pickle.dumps(request))
    assert restored == request
    assert (
        replace(
            request, options=replace(options, sampling_plan=SamplingPlan([LayerSpec()]))
        )
        != request
    )
    with pytest.raises(SamplingContractError, match="seed schedule"):
        replace(request, options=replace(options, num_workers=2)).validate()
    with pytest.raises(SamplingContractError, match="wire version"):
        replace(request, wire_version=1).validate()
