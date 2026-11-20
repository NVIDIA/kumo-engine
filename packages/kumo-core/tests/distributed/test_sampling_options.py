# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling configuration owns a plain plan dictionary without stale cache keys."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import pytest
from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchRequest,
    ConsumerToken,
    JsonValue,
    SamplingContractError,
    SamplingOptions,
    WeightedSeedSampling,
    WeightedSeedSamplingArguments,
)


def test_sampling_options_copy_nested_inputs_and_exports() -> None:
    """Copy caller inputs once and expose the normal owned plan to the native adapter."""
    fanout: dict[str, JsonValue] = {"fanout": 4, "mode": "uniform"}
    layers: list[JsonValue] = [{"default_fanout": fanout}]
    projections = {"users": ["age"]}
    selected = [3, 1, 3]
    weights: WeightedSeedSamplingArguments = {"num_samples": 3, "false_weight": 2.0}
    options = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        sampling_plan={"layers": layers},
        projections=projections,
        selected_stage_node_ids=selected,
        weighted_seed_sampling=weights,
    )
    digest = options.digest
    fanout["fanout"] = 99
    layers.append(None)
    projections["users"].append("name")
    selected.append(8)
    weights["num_samples"] = 20
    native = options.to_native_kwargs()
    assert native["sampling_plan"] == {
        "layers": [{"default_fanout": {"fanout": 4, "mode": "uniform"}}]
    }
    assert options.selected_stage_node_ids == (3, 1, 3)
    assert options.weighted_seed_sampling == WeightedSeedSampling(
        num_samples=3, false_weight=2.0
    )
    assert native["projections"] == {"users": ["age"]}
    native_projection = native["projections"]
    native_selection = native["selected_stage_node_ids"]
    assert native_projection is not None and native_selection is not None
    native_projection["users"].append("external mutation")
    native_selection.clear()
    assert options.to_native_kwargs()["projections"] == {"users": ["age"]}
    assert options.to_native_kwargs()["selected_stage_node_ids"] == [3, 1, 3]
    assert options.digest == digest


def test_dataclasses_are_frozen_and_pickle_roundtrip_preserves_identity() -> None:
    """Configuration fields and scalar weights are frozen; plans remain ordinary dicts."""
    options = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        sampling_plan={"layers": []},
        weighted_seed_sampling={"num_samples": 4, "use_sample_weights": True},
    )
    assert isinstance(options.sampling_plan, dict)
    assert options.weighted_seed_sampling is not None
    for obj, name, value in (
        (options, "batch_size", 3),
        (options.weighted_seed_sampling, "num_samples", 10),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, value)
    restored = pickle.loads(pickle.dumps(options))
    assert isinstance(restored, SamplingOptions)
    assert restored == options
    assert restored.digest == options.digest
    assert restored.to_native_kwargs() == options.to_native_kwargs()


def test_canonical_identity_ignores_mapping_order_and_tracks_plan_edits() -> None:
    """Equivalent inputs have equal keys, and editing the owned plan changes its key."""
    first = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        sampling_plan={"layers": [], "version": 1},
        projections={"users": ["age"], "orders": ["price"]},
    )
    second = SamplingOptions.from_kwargs(
        stage="train",
        batch_size=2,
        sampling_plan={"version": 1, "layers": []},
        projections={"orders": ["price"], "users": ["age"]},
    )
    assert first == second
    expected = first.digest
    assert second.digest == expected
    first.sampling_plan["version"] = 2
    assert first.digest != expected
    assert second.digest == expected


def test_sampling_only_removes_schedule_without_mutating_original() -> None:
    """Trainer selection/sharding is excluded from the options sent to an actor."""
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
    assert original.has_seed_schedule
    assert not sampling.has_seed_schedule
    assert sampling.sampling_only() is sampling
    assert original.selected_stage_node_ids == (1, 3)
    assert sampling.projections == original.projections
    assert sampling.rnd_seed == 42
    assert sampling.digest != original.digest


def test_omitted_class_weights_preserve_native_sample_weight_mode() -> None:
    """Default class weights must not make native scheduling require binary sidecars."""
    settings = WeightedSeedSampling(num_samples=5, use_sample_weights=True)
    assert settings.to_native_kwargs() == {"num_samples": 5, "use_sample_weights": True}
    assert replace(settings, false_weight=0.0).to_native_kwargs() == {
        "num_samples": 5,
        "use_sample_weights": True,
        "false_weight": 0.0,
    }


def test_request_keeps_typed_options_and_validates_without_json_decoding() -> None:
    """The request transports the frozen object; serialization is only an identity detail."""
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
    with patch(
        "kumo_core.diskgraph.batch_contract.orjson.loads",
        side_effect=AssertionError("typed options must not decode JSON"),
    ):
        assert request.validate() is options
        assert request.sampling_plan_digest == options.digest
    with pytest.raises(SamplingContractError, match="seed schedule"):
        replace(request, options=replace(options, num_workers=2)).validate()
    with pytest.raises(SamplingContractError, match="wire version"):
        replace(request, wire_version=1).validate()


def test_trainer_selection_is_not_serialized_for_an_unused_digest() -> None:
    """Metadata-only scheduling should not serialize a potentially large ID selection."""
    with patch(
        "kumo_core.diskgraph.batch_contract.canonical_json",
        side_effect=AssertionError("trainer schedule does not need a cache key"),
    ):
        options = SamplingOptions.from_kwargs(
            stage="train", batch_size=2, selected_stage_node_ids=[1, 2, 3]
        )
        assert options.selected_stage_node_ids == (1, 2, 3)
    request = BatchRequest(
        ArtifactIdentity("snapshot", "task", "mapping"),
        ConsumerToken("job", 0, "train", 0),
        0,
        0,
        1,
        0,
        (1, 2),
        options.sampling_only(),
        42,
        1,
    )
    expected = request.sampling_plan_digest
    restored = pickle.loads(pickle.dumps(request))
    assert isinstance(restored, BatchRequest)
    assert restored.sampling_plan_digest == expected
    request.options.sampling_plan["version"] = 2
    assert request.sampling_plan_digest != expected
    assert restored.sampling_plan_digest == expected


def test_direct_configuration_construction_copies_the_plan() -> None:
    """The ordinary dataclass constructor takes ownership without a wrapper API."""
    plan: dict[str, JsonValue] = {"layers": [], "metadata": {"label": "original"}}
    options = SamplingOptions(stage="train", batch_size=2, sampling_plan=plan)
    plan["metadata"] = "caller edit"
    assert options.sampling_plan == {"layers": [], "metadata": {"label": "original"}}
    assert options.to_native_kwargs()["sampling_plan"] is options.sampling_plan
