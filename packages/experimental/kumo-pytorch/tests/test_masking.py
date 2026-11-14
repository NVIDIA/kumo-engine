# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from kumo.pytorch.config import MaskingConfig
from kumo.pytorch.data.masking import mask_event_values


def _run_mask(
    keys: torch.Tensor,
    masking: MaskingConfig,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = keys.to(device)
    values = torch.arange(keys.numel(), device=device).reshape_as(keys) + 100
    return mask_event_values(
        key_ids=keys,
        value_ids=values,
        valid=torch.ones_like(keys, dtype=torch.bool),
        event_owner=torch.zeros(keys.shape[0], dtype=torch.long, device=device),
        event_history_position=torch.arange(keys.shape[0], device=device),
        entity_ids=torch.tensor([1], device=device),
        masking=masking,
        seed=1729,
        epoch=0,
        mask_id=2,
        unknown_id=1,
    )


def test_masking_draws_vary_at_the_requested_grain() -> None:
    token_keys = torch.arange(10, 26).repeat(4_096, 1)
    _inputs, _targets, token_loss = _run_mask(
        token_keys,
        MaskingConfig(0.15, 0.0, 0.0, 0.0),
    )
    assert 0.14 < token_loss.float().mean() < 0.16

    event_keys = torch.arange(10, 14).repeat(4_096, 1)
    _inputs, _targets, event_loss = _run_mask(
        event_keys,
        MaskingConfig(0.0, 0.10, 0.0, 0.0),
    )
    selected_per_event = event_loss.sum(dim=1)
    assert ((selected_per_event == 0) | (selected_per_event == event_keys.shape[1])).all()
    assert 0.085 < event_loss.any(dim=1).float().mean() < 0.115

    semantic_keys = torch.arange(10, 266).repeat(32, 1)
    _inputs, _targets, semantic_loss = _run_mask(
        semantic_keys,
        MaskingConfig(0.0, 0.0, 0.10, 0.0),
    )
    assert (semantic_loss.all(dim=0) | ~semantic_loss.any(dim=0)).all()
    assert 0.075 < semantic_loss.any(dim=0).float().mean() < 0.125


def test_unknown_draws_vary_within_selected_positions() -> None:
    keys = torch.arange(10, 26).repeat(2_048, 1)
    inputs, targets, loss = _run_mask(
        keys,
        MaskingConfig(1.0, 0.0, 0.0, 0.10),
    )
    unknown = inputs == 1
    assert 0.08 < unknown.float().mean() < 0.12
    assert (unknown | loss).all()
    assert (targets[unknown] == -100).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_masking_is_bitwise_equal_on_cpu_and_cuda() -> None:
    keys = torch.arange(10, 26).repeat(256, 1)
    masking = MaskingConfig(0.15, 0.10, 0.10, 0.10)
    cpu = _run_mask(keys, masking)
    cuda = tuple(value.cpu() for value in _run_mask(keys, masking, device="cuda"))
    for cpu_value, cuda_value in zip(cpu, cuda, strict=True):
        assert torch.equal(cpu_value, cuda_value)


def test_semantic_key_masking_is_history_wide() -> None:
    keys = torch.tensor([[10, 20], [10, 30], [20, 30]])
    values = torch.arange(6).reshape(3, 2) + 100
    valid = torch.ones_like(keys, dtype=torch.bool)
    inputs, targets, loss = mask_event_values(
        key_ids=keys,
        value_ids=values,
        valid=valid,
        event_owner=torch.zeros(3, dtype=torch.long),
        event_history_position=torch.arange(3),
        entity_ids=torch.tensor([7]),
        masking=MaskingConfig(0.0, 0.0, 1.0, 0.0),
        seed=11,
        epoch=2,
        mask_id=2,
        unknown_id=1,
    )
    assert loss.all()
    assert torch.equal(inputs, torch.full_like(inputs, 2))
    assert torch.equal(targets, values)


def test_unknown_corruption_still_leaves_one_prediction() -> None:
    keys = torch.tensor([[10, 20]])
    values = torch.tensor([[100, 200]])
    inputs, targets, loss = mask_event_values(
        key_ids=keys,
        value_ids=values,
        valid=torch.ones_like(keys, dtype=torch.bool),
        event_owner=torch.zeros(1, dtype=torch.long),
        event_history_position=torch.zeros(1, dtype=torch.long),
        entity_ids=torch.tensor([7]),
        masking=MaskingConfig(1.0, 0.0, 0.0, 1.0),
        seed=11,
        epoch=0,
        mask_id=2,
        unknown_id=1,
    )
    assert loss.sum() == 1
    assert (inputs == 1).sum() == 1
    assert targets[loss].item() in values.tolist()[0]


def test_masking_policy_can_allow_zero_targets() -> None:
    keys = torch.tensor([[10, 20]])
    inputs, targets, loss = _run_mask(
        keys,
        MaskingConfig(
            0.0,
            0.0,
            0.0,
            0.0,
            ensure_at_least_one=False,
        ),
    )

    assert not loss.any()
    assert (targets == -100).all()
    assert (inputs != 2).all()


def test_unknown_replacement_can_remain_supervised() -> None:
    keys = torch.tensor([[10, 20]])
    inputs, targets, loss = _run_mask(
        keys,
        MaskingConfig(
            1.0,
            0.0,
            0.0,
            1.0,
            supervise_unknown=True,
        ),
    )

    assert loss.all()
    assert (inputs == 1).all()
    assert (targets != -100).all()


def test_mask_selection_composition_is_explicit() -> None:
    keys = torch.tensor([[10, 20]])
    _inputs, _targets, loss = _run_mask(
        keys,
        MaskingConfig(
            1.0,
            0.0,
            1.0,
            0.0,
            combine="intersection",
            ensure_at_least_one=False,
        ),
    )

    assert not loss.any()


@pytest.mark.parametrize("ensure", [False, True])
@pytest.mark.parametrize("supervise_unknown", [False, True])
@pytest.mark.parametrize("selected_probability", [0.0, 1.0])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_masking_guarantee_handles_interleaved_and_empty_owners(
    ensure: bool,
    supervise_unknown: bool,
    selected_probability: float,
    noncontiguous: bool,
) -> None:
    keys = torch.tensor([[10, 20], [10, 20], [10, 20], [10, 20]])
    values = torch.arange(8).reshape(4, 2) + 100
    valid = torch.tensor([[False, True], [True, True], [True, True], [False, False]])
    if noncontiguous:
        keys, values, valid = (value.T.contiguous().T for value in (keys, values, valid))
    inputs, targets, loss = mask_event_values(
        key_ids=keys,
        value_ids=values,
        valid=valid,
        event_owner=torch.tensor([2, 0, 2, 3]),
        event_history_position=torch.tensor([1, 0, 0, 0]),
        entity_ids=torch.tensor([7, 8, 9, 10]),
        masking=MaskingConfig(
            selected_probability,
            0.0,
            0.0,
            1.0,
            ensure_at_least_one=ensure,
            supervise_unknown=supervise_unknown,
        ),
        seed=11,
        epoch=2,
        mask_id=2,
        unknown_id=1,
    )
    expected_inputs = values.clone()
    expected_loss = torch.zeros_like(valid)
    if selected_probability:
        expected_inputs[valid] = 1
        if supervise_unknown:
            expected_loss = valid.clone()
    if ensure and (not selected_probability or not supervise_unknown):
        # First valid token in tensor order, including when history positions differ.
        expected_inputs[0, 1] = expected_inputs[1, 0] = 2
        expected_loss[0, 1] = expected_loss[1, 0] = True
    expected_targets = torch.full_like(values, -100)
    expected_targets[expected_loss] = values[expected_loss]
    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)
    assert torch.equal(loss, expected_loss)
