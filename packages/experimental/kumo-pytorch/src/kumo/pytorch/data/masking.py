# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deterministic masking for Python and engine-backed batches."""

from __future__ import annotations

import torch

from kumo.pytorch.config import MaskingConfig

_UINT32_MODULUS = 4_294_967_296
_UINT32_MASK = _UINT32_MODULUS - 1
_HASH_MULTIPLIER = 0x45D9F3B
_COORDINATE_SALT = 0x9E3779B9


def _hash32(value: torch.Tensor) -> torch.Tensor:
    """Spread changes in an integer across a portable 32-bit domain."""

    value = torch.bitwise_and(value, _UINT32_MASK)
    value = torch.bitwise_and(
        torch.bitwise_xor(value, torch.bitwise_right_shift(value, 16)) * _HASH_MULTIPLIER,
        _UINT32_MASK,
    )
    value = torch.bitwise_and(
        torch.bitwise_xor(value, torch.bitwise_right_shift(value, 16)) * _HASH_MULTIPLIER,
        _UINT32_MASK,
    )
    return torch.bitwise_and(
        torch.bitwise_xor(value, torch.bitwise_right_shift(value, 16)),
        _UINT32_MASK,
    )


def _uniform(
    reference: torch.Tensor,
    *coordinates: torch.Tensor,
    seed: int,
    epoch: int,
    domain: int,
) -> torch.Tensor:
    """Return reproducible values in [0, 1) using only portable integer math."""

    state = _hash32(
        torch.full_like(
            reference,
            (seed + epoch * 1_000_003 + domain * 97_409) % _UINT32_MODULUS,
            dtype=torch.long,
        )
    )
    for index, coordinate in enumerate(coordinates, start=1):
        mixed = torch.remainder(
            coordinate.to(torch.long) + index * _COORDINATE_SALT,
            _UINT32_MODULUS,
        )
        state = _hash32(torch.bitwise_xor(state, _hash32(mixed)))
    return state.to(torch.float64) / _UINT32_MODULUS


def _below(probability: float, draws: torch.Tensor) -> torch.Tensor:
    if probability <= 0.0:
        return torch.zeros_like(draws, dtype=torch.bool)
    if probability >= 1.0:
        return torch.ones_like(draws, dtype=torch.bool)
    return draws < probability


def mask_event_values(
    *,
    key_ids: torch.Tensor,
    value_ids: torch.Tensor,
    valid: torch.Tensor,
    event_owner: torch.Tensor,
    event_history_position: torch.Tensor,
    entity_ids: torch.Tensor,
    masking: MaskingConfig,
    seed: int,
    epoch: int,
    mask_id: int,
    unknown_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt event values while keeping semantic-key decisions history-wide.

    A key decision depends on the example's entity ID and semantic key ID, but not
    on event identity. Every occurrence of that key in the history therefore
    receives the same key-level decision. Event and token decisions use their
    corresponding coordinates. The function guarantees at least one predicted
    token for every non-empty history.
    """

    if key_ids.shape != value_ids.shape or key_ids.shape != valid.shape:
        raise ValueError("key IDs, value IDs, and validity must have identical shapes")
    if key_ids.ndim != 2:
        raise ValueError("event token tensors must have shape [events, tokens]")
    if event_owner.shape != (key_ids.shape[0],):
        raise ValueError("event_owner must contain one row index per event")
    if event_history_position.shape != event_owner.shape:
        raise ValueError("event_history_position must contain one position per event")

    token_width = key_ids.shape[1]
    owners = event_owner[:, None].expand_as(key_ids)
    owner_entities = entity_ids[event_owner][:, None].expand_as(key_ids)
    event_positions = event_history_position[:, None].expand_as(key_ids)
    token_positions = torch.arange(token_width, device=key_ids.device, dtype=torch.long)[
        None, :
    ].expand_as(key_ids)

    token_selected = _below(
        masking.token_probability,
        _uniform(
            key_ids,
            owner_entities,
            event_positions,
            token_positions,
            seed=seed,
            epoch=epoch,
            domain=1,
        ),
    )
    event_selected = _below(
        masking.event_probability,
        _uniform(
            key_ids,
            owner_entities,
            event_positions,
            torch.zeros_like(key_ids),
            seed=seed,
            epoch=epoch,
            domain=2,
        ),
    )
    key_selected = _below(
        masking.semantic_type_probability,
        _uniform(
            key_ids,
            owner_entities,
            key_ids,
            torch.zeros_like(key_ids),
            seed=seed,
            epoch=epoch,
            domain=3,
        ),
    )
    if masking.combine == "union":
        selected = valid & (token_selected | event_selected | key_selected)
    elif masking.combine == "intersection":
        selected = valid & token_selected & event_selected & key_selected
    else:
        raise ValueError("mask selections must combine by 'union' or 'intersection'")

    unknown = selected & _below(
        masking.unknown_probability,
        _uniform(
            key_ids,
            owner_entities,
            event_positions,
            token_positions,
            key_ids,
            seed=seed,
            epoch=epoch,
            domain=4,
        ),
    )

    if masking.ensure_at_least_one:
        # Reduce first flat positions by owner once, rather than scanning every
        # event/token for each owner. Flat order matches nonzero's row order.
        positions = torch.arange(key_ids.numel(), device=key_ids.device).reshape_as(key_ids)
        sentinel = key_ids.numel()
        first_valid = torch.full_like(entity_ids, sentinel, dtype=torch.long)
        first_selected = first_valid.clone()
        first_valid.scatter_reduce_(
            0,
            owners.reshape(-1),
            positions.masked_fill(~valid, sentinel).reshape(-1),
            reduce="amin",
            include_self=True,
        )
        first_selected.scatter_reduce_(
            0,
            owners.reshape(-1),
            positions.masked_fill(~selected, sentinel).reshape(-1),
            reduce="amin",
            include_self=True,
        )
        fallback = first_valid[(first_selected == sentinel) & (first_valid != sentinel)]
        first_event, first_token = fallback // token_width, fallback % token_width
        selected[first_event, first_token] = True
        unknown[first_event, first_token] = False
        if not masking.supervise_unknown:
            predicted = selected & ~unknown
            first_predicted = torch.full_like(entity_ids, sentinel, dtype=torch.long)
            first_predicted.scatter_reduce_(
                0,
                owners.reshape(-1),
                positions.masked_fill(~predicted, sentinel).reshape(-1),
                reduce="amin",
                include_self=True,
            )
            fallback = first_selected[(first_predicted == sentinel) & (first_selected != sentinel)]
            unknown[fallback // token_width, fallback % token_width] = False

    loss_mask = selected if masking.supervise_unknown else selected & ~unknown
    inputs = value_ids.clone()
    inputs[unknown] = unknown_id
    inputs[selected & ~unknown] = mask_id
    targets = torch.full_like(value_ids, -100)
    targets[loss_mask] = value_ids[loss_mask]
    return inputs, targets, loss_mask
