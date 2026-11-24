# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token / whole-event / key-level MLM for key-value-time batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.outputs import ModelOutput
from tfm.features.tokenizers.key_value_time import MASK, UNK

# splitmix64's constants as signed int64.
_GOLDEN, _MIX1, _MIX2 = -0x61C8864680B583EB, -0x40A7B892E31B1A47, -0x6B2FB644ECCEEE15


def _shr(h: Tensor, s: int) -> Tensor:
    """Logical right shift of int64 `h` by `s` bits."""
    return (h >> s) & ((1 << (64 - s)) - 1)


def _uniform(h: Tensor) -> Tensor:
    """Each int64 of `h`, mixed by splitmix64's finaliser, as a uniform float in [0, 1)."""
    h = (h ^ _shr(h, 30)) * _MIX1
    h = (h ^ _shr(h, 27)) * _MIX2
    return _shr(h ^ _shr(h, 31), 40).float() * 2.0**-24


@dataclass(frozen=True)
class TokenEventKeyMLM:
    """PRAGMA-style masked modeling over key-value-time events.

    `label_smoothing` is the cross-entropy smoothing the PRAGMA paper specifies
    for its tied 3d MLM head. It matters here because the value vocabulary is
    bucketed: adjacent amount or time buckets are near-synonyms, so a hard
    one-hot target over-penalizes an off-by-one-bucket prediction. Set it to
    0.0 for an unsmoothed baseline.
    """

    name: str = "token_event_key_mlm"
    p_token: float = 0.15
    p_event: float = 0.10
    p_key: float = 0.10
    p_unk: float = 0.10
    label_smoothing: float = 0.1
    ignore_index: int = -100
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            from tfm.contracts.errors import IncompatibleCompositionError

            raise IncompatibleCompositionError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}."
            )

    def required_outputs(self) -> tuple[str, ...]:
        return ("logits",)

    def target_level(self) -> str:
        return "event"


def apply_mlm_mask(
    batch: KeyValueTimeBatch,
    spec: TokenEventKeyMLM,
    *,
    generator: torch.Generator | None = None,
) -> KeyValueTimeBatch:
    """Mask value ids; keep key ids. Unk-dropout positions are excluded from loss.

    A token is selected with `p_token`, every token of an event with `p_event`, and every
    token of a row whose key id draws under `p_key`: one draw per row and key id, a hash
    of the key id and a per-row seed, so no tensor scales with the vocabulary. Runs on
    the batch's device without a host sync.
    """
    token_mask = batch.token_mask.bool()
    device = token_mask.device
    rnd = torch.rand(token_mask.shape, generator=generator, device=device)
    selected = token_mask & (rnd < spec.p_token)
    ev_rnd = torch.rand(batch.event_mask.shape, generator=generator, device=device)
    selected |= token_mask & (ev_rnd < spec.p_event).unsqueeze(-1)
    batch_size = batch.key_ids.shape[0]
    seed = torch.randint(0, 2**62, (batch_size, 1), generator=generator, device=device)
    keys = batch.key_ids.reshape(batch_size, -1).long()
    key_hit = _uniform(keys * _GOLDEN + seed).view_as(token_mask)
    selected |= token_mask & (key_hit < spec.p_key)
    unk_draw = torch.rand(token_mask.shape, generator=generator, device=device)
    unk_sel = selected & (unk_draw < spec.p_unk)
    # Tiny batches can end with no supervised token; mask the first valid one so CE is defined.
    force = (token_mask.any() & ~(selected & ~unk_sel).any()).view(1)
    first = token_mask.view(-1).int().argmax().view(1)
    pin = torch.zeros_like(token_mask).view(-1).scatter_(0, first, force).view_as(token_mask)
    selected |= pin
    unk_sel &= ~pin
    mask_sel = selected & ~unk_sel
    labels = batch.value_ids.clone()
    labels[~selected] = spec.ignore_index
    labels[unk_sel] = spec.ignore_index
    value_ids = batch.value_ids.clone()
    value_ids[mask_sel] = MASK
    value_ids[unk_sel] = UNK
    return KeyValueTimeBatch(
        key_ids=batch.key_ids,
        value_ids=value_ids,
        positions=batch.positions,
        token_mask=batch.token_mask,
        event_mask=batch.event_mask,
        time_log=batch.time_log,
        events=batch.events,
        prof_key_ids=batch.prof_key_ids,
        prof_value_ids=batch.prof_value_ids,
        prof_positions=batch.prof_positions,
        prof_mask=batch.prof_mask,
        prof_time_log=batch.prof_time_log,
        entity_ids=batch.entity_ids,
        sequence_ids=batch.sequence_ids,
        fraud_labels=batch.fraud_labels,
        numeric_features=batch.numeric_features,
        mlm_labels=labels,
        elapsed_log=batch.elapsed_log,
    )


class TokenEventKeyMLMLoss(nn.Module):
    """Cross-entropy over masked value positions only.

    `[UNK]`-dropout positions are already `ignore_index` in `mlm_labels`, so
    they contribute no gradient and are also excluded from the smoothing mass.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.crit = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

    def forward(self, output: ModelOutput, batch: KeyValueTimeBatch) -> Tensor:
        if output.logits is None or batch.mlm_labels is None:
            raise ValueError("TokenEventKeyMLM requires logits and mlm_labels.")
        logits = output.logits
        labels = batch.mlm_labels
        return self.crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
