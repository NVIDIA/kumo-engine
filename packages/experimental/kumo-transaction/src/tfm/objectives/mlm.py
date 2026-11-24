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
    """Mask value ids; keep key ids. Unk-dropout positions are excluded from loss."""
    token_mask = batch.token_mask.bool()
    selected = torch.zeros_like(token_mask)
    rnd = torch.rand(token_mask.shape, generator=generator, device=token_mask.device)
    selected |= token_mask & (rnd < spec.p_token)
    ev_rnd = torch.rand(batch.event_mask.shape, generator=generator, device=token_mask.device)
    selected |= token_mask & (ev_rnd < spec.p_event).unsqueeze(-1)
    batch_size, n_events, _n_tok = batch.key_ids.shape
    for b in range(batch_size):
        valid_keys = batch.key_ids[b][token_mask[b]]
        if valid_keys.numel() == 0:
            continue
        for key in valid_keys.unique():
            draw = torch.rand((), generator=generator, device=token_mask.device)
            if float(draw) < spec.p_key:
                selected[b] |= token_mask[b] & (batch.key_ids[b] == key)
    if not bool(selected.any()) and bool(token_mask.any()):
        # Tiny batches can sample zero masks; force one valid token so CE is defined.
        idx = token_mask.view(-1).nonzero(as_tuple=False)[0]
        selected.view(-1)[idx] = True
    unk_draw = torch.rand(token_mask.shape, generator=generator, device=token_mask.device)
    unk_sel = selected & (unk_draw < spec.p_unk)
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
        hour=batch.hour,
        dow=batch.dow,
        dom=batch.dom,
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
