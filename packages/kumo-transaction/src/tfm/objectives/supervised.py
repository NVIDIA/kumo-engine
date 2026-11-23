# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-level supervised objective for task heads.

Composed alongside a pretraining objective so `Binary` heads receive gradient.
Without this, heads are forward-only and their parameters never update.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import MissingLabelError
from tfm.contracts.outputs import ModelOutput


@dataclass(frozen=True)
class SupervisedHead:
    """Binary cross-entropy on a named task head over pooled embeddings.

    Contract declarations required for any new objective:

    - **Label horizon.** None of its own. It consumes the sequence-level label
      already reduced by `tfm.data.collate.sequence_label`, which keeps an
      event label only when `available_at <= cutoff_time`. The horizon is
      therefore whatever the recipe's `label_delay` established upstream.
    - **Cutoff semantics.** Labels are read from `TransactionBatch.fraud_labels`,
      built from events already truncated to `sequence.cutoff_time`. This
      objective never re-reads raw events and cannot widen the window.
    - **Required outputs.** `predictions[head]` and `embeddings`.
    - **Gradient behavior.** Gradient flows into the named head and, through the
      pooled embedding, into the backbone. Scaled by `weight`. With
      `detach_backbone`, the head loss stops at the pooled embedding, so the head
      trains as a linear probe while the backbone still learns from the
      pretraining objective only.
    - **Leakage checks.** Raises `MissingLabelError` rather than silently
      substituting zeros for absent labels. Contributes no loss for sequences
      whose label was unavailable at cutoff (encoded as NaN by the collator).
    """

    name: str = "supervised_head"
    head: str = "fraud"
    weight: float = 1.0
    detach_backbone: bool = False
    support_tier: str = "experimental"

    def required_outputs(self) -> tuple[str, ...]:
        return ("predictions", "embeddings")

    def target_level(self) -> str:
        return "sequence"

    def label_horizon(self) -> str:
        return "inherited-from-sequence-label"

    def cutoff_semantics(self) -> str:
        return "labels available at sequence.cutoff_time only"


class SupervisedHeadLoss(nn.Module):
    def __init__(self, spec: SupervisedHead) -> None:
        super().__init__()
        self.spec = spec
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, output: ModelOutput, batch: TransactionBatch) -> Tensor:
        if output.predictions is None or self.spec.head not in output.predictions:
            raise MissingLabelError(
                f"SupervisedHead({self.spec.head!r}) requires a matching head on the model. "
                f"Add heads={{{self.spec.head!r}: Binary()}}."
            )
        if batch.fraud_labels is None:
            raise MissingLabelError(
                f"SupervisedHead({self.spec.head!r}) requires batch.fraud_labels. "
                "Collate sequences with tfm.data.collate_sequences."
            )
        logits = output.predictions[self.spec.head]
        targets = batch.fraud_labels.to(logits.dtype)
        observed = ~targets.isnan()
        if not bool(observed.any()):
            return logits.sum() * 0.0
        return self.crit(logits[observed], targets[observed])
