# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental composition API for the hierarchical stacks and new building blocks.

These names are **not** part of the core `tfm.__all__` surface. Import from
`tfm.experimental` (or the underlying modules). See ADRs 0007, 0012, and 0014.
"""

from __future__ import annotations

from collections.abc import Sequence

from tfm.adaptation.fusion import JointFusionHead, PeriodicEmbedding
from tfm.api.validation import validate_composition
from tfm.attention.eager import EagerAttention
from tfm.compositions.flat_encoder import FlatEncoder
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.protocols import AttentionSpec
from tfm.layers.flat_encoder import FlatEncoderModel, FlatEncoderReadout
from tfm.layers.gru import GRUBackbone
from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.objectives.contrastive import SubsequenceContrastive
from tfm.objectives.masked_fields import MaskedFields
from tfm.objectives.mlm import TokenEventKeyMLM
from tfm.objectives.next_event_fields import NextEventFields
from tfm.objectives.past_reconstruction import PastReconstruction
from tfm.pooling.last_event import LastEventMeanPool
from tfm.pooling.last_token import LastTokenPool
from tfm.pooling.mean import MeanPool
from tfm.pooling.registry import Readout
from tfm.pooling.usr_token import UsrTokenPool
from tfm.position.none import NoPE
from tfm.position.rope import RoPE
from tfm.position.time_rope import TimeRoPE
from tfm.training.lora import LoRA

__all__ = [
    "FlatEncoder",
    "FlatEncoderModel",
    "GRUBackbone",
    "JointFusionHead",
    "LastEventMeanPool",
    "LoRA",
    "MaskedFields",
    "MeanPool",
    "NextEventFields",
    "PastReconstruction",
    "PeriodicEmbedding",
    "ProfileEventDecoder",
    "ProfileEventDecoderModel",
    "ProfileHistory",
    "ProfileHistoryModel",
    "SubsequenceContrastive",
    "TimeRoPE",
    "TokenEventKeyMLM",
    "UsrTokenPool",
    "compose_blueprint_v2",
    "compose_flat_encoder",
    "compose_profile_history",
]

V2Objective = NextEventFields | TokenEventKeyMLM | PastReconstruction | SubsequenceContrastive


def compose_profile_history(
    granularity: ProfileHistory | None = None,
    position: TimeRoPE | None = None,
    attention: AttentionSpec | None = None,
    objective: TokenEventKeyMLM | None = None,
    readout: Readout | None = None,
) -> ProfileHistoryModel:
    """Validate and build the experimental profile / event / history encoder.

    `readout` defaults to `UsrTokenPool()`, PRAGMA's `[USR]` slot. Because the
    history encoder is bidirectional, `LastTokenPool` and `MeanPool` are also valid.
    """
    granularity = ProfileHistory() if granularity is None else granularity
    position = TimeRoPE() if position is None else position
    attn: AttentionSpec = EagerAttention() if attention is None else attention
    objective = TokenEventKeyMLM() if objective is None else objective
    readout = UsrTokenPool() if readout is None else readout
    validate_composition(granularity, position, attn, objective, readout, None)
    return ProfileHistoryModel(granularity, objective, attn, readout)


def compose_flat_encoder(
    granularity: FlatEncoder,
    position: RoPE | None = None,
    attention: AttentionSpec | None = None,
    objective: MaskedFields | None = None,
    readout: FlatEncoderReadout | None = None,
) -> FlatEncoderModel:
    """Validate and build the experimental flat BERT-style encoder (ADR 0019).

    `granularity` has no default: it carries the tokenizer's `FieldVocabulary`.
    `readout` defaults to `LastEventMeanPool()`, the scored transaction's tokens.
    """
    position = RoPE() if position is None else position
    attn: AttentionSpec = EagerAttention() if attention is None else attention
    objective = MaskedFields() if objective is None else objective
    readout = LastEventMeanPool() if readout is None else readout
    validate_composition(granularity, position, attn, objective, readout, None)
    return FlatEncoderModel(granularity, objective, attn, readout)


def compose_blueprint_v2(
    granularity: ProfileEventDecoder | None = None,
    position: TimeRoPE | NoPE | None = None,
    attention: AttentionSpec | None = None,
    objective: V2Objective | Sequence[V2Objective] | None = None,
    readout: LastTokenPool | MeanPool | None = None,
) -> ProfileEventDecoderModel:
    """Validate and build the Blueprint v2 profile / event encoders + causal decoder.

    `position` times the history decoder: `TimeRoPE()` (default) rotates
    attention by elapsed log-time; `NoPE()` leaves it causal order and calendar only.

    `objective` is `NextEventFields()` alone, or a sequence of `NextEventFields()`
    plus any of `TokenEventKeyMLM()`, `PastReconstruction()`, and
    `SubsequenceContrastive()` as auxiliary losses. The default adds PRAGMA's
    within-event masking: `(NextEventFields(), TokenEventKeyMLM())`.
    """
    granularity = ProfileEventDecoder() if granularity is None else granularity
    position = TimeRoPE() if position is None else position
    attn: AttentionSpec = EagerAttention() if attention is None else attention
    if objective is None:
        objective = (NextEventFields(), TokenEventKeyMLM())
    readout = LastTokenPool() if readout is None else readout
    validate_composition(granularity, position, attn, objective, readout, None)
    specs = list(objective) if isinstance(objective, Sequence) else [objective]
    next_event = next(s for s in specs if isinstance(s, NextEventFields))
    mlm = next((s for s in specs if isinstance(s, TokenEventKeyMLM)), None)
    auxiliary = tuple(
        s for s in specs if isinstance(s, (PastReconstruction, SubsequenceContrastive))
    )
    return ProfileEventDecoderModel(
        granularity, next_event, mlm, attn, readout, auxiliary, position=position
    )
