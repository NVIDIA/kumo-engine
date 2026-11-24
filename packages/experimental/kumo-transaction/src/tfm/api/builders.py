# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build models from a recipe config, one building-block slot per key (ADR 0014).

Each `model.composition` key selects a component; nothing is hard-coded per
stack. `tfm.api.validation` then decides whether the combination is valid, so
a config such as `granularity: flat` with `position: time_rope` fails with the
rule it breaks instead of silently training something else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from torch import nn

from tfm.api.model import TrxFM
from tfm.attention.registry import build_attention
from tfm.compositions.flat import Flat
from tfm.compositions.flat_encoder import FlatEncoder
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.compositions.profile_history import ProfileHistory
from tfm.config import CyclicPeriodConfig, GranularityConfig, RecipeConfig
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import CALENDAR, CyclicTime
from tfm.contracts.tokenizers import FieldVocabulary
from tfm.experimental import compose_blueprint_v2, compose_flat_encoder, compose_profile_history
from tfm.heads.binary import Binary
from tfm.layers.decoder import Decoder
from tfm.layers.flat_encoder import FlatEncoderModel
from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.objectives.contrastive import SubsequenceContrastive
from tfm.objectives.masked_fields import MaskedFields
from tfm.objectives.mlm import TokenEventKeyMLM
from tfm.objectives.next_event import NextEvent
from tfm.objectives.next_event_fields import NextEventFields
from tfm.objectives.past_reconstruction import PastReconstruction
from tfm.objectives.supervised import SupervisedHead
from tfm.pooling.registry import build_readout
from tfm.position.none import NoPE
from tfm.position.rope import RoPE
from tfm.position.time_rope import TimeRoPE

if TYPE_CHECKING:
    from tfm.api.graph import GraphBuildContext

Objective = (
    NextEvent
    | NextEventFields
    | TokenEventKeyMLM
    | MaskedFields
    | PastReconstruction
    | SubsequenceContrastive
)


def build_position(name: str) -> NoPE | RoPE | TimeRoPE:
    """The position spec `model.composition.position` names."""
    if name == "nope":
        return NoPE()
    if name == "rope":
        return RoPE()
    if name == "time_rope":
        return TimeRoPE()
    raise IncompatibleCompositionError(
        f"Unknown position {name!r}; use 'nope', 'rope', or 'time_rope'."
    )


def build_objectives(config: RecipeConfig) -> list[Objective]:
    """The primary objective, then each `objective.auxiliary` entry in order."""
    section = config.objective
    mlm = TokenEventKeyMLM(**section.mlm.model_dump())
    primary = config.model.composition.objective
    specs: list[Objective] = []
    if primary == "next_event":
        specs.append(NextEvent())
    elif primary == "next_event_fields":
        specs.append(NextEventFields())
    elif primary == "masked_fields":
        specs.append(MaskedFields(**section.masked_fields.model_dump()))
    else:
        specs.append(mlm)
    for name in section.auxiliary:
        if name == "token_event_key_mlm":
            specs.append(mlm)
        elif name == "past_reconstruction":
            specs.append(PastReconstruction(**section.past_reconstruction.model_dump()))
        else:
            specs.append(SubsequenceContrastive(**section.contrastive.model_dump()))
    return specs


def trxfm_from_config(
    config: RecipeConfig,
    *,
    vocab_size: int,
    backbone: Any | None = None,
    head_objectives: Sequence[SupervisedHead] = (),
) -> TrxFM:
    """The flat stack. `backbone` replaces the eager decoder (HuggingFace, GRU)."""
    composition = config.model.composition
    dec = composition.granularity.decoder
    decoder = Decoder(
        depth=dec.depth,
        mask=dec.mask,
        hidden_size=dec.hidden_size,
        num_heads=dec.num_heads,
        dropout=dec.dropout,
        vocab_size=vocab_size,
        # The context must cover the collated row. A longer configured context is kept.
        max_seq_len=max(dec.max_seq_len, config.data.max_tokens),
        ffn_size=dec.ffn_size,
    )
    # Typed loosely on purpose: validate_composition rejects what the flat stack
    # cannot run (TimeRoPE, MLM, UsrTokenPool) with the rule it breaks.
    position: Any = build_position(composition.position)
    readout: Any = build_readout(composition.readout)
    objectives: list[Any] = [*build_objectives(config), *head_objectives]
    return TrxFM(
        granularity=Flat(decoder),
        position=position,
        attention=build_attention(composition.attention),
        objective=objectives,
        readout=readout,
        heads={name: Binary() for name in composition.heads},
        backbone=backbone,
    )


def _time_features(gran: GranularityConfig) -> tuple[CyclicTime, ...]:
    """`CALENDAR`'s arrays first, in its order, then the others by name, so the order the
    keys are written in never changes which weights read which array."""
    calendar = [f.array for f in CALENDAR]

    def rank(name: str) -> tuple[int, str]:
        return (calendar.index(name), "") if name in calendar else (len(calendar), name)

    return tuple(
        CyclicTime(name, c.period, c.origin)
        if isinstance(c, CyclicPeriodConfig)
        else CyclicTime(name, float(c))
        for name, c in sorted(gran.time_features.items(), key=lambda item: rank(item[0]))
    )


def profile_history_from_config(config: RecipeConfig, *, vocab_size: int) -> ProfileHistoryModel:
    """The PRAGMA-style bidirectional profile / event / history stack."""
    composition = config.model.composition
    gran = composition.granularity
    spec = ProfileHistory(
        hidden_size=gran.decoder.hidden_size,
        num_heads=gran.decoder.num_heads,
        profile_depth=gran.profile_depth,
        event_depth=gran.event_depth,
        history_depth=gran.history_depth,
        vocab_size=vocab_size,
        dropout=gran.decoder.dropout,
        varlen=gran.varlen,
        time_features=_time_features(gran),
    )
    position: Any = build_position(composition.position)
    objectives: Any = build_objectives(config)
    return compose_profile_history(
        granularity=spec,
        position=position,
        attention=build_attention(composition.attention),
        objective=objectives[0] if len(objectives) == 1 else objectives,
        readout=build_readout(composition.readout),
    )


def flat_encoder_from_config(
    config: RecipeConfig, *, vocab_size: int, vocabulary: FieldVocabulary | None
) -> FlatEncoderModel:
    """The experimental flat BERT-style encoder (ADR 0019)."""
    if vocabulary is None:
        raise IncompatibleCompositionError(
            "granularity.type flat_encoder needs the tokenizer's FieldVocabulary to mask "
            "each field within its own ids; pass `vocabulary`."
        )
    if vocabulary.vocab_size != vocab_size:
        raise IncompatibleCompositionError(
            f"FieldVocabulary covers {vocabulary.vocab_size} ids, but vocab_size is {vocab_size}."
        )
    composition = config.model.composition
    dec = composition.granularity.decoder
    if dec.ffn_size not in (None, 4 * dec.hidden_size):
        raise IncompatibleCompositionError(
            f"flat_encoder blocks use a 4 * hidden_size feed-forward; unset ffn_size "
            f"(got {dec.ffn_size})."
        )
    windows = config.data.pretrain_windows
    widest = max(config.data.max_tokens, (windows.max_tokens or 0) if windows else 0)
    spec = FlatEncoder(
        vocabulary=vocabulary,
        depth=dec.depth,
        hidden_size=dec.hidden_size,
        num_heads=dec.num_heads,
        dropout=dec.dropout,
        max_seq_len=max(dec.max_seq_len, widest),
    )
    position: Any = build_position(composition.position)
    objectives: Any = build_objectives(config)
    readout: Any = build_readout(composition.readout)
    return compose_flat_encoder(
        granularity=spec,
        position=position,
        attention=build_attention(composition.attention),
        objective=objectives[0] if len(objectives) == 1 else objectives,
        readout=readout,
    )


def profile_event_spec_from_config(config: RecipeConfig, *, vocab_size: int) -> ProfileEventDecoder:
    """Public sequence specification shared by token and graph model builders."""
    composition = config.model.composition
    gran = composition.granularity
    spec = ProfileEventDecoder(
        hidden_size=gran.decoder.hidden_size,
        num_heads=gran.decoder.num_heads,
        profile_depth=gran.profile_depth,
        event_depth=gran.event_depth,
        decoder_depth=gran.decoder.depth,
        vocab_size=vocab_size,
        dropout=gran.decoder.dropout,
        varlen=gran.varlen,
        time_features=_time_features(gran),
    )
    return spec


def profile_event_decoder_from_config(
    config: RecipeConfig, *, vocab_size: int
) -> ProfileEventDecoderModel:
    """The Blueprint v2 stack: profile and event encoders, causal decoder."""
    composition = config.model.composition
    spec = profile_event_spec_from_config(config, vocab_size=vocab_size)
    position: Any = build_position(composition.position)
    readout: Any = build_readout(composition.readout)
    objectives: Any = build_objectives(config)
    return compose_blueprint_v2(
        granularity=spec,
        position=position,
        attention=build_attention(composition.attention),
        objective=objectives,
        readout=readout,
    )


def model_from_config(
    config: RecipeConfig,
    *,
    vocab_size: int,
    backbone: Any | None = None,
    head_objectives: Sequence[SupervisedHead] = (),
    vocabulary: FieldVocabulary | None = None,
    graph_context: GraphBuildContext | None = None,
) -> nn.Module:
    """Dispatch on `model.composition.granularity.type`.

    `vocabulary` is the tokenizer's field layout, which only `flat_encoder` reads.
    """
    kind = config.model.composition.granularity.type
    if graph_context is not None:
        from tfm.api.graph import build_graph_architecture

        return build_graph_architecture(kind, config, graph_context)
    if kind == "flat":
        return trxfm_from_config(
            config, vocab_size=vocab_size, backbone=backbone, head_objectives=head_objectives
        )
    if backbone is not None or head_objectives:
        raise IncompatibleCompositionError(
            f"backbone and head objectives apply to the flat stack only, not {kind!r}."
        )
    if kind == "flat_encoder":
        return flat_encoder_from_config(config, vocab_size=vocab_size, vocabulary=vocabulary)
    if kind not in _HIERARCHICAL:
        raise IncompatibleCompositionError(
            f"Unknown architecture {kind!r}; graph models require a GraphBuildContext"
        )
    return _HIERARCHICAL[kind](config, vocab_size=vocab_size)


_HIERARCHICAL: dict[str, Callable[..., ProfileHistoryModel | ProfileEventDecoderModel]] = {
    "profile_history": profile_history_from_config,
    "profile_event_decoder": profile_event_decoder_from_config,
}
