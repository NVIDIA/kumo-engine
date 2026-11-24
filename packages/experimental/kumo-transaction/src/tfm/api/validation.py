# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction-time composition validation (ADR 0014).

Four rules connect the building-block slots, and each error names the rule it
enforces:

1. A causal objective needs a causal history model and past-only time.
2. Masked modeling needs bidirectional attention inside its unit.
3. The readout must see every event up to the cutoff.
4. Variable-length events need a boundary (checked through tokenizer layouts).

The remaining checks say what a structure implements, for example that the
flat decoder carries no per-token time positions. Implementation limits of a
training backend are checked by its capabilities (ADR 0011), not here.
"""

from __future__ import annotations

from collections.abc import Sequence

from tfm.compositions.flat import Flat
from tfm.compositions.flat_encoder import FlatEncoder
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.protocols import AttentionSpec
from tfm.contracts.tokenizers import LAYOUT_FLAT_FIELDS, SequenceTokenizer
from tfm.heads.binary import Binary
from tfm.layers.decoder import Decoder
from tfm.objectives.contrastive import SubsequenceContrastive
from tfm.objectives.masked_fields import MaskedFields
from tfm.objectives.mlm import TokenEventKeyMLM
from tfm.objectives.next_event import NextEvent
from tfm.objectives.next_event_fields import NextEventFields
from tfm.objectives.past_reconstruction import PastReconstruction
from tfm.objectives.supervised import SupervisedHead
from tfm.pooling.last_event import LastEventMeanPool
from tfm.pooling.last_token import LastTokenPool
from tfm.pooling.mean import MeanPool
from tfm.pooling.usr_token import UsrTokenPool
from tfm.position.none import NoPE
from tfm.position.rope import RoPE
from tfm.position.time_rope import TimeRoPE

CAUSAL_OBJECTIVES: tuple[type, ...] = (NextEvent, NextEventFields)
MASKED_OBJECTIVES: tuple[type, ...] = (TokenEventKeyMLM, MaskedFields)
V2_AUXILIARIES: tuple[type, ...] = (TokenEventKeyMLM, PastReconstruction, SubsequenceContrastive)

RULE_CAUSAL = "rule 1 (a causal objective needs a causal history model)"
RULE_MASKED = "rule 2 (masked modeling needs bidirectional attention inside its unit)"
RULE_READOUT = "rule 3 (the readout must see every event up to the cutoff)"


def _names(specs: Sequence[object]) -> list[str]:
    return [type(s).__name__ for s in specs]


def _as_list(objective: object) -> list[object]:
    return list(objective) if isinstance(objective, (list, tuple)) else [objective]


def _validate_attention(attention: object, *, causal: bool) -> None:
    """Accept any AttentionSpec that can build a kernel with the needed mode.

    Structural rather than ``isinstance(attention, EagerAttention)`` so optional
    kernels can be substituted without editing layer code. ``EagerAttention`` is
    the only kernel in the core export surface and remains the default.
    """
    if not isinstance(attention, AttentionSpec):
        raise IncompatibleCompositionError(
            f"attention must be an AttentionSpec exposing name, capabilities(), and "
            f"build(). Got {type(attention).__name__}."
        )
    mode = "causal" if causal else "bidirectional"
    if not attention.capabilities().get(mode, False):
        raise IncompatibleCompositionError(
            f"Attention kernel {attention.name!r} does not advertise {mode} support."
        )


def _validate_heads(heads: dict[str, object] | None) -> None:
    for name, head in (heads or {}).items():
        if not isinstance(head, Binary):
            raise IncompatibleCompositionError(
                f"Head {name!r} must be Binary(), got {type(head).__name__}."
            )


def _validate_head_objectives(specs: list[object], heads: dict[str, object] | None) -> None:
    """Every SupervisedHead must name a head that exists on the model."""
    available = set(heads or {})
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, SupervisedHead):
            continue
        if spec.head not in available:
            raise IncompatibleCompositionError(
                f"SupervisedHead(head={spec.head!r}) has no matching task head. "
                f"Declared heads: {sorted(available)}."
            )
        if spec.head in seen:
            raise IncompatibleCompositionError(
                f"Duplicate SupervisedHead objective for head {spec.head!r}."
            )
        seen.add(spec.head)


def _reject_masked_on_causal(specs: list[object], structure: str) -> None:
    masked = [s for s in specs if isinstance(s, MASKED_OBJECTIVES)]
    if masked:
        raise IncompatibleCompositionError(
            f"{_names(masked)} on {structure} breaks {RULE_MASKED}: {structure} reads "
            "one causal stream with no bidirectional per-event unit. Use FlatEncoder for "
            "MaskedFields, ProfileHistory, or ProfileEventDecoder with TokenEventKeyMLM as "
            "an auxiliary."
        )


def validate_composition(
    granularity: object,
    position: object,
    attention: object,
    objective: object,
    readout: object,
    heads: dict[str, object] | None,
) -> None:
    specs = _as_list(objective)
    if isinstance(granularity, ProfileHistory):
        _validate_profile_history(position, attention, specs, readout)
    elif isinstance(granularity, ProfileEventDecoder):
        _validate_profile_event_decoder(position, attention, specs, readout)
    elif isinstance(granularity, Flat):
        _validate_flat(granularity, position, attention, specs, readout, heads)
    elif isinstance(granularity, FlatEncoder):
        _validate_flat_encoder(position, attention, specs, readout)
    else:
        raise IncompatibleCompositionError(
            "Supported granularities are Flat(Decoder(...)) or experimental FlatEncoder(...) "
            f"/ ProfileHistory(...) / ProfileEventDecoder(...). Got {type(granularity).__name__}."
        )
    _validate_heads(heads)


def _validate_flat_encoder(
    position: object,
    attention: object,
    specs: list[object],
    readout: object,
) -> None:
    if not isinstance(position, RoPE):
        raise IncompatibleCompositionError(
            f"FlatEncoder requires position=RoPE(), got {type(position).__name__}: without "
            "a causal mask, attention alone cannot tell token order, so NoPE would read a "
            "row as an unordered bag of field tokens."
        )
    _validate_attention(attention, causal=False)
    causal = [s for s in specs if isinstance(s, CAUSAL_OBJECTIVES)]
    if causal:
        raise IncompatibleCompositionError(
            f"{_names(causal)} on FlatEncoder breaks {RULE_CAUSAL}: every position "
            "attends to the tokens it would predict. Use Flat(Decoder) for next-event "
            "objectives."
        )
    if len(specs) != 1 or not isinstance(specs[0], MaskedFields):
        raise IncompatibleCompositionError(
            f"FlatEncoder trains exactly one MaskedFields() objective. Got {_names(specs)}."
        )
    if not isinstance(readout, (LastEventMeanPool, MeanPool, UsrTokenPool)):
        raise IncompatibleCompositionError(
            "FlatEncoder reads out with LastEventMeanPool, MeanPool, or UsrTokenPool "
            f"(the <bos> slot). Got {type(readout).__name__}."
        )


def _validate_flat(
    granularity: Flat,
    position: object,
    attention: object,
    specs: list[object],
    readout: object,
    heads: dict[str, object] | None,
) -> None:
    if not isinstance(granularity.model, Decoder):
        raise IncompatibleCompositionError("Flat.model must be a Decoder.")
    if granularity.model.mask != "causal":
        raise IncompatibleCompositionError(
            "Next-event pretraining requires Decoder(mask='causal'). "
            "Set mask='causal' on the Decoder."
        )
    if not isinstance(position, (NoPE, RoPE)):
        raise IncompatibleCompositionError(
            f"Flat(Decoder) supports NoPE or RoPE, got {type(position).__name__}: flat "
            "batches carry no per-token time positions for TimeRoPE to rotate by. The "
            "causal mask supplies order, and time is encoded as tokens by the tokenizer."
        )
    _validate_attention(attention, causal=True)
    _reject_masked_on_causal(specs, "Flat(Decoder)")
    pretrain = [s for s in specs if isinstance(s, NextEvent)]
    if len(pretrain) != 1:
        raise IncompatibleCompositionError(
            "Flat(Decoder) requires exactly one NextEvent() pretraining objective. "
            f"Got {_names(specs)}."
        )
    _validate_head_objectives(specs, heads)
    unknown = [s for s in specs if not isinstance(s, (NextEvent, SupervisedHead))]
    if unknown:
        raise IncompatibleCompositionError(
            f"Flat(Decoder) trains NextEvent plus optional SupervisedHead objectives; "
            f"{_names(unknown)} need per-event slots. Use ProfileEventDecoder."
        )
    if isinstance(readout, UsrTokenPool):
        raise IncompatibleCompositionError(
            f"UsrTokenPool on Flat(Decoder) breaks {RULE_READOUT}: under a causal mask the "
            "first position has seen nothing after it. Use LastTokenPool or MeanPool."
        )
    if not isinstance(readout, (LastTokenPool, MeanPool)):
        raise IncompatibleCompositionError(
            f"Flat(Decoder) reads out with LastTokenPool or MeanPool, got {type(readout).__name__}."
        )


def _validate_profile_history(
    position: object,
    attention: object,
    specs: list[object],
    readout: object,
) -> None:
    if not isinstance(position, TimeRoPE):
        raise IncompatibleCompositionError(
            f"ProfileHistory requires position=TimeRoPE(). Got {type(position).__name__}."
        )
    _validate_attention(attention, causal=False)
    causal = [s for s in specs if isinstance(s, CAUSAL_OBJECTIVES)]
    if causal:
        raise IncompatibleCompositionError(
            f"{_names(causal)} on ProfileHistory breaks {RULE_CAUSAL}: its history encoder "
            "is bidirectional, so every slot sees the event it would predict. Use "
            "ProfileEventDecoder for next-event objectives."
        )
    if len(specs) != 1 or not isinstance(specs[0], TokenEventKeyMLM):
        raise IncompatibleCompositionError(
            f"ProfileHistory trains exactly one TokenEventKeyMLM() objective. Got {_names(specs)}."
        )
    if not isinstance(readout, (UsrTokenPool, LastTokenPool, MeanPool)):
        raise IncompatibleCompositionError(
            "ProfileHistory reads out with UsrTokenPool, LastTokenPool, or MeanPool. "
            f"Got {type(readout).__name__}."
        )


def _validate_profile_event_decoder(
    position: object,
    attention: object,
    specs: list[object],
    readout: object,
) -> None:
    if not isinstance(position, (TimeRoPE, NoPE)):
        raise IncompatibleCompositionError(
            "ProfileEventDecoder is timed by TimeRoPE() (elapsed log-time, past events only) "
            f"or NoPE() (causal order and calendar only). Got {type(position).__name__}."
        )
    # Encoders are bidirectional within a unit; the history decoder is causal.
    _validate_attention(attention, causal=False)
    _validate_attention(attention, causal=True)
    names = _names(specs)
    if sum(isinstance(s, NextEventFields) for s in specs) != 1:
        raise IncompatibleCompositionError(
            f"ProfileEventDecoder requires exactly one NextEventFields() objective. Got {names}."
        )
    extra = [s for s in specs if not isinstance(s, NextEventFields)]
    unsupported = [s for s in extra if not isinstance(s, V2_AUXILIARIES)]
    if unsupported:
        raise IncompatibleCompositionError(
            "ProfileEventDecoder accepts NextEventFields() plus auxiliary "
            f"{[t.__name__ for t in V2_AUXILIARIES]}. Got {names}."
        )
    kinds = [type(s) for s in extra]
    if len(set(kinds)) != len(kinds):
        raise IncompatibleCompositionError(
            f"Each auxiliary objective may appear once. Got {names}."
        )
    if isinstance(readout, UsrTokenPool):
        raise IncompatibleCompositionError(
            f"UsrTokenPool on ProfileEventDecoder breaks {RULE_READOUT}: under the causal "
            "decoder, slot 0 has seen only the profile. Use LastTokenPool or MeanPool."
        )
    if not isinstance(readout, (LastTokenPool, MeanPool)):
        raise IncompatibleCompositionError(
            "ProfileEventDecoder reads out with LastTokenPool or MeanPool. "
            f"Got {type(readout).__name__}."
        )


def accepted_layouts_for(granularity: object) -> frozenset[str]:
    """Layouts a granularity can consume (rule 4)."""
    accepted = getattr(granularity, "accepted_layouts", None)
    if accepted is None:
        return frozenset({LAYOUT_FLAT_FIELDS})
    return frozenset(accepted)


def validate_tokenizer_layout(tokenizer: SequenceTokenizer, granularity: object) -> None:
    """Reject tokenizer / history-model pairs that do not share a layout (rule 4)."""
    accepted = accepted_layouts_for(granularity)
    if tokenizer.layout not in accepted:
        gran_name = getattr(granularity, "name", type(granularity).__name__)
        raise IncompatibleCompositionError(
            f"Tokenizer {tokenizer.name!r} emits layout {tokenizer.layout!r}, but "
            f"granularity {gran_name!r} accepts {sorted(accepted)}: each structure needs "
            "the event boundary its layout carries (rule 4, ADR 0014)."
        )
