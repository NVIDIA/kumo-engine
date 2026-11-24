# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tfm import EagerAttention, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.api.validation import validate_composition
from tfm.attention.eager import EagerAttention as Eager
from tfm.compositions.flat import Flat as FlatG
from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.experimental import compose_profile_history
from tfm.layers.decoder import Decoder as Dec
from tfm.objectives.mlm import TokenEventKeyMLM
from tfm.objectives.next_event import NextEvent as NE
from tfm.pooling.last_token import LastTokenPool as LTP
from tfm.pooling.usr_token import UsrTokenPool
from tfm.position.time_rope import TimeRoPE


def test_m1_export_excludes_experimental() -> None:
    import tfm

    assert "ProfileHistory" not in tfm.__all__
    assert "compose_profile_history" not in tfm.__all__
    assert "UsrTokenPool" not in tfm.__all__
    assert "TimeRoPE" not in tfm.__all__


def test_compose_profile_history_validates() -> None:
    model = compose_profile_history(
        granularity=ProfileHistory(hidden_size=32, num_heads=4, vocab_size=64)
    )
    assert model.hidden_size == 32
    assert model.vocab_size == 64


def test_profile_history_rejects_next_event() -> None:
    with pytest.raises(IncompatibleCompositionError, match="rule 1"):
        validate_composition(
            ProfileHistory(vocab_size=32),
            TimeRoPE(),
            Eager(),
            NE(),
            UsrTokenPool(),
            None,
        )


def test_flat_rejects_time_rope() -> None:
    with pytest.raises(IncompatibleCompositionError, match="NoPE"):
        validate_composition(
            FlatG(Dec(mask="causal", vocab_size=32)),
            TimeRoPE(),
            Eager(),
            NE(),
            LTP(),
            None,
        )


def test_profile_history_accepts_any_readout_that_sees_every_event() -> None:
    """Its history encoder is bidirectional, so rule 3 holds for every readout."""
    from tfm.pooling.mean import MeanPool

    for readout in (UsrTokenPool(), LTP(), MeanPool()):
        validate_composition(
            ProfileHistory(vocab_size=32),
            TimeRoPE(),
            Eager(),
            TokenEventKeyMLM(),
            readout,
            None,
        )


def test_flat_rejects_usr_readout_by_rule_3() -> None:
    with pytest.raises(IncompatibleCompositionError, match="rule 3"):
        validate_composition(
            FlatG(Dec(mask="causal", vocab_size=32)), NoPE(), Eager(), NE(), UsrTokenPool(), None
        )


def test_trxfm_still_rejects_profile_history_kw() -> None:
    with pytest.raises(IncompatibleCompositionError):
        TrxFM(
            granularity=ProfileHistory(vocab_size=32),  # type: ignore[arg-type]
            position=NoPE(),
            attention=EagerAttention(),
            objective=NextEvent(),
            readout=LastTokenPool(),
        )
