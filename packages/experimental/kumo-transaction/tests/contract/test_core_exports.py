# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal per-component contracts for every name in the core export surface.

AGENTS.md requires each exported component to have a minimal example and a
contract test. Broader end-to-end behavior lives in the other contract modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

import tfm
from tfm import (
    Binary,
    Decoder,
    EagerAttention,
    Flat,
    LastTokenPool,
    NextEvent,
    NoPE,
    TrxFM,
)
from tfm.attention.eager import EagerAttentionKernel
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError, MissingLabelError
from tfm.contracts.protocols import AttentionSpec
from tfm.heads.binary import BinaryHead
from tfm.layers.decoder import DecoderModule
from tfm.objectives.next_event import NextEventLoss
from tfm.objectives.supervised import SupervisedHead
from tfm.pooling.last_token import LastTokenPoolModule
from tfm.position.rope import RoPE

CORE_EXPORTS = (
    "Binary",
    "Decoder",
    "EagerAttention",
    "Flat",
    "LastTokenPool",
    "NextEvent",
    "NoPE",
    "TrxFM",
)


def _batch(ids: list[list[int]], mask: list[list[int]], **kwargs: Tensor) -> TransactionBatch:
    return TransactionBatch(
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.tensor(mask, dtype=torch.long),
        **kwargs,
    )


def test_core_export_surface_is_exactly_the_documented_names() -> None:
    assert tuple(sorted(tfm.__all__)) == CORE_EXPORTS


# --- Flat ------------------------------------------------------------------


def test_flat_wraps_decoder_and_declares_accepted_layout() -> None:
    flat = Flat(Decoder())
    assert flat.accepted_layouts == frozenset({"flat_fields", "flat_key_value"})
    assert isinstance(flat.model, Decoder)


def test_flat_rejects_non_decoder_model() -> None:
    with pytest.raises(IncompatibleCompositionError, match="must be a Decoder"):
        TrxFM(granularity=Flat(object()))  # type: ignore[arg-type]


# --- Decoder ---------------------------------------------------------------


def test_decoder_defaults_and_shape() -> None:
    spec = Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=20)
    module = DecoderModule(spec)
    ids = torch.tensor([[1, 2, 3]])
    hidden = module(ids, torch.ones_like(ids), causal=True)
    assert hidden.shape == (1, 3, 16)
    assert module.logits(hidden).shape == (1, 3, 20)


def test_decoder_rejects_non_causal_mask() -> None:
    with pytest.raises(IncompatibleCompositionError, match="mask='causal'"):
        Decoder(mask="bidirectional")


def test_decoder_rejects_indivisible_head_split() -> None:
    with pytest.raises(IncompatibleCompositionError, match="divisible"):
        Decoder(hidden_size=10, num_heads=4)


# --- NoPE ------------------------------------------------------------------


def test_nope_is_identity() -> None:
    hidden = torch.randn(2, 3, 8)
    assert NoPE().name == "nope"
    assert torch.equal(NoPE().apply(hidden), hidden)


# --- EagerAttention --------------------------------------------------------


def test_eager_attention_spec_builds_kernel_and_declares_capabilities() -> None:
    spec = EagerAttention()
    assert isinstance(spec, AttentionSpec)
    assert spec.capabilities() == {"causal": True, "bidirectional": True}
    assert isinstance(spec.build(), EagerAttentionKernel)


def test_eager_attention_masks_future_and_padding() -> None:
    kernel = EagerAttentionKernel()
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0]])
    out = kernel(query, key, value, mask, causal=True)
    assert out.shape == (1, 1, 4, 8)
    # Position 0 attends only to itself under a causal mask.
    assert torch.allclose(out[0, 0, 0], value[0, 0, 0], atol=1e-5)
    # Perturbing a padded column must not change any output row.
    value_pad = value.clone()
    value_pad[0, 0, 3] += 100.0
    out_pad = kernel(query, key, value_pad, mask, causal=True)
    assert torch.allclose(out, out_pad, atol=1e-5)


def test_attention_spec_is_actually_invoked_by_the_decoder() -> None:
    """Guards the extension point: TrxFM(attention=...) must reach every block."""
    calls: list[bool] = []

    class SpyKernel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = EagerAttentionKernel()

        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor,
            *,
            causal: bool,
            segment_ids: Tensor | None = None,
        ) -> Tensor:
            calls.append(causal)
            return self.inner(
                query, key, value, attention_mask, causal=causal, segment_ids=segment_ids
            )

    @dataclass(frozen=True)
    class SpyAttention:
        name: str = "spy"

        def capabilities(self) -> dict[str, bool]:
            return {"causal": True, "bidirectional": True}

        def build(self) -> nn.Module:
            return SpyKernel()

    depth = 3
    model = TrxFM(
        granularity=Flat(Decoder(depth=depth, hidden_size=16, num_heads=4, vocab_size=20)),
        attention=SpyAttention(),
    )
    model(_batch([[1, 2, 3]], [[1, 1, 1]]), compute_loss=False)
    assert calls == [True] * depth
    assert model.to_config()["attention"] == "spy"


def test_attention_must_be_an_attention_spec() -> None:
    with pytest.raises(IncompatibleCompositionError, match="AttentionSpec"):
        TrxFM(granularity=Flat(Decoder()), attention=object())  # type: ignore[arg-type]


def test_attention_must_advertise_causal_support_for_flat_decoder() -> None:
    @dataclass(frozen=True)
    class BidirectionalOnly:
        name: str = "bidir-only"

        def capabilities(self) -> dict[str, bool]:
            return {"causal": False, "bidirectional": True}

        def build(self) -> nn.Module:
            return EagerAttentionKernel()

    with pytest.raises(IncompatibleCompositionError, match="causal"):
        TrxFM(granularity=Flat(Decoder()), attention=BidirectionalOnly())


# --- LastTokenPool ---------------------------------------------------------


def test_last_token_pool_selects_final_valid_position() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert LastTokenPool().name == "last_token"
    pooled = LastTokenPoolModule()(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 3])


# --- NextEvent -------------------------------------------------------------


def test_next_event_declares_its_contract() -> None:
    spec = NextEvent()
    assert spec.required_outputs() == ("logits",)
    assert spec.target_level() == "event"
    assert spec.ignore_index == -100


def test_next_event_targets_are_shifted_by_one() -> None:
    """Position t must be scored against token t+1, never against itself."""
    vocab = 6
    loss_fn = NextEventLoss(ignore_index=-100)
    ids = torch.tensor([[1, 2, 3]])
    batch = _batch([[1, 2, 3]], [[1, 1, 1]])

    # Logits that predict the *next* token perfectly should beat logits that
    # predict the current token.
    shifted = torch.full((1, 3, vocab), -10.0)
    current = torch.full((1, 3, vocab), -10.0)
    for t in range(3):
        if t + 1 < 3:
            shifted[0, t, int(ids[0, t + 1])] = 10.0
        current[0, t, int(ids[0, t])] = 10.0

    from tfm.contracts.outputs import ModelOutput

    hidden = torch.zeros(1, 3, 2)
    shifted_loss = loss_fn(ModelOutput(hidden_states=hidden, logits=shifted), batch)
    current_loss = loss_fn(ModelOutput(hidden_states=hidden, logits=current), batch)
    assert float(shifted_loss) < float(current_loss)


def test_next_event_ignores_padding_targets() -> None:
    loss_fn = NextEventLoss(ignore_index=-100)
    from tfm.contracts.outputs import ModelOutput

    logits = torch.randn(1, 4, 7)
    hidden = torch.zeros(1, 4, 2)
    padded = _batch([[1, 2, 0, 0]], [[1, 1, 0, 0]])
    base = loss_fn(ModelOutput(hidden_states=hidden, logits=logits), padded)

    # Changing the token ids under padding must not change the loss.
    other = _batch([[1, 2, 5, 6]], [[1, 1, 0, 0]])
    assert torch.allclose(base, loss_fn(ModelOutput(hidden_states=hidden, logits=logits), other))


def test_flat_decoder_requires_exactly_one_next_event_objective() -> None:
    with pytest.raises(IncompatibleCompositionError, match="exactly one NextEvent"):
        TrxFM(granularity=Flat(Decoder()), objective=[NextEvent(), NextEvent()])


# --- Binary ----------------------------------------------------------------


def test_binary_head_shape_and_spec() -> None:
    spec = Binary()
    assert spec.name == "binary"
    assert spec.target_level == "sequence"
    head = BinaryHead(8)
    assert head(torch.randn(3, 8)).shape == (3,)


def test_binary_head_prediction_is_exposed_per_name() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)),
        heads={"fraud": Binary(), "churn": Binary()},
    )
    out = model(_batch([[1, 2, 3]], [[1, 1, 1]]), compute_loss=False)
    assert out.predictions is not None
    assert set(out.predictions) == {"fraud", "churn"}
    assert out.predictions["fraud"].shape == (1,)


def test_binary_head_is_forward_only_without_a_supervised_objective() -> None:
    """Documents the default: heads produce predictions but receive no gradient."""
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)),
        heads={"fraud": Binary()},
    )
    batch = _batch([[1, 2, 3]], [[1, 1, 1]], fraud_labels=torch.tensor([1.0]))
    model(batch).loss.backward()
    assert model.task_heads["fraud"].proj.weight.grad is None


def test_supervised_head_gives_binary_a_gradient() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)),
        objective=[NextEvent(), SupervisedHead(head="fraud")],
        heads={"fraud": Binary()},
    )
    batch = _batch(
        [[1, 2, 3], [4, 5, 0]], [[1, 1, 1], [1, 1, 0]], fraud_labels=torch.tensor([1.0, 0.0])
    )
    out = model(batch)
    assert out.losses is not None
    assert set(out.losses) == {"causal_next_event", "supervised_head:fraud"}
    assert torch.allclose(out.loss, sum(out.losses.values(), start=torch.zeros(())))
    out.loss.backward()
    grad = model.task_heads["fraud"].proj.weight.grad
    assert grad is not None and float(grad.norm()) > 0.0


def test_supervised_head_probe_mode_keeps_head_loss_off_the_backbone() -> None:
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)),
        objective=[NextEvent(), SupervisedHead(head="fraud", detach_backbone=True)],
        heads={"fraud": Binary()},
    )
    batch = _batch([[1, 2, 3]], [[1, 1, 1]], fraud_labels=torch.tensor([1.0]))
    out = model(batch)
    # Only the head loss, so no path into the backbone when detached.
    out.losses["supervised_head:fraud"].backward()
    assert model.task_heads["fraud"].proj.weight.grad is not None
    assert model.backbone.embed.weight.grad is None


def test_supervised_head_requires_a_matching_declared_head() -> None:
    with pytest.raises(IncompatibleCompositionError, match="no matching task head"):
        TrxFM(
            granularity=Flat(Decoder()),
            objective=[NextEvent(), SupervisedHead(head="chargeback")],
            heads={"fraud": Binary()},
        )


def test_supervised_head_raises_instead_of_inventing_labels() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)),
        objective=[NextEvent(), SupervisedHead(head="fraud")],
        heads={"fraud": Binary()},
    )
    with pytest.raises(MissingLabelError, match="fraud_labels"):
        model(_batch([[1, 2, 3]], [[1, 1, 1]]))


def test_heads_must_be_binary() -> None:
    with pytest.raises(IncompatibleCompositionError, match="must be Binary"):
        TrxFM(granularity=Flat(Decoder()), heads={"fraud": object()})  # type: ignore[dict-item]


# --- TrxFM -----------------------------------------------------------------


def test_trxfm_default_composition_is_the_core_surface() -> None:
    model = TrxFM(granularity=Flat(Decoder(hidden_size=16, num_heads=4, vocab_size=20)))
    assert isinstance(model.position, NoPE)
    assert isinstance(model.attention, EagerAttention)
    assert isinstance(model.objective_spec, NextEvent)
    assert isinstance(model.readout, LastTokenPool)
    config = model.to_config()
    assert config["position"] == "nope"
    assert config["attention"] == "eager"
    assert config["objective"] == "next_event"
    assert config["readout"] == "last_token"
    assert config["head_objectives"] == []


class _RopeBackbone(nn.Module):
    """Stands in for a backbone that applies rotary positions itself."""

    position = "rope"
    hidden_size = 16
    vocab_size = 20

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, causal: bool = True) -> Tensor:
        return self.embed(input_ids)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.head(hidden)


def test_to_config_reports_the_declared_position_and_head_objectives() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(hidden_size=16, num_heads=4, vocab_size=20)),
        position=RoPE(),
        objective=[NextEvent(), SupervisedHead(head="fraud", weight=0.5)],
        heads={"fraud": Binary()},
        backbone=_RopeBackbone(),
    )
    config = model.to_config()
    assert config["position"] == "rope"
    assert config["head_objectives"] == [{"head": "fraud", "weight": 0.5, "detach_backbone": False}]


def test_trxfm_exposes_hidden_and_vocab_size() -> None:
    model = TrxFM(granularity=Flat(Decoder(hidden_size=32, num_heads=4, vocab_size=48)))
    assert model.hidden_size == 32
    assert model.vocab_size == 48


def test_trxfm_rejects_incomplete_optional_backbone() -> None:
    class NoDims(nn.Module):
        def logits(self, hidden: Tensor) -> Tensor:
            return hidden

    with pytest.raises(IncompatibleCompositionError, match="hidden_size and vocab_size"):
        TrxFM(granularity=Flat(Decoder()), backbone=NoDims())
