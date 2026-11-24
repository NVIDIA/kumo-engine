# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`TrxFM.to_config()` describes the composition that runs, and the cost estimate trusts it.

With an optional backbone the `Decoder` spec is validated but never built, so
`to_config()` reports the backbone's own `describe()` instead of the spec, and
`cost_from_composition` refuses a backbone whose blocks it does not model.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from tfm import Decoder, Flat, TrxFM
from tfm.contracts.errors import IncompatibleCompositionError, SchemaError
from tfm.diagnose.cost import cost_from_composition
from tfm.extras.capability import extra_available
from tfm.position.rope import RoPE

VOCAB = 32


class _Backbone(nn.Module):
    """A minimal optional backbone: hidden states and logits, nothing else."""

    hidden_size = 8
    vocab_size = VOCAB

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, causal: bool) -> Tensor:
        return torch.zeros(*input_ids.shape, self.hidden_size)

    def logits(self, hidden: Tensor) -> Tensor:
        return torch.zeros(*hidden.shape[:-1], VOCAB)


class _DescribedBackbone(_Backbone):
    def describe(self) -> dict[str, object]:
        return {"depth": 3, "hidden_size": self.hidden_size}


class _MisdescribedBackbone(_Backbone):
    def describe(self) -> object:
        return ["depth", 3]


def _decoder() -> Flat:
    return Flat(Decoder(depth=1, hidden_size=8, num_heads=2, vocab_size=VOCAB, max_seq_len=16))


def test_the_eager_decoder_reports_its_spec() -> None:
    config = TrxFM(granularity=_decoder()).to_config()

    assert config["backbone"] == "DecoderModule"
    assert config["granularity"] == {
        "type": "flat",
        "decoder": {
            "depth": 1,
            "mask": "causal",
            "hidden_size": 8,
            "num_heads": 2,
            "dropout": 0.1,
            "vocab_size": VOCAB,
            "max_seq_len": 16,
            "ffn_size": None,
        },
    }


def test_an_optional_backbone_without_describe_reports_no_decoder() -> None:
    config = TrxFM(granularity=_decoder(), backbone=_Backbone()).to_config()

    assert config["backbone"] == "_Backbone"
    assert config["granularity"] == {"type": "flat", "decoder": None}


def test_an_optional_backbone_reports_its_own_describe() -> None:
    config = TrxFM(granularity=_decoder(), backbone=_DescribedBackbone()).to_config()

    assert config["granularity"] == {"type": "flat", "decoder": {"depth": 3, "hidden_size": 8}}


def test_describe_must_return_a_dict() -> None:
    model = TrxFM(granularity=_decoder(), backbone=_MisdescribedBackbone())

    with pytest.raises(IncompatibleCompositionError, match=r"describe\(\) must return a dict"):
        model.to_config()


def test_cost_counts_the_eager_decoder() -> None:
    report = cost_from_composition(TrxFM(granularity=_decoder()).to_config())

    assert report.depth == 1
    assert report.seq_len == 16


def test_cost_refuses_a_backbone_it_does_not_model() -> None:
    config = TrxFM(granularity=_decoder(), backbone=_DescribedBackbone()).to_config()

    with pytest.raises(SchemaError, match="counts the eager DecoderModule"):
        cost_from_composition(config, seq_len=16)


@pytest.mark.skipif(not extra_available("transformers"), reason="transformers extra not installed")
def test_the_llama_backbone_describes_its_own_dimensions() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from tfm.extras.huggingface import HuggingFaceCausalBackbone

    torch.manual_seed(0)
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=16,
            intermediate_size=40,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    )
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=VOCAB)),
        position=RoPE(),
        backbone=HuggingFaceCausalBackbone(llama),
    )

    config = model.to_config()

    assert config["backbone"] == "HuggingFaceCausalBackbone"
    assert config["granularity"] == {
        "type": "flat",
        "decoder": {
            "depth": 2,
            "mask": "causal",
            "hidden_size": 16,
            "num_heads": 4,
            "num_key_value_heads": 2,
            "dropout": 0.0,
            "vocab_size": VOCAB,
            "max_seq_len": 64,
            "ffn_size": 40,
            "tie_word_embeddings": False,
        },
    }
