# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tfm import Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.contracts.batch import TransactionBatch
from tfm.pooling.last_token import LastTokenPoolModule


def test_padding_does_not_affect_pooling() -> None:
    pool = LastTokenPoolModule()
    hidden_short = torch.randn(1, 2, 8)
    mask_short = torch.tensor([[1, 1]])
    hidden_pad = torch.zeros(1, 4, 8)
    hidden_pad[:, :2] = hidden_short
    hidden_pad[:, 2:] = 3.14
    mask_pad = torch.tensor([[1, 1, 0, 0]])
    a = pool(hidden_short, mask_short)
    b = pool(hidden_pad, mask_pad)
    assert torch.allclose(a, b)


def test_causal_future_perturbation_does_not_change_prefix() -> None:
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=2, hidden_size=32, num_heads=4, vocab_size=40, dropout=0.0)),
        position=NoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
    )
    model.eval()
    ids = torch.tensor([[2, 5, 7, 9, 11]], dtype=torch.long)
    mask = torch.ones_like(ids)
    batch = TransactionBatch(input_ids=ids, attention_mask=mask)
    with torch.no_grad():
        base = model(batch, compute_loss=False).hidden_states
        ids2 = ids.clone()
        ids2[0, -1] = 3
        perturbed = model(
            TransactionBatch(input_ids=ids2, attention_mask=mask),
            compute_loss=False,
        ).hidden_states
    assert torch.allclose(base[:, :-1, :], perturbed[:, :-1, :], atol=1e-5)


def test_output_contract_fields() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20, dropout=0.0)),
        position=NoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
    )
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    out = model(batch)
    assert out.hidden_states is not None
    assert out.embeddings is not None
    assert out.logits is not None
    assert out.loss is not None


def test_tokenizer_manifest_roundtrip() -> None:
    import json

    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    tok = TransactionFieldTokenizer()
    payload = json.loads(tok.dumps())
    assert payload["tokenizer"] == "transaction_field_tokenizer"
    assert payload["layout"] == "flat_fields"
    assert payload["vocab_size"] == tok.vocab_size
    restored = TransactionFieldTokenizer(merchant_hash_size=int(payload["merchant_hash_size"]))
    assert restored.vocab_size == tok.vocab_size
    assert restored.pad_id == tok.pad_id
