# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tfm import Binary, Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.contracts.batch import TransactionBatch
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.pooling.last_token import LastTokenPoolModule


def _batch(ids: list[list[int]], mask: list[list[int]]) -> TransactionBatch:
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention = torch.tensor(mask, dtype=torch.long)
    labels = input_ids.clone()
    labels[attention == 0] = -100
    return TransactionBatch(input_ids=input_ids, attention_mask=attention, labels=labels)


def test_decoder_forward_shapes() -> None:
    model = TrxFM(
        granularity=Flat(Decoder(depth=2, hidden_size=32, num_heads=4, vocab_size=50, dropout=0.0)),
        position=NoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
        heads={"fraud": Binary()},
    )
    batch = _batch([[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 1, 1, 0], [1, 1, 0, 0]])
    out = model(batch)
    assert out.hidden_states.shape == (2, 4, 32)
    assert out.embeddings is not None and out.embeddings.shape == (2, 32)
    assert out.logits is not None and out.logits.shape == (2, 4, 50)
    assert out.loss is not None


def test_last_token_pool_ignores_padding() -> None:
    pool = LastTokenPoolModule()
    states = torch.zeros(1, 4, 3)
    states[0, 1] = torch.tensor([9.0, 8.0, 7.0])
    states[0, 3] = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([[1, 1, 0, 0]])
    pooled = pool(states, mask)
    assert torch.equal(pooled[0], states[0, 1])


def test_metrics_keys() -> None:
    import numpy as np

    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    m = classification_metrics(y, s, k=2)
    for key in ("roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "calibration"):
        assert key in m
