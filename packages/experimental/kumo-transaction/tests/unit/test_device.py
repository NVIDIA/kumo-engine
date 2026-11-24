# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tfm import Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.contracts.batch import TransactionBatch
from tfm.data.collate import minibatches
from tfm.inference.embedding import extract_embeddings
from tfm.training.device import device_environment, resolve_device
from tfm.training.trainer import train_model


def test_resolve_device_cpu() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert device.index is None


def test_resolve_device_auto_matches_availability() -> None:
    device = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected


def test_resolve_device_rejects_unknown_spec() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime.device"):
        resolve_device("tpu")


def test_resolve_device_cuda_requires_runtime() -> None:
    if torch.cuda.is_available():
        device = resolve_device("cuda")
        assert device.type == "cuda"
        env = device_environment(device)
        assert env["gpu_name"]
        assert env["gpu_count"] >= 1
    else:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_eager_train_and_embed_on_cuda() -> None:
    device = resolve_device("cuda")
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=32, dropout=0.0)),
        position=NoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
    )
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        labels=torch.tensor([[2, 3, -100, -100], [5, -100, -100, -100]], dtype=torch.long),
        fraud_labels=torch.tensor([0.0, 1.0]),
        numeric_features=torch.zeros(2, 3),
        entity_ids=("a", "b"),
        sequence_ids=("s0", "s1"),
        event_ids=("e0", "e1"),
    )
    stats = train_model(
        model,
        minibatches(batch, 2),
        max_steps=1,
        learning_rate=1e-3,
        seed=0,
        device=device,
    )
    assert next(model.parameters()).device.type == "cuda"
    assert stats["steps"] == 1.0
    assert math.isfinite(stats["final_loss"])
    table = extract_embeddings(model, batch, device=device)
    assert table.vectors.shape == (2, 16)
    moved = batch.to(device)
    out = model(moved, compute_loss=True)
    assert out.hidden_states is not None and out.hidden_states.device.type == "cuda"
    assert out.embeddings is not None and out.embeddings.device.type == "cuda"
    assert out.loss is not None and out.loss.device.type == "cuda"
