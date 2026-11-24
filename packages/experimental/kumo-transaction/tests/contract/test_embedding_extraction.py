# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction settings change speed, not which state is read."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from tfm.adaptation.batching import select_rows
from tfm.config import load_recipe_config
from tfm.inference.embedding import extract_embeddings, trim_padding
from tfm.runtime import build_model, prepare_data
from tfm.training.reproducibility import seed_everything

CONFIG = "recipes/blueprint_v1/config.smoke.yaml"
CPU = torch.device("cpu")


def _setup():
    config = load_recipe_config(CONFIG)
    seed_everything(config.seed)
    data = prepare_data(config)
    model = build_model(config, data.tokenizer.vocab_size)
    return data, model


def test_trimming_drops_only_columns_no_row_attends_to() -> None:
    data, _ = _setup()
    batch = data.batches[0]
    trimmed = trim_padding(batch)
    width = int(batch.attention_mask.sum(dim=1).max())
    assert trimmed.input_ids.shape == (batch.batch_size, width)
    assert torch.equal(trimmed.input_ids, batch.input_ids[:, :width])
    assert int(trimmed.attention_mask.sum()) == int(batch.attention_mask.sum())
    packed = dataclasses.replace(batch, segment_ids=torch.zeros_like(batch.input_ids))
    assert trim_padding(packed) is packed


def test_batch_size_and_trimming_leave_embeddings_unchanged() -> None:
    data, model = _setup()
    batch = data.batches[0]
    short = select_rows(batch, range(batch.batch_size))
    short.attention_mask[0, 5:] = 0
    reference = extract_embeddings(model, short, device=CPU, minibatch_size=1).vectors
    batched = extract_embeddings(model, short, device=CPU, minibatch_size=7).vectors
    np.testing.assert_allclose(batched, reference, rtol=1e-4, atol=1e-5)


def test_bf16_extraction_is_close_to_fp32() -> None:
    data, model = _setup()
    batch = data.batches[0]
    fp32 = extract_embeddings(model, batch, device=CPU).vectors
    bf16 = extract_embeddings(model, batch, device=CPU, precision="bf16").vectors
    assert bf16.dtype == np.float32
    cosine = (fp32 * bf16).sum(1) / (np.linalg.norm(fp32, axis=1) * np.linalg.norm(bf16, axis=1))
    assert cosine.min() > 0.99
