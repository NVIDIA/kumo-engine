# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tests.e2e.throughput import BATCH_SIZE, _window


def test_timed_window_counts_batches_even_when_a_batch_has_no_labels():
    weight = torch.nn.Parameter(torch.ones(()))
    batch = SimpleNamespace(
        batch_size=BATCH_SIZE,
        metadata={"is_padding": torch.zeros(BATCH_SIZE, dtype=torch.bool)},
        inputs={},
    )
    stream = iter([batch, batch, batch])
    losses = iter([None, weight.square()])
    model = SimpleNamespace(batch_loss=lambda batch, device: next(losses))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    result = _window(stream, model, optimizer, torch.device("cpu"), 2)
    assert (result["batches"], result["samples"], result["optimizer_updates"]) == (2, 64, 1)
    assert weight.item() < 1 and next(stream) is batch
