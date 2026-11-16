# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch_geometric.testing import withCUDA

from kumo_core.training.supervision.tensor_ops import batched_arange


@withCUDA
def test_batched_arange(device: torch.device) -> None:
    count = torch.tensor([0, 2, 4, 2, 0, 0, 2], device=device)
    batch, arange = batched_arange(count)

    assert batch.device == device
    assert arange.device == device
    assert batch.tolist() == [1, 1, 2, 2, 2, 2, 3, 3, 6, 6]
    assert arange.tolist() == [0, 1, 0, 1, 2, 3, 0, 1, 0, 1]
