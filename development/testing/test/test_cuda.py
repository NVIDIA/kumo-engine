# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kumo.testing import withCUDA


@withCUDA
def test_cuda(device):
    x = torch.empty((5, 5), device=device)
    assert x.device == device
