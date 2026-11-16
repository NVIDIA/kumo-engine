# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kumo.utils.sampling import random_sample_from_mask


def test_random_sample_from_mask():
    a = torch.rand(5000) > 0.5
    if a.sum() < 2000:
        a = torch.logical_not(a)
    assert a.sum() > 2000

    b = random_sample_from_mask(a, 1000)
    assert b.sum() == 1000
    assert torch.logical_and(a, b).sum() == 1000
