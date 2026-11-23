# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padded tensor storage preserves row boundaries, values and fill positions."""

from math import prod

import numpy as np
import pytest
import torch

from kumo.pytorch.contract.batch import Padded
from kumo.pytorch.output import layout_array


@pytest.mark.parametrize("lengths", ((), (0, 0), (2, 0, 1), (1, 4, 2)))
@pytest.mark.parametrize("trailing", ((), (2,), (2, 3)))
@pytest.mark.parametrize("dtype", (torch.int64, torch.float32, torch.bfloat16, torch.bool))
def test_padded_tensor_layout_preserves_rows(lengths, trailing, dtype):
    boundaries = np.array((0, *np.cumsum(lengths)), dtype=np.int64)
    values = torch.arange(sum(lengths) * prod(trailing)).reshape(sum(lengths), *trailing)
    values = values.to(dtype)
    fill = True if dtype == torch.bool else -7
    expected = torch.full((len(lengths), max(lengths, default=0), *trailing), fill, dtype=dtype)
    for row, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        expected[row, : stop - start] = values[start:stop]

    actual = layout_array(values, boundaries, Padded(), fill=fill)

    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
