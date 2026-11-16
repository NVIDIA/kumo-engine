# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import get_args

import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.types.features import TimeUnit
from kumo_core.features.time import get_rel_time


@withCUDA
@pytest.mark.parametrize('time_unit', get_args(TimeUnit))
def test_get_rel_time(time_unit: TimeUnit, device: torch.device) -> None:
    time = torch.arange(5, device=device).view(-1, 1)
    seed_time = 2 * time

    out = get_rel_time(time, seed_time, time_unit)
    assert out.size() == (5, 1)
    assert out.dtype == torch.float
    assert out.device == device

    expected = torch.arange(5, dtype=torch.float, device=device).view(-1, 1)
    if time_unit == 'seconds':
        expected *= 1
    elif time_unit == 'minutes':
        expected /= 60
    elif time_unit == 'hours':
        expected /= 60 * 60
    elif time_unit == 'days':
        expected /= 60 * 60 * 24
    elif time_unit == 'weeks':
        expected /= 60 * 60 * 24 * 7
    elif time_unit == 'months':
        expected /= 60 * 60 * 24 * 30
    else:
        raise AssertionError()

    assert torch.allclose(out, expected)


def test_unsupported_time_unit() -> None:
    with pytest.raises(ValueError, match="Unsupported time unit"):
        get_rel_time(
            torch.empty(0, 1),
            torch.empty(0),
            time_unit='years',
        )
