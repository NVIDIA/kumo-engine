# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.modeling.heads.prediction_time import PredictionTimeEncoder


@withCUDA
@pytest.mark.parametrize('out_channels', [32])
@pytest.mark.parametrize('unix_step_size', [None, NANOSECONDS_IN_DAY])
@pytest.mark.parametrize('kernel_size', [0, 5])
@pytest.mark.parametrize('length', [1, 2])
def test_prediction_time_encoder(
    out_channels: int,
    unix_step_size: int | None,
    kernel_size: int,
    length: int,
    device: torch.device,
) -> None:
    model = PredictionTimeEncoder(
        out_channels,
        unix_step_size=unix_step_size,
        kernel_size=kernel_size,
    ).to(device)
    model.reset_parameters()
    assert str(model) == f'PredictionTimeEncoder({out_channels})'

    time = torch.randint(0, 100_000, (2, ), device=device)

    if unix_step_size is None and (length != 1 or kernel_size != 0):
        with pytest.raises(ValueError, match="'unix_step_size' needs to be"):
            model(time, length)
        return

    out = model(time, length)
    assert out.device == device
    assert out.size() == (2, length, out_channels)
