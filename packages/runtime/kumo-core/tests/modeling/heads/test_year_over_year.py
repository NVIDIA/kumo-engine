# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import is_dataclass

import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.modeling.heads.year_over_year import YearOverYearEncoder


@withCUDA
@pytest.mark.parametrize('in_channels', [8])
@pytest.mark.parametrize('out_channels', [32])
@pytest.mark.parametrize('kernel_size', [0, 5])
@pytest.mark.parametrize('length', [1, 2])
def test_year_over_year_encoder(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    length: int,
    device: torch.device,
) -> None:
    model = YearOverYearEncoder(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
    ).to(device)
    model.reset_parameters()
    assert str(model) == f'YearOverYearEncoder({in_channels}, {out_channels})'

    past = torch.randn(2, length + 4 * kernel_size, in_channels, device=device)
    out = model(past)
    assert out.device == device
    assert out.size() == (2, length, out_channels)


def test_year_over_year_encoder_config() -> None:
    x = torch.randn(3, 30, 1)

    model1 = YearOverYearEncoder(
        in_channels=1,
        out_channels=16,
    )
    out1 = model1(x)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.modeling.heads.year_over_year.YearOverYearEncoder')

    model2 = YearOverYearEncoder.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(x)

    assert torch.allclose(out1, out2)
