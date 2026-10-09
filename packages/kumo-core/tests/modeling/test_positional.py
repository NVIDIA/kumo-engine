# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.modeling.positional import PositionalEncoding


@withCUDA
@pytest.mark.parametrize('num_positions', [4])
@pytest.mark.parametrize('channels', [2])
def test_positional_encoding(
    num_positions: int,
    channels: int,
    device: torch.device,
) -> None:
    model = PositionalEncoding(num_positions, channels).to(device)
    assert str(model) == f'PositionalEncoding({num_positions}, {channels})'

    out = model()
    assert out.dtype == torch.float
    assert out.device == device
    assert out.size() == (num_positions, channels)

    pos = torch.arange(num_positions, dtype=out.dtype, device=device)
    assert torch.allclose(out[:, 0], pos.sin())
    assert torch.allclose(out[:, 1], pos.cos())


def test_odd_channels() -> None:
    with pytest.raises(ValueError, match="with odd channel size"):
        PositionalEncoding(4, 3)
