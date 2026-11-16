# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.encoders.sequence.normalized import NormalizedSequenceEncoder


@withCUDA
@pytest.mark.parametrize('out_channels', [16])
@pytest.mark.parametrize('length', [1, 4])
def test_normalized_sequence_encoder(
    out_channels: int,
    length: int,
    device: torch.device,
) -> None:
    model = NormalizedSequenceEncoder(
        in_channels=1,
        out_channels=out_channels,
        length=length,
    ).to(device)

    model.reset_parameters()
    assert str(model) == (f'NormalizedSequenceEncoder(1, {out_channels}, '
                          f'length={length})')

    seq = torch.arange(2 * length, dtype=torch.float, device=device)

    out = model(seq.view(-1, length, 1))
    assert out.device == device
    assert out.size() == (2, out_channels)

    out = model(seq.view(-1, length))
    assert out.device == device
    assert out.size() == (2, out_channels)


def test_invalid_normalized_sequence_encoder() -> None:
    with pytest.raises(ValueError, match="only supports univariate sequences"):
        NormalizedSequenceEncoder(in_channels=2, out_channels=16, length=4)


@withCUDA
def test_normalized_sequence_encoder_na(device: torch.device) -> None:
    model = NormalizedSequenceEncoder(1, 16, length=4).to(device)
    seq = torch.tensor([
        [float('NaN'), 2, 3, float('NaN')],
        [5, 6, float('NaN'), float('NaN')],
        [float('NaN'), float('NaN'),
         float('NaN'), float('NaN')],
    ], device=device)
    out = model(seq)
    assert out.device == device
    assert out.size() == (3, 16)
    assert not out.isnan().any()
    last_value = torch.tensor([[3.0], [6.0], [0.0]], device=device)
    expected = model.lin((seq - last_value).nan_to_num()) + last_value
    assert out.allclose(expected)
