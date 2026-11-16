# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.encoders.sequence.decomposed import DecomposedSequenceEncoder


@withCUDA
@pytest.mark.parametrize('in_channels', [1, 8])
@pytest.mark.parametrize('out_channels', [16])
@pytest.mark.parametrize('length', [1, 4])
@pytest.mark.parametrize('kernel_size', [2, 3])
def test_decomposed_sequence_encoder(
    in_channels: int,
    out_channels: int,
    length: int,
    kernel_size: int,
    device: torch.device,
) -> None:
    model = DecomposedSequenceEncoder(
        in_channels,
        out_channels,
        length,
        kernel_size,
    ).to(device)
    model.reset_parameters()
    assert str(model) == (f'DecomposedSequenceEncoder({in_channels}, '
                          f'{out_channels}, length={length})')

    seq = torch.arange(
        2 * length * in_channels,
        dtype=torch.float,
        device=device,
    ).view(-1, length, in_channels)
    seq = seq.squeeze(-1) if in_channels == 1 else seq

    out = model(seq)
    assert out.device == device
    assert out.size() == (2, out_channels)


@withCUDA
def test_decomposed_sequence_encoder_na(device: torch.device) -> None:
    model = DecomposedSequenceEncoder(1, 16, length=4, kernel_size=1)
    model = model.to(device)
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

    avg = torch.tensor([
        [2.0, 2.5, 2.5, 3.0],
        [5.5, 5.5, 6.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], device=device)
    assert out.allclose(
        model.trend_lin(avg) + model.seasonal_lin((seq - avg).nan_to_num()),
        atol=1e-6,
    )
