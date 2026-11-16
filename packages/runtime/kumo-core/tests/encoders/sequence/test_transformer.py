# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.encoders.sequence.transformer import TransformerSequenceEncoder


@withCUDA
@pytest.mark.parametrize('in_channels', [1, 8])
@pytest.mark.parametrize('out_channels', [16])
@pytest.mark.parametrize('length', [1, 4])
@pytest.mark.parametrize('hidden_channels', [None, 6])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('num_heads', [1, 4])
@pytest.mark.parametrize('dropout', [0.0])
@pytest.mark.parametrize('act', [None, 'relu', 'leaky_relu', 'elu', 'gelu'])
def test_transformer_sequence_encoder(
    in_channels: int,
    out_channels: int,
    length: int,
    hidden_channels: int | None,
    num_layers: int,
    num_heads: int,
    dropout: float,
    act: str | None,
    device: torch.device,
) -> None:
    model = TransformerSequenceEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        act=act,
    ).to(device)
    model.reset_parameters()
    assert str(model) == (f'TransformerSequenceEncoder({in_channels}, '
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
def test_transformer_sequence_encoder_na(device: torch.device) -> None:
    model = TransformerSequenceEncoder(1, 16, length=4).to(device)
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
