# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.encoders.sequence.mlp import MLPSequenceEncoder


@withCUDA
@pytest.mark.parametrize('in_channels', [1, 8])
@pytest.mark.parametrize('out_channels', [16])
@pytest.mark.parametrize('length', [1, 4])
@pytest.mark.parametrize('hidden_channels', [None, 6])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('dropout', [0.0])
@pytest.mark.parametrize('act', [None, 'relu'])
@pytest.mark.parametrize('act_first', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
def test_decomposed_sequence_encoder(
    in_channels: int,
    out_channels: int,
    length: int,
    hidden_channels: int | None,
    num_layers: int,
    dropout: float,
    act: str | None,
    act_first: bool,
    norm: str | None,
    device: torch.device,
) -> None:
    model = MLPSequenceEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        act_first=act_first,
        norm=norm,
    ).to(device)
    model.reset_parameters()
    assert str(model) == (f'MLPSequenceEncoder({in_channels}, {out_channels}, '
                          f'length={length})')

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
def test_mlp_sequence_encoder_na(device: torch.device) -> None:
    model = MLPSequenceEncoder(1, 16, length=4).to(device)
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
    assert out.allclose(model.mlp(seq.nan_to_num()))
