# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pytest
import torch
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.relative_time import RelTimeEncoder


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
@pytest.mark.parametrize('use_seq_id', [True, False])
def test_enc_mode(
    enc_mode: Literal['fuse', 'column'],
    use_seq_id: bool,
    device: torch.device,
) -> None:
    """Test the head linear layer of FusedNumericalEncoder."""
    num_rows = 10
    num_cols = 3
    feat = torch.randint(0, 1000, (num_rows, num_cols), device=device)
    feat[-1] = -1  # N/A handling
    out_channels = 8

    encoder = RelTimeEncoder(
        num_cols=num_cols,
        out_channels=out_channels,
        enc_mode=enc_mode,
        use_seq_id=use_seq_id,
    ).to(device)

    seed_time = torch.full((num_rows, ), fill_value=int(feat.max()),
                           device=device)
    batch = torch.tensor(
        [0] * (num_rows // 2) + [1] * (num_rows // 2),
        device=device,
    )
    out = encoder(feat, seed_time, batch)
    assert out.isnan().sum() == 0

    if enc_mode == 'fuse':
        assert out.size() == (num_rows, out_channels)
    else:
        assert enc_mode == 'column'
        assert out.size() == (num_rows, num_cols, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
@pytest.mark.parametrize('use_seq_id', [True, False])
def test_gradient(
    enc_mode: Literal['fuse', 'column'],
    use_seq_id: bool,
    device: torch.device,
) -> None:
    num_rows = 10
    num_cols = 3
    feat = torch.randint(0, 1000, (num_rows, num_cols), device=device)
    feat[-1] = -1  # N/A handling
    out_channels = 8

    encoder = RelTimeEncoder(
        num_cols=num_cols,
        out_channels=out_channels,
        enc_mode=enc_mode,
        use_seq_id=use_seq_id,
    ).to(device)

    seed_time = torch.full((2, ), fill_value=int(feat.max()), device=device)
    batch = torch.tensor(
        [0] * (num_rows // 2) + [1] * (num_rows // 2),
        device=device,
    )

    with modeling_context.explaining():
        out = encoder(feat, seed_time, batch)

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert grad is not None
    assert grad.size() == (num_rows, num_cols)
    assert grad[-1].equal(torch.zeros_like(grad[-1]))

    # Test uniformative relative time:
    with modeling_context.explaining():
        out = encoder(
            seed_time[batch].view(-1, 1).expand(-1, num_cols),
            seed_time,
            batch,
        )

    out.backward(torch.randn_like(out))
    grad = encoder.get_grad_wrt_input()
    if use_seq_id:
        assert (grad != 0.0).any()
    else:
        assert (grad == 0.0).all()
