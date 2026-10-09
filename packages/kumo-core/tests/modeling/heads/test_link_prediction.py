# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import is_dataclass

import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.modeling.heads.link_prediction import LinkPredHead


@withCUDA
@pytest.mark.parametrize('normalize_vectors', [False, True])
@pytest.mark.parametrize('hidden_channels', [None, 6])
@pytest.mark.parametrize('num_layers', [1])
@pytest.mark.parametrize('dropout', [0.0])
@pytest.mark.parametrize('act', [None, 'relu'])
@pytest.mark.parametrize('act_first', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
@pytest.mark.parametrize('share_negatives', [False, True])
def test_link_pred_head(
    normalize_vectors: bool,
    hidden_channels: int | None,
    num_layers: int,
    dropout: float,
    act: str | None,
    act_first: bool,
    norm: str | None,
    share_negatives: bool,
    device: torch.device,
) -> None:
    model = LinkPredHead(
        in_channels=4,
        use_rhs_projector=True,
        use_mlp_head=False,
        embedding_dim=8,
        normalize_vectors=normalize_vectors,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        act_first=act_first,
        norm=norm,
    ).to(device)
    model.reset_parameters()

    assert str(model) == ('LinkPredHead(4, 8, use_rhs_projector=True, '
                          'use_mlp_head=False)')

    out = model(
        lhs_x=torch.randn(3, 4, device=device),
        rhs_pos_x=torch.randn(3, 4, device=device),
        rhs_neg_x=torch.randn(3, 2, 4, device=device),
        share_negatives=share_negatives,
    )

    assert isinstance(out.pos_score, Tensor)
    assert out.pos_score.device == device
    assert out.pos_score.size() == (3, )
    assert isinstance(out.neg_score, Tensor)
    assert out.neg_score.device == device
    if share_negatives:
        assert out.neg_score.size() == (3, 6)
    else:
        assert out.neg_score.size() == (3, 2)
    assert isinstance(out.lhs_x, Tensor)
    assert out.lhs_x.device == device
    assert out.lhs_x.size() == (3, 8)
    assert isinstance(out.rhs_pos_x, Tensor)
    assert out.rhs_pos_x.device == device
    assert out.rhs_pos_x.size() == (3, 8)
    assert isinstance(out.rhs_neg_x, Tensor)
    assert out.rhs_neg_x.device == device
    if share_negatives:
        assert out.rhs_neg_x.size() == (6, 8)
    else:
        assert out.rhs_neg_x.size() == (3, 2, 8)


def test_link_pred_head_config() -> None:
    lhs_x = torch.randn(3, 4)
    rhs_pos_x = torch.randn(3, 4)
    rhs_neg_x = torch.randn(3, 2, 4)

    model1 = LinkPredHead(
        in_channels=4,
        use_rhs_projector=True,
        use_mlp_head=False,
        embedding_dim=8,
        normalize_vectors=True,
        hidden_channels=None,
        num_layers=2,
        dropout=0.0,
        act='relu',
        act_first=False,
        norm='batch_norm',
    )
    out1 = model1(lhs_x, rhs_pos_x, rhs_neg_x)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.modeling.heads.link_prediction.LinkPredHead')

    model2 = LinkPredHead.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(lhs_x, rhs_pos_x, rhs_neg_x)

    assert torch.allclose(out1.pos_score, out2.pos_score)
    assert torch.allclose(out1.neg_score, out2.neg_score)
