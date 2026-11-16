# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import is_dataclass

import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.modeling.heads.node_prediction import NodePredHead


@withCUDA
@pytest.mark.parametrize('hidden_channels', [None, 6])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('dropout', [0.0, 0.5])
@pytest.mark.parametrize('act', [None, 'relu', 'leaky_relu', 'elu', 'gelu'])
@pytest.mark.parametrize('act_first', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
@pytest.mark.parametrize('param_keys', [[], ['mean'], ['mean', 'std']])
@pytest.mark.parametrize('prediction_time_encodings', [False, True])
def test_node_pred_head(
    hidden_channels: int | None,
    num_layers: int,
    dropout: float,
    act: str | None,
    act_first: bool,
    norm: str | None,
    param_keys: list[str],
    prediction_time_encodings: bool,
    device: torch.device,
) -> None:
    model = NodePredHead(
        in_channels=4,
        out_channels=1,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        act_first=act_first,
        norm=norm,
        param_keys=param_keys,
        prediction_time_encodings=prediction_time_encodings,
    ).to(device)
    model.reset_parameters()

    if len(param_keys) == 0:
        assert str(model) == 'NodePredHead(4, 1)'
    else:
        assert str(model) == f'NodePredHead(4, 1, param_keys={param_keys})'

    x = torch.randn(3, 4, device=device)
    pred_time: Tensor | None = None
    if prediction_time_encodings is not None:
        unix_year = 365 * NANOSECONDS_IN_DAY
        pred_time = torch.randint(0, unix_year, (3, ), device=device)

    out = model(x, pred_time)
    if len(param_keys) == 0:
        assert isinstance(out, Tensor)
        assert out.device == device
        assert out.size() == (3, 1)
    else:
        assert isinstance(out, dict)
        for key, value in out.items():
            assert key in param_keys
            assert value.device == device
            assert value.size() == (3, 1)


def test_node_pred_head_config() -> None:
    x = torch.randn(3, 4)

    model1 = NodePredHead(
        in_channels=4,
        out_channels=1,
        hidden_channels=None,
        num_layers=2,
        dropout=0.0,
        act='leaky_relu',
        act_first=False,
        norm=None,
        param_keys=[],
        prediction_time_encodings=False,
    )
    out1 = model1(x)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.modeling.heads.node_prediction.NodePredHead')

    model2 = NodePredHead.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(x)

    assert torch.allclose(out1, out2)
