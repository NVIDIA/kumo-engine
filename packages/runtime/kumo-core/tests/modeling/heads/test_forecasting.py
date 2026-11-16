# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.modeling.heads.forecasting import ForecastingHead


@withCUDA
@pytest.mark.parametrize('out_channels', [1, 8])
@pytest.mark.parametrize('past_channels', [0, 16])
@pytest.mark.parametrize('past_year_channels', [0, 12])
@pytest.mark.parametrize('forecast_length', [1, 3])
@pytest.mark.parametrize('hidden_channels', [None, 6])
@pytest.mark.parametrize('pos_enc_channels', [None, 6])
@pytest.mark.parametrize('norm', [None, 'layer_norm'])
@pytest.mark.parametrize('param_keys', [[], ['mean', 'std']])
@pytest.mark.parametrize('prediction_time_encodings', [True, False])
@pytest.mark.parametrize('handle_new_entities', [False])
def test_forecasting_head(
    out_channels: int,
    past_channels: int,
    past_year_channels: int,
    forecast_length: int,
    hidden_channels: int | None,
    pos_enc_channels: int | None,
    norm: str | None,
    param_keys: list[str],
    prediction_time_encodings: bool,
    handle_new_entities: bool,
    device: torch.device,
) -> None:
    model = ForecastingHead(
        in_channels=4,
        out_channels=out_channels,
        past_channels=past_channels,
        past_year_channels=past_year_channels,
        unix_step_size=NANOSECONDS_IN_DAY,
        hidden_channels=hidden_channels,
        pos_enc_channels=pos_enc_channels,
        norm=norm,
        param_keys=param_keys,
        prediction_time_encodings=prediction_time_encodings,
        num_entities=3,
        handle_new_entities=handle_new_entities,
    ).to(device)
    model.reset_parameters()

    if len(param_keys) == 0:
        assert str(model) == (f'ForecastingHead(4, {out_channels}, '
                              f'past_channels={past_channels}, '
                              f'past_year_channels={past_year_channels})')
    else:
        assert str(model) == (f'ForecastingHead(4, {out_channels}, '
                              f'past_channels={past_channels}, '
                              f'past_year_channels={past_year_channels}, '
                              f'param_keys={param_keys})')

    x = torch.randn(3, 4, device=device)
    node_id: Tensor | None = None
    if not handle_new_entities:
        node_id = torch.tensor([0, 1, 2], device=device)

    past: Tensor | None = None
    if past_channels > 0:
        past = torch.randn(3, past_channels, device=device)

    past_year: Tensor | None = None
    if past_year_channels > 0:
        past_year = torch.randn(3, forecast_length, past_year_channels,
                                device=device)

    pred_time: Tensor | None = None
    if prediction_time_encodings is not None:
        unix_year = 365 * NANOSECONDS_IN_DAY
        pred_time = torch.randint(0, unix_year, (3, ), device=device)

    out = model(x, forecast_length, past, past_year, pred_time, node_id)

    if len(param_keys) == 0:
        assert isinstance(out, Tensor)
        assert out.device == device
        assert out.size() == (3, forecast_length, out_channels)
    else:
        assert isinstance(out, dict)
        for key, value in out.items():
            assert key in param_keys
            assert value.device == device
            assert value.size() == (3, forecast_length, out_channels)
