# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn import MLP
from typing_extensions import Self

from kumo_core.modeling.base import Module
from kumo_core.modeling.embedding import Embedding
from kumo_core.modeling.positional import PositionalEncoding
from kumo_core.modeling.heads.prediction_time import PredictionTimeEncoder


@register(with_target=True)
class ForecastingHead(Module, ConfigMixin):
    r"""An implicit function-based forecasting head.

    Forecasts are computed with shared parameters across time but conditioned
    based on a positional encoding as

    .. math::

        \hat{y}_t = f(x, t)

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        past_channels: Size of past sequence embeddings.
        past_year_channels: Size of past year sequence embeddings.
        unix_step_size: The UNIX step size.
        hidden_channels: Size of each hidden sample.
        pos_enc_channels: Size of relative positional encoding along the
            forecasting horizon.
        num_layers: The number of layers.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
        act_first: If set to :obj:`True`, activation is applied before
            normalization.
        norm: The normalization function to use.
        param_keys: The output variables per prediction if probabilistic
            modeling is used. If non-empty, the model returns distribution
            parameters instead of point predictions.
        prediction_time_encodings: If set to :obj:`True`, will use temporal
            encodings of training/prediction table timestamps.
        num_entities: The number of entities.
        handle_new_entities: If set to :obj:`False`, will learn an additional
            entity-specific head.
    """
    LAZY_ATTRS = {
        'in_channels', 'out_channels', 'past_channels', 'past_year_channels',
        'unix_step_size', 'param_keys', 'num_entities'
    }
    MAX_FORECAST_LENGTH = 1_000

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        past_channels: int | None = None,
        past_year_channels: int | None = None,
        unix_step_size: int | None = None,
        hidden_channels: int | None = None,
        pos_enc_channels: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        act: str | None = 'relu',
        act_first: bool = False,
        norm: str | None = 'batch_norm',
        param_keys: list[str] | None = None,
        prediction_time_encodings: bool = False,
        num_entities: int | None = None,
        handle_new_entities: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, past_channels,
                         past_year_channels, unix_step_size, hidden_channels,
                         pos_enc_channels, num_layers, dropout, act, act_first,
                         norm, param_keys, prediction_time_encodings,
                         num_entities, handle_new_entities)

    def init_modules(self) -> None:
        in_channels = self.in_channels
        in_channels += self.past_channels
        in_channels += self.past_year_channels

        self.pos_enc = PositionalEncoding(
            self.MAX_FORECAST_LENGTH,
            self.pos_enc_channels or self.in_channels,
        )
        in_channels += self.pos_enc.channels

        self.pred_time_encoder: PredictionTimeEncoder | None = None
        if self.prediction_time_encodings:
            self.pred_time_encoder = PredictionTimeEncoder(
                unix_step_size=self.unix_step_size)
            in_channels += self.pred_time_encoder.out_channels

        norm_kwargs: dict[str, Any] | None = None
        if self.norm == 'layer_norm':
            norm_kwargs = dict(mode='node')

        hidden_channels = self.hidden_channels or self.in_channels
        out_channels = self.out_channels * max(len(self.param_keys), 1)

        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=max(self.num_layers - 1, 1),
            dropout=self.dropout,
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

        self.lookup_embedding: Embedding | None = None
        if not self.handle_new_entities:
            self.lookup_embedding = Embedding(
                num_embeddings=self.num_entities,
                embedding_dim=(hidden_channels + 1) * out_channels,
            )

    def reset_parameters(self) -> None:
        self.pos_enc.reset_parameters()
        if self.pred_time_encoder is not None:
            self.pred_time_encoder.reset_parameters()
        self.mlp.reset_parameters()
        self.lin.reset_parameters()
        if self.lookup_embedding is not None:
            self.lookup_embedding.reset_parameters()

    def order_by(
        self,
        index_map: pd.DataFrame,
    ) -> Self:
        r"""Re-orders the lookup embeddings according to ``index_map``.

        See :meth:`~kumo_core.modeling.embedding.Embedding.order_by` for details.
        """
        if self.lookup_embedding is not None:
            self.lookup_embedding.order_by(
                index_map,
                default_init='zero',
                num_embeddings=self.num_entities,
            )
        return self

    def forward(
        self,
        x: Tensor,
        length: int,
        past: Tensor | None = None,
        past_year: Tensor | None = None,
        pred_time: Tensor | None = None,
        node_id: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        r"""The forward pass.

        Args:
            x: The entity embeddings.
            length: The forecast length.
            past: The past label sequence per entity. Only necessary to input
                in case ``past_length > 0``.
            past_year: The past label sequence of the previous year per entity.
                Only necessary to input in case ``past_year_channels > 0``.
            pred_time: The prediction time per entity. Only necessary to input
                in case ``prediction_time_encodings`` is set.
            node_id: The node ID of entities. Only necessary to input in case
                ``handle_new_entities`` is set to ``False``.
        """
        assert length >= 1
        assert length <= self.MAX_FORECAST_LENGTH

        xs = [
            x.unsqueeze(1).expand(-1, length, -1),
            self.pos_enc()[None, :length].expand(x.size(0), -1, -1),
        ]

        if self.past_channels > 0:
            assert past is not None and past.dim() == 2
            assert past.size(0) == x.size(0)
            assert past.size(1) == self.past_channels
            xs.append(past.unsqueeze(1).expand(-1, length, -1))

        if self.past_year_channels > 0:
            assert past_year is not None and past_year.dim() == 3
            assert past_year.size(0) == x.size(0)
            assert past_year.size(1) == length
            assert past_year.size(2) == self.past_year_channels
            xs.append(past_year)

        if self.pred_time_encoder is not None:
            assert pred_time is not None
            xs.append(self.pred_time_encoder(pred_time, length))

        h = self.mlp(torch.cat(xs, dim=-1).flatten(0, 1))
        out = self.lin(h)
        out = out.view(x.size(0), length, -1)

        if self.lookup_embedding is not None:
            assert node_id is not None
            weight = self.lookup_embedding(node_id)
            out_channels = self.out_channels * max(len(self.param_keys), 1)
            weight, bias = weight[:, out_channels:], weight[:, :out_channels]
            weight = weight.view(weight.size(0), h.size(-1), out_channels)
            bias = bias.view(bias.size(0), 1, out_channels)
            h = h.view(-1, length, h.size(-1))
            out = out + h @ weight + bias

        if len(self.param_keys) == 0:
            return out

        outs = out.split(out.size(-1) // len(self.param_keys), dim=-1)
        return {key: value for key, value in zip(self.param_keys, outs)}

    def __repr__(self) -> str:
        extra_repr = ''
        if len(self.param_keys) > 0:
            extra_repr = f', param_keys={self.param_keys}'
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, past_channels={self.past_channels}, '
                f'past_year_channels={self.past_year_channels}{extra_repr})')
