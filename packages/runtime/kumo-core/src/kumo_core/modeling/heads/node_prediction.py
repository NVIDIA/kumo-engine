# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn import MLP

from kumo_core.modeling.base import Module
from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.modeling.heads.prediction_time import PredictionTimeEncoder


@register(with_target=True)
class NodePredHead(Module, ConfigMixin):
    r"""An MLP-based node prediction head.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        hidden_channels: Size of each hidden sample. If set to :obj:`None`,
            will set ``hidden_channels = in_channels``.
        num_layers: The number of layers.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
        act_first: If set to :obj:`True`, activation is
            applied before normalization.
        norm: The normalization function to use.
        param_keys: The output variables per prediction if probabilistic
            modeling is used. If non-empty, the model returns distribution
            parameters instead of point predictions.
        prediction_time_encodings: If set to :obj:`True`, will use temporal
            encodings of training/prediction table timestamps.
    """
    LAZY_ATTRS = {'in_channels', 'out_channels', 'param_keys'}

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        act: str | None = 'relu',
        act_first: bool = False,
        norm: str | None = 'batch_norm',
        param_keys: list[str] | None = None,
        prediction_time_encodings: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, hidden_channels,
                         num_layers, dropout, act, act_first, norm, param_keys,
                         prediction_time_encodings)

    def init_modules(self) -> None:
        in_channels = self.in_channels

        self.pred_time_encoder: PredictionTimeEncoder | None = None
        if self.prediction_time_encodings:
            self.pred_time_encoder = PredictionTimeEncoder(
                unix_step_size=NANOSECONDS_IN_DAY,  # One day.
            )
            in_channels += self.pred_time_encoder.out_channels

        norm_kwargs: dict[str, Any] | None = None
        if self.norm == 'layer_norm':
            norm_kwargs = dict(mode='node')

        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels or self.in_channels,
            out_channels=self.out_channels * max(len(self.param_keys), 1),
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=norm_kwargs,
        )

    def reset_parameters(self) -> None:
        if self.pred_time_encoder is not None:
            self.pred_time_encoder.reset_parameters()
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        pred_time: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        r"""The forward pass.

        Args:
            x: The entity embeddings.
            pred_time: The prediction time per entity.
        """
        if self.pred_time_encoder is not None:
            assert pred_time is not None
            pred_time_enc = self.pred_time_encoder(pred_time, length=1)
            pred_time_enc = pred_time_enc.squeeze(1)
            x = torch.cat([x, pred_time_enc], dim=-1)

        out = self.mlp(x)

        if len(self.param_keys) == 0:
            return out

        outs = out.split(out.size(-1) // len(self.param_keys), dim=-1)
        return {key: value for key, value in zip(self.param_keys, outs)}

    def __repr__(self) -> str:
        extra_repr = ''
        if len(self.param_keys) > 0:
            extra_repr = f', param_keys={self.param_keys}'
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}{extra_repr})')
