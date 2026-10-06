# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn import MLP

from kumo_core.encoders.sequence.base import SequenceEncoder


@register(with_target=True)
class MLPSequenceEncoder(SequenceEncoder, ConfigMixin):
    r"""An MLP-based sequence encoder.

    Args:
        hidden_channels: Size of each hidden sample. If set to :obj:`None`,
            will set ``hidden_channels = out_channels``.
        num_layers: The number of layers.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
        act_first: If set to :obj:`True`, activation is applied before
            normalization.
        norm: The normalization function to use.
    """
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        length: int | None = None,
        hidden_channels: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.0,
        act: str | None = 'relu',
        act_first: bool = False,
        norm: str | None = 'batch_norm',
    ) -> None:
        super().__init__(in_channels, out_channels, length, hidden_channels,
                         num_layers, dropout, act, act_first, norm)

    def init_modules(self) -> None:
        norm_kwargs: dict[str, Any] | None = None
        if self.norm == 'layer_norm':
            norm_kwargs = dict(mode='node')

        self.mlp = MLP(
            in_channels=self.length * self.in_channels,
            hidden_channels=self.hidden_channels or self.out_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=norm_kwargs,
        )

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()

    def forward(self, seq: Tensor) -> Tensor:
        return self.mlp(seq.flatten(start_dim=1).nan_to_num())
