# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn.resolver import activation_resolver

from kumo_core.modeling.positional import PositionalEncoding
from kumo_core.encoders.sequence.base import SequenceEncoder


@register(with_target=True)
class TransformerSequenceEncoder(SequenceEncoder, ConfigMixin):
    r"""A Transformer-based sequence encoder.

    Args:
        hidden_channels: Size of each hidden sample. If set to :obj:`None`,
            will set ``hidden_channels = out_channels``.
        num_layers: The number of layers.
        num_heads: The number of heads.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
    """
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        length: int | None = None,
        hidden_channels: int | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        act: str | None = 'relu',
    ) -> None:
        super().__init__(in_channels, out_channels, length, hidden_channels,
                         num_layers, num_heads, dropout, act)

    def init_modules(self) -> None:
        self.lin = torch.nn.Linear(self.in_channels, self.out_channels)
        self.pos_enc = PositionalEncoding(self.length, self.out_channels)

        if self.act is not None:
            act = activation_resolver(self.act)
        else:
            act = torch.nn.Identity()

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            layer = torch.nn.TransformerEncoderLayer(
                self.out_channels,
                self.num_heads,
                dim_feedforward=self.hidden_channels or self.out_channels,
                dropout=self.dropout,
                activation=act,
                batch_first=True,
            )
            self.layers.append(layer)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.pos_enc.reset_parameters()
        for layer in self.layers:
            for child in layer.children():
                if isinstance(child, torch.nn.MultiheadAttention):
                    child._reset_parameters()
                elif hasattr(child, 'reset_parameters'):
                    child.reset_parameters()

    def forward(self, seq: Tensor) -> Tensor:
        if seq.dim() == 2:
            assert self.in_channels == 1
            seq = seq.unsqueeze(-1)

        x = self.lin(seq.nan_to_num()) + self.pos_enc().unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x[:, -1]
