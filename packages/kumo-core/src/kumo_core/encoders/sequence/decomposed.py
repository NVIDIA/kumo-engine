# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register

from kumo_core.encoders.sequence.base import SequenceEncoder


@register(with_target=True)
class DecomposedSequenceEncoder(SequenceEncoder, ConfigMixin):
    r"""The decomposition sequence encoder from the
    `"Are Transformers Effective for Time Series Forecasting?"
    <https://arxiv.org/abs/2205.13504>`_ paper.

    :class:`DecomposedSequenceEncoder` decomposes the sequence into a
    trend component by a moving average kernel and a remainder (seasonal)
    component. Then, two one-layer linear layers are applied to each component
    and summed up to get the final prediction.

    Args:
        kernel_size: The kernel size of the moving average
            kernel.
    """
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        length: int | None = None,
        kernel_size: int = 12,
    ) -> None:
        super().__init__(in_channels, out_channels, length, kernel_size)

    def init_modules(self) -> None:
        self.avg = torch.nn.AvgPool1d(
            kernel_size=2 * self.kernel_size + 1,
            stride=1,
            padding=self.kernel_size,
            count_include_pad=True,
        )
        self.trend_lin = torch.nn.Linear(
            self.length * self.in_channels,
            self.out_channels,
        )
        self.seasonal_lin = torch.nn.Linear(
            self.length * self.in_channels,
            self.out_channels,
            bias=False,
        )

    def reset_parameters(self) -> None:
        self.trend_lin.reset_parameters()
        self.seasonal_lin.reset_parameters()

    def forward(self, seq: Tensor) -> Tensor:
        if seq.dim() == 2:
            assert self.in_channels == 1
            seq = seq.unsqueeze(-1)

        seq_perm = seq.permute(0, 2, 1)
        seq_trend = self.avg(seq_perm.nan_to_num())
        # Only consider non-N/A values during averaging.
        # We compute the average over the local sequence and divide this by the
        # average of valid entries in the local sequence. That means, we are
        # basically converting the average into a sum, and divide by the real
        # count of valid entries.
        seq_trend = seq_trend / self.avg((~seq_perm.isnan()).float())
        seq_trend = seq_trend.permute(0, 2, 1).nan_to_num()

        seq_seasonal = (seq - seq_trend).nan_to_num()

        out_trend = self.trend_lin(seq_trend.flatten(start_dim=1))
        out_seasonal = self.seasonal_lin(seq_seasonal.flatten(start_dim=1))

        return out_trend + out_seasonal
