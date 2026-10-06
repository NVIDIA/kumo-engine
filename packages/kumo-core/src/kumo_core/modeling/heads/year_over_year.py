# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register

from kumo_core.modeling.base import Module


@register(with_target=True)
class YearOverYearEncoder(Module, ConfigMixin):
    r"""Encodes the year-over-year features via a 1D-CNN.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        kernel_size: The kernel size of the 1D-CNN.
    """
    LAZY_ATTRS = {'in_channels', 'out_channels'}

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int = 5,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size)

    def init_modules(self) -> None:
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2 * self.kernel_size + 1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=2 * self.kernel_size + 1,
        )

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, past: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            past: The past year sequence of shape
                ``[batch_size, length, in_channels]``.
        """
        out = past.transpose(1, 2)
        out = self.conv1(out)
        out = out.relu()
        out = self.conv2(out)
        out = out.transpose(1, 2)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
