# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor

from kumo_core.encoders.time import TimeEncoder


class PredictionTimeEncoder(torch.nn.Module):
    r"""Encodes the prediction time for use in model heads.

    Args:
        out_channels: Size of each output sample.
        unix_step_size: The UNIX step size in case of forecasting heads.
        kernel_size: The kernel size of the 1D-CNN.
    """
    def __init__(
        self,
        out_channels: int = 32,
        unix_step_size: int | None = None,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.time_encoder = TimeEncoder(unix_step_size)

        self.conv1 = torch.nn.Conv1d(
            in_channels=TimeEncoder.out_channels,
            out_channels=out_channels,
            kernel_size=2 * kernel_size + 1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2 * kernel_size + 1,
        )

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, time: Tensor, length: int) -> Tensor:
        r"""The forward pass.

        Args:
            time: The prediction time per entity.
            length: The forecast length.
        """
        time_enc = self.time_encoder(
            time=time,
            length=length,
            padding_size=2 * self.kernel_size,
        )

        out = time_enc.transpose(1, 2)
        out = self.conv1(out)
        out = out.relu()
        out = self.conv2(out)
        out = out.transpose(1, 2)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
