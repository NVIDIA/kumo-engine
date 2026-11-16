# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor


class PositionalEncoding(torch.nn.Module):
    r"""A 1-dimensional positional frequency encoding.

    Args:
        num_positions: The number of positions for which to create positional
            encodings.
        channels: Number of channels for the positional encoding. Must be
            divisible by ``2``.
    """
    def __init__(
        self,
        num_positions: int,
        channels: int,
    ) -> None:
        super().__init__()

        if channels % 2 != 0:
            raise ValueError(f"Cannot create a positional encoding with odd "
                             f"channel size (got {channels})")

        self.num_positions = num_positions
        self.channels = channels

        self.pe: Tensor
        self.register_buffer('pe', torch.empty(num_positions, channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pos = torch.arange(
            self.num_positions,
            dtype=self.pe.dtype,
            device=self.pe.device,
        )
        arange = torch.arange(
            self.channels // 2,
            dtype=self.pe.dtype,
            device=self.pe.device,
        )
        mul_term = 1.0 / 10000.0**arange
        self.pe[:, 0::2] = (pos.view(-1, 1) * mul_term.view(1, -1)).sin()
        self.pe[:, 1::2] = (pos.view(-1, 1) * mul_term.view(1, -1)).cos()

    def forward(self) -> Tensor:
        """"""  # noqa: D419
        return self.pe

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_positions}, '
                f'{self.channels})')
