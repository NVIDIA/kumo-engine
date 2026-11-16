# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from kumo_core.modeling.base import Module


class SequenceEncoder(Module, ABC):
    r"""An abstract base class for defining sequence encoders.

    Args:
        in_channels: The channel size of any given element in the sequence,
            *e.g.*, ``1`` in the univariate case.
        out_channels: The output channel size of the *aggregated* sequence.
        length: The length of the sequence.
    """
    LAZY_ATTRS = {'in_channels', 'out_channels', 'length'}

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        length: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, length, *args, **kwargs)

    @abstractmethod
    def reset_parameters(self) -> None:
        pass

    @abstractmethod
    def forward(self, seq: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            seq: The sequence per entity of shape
            ``[batch_size, length, in_channels]``.

        Returns:
            The encoded sequence of shape ``[batch_size, out_channels]``.
        """

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, length={self.length})')
