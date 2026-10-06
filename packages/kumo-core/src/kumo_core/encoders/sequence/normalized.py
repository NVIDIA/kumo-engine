# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register

from kumo_core.encoders.sequence.base import SequenceEncoder


@register(with_target=True)
class NormalizedSequenceEncoder(SequenceEncoder, ConfigMixin):
    r"""The normalized sequence encoder from the
    `"Are Transformers Effective for Time Series Forecasting?"
    <https://arxiv.org/abs/2205.13504>`_ paper.

    :class:`NormalizedSequenceEncoder` first subtracts the input by the last
    value of the sequence. Then, the input goes through a linear layer, and the
    subtracted part is added back before making the final prediction.
    The subtraction and addition act as a simple normalization for the input
    sequence.
    """
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        length: int | None = None,
    ) -> None:
        super().__init__(in_channels, out_channels, length)

    def init_modules(self) -> None:
        if self.in_channels != 1:
            raise ValueError(f"'{self.__class__.__name__}' only supports "
                             f"univariate sequences")

        self.lin = torch.nn.Linear(self.length, self.out_channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, seq: Tensor) -> Tensor:
        seq = seq.squeeze(-1) if seq.dim() == 3 else seq

        # Find the last non-NaN value in every sequence:
        last_index = (~seq.isnan()).cumsum(1).argmax(1, keepdim=True)
        last_value = seq.gather(1, last_index)
        out = self.lin((seq - last_value).nan_to_num())
        return out + last_value.nan_to_num()
