# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from typing import Generic, Literal, TypeVar

import torch
from kumoapi.encoder import Encoder
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor

import kumo_core.modeling.context as modeling_context
from kumo_core.modeling.gradient import GradientMixin

InputT = TypeVar('InputT', Tensor, MultiNestedTensor, MultiEmbeddingTensor)
OutputT = TypeVar('OutputT', Tensor, MultiNestedTensor, MultiEmbeddingTensor)


class FusedEncoder(torch.nn.Module, GradientMixin, Generic[InputT, OutputT]):
    r"""A fused encoder that takes a dictionary of encoders and parallelizes
    them.

    Args:
        col_encoder_dict: A mapping from column names to encoders to fuse.
            Encoders need to be operate on the same stype.
        exclude_cols: A list of column names to be excluded.
        out_channels: The number of output channels.
        enc_mode: If ``"fuse"``, the output size is
            ``[num_rows, out_channels]``. If ``"column"``, the output size is
            ``[num_rows, num_cols, out_channels]``.
        device: The device to use.
    """
    SUPPORTED_ENCODERS: set[type[Encoder]]

    def __init__(
        self,
        col_encoder_dict: dict[str, Encoder],
        exclude_cols: list[str],
        out_channels: int,
        enc_mode: Literal["fuse", "column"],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.col_names = list(col_encoder_dict.keys())
        self.num_cols = len(col_encoder_dict)
        self.exclude_cols = exclude_cols
        self.out_channels = out_channels
        self.enc_mode = enc_mode
        for col_name, encoder in col_encoder_dict.items():
            if encoder.__class__ not in self.SUPPORTED_ENCODERS:
                raise ValueError(
                    f"The column '{col_name}' has "
                    f"{encoder.__class__.__name__}, but "
                    f"{self.__class__.__name__} doesn't support this encoder "
                    f"and only supports: {self.SUPPORTED_ENCODERS}.")
        if enc_mode not in {'fuse', 'column'}:
            raise ValueError(f"Unsupported encoder mode '{enc_mode}'.")

        self.exclude_cols_mask: Tensor
        self.register_buffer(
            'exclude_cols_mask',
            self._create_exclude_cols_mask(device=device),
        )

    def validate_col_order(self, col_names: list[str] | None) -> None:
        if col_names is None:
            return
        if col_names != self.col_names:
            msg = (
                f"The order of columns in the input tensor is different from"
                f" the order of columns in the encoder:\n"
                f"  * Column names in TensorFrame: {col_names}\n"
                f"  * Column names in {self.__class__.__name__}: "
                f"{self.col_names}\n"
                f"The cause is either:\n"
                f"  1. The column order has changed in the input TensorFrame "
                f"between calls, or\n"
                f"  2. The column order used to initialize the "
                f"{self.__class__.__name__} is wrong.")
            raise ValueError(msg)

    def _create_exclude_cols_mask(
        self,
        device: torch.device | None = None,
    ) -> Tensor:
        exclude_cols_mask = torch.zeros(
            self.num_cols,
            dtype=torch.bool,
            device=device,
        )
        for idx, col_name in enumerate(self.col_names):
            if col_name in self.exclude_cols:
                exclude_cols_mask[idx] = True
        return exclude_cols_mask

    def forward(
        self,
        feat: InputT,
        col_names: list[str] | None = None,
    ) -> Tensor:
        out = self.forward_learnable(self.forward_fixed(feat, col_names))

        if modeling_context.is_debugging() and out.isnan().sum() > 0:
            raise ValueError(f"'{self.__class__.__name__}' produced NaN "
                             f"values for one or multiple of the following "
                             f"column names: {self.col_names}")

        return out

    @abstractmethod
    def forward_fixed(
        self,
        feat: InputT,
        col_names: list[str] | None = None,
    ) -> OutputT:
        """Forward pass of fixed part of the encoder that doesn't change across
        training iterations.

        This is useful for caching the intermediate results of the encoder
        in models that use RHS features.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_learnable(self, feat: OutputT) -> Tensor:
        """Forward pass of the learnable part of the encoder that changes
        across training iterations.
        """
        raise NotImplementedError
