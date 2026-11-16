# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
from kumoapi.encoder import EmbeddingScaler, Encoder, NumericalList
from kumoapi.typing import ColStatType
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor
from torch_geometric.utils.sparse import ptr2index

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.base import FusedEncoder


class FusedSequenceEncoder(FusedEncoder[
        MultiEmbeddingTensor,
        MultiEmbeddingTensor,
]):
    r"""A fused encoder that takes a dictionary of sequence encoders and
    parallelizes them.

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
    SUPPORTED_ENCODERS: set[type[Encoder]] = {NumericalList}

    def __init__(
        self,
        col_encoder_dict: dict[str, Encoder],
        exclude_cols: list[str],
        out_channels: int,
        enc_mode: Literal['fuse', 'column'],
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            col_encoder_dict,
            exclude_cols,
            out_channels,
            enc_mode,
            device,
        )
        self.col_dims = []
        self.scalers: list[EmbeddingScaler | None] = []
        for encoder in col_encoder_dict.values():
            assert isinstance(encoder, NumericalList)
            self.col_dims.append(
                encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH])
            self.scalers.append(encoder.scaler)
        self.total_col_dims = sum(self.col_dims)
        offsets, scales = self._create_offsets_and_scales(
            col_encoder_dict, device)
        self.offsets: Tensor
        self.register_buffer('offsets', offsets)
        self.scales: Tensor
        self.register_buffer('scales', scales)
        self.impute_values: Tensor
        self.register_buffer(
            'impute_values',
            self._create_impute_values(col_encoder_dict, device),
        )
        self.exclude_cols_mask: Tensor = (
            self.exclude_cols_mask.repeat_interleave(
                torch.tensor(self.col_dims, device=device)))
        if self.enc_mode == "fuse":
            self.lin = torch.nn.Linear(self.total_col_dims, out_channels,
                                       device=device)
        else:  # enc_mode == "column"
            self.lins = torch.nn.ModuleList([
                torch.nn.Linear(dim, out_channels, device=device)
                for dim in self.col_dims
            ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.enc_mode == "fuse":
            self.lin.reset_parameters()
        else:  # enc_mode == "column"
            for lin in self.lins:
                lin.reset_parameters()

    def _create_offsets_and_scales(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Creates offsets and scales for normalization.

        For each ``EmbeddingScaler`` type, the offsets and scales are
        computed as follows:

        * None: offset = 0.0, scale = 1.0
        * Standard: offset = mean, scale = std
        * Cosine: offset = 0.0, scale = 1.0. We later replace the scale with
            the embedding norm.
        """
        offsets = torch.zeros(self.total_col_dims, device=device)
        scales = torch.ones(self.total_col_dims, device=device)
        start = 0
        for col_dim, config in zip(self.col_dims, col_encoder_dict.values()):
            end = start + col_dim
            if config.scaler == EmbeddingScaler.STANDARD:
                offsets[start:end] = config._stats[ColStatType.SEQUENCE_MEAN]
                scales[start:end] = config._stats[ColStatType.SEQUENCE_STD]
            start = end
        return offsets, scales

    def _create_impute_values(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.zeros(self.total_col_dims, device=device)

    def _normalize(self, values: Tensor) -> Tensor:
        """Normalizes values according to the specified scalers per column."""
        scales = self.scales
        start = 0
        for col_dim, scaler in zip(self.col_dims, self.scalers):
            end = start + col_dim
            if scaler == EmbeddingScaler.COSINE:
                # Cosine normalization requires individual scales per row.
                # If we haven't already, we need to expand the tensor into
                # shape `[num_rows, num_total_cols]`:
                if scales.dim() == 1:
                    scales = scales.unsqueeze(0).repeat(values.size(0), 1)
                scale = values[:, start:end].norm(p=2, dim=-1, keepdim=True)
                scale = torch.where(scale < 1e-5, 1.0, scale)
                scales[:, start:end] = scale
            start = end
        return (values - self.offsets) / scales

    def forward_fixed(
        self,
        feat: MultiEmbeddingTensor,
        col_names: list[str] | None = None,
    ) -> MultiEmbeddingTensor:
        self.validate_col_order(col_names)
        values = feat.values
        na_mask = values.isnan() | values.isinf()
        values = torch.where(
            na_mask.view(feat.num_rows, self.total_col_dims),
            self.impute_values,
            values.view(feat.num_rows, self.total_col_dims),
        )
        if modeling_context.is_explaining():
            self.save_for_backward(values, feat.offset)
        values = self._normalize(values)
        if len(self.exclude_cols) > 0:
            values = torch.where(self.exclude_cols_mask, 0.0, values)
        feat.values = values
        return feat

    def forward_learnable(self, feat: MultiEmbeddingTensor) -> Tensor:
        if self.enc_mode == "fuse":
            return self.lin(feat.values)
        else:  # enc_mode == "column"
            outs: list[Tensor] = []
            start = 0
            for col_idx, lin in enumerate(self.lins):
                end = start + self.col_dims[col_idx]
                outs.append(lin(feat.values[:, start:end]))
                start = end
            return torch.stack(outs, dim=1)

    def _get_grad_wrt_input(self) -> Tensor:
        values, offset = self.saved_tensors

        assert values is not None and values.grad is not None
        assert offset is not None
        in_grad = values.new_zeros(values.size(0), offset.numel() - 1)
        batch = ptr2index(offset).view(1, -1).expand_as(values)
        in_grad.scatter_reduce_(1, batch, values.grad.abs(), reduce='sum')
        return in_grad
