# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Literal

import torch
from kumoapi.encoder import (
    Encoder,
    MaxLogNumerical,
    MinLogNumerical,
    NAStrategy,
    Numerical,
)
from kumoapi.typing import ColStatType
from torch import Tensor
from torch.nn import Linear, Parameter, init

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.base import FusedEncoder
from kumo_core.encoders.scaling import get_offset_and_scale


class FusedNumericalEncoder(FusedEncoder[Tensor, Tensor]):
    r"""A fused encoder that takes a dictionary of numerical encoders and
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
    SUPPORTED_ENCODERS: set[type[Encoder]] = {
        Numerical,
        MaxLogNumerical,
        MinLogNumerical,
    }

    def __init__(
        self,
        col_encoder_dict: dict[str, Encoder],
        exclude_cols: list[str],
        out_channels: int,
        enc_mode: Literal["fuse", "column"],
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            col_encoder_dict,
            exclude_cols,
            out_channels,
            enc_mode,
            device,
        )

        offsets, scales = self._create_offsets_and_scales(
            col_encoder_dict, device)
        self.register_buffer('offsets', offsets)
        self.register_buffer('scales', scales)

        self.impute_values: Tensor
        self.register_buffer(
            'impute_values',
            self._create_impute_values(col_encoder_dict, device),
        )

        self.log_mask: Tensor
        self.register_buffer(
            'log_mask',
            self._create_log_mask(col_encoder_dict, device),
        )
        self.has_log_encoder = any(self.log_mask)

        self.separate_na_mask: Tensor
        self.register_buffer(
            'separate_na_mask',
            self._create_separate_na_mask(col_encoder_dict, device),
        )
        self.has_separate_na_encoder = any(self.separate_na_mask)

        if enc_mode == "fuse":
            self.lin = Linear(self.num_cols, out_channels, device=device)
            self.na_lin = Linear(self.num_cols, out_channels, device=device)
        else:
            assert enc_mode == 'column'
            self.weight = Parameter(
                torch.empty(self.num_cols, self.out_channels, device=device))
            self.na_weight = Parameter(
                torch.empty(self.num_cols, self.out_channels, device=device))
            self.bias = Parameter(
                torch.empty(self.num_cols, self.out_channels, device=device))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.enc_mode == "fuse":
            self.lin.reset_parameters()
            self.na_lin.reset_parameters()
        else:  # enc_mode == "column":
            # NOTE: This is the same as how nn.Linear inits its params.
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            init.kaiming_uniform_(self.na_weight, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _create_offsets_and_scales(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Creates offsets and scales for numerical encoders.

        For each encoder type, the offsets and scales are computed as follows:

        * ``Numerical``:
            1. If ``scaler`` is ``None``: ``(feat - 0.0) / 1.0``
            2. If ``scaler`` is ``MINMAX``: ``(feat - min) / (max - min)``
            3. If ``scaler`` is ``ROBUST``: ``(feat - q50) / (q75 - q25)``
            4. If ``scaler`` is ``STANDARD``: ``(feat - mean) / std``
        * ``MaxLogNumerical``: ``(feat - (min - 1)) / 1.0``
        * ``MinLogNumerical``: ``(feat - (max + 1)) / -1.0``
        """
        offsets = torch.zeros(self.num_cols, device=device)
        scales = torch.ones(self.num_cols, device=device)
        for idx, encoder in enumerate(col_encoder_dict.values()):
            offsets[idx], scales[idx] = get_offset_and_scale(encoder)
        return offsets, scales

    def _create_impute_values(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        impute_values = torch.zeros(self.num_cols, device=device)
        for idx, (col_name, encoder) in enumerate(col_encoder_dict.items()):
            if encoder.na_strategy == NAStrategy.MEAN:
                # TODO Intermediate fix to unblock all N/A rows.
                mean = encoder._stats[ColStatType.MEAN] or 0
                mean = min(mean, torch.finfo(torch.float).max)
                mean = max(mean, torch.finfo(torch.float).min)
                impute_values[idx] = mean
            elif encoder.na_strategy == NAStrategy.ZERO:
                impute_values[idx] = 0.
            elif encoder.na_strategy == NAStrategy.SEPARATE:
                impute_values[idx] = float('NaN')
            else:
                raise ValueError(
                    f"Unsupported N/A strategy {encoder.na_strategy} in "
                    f"column '{col_name}'. Only 'mean', 'zero' and 'separate' "
                    f"are supported.")
        return impute_values

    def _create_log_mask(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        mask = torch.zeros(self.num_cols, dtype=torch.bool, device=device)
        for idx, encoder in enumerate(col_encoder_dict.values()):
            if isinstance(encoder, MaxLogNumerical | MinLogNumerical):
                mask[idx] = True
        return mask

    def _create_separate_na_mask(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        mask = torch.zeros(self.num_cols, dtype=torch.bool, device=device)
        for idx, encoder in enumerate(col_encoder_dict.values()):
            if encoder.na_strategy == NAStrategy.SEPARATE:
                mask[idx] = True
        return mask & ~self.exclude_cols_mask

    def forward_fixed(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        self.validate_col_order(col_names)
        # 1. Impute N/A values
        na_mask = feat.isnan() | feat.isinf()
        feat = torch.where(na_mask, self.impute_values, feat)
        # 2. Apply offsets and scales
        feat = (feat - self.offsets) / self.scales
        # 3. Apply log transform
        if self.has_log_encoder:
            feat = torch.where(self.log_mask, feat.clamp(min=1).log(), feat)
        # 4. Clamp values for safety reasons: Since encoders are inferred
        #    based on approximate stats, we may not always detect highly
        #    skewed distributions (and as such, no log-scaling is applied).
        #    In such (rare) cases, standard or robust normalization preserves
        #    very large values, leading to either exploding gradients or NaNs.
        feat = feat.clamp(min=-1000, max=1000)

        return feat

    def forward_learnable(self, feat: Tensor) -> Tensor:
        if self.has_separate_na_encoder:
            na_mask = feat.isnan().to(feat.dtype)
            feat = feat.nan_to_num()

        if modeling_context.is_explaining():
            self.save_for_backward(feat)

        if self.exclude_cols:
            feat = torch.where(self.exclude_cols_mask, 0.0, feat)

        if self.enc_mode == "fuse":
            # [nrows, ncols] -> [nrows, outch]
            out = self.lin(feat)
            if self.has_separate_na_encoder:
                out = out + self.na_lin(na_mask)
            return out
        else:  # enc_mode == "column"
            # [nrows, ncols], [ncols, outch] -> [nrows, ncols, outch]
            out = torch.einsum("ij,jk->ijk", feat, self.weight)
            # [nrows, ncols, outch] + [ncols, outch] -> [nrows, ncols, outch]
            if self.has_separate_na_encoder:
                out = out + torch.einsum("ij,jk->ijk", na_mask, self.na_weight)
            return out + self.bias

    def _get_grad_wrt_input(self) -> Tensor:
        feat = self.saved_tensors[0]

        assert feat is not None and feat.grad is not None
        return feat.grad.abs()
