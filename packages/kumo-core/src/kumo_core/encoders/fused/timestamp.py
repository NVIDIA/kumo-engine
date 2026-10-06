# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Literal

import pandas as pd
import torch
from kumoapi.encoder import Datetime, Encoder
from kumoapi.typing import ColStatType
from torch import Tensor

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.encoders.fused.base import FusedEncoder
from kumo_core.types.features import (
    DAYS_PER_WEEK,
    HOURS_PER_DAY,
    MINUTES_PER_HOUR,
    TIMESTAMP_PREFIX,
)


class FusedTimestampEncoder(FusedEncoder[Tensor, Tensor]):
    r"""A fused encoder that takes a dictionary of timestamp encoders and
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
    SUPPORTED_ENCODERS: set[type[Encoder]] = {Datetime}
    # NOTE: The first `index_boundary` features, `feat[:, :, :index_boundary]`,
    # are the meaningful features while the rest of the features,
    # `feat[:, :, index_boundary:]`, are just denominators to normalise
    # features into the range of [0, 1].
    INPUT_INDEX_BOUNDARY = TimestampTensorMapper.index_boundary

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

        min_times: list[pd.Timestamp] = []
        max_times: list[pd.Timestamp] = []
        for encoder in col_encoder_dict.values():
            assert isinstance(encoder, Datetime)
            if encoder.include_year and encoder.num_year_periods is None:
                min_time = encoder._stats[ColStatType.MIN]
                max_time = encoder._stats[ColStatType.MAX]
                assert min_time.startswith(TIMESTAMP_PREFIX)
                assert max_time.startswith(TIMESTAMP_PREFIX)
                prefix_len = len(TIMESTAMP_PREFIX)
                min_times.append(pd.Timestamp(min_time[prefix_len:]))
                max_times.append(pd.Timestamp(max_time[prefix_len:]))
            else:
                min_times.append(pd.Timestamp.min)
                max_times.append(pd.Timestamp.max)

        self.max_num_year_periods = 1
        for i, encoder in enumerate(col_encoder_dict.values()):
            if not encoder.include_year:
                continue
            num_year_periods = encoder.num_year_periods
            if num_year_periods is None:
                max_offset = max_times[i].year - min_times[i].year
                num_year_periods = math.ceil(math.log(max_offset + 1, 2))
            self.max_num_year_periods = max(
                self.max_num_year_periods,
                num_year_periods,
            )

        # Note: INPUT_INDEX cannot be defined as a class property to avoid
        # dynamo from producing graph breaks:
        self.INPUT_INDEX = TimestampTensorMapper.index

        self.year_repeats: Tensor
        self.register_buffer(
            'year_repeats',
            torch.tensor([
                self.max_num_year_periods
                if idx == self.INPUT_INDEX["year"] else 1
                for idx in range(self.INPUT_INDEX_BOUNDARY)
            ], device=device))
        self.register_buffer(
            'year_bases',
            torch.arange(1., self.max_num_year_periods + 1, device=device)**2,
        )
        self.exclude_feat_mask: Tensor
        self.register_buffer(
            'exclude_feat_mask',
            self._create_exclude_feat_mask(col_encoder_dict, device),
        )
        self.lin = torch.nn.Linear(
            self.num_cols * self.num_expanded_feats * 2,
            out_channels,
            device=device,
        )
        self.reset_parameters()

    @property
    def num_expanded_feats(self) -> int:
        """The number of the total features per column after expansion. We
        expand a single value of 'year' feature into max_num_year_periods
        number of values. See :meth:`forward_year` for how it's expanded.
        """
        return self.INPUT_INDEX_BOUNDARY - 1 + self.max_num_year_periods

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def _create_exclude_feat_mask(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        """Creates a mask to exclude features from the encoder.

        For example, if one encoder has ``include_minute=False``, then the
        minute feature of the column will be filled with zero.

        Returns:
            A :class:`~torch.Tensor` of size
            ``[num_cols, num_expanded_feats]``.
        """
        exclude_feat_mask = torch.zeros(
            self.num_cols,
            self.INPUT_INDEX_BOUNDARY,
            dtype=torch.bool,
            device=device,
        )
        for idx, encoder in enumerate(col_encoder_dict.values()):
            assert isinstance(encoder, Datetime)
            exclude_feat_mask[
                idx, self.INPUT_INDEX['minute']] = not encoder.include_minute
            exclude_feat_mask[
                idx, self.INPUT_INDEX['hour']] = not encoder.include_hour
            exclude_feat_mask[
                idx, self.INPUT_INDEX['year']] = not encoder.include_year
            exclude_feat_mask[
                idx, self.
                INPUT_INDEX['day_of_week']] = not encoder.include_day_of_week
            exclude_feat_mask[
                idx, self.
                INPUT_INDEX['day_of_month']] = not encoder.include_day_of_month
            exclude_feat_mask[
                idx, self.
                INPUT_INDEX['day_of_year']] = not encoder.include_day_of_year
        # [num_cols, INPUT_INDEX_BOUNDARY] -> [num_cols, num_expanded_feats]
        return exclude_feat_mask.repeat_interleave(self.year_repeats, dim=1)

    def get_scales(self, feat: Tensor) -> Tensor:
        """Gets the scales for each feature.

        For example, if the input is ``feat[:, :, [..., year, ...]]``
        then the scales will be
        ``scales[:, :, [..., year, ...]] = [1., 4., 9., ...]``.

        Args:
            feat: The input :class:`~torch.Tensor` of size
                ``[num_rows, num_cols, in_channels]``.

        Returns:
            A :class:`~torch.Tensor` of size
            ``[num_rows, num_cols, INPUT_INDEX_BOUNDARY]``.
        """
        # [num_rows, num_cols, in_channels] ->
        # [num_rows, num_cols, INPUT_INDEX_BOUNDARY]
        num_rows, num_cols, _ = feat.size()
        scales = torch.ones(
            num_rows,
            num_cols,
            self.INPUT_INDEX_BOUNDARY,
            device=feat.device,
        )
        # === scales that are the same across all rows and columns ===
        scales[:, :, self.INPUT_INDEX['minute']] = MINUTES_PER_HOUR
        scales[:, :, self.INPUT_INDEX['hour']] = HOURS_PER_DAY
        # NOTE no-op for year feature as we handle in forward_year():
        scales[:, :, self.INPUT_INDEX['year']] = 1.
        scales[:, :, self.INPUT_INDEX['day_of_week']] = DAYS_PER_WEEK
        # === scales that are different across rows and columns ===
        scales[:, :, self.INPUT_INDEX[
            'day_of_month']] = feat[:, :, self.INPUT_INDEX['days_per_month']]
        scales[:, :, self.
               INPUT_INDEX['day_of_year']] = feat[:, :, self.
                                                  INPUT_INDEX['days_per_year']]
        return scales

    def forward_year(self, feat: Tensor) -> Tensor:
        """Encodes year feature into cyclic feature representation.

        A single 'year' feature is expanded into the max_num_year_periods
        number of features and encoded into cyclic features with different
        frequencies.

        Args:
            feat: The input :class:`~torch.Tensor` of size
                ``[num_rows, num_cols, INPUT_INDEX_BOUNDARY]``.

        Returns:
            A :class:`~torch.Tensor` of size
            ``[num_rows, num_cols, num_expanded_feats]``.
        """
        # feat_year: [num_rows, num_cols]
        feat_year = feat[:, :, self.INPUT_INDEX['year']].unsqueeze(-1) - 1970
        # feat_year_expanded: [num_rows, num_cols, max_num_year_periods]
        feat_year_expanded = feat_year.expand(-1, -1,
                                              self.max_num_year_periods)
        feat_year_expanded = feat_year_expanded % self.year_bases
        feat_year_expanded = feat_year_expanded / self.year_bases
        return torch.cat([
            feat[:, :, :self.INPUT_INDEX['year']],
            feat_year_expanded,
            feat[:, :, self.INPUT_INDEX['year'] + 1:],
        ], dim=-1)

    def forward_fixed(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [num_rows, num_cols, in_channels]
        self.validate_col_order(col_names)
        # NOTE Only checking at 0th feature is sufficient as TensorMapper fills
        # NaN with -1 across all features of a NaN row of a column.
        na_mask = feat[:, :, 0] == -1
        # feat_sub: [num_rows, num_cols, INPUT_INDEX_BOUNDARY]
        feat_sub = feat[:, :, :self.INPUT_INDEX_BOUNDARY]
        feat_sub = feat_sub / self.get_scales(feat)
        # feat_sub_year_expanded: [num_rows, num_cols, num_expanded_feats]
        feat_sub_year_expanded = self.forward_year(feat_sub)
        if modeling_context.is_explaining():
            self.save_for_backward(feat_sub_year_expanded)
        feat_sub_year_expanded = 2 * math.pi * feat_sub_year_expanded
        # feat_cyclic: [num_rows, num_cols, num_expanded_feats, 2]
        feat_cyclic = torch.stack(
            [feat_sub_year_expanded.sin(),
             feat_sub_year_expanded.cos()], dim=-1)
        if len(self.exclude_cols) > 0:
            feat_cyclic = torch.where(
                self.exclude_cols_mask.view(self.num_cols, 1, 1),
                0.0,
                feat_cyclic,
            )
        na_mask = na_mask.view(-1, self.num_cols, 1, 1)
        exclude_feat_mask = self.exclude_feat_mask.view(
            1, self.num_cols, self.num_expanded_feats, 1)
        return torch.where(
            na_mask | exclude_feat_mask,
            0.0,
            feat_cyclic,
        )

    def forward_learnable(self, feat: Tensor) -> Tensor:
        num_rows, num_cols, num_feats, num_cycles = feat.size()
        if self.enc_mode == "fuse":
            feat = feat.view(num_rows, num_cols * num_feats * num_cycles)
            return self.lin(feat)
        else:  # enc_mode == "column"
            feat = torch.einsum(
                'ijk,jkl->ijl',
                feat.view(num_rows, num_cols, num_feats * num_cycles),
                self.lin.weight.view(
                    self.num_cols,
                    self.num_expanded_feats * 2,
                    self.out_channels,
                ),
            )
            return feat + self.lin.bias

    def _get_grad_wrt_input(self) -> Tensor:
        feat, = self.saved_tensors

        assert feat is not None and feat.grad is not None
        return feat.grad.abs().sum(dim=-1)
