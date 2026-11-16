# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
from kumoapi.encoder import Encoder, MultiCategorical, NAStrategy
from kumoapi.typing import ColStatType
from torch import Tensor
from torch.nn import EmbeddingBag
from torch_frame.data import MultiNestedTensor
from torch_geometric.utils import cumsum, scatter
from torch_geometric.utils.sparse import ptr2index

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.base import FusedEncoder


def _get_num_min_occ_categories(encoder: MultiCategorical) -> int:
    r"""The number of categories that meet the min_occ constraints."""
    category_counts = encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS]
    return sum(count >= encoder.min_occ for _, count in category_counts)


def _get_na_threshold_and_num_embeddings(
        encoder: MultiCategorical) -> tuple[int, int]:
    r"""The number of categories that meet the min_occ constraints."""
    # Treat 'values>=num_min_occ_categories' as N/A.
    na_threshold = num_embeddings = _get_num_min_occ_categories(encoder)

    if encoder.na_strategy == NAStrategy.SEPARATE:
        num_embeddings = num_embeddings + 1

    return na_threshold, num_embeddings


class FusedMultiCategoricalEncoder(FusedEncoder[
        MultiNestedTensor,
        MultiNestedTensor,
]):
    r"""A fused encoder that takes a dictionary of multicategorical encoders
    and parallelizes them.

    Args:
        col_encoder_dict: A mapping from column names to encoders to fuse.
            Encoders need to be operate on the same stype.
        exclude_cols: A list of column names to be excluded.
        out_channels: The number of output channels.
        enc_mode: If ``"fuse"``, the output size is
            ``[num_rows, out_channels]``. If ``"column"``, the output size is
            ``[num_rows, num_cols, out_channels]``.
        device: The device to use.

    .. warning::
        :class:`FusedCategoricalEncoder` has an assumption that the value ``0``
        in the input tensor is always the category with the most count.
    """
    SUPPORTED_ENCODERS: set[type[Encoder]] = {MultiCategorical}

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
        num_embeddings_list = []
        na_thresholds = []
        for cfg in col_encoder_dict.values():
            assert isinstance(cfg, MultiCategorical)
            (na_threshold,
             num_embeddings) = _get_na_threshold_and_num_embeddings(cfg)
            na_thresholds.append(na_threshold)
            num_embeddings_list.append(num_embeddings)

        self.cumulative_offset: Tensor
        self.register_buffer(
            'cumulative_offset',
            cumsum(
                torch.tensor(
                    num_embeddings_list[:-1],
                    dtype=torch.long,
                    device=device,
                )),
        )
        self.register_buffer(
            'na_thresholds',
            torch.tensor(na_thresholds, dtype=torch.long, device=device),
        )
        # `+ 1` for the additional index `padding_idx` to which we map N/A
        # values and excluded column values. Note that these zero vectors
        # are excluded from the reduction as documented in PyTorch docs.
        self.total_num_embeddings = sum(num_embeddings_list) + 1
        self.zero_vector_index = self.total_num_embeddings - 1

        self.register_buffer(
            'impute_values',
            self._create_impute_values(col_encoder_dict, device),
        )
        self.emb = EmbeddingBag(
            num_embeddings=self.total_num_embeddings,
            embedding_dim=out_channels,
            mode="sum",
            padding_idx=self.zero_vector_index,
            include_last_offset=True,
            device=device,
        )

    def reset_parameters(self) -> None:
        self.emb.reset_parameters()

    def _create_impute_values(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        impute_values: list[int] = []
        for idx, (col_name, encoder) in enumerate(col_encoder_dict.items()):
            assert isinstance(encoder, MultiCategorical)
            if col_name in self.exclude_cols:
                # exclude_cols should always be prioritised over na_strategy,
                # and values of exclude_cols will be mapped to zero vectors,
                # which is the same behaviour as NA_STRATEGY.ZERO.
                impute_values.append(self.zero_vector_index -
                                     self.cumulative_offset[idx])
            elif encoder.na_strategy == NAStrategy.ZERO:
                # NOTE Since cumulative offset will be added to this impute
                # value, we want this impute value to satisfy:
                #     zero_vector_index = impute_value + cumulative_offset
                impute_values.append(self.zero_vector_index -
                                     self.cumulative_offset[idx])
            elif encoder.na_strategy == NAStrategy.MOST_FREQUENT:
                impute_values.append(0)
            elif encoder.na_strategy == NAStrategy.SEPARATE:
                impute_values.append(_get_num_min_occ_categories(encoder))
            else:
                raise ValueError(
                    f"Unsupported N/A Strategy: {encoder.na_strategy}."
                    " Only 'zero', 'most_frequent' and 'separate' are"
                    "supported.")
        return torch.tensor(impute_values, device=device)

    def forward_fixed(
        self,
        feat: MultiNestedTensor,
        col_names: list[str] | None = None,
    ) -> MultiNestedTensor:
        """Encodes a multi nested tensor into a tensor by applying the
        following operations.

        1. Map category indices that (A) have no values or (B) have too few
            occurences in each category to the following values according to
            the N/A strategy of each column:
            1.1. the most frequent category index if MOST_FREQUENT,
            1.2. the last index of EmbeddingBag that points to a zero vector
                if ZERO, and
            1.3. a new category if SEPARATE.
        2. Map local category indices to global category indices.
        3. Fetch embeddings from EmbeddingBag with the global category indices.
        """
        self.validate_col_order(col_names)
        counts = feat.offset.diff()
        # 1. N/A and exclude_cols masking
        na_thresholds = self.na_thresholds.repeat(feat.num_rows)
        na_thresholds = na_thresholds.repeat_interleave(
            counts,
            output_size=feat.values.size(0),
        )
        na_mask = (feat.values == -1) | (na_thresholds <= feat.values)
        if self.exclude_cols:
            exclude_cols_mask = self.exclude_cols_mask.repeat(feat.num_rows)
            exclude_cols_mask = exclude_cols_mask.repeat_interleave(
                counts,
                output_size=feat.values.size(0),
            )
            na_mask = na_mask | exclude_cols_mask
        impute_values = self.impute_values.repeat(feat.num_rows)
        impute_values = impute_values.repeat_interleave(
            counts,
            output_size=feat.values.size(0),
        )
        values = torch.where(
            na_mask,
            impute_values,
            feat.values,
        )
        # 2. map local category indices to global category indices
        cumulative_offset = self.cumulative_offset.repeat(feat.num_rows)
        cumulative_offset = cumulative_offset.repeat_interleave(
            counts,
            output_size=feat.values.size(0),
        )
        values = values + cumulative_offset
        # `padding_idx+cumulative_offset` can be larger than
        # `total_num_embeddings-1`, so we clamp the values.
        values = values.clamp(max=self.total_num_embeddings - 1)

        return MultiNestedTensor(
            num_rows=feat.num_rows,
            num_cols=feat.num_cols,
            values=values,
            offset=feat.offset,
        )

    def forward_learnable(self, feat: MultiNestedTensor) -> Tensor:
        # 3. Fetch embedings from EmbeddingBag
        offset = feat.offset
        if self.enc_mode == 'fuse':
            # When enc_mode is "fuse", we slice the original `offset` of size
            # [num_rows*num_cols+1] to [num_rows+1] so that embedding bag will
            # reduce the embeddings of each row across columns.
            offset = offset[::feat.num_cols]
        out = self.emb(feat.values, offset)

        if self.enc_mode == "column":
            out = out.view(feat.num_rows, feat.num_cols, self.out_channels)

        if modeling_context.is_explaining():
            self.save_for_backward(feat.values, feat.offset, out)

        return out

    def _get_grad_wrt_input(self) -> Tensor:
        index, offset, out = self.saved_tensors
        assert index is not None
        assert offset is not None
        assert out is not None and out.grad is not None

        batch = ptr2index(offset)
        batch_size = offset.numel() - 1

        num_cols = self.cumulative_offset.numel()
        if self.enc_mode == 'fuse':
            out_grad = out.grad[ptr2index(offset[::num_cols])]
        else:
            out_grad = out.grad.view(-1, out.size(-1))[batch]

        sparse_emb = self.emb.weight[index]

        in_grad = scatter(sparse_emb * out_grad, batch, dim_size=batch_size,
                          reduce='sum')
        in_grad = in_grad.sum(dim=-1).abs().view(-1, num_cols)
        return in_grad
