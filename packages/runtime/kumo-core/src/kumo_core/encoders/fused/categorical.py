# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
from kumoapi.encoder import Encoder, Hash, Index, NAStrategy
from kumoapi.typing import ColStatType
from torch import Tensor
from torch.nn import Embedding, EmbeddingBag

import kumo_core.modeling.context as modeling_context
import logging
from kumo_core.encoders.fused.base import FusedEncoder

logger = logging.getLogger(__name__)


def _get_num_min_occ_categories(encoder: Index) -> int:
    r"""The number of categories that meet the min_occ constraints."""
    category_counts = encoder._stats[ColStatType.CATEGORY_COUNTS]
    return sum(count >= encoder.min_occ for _, count in category_counts)


def _get_na_threshold_and_num_embeddings(
        encoder: Index | Hash) -> tuple[int, int]:
    r"""The number of categories that meet the min_occ constraints."""
    if isinstance(encoder, Index):
        # Treat 'values>=num_min_occ_categories' as N/A.
        na_threshold = num_embeddings = _get_num_min_occ_categories(encoder)
    else:
        assert isinstance(encoder, Hash)
        # N/A threshold is used to distinguish between these two cases:
        #   (1) [0, len(category_counts)-1]: known categories
        #   (2) [len(category_counts), +inf): N/A
        # modulo op squashes both cases into the range [0, num_components-1],
        # but values in the case (2) will then be masked as N/A.
        na_threshold = len(encoder._stats[ColStatType.CATEGORY_COUNTS])
        num_embeddings = encoder.num_components

    if encoder.na_strategy == NAStrategy.SEPARATE:
        num_embeddings = num_embeddings + 1

    return na_threshold, num_embeddings


class FusedCategoricalEncoder(FusedEncoder[Tensor, Tensor]):
    r"""A fused encoder that takes a dictionary of categorical encoders and
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

    .. warning::
        :class:`FusedCategoricalEncoder` has an assumption that the value ``0``
        in the input tensor is always the category with the most count.
    """
    SUPPORTED_ENCODERS: set[type[Encoder]] = {Index, Hash}

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
        self.should_run_hash = any(
            isinstance(cfg, Hash) for cfg in col_encoder_dict.values())

        num_embeddings_list = []
        na_thresholds = []
        for cfg in col_encoder_dict.values():
            (na_threshold,
             num_embeddings) = _get_na_threshold_and_num_embeddings(cfg)
            na_thresholds.append(na_threshold)
            num_embeddings_list.append(num_embeddings)

        self.register_buffer(
            "na_thresholds",
            torch.tensor(na_thresholds, dtype=torch.long, device=device),
        )

        # + 1 to create a new category to map category indices to zero vectors:
        # * if columns are in exclude_cols, or
        # * if columns have zero na strategy
        self.total_num_embeddings = sum(num_embeddings_list) + 1
        self.zero_vector_index = self.total_num_embeddings - 1

        self.cumulative_offset: Tensor
        self.register_buffer(
            'cumulative_offset',
            torch.cumsum(
                torch.tensor(
                    [0] + num_embeddings_list[:-1],
                    dtype=torch.long,
                    device=device,
                ),
                dim=0,
            ),
        )
        self.register_buffer(
            'modulo_values',
            self._get_hash_modulo_values(col_encoder_dict, device),
        )
        self.impute_values: Tensor
        self.register_buffer(
            'impute_values',
            self._get_impute_values(col_encoder_dict, device),
        )

        self.emb: Embedding | EmbeddingBag
        if self.enc_mode == 'column':
            self.emb = Embedding(
                num_embeddings=self.total_num_embeddings,
                embedding_dim=self.out_channels,
                padding_idx=self.zero_vector_index,
                device=device,
            )
        else:
            assert self.enc_mode == 'fuse'
            self.emb = EmbeddingBag(
                num_embeddings=self.total_num_embeddings,
                embedding_dim=self.out_channels,
                mode='sum',
                padding_idx=self.zero_vector_index,
                device=device,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.emb.reset_parameters()

    def _get_impute_values(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        impute_values = torch.zeros(self.num_cols, dtype=torch.long,
                                    device=device)
        for idx, cfg in enumerate(col_encoder_dict.values()):
            if cfg.na_strategy == NAStrategy.ZERO:
                # NOTE Since cumulative offset will be added to this impute
                # value, we want this impute value to satisfy:
                #     zero_vector_index = impute_value + cumulative_offset
                impute_values[
                    idx] = self.zero_vector_index - self.cumulative_offset[idx]
            elif cfg.na_strategy == NAStrategy.MOST_FREQUENT:
                # Categories are always sorted with count in descending order,
                # so the category 0 is always the category with the most count.
                impute_values[idx] = 0
            else:
                assert cfg.na_strategy == NAStrategy.SEPARATE
                if isinstance(cfg, Hash):
                    # 0, ..., num_components-1 are non-N/A indices, and
                    # num_components is the index for the separate category.
                    impute_values[idx] = cfg.num_components
                else:
                    impute_values[idx] = _get_num_min_occ_categories(cfg)

        return impute_values

    def _get_hash_modulo_values(
        self,
        col_encoder_dict: dict[str, Encoder],
        device: torch.device | None = None,
    ) -> Tensor:
        modulo_values = torch.zeros(self.num_cols, dtype=torch.long,
                                    device=device)
        for idx, encoder in enumerate(col_encoder_dict.values()):
            assert isinstance(encoder, tuple(self.SUPPORTED_ENCODERS))
            if isinstance(encoder, Hash):
                modulo_values[idx] = encoder.num_components
            else:
                assert isinstance(encoder, Index)
                modulo_values[idx] = max(_get_num_min_occ_categories(encoder),
                                         1)
        return modulo_values

    def forward_fixed(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        self.validate_col_order(col_names)
        na_mask = (feat == -1) | (feat >= self.na_thresholds)
        if self.should_run_hash:
            feat = feat % self.modulo_values
        feat = torch.where(na_mask, self.impute_values, feat)
        feat = feat + self.cumulative_offset
        if self.exclude_cols:
            feat = torch.where(
                self.exclude_cols_mask,
                self.zero_vector_index,
                feat,
            )
        return feat

    def forward_learnable(self, feat: Tensor) -> Tensor:
        out = self.emb(feat)
        if modeling_context.is_explaining():
            self.save_for_backward(feat, out)
        return out

    def _get_grad_wrt_input(self) -> Tensor:
        index, out = self.saved_tensors
        assert index is not None
        assert out is not None and out.grad is not None

        sparse_emb = self.emb.weight[index.view(-1)].view(
            *index.size(), self.emb.embedding_dim)

        if self.enc_mode == 'fuse':
            return torch.einsum('ik,ijk->ij', out.grad, sparse_emb).abs()
        else:
            return torch.einsum('ijk,ijk->ij', out.grad, sparse_emb).abs()
