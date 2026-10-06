# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
import torch.nn.functional as F
from kumoapi.encoder import Encoder, TextEmbedding
from torch import Tensor
from torch.nn import Linear
from torch_frame.data import MultiNestedTensor

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.fused.base import FusedEncoder
from kumo_core.embedding_models import TokenEmbeddingModel, resolve_encoder
from kumo_core.glove import TokenEmbeddingTable


class FusedTextEncoder(FusedEncoder[MultiNestedTensor, Tensor]):
    r"""A fused encoder that takes a dictionary of text encoders and
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
    SUPPORTED_ENCODERS: set[type[Encoder]] = {TextEmbedding}

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
        # TODO: Add support for exclude_cols if necessary in the future.
        if self.exclude_cols_mask.any():
            raise ValueError(f"Currently, 'exclude_cols' is not supported for "
                             f"'{self.__class__.__name__}'")
        # One table per distinct model; each column is looked up in its own.
        models: dict[tuple[str, str], TokenEmbeddingModel] = {}
        col_group: list[int] = []
        self.col_dims: list[int] = []
        for encoder in col_encoder_dict.values():
            model = resolve_encoder(encoder)
            key = (model.family, model.fingerprint())
            models.setdefault(key, model)
            col_group.append(list(models).index(key))
            self.col_dims.append(model.dim)
        self.tables = torch.nn.ModuleList([
            TokenEmbeddingTable.instance(model).initialize(device)
            for model in models.values()
        ])
        self.groups = [[idx for idx, g in enumerate(col_group) if g == group]
                       for group in range(len(models))]
        dims = [model.dim for model in models.values()]
        self.embedding_dim = sum(dims) if enc_mode == 'fuse' else max(dims)
        self.col_group: Tensor
        self.register_buffer('col_group', torch.tensor(col_group,
                                                       device=device),
                             persistent=False)

        self.lin: Linear | torch.nn.ModuleList
        if enc_mode == 'fuse':
            self.lin = Linear(self.embedding_dim, out_channels, device=device)
        else:
            assert enc_mode == 'column'
            self.lin = torch.nn.ModuleList([
                Linear(self.col_dims[idx], out_channels, device=device)
                for idx, col in enumerate(col_encoder_dict.keys())
                if col not in self.exclude_cols
            ])

        # Note: Needs to have a non-tensor version of the mask to avoid
        # a graph break due to dynamo seeing it as dynamic control flow.
        self.exclude_cols_mask_list = self.exclude_cols_mask.tolist()

    def reset_parameters(self) -> None:
        if self.enc_mode == "fuse":
            assert isinstance(self.lin, Linear)
            self.lin.reset_parameters()
        else:
            assert isinstance(self.lin, torch.nn.ModuleList)
            for layer in self.lin:
                layer.reset_parameters()

    def forward_fixed(
        self,
        feat: MultiNestedTensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        self.validate_col_order(col_names)
        if len(self.groups) == 1:
            offset = feat.offset
            if self.enc_mode == 'fuse':
                # Slice the [num_rows*num_cols+1] offset to [num_rows+1] so
                # that each row averages its tokens across columns.
                offset = offset[::self.num_cols]
            out = self._embed(self.tables[0], feat.values, offset)
            if self.enc_mode == 'column':
                out = out.view(feat.num_rows, feat.num_cols,
                               self.embedding_dim)
        else:
            out = self._embed_groups(feat)
        if modeling_context.is_explaining():
            self.save_for_backward(out)
        return out

    def _embed_groups(self, feat: MultiNestedTensor) -> Tensor:
        counts = feat.offset.diff().view(feat.num_rows, feat.num_cols)
        value_group = self.col_group.repeat(feat.num_rows).repeat_interleave(
            counts.flatten())
        outs = []
        for group, (table, cols) in enumerate(zip(self.tables, self.groups)):
            group_counts = counts[:, cols]
            if self.enc_mode == 'fuse':
                group_counts = group_counts.sum(dim=1)
            offset = F.pad(group_counts.flatten().cumsum(0), (1, 0))
            outs.append(
                self._embed(table, feat.values[value_group == group], offset))
        if self.enc_mode == 'fuse':
            return torch.cat(outs, dim=1)
        out = outs[0].new_zeros(feat.num_rows, feat.num_cols,
                                self.embedding_dim)
        for group_out, cols in zip(outs, self.groups):
            out[:, cols, :group_out.size(1)] = group_out.view(
                feat.num_rows, len(cols), -1)
        return out

    @staticmethod
    def _embed(table: TokenEmbeddingTable, values: Tensor,
               offset: Tensor) -> Tensor:
        return F.embedding_bag(values, table.emb, offset, mode='mean',
                               include_last_offset=True)

    def forward_learnable(self, feat: Tensor) -> Tensor:
        num_rows = feat.size(0)
        if self.enc_mode == 'fuse':
            return self.lin(feat)
        else:
            assert isinstance(self.lin, torch.nn.ModuleList)
            outs = []
            # TODO: Remove the for-loop
            for idx in range(self.num_cols):
                if self.exclude_cols_mask_list[idx]:
                    outs.append(
                        torch.zeros(num_rows, self.out_channels,
                                    device=feat.device))
                else:
                    outs.append(
                        self.lin[idx](feat[:, idx, :self.col_dims[idx]]))
            return torch.stack(outs, dim=1)

    def _get_grad_wrt_input(self) -> Tensor:
        feat, = self.saved_tensors
        assert feat is not None and feat.grad is not None
        if self.enc_mode == "column":
            return feat.grad.abs().sum(dim=-1)
        else:  # self.enc_mode == "fuse"
            # Note that in fuse mode, the gradients across different columns
            # are the same due to the GPU memory restriction.
            return feat.grad.abs().sum(dim=1, keepdim=True).repeat(
                1, self.num_cols) / self.num_cols
