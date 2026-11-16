# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, NamedTuple

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn import MLP

from kumo_core.modeling.base import Module


class LinkPredHeadOutput(NamedTuple):
    pos_score: Tensor
    neg_score: Tensor | None
    lhs_x: Tensor
    rhs_pos_x: Tensor
    rhs_neg_x: Tensor | None


@register(with_target=True)
class LinkPredHead(Module, ConfigMixin):
    r"""A link prediction head.

    Args:
        in_channels: Size of each input sample.
        use_rhs_projector: Whether to use a RHS projector. For two-tower
            models, set to :obj:`True` since we need it to map GNN-generated
            RHS embeddings into ``embedding_dim``. For other models, set to
            :obj:`False` because they use shallow RHS embeddings whose
            dimensionality is already ``embedding_dim``.
        use_mlp_head: Whether to use MLP scorer (MLP head used by ID-GNN to
            score RHS nodes given LHS nodes). If ID/Hybrid GNNs, set to
            :obj:`True`, otherwise :obj:`False`.
        embedding_dim: Dimension of output embeddings.
        normalize_vectors: Whether to normalize embedding vectors. Only
            supported for two-tower GNN models.
        hidden_channels: Size of each hidden sample. If set to :obj:`None`,
            will set ``hidden_channels = in_channels``.
        num_layers: The number of layers.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
        act_first: If set to :obj:`True`, activation is applied before
            normalization.
        norm: The normalization function to use.
    """
    LAZY_ATTRS = {'in_channels', 'use_rhs_projector', 'use_mlp_head'}

    def __init__(
        self,
        in_channels: int | None = None,
        use_rhs_projector: bool | None = None,
        use_mlp_head: bool | None = None,
        embedding_dim: int = 32,
        normalize_vectors: bool = False,
        hidden_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        act: str | None = 'relu',
        act_first: bool = False,
        norm: str | None = 'batch_norm',
    ):
        super().__init__(
            in_channels=in_channels,
            use_rhs_projector=use_rhs_projector,
            use_mlp_head=use_mlp_head,
            embedding_dim=embedding_dim,
            normalize_vectors=normalize_vectors,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            act_first=act_first,
            norm=norm,
        )

    def init_modules(self) -> None:
        self.lhs_projector = Linear(self.in_channels, self.embedding_dim)

        if self.use_rhs_projector:
            # RHS projector used by two-tower GNN model:
            self.rhs_projector = Linear(self.in_channels, self.embedding_dim)

        if self.use_mlp_head:
            # MLP scorer used by ID/Hybrid GNN models:
            norm_kwargs: dict[str, Any] | None = None
            if self.norm == 'layer_norm':
                norm_kwargs = dict(mode='node')

            self.mlp_head = MLP(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels or self.in_channels,
                out_channels=1,  # Predict scalar logits.
                num_layers=self.num_layers,
                dropout=self.dropout,
                act=self.act,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=norm_kwargs,
            )

    def reset_parameters(self) -> None:
        self.lhs_projector.reset_parameters()
        if self.use_rhs_projector:
            self.rhs_projector.reset_parameters()
        if self.use_mlp_head:
            self.mlp_head.reset_parameters()

    def lhs_forward(self, x: Tensor) -> Tensor:
        x = self.lhs_projector(x)
        if self.normalize_vectors:
            x = F.normalize(x, dim=-1)
        return x

    def rhs_forward(self, x: Tensor) -> Tensor:
        assert self.use_rhs_projector
        x = self.rhs_projector(x)
        if self.normalize_vectors:
            x = F.normalize(x, dim=-1)
        return x

    def forward(
        self,
        lhs_x: Tensor,
        rhs_pos_x: Tensor,
        rhs_neg_x: Tensor | None = None,
        share_negatives: bool = False,
    ) -> LinkPredHeadOutput:
        """Returns distances between positive and negative pairs of nodes.

        .. note::
            This is only used in two-tower GNN code path.
        """
        lhs_x = self.lhs_forward(lhs_x)
        rhs_pos_x = self.rhs_forward(rhs_pos_x)
        if rhs_neg_x is not None:
            rhs_neg_x = self.rhs_forward(rhs_neg_x)

        # Ensure extra dimension when there is only 1 negative.
        if rhs_neg_x is not None and rhs_neg_x.dim() < 3:
            rhs_neg_x = rhs_neg_x.unsqueeze(1)

        # lhs_x: [batch_size, emb_dim]
        # rhs_pos_x: [batch_size, emb_dim]
        # rhs_neg_x: [batch_size, neg_amount, emb_dim]
        pos_score = (lhs_x * rhs_pos_x).sum(dim=-1)
        neg_score: Tensor | None = None
        if rhs_neg_x is not None:
            if share_negatives:
                # [batch_size * neg_amount, emb_dim]
                rhs_neg_x = rhs_neg_x.view(-1, rhs_neg_x.size(-1))
                # [batch_size, emb_dim] @ [emb_dim, batch_size * neg_amount]
                # -> [batch_size, batch_size * neg_amount]
                neg_score = lhs_x @ rhs_neg_x.t()
            else:
                # [batch_size, neg_amount]
                neg_score = (lhs_x.unsqueeze(1) * rhs_neg_x).sum(dim=-1)

        return LinkPredHeadOutput(
            pos_score=pos_score,
            neg_score=neg_score,
            lhs_x=lhs_x,
            rhs_pos_x=rhs_pos_x,
            rhs_neg_x=rhs_neg_x,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'{self.embedding_dim}, '
                f'use_rhs_projector={self.use_rhs_projector}, '
                f'use_mlp_head={self.use_mlp_head})')
