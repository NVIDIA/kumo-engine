# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from kumo_core.modeling.base import Module
from kumo_core.modeling.gnn.positional import PositionalEncoding


class TransformerEncoder(torch.nn.Module):
    r"""A Transformer encoder model.

    Args:
        channels: The hidden feature dimensionality.
        num_heads: The number of heads in the MultiHeadAttention models
        num_layers: Number of transformer layers to stack.
        dropout: Dropout value for the TransformerEncoderLayer.
        act: The non-linear activation function to use.
        apply_final_norm: If :obj:`True`, it applies a final normalization
            before returning the output.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        act: str | None = 'relu',
        apply_final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.apply_final_norm = apply_final_norm

        if self.act is not None:
            act_fn = activation_resolver(self.act)
        else:
            act_fn = torch.nn.Identity()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=self.num_heads,
            dropout=self.dropout,
            dim_feedforward=self.channels * 4,
            activation=act_fn,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=torch.nn.LayerNorm(self.channels)
            if self.apply_final_norm else None,
            enable_nested_tensor=True,
        )

    def reset_parameters(self) -> None:
        for layer in self.transformer.layers:
            for child in layer.children():
                if isinstance(child, torch.nn.MultiheadAttention):
                    child._reset_parameters()
                elif hasattr(child, 'reset_parameters'):
                    child.reset_parameters()

    def forward(self, data: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        In the following: :math:`B` is the batch size, :math:`S` is the
        sequence length, and :math:`D` is the feature dimensionality
        (matching ``channels``).

        Args:
            data: input node embeddings for :math:`B` disjoint subgraphs with a
                maximum of :math:`S` nodes each. Shape: :math:`(B, S, D)`.
            mask: Allows for specified elements to be ignored by the attention.
                Typically used to mask out padding values.
                If a BoolTensor is provided, the positions with the value of
                :obj:`True` will be ignored while the position with the value
                of :obj:`False` will be unchanged. Shape: :math:`(B, S)`.

        Returns:
            Tensor of shape :math:`(B, S, D)`.
        """
        return self.transformer(data, src_key_padding_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  channels={self.channels},\n'
                f'  num_layers={self.num_layers},\n'
                f'  num_heads={self.num_heads},\n'
                f'  dropout={self.dropout},\n'
                f'  act={self.act},\n'
                f'  apply_final_norm={self.apply_final_norm},\n'
                f')')


@register(with_target=True)
class GraphTransformer(Module, ConfigMixin):
    r"""A Graph Transformer model.

    Args:
        channels: The hidden feature dimensionality.
        num_heads: The number of heads for multi-head attention.
        num_layers: Number of transformer layers to stack.
        dropout: Dropout value of transformer layers.
        act: The non-linear activation function to use.
        apply_final_norm: If :obj:`True`, it applies a final normalization
            before returning the output.
        seq_len: The maximum sequence length.
        positional_encodings: The positional encodings to use.
    """
    LAZY_ATTRS = {'seq_len'}

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        act: str | None = 'relu',
        apply_final_norm: bool = False,
        seq_len: int | None = None,
        positional_encodings: list[PositionalEncoding] | None = None,
    ) -> None:
        super().__init__(
            channels,
            num_heads,
            num_layers,
            dropout,
            act,
            apply_final_norm,
            seq_len,
        )
        positional_encodings = positional_encodings or []
        self.positional_encodings = torch.nn.ModuleList(positional_encodings)

    def init_modules(self) -> None:
        self.transformer = TransformerEncoder(
            channels=self.channels,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            apply_final_norm=self.apply_final_norm,
        )

    def reset_parameters(self) -> None:
        self.transformer.reset_parameters()
        for positional_encoding in self.positional_encodings:
            positional_encoding.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        batch_size: int,
        entity_table_name: str,
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x_dict: The node features for every node type.
            edge_index_dict: The edge indices for every edge type.
            batch_dict: The node to subgraph mapping for every node type.
            batch_size: The batch size :math:`B`.
            entity_table_name: Name of the entity table.
            num_sampled_nodes_dict: The number of sampled nodes for every node
                type.

        Returns:
            Tensor of shape :math:`(B, S, D)` where the seed nodes coming from
            the entity table are the first nodes in each batch.
        """
        x_dict = self.add_positional_encodings(x_dict, edge_index_dict,
                                               num_sampled_nodes_dict)

        batched_data, padding_masks = self.get_transformer_batch_data(
            x_dict, batch_dict, entity_table_name, batch_size,
            seq_len=self.seq_len)
        return self.transformer(batched_data, mask=padding_masks)

    def add_positional_encodings(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        for positional_encoding in self.positional_encodings:
            pos_dict = positional_encoding(edge_index_dict,
                                           num_sampled_nodes_dict)
            for table_name, pos_embeds in pos_dict.items():
                x_dict[table_name] = x_dict[table_name] + pos_embeds
        return x_dict

    @staticmethod
    def get_transformer_batch_data(
        x_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        entity_table_name: str,
        batch_size: int,
        seq_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Get batched data for transformer.

        Args:
            x_dict: The node features for every node type.
            batch_dict: The node to subgraph mapping for every node type.
            entity_table_name: Name of the entity table.
            batch_size: The batch size :math:`B`.
            seq_len: Max sequence length :math:`S`.

        Returns:
            A tuple with:
                - Batched data of shape :math:`(B, S, D)`
                - Padding mask (True means 'ignore'). Shape: :math:`(B, S)`
        """
        # stack all values, except seed nodes
        node_embeds_list = []
        node2subgraph_list = []
        for table_name, node_embeds in x_dict.items():
            node2subgraph = batch_dict[table_name]

            if table_name == entity_table_name:
                # only add values NOT corresponding to seed nodes
                node_embeds_list.append(node_embeds[batch_size:])
                node2subgraph_list.append(node2subgraph[batch_size:])
            else:
                node_embeds_list.append(node_embeds)
                node2subgraph_list.append(node2subgraph)

        node_embeds = torch.cat(node_embeds_list, dim=0)
        node2subgraph = torch.cat(node2subgraph_list, dim=0)
        seed_nodes = x_dict[entity_table_name][:batch_size]

        # Sort elements by bucket index: it's a precond for to_dense_batch
        sorted_node2subgraph, sorted_idx = node2subgraph.sort()
        sorted_node_embeds = node_embeds[sorted_idx]

        # batched: (B, S - 1, D), masks: (B, S - 1) bool values are inverted
        batched, masks = to_dense_batch(sorted_node_embeds,
                                        sorted_node2subgraph,
                                        max_num_nodes=seq_len - 1,
                                        batch_size=batch_size)

        # Add seeds as first node in each element
        batched = torch.cat((seed_nodes.unsqueeze(1), batched), dim=1)
        # Invert mask value and add mask for seeds
        masks = torch.cat((masks.new_zeros((batch_size, 1)), ~masks), dim=1)
        return batched, masks

    def get_extra_state(self) -> dict[str, Any]:
        """Get extra state.

        This function is called when building the module's state_dict().

        Returns:
            A dictionary with the sequence length.
        """
        return {'seq_len': self.seq_len}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Set extra state.

        This function is called from load_state_dict() to handle any extra
        state found within the state_dict.

        Args:
            state: A dictionary with the sequence length.
        """
        self.seq_len = state['seq_len']

    def __repr__(self) -> str:
        pos_encodings = [
            str(pos_encoding) for pos_encoding in self.positional_encodings
        ]
        return (f'{self.__class__.__name__}(\n'
                f'  channels={self.channels},\n'
                f'  num_heads={self.num_heads},\n'
                f'  num_layers={self.num_layers},\n'
                f'  dropout={self.dropout},\n'
                f'  act={self.act},\n'
                f'  apply_final_norm={self.apply_final_norm},\n'
                f'  positional_encodings={pos_encodings},\n'
                f')')

    @staticmethod
    def get_seq_len(total_num_nodes: int, batch_size: int) -> int:
        """Computes avg size of graph and gets the closest ceiling power of 2.

        First, it computes the average subgraph size based on the total number
        of nodes and the batch size. Then doubles it, since this is just the
        average, in order to include most of the nodes per subgraph. Finally,
        it computes the closest ceiling power of 2, so be used as sequence
        length.
        """
        num_nodes = total_num_nodes * 2 // batch_size
        if num_nodes <= 0:
            return 1
        if num_nodes >= 256:
            return 256

        # Find the next power of 2
        power = 1
        while power < num_nodes:
            power *= 2
        return power
