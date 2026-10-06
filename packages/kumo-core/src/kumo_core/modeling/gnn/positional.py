# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import EdgeType, NodeType


class PositionalEncoding(torch.nn.Module, ABC, ConfigMixin):
    """Abstract base class for positional encoders.

    Classes that inherit from this should implement the forward method to
    compute positional embeddings for nodes in a graph.
    """
    @abstractmethod
    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""

    @abstractmethod
    def forward(
        self,
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        """Compute positional embeddings.

        Args:
            edge_index_dict: The edge indices for every edge type.
            num_sampled_nodes_dict: The number of sampled nodes for every node
                type.

        Returns:
            Dictionary mapping node types to their positional embeddings.
        """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@register(with_target=True)
class HopPositionalEncoding(PositionalEncoding):
    """Computes positional encodings based on hop distance to the seed nodes.

    Args:
        channels: Feature dimensionality.
        num_hops: Max number of hops to consider.
    """
    def __init__(self, channels: int, num_hops: int) -> None:
        assert channels > 0, "channels must be greater than 0"
        assert num_hops > 0, "num_hops must be greater than 0"
        super().__init__()
        self.channels = channels
        self.num_hops = num_hops

        self.emb = torch.nn.Embedding(self.num_hops + 1, channels)

    def reset_parameters(self) -> None:
        self.emb.reset_parameters()

    def forward(
        self,
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        if num_sampled_nodes_dict is None:
            raise ValueError(f"`num_sampled_nodes_dict` must be provided for "
                             f"`{self.__class__.__name__}`")

        device = self.emb.weight.device

        hop_dict = {}
        for table_name, num_sampled_nodes in num_sampled_nodes_dict.items():
            hop = torch.arange(self.num_hops + 1, device=device)
            hop = hop.repeat_interleave(
                torch.tensor(num_sampled_nodes, device=device))
            hop_dict[table_name] = hop

        return {
            table_name: self.emb(hop_dict[table_name])
            for table_name in num_sampled_nodes_dict.keys()
        }


@register(with_target=True)
class RandomGNNPositionalEncoding(PositionalEncoding):
    """A random GNN used to compute positional encodings.

    Creates a random positional embedding for each node in the graph, and
    passes it through a GNN to compute a positional encoding.

    Args:
        channels: Feature dimensionality.
        edge_types: List of edge types to consider.
        num_layers: Number of GNN layers to stack.
        act: The non-linear activation function to use.
    """
    def __init__(
        self,
        channels: int,
        edge_types: list[EdgeType],
        num_layers: int = 2,
        act: str = 'selu',
    ) -> None:
        assert channels > 0, "channels must be greater than 0"
        assert num_layers > 0, "num_layers must be greater than 0"
        assert len(edge_types) > 0, "edge_types must be non-empty"
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.edge_types = edge_types
        self.act = act

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                tuple(edge): SAGEConv(channels, channels)
                for edge in edge_types
            }
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)

        self.act_fn = activation_resolver(act)
        self.lin = torch.nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[NodeType, Tensor]:
        if num_sampled_nodes_dict is None:
            raise ValueError(f"`num_sampled_nodes_dict` must be provided for "
                             f"`{self.__class__.__name__}`")

        device = self.lin.weight.device
        pos_dict = {
            k:
            torch.randn(
                (sum(v), self.channels),
                generator=generator,
                device=device,
            )
            for k, v in num_sampled_nodes_dict.items()
        }

        for conv in self.convs:
            pos_dict = conv(pos_dict, edge_index_dict)
            pos_dict = {k: self.act_fn(v) for k, v in pos_dict.items()}

        return {k: self.lin(v) for k, v in pos_dict.items()}


@register(with_target=True)
class InvariantRandomGNNPositionalEncoding(PositionalEncoding):
    """A table-invariant random GNN used to compute positional encodings.

    Creates a random positional embedding for each node in the graph, and
    passes it through a GNN to compute a positional encoding.

    The main difference to :class:`RandomGNNPositionalEncoding` is that this
    positional encoding is table/graph-invariant, and is thus not tied to a
    specific graph.

    Args:
        channels: Feature dimensionality.
        num_layers: Number of GNN layers to stack.
        act: The non-linear activation function to use.
    """
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        act: str = 'selu',
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_layers = num_layers
        self.act = act

        self.convs = torch.nn.ModuleList(
            [SAGEConv(channels, channels) for _ in range(num_layers)])
        self.act_fn = activation_resolver(act)
        self.lin = torch.nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[NodeType, Tensor]:
        if num_sampled_nodes_dict is None:
            raise ValueError(f"`num_sampled_nodes_dict` must be provided for "
                             f"`{self.__class__.__name__}`")

        device = self.lin.weight.device

        # Convert the heterogeneous `edge_index_dict` into a homogeneous single
        # `edge_index` by grouping all nodes and offsetting source and
        # destination node indices according to node type:
        num_nodes = 0
        cumsum_dict: dict[NodeType, int] = {}
        for table_name, num_sampled_nodes in num_sampled_nodes_dict.items():
            cumsum_dict[table_name] = num_nodes
            num_nodes += sum(num_sampled_nodes)

        edge_indices: list[Tensor] = []
        for (src, _, dst), edge_index in edge_index_dict.items():
            offset = torch.tensor(
                [[cumsum_dict[src]], [cumsum_dict[dst]]],
                device=device,
            )
            edge_index = edge_index + offset
            edge_indices.append(edge_index)

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(edge_indices, dim=1)

        pos = torch.randn(
            (num_nodes, self.channels),
            generator=generator,
            device=device,
        )

        for conv in self.convs:
            pos = self.act_fn(conv(pos, edge_index))
        pos = self.lin(pos)

        cumsums = list(cumsum_dict.values()) + [num_nodes]
        pos_dict = {
            table_name: pos[start:end]
            for table_name, start, end in zip(
                cumsum_dict.keys(),
                cumsums[:-1],
                cumsums[1:],
            )
        }

        return pos_dict


@register(with_target=True)
class InvariantTableTypePositionalEncoding(PositionalEncoding):
    """Data-agnostic positional encoding based on table type.

    This positional encoding is data-agnostic, meaning it does not depend on
    the specific data in the graph. It is computed by creating a random
    embedding for each table type and encoding them with a linear layer.

    Args:
        channels: feature dimensionality.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.lin = torch.nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(
        self,
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[NodeType, Tensor]:
        if num_sampled_nodes_dict is None:
            raise ValueError(f"`num_sampled_nodes_dict` must be provided for "
                             f"`{self.__class__.__name__}`")

        type_embeds = torch.randn(
            (len(num_sampled_nodes_dict), self.channels),
            generator=generator,
            device=self.lin.weight.device,
        )
        type_embeds = self.lin(F.normalize(type_embeds, dim=1))

        return {
            name: type_embeds[i].expand(sum(num_sampled_nodes_dict[name]), -1)
            for i, name in enumerate(num_sampled_nodes_dict)
        }


# Global mapping from string name to positional encoding class
_POS_ENCODING_CLASSES = {
    "hop": HopPositionalEncoding,
    "random_gnn": RandomGNNPositionalEncoding,
    "invariant_table_type": InvariantTableTypePositionalEncoding,
    "invariant_random_gnn": InvariantRandomGNNPositionalEncoding,
}


def positional_encoding_resolver(
    positional_encoding_name: str,
    **kwargs: Any,
) -> PositionalEncoding:
    """Resolve a positional embedding class from a string and its parameters.

    Args:
        positional_encoding_name: The name of the positional encoding class.
        **kwargs: Keyword arguments to pass to the positional encoding class
        constructor.
    """
    if positional_encoding_name not in _POS_ENCODING_CLASSES:
        raise ValueError(f"Could not resolve '{positional_encoding_name}' "
                         f"among the available positional "
                         f"encodings: {list(_POS_ENCODING_CLASSES.keys())}")

    cls = _POS_ENCODING_CLASSES[positional_encoding_name]

    # Filter only those accepted by the class's __init__
    sig = inspect.signature(cls.__init__)
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items() if k in sig.parameters and v is not None
    }
    return cls(**filtered_kwargs)
