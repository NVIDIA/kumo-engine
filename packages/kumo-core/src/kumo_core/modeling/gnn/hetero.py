# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Dropout, Linear
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.index import index2ptr
from torch_geometric.nn import Aggregation, BatchNorm, MultiAggregation
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import scatter

from kumo_core.modeling.base import Module


class HeteroPostLayer(torch.nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        channels: int,
        dropout: float,
        act: str | None,
        act_first: bool,
        norm: str | None,
        norm_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.layer_dict: ModuleDict[torch.nn.ModuleList] = ModuleDict()
        for key in node_types:
            layers = []
            if act is not None and act_first:
                layers.append(activation_resolver(act))

            if norm is not None:
                norm_kwargs = norm_kwargs or {}
                if device is not None:
                    norm_kwargs['device'] = device
                norm_layer = normalization_resolver(norm, channels,
                                                    **norm_kwargs)
                if (isinstance(norm_layer, BatchNorm)
                        and norm_layer.module.track_running_stats):
                    norm_layer.allow_single_element = True
                layers.append(norm_layer)

            if act is not None and not act_first:
                layers.append(activation_resolver(act))

            layers.append(Dropout(p=dropout))
            self.layer_dict[key] = torch.nn.ModuleList(layers)

    def reset_parameters(self) -> None:
        for layers in self.layer_dict.values():
            for layer in layers:
                attr = getattr(layer, 'reset_parameters', None)
                if attr is not None and callable(attr):
                    layer.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        out_dict: dict[NodeType, Tensor] = {}
        for key, x in x_dict.items():
            for layer in self.layer_dict[key]:
                x = layer(x)
            out_dict[key] = x
        return out_dict


class HeteroLinear(torch.nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.lin_dict = ModuleDict()
        for key in node_types:
            self.lin_dict[key] = Linear(
                in_channels,
                out_channels,
                bias=bias,
                device=device,
            )

    def reset_parameters(self) -> None:
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        return {key: self.lin_dict[key](x) for key, x in x_dict.items()}


class HeteroIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def reset_parameters(self) -> None:
        pass

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        return x_dict


class ForeignKeyToPrimaryKeyConv(torch.nn.Module):
    r"""A simplified GNN layer that is designed to send messages from foreign
    keys to primary keys.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str | list[str] | Aggregation,
        aggr_kwargs: dict[str, Any] | None = None,
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if aggr == 'patch_transformer':
            aggr_kwargs = aggr_kwargs or {}
            aggr_kwargs.setdefault('in_channels', in_channels)
            aggr_kwargs.setdefault('hidden_channels', in_channels)
            aggr_kwargs.setdefault('out_channels', out_channels)
            aggr_kwargs.setdefault('device', device)

        aggr = aggregation_resolver(aggr, **(aggr_kwargs or {}))

        if isinstance(aggr, MultiAggregation):
            in_channels = aggr.get_out_channels(in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.aggr = aggr
        self.lin = Linear(in_channels, out_channels, bias=bias, device=device)

    def reset_parameters(self) -> None:
        self.aggr.reset_parameters()
        self.lin.reset_parameters()

    def empty_output(self, num_nodes: int) -> Tensor:
        if not self.bias:
            return self.lin.weight.new_zeros(num_nodes, self.out_channels)
        return self.lin.bias.view(1, -1).expand(num_nodes, self.out_channels)

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
        deterministic: bool = False,
    ) -> Tensor:

        if edge_index.size(1) == 0:
            # No need for aggregation, just return bias shared for all nodes:
            return self.empty_output(x_dst.size(0))
        assert x_dst.size(0), "Missing destination nodes with inbound edges."

        if deterministic:
            col, perm = edge_index[1].sort(stable=True)
            ptr = index2ptr(col, size=x_dst.size(0))
            out = self.aggr(x_src[edge_index[0][perm]], ptr=ptr,
                            dim_size=x_dst.size(0))
        else:
            out = self.aggr(x_src[edge_index[0]], edge_index[1],
                            dim_size=x_dst.size(0))

        out = self.lin(out)

        return out


class PrimaryKeyToForeignKeyConv(torch.nn.Module):
    r"""A simplified GNN layer that is designed to send messages from primary
    keys to foreign keys.

    Specifically, this module simplifies to a single gather (no aggregation
    required since every foreign key only receives information from a single
    primary key).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.lin = Linear(in_channels, out_channels, bias=bias, device=device)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def empty_output(self, num_nodes: int) -> Tensor:
        if not self.bias:
            return self.lin.weight.new_zeros(num_nodes, self.out_channels)
        return self.lin.bias.view(1, -1).expand(num_nodes, self.out_channels)

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
    ) -> Tensor:

        if edge_index.size(1) == 0:
            # No need for gathering, just return bias shared for all nodes:
            return self.empty_output(x_dst.size(0))

        out = scatter(x_src[edge_index[0]], edge_index[1],
                      dim_size=x_dst.size(0), reduce='any')
        out = self.lin(out)

        return out


def get_hetero_conv_module_dict(
    edge_types: list[EdgeType],
    in_channels: int,
    out_channels: int,
    aggr: str | list[str] | Aggregation | None = None,
    aggr_kwargs: dict[str, Any] | None = None,
    bias: bool = True,
    device: torch.device | None = None,
) -> ModuleDict:

    module_dict = ModuleDict()
    for edge_type in edge_types:
        # Currently, we leverage the relation name to determine whether
        # messages are passed from fkey->pkey or from pkey->fkey.
        # Specifically, every original edge type denotes a fkey->pkey flow,
        # while all reverse edge types denote a pkey->fkey flow.
        # TODO (matthias) Replace with a dedicated `EdgeType` configuration.
        _, rel, _ = edge_type
        if rel.startswith('rev_'):  # pkey->fkey:
            conv: torch.nn.Module = PrimaryKeyToForeignKeyConv(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                device=device,
            )
        else:  # fkey->pkey
            conv = ForeignKeyToPrimaryKeyConv(
                in_channels=in_channels,
                out_channels=out_channels,
                aggr=aggr,
                aggr_kwargs=aggr_kwargs,
                bias=bias,
                device=device,
            )
        module_dict[tuple(edge_type)] = conv

    return module_dict


@register(with_target=True)
class HeteroGNN(Module, ConfigMixin):
    r"""A heterogeneous GNN model with support for custom aggregations.

    Args:
        channels: The hidden feature dimensionality.
        num_mp_layers: The number of GNN layers.
        metadata: The metadata of a :class:`~torch_geometric.data.HeteroData`.
        num_pre_mp_layers: The number of transformation layers prior to GNN
            execution.
        aggr: The aggregation scheme of the underlying GNN layer.
        dropout: Dropout probability of each hidden embedding.
        act: The non-linear activation function to use.
        act_first: If set to :obj:`True`, activation is applied before
            normalization.
        norm: The normalization function to use.
        bias: If set to :obj:`False`, the GNN will not learn an additive bias.
        learn_skip: If set to :obj:`False`, will apply a non-learnable skip
            connection with an identity mapping. If set to :obj:`True`, will
            apply a learnable skip connection with a linear projection.
        device: The device to use.
    """
    LAZY_ATTRS = {'metadata'}

    def __init__(
        self,
        channels: int,
        num_mp_layers: int,
        *,
        metadata: Metadata | None = None,
        num_pre_mp_layers: int = 0,
        aggr: str | list[str] | None = None,
        aggr_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        act: str | None = 'relu',
        act_first: bool = False,
        norm: str | None = 'batch_norm',
        bias: bool = True,
        learn_skip: bool = True,
        deterministic: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            channels,
            num_mp_layers,
            metadata,
            num_pre_mp_layers,
            aggr or ['sum', 'mean', 'max'],
            aggr_kwargs,
            dropout,
            act,
            act_first,
            norm,
            bias,
            learn_skip,
            deterministic,
            device,
        )

    def init_modules(self) -> None:
        norm_kwargs: dict[str, Any] | None = None
        if self.norm == 'layer_norm':
            norm_kwargs = dict(mode='node')

        assert isinstance(self.metadata, Iterable)
        node_types, edge_types = self.metadata

        # Pre-Message Passing Layers ##########################################
        self.lins = torch.nn.ModuleList()
        self.post_lins = torch.nn.ModuleList()
        for _ in range(self.num_pre_mp_layers):
            self.lins.append(
                HeteroLinear(
                    node_types,
                    in_channels=self.channels,
                    out_channels=self.channels,
                    bias=self.bias,
                    device=self.device,
                ))
            self.post_lins.append(
                HeteroPostLayer(
                    node_types,
                    channels=self.channels,
                    dropout=self.dropout,
                    act=self.act,
                    act_first=self.act_first,
                    norm=self.norm,
                    norm_kwargs=norm_kwargs,
                    device=self.device,
                ))

        # Message Passing Layers ##############################################
        self.hetero_convs = torch.nn.ModuleList()
        self.post_convs = torch.nn.ModuleList()
        self.root_skips = torch.nn.ModuleList()
        for _ in range(self.num_mp_layers):
            self.hetero_convs.append(
                get_hetero_conv_module_dict(
                    edge_types,
                    in_channels=self.channels,
                    out_channels=self.channels,
                    aggr=self.aggr,
                    aggr_kwargs=self.aggr_kwargs,
                    bias=self.bias,
                    device=self.device,
                ))

            if self.learn_skip:
                self.root_skips.append(
                    HeteroLinear(
                        node_types,
                        in_channels=self.channels,
                        out_channels=self.channels,
                        bias=self.bias,
                        device=self.device,
                    ))
            else:
                self.root_skips.append(HeteroIdentity())

            self.post_convs.append(
                HeteroPostLayer(
                    node_types,
                    channels=self.channels,
                    dropout=self.dropout,
                    act=self.act,
                    act_first=self.act_first,
                    norm=self.norm,
                    norm_kwargs=norm_kwargs,
                    device=self.device,
                ))

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()
        for post_lin in self.post_lins:
            post_lin.reset_parameters()
        for hetero_conv in self.hetero_convs:
            for conv in hetero_conv.values():
                conv.reset_parameters()
        for root_skip in self.root_skips:
            root_skip.reset_parameters()
        for post_conv in self.post_convs:
            post_conv.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> dict[NodeType, Tensor]:
        r"""The forward pass.

        Args:
            x_dict: The node features for every node type.
            edge_index_dict: The edge indices for every edge_type.
            num_sampled_nodes_dict: The number of sampled nodes per hop in
                every node type. Useful in neighbor sampling scenarios to only
                operate on minimal-sized representations.
            num_sampled_edges_dict: The number of sampled edges per hop in
                every edge type. Useful in neighbor sampling scenarios to only
                operate on minimal-sized representations.
        """
        if num_sampled_nodes_dict is not None:
            assert num_sampled_edges_dict is not None
            for key, num_sampled_nodes in num_sampled_nodes_dict.items():
                assert len(num_sampled_nodes) <= self.num_mp_layers + 1, (
                    f"Mismatch between number of hops "
                    f"({len(num_sampled_nodes) - 1}) and layers "
                    f"({self.num_mp_layers}) for key {key}. If this is "
                    f"intended, pass num_sampled_nodes_dict as None.")
        else:
            assert num_sampled_edges_dict is None

        for lin, post_lin in zip(self.lins, self.post_lins):
            x_dict = lin(x_dict)
            x_dict = post_lin(x_dict)

        # Ensure that number of sampled nodes and edges have correct shape.
        # This is, e.g., violated in case of single tables/no edge types.
        if num_sampled_nodes_dict is not None:
            num_sampled_nodes_dict = {
                key: value + [0] * max(self.num_mp_layers + 1 - len(value), 0)
                for key, value in num_sampled_nodes_dict.items()
            }
        if num_sampled_edges_dict is not None:
            num_sampled_edges_dict = {
                key: value + [0] * max(self.num_mp_layers - len(value), 0)
                for key, value in num_sampled_edges_dict.items()
            }

        for i, hetero_conv in enumerate(self.hetero_convs):
            # Iteratively trim node features and edge indices such that we only
            # maintain the information needed for the current layer:
            x_src_dict = x_dst_dict = x_dict
            if (num_sampled_nodes_dict is not None
                    and len(num_sampled_nodes_dict) > 0):
                x_dst_dict = {
                    key: x[:x.size(0) - num_sampled_nodes_dict[key][-(i + 1)]]
                    for key, x in x_src_dict.items()
                }
            if (num_sampled_edges_dict is not None
                    and len(num_sampled_edges_dict) > 0 and i > 0):
                edge_index_dict = {
                    key: e[:, :e.size(1) - num_sampled_edges_dict[key][-i]]
                    for key, e in edge_index_dict.items()
                }

            # Apply GNN layers over each edge type sequentially:
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                x_src, x_dst = x_src_dict[src], x_dst_dict[dst]
                conv = hetero_conv[edge_type]
                if isinstance(conv, ForeignKeyToPrimaryKeyConv):
                    out = conv(x_src, x_dst, edge_index, self.deterministic)
                else:
                    out = conv(x_src, x_dst, edge_index)
                out_dict[dst].append(out)

            # Group embeddings pointing to the same destination node type:
            for node_type, out in out_dict.items():
                if len(out) == 1:
                    out_dict[node_type] = out[0]
                elif len(out) > 1:
                    out_dict[node_type] = sum(out)

            # Apply post-processing (skip-connection + normalization):
            root_dict = self.root_skips[i](x_dst_dict)
            for key in root_dict.keys():
                if key not in out_dict:
                    out_dict[key] = root_dict[key]
                else:
                    out_dict[key] = out_dict[key] + root_dict[key]

            x_dict = self.post_convs[i](out_dict)

        return x_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  channels={self.channels},\n'
                f'  num_mp_layers={self.num_mp_layers},\n'
                f'  num_pre_mp_layers={self.num_pre_mp_layers},\n'
                f'  aggr={self.aggr},\n'
                f'  dropout={self.dropout},\n'
                f'  act={self.act},\n'
                f'  act_first={self.act_first},\n'
                f'  norm={self.norm},\n'
                f'  bias={self.bias},\n'
                f'  learn_skip={self.learn_skip},\n'
                f')')
