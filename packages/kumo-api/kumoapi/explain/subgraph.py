# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic.dataclasses import dataclass

from kumoapi.task import TaskType
from kumoapi.typing import _FORBID_EXTRA_CONFIG, Stype


@dataclass
class Feature:
    r"""Represents a feature column in the
    :class:`~kumoapi.explain.subgraph.SubgraphFeatureStore`.

    Args:
        table_name (str): The name of the table.
        name (str): The column name of the feature.
        value (Any): The value of the feature.
        stype (Stype): The stype of the feature.
        score (float, optional): Feature importance score if available.
    """
    table_name: Optional[str] = None
    name: Optional[str] = None
    value: Optional[Any] = None
    stype: Optional[Stype] = None
    score: Optional[float] = None


@dataclass(config=_FORBID_EXTRA_CONFIG)
class SubgraphTopFeatureRequest:
    r"""Request to get the `top_k` features for a subgraph
    based on the feature score.

    Args:
        top_k (int): The number of top features to get.
    """
    top_k: int = 3


@dataclass
class SubgraphNode:
    r"""Represents a node in the subgraph.

    Args:
        id (int): The unique integer ID of the node.
        pkey_name (str, optional): The name of the primary key column.
        pkey (Any, optional): The value of the primary key.
    """
    id: int
    pkey_name: Optional[str] = None
    pkey: Optional[Any] = None


@dataclass
class AggregatedNodes:
    r"""Aggregates multiple nodes in the subgraph in case
    the number of nodes exceeds the limit.

    Args:
        id (int): The unique integer ID of the node. The ID is negative
            for aggregated nodes.
        count (int): The number of nodes aggregated.
        start_time (datetime, optional): The start time of the aggregation
            period if available.
        end_time (datetime, optional): The end time of the aggregation
            period if available.
        pkey_name (str, optional): The name of the primary key column
            if available.
    """
    id: int
    count: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    pkey_name: Optional[str] = None


@dataclass
class SubgraphStore:
    r"""Stores the subgraph structure, including the ID for entity (root)
    node, nodes and edges. The nodes are stored in a dictionary with the
    node ID as the key. The edges are stored in a dictionary with the source
    node ID as the key and a dictionary of relation types to a list of
    destination node IDs. The tables for each node are stored in a dictionary
    with the table name as the key.

    Args:
        root_id (int): The ID of the root node.
        task_type (TaskType): The type of the task.
        seed_time (datetime, optional): The seed time of the subgraph.
        nodes (Dict[int, Union[SubgraphNode, AggregatedNodes]], optional):
            The nodes in the subgraph.
        edges (Dict[int, Dict[str, List[int]]], optional): The edges in
            the subgraph, where the key is the source node ID and the value
            is a dictionary of relation types to a list of destination node
            IDs.
        tables (Dict[str, List[int]], optional): The list of node IDs
            for each table.
    """
    root_id: int
    task_type: TaskType
    seed_time: Optional[datetime] = None
    nodes: Dict[int, Union[SubgraphNode,
                           AggregatedNodes]] = field(default_factory=dict)
    edges: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    tables: Dict[str, List[int]] = field(default_factory=dict)

    def __getitem__(
        self,
        node_id: int,
    ) -> Optional[Union[SubgraphNode, AggregatedNodes]]:
        r"""Get a node by ID. If the node does not exist, return :obj:`None`.

        Returns:
            Optional[Union[SubgraphNode, AggregatedNodes]]: The node to get.
        """
        if node_id not in self.nodes:
            return None
        return self.nodes[node_id]

    def get_table_node_ids(self, table_name: str) -> List[int]:
        r"""Returns the list of node IDs for a table.

        Args:
            table_name (str): The name of the table.

        Returns:
            List[int]: The list of node IDs for the table.
        """
        return self.tables.get(table_name, [])

    def degree(self, node_id: int) -> int:
        r"""Returns the degree of a node."""
        if node_id not in self.edges:
            return 0
        return sum(len(v) for v in self.edges[node_id].values())

    def add_node(
        self,
        node: Union[SubgraphNode, AggregatedNodes],
        table_name: str,
    ) -> None:
        r"""Adds a node to the subgraph.

        Args:
            node (Union[SubgraphNode, AggregatedNodes]): The node to add.
        """
        self.nodes[node.id] = node
        if table_name not in self.tables:
            self.tables[table_name] = [node.id]
        else:
            self.tables[table_name].append(node.id)

    def add_edge(
        self,
        node_a: SubgraphNode,
        node_b: Union[SubgraphNode, AggregatedNodes],
        rel: str,
    ) -> None:
        r"""Adds an edge between two nodes in the subgraph for
        a specific relation type.

        Args:
            node_a (SubgraphNode): The source node.
            node_b (Union[SubgraphNode, AggregatedNodes]): The
                destination node.
            rel (str): The relation type.
        """
        if node_a.id not in self.edges:
            self.edges[node_a.id] = {}
        if rel not in self.edges[node_a.id]:
            self.edges[node_a.id][rel] = []
        if node_b.id not in self.edges[node_a.id][rel]:
            self.edges[node_a.id][rel].append(node_b.id)


@dataclass
class SubgraphFeatureStore:
    r"""Stores the feature columns records for column for
    each node in the subgraph.

    Args:
        features (Dict[int, Dict[str, Feature]], optional): Dictionary
            of feature columns for each node, where key for each node
            is the column name and the value is the feature column.
    """
    features: Dict[int, Dict[str, Feature]] = field(default_factory=dict)

    def add(
        self,
        node_id: int,
        cols: List[Feature],
    ) -> None:
        r"""Adds the feature columns for a node.

        Args:
            node_id (int): The ID of the node.
            cols (List[Feature]): The list of features.
        """
        if node_id not in self.features:
            self.features[node_id] = {}
        for col in cols:
            assert col.name is not None
            self.features[node_id][col.name] = col

    def __getitem__(self, node_id: int) -> Dict[str, Feature]:
        r"""Get features for a node or a table.

        Args:
            node_id (int): The node ID.

        Returns:
            Dict[str, Feature]: The features for a node.
        """
        return self.features[node_id]


@dataclass
class Subgraph:
    r"""Represents a subgraph with the structure and feature columns.

    Args:
        subgraph_store (SubgraphStore, optional): The subgraph store, which
            mainly stores the subgraph structure.
        feature_store (FeatureStore, optional): The subgraph feature store,
            which stores the features for each node.
    """
    subgraph_store: Optional[SubgraphStore] = None
    feature_store: Optional[SubgraphFeatureStore] = None
