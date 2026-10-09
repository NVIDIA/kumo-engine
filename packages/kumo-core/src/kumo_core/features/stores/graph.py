# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch_geometric
from torch import Tensor
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.typing import EdgeType
from torch_geometric.utils.mixin import CastMixin

REV = 'rev_'


@dataclass(init=False, repr=False)
class GraphStoreAttr(CastMixin):
    edge_type: EdgeType
    size: tuple[int, int] | None = None

    # Only allow sorted CSC edges:
    layout: EdgeLayout = EdgeLayout.CSC
    is_sorted: bool = True

    def __init__(
        self,
        edge_type: EdgeType,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.edge_type = edge_type
        self.size = size

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'edge_type={self.edge_type}, '
                f'size={self.size}'
                f')')


class GraphStore(torch_geometric.data.GraphStore, ABC):  # type: ignore
    @abstractmethod
    def _get_edge_index(self, attr: GraphStoreAttr) -> tuple[Tensor, Tensor]:
        r"""Synchronously obtains a CSC edge tensor from the graph store.

        .. note::
            This function may not be implemented by all graph store
            implementations.
        """

    @abstractmethod
    def get_all_edge_attrs(self) -> list[GraphStoreAttr]:
        r"""Returns all edge attributes."""

    # Implementation ##########################################################

    def __init__(self) -> None:
        super().__init__(edge_attr_cls=GraphStoreAttr)

    def get_edge_index(
        self,
        attr: GraphStoreAttr | EdgeType,
    ) -> tuple[Tensor, Tensor]:
        r"""Synchronously obtains a CSC edge tensor from the graph store."""
        if not isinstance(attr, GraphStoreAttr):
            attr = GraphStoreAttr(attr)
        return self._get_edge_index(attr)

    @property
    def edge_types(self) -> list[EdgeType]:
        r"""Returns all edge types registered to the graph store."""
        return [attr.edge_type for attr in self.get_all_edge_attrs()]

    def get_foreign_keys(self, table_name: str) -> list[tuple[str, str]]:
        r"""Returns the foreign keys as a list of
        ``(column_name, destination_table)`` pairs for a specific table
        ``table_name`` registered in the graph store.
        """
        fkeys = []
        for (src, rel, dst) in self.edge_types:
            if src == table_name and not rel.startswith(REV):
                fkeys.append((rel, dst))
        return fkeys

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_edge_types={len(self.edge_types)})')

    # Not supported ###########################################################

    def coo(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def csr(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _put_edge_index(
        self,
        edge_index: tuple[Tensor, Tensor],
        attr: GraphStoreAttr,
    ) -> bool:
        raise NotImplementedError

    def _remove_edge_index(self, attr: GraphStoreAttr) -> bool:
        raise NotImplementedError
