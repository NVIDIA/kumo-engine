# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.typing import EdgeType
from torch_geometric.utils.sparse import index2ptr

from kumo_core.features.stores.graph import GraphStore, GraphStoreAttr
from kumo_core.features.stores.graph import REV
from kumo_core.encoders.index_mapping import Mapper
from kumo_core.features.time import to_datetime


class InMemoryGraphStore(GraphStore):
    r"""Test-only graph fixture; not part of the production runtime.

    Args:
        df_dict: A dictionary of raw data frames.
        fkeys: The foreign keys connecting different data frames. A foreign key
            is given as a tuple, describing source and destination tables in
            dot syntax, *i.e.* ``("FACT.FKEY_NAME", "ENTITY.PKEY_NAME")``.
        time_col_dict: A dictionary holding the time column names of temporal
            tables.
    """
    def __init__(
        self,
        df_dict: dict[str, pd.DataFrame],
        fkeys: list[tuple[str, str]],
        time_col_dict: dict[str, str],
    ) -> None:
        super().__init__()

        self._fkeys = fkeys

        index_maps: dict[str, Mapper] = {}
        for pkey in {pkey for (fkey, pkey) in fkeys}:
            table_name, col_name = pkey.rsplit('.', maxsplit=1)
            index_maps[table_name] = Mapper(df_dict[table_name][col_name])

        self._csc_dict: dict[EdgeType, tuple[Tensor, Tensor]] = {}
        for fkey, pkey in fkeys:
            src_table_name, src_col_name = fkey.rsplit('.', maxsplit=1)
            dst_table_name, _ = pkey.rsplit('.', maxsplit=1)

            src_df = df_dict[src_table_name]
            dst_df = df_dict[dst_table_name]

            src = torch.arange(len(src_df))
            dst_index_map = index_maps[dst_table_name]
            dst = dst_index_map.map(src_df[src_col_name], strict=True)
            mask = dst >= 0
            src, dst = src[mask], dst[mask]

            src_node_time: Tensor | None = None
            if time_col_name := time_col_dict.get(src_table_name):
                ser = df_dict[src_table_name][time_col_name]
                ser = to_datetime(ser)
                src_node_time = torch.from_numpy(ser.astype(int).to_numpy())

            # Stable Torch sorting avoids optional PyG native sorting kernels.
            order = torch.arange(src.numel())
            if src_node_time is not None:
                order = order[src_node_time[src].argsort(stable=True)]
            order = order[dst[order].argsort(stable=True)]
            row, col = src[order], dst[order]
            colptr = index2ptr(col, size=len(dst_df))
            edge_type = (src_table_name, src_col_name, dst_table_name)
            self._csc_dict[edge_type] = (row, colptr)

            row, col = dst, src  # NOTE No sort and no time handling needed.
            colptr = index2ptr(col, size=len(src_df))
            edge_type = (dst_table_name, f'{REV}{src_col_name}',
                         src_table_name)
            self._csc_dict[edge_type] = (row, colptr)

    def _get_edge_index(self, attr: GraphStoreAttr) -> tuple[Tensor, Tensor]:
        return self._csc_dict[attr.edge_type]

    def get_all_edge_attrs(self) -> list[GraphStoreAttr]:
        attrs: list[GraphStoreAttr] = []

        for fkey, pkey in self._fkeys:
            src_table_name, rel = fkey.rsplit('.', maxsplit=1)
            dst_table_name, _ = pkey.rsplit('.', maxsplit=1)

            edge_type = (src_table_name, rel, dst_table_name)
            rev_edge_type = (dst_table_name, f'{REV}{rel}', src_table_name)

            attr = GraphStoreAttr(
                edge_type, size=(
                    self._csc_dict[rev_edge_type][1].size(0) - 1,
                    self._csc_dict[edge_type][1].size(0) - 1,
                ))
            attrs.append(attr)

            attr = GraphStoreAttr(
                rev_edge_type, size=(
                    self._csc_dict[edge_type][1].size(0) - 1,
                    self._csc_dict[rev_edge_type][1].size(0) - 1,
                ))
            attrs.append(attr)

        return attrs
