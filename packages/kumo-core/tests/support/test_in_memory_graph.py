# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pandas as pd
import pytest
import torch

from kumo_core.features.stores.graph import GraphStoreAttr
from tests.support.in_memory_graph import InMemoryGraphStore


def test_in_memory_graph_store() -> None:
    df_dict = {}
    df_dict['USER'] = pd.DataFrame({'USER_ID': ['user_0', 'user_1']})
    df_dict['ITEM'] = pd.DataFrame({'ITEM_ID': ['item_0', 'item_1', 'item_2']})
    df_dict['FACT'] = pd.DataFrame({
        'USER_ID': ['user_0', 'user_1', 'user_0', 'user_1', 'user_1'],
        'ITEM_ID': ['item_0', 'item_2', 'item_0', 'item_1', 'item_3'],
        'TIME': ['1991', '1990', '1990', '1992', '1991'],
    })

    graph_store = InMemoryGraphStore(
        df_dict=df_dict,
        fkeys=[
            ('FACT.USER_ID', 'USER.USER_ID'),
            ('FACT.ITEM_ID', 'ITEM.ITEM_ID'),
        ],
        time_col_dict={'FACT': 'TIME'},
    )
    assert str(graph_store) == 'InMemoryGraphStore(num_edge_types=4)'

    attrs = graph_store.get_all_edge_attrs()
    assert len(attrs) == 4
    assert attrs[0] == GraphStoreAttr(('FACT', 'USER_ID', 'USER'), (5, 2))
    assert attrs[1] == GraphStoreAttr(('USER', 'rev_USER_ID', 'FACT'), (2, 5))
    assert attrs[2] == GraphStoreAttr(('FACT', 'ITEM_ID', 'ITEM'), (5, 3))
    assert attrs[3] == GraphStoreAttr(('ITEM', 'rev_ITEM_ID', 'FACT'), (3, 5))
    assert str(attrs[0]) == ("GraphStoreAttr("
                             "edge_type=('FACT', 'USER_ID', 'USER'), "
                             "size=(5, 2))")

    edge_types = graph_store.edge_types
    assert len(edge_types) == 4
    assert edge_types[0] == ('FACT', 'USER_ID', 'USER')
    assert edge_types[1] == ('USER', 'rev_USER_ID', 'FACT')
    assert edge_types[2] == ('FACT', 'ITEM_ID', 'ITEM')
    assert edge_types[3] == ('ITEM', 'rev_ITEM_ID', 'FACT')

    assert graph_store.get_foreign_keys('USER') == []
    assert graph_store.get_foreign_keys('FACT') == [
        ('USER_ID', 'USER'),
        ('ITEM_ID', 'ITEM'),
    ]
    assert graph_store.get_foreign_keys('ITEM') == []

    row, colptr = graph_store[('FACT', 'USER_ID', 'USER')]
    assert colptr.equal(torch.tensor([0, 2, 5]))
    assert row.equal(torch.tensor([2, 0, 1, 4, 3]))

    row, colptr = graph_store[('USER', 'rev_USER_ID', 'FACT')]
    assert colptr.equal(torch.tensor([0, 1, 2, 3, 4, 5]))
    assert row.equal(torch.tensor([0, 1, 0, 1, 1]))

    row, colptr = graph_store[('FACT', 'ITEM_ID', 'ITEM')]
    assert colptr.equal(torch.tensor([0, 2, 3, 4]))
    assert row.equal(torch.tensor([2, 0, 3, 1]))

    row, colptr = graph_store[('ITEM', 'rev_ITEM_ID', 'FACT')]
    assert colptr.equal(torch.tensor([0, 1, 2, 3, 4, 4]))
    assert row.equal(torch.tensor([0, 2, 0, 1]))


@pytest.mark.parametrize('src_dtype', ['Int64', float, str])
@pytest.mark.parametrize('dst_dtype', ['Int64', float, str])
def test_dtype(
    src_dtype: Literal['int64'] | type[float] | type[str],
    dst_dtype: Literal['int64'] | type[float] | type[str],
) -> None:
    df_dict = {}
    df_dict['SRC'] = pd.DataFrame({
        'SRC_ID':
        pd.Series([10, 20, None, 10, 20, 30, 20], dtype=src_dtype),
    })
    df_dict['DST'] = pd.DataFrame({
        'DST_ID':
        pd.Series([30, 20, 10], dtype=dst_dtype),
    })

    if ((src_dtype == str and dst_dtype != str)
            or (src_dtype != str and dst_dtype == str)):
        with pytest.raises(RuntimeError, match=r"incompatible data type"):
            InMemoryGraphStore(
                df_dict=df_dict,
                fkeys=[('SRC.SRC_ID', 'DST.DST_ID')],
                time_col_dict={},
            )
        return

    graph_store = InMemoryGraphStore(
        df_dict=df_dict,
        fkeys=[('SRC.SRC_ID', 'DST.DST_ID')],
        time_col_dict={},
    )

    row, colptr = graph_store._csc_dict[('SRC', 'SRC_ID', 'DST')]
    assert row.equal(torch.tensor([5, 1, 4, 6, 0, 3]))
    assert colptr.equal(torch.tensor([0, 1, 4, 6]))

    row, colptr = graph_store._csc_dict[('DST', 'rev_SRC_ID', 'SRC')]
    assert row.equal(torch.tensor([2, 1, 2, 1, 0, 1]))
    assert colptr.equal(torch.tensor([0, 1, 2, 2, 3, 4, 5, 6]))
