# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# type: ignore

import os
import pathlib
from os import path as osp

import pandas as pd
import pytest
import torch
from kumoapi.train import TrainingStage
from kumoapi.typing import Dtype

from io_utils import json
from kumo.config.task import TimePeriod
from kumo.data.utils import ENTITY_INCLUDE_BIT
from kumo.graph.column import (
    BaseColumn,
    BoolColumn,
    ForeignKey,
    PrimaryKey,
    TimeColumn,
)
from kumo.graph.graph import Graph
from kumo.graph.pandas_view import PandasView
from kumo.graph.table import Table
from kumo.graph.table_data import (
    InMemoryDataAttributes,
    InMemoryDataAttributesConfig,
    TableData,
    TableDataAttributes,
)
from kumo.graph.utils import get_fkey_data

# Unit Tests ##################################################################


@pytest.mark.parametrize('is_pseudo', [True, False])
def test_attribute_serialization(tmp_path: pathlib.Path, is_pseudo: bool):
    tmp_path_ = tmp_path.resolve().as_posix()

    # Setup:
    table = Table(table_name='test_table', data_source='test_ds')
    table.set_pkey(PrimaryKey("pkey"))
    table.add_fkey(ForeignKey("fkey", dst_table="x", dst_key="y"))

    in_mem_attrs = InMemoryDataAttributes()
    in_mem_attrs.time_col = TimeColumn(name='time')
    in_mem_attrs.weight_col = BaseColumn(name='weight', dtype=Dtype.int)
    in_mem_attrs.entity_include_bit_col = BoolColumn(name=ENTITY_INCLUDE_BIT)
    if is_pseudo:
        time_periods = dict(
            {TrainingStage.VAL: TimePeriod.cast(0, 1, is_pseudo=True)})
    else:
        time_periods = dict(
            {TrainingStage.VAL: TimePeriod.cast('2000-01-01', '2000-01-02')})

    attrs = TableDataAttributes(
        name='test',
        num_nodes=1000,
        data_version='-1',
        in_memory_attrs=in_mem_attrs.config(
            data_cls=InMemoryDataAttributesConfig),
        table_metadata_config=table.metadata_config(),
        time_periods=time_periods,
    )

    path = osp.join(tmp_path_, 'ser.json')
    json.save_json(attrs, path)
    attrs_deserialized = json.load_json(path, TableDataAttributes)

    # Compare in-memory attributes:
    assert in_mem_attrs == InMemoryDataAttributes.from_config(
        attrs_deserialized.in_memory_attrs)

    # Compare other attributes:
    attrs.in_memory_attrs = None
    attrs_deserialized.in_memory_attrs = None
    assert attrs == attrs_deserialized


def test_non_unique_index_error():
    class MockConnector:
        def read_table(*args, **kwargs):
            df = pd.DataFrame([1, 1, 1], columns=['non_unique_idx'])
            df.set_index('non_unique_idx', inplace=True)
            return df

    pkey = PrimaryKey('non_unique_idx')
    pd_view = PandasView(MockConnector(), 'TABLE', cols=[], pkey=pkey)

    with pytest.raises(ValueError, match="has more than 50% duplicated"):
        pd_view()


def test_non_unique_index_drop():
    class MockConnector:
        def read_table(*args, **kwargs):
            df = pd.DataFrame([1, 2, 1], columns=['non_unique_idx'])
            df.set_index('non_unique_idx', inplace=True)
            return df

    pkey = PrimaryKey('non_unique_idx', drop_duplicates=True)
    pd_view = PandasView(MockConnector(), 'TABLE', cols=[], pkey=pkey)

    assert pd_view().shape == (2, 0)


# E2E Tests ###################################################################


def test_table_data_memory(create_graph, tmp_path: pathlib.Path):
    graph: Graph = create_graph(name='FINANCIAL', parquet=True, infer=True,
                                temporal=True)

    graph['LOAN', 'DATE'].metadata = {'TS_FORMAT': 'yyyyMMdd'}
    graph['ACCOUNT', 'DATE'].metadata = {'TS_FORMAT': 'yyyyMMdd'}
    table_name = 'LOAN'
    table_data = TableData(graph[table_name])

    # Basic attributes & properties:
    assert table_data.persist_dir is None
    assert hasattr(table_data, 'data')

    assert str(table_data) == (f'TableData(\n'
                               f'  name=parquet.LOAN,\n'
                               f'  data_version={table_data.data_version},\n'
                               f'  persist_dir=MEMORY,\n'
                               f'  num_nodes=682\n'
                               f')')

    # Index map:
    assert len(table_data.index_map) == table_data.num_nodes

    # Time:
    assert isinstance(table_data.time, torch.Tensor)
    # ns-canonical: the time column is int64 NANOSECONDS regardless of the source
    # datetime resolution (pandas-2 ``astype(int)`` would yield seconds for a
    # ``datetime64[s]`` source). FINANCIAL dates are 1990s -> ns since epoch are
    # ~7e17, far beyond any seconds (~7e8) magnitude; assert the ns scale.
    _finite = table_data.time[table_data.time != torch.iinfo(
        table_data.time.dtype).max]
    assert int(_finite.abs().max()) > 10**17, (
        "time column is not nanoseconds (resolution bug)")

    # Save:
    save_out_dir = osp.join(tmp_path.resolve().as_posix(), '_saved_data')
    os.makedirs(save_out_dir, exist_ok=True)
    table_data.save_data(save_out_dir)

    assert set(os.listdir(save_out_dir)) == {
        'attributes.json',
    }

    # Load:
    table_data_2 = TableData.__new__(TableData)
    table_data_2.load_data(
        save_out_dir,
        pkey=graph[table_name].pkey,
        fkeys=graph[table_name].fkeys,
    )

    # TODO(manan): better checks
    assert str(table_data_2) == str(table_data)

    # Ensure lazy loading works as intended:
    assert not hasattr(table_data_2, '_data')

    # Special case: loading time tensor does not also load data
    assert isinstance(table_data_2.time, torch.Tensor)
    assert not hasattr(table_data_2, '_data')

    assert isinstance(table_data_2.data, pd.DataFrame)
    assert len(table_data_2._fkey_data) > 0
    assert isinstance(get_fkey_data(list(table_data_2._fkey_data.values())[0]),
                      pd.Series)

    assert len(table_data_2.index_map) == table_data_2.num_nodes
