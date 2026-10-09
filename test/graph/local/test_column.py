# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.util.cache_seeding import seed_col_stats_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumoapi.encoder import (
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    Scaler,
)
from kumoapi.typing import Dtype, Stype

from kumo.connector.connector import DEFAULT_DATA_VERSION
from kumo.connector.pandas_connector import PandasConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.data import get_active_cache
from kumo.graph.column import Column, EncoderGenerator, ForeignKey, TimeColumn
from kumo.graph.graph import Graph
from kumo.graph.table import Table
from kumo.stats.stats import Stats

# TODO: Need to write a lot of tests after defining mock objects
#   for connector, table.

# Define mock connector inheriting a real connector so it has all methods
# but takes a fake name and path


class MockConnector(ParquetConnector):
    name: str = 'mock_connector'

    def __init__(self):
        super().__init__(name=self.name, root_dir='')

    def get_data_version(self, table: Any) -> Any:
        return DEFAULT_DATA_VERSION


class MockTable(Table):
    def __init__(self):
        super().__init__(table_name='mock_table', data_source='mock_table',
                         data_version=DEFAULT_DATA_VERSION)


def test_multicategorical_delimiter_survives_column_config_round_trip():
    col = Column(
        name='tags',
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        multicategorical_delimiter='|',
    )

    config = col.config()
    restored = Column.from_config(config)

    assert config.multicategorical_delimiter == '|'
    assert col.base_config().multicategorical_delimiter == '|'
    assert restored.multicategorical_delimiter == '|'


# helper method for manipulating the stats used in encoder inference for each
# test


def update_stats_cache(col: Column, stats_dict: Dict[ColStatType, Any]):
    get_active_cache().add_col_stats(col.table.fingerprint(),
                                     {col.base_config(): stats_dict})


def test_get_encoder_numerical():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(11, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MIN:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MAX:
        Stats(11, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEAN:
        Stats(0.5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEDIAN:
        Stats(0.5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STD:
        Stats(1, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.KURTOSIS:
        Stats(500, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.QUANTILES:
        Stats([1, 2, 3], approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='value', dtype=Dtype.float, stype=Stype.numerical,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)

    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Numerical)

    assert EncoderGenerator.get_numerical_encoder(
        col, stats_dict).scaler == Scaler.ROBUST

    stats_dict[ColStatType.MEDIAN] = Stats(0, approx=False,
                                           data_version=DEFAULT_DATA_VERSION)
    stats_dict[ColStatType.STD] = Stats(0.1, approx=False,
                                        data_version=DEFAULT_DATA_VERSION)

    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict),
                      MaxLogNumerical)

    stats_dict[ColStatType.MEDIAN] = Stats(1.0, approx=False,
                                           data_version=DEFAULT_DATA_VERSION)
    stats_dict[ColStatType.STD] = Stats(0.1, approx=False,
                                        data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict),
                      MinLogNumerical)

    stats_dict[ColStatType.STD] = Stats(None, approx=False,
                                        data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Numerical)


def test_get_encoder_categorical():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_NA:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.COUNT:
        Stats(20, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.CATEGORY_COUNTS:
        Stats([[0, 10], [1, 5], [2, 2], [3, 2], [4, 1]], approx=False,
              data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='category', dtype=Dtype.string, stype=Stype.categorical,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Index)

    # Test unique fraction is smaller than the threshold
    stats_dict['NUM_UNIQUE'] = Stats(9, approx=False,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(1000, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 100]
                                           for i in range(8)] + [[9, 200]],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Index)

    # Test unique fraction is smaller than the 20 * threshold
    stats_dict['NUM_UNIQUE'] = Stats(20, approx=False,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(1000, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 50] for i in range(20)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Hash)

    # Test unique fraction is larger than the 20 * threshold
    stats_dict['NUM_UNIQUE'] = Stats(2001, approx=False,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(102000, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 50] for i in range(2001)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Null)

    # Test float dtype with Hash condition:
    col_f = Column(name='category', dtype=Dtype.float, stype=Stype.categorical,
                   stats=stats_dict)
    col_f.table = MockTable()
    col_f.connector = MockConnector()

    stats_dict['NUM_UNIQUE'] = Stats(20, approx=False,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(1000, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 50] for i in range(20)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col_f, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col_f, stats_dict), Hash)


def test_get_encoder_hash_components():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_NA:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.COUNT:
        Stats(20, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.CATEGORY_COUNTS:
        Stats([[0, 10], [1, 5], [2, 2], [3, 2], [4, 1]], approx=False,
              data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='category', dtype=Dtype.string, stype=Stype.ID,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()

    unique = 5010
    stats_dict['NUM_UNIQUE'] = Stats(unique, approx=False,
                                     num_entries=unique * 2,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(unique * 2, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 2] for i in range(unique)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    enc = EncoderGenerator.get_encoder(col, stats_dict)
    assert isinstance(enc, Hash)
    assert enc.num_components == 3989

    unique = 50010
    stats_dict['NUM_UNIQUE'] = Stats(unique, approx=False,
                                     num_entries=unique * 2,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(unique * 2, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 2] for i in range(unique)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    enc = EncoderGenerator.get_encoder(col, stats_dict)
    assert isinstance(enc, Hash)
    assert enc.num_components == 19997

    unique = 200010
    stats_dict['NUM_UNIQUE'] = Stats(unique, approx=False,
                                     num_entries=unique * 2,
                                     data_version=DEFAULT_DATA_VERSION)
    stats_dict['COUNT'] = Stats(unique * 2, approx=False,
                                data_version=DEFAULT_DATA_VERSION)
    stats_dict['CATEGORY_COUNTS'] = Stats([[i, 2] for i in range(unique)],
                                          approx=False,
                                          data_version=DEFAULT_DATA_VERSION)
    update_stats_cache(col, stats_dict)
    enc = EncoderGenerator.get_encoder(col, stats_dict)
    assert isinstance(enc, Hash)
    assert enc.num_components == 99991


def test_get_encoder_binary():
    # Test encoder generation for single non-NA value:
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_NA:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(1, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.COUNT:
        Stats(20, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.CATEGORY_COUNTS:
        Stats([[i, 1] for i in range(20)], approx=False,
              data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='category', dtype=Dtype.string, stype=Stype.categorical,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Index)


def test_get_encoder_sequence():
    # Test encoder generation for single non-NA value:
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.SEQUENCE_MIN_LENGTH:
        Stats(128, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.SEQUENCE_MAX_LENGTH:
        Stats(128, approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='sequence', dtype=Dtype.floatlist, stype=Stype.sequence,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict),
                      NumericalList)


def test_get_encoder_multicategorical():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(11, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE_MULTI:
        Stats(3, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MULTI_CATEGORIES_SEPARATOR:
        Stats('|', approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='text', dtype=Dtype.string, stype=Stype.multicategorical,
                 multicategorical_delimiter='|', stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict),
                      MultiCategorical)


def test_get_encoder_id():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_NA:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(10, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.COUNT:
        Stats(20, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.CATEGORY_COUNTS:
        Stats([[i, 2] for i in range(10)], approx=False,
              data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='id', dtype=Dtype.string, stype=Stype.ID,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Index)

    # Test num unique > 0.8 * num rows (10000)
    stats_dict[ColStatType.NUM_UNIQUE] = Stats(
        7812, approx=False, data_version=DEFAULT_DATA_VERSION,
        num_entries=10000)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Hash)

    stats_dict[ColStatType.NUM_UNIQUE] = Stats(
        8012, approx=False, data_version=DEFAULT_DATA_VERSION,
        num_entries=10000)
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Null)


def test_get_encoder_text():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(11, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STRING_GLOVE_OVERLAP:
        Stats(0.5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STRING_MAX_TOKENS:
        Stats(10, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STRING_AVG_TOKENS:
        Stats(4.3, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STRING_AVG_NON_CHAR:
        Stats(0.4, approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='text', dtype=Dtype.string, stype=Stype.text,
                 stats=stats_dict)

    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)

    assert EncoderGenerator.get_encoder(col, stats_dict) == Null()


def test_invalid_col():
    with pytest.raises(ValueError, match="has no data type"):
        Column("x").validate()

    with pytest.raises(ValueError, match="has no data type"):
        Column("x", stype=Stype.categorical).validate()

    with pytest.raises(ValueError, match="has no semantic type"):
        Column("x", dtype=Dtype.int).validate()


def test_invalid_time_col():
    with pytest.raises(ValueError,
                       match="Must be either 'time' or 'date'"):
        TimeColumn("x", Dtype.int)

    assert TimeColumn("x", Dtype.time)


def test_foreign_key_fully_qualified_name():
    fkey = ForeignKey(name="fkey", dst_table="table1", dst_key="key1",
                      dst_data_source="source1")
    assert fkey.dst_table == "source1.table1"
    fkey = ForeignKey(name="fkey", dst_table="source1.table1", dst_key="key1")
    assert fkey.dst_table == "source1.table1"


def test_encoder_for_none_stats():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MIN:
        Stats(0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MAX:
        Stats(1, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEAN:
        Stats(0.5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEDIAN:
        Stats(0.5, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STD:
        Stats(1, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.KURTOSIS:
        Stats(500, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.QUANTILES:
        Stats([1, 2, 3], approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='value', dtype=Dtype.float, stype=Stype.numerical,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Numerical)


def test_encoder_for_missing_col():
    stats_dict: Dict[ColStatType, Any] = {
        ColStatType.NA_FRACTION:
        Stats(1.0, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.NUM_UNIQUE:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MIN:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MAX:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEAN:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.MEDIAN:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.STD:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.KURTOSIS:
        Stats(None, approx=False, data_version=DEFAULT_DATA_VERSION),
        ColStatType.QUANTILES:
        Stats([], approx=False, data_version=DEFAULT_DATA_VERSION),
    }

    col = Column(name='value', dtype=Dtype.float, stype=Stype.numerical,
                 stats=stats_dict)
    col.table = MockTable()
    col.connector = MockConnector()
    update_stats_cache(col, stats_dict)
    assert isinstance(EncoderGenerator.get_encoder(col, stats_dict), Null)


def test_get_max_log_encoder():
    df = pd.DataFrame({'col': np.exp(np.arange(10))})
    connector = PandasConnector({'table1': df})
    col = Column(name='col', stype=Stype.numerical, dtype=Dtype.float)

    graph = Graph("test1")
    graph.connect(connector)
    graph.add('table1', connector.name, cols=[col])
    seed_col_stats_cache(
        df, table_name='table1', connector_name=connector.name,
        data_version=connector.get_data_version('table1'))

    ret = EncoderGenerator.get_encoder(
        col, graph['table1'].calculate_stats_for_column(col))
    assert isinstance(ret, MaxLogNumerical)


def test_get_min_log_encoder():
    df = pd.DataFrame({'col': -np.exp(np.arange(10))})
    connector = PandasConnector({'table2': df})
    col = Column(name='col', stype=Stype.numerical, dtype=Dtype.float)

    graph = Graph("test1")
    graph.connect(connector)
    graph.add('table2', connector.name, cols=[col])
    seed_col_stats_cache(
        df, table_name='table2', connector_name=connector.name,
        data_version=connector.get_data_version('table2'))

    ret = EncoderGenerator.get_encoder(
        col, graph['table2'].calculate_stats_for_column(col))
    assert isinstance(ret, MinLogNumerical)


def test_foreign_key_column_stats():
    fk = ForeignKey(
        name='article_id',
        dtype=Dtype.int64,
        dst_table='hm.articles',
        dst_key='article_id',
        dst_data_source='hm',
        stats={
            ColStatType.NUM_UNIQUE:
            Stats(
                value=1231231,
                approx=False,
                data_version="1",
                num_entries=1231231,
            )
        },
    )
    assert ForeignKey.from_config(fk.config()) == fk
