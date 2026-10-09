# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.types import ColStatType, TaskType
from kumo_core.types.conversions import (
    dtype_to_numpy,
    dtype_to_pandas,
    dtype_to_torch,
)
from kumo_core.types.edges import EdgeKey
from kumoapi.typing import Dtype, Stype

from statedb.config.typing import DTYPE_TO_PANDAS, to_dtype


def test_dtype():
    assert Dtype('bool') == Dtype.bool
    assert Dtype('int') == Dtype.int
    assert Dtype('byte') == Dtype.byte
    assert Dtype('int32') == Dtype.int32
    assert Dtype('int64') == Dtype.int64
    assert Dtype('float') == Dtype.float
    assert Dtype('float32') == Dtype.float32
    assert Dtype('float64') == Dtype.float64
    assert Dtype('string') == Dtype.string
    assert Dtype('date') == Dtype.date
    assert Dtype('time') == Dtype.time
    assert Dtype('timedelta') == Dtype.timedelta
    assert Dtype('floatlist') == Dtype.floatlist
    assert Dtype('intlist') == Dtype.intlist
    assert Dtype('stringlist') == Dtype.stringlist

    # `str` is the bare value: `Dtype`/`Stype` are now kumoapi's, and
    # kumoapi never prefixed. Only `kumo/data/cache/exact_stats.py`
    # read the old `kumo.` prefix, and it strips an optional one.
    assert str(Dtype.bool) == 'bool'
    assert str(Dtype.int) == 'int'
    assert str(Dtype.byte) == 'byte'
    assert str(Dtype.int32) == 'int32'
    assert str(Dtype.int64) == 'int64'
    assert str(Dtype.float) == 'float'
    assert str(Dtype.float32) == 'float32'
    assert str(Dtype.float64) == 'float64'
    assert str(Dtype.string) == 'string'
    assert str(Dtype.date) == 'date'
    assert str(Dtype.time) == 'time'
    assert str(Dtype.timedelta) == 'timedelta'
    assert str(Dtype.floatlist) == 'floatlist'
    assert str(Dtype.intlist) == 'intlist'
    assert str(Dtype.stringlist) == 'stringlist'

    assert Dtype.bool.title_case() == 'Bool'
    assert Dtype.int.title_case() == 'Int'
    assert Dtype.byte.title_case() == 'Byte'
    assert Dtype.int32.title_case() == 'Int32'
    assert Dtype.int64.title_case() == 'Int64'
    assert Dtype.float.title_case() == 'Float'
    assert Dtype.float32.title_case() == 'Float32'
    assert Dtype.float64.title_case() == 'Float64'
    assert Dtype.string.title_case() == 'String'
    assert Dtype.date.title_case() == 'Date'
    assert Dtype.time.title_case() == 'Time'
    assert Dtype.timedelta.title_case() == 'Timedelta'
    assert Dtype.floatlist.title_case() == 'FloatList'
    assert Dtype.intlist.title_case() == 'IntList'
    assert Dtype.stringlist.title_case() == 'StringList'

    assert dtype_to_numpy(Dtype.bool) == bool
    assert dtype_to_numpy(Dtype.int) == np.int64
    assert dtype_to_numpy(Dtype.byte) == np.byte
    assert dtype_to_numpy(Dtype.int32) == np.int32
    assert dtype_to_numpy(Dtype.int64) == np.int64
    assert dtype_to_numpy(Dtype.float) == np.float32
    assert dtype_to_numpy(Dtype.float32) == np.float32
    assert dtype_to_numpy(Dtype.float64) == np.float64
    assert dtype_to_numpy(Dtype.string) == object
    assert dtype_to_numpy(Dtype.date) == np.dtype('datetime64[ns]')
    assert dtype_to_numpy(Dtype.time) == np.dtype('datetime64[ns]')
    assert dtype_to_numpy(Dtype.timedelta) == np.dtype('timedelta64[ns]')
    assert dtype_to_numpy(Dtype.floatlist) == object
    assert dtype_to_numpy(Dtype.intlist) == object
    assert dtype_to_numpy(Dtype.stringlist) == object

    assert dtype_to_pandas(Dtype.bool) == pd.BooleanDtype()
    assert dtype_to_pandas(Dtype.int) == pd.Int64Dtype()
    assert dtype_to_pandas(Dtype.byte) == pd.UInt8Dtype()
    assert dtype_to_pandas(Dtype.int32) == pd.Int32Dtype()
    assert dtype_to_pandas(Dtype.int64) == pd.Int64Dtype()
    assert dtype_to_pandas(Dtype.float) == np.float32
    assert dtype_to_pandas(Dtype.float32) == np.float32
    assert dtype_to_pandas(Dtype.float64) == np.float64
    assert dtype_to_pandas(Dtype.string) == DTYPE_TO_PANDAS[Dtype.string]
    assert dtype_to_pandas(Dtype.date) == np.dtype('datetime64[ns]')
    assert dtype_to_pandas(Dtype.time) == np.dtype('datetime64[ns]')
    assert dtype_to_pandas(Dtype.timedelta) == np.dtype('timedelta64[ns]')
    assert dtype_to_pandas(Dtype.floatlist) == np.dtype('object')
    assert dtype_to_pandas(Dtype.intlist) == np.dtype('object')
    assert dtype_to_pandas(Dtype.stringlist) == np.dtype('object')

    assert dtype_to_torch(Dtype.bool) == torch.bool
    assert dtype_to_torch(Dtype.int) == torch.int64
    assert dtype_to_torch(Dtype.byte) == torch.uint8
    assert dtype_to_torch(Dtype.int32) == torch.int32
    assert dtype_to_torch(Dtype.int64) == torch.int64
    assert dtype_to_torch(Dtype.float) == torch.float32
    assert dtype_to_torch(Dtype.float32) == torch.float32
    assert dtype_to_torch(Dtype.float64) == torch.float64
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.string)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.date)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.time)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.timedelta)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.floatlist)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.intlist)
    with pytest.raises(KeyError):
        dtype_to_torch(Dtype.stringlist)

    assert Dtype.bool.is_bool()
    assert Dtype.int.is_int()
    assert Dtype.byte.is_int()
    assert Dtype.int32.is_int()
    assert Dtype.int64.is_int()
    assert Dtype.float.is_float()
    assert Dtype.float32.is_float()
    assert Dtype.float64.is_float()
    assert Dtype.string.is_string()
    assert Dtype.date.is_temporal()
    assert Dtype.time.is_temporal()
    assert not Dtype.timedelta.is_temporal()
    assert Dtype.timedelta.is_numerical()
    assert Dtype.date.is_timestamp()
    assert Dtype.time.is_timestamp()
    assert Dtype.floatlist.is_list()
    assert Dtype.intlist.is_list()
    assert Dtype.stringlist.is_list()

    assert Dtype.floatlist.list_element_type() == Dtype.float
    assert Dtype.intlist.list_element_type() == Dtype.int
    assert Dtype.stringlist.list_element_type() == Dtype.string


def test_to_dtype_object_datetime_series():
    # Snowflake object-dtype timestamps must type temporal so MIN/MAX survive.
    timestamp_ser = pd.Series(
        [pd.Timestamp("2020-01-01"),
         pd.Timestamp("2021-06-15")], dtype=object)
    assert timestamp_ser.dtype == object
    assert to_dtype(timestamp_ser.dtype, ser=timestamp_ser) == Dtype.date

    datetime_ser = pd.Series(
        [datetime.datetime(2020, 1, 1),
         datetime.datetime(2021, 6, 15)], dtype=object)
    assert to_dtype(datetime_ser.dtype, ser=datetime_ser) == Dtype.date

    date_ser = pd.Series(
        [datetime.date(2020, 1, 1),
         datetime.date(2021, 6, 15)], dtype=object)
    assert to_dtype(date_ser.dtype, ser=date_ser) == Dtype.date

    # np.datetime64 scalars are not a datetime.date subclass; infer_dtype must
    # still recognize them as temporal.
    np_dt64_ser = pd.Series(
        [np.datetime64("2020-01-01"),
         np.datetime64("2021-06-15")], dtype=object)
    assert np_dt64_ser.dtype == object
    assert to_dtype(np_dt64_ser.dtype, ser=np_dt64_ser) == Dtype.date

    # Out-of-range timestamps must still be recognized as temporal.
    out_of_range_ser = pd.Series(
        [datetime.datetime(3122, 5, 1),
         datetime.datetime(2020, 1, 1)], dtype=object)
    assert to_dtype(out_of_range_ser.dtype, ser=out_of_range_ser) == Dtype.date

    # Leading nulls must not confuse the inference.
    leading_null_ser = pd.Series([None, pd.Timestamp("2020-01-01")],
                                 dtype=object)
    assert to_dtype(leading_null_ser.dtype, ser=leading_null_ser) == Dtype.date

    # MIN/MAX must be supported for the inferred temporal dtype.
    assert ColStatType.MIN.supports(Dtype.date)
    assert ColStatType.MAX.supports(Dtype.date)

    # Regression guard: ordinary object series remain Dtype.string.
    string_ser = pd.Series(["a", "b", "c"], dtype=object)
    assert to_dtype(string_ser.dtype, ser=string_ser) == Dtype.string

    # Regression guard: numeric-object series must not be typed temporal.
    numeric_ser = pd.Series([1, 2, 3], dtype=object)
    assert to_dtype(numeric_ser.dtype, ser=numeric_ser) == Dtype.string

    # Regression guard: object list series remain list dtypes.
    list_ser = pd.Series([[1.0, 2.0], [3.0]], dtype=object)
    assert to_dtype(list_ser.dtype, ser=list_ser) == Dtype.floatlist
    int_list_ser = pd.Series([[1, 2], [3]], dtype=object)
    assert to_dtype(int_list_ser.dtype, ser=int_list_ser) == Dtype.intlist
    str_list_ser = pd.Series([["a", "b"], ["c"]], dtype=object)
    assert to_dtype(str_list_ser.dtype, ser=str_list_ser) == Dtype.stringlist


def test_stype():
    assert Stype('numerical') == Stype.numerical
    assert Stype('categorical') == Stype.categorical
    assert Stype('multicategorical') == Stype.multicategorical
    assert Stype('ID') == Stype.ID
    assert Stype('text') == Stype.text
    assert Stype('timestamp') == Stype.timestamp
    assert Stype('sequence') == Stype.sequence

    # `str` is the bare value: `Dtype`/`Stype` are now kumoapi's, and
    # kumoapi never prefixed. Only `kumo/data/cache/exact_stats.py`
    # read the old `kumo.` prefix, and it strips an optional one.
    assert str(Stype.numerical) == 'numerical'
    assert str(Stype.categorical) == 'categorical'
    assert str(Stype.multicategorical) == 'multicategorical'
    assert str(Stype.ID) == 'ID'
    assert str(Stype.text) == 'text'
    assert str(Stype.timestamp) == 'timestamp'
    assert str(Stype.sequence) == 'sequence'

    assert Stype.numerical.title_case() == 'Numerical'
    assert Stype.categorical.title_case() == 'Categorical'
    assert Stype.multicategorical.title_case() == 'Multicategorical'
    assert Stype.ID.title_case() == 'ID'
    assert Stype.text.title_case() == 'Text'
    assert Stype.timestamp.title_case() == 'Timestamp'
    assert Stype.sequence.title_case() == 'Sequence'


def test_deprecated_task_type():
    task_type = TaskType('classification')
    assert task_type == TaskType.MULTICLASS_CLASSIFICATION

    task_type = TaskType('multilabel')
    assert task_type == TaskType.MULTILABEL_CLASSIFICATION

    with pytest.raises(ValueError):
        task_type = TaskType('test')


def test_edge_key():
    edge_key = EdgeKey(src_table='table1', link_key='col', dst_table='table2')
    assert edge_key.to_str() == "table1__col__table2"
    assert edge_key.to_tuple() == ('table1', 'col', 'table2')
    rev_edge_key = edge_key.reverse()
    assert rev_edge_key.to_str() == "table2__rev_col__table1"
    assert rev_edge_key.to_tuple() == ('table2', 'rev_col', 'table1')
    assert rev_edge_key == EdgeKey(
        src_table='table2',
        link_key='col',
        dst_table='table1',
        is_reverse=True,
    )
    # Reverse of a reverse edge should be equal to the original edge.
    assert rev_edge_key.reverse() == edge_key

    # Creating a reverse edge key with the 'rev_' prefix should fail:
    with pytest.raises(ValueError):
        EdgeKey('table2', 'rev_col', 'table1', is_reverse=True)
