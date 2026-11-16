# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Union

import pandas as pd
import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Constant
from kumoapi.typing import Dtype


@pytest.mark.parametrize('const, dtype', [
    ('abc', Dtype.int),
    ('abc', Dtype.time),
    ('2020-99-99', Dtype.time),
    ('(}rubbish{)', Dtype.intlist),
    ('(}rubbish{)', Dtype.floatlist),
    ('(}rubbish{)', Dtype.stringlist),
])
def test_invalid_constant(const: str, dtype: Dtype):
    with pytest.raises(TypeError, match='not a valid instance of'):
        Constant(value=const, dtype_maybe=dtype)


@pytest.mark.parametrize('value, dtype', [
    ('0', Dtype.int),
    ('0', Dtype.float),
    ('1e-4', Dtype.float),
    ('True', Dtype.bool),
    ('FALSE', Dtype.bool),
    ('2020-01-01', Dtype.time),
    ('2020-01-01 20:11', Dtype.time),
    ('"2020-01-01"', Dtype.string),
    ([
        Constant(value='1', dtype_maybe=Dtype.int),
        Constant(value='2', dtype_maybe=Dtype.int),
        Constant(value='3', dtype_maybe=Dtype.int)
    ], Dtype.intlist),
    ([
        Constant(value='1.0', dtype_maybe=Dtype.float),
        Constant(value='2', dtype_maybe=Dtype.int),
        Constant(value='3.0', dtype_maybe=Dtype.float)
    ], Dtype.stringlist),
    ([
        Constant(value='"a"', dtype_maybe=Dtype.string),
        Constant(value='"b"', dtype_maybe=Dtype.string)
    ], Dtype.stringlist),
    ('NULL', None),
    (None, None),
])
def test_column_serialization(value: Union[str, List[Constant]],
                              dtype: Optional[Dtype]):
    const = Constant(value=value, dtype_maybe=dtype)
    cfg = to_json(const)
    assert const == from_json(cfg, Constant)


@pytest.mark.parametrize('value, exp_dtype, exp_str', [
    (0, Dtype.int, '0'),
    (0.0, Dtype.float, '0.0'),
    (1e-4, Dtype.float, '0.0001'),
    (True, Dtype.bool, 'True'),
    ('kumo_timestamp_2020-01-01', Dtype.time, '2020-01-01 00:00:00'),
    (pd.Timestamp('2020-01-01 20:11'), Dtype.time, '2020-01-01 20:11:00'),
    ('2020-01-01', Dtype.string, '\"2020-01-01\"'),
    ([1, 2, 3], Dtype.intlist, '(1, 2, 3)'),
    ([1.0, 2, 3.0], Dtype.floatlist, '(1.0, 2, 3.0)'),
    (["a", "b"], Dtype.stringlist, '("a", "b")'),
    (None, None, 'NULL'),
])
def test_from_value(value: Any, exp_dtype: Optional[Dtype], exp_str: str):
    const = Constant.from_value(value)
    cfg = to_json(const)
    assert const == from_json(cfg, Constant)
    assert const.dtype_maybe == exp_dtype
    assert const.to_string() == exp_str
