# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_snowflake.types import dtype_for, dtype_for_described
from kumoapi.typing import Dtype


@pytest.mark.parametrize('name,scale,expected', [
    ('FIXED', 0, Dtype.int64),
    ('FIXED', 2, Dtype.float64),
    ('REAL', None, Dtype.float64),
    ('TIMESTAMP', None, Dtype.time),
    ('MAP', None, Dtype.unsupported),
    ('VECTOR', None, Dtype.unsupported),
])
def test_described_aliases(name, scale, expected):
    assert dtype_for_described(name, scale) == expected


@pytest.mark.parametrize('name,scale,expected', [
    ('NUMBER', 0, Dtype.int64),
    ('NUMBER', 2, Dtype.float64),
    ('DECIMAL', '0', Dtype.int64),
    ('NUMERIC', '2', Dtype.float64),
    ('INTEGER', None, Dtype.int64),
    ('INT', None, Dtype.int64),
    ('BIGINT', None, Dtype.int64),
    ('FLOAT', None, Dtype.float64),
    ('DOUBLE', None, Dtype.float64),
    ('REAL', None, Dtype.float64),
    ('TEXT', None, Dtype.string),
    ('VARCHAR', None, Dtype.string),
    ('CHAR', None, Dtype.string),
    ('DATETIME', None, Dtype.time),
    ('VARBINARY', None, Dtype.binary),
    ('BINARY', None, Dtype.binary),
    ('DATE', None, Dtype.date),
    ('TIMESTAMP_NTZ', None, Dtype.time),
    ('TIMESTAMP_LTZ', None, Dtype.time),
    ('TIMESTAMP_TZ', None, Dtype.time),
    ('VARIANT', None, Dtype.string),
    ('ARRAY', None, Dtype.string),
    ('VECTOR', None, Dtype.unsupported),
    ('OBJECT', None, Dtype.unsupported),
    ('TIME', None, Dtype.unsupported),
    ('GEOGRAPHY', None, Dtype.unsupported),
    ('NOPE', None, Dtype.unsupported),
    ('text', None, Dtype.string),
    (' Boolean ', None, Dtype.bool),
])
def test_catalog_types(name, scale, expected):
    assert dtype_for(name, scale) == expected
