# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from kumoapi.typing import Dtype, Stype

from kumoai_studio.graph import Column


def test_create():
    r"""Test basic column creation."""
    c = Column(name='test', stype=Stype.numerical, dtype=Dtype.int32)
    d = Column('test', 'numerical', 'int32')

    assert c.stype == Stype.numerical
    assert c.dtype == Dtype.int32

    assert str(c) == ('Column(name="test", stype="numerical", dtype="int32")')
    assert str(d) == str(c)
    assert c == d

    e = deepcopy(c)
    e.timestamp_format = 'yyyy-MM-dd'
    assert str(e) == ('Column(name="test", stype="numerical", dtype="int32", '
                      'timestamp_format="yyyy-MM-dd")')


def test_update():
    r"""Test column update behaves as expected."""
    c = Column(name='test', stype=Stype.numerical, dtype=Dtype.int32)
    c2 = Column(name='test', stype=Stype.categorical, dtype=Dtype.int32)

    assert c == c
    assert c != c2
    assert c.update(c2) == c
    assert c.update(c2, override=True) == c2


def test_readonly_name():
    r"""Test that column name cannot be modified after create."""
    c = Column(name='test', stype=Stype.numerical, dtype=Dtype.int32)
    with pytest.raises(AttributeError):
        c.name = 'test2'


@pytest.mark.parametrize('delimiter', [',', '|', ';', '\t'])
def test_scalar_string_multicategorical_delimiter(delimiter):
    column = Column(
        name='interests',
        stype='multicategorical',
        dtype='string',
        multicategorical_delimiter=delimiter,
    )

    assert column.multicategorical_delimiter == delimiter
    assert f'multicategorical_delimiter={delimiter!r}' in repr(column)
    assert deepcopy(column) == column
    assert hash(deepcopy(column)) == hash(column)


def test_resolved_scalar_string_multicategorical_requires_delimiter():
    with pytest.raises(ValueError,
                       match='requires multicategorical_delimiter'):
        Column('interests', 'multicategorical', 'string')


def test_incomplete_multicategorical_metadata_can_be_inferred_later():
    stype_only = Column('interests', stype='multicategorical')
    dtype_only = Column('interests', dtype='string')

    assert stype_only.stype == Stype.multicategorical
    assert stype_only.dtype is None
    assert dtype_only.stype is None
    assert dtype_only.dtype == Dtype.string


def test_mutable_metadata_can_be_staged_before_delimiter_is_assigned():
    column = Column('interests')

    column.stype = Stype.multicategorical
    column.dtype = Dtype.string
    column.multicategorical_delimiter = '|'

    column._validate_multicategorical_delimiter()


@pytest.mark.parametrize('delimiter', ['', ':', '::', '[]', '["a", "b"]'])
def test_unsupported_multicategorical_delimiter(delimiter):
    with pytest.raises(ValueError,
                       match='multicategorical_delimiter must be one of'):
        Column(
            name='interests',
            stype='multicategorical',
            dtype='string',
            multicategorical_delimiter=delimiter,
        )


@pytest.mark.parametrize('dtype', ['stringlist', 'intlist'])
def test_native_multicategorical_array_does_not_use_delimiter(dtype):
    assert Column('interests', 'multicategorical',
                  dtype).multicategorical_delimiter is None
    with pytest.raises(ValueError, match='only valid for a scalar string'):
        Column('interests', 'multicategorical', dtype,
               multicategorical_delimiter='|')


@pytest.mark.parametrize(
    ('stype', 'dtype'),
    [
        ('categorical', 'string'),
        ('text', 'string'),
        ('multicategorical', 'binary'),
    ],
)
def test_non_scalar_string_multicategorical_column_rejects_delimiter(
        stype, dtype):
    with pytest.raises(ValueError,
                       match='only valid for a scalar string|requires dtype'):
        Column('interests', stype, dtype, multicategorical_delimiter=';')


@pytest.mark.parametrize('dtype', ['binary', 'floatlist'])
def test_resolved_multicategorical_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match='requires dtype'):
        Column('interests', 'multicategorical', dtype)


def test_incompatible_type_mutation_clears_stale_delimiter():
    column = Column('interests', 'multicategorical', 'string',
                    multicategorical_delimiter='|')

    column.stype = Stype.categorical
    assert column.stype == Stype.categorical
    assert column.multicategorical_delimiter is None


def test_invalid_delimiter_mutation_is_rejected_without_corrupting_column():
    column = Column('interests', 'multicategorical', 'string',
                    multicategorical_delimiter='|')

    with pytest.raises(ValueError,
                       match='multicategorical_delimiter must be one of'):
        setattr(column, 'multicategorical_delimiter', ':')

    assert column.multicategorical_delimiter == '|'
