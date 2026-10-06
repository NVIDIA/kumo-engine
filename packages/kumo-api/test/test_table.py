# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from typing import get_args

import pytest
from pydantic import ValidationError

from kumoapi.json_serde import from_json, to_json_dict
from kumoapi.source_table import SourceColumn
from kumoapi.table import Column, ColumnMetadataRequest
from kumoapi.typing import (
    SUPPORTED_MULTICATEGORICAL_DELIMITERS,
    Dtype,
    MulticategoricalDelimiter,
    Stype,
)


@pytest.mark.parametrize('delimiter', [',', '|', ';', '\t'])
def test_scalar_string_multicategorical_delimiters_round_trip(delimiter):
    column = Column(
        name='interests',
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        multicategorical_delimiter=delimiter,
    )

    payload = to_json_dict(column)

    assert payload['multicategorical_delimiter'] == delimiter
    assert from_json(payload, Column) == column
    assert SUPPORTED_MULTICATEGORICAL_DELIMITERS == {',', '|', ';', '\t'}


def test_public_delimiter_type_exposes_exact_schema_values():
    assert get_args(MulticategoricalDelimiter) == (',', '|', ';', '\t')


@pytest.mark.parametrize('delimiter', ['', ':', '::', '[', '{', '"', '[]'])
def test_unsupported_multicategorical_delimiters_are_rejected(delimiter):
    with pytest.raises((ValidationError, ValueError)):
        Column(
            name='interests',
            stype=Stype.multicategorical,
            dtype=Dtype.string,
            multicategorical_delimiter=delimiter,
        )


@pytest.mark.parametrize('dtype', [Dtype.stringlist, Dtype.intlist])
def test_native_multicategorical_arrays_do_not_accept_a_delimiter(dtype):
    column = Column(
        name='interests',
        stype=Stype.multicategorical,
        dtype=dtype,
    )
    assert column.multicategorical_delimiter is None

    with pytest.raises((ValidationError, ValueError),
                       match='only valid for a scalar string'):
        Column(
            name='interests',
            stype=Stype.multicategorical,
            dtype=dtype,
            multicategorical_delimiter='|',
        )


@pytest.mark.parametrize(
    ('stype', 'dtype'),
    [
        (Stype.categorical, Dtype.string),
        (Stype.text, Dtype.string),
        (Stype.multicategorical, Dtype.binary),
    ],
)
def test_non_scalar_string_multicategorical_columns_reject_a_delimiter(
        stype, dtype):
    with pytest.raises((ValidationError, ValueError),
                       match='only valid for a scalar string|requires dtype'):
        Column(
            name='interests',
            stype=stype,
            dtype=dtype,
            multicategorical_delimiter=',',
        )


def test_resolved_scalar_string_multicategorical_column_requires_delimiter():
    with pytest.raises((ValidationError, ValueError),
                       match='requires multicategorical_delimiter'):
        Column(
            name='interests',
            stype=Stype.multicategorical,
            dtype=Dtype.string,
        )


@pytest.mark.parametrize('dtype', [Dtype.binary, Dtype.floatlist])
def test_resolved_multicategorical_column_rejects_unsupported_dtype(dtype):
    with pytest.raises((ValidationError, ValueError),
                       match="requires dtype='string'"):
        Column(
            name='interests',
            stype=Stype.multicategorical,
            dtype=dtype,
        )


def test_multicategorical_stype_supports_only_contract_dtypes():
    assert Stype.multicategorical.supports_dtype(Dtype.string)
    assert Stype.multicategorical.supports_dtype(Dtype.stringlist)
    assert Stype.multicategorical.supports_dtype(Dtype.intlist)
    assert not Stype.multicategorical.supports_dtype(Dtype.binary)
    assert not Stype.multicategorical.supports_dtype(Dtype.floatlist)


@pytest.mark.parametrize('delimiter', [',', '|', ';', '\t'])
def test_metadata_request_carries_delimiter_before_types_are_inferred(
        delimiter):
    request = ColumnMetadataRequest(
        name='interests',
        multicategorical_delimiter=delimiter,
    )

    assert request.stype is None
    assert request.dtype is None
    assert request.multicategorical_delimiter == delimiter


@pytest.mark.parametrize(
    ('stype', 'dtype'),
    [
        (Stype.categorical, None),
        (None, Dtype.int),
        (Stype.multicategorical, Dtype.stringlist),
    ],
)
def test_metadata_request_rejects_known_incompatible_types(stype, dtype):
    with pytest.raises((ValidationError, ValueError),
                       match='only valid for a scalar string'):
        ColumnMetadataRequest(
            name='interests',
            stype=stype,
            dtype=dtype,
            multicategorical_delimiter='|',
        )


def test_source_column_preserves_a_proven_delimiter():
    source_column = SourceColumn(
        name='interests',
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        is_primary=False,
        multicategorical_delimiter=';',
    )

    assert from_json(to_json_dict(source_column),
                     SourceColumn) == source_column


def test_resolved_source_column_requires_scalar_delimiter_proof():
    with pytest.raises((ValidationError, ValueError),
                       match='requires multicategorical_delimiter'):
        SourceColumn(
            name='interests',
            stype=Stype.multicategorical,
            dtype=Dtype.string,
            is_primary=False,
        )
