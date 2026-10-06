# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Tests for the column typing vocabulary.

`Dtype` and `Stype` live here because they are wire contract: a client and the
engine must agree on what a column *is* before either can act on it. The
methods below were previously duplicated in `kumo_core`, which let the two
definitions drift; these tests pin the merged behaviour.

Conversions to numpy/pandas/torch dtypes deliberately do **not** live here.
They need torch and pyarrow, which this package must not depend on, and they
remain free functions in `kumo_core.types.conversions`.
"""
import pytest

from kumoapi.typing import Dtype, Stype


class TestTemporal:
    def test_is_temporal_matches_is_timestamp(self) -> None:
        """Two names for one predicate; both are in use across the codebase."""
        for dtype in Dtype:
            assert dtype.is_temporal() == dtype.is_timestamp()

    @pytest.mark.parametrize('dtype', [Dtype.date, Dtype.time])
    def test_dates_and_times_are_temporal(self, dtype: Dtype) -> None:
        assert dtype.is_temporal()

    @pytest.mark.parametrize('dtype', [Dtype.string, Dtype.int64, Dtype.bool])
    def test_nothing_else_is_temporal(self, dtype: Dtype) -> None:
        assert not dtype.is_temporal()

    def test_a_timestamp_stype_requires_a_real_date(self) -> None:
        """A string is not a timestamp.

        The engine cannot ingest a string time column, so accepting one here
        only moves the failure later. A caller holding timestamps as text
        declares the column's dtype as `date` and converts on the way in.
        """
        assert Stype.timestamp.supports_dtype(Dtype.date)
        assert Stype.timestamp.supports_dtype(Dtype.time)
        assert not Stype.timestamp.supports_dtype(Dtype.string)

    @pytest.mark.parametrize('dtype,expected', [
        (Dtype.string, True),
        (Dtype.int64, True),
        (Dtype.float64, False),
        (Dtype.date, False),
    ])
    def test_what_can_be_cast_to_a_date(self, dtype: Dtype,
                                        expected: bool) -> None:
        """Strings parse; integers are read as epoch offsets."""
        assert dtype.supports_cast_to_temporal() is expected

    def test_casting_is_not_native_support(self) -> None:
        """`supports_cast_from_dtype` asks a different question.

        A string may be *cast* to a timestamp even though a timestamp does not
        natively accept one.
        """
        assert Stype.timestamp.supports_cast_from_dtype(Dtype.string)
        assert not Stype.timestamp.supports_dtype(Dtype.string)

    def test_numerical_can_be_cast_from_a_date(self) -> None:
        assert Stype.numerical.supports_cast_from_dtype(Dtype.date)
        assert not Stype.numerical.supports_cast_from_dtype(Dtype.string)


class TestLists:
    @pytest.mark.parametrize('dtype,element', [
        (Dtype.floatlist, Dtype.float),
        (Dtype.intlist, Dtype.int),
        (Dtype.stringlist, Dtype.string),
    ])
    def test_element_type(self, dtype: Dtype, element: Dtype) -> None:
        assert dtype.list_element_type() is element

    def test_a_scalar_has_no_element_type(self) -> None:
        with pytest.raises(AssertionError, match='not a list dtype'):
            Dtype.int64.list_element_type()


class TestTitleCase:
    @pytest.mark.parametrize('dtype,label', [
        (Dtype.int64, 'Int64'),
        (Dtype.floatlist, 'FloatList'),
        (Dtype.stringlist, 'StringList'),
        (Dtype.string, 'String'),
    ])
    def test_dtype_labels(self, dtype: Dtype, label: str) -> None:
        assert dtype.title_case() == label

    @pytest.mark.parametrize('stype,label', [
        (Stype.numerical, 'Numerical'),
        (Stype.multicategorical, 'Multicategorical'),
        (Stype.ID, 'ID'),
    ])
    def test_stype_labels(self, stype: Stype, label: str) -> None:
        """`ID` keeps its capitalisation; it is an initialism, not a word."""
        assert stype.title_case() == label


class TestSupportsDtype:
    def test_binary_is_not_text(self) -> None:
        """`text` means language content, which binary data does not carry."""
        assert Stype.text.supports_dtype(Dtype.string)
        assert not Stype.text.supports_dtype(Dtype.binary)
        assert not Stype.sequence.supports_dtype(Dtype.binary)
        assert not Stype.image.supports_dtype(Dtype.binary)

    def test_unsupported_accepts_anything(self) -> None:
        """A sentinel stype answers rather than raising, so a caller iterating
        every pairing does not have to special-case it.
        """
        for dtype in Dtype:
            assert Stype.unsupported.supports_dtype(dtype)


def test_is_public_facing() -> None:
    for dtype in Dtype:
        assert dtype.is_public_facing()
