# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for pandas-free forward and serialized mapper paths."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
    TextTensorMapper,
    TimestampTensorMapper,
)
from tests.support.embedding_models import TinyModel


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("dictionary", [False, True])
def test_duration_units(unit: str, dictionary: bool) -> None:
    """Nullable durations retain second units, including dictionary storage."""
    array = pa.array([2, None, -3], type=pa.duration(unit))
    if dictionary:
        array = array.dictionary_encode()
    scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]

    result = NumericalTensorMapper()(array)

    expected = torch.tensor([[2 / scale], [float("nan")], [-3 / scale]])
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize(
    "counts,array,expected",
    [
        (
            [[0.0, 2], [1.5, 1]],
            pa.array([-0.0, 0.0, None, 1.5, 2.0]),
            [0, 0, -1, 1, -1],
        ),
        ([[1, 1]], pa.array([1.0, 1.5, None]), [0, -1, -1]),
        ([[1, 1]], pa.array(["1", "other", None]), [-1, -1, -1]),
        ([[True, 2], [False, 1]], pa.array([True, None, False]), [0, -1, 1]),
        ([[b"a", 1]], pa.array([b"a", b"b", None]), [0, -1, -1]),
        (
            [[Decimal("1.25"), 1]],
            pa.array([Decimal("1.25"), None, Decimal("2.5")]),
            [0, -1, -1],
        ),
    ],
)
def test_category_lookup(
    counts: list[list[object]],
    array: pa.Array,
    expected: list[int],
    dictionary: bool,
) -> None:
    """Arrow lookup retains null, numeric, dictionary, and binary semantics."""
    mapper = CategoricalTensorMapper(counts)
    if dictionary:
        array = array.dictionary_encode()

    assert mapper(array).view(-1).tolist() == expected


def test_dictionary_with_empty_dictionary() -> None:
    """Null dictionary indices do not gather from an empty dictionary."""
    array = pa.DictionaryArray.from_arrays(
        pa.array([None, None], pa.int8()), pa.array([], pa.string())
    )

    assert CategoricalTensorMapper([["a", 1]])(array).view(-1).tolist() == [-1, -1]


@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.string()), pa.large_list(pa.string()), pa.list_(pa.string(), 2)],
)
def test_multicategorical_visible_rows(list_type: pa.DataType) -> None:
    """Slices, empty strings, unknown values, and null list rows retain CSR order."""
    array = pa.array(
        [["ignored", "ignored"], ["b", ""], None, [None, "a"], ["ignored", "ignored"]],
        type=list_type,
    ).slice(1, 3)
    mapper = MultiCategoricalTensorMapper([["a", 2], ["b", 1]], None)

    result = mapper(array)

    assert result.offset.tolist() == [0, 1, 2, 4]
    assert result.values.tolist() == [1, -1, -1, 0]


@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.float32()), pa.large_list(pa.float32()), pa.list_(pa.float32(), 2)],
)
@pytest.mark.parametrize("serialized", [False, True])
def test_sequence_visible_null_rows(list_type: pa.DataType, serialized: bool) -> None:
    """Null slots never expose hidden children, including fixed-width lists."""
    array = pa.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], type=list_type)
    if isinstance(array, pa.FixedSizeListArray):
        array = pa.FixedSizeListArray.from_arrays(
            array.values, 2, mask=pa.array([False, True, False])
        )
    else:
        array = type(array).from_arrays(
            array.offsets, array.values, mask=pa.array([False, True, False])
        )
    array = array.slice(1)
    mapper = SequenceTensorMapper(0, 3)

    result = mapper.post_forward(array) if serialized else mapper(array)

    torch.testing.assert_close(
        result.values,
        torch.tensor([[float("nan")] * 3, [50.0, 60.0, float("nan")]]),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "mapper", [MultiCategoricalTensorMapper([], None), TextTensorMapper()]
)
@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.int64()), pa.large_list(pa.int64()), pa.list_(pa.int64(), 2)],
)
def test_serialized_nested_slices(mapper: TensorMapper, list_type: pa.DataType) -> None:
    """Serialized nested tensors own only the selected child values and offsets."""
    array = pa.array([[8, 9], [1, 2], [3, -1]], type=list_type).slice(1)

    result = mapper.post_forward(array)

    assert result.offset.tolist() == [0, 2, 4]
    assert result.values.tolist() == [1, 2, 3, -1]
    result.values.fill_(0)
    assert array.to_pylist() == [[1, 2], [3, -1]]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamp_units_and_timezones(unit: str) -> None:
    """Timestamp resolutions and timezones normalize to the same UNIX seconds."""
    array = pa.array(
        [datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), None],
        type=pa.timestamp(unit, tz="America/New_York"),
    )
    mapper = TimestampTensorMapper()

    result = mapper(array)

    assert result[0, mapper.index["unix"]].item() == 1_577_934_245
    assert result[1].tolist() == [-1] * len(mapper.index)


@pytest.mark.parametrize(
    "values,format,expected",
    [
        (["2020-01-02T03:04:05.123456789", "bad", None], None, 1_577_934_245),
        (["2020-01-02T03:04:05+02:00", "bad", None], None, 1_577_927_045),
        (["02/01/2020 03:04:05", "bad", None], "%d/%m/%Y %H:%M:%S", 1_577_934_245),
        (
            ["02/01/2020 03:04:05.123456789", "bad", None],
            "%d/%m/%Y %H:%M:%S.%f",
            1_577_934_245,
        ),
    ],
)
def test_timestamp_strings(
    values: list[str | None], format: str | None, expected: int
) -> None:
    """ISO, timezone, and explicit-format parsing coerce invalid strings to null."""
    mapper = TimestampTensorMapper(format)

    result = mapper(pa.array(values))

    assert result[:, mapper.index["unix"]].tolist() == [expected, -1, -1]


def test_encoding_does_not_construct_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """All forward paths work with pandas construction and parsing disabled."""
    text = TextTensorMapper(TinyModel(10))
    text.model.tokenize(pa.array([], pa.string()))
    cases = [
        (NumericalTensorMapper(), pa.array([1, None, 2], pa.duration("ms"))),
        (
            CategoricalTensorMapper([["a", 2]]),
            pa.array(["a", None, "b"]).dictionary_encode(),
        ),
        (
            MultiCategoricalTensorMapper([["a", 2], ["b", 1]], "|"),
            pa.array(["a|b", None, ""]),
        ),
        (
            MultiCategoricalTensorMapper([["a", 2]], None),
            pa.array([["a", None], [], None]),
        ),
        (SequenceTensorMapper(0, 3), pa.array(["[1, 2]", None, "[]"])),
        (SequenceTensorMapper(0, 3), pa.array([[1.0, None], None, []])),
        (TimestampTensorMapper(), pa.array(["2020-01-01", "bad", None])),
        (text, pa.array(["The, OF!", None, "unknown"])),
    ]

    def reject_pandas(*args: object, **kwargs: object) -> None:
        raise AssertionError("Forward encoding must not construct pandas objects")

    with monkeypatch.context() as patch:
        for name in ("Series", "Index", "to_datetime"):
            patch.setattr(pd, name, reject_pandas)
        for mapper, array in cases:
            result = mapper(array)
            assert len(result) == len(array)
    tokens = text(pa.array(["The, OF!", None, "unknown"]))
    assert tokens.offset.tolist() == [0, 2, 2, 2]
    assert tokens.values.tolist() == [0, 3]


@pytest.mark.parametrize(
    "array",
    [
        pa.array([[True, None], [False]], type=pa.list_(pa.bool_())),
        pa.array([[Decimal("1.0"), None], [Decimal("0.0")]]),
    ],
)
def test_sequence_nonprimitive_numbers(array: pa.Array) -> None:
    """Numeric child values normalize nullable booleans and decimal objects."""
    result = SequenceTensorMapper(0, 2)(array)

    torch.testing.assert_close(
        result.values,
        torch.tensor([[1.0, float("nan")], [0.0, float("nan")]]),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "format,values",
    [
        (None, ["1500-01-01", "2500-01-01", "invalid"]),
        ("%Y-%m-%d", ["1500-01-01", "2500-01-01", "invalid"]),
    ],
)
def test_timestamp_out_of_bounds(format: str | None, values: list[str]) -> None:
    """String timestamps outside the nanosecond range are missing, never wrapped."""
    result = TimestampTensorMapper(format)(pa.array(values))

    assert result.tolist() == [[-1] * len(TimestampTensorMapper.index)] * 3


def test_fractional_timestamp_parser_retains_nanoseconds() -> None:
    """ISO and strptime paths retain all nine fraction digits before encoding."""
    from kumo_core.encoders.tensor_mapper import _timestamp_nanoseconds

    for values, format in [
        (["2020-01-02T03:04:05.123456789", "invalid"], None),
        (["02/01/2020 03:04:05.123456789", "invalid"], "%d/%m/%Y %H:%M:%S.%f"),
    ]:
        result = _timestamp_nanoseconds(pa.array(values), format)
        assert result.tolist() == [1_577_934_245_123_456_789, -(2**63)]


def test_arrow_text_matches_unicode_normalization() -> None:
    """Arrow normalization retains Python casing, whitespace, and HTML semantics."""
    from kumo_core.glove import VocabularyIndex

    words = pd.Index(["ος", "οσ", "i", "i\u0307", "hello", "world"])
    index = VocabularyIndex(words)
    rows = ["ΟΣ", "İ", "Hello\u00a0world", "hello<br\u00a0/>world", None, ""]
    expected_offsets, expected_tokens = index.tokenize(pd.Series(rows))

    offsets, tokens = index.tokenize(pa.array(rows))

    assert offsets.equal(expected_offsets)
    assert tokens.equal(expected_tokens)


@pytest.mark.parametrize(
    "value", ["January 2, 2020", "2020/01/02", "02-Jan-2020", "2020-01-02"]
)
def test_inferred_timestamp_formats(value: str) -> None:
    """Non-ISO date formats remain supported without pandas parsing."""
    mapper = TimestampTensorMapper()

    result = mapper(pa.array([value, None]))

    assert result[:, mapper.index["unix"]].tolist() == [1_577_923_200, -1]


def test_invalid_calendar_dates() -> None:
    """Invalid dates do not roll over into a different calendar day."""
    mapper = TimestampTensorMapper()

    result = mapper(pa.array(["2021-02-29", "2020-02-30", "2020-13-01"]))

    assert result.tolist() == [[-1] * len(mapper.index)] * 3


def test_numeric_timestamp_missing_values() -> None:
    """Numeric timestamps retain nanosecond units and coerce non-finite values."""
    mapper = TimestampTensorMapper()

    result = mapper(pa.array([1_577_923_200_000_000_000.0, None, float("inf")]))

    assert result[:, mapper.index["unix"]].tolist() == [1_577_923_200, -1, -1]


@pytest.mark.parametrize("values", [["a", "a"], [0.0, -0.0], [None], [float("nan")]])
@pytest.mark.parametrize("multicategorical", [False, True])
def test_invalid_vocabulary_is_rejected(
    values: list[object], multicategorical: bool
) -> None:
    """Invalid vocabularies retain the previous categorical mapper validation."""
    counts = [[value, 1] for value in values]

    with pytest.raises(ValueError, match="Categorical categories"):
        if multicategorical:
            MultiCategoricalTensorMapper(counts, None)
        else:
            CategoricalTensorMapper(counts)


@pytest.mark.parametrize("format", [None, "%Y-%m-%dT%H:%M:%S.%f"])
def test_timestamp_parser_checks_nanosecond_boundaries(format: str | None) -> None:
    """Preserve boundary precision and reject wraparound in older Arrow releases."""
    from kumo_core.encoders.tensor_mapper import _timestamp_nanoseconds

    values = pa.array(
        [
            "1677-09-21T00:12:43.145224193",
            "1677-09-21T00:12:43.145224191",
            "2262-04-11T23:47:16.854775807",
            "2262-04-11T23:47:16.854775808",
            "2000-01-01T00:00:00.000000001",
        ]
    )

    result = _timestamp_nanoseconds(values, format)

    assert result.tolist() == [
        -(2**63) + 1,
        -(2**63),
        2**63 - 1,
        -(2**63),
        946684800000000001,
    ]
