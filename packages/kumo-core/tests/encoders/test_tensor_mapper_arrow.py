# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden tests: native ``pyarrow.Array`` mapper path == the pandas path."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    ImageTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
    TextTensorMapper,
    TimestampTensorMapper,
    UnsupportedTensorMapper,
)
from kumo_core.types.features import TensorData
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


def _pandas_reference(
    mapper: NumericalTensorMapper
    | CategoricalTensorMapper
    | MultiCategoricalTensorMapper
    | SequenceTensorMapper,
    series: pd.Series,
) -> TensorData:
    """Retain the pandas reference calculations for Arrow encoding parity."""
    if isinstance(mapper, NumericalTensorMapper):
        if pd.api.types.is_timedelta64_dtype(series):
            series = series.dt.total_seconds()
        dtype = str(torch.get_default_dtype()).split(".")[-1]
        return torch.from_numpy(series.to_numpy().astype(dtype)).view(-1, 1)
    if isinstance(mapper, CategoricalTensorMapper):
        return mapper.mapper.map(series).view(-1, 1)
    if isinstance(mapper, MultiCategoricalTensorMapper):
        rowptr, index = mapper.mapper.map_multi(series, mapper.sep)
        return MultiNestedTensor(
            num_rows=len(series), num_cols=1, values=index, offset=rowptr
        )
    return mapper._forward_pandas(series)


def _assert_same(pandas_out, arrow_out) -> None:
    assert type(pandas_out) is type(arrow_out)
    if isinstance(pandas_out, MultiNestedTensor):
        assert pandas_out.values.equal(arrow_out.values)
        assert pandas_out.offset.equal(arrow_out.offset)
    elif isinstance(pandas_out, MultiEmbeddingTensor):
        assert pandas_out.values.allclose(arrow_out.values, equal_nan=True)
        assert pandas_out.offset.equal(arrow_out.offset)
    else:
        assert isinstance(pandas_out, Tensor)
        assert pandas_out.shape == arrow_out.shape
        if pandas_out.is_floating_point():
            assert pandas_out.allclose(arrow_out, equal_nan=True)
        else:
            assert pandas_out.equal(arrow_out)


def test_numerical_arrow_matches_pandas() -> None:
    mapper = NumericalTensorMapper()
    ser = pd.Series([1.0, 2.0, None, 4.0])
    arr = pa.array([1.0, 2.0, None, 4.0], type=pa.float64())
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))
    # Integer column with nulls: arrow upcasts int -> float64+NaN, matching the
    # pandas default for a nullable-int column.
    ser_i = pd.Series([1, 2, None, 4])
    arr_i = pa.array([1, 2, None, 4], type=pa.int64())
    _assert_same(_pandas_reference(mapper, ser_i), mapper(arr_i))
    # ChunkedArray input preserves the same row order and null values.
    chunked = pa.chunked_array([[1.0, 2.0], [None, 4.0]], type=pa.float64())
    _assert_same(_pandas_reference(mapper, ser), mapper(chunked))


def test_numerical_duration_arrow_matches_pandas() -> None:
    # A ``duration`` column must be normalized to seconds to match the stats
    # the encoder computes via ``Series.dt.total_seconds()``. The native path
    # normalizes durations before conversion so units stay consistent; a raw
    # ``astype(float)`` would keep nanoseconds and diverge by ~1e9.
    mapper = NumericalTensorMapper()
    ser = pd.Series(pd.to_timedelta(["2 days", "1 days", None]))
    arr = pa.array(ser)  # inferred as duration[ns]
    assert pa.types.is_duration(arr.type)
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))


def test_categorical_str_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[["A", 2], ["B", 1]])
    ser = pd.Series(["A", "B", None, "A", "C"])  # 'C' is out-of-vocab -> -1
    arr = pa.array(["A", "B", None, "A", "C"])
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))
    # Dictionary-encoded arrow exercises the zero-copy indices path.
    _assert_same(_pandas_reference(mapper, ser), mapper(arr.dictionary_encode()))


def test_categorical_int_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[[10, 2], [20, 1]])
    ser = pd.Series([10, 20, None, 10, 30])
    arr = pa.array([10, 20, None, 10, 30], type=pa.int64())
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))
    # Dictionary-encoded int column also routes through the zero-copy path.
    _assert_same(_pandas_reference(mapper, ser), mapper(arr.dictionary_encode()))


def test_categorical_all_null_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[["A", 2], ["B", 1]])
    ser = pd.Series([None, None, None], dtype=object)
    arr = pa.array([None, None, None], type=pa.string())
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))


def test_multicategorical_list_arrow_matches_pandas() -> None:
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None
    )
    # No nulls and no empty lists, so this hits the native CSR fast path (a
    # list-level null would defer to the pandas fallback).
    ser = pd.Series([[1, 2], [2, 3], [1], [1, 1]])
    arr = pa.array([[1, 2], [2, 3], [1], [1, 1]], type=pa.list_(pa.int64()))
    assert arr.null_count == 0
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))
    # LargeListArray takes the same native path.
    large = pa.array([[1, 2], [2, 3], [1], [1, 1]], type=pa.large_list(pa.int64()))
    _assert_same(_pandas_reference(mapper, ser), mapper(large))


def test_multicategorical_null_arrow_matches_pandas() -> None:
    # A list-level null forces the pandas fallback; it must still match.
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None
    )
    ser = pd.Series([[1, 2], [2, 3], None, [1]])
    arr = pa.array([[1, 2], [2, 3], None, [1]], type=pa.list_(pa.int64()))
    assert arr.null_count == 1
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))


@pytest.mark.parametrize("categories", [("a", "b"), (1, 2)])
@pytest.mark.parametrize("large_list", [False, True])
@pytest.mark.parametrize("with_null", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_multicategorical_empty_lists_preserve_legacy_tokens(
    categories: tuple[str, str] | tuple[int, int],
    large_list: bool,
    with_null: bool,
    chunked: bool,
) -> None:
    """Empty Arrow lists retain one -1 token, just like null rows on main."""
    first, second = categories
    mapper = MultiCategoricalTensorMapper([[first, 2], [second, 1]], sep=None)
    rows = [[first], [], [second, first]]
    value_type = pa.string() if isinstance(first, str) else pa.int64()
    list_type = pa.large_list(value_type) if large_list else pa.list_(value_type)
    array = pa.array(rows + [None] if with_null else rows, type=list_type)
    source = pa.chunked_array([array[:2], array[2:]]) if chunked else array

    result = mapper(source)

    assert result.offset.tolist() == ([0, 1, 2, 4, 5] if with_null else [0, 1, 2, 4])
    assert result.values.tolist() == ([0, -1, 1, 0, -1] if with_null else [0, -1, 1, 0])


@pytest.mark.parametrize("value_type", [pa.string(), pa.int64()])
def test_multicategorical_all_empty_lists_preserve_legacy_tokens(
    value_type: pa.DataType,
) -> None:
    """All-empty rows still emit missing tokens when the vocabulary is empty."""
    mapper = MultiCategoricalTensorMapper([], sep=None)
    array = pa.array([[], []], type=pa.list_(value_type))

    result = mapper(array)

    assert result.offset.tolist() == [0, 1, 2]
    assert result.values.tolist() == [-1, -1]


def test_multicategorical_sliced_list_arrow_matches_pandas() -> None:
    # A head slice keeps offsets[0] == 0 but ``arr.values`` still spans the
    # full child buffer; the native guard must defer this to the pandas
    # fallback rather than build a mismatched tensor.
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None
    )
    arr = pa.array([[1, 2], [2, 3], [1], [1, 1]], type=pa.list_(pa.int64()))
    sliced = arr.slice(0, 2)  # head slice: values buffer not trimmed
    ser = pd.Series([[1, 2], [2, 3]])
    _assert_same(_pandas_reference(mapper, ser), mapper(sliced))


def test_sequence_list_arrow_matches_pandas() -> None:
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0], None])
    arr = pa.array([[1.0, 2.0], [3.0, 4.0], None], type=pa.list_(pa.float64()))
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))
    # LargeListArray takes the same native path.
    large = pa.array([[1.0, 2.0], [3.0, 4.0], None], type=pa.large_list(pa.float64()))
    _assert_same(_pandas_reference(mapper, ser), mapper(large))


def test_sequence_truncation_arrow_matches_pandas() -> None:
    # Rows longer than ``max_length`` keep the prefix; shorter rows pad w/ NaN.
    mapper = SequenceTensorMapper(min_length=1, max_length=2)
    ser = pd.Series([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    arr = pa.array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]], type=pa.list_(pa.float64()))
    _assert_same(_pandas_reference(mapper, ser), mapper(arr))


def test_sequence_sliced_list_arrow_matches_pandas() -> None:
    # Head slice: native guard must defer to the pandas fallback (see the
    # multicategorical sliced test for the underlying pyarrow behavior).
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    arr = pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type=pa.list_(pa.float64()))
    sliced = arr.slice(0, 2)
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0]])
    _assert_same(_pandas_reference(mapper, ser), mapper(sliced))


@pytest.mark.parametrize("empty", [False, True])
def test_all_mappers_accept_chunked_arrow(empty: bool) -> None:
    """Every mapper shares the Arrow array and chunked-array input contract."""
    cases: list[tuple[TensorMapper, pa.Array]] = [
        (NumericalTensorMapper(), pa.array([1.0, None, 3.0])),
        (CategoricalTensorMapper([["a", 2]]), pa.array(["a", None, "z"])),
        (MultiCategoricalTensorMapper([[1, 2]], None), pa.array([[1], [], None])),
        (SequenceTensorMapper(1, 2), pa.array([[1.0], [2.0, 3.0], None])),
        (TimestampTensorMapper(), pa.array(["2000-01-01", None, "2000-01-03"])),
        (TextTensorMapper(), pa.array(["hello", None, "world"])),
        (ImageTensorMapper(), pa.array(["a", None, "b"])),
        (UnsupportedTensorMapper(), pa.array([1, None, 2])),
    ]
    for mapper, array in cases:
        if empty:
            array = array.slice(0, 0)
        chunked = pa.chunked_array([array.slice(0, 1), array.slice(1)])

        _assert_same(mapper(array), mapper(chunked))
        _assert_same(mapper(array), mapper.forward(array))


@pytest.mark.parametrize("mapper", [NumericalTensorMapper(), ImageTensorMapper()])
def test_mapper_rejects_non_arrow_input(mapper: TensorMapper) -> None:
    """Legacy pandas and NumPy inputs fail at the public mapper boundary."""
    series = pd.Series([1, 2])
    for values in (series, series.to_numpy()):
        with pytest.raises(TypeError, match="Arrow array"):
            mapper(values)  # type: ignore[arg-type]


def test_categorical_float_arrow_preserves_zero_and_unknown_categories() -> None:
    """Float Arrow inputs retain signed-zero equivalence and unknown-value codes."""
    mapper = CategoricalTensorMapper(
        category_counts=[[0.0, 25], [1.5, 20], [2.5, 15], [3.5, 10]]
    )
    values = pa.array([0.0, -0.0, 1.5, 3.5, 4.5], type=pa.float64())

    mapped = mapper(values)

    assert mapped.view(-1).tolist() == [0, 0, 1, 3, -1]


@pytest.mark.parametrize(
    "mapper,array",
    [
        (NumericalTensorMapper(), pa.array([1.0, None, 3.0])),
        (CategoricalTensorMapper([["a", 2]]), pa.array(["a", None, "z"])),
        (MultiCategoricalTensorMapper([[1, 2]], None), pa.array([[1], [], None])),
        (SequenceTensorMapper(0, 3), pa.array([[1.0], [], [2.0, 3.0]])),
        (TimestampTensorMapper(), pa.array(["2000-01-01", None, "2000-01-03"])),
        (TextTensorMapper(), pa.array(["hello", None, "world"])),
        (ImageTensorMapper(), pa.array(["a", None, "b"])),
        (UnsupportedTensorMapper(), pa.array([1, None, 2])),
    ],
)
def test_chunk_encoding_preserves_input_chunks(
    mapper: TensorMapper,
    array: pa.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encode slices, empty chunks, and zero-chunk columns without combining."""
    source = pa.chunked_array([array[:1], array[:0], array[1:]])
    from kumo_core.encoders import tensor_mapper

    expected = mapper(array)
    original_chunks = tensor_mapper._arrow_chunks
    seen = []

    def checked_chunks(values: pa.Array | pa.ChunkedArray) -> list[pa.Array]:
        assert values is source
        chunks = original_chunks(values)
        seen.extend(chunks)
        return chunks

    monkeypatch.setattr(tensor_mapper, "_arrow_chunks", checked_chunks)
    _assert_same(expected, mapper(source))
    assert len(seen) == (
        0
        if mapper.stype in {tensor_mapper.Stype.image, tensor_mapper.Stype.unsupported}
        else 3
    )
    monkeypatch.undo()
    _assert_same(mapper(array[:0]), mapper(pa.chunked_array([], type=array.type)))


@pytest.mark.parametrize("dictionary", [False, True])
def test_dictionary_chunks_keep_independent_vocabularies(dictionary: bool) -> None:
    """Different per-chunk dictionary orders retain global category codes."""
    chunks = [pa.array(["a", "b", None]), pa.array(["b", "z", "a"])]
    if dictionary:
        chunks = [chunk.dictionary_encode() for chunk in chunks]
    result = CategoricalTensorMapper([["a", 2], ["b", 1]])(pa.chunked_array(chunks))
    assert result.flatten().tolist() == [0, 1, -1, 1, -1, 0]


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_chunked_numerical_output_owns_its_memory(default_dtype: torch.dtype) -> None:
    """Writing encoded values never mutates the Arrow source buffers."""
    source = pa.chunked_array([[1.0, 2.0], [3.0]], type=pa.float64())
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        result = NumericalTensorMapper()(source)
        assert result.dtype == default_dtype
        result.fill_(0)
    finally:
        torch.set_default_dtype(previous)
    assert source.to_pylist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "rows",
    [
        [None, "2020-01-01", "01/02/2020"],
        ["invalid", "2020-01-01", "01/02/2020"],
        [None, "2020-01-01", "2020-01-02 00:00:00"],
    ],
)
def test_timestamp_format_inference_spans_chunks(rows: list[str | None]) -> None:
    """Chunk boundaries must not change pandas' column-wide format inference."""
    array = pa.array(rows)
    mapper = TimestampTensorMapper()
    _assert_same(
        mapper(array), mapper(pa.chunked_array([array[:1], array[1:2], array[2:]]))
    )


@pytest.mark.parametrize("sequence", [False, True])
def test_leading_null_chunk_preserves_string_backward(sequence: bool) -> None:
    """Missing values in the first chunk do not select list output formatting."""
    array = pa.array([None, "[1,2]"] if sequence else [None, "a,b"])
    mapper = (
        SequenceTensorMapper(0, 2)
        if sequence
        else MultiCategoricalTensorMapper([["a", 2], ["b", 1]], ",")
    )
    expected = mapper(array)
    reference = mapper.backward(expected)
    mapper._is_string_dtype = None
    actual = mapper(pa.chunked_array([array[:1], array[1:]]))
    _assert_same(expected, actual)
    pd.testing.assert_series_equal(reference, mapper.backward(actual))


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_duration_chunks_divide_directly_into_destination(
    unit: str,
    dtype: torch.dtype,
    sliced: bool,
    chunked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectorized duration scaling matches pandas without a seconds buffer."""
    ticks = [
        0,
        1,
        -1,
        1_234_567_891,
        None,
        -(2**63),
        -(2**63) + 1,
        2**63 - 1,
        1,
        None,
        3,
        -4,
        None,
        6,
        7,
        8,
    ]
    array = pa.array(ticks, type=pa.duration(unit))
    if sliced:
        array = array.slice(3, 12)
    source = pa.chunked_array([array[:2], array[2:]]) if chunked else array
    previous = torch.get_default_dtype()
    destinations = []
    divide = np.divide

    def divide_into_destination(*args: object, **kwargs: object) -> np.ndarray:
        output = kwargs["out"]
        assert isinstance(output, np.ndarray)
        destinations.append(output)
        return divide(*args, **kwargs)

    try:
        torch.set_default_dtype(dtype)
        mapper = NumericalTensorMapper()
        expected = _pandas_reference(mapper, array.to_pandas())
        monkeypatch.setattr(np, "divide", divide_into_destination)
        actual = mapper(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert len(destinations) == (2 if chunked else 1)
        assert all(np.shares_memory(output, actual.numpy()) for output in destinations)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_duration_chunks_without_nulls(unit: str) -> None:
    """Duration buffers without a validity bitmap use the same vectorized path."""
    array = pa.array([1, -1, 1_234_567_891], type=pa.duration(unit))
    mapper = NumericalTensorMapper()
    expected = _pandas_reference(mapper, array.to_pandas())
    actual = mapper(pa.chunked_array([array[:1], array[1:]]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "arrow_type",
    [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.bool_(),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [0, 1, 7, 8, 9])
@pytest.mark.parametrize("nullable", [False, True])
def test_primitive_arrow_buffers_match_pandas(
    arrow_type: pa.DataType,
    dtype: torch.dtype,
    offset: int,
    nullable: bool,
) -> None:
    """Native buffers and packed bits preserve values across sliced chunks."""
    if pa.types.is_boolean(arrow_type):
        rows = [index % 3 == 0 for index in range(32)]
    elif pa.types.is_floating(arrow_type):
        rows = [index / 4 - 3 for index in range(32)]
    else:
        rows = [index % 13 for index in range(32)]
    if nullable:
        rows = [None if index % 7 == 2 else value for index, value in enumerate(rows)]
    array = pa.array(rows, type=arrow_type).slice(offset, 17)
    source = pa.chunked_array([array[:3], array[3:8], array[8:]])
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        mapper = NumericalTensorMapper()
        expected = _pandas_reference(mapper, array.to_pandas())
        actual = mapper(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        actual.fill_(0)
        assert array.equals(pa.array(rows, type=arrow_type).slice(offset, 17))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("arrow_type", [pa.int64(), pa.uint64()])
def test_nullable_integer_rounding_is_independent_of_chunks(
    arrow_type: pa.DataType,
) -> None:
    """Nullable columns retain the legacy float64-to-float32 rounding step."""
    edge = -(2**63) if pa.types.is_signed_integer(arrow_type) else 2**64 - 1
    array = pa.array([2**62 + 2**38 + 1, None, 2**63 - 1, edge], type=arrow_type)
    mapper = NumericalTensorMapper()
    expected = _pandas_reference(mapper, array.to_pandas())
    actual = mapper(pa.chunked_array([array[:1], array[1:2], array[2:]]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("arrow_type", [pa.int64(), pa.uint32(), pa.float32()])
def test_nullable_numeric_reads_arrow_values_without_materializing_numpy(
    arrow_type: pa.DataType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only views of original Arrow values reach destination conversion."""
    from kumo_core.encoders import tensor_mapper

    array = pa.array([8, 4, None, 3, 2, 1], type=arrow_type).slice(1)
    chunks = [array[:2], array[2:]]
    source = pa.chunked_array(chunks)
    copies = []
    copy_numpy = tensor_mapper._copy_numpy

    def copy_arrow_view(
        destination: Tensor, values: np.ndarray, *, via_float64: bool = False
    ) -> None:
        chunk = chunks[len(copies)]
        dtype = np.dtype(chunk.type.to_pandas_dtype())
        assert values.dtype == dtype
        assert (
            values.ctypes.data
            == chunk.buffers()[1].address + chunk.offset * dtype.itemsize
        )
        copies.append(destination.numpy())
        copy_numpy(destination, values, via_float64=via_float64)

    monkeypatch.setattr(tensor_mapper, "_copy_numpy", copy_arrow_view)
    result = NumericalTensorMapper()(source)
    assert len(copies) == len(chunks)
    assert all(np.shares_memory(output, result.numpy()) for output in copies)


def test_boolean_bits_write_directly_into_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed Boolean values use final tensor slices as the bitwise output."""
    array = pa.array([True, None, False, True] * 5).slice(3, 13)
    bitwise_and = np.bitwise_and
    outputs = []

    def unpack_into_destination(*args: object, **kwargs: object) -> np.ndarray:
        output = kwargs["out"]
        assert isinstance(output, np.ndarray)
        outputs.append(output)
        return bitwise_and(*args, **kwargs)

    monkeypatch.setattr(np, "bitwise_and", unpack_into_destination)
    actual = NumericalTensorMapper()(array)
    expected = torch.tensor(
        [
            [float(value) if value is not None else float("nan")]
            for value in array.to_pylist()
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert len(outputs) == 8
    assert all(np.shares_memory(output, actual.numpy()) for output in outputs)


def test_numerical_null_type_and_all_null_chunks() -> None:
    """Entirely missing chunks need no values or validity conversion."""
    mapper = NumericalTensorMapper()
    for arrow_type in (pa.null(), pa.int64(), pa.float32(), pa.bool_()):
        array = pa.array([None, None], type=arrow_type)
        actual = mapper(pa.chunked_array([array[:1], array[1:]]))
        assert actual.shape == (2, 1)
        assert actual.isnan().all()
