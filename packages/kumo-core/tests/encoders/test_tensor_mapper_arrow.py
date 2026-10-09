# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden tests: native ``pyarrow.Array`` mapper path == the pandas path."""
import pandas as pd
import pyarrow as pa
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, NumericalTensorMapper, SequenceTensorMapper


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
    _assert_same(mapper(ser), mapper(arr))
    # Integer column with nulls: arrow upcasts int -> float64+NaN, matching the
    # pandas default for a nullable-int column.
    ser_i = pd.Series([1, 2, None, 4])
    arr_i = pa.array([1, 2, None, 4], type=pa.int64())
    _assert_same(mapper(ser_i), mapper(arr_i))
    # ChunkedArray input is combined before the numpy conversion.
    chunked = pa.chunked_array([[1.0, 2.0], [None, 4.0]], type=pa.float64())
    _assert_same(mapper(ser), mapper(chunked))


def test_numerical_duration_arrow_matches_pandas() -> None:
    # A ``duration`` column must be normalized to seconds to match the stats
    # the encoder computes via ``Series.dt.total_seconds()``. The native path
    # defers duration types to ``forward`` so the units stay consistent; a raw
    # ``astype(float)`` would keep nanoseconds and diverge by ~1e9.
    mapper = NumericalTensorMapper()
    ser = pd.Series(pd.to_timedelta(['2 days', '1 days', None]))
    arr = pa.array(ser)  # inferred as duration[ns]
    assert pa.types.is_duration(arr.type)
    _assert_same(mapper(ser), mapper(arr))


def test_categorical_str_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[['A', 2], ['B', 1]])
    ser = pd.Series(['A', 'B', None, 'A', 'C'])  # 'C' is out-of-vocab -> -1
    arr = pa.array(['A', 'B', None, 'A', 'C'])
    _assert_same(mapper(ser), mapper(arr))
    # Dictionary-encoded arrow exercises the zero-copy indices path.
    _assert_same(mapper(ser), mapper(arr.dictionary_encode()))


def test_categorical_int_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[[10, 2], [20, 1]])
    ser = pd.Series([10, 20, None, 10, 30])
    arr = pa.array([10, 20, None, 10, 30], type=pa.int64())
    _assert_same(mapper(ser), mapper(arr))
    # Dictionary-encoded int column also routes through the zero-copy path.
    _assert_same(mapper(ser), mapper(arr.dictionary_encode()))


def test_categorical_all_null_arrow_matches_pandas() -> None:
    mapper = CategoricalTensorMapper(category_counts=[['A', 2], ['B', 1]])
    ser = pd.Series([None, None, None], dtype=object)
    arr = pa.array([None, None, None], type=pa.string())
    _assert_same(mapper(ser), mapper(arr))


def test_multicategorical_list_arrow_matches_pandas() -> None:
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None)
    # No nulls and no empty lists, so this hits the native CSR fast path (a
    # list-level null would defer to the pandas fallback).
    ser = pd.Series([[1, 2], [2, 3], [1], [1, 1]])
    arr = pa.array([[1, 2], [2, 3], [1], [1, 1]], type=pa.list_(pa.int64()))
    assert arr.null_count == 0
    _assert_same(mapper(ser), mapper(arr))
    # LargeListArray takes the same native path.
    large = pa.array([[1, 2], [2, 3], [1], [1, 1]],
                     type=pa.large_list(pa.int64()))
    _assert_same(mapper(ser), mapper(large))


def test_multicategorical_null_arrow_matches_pandas() -> None:
    # A list-level null forces the pandas fallback; it must still match.
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None)
    ser = pd.Series([[1, 2], [2, 3], None, [1]])
    arr = pa.array([[1, 2], [2, 3], None, [1]], type=pa.list_(pa.int64()))
    assert arr.null_count == 1
    _assert_same(mapper(ser), mapper(arr))


def test_multicategorical_empty_list_arrow_matches_pandas() -> None:
    # An empty list is deferred to the pandas fallback so the native path keeps
    # ``map_multi``'s (pre-existing) behavior of emitting a spurious -1.
    mapper = MultiCategoricalTensorMapper(category_counts=[[1, 2], [2, 1]],
                                          sep=None)
    arr = pa.array([[1], [], [2, 1]], type=pa.list_(pa.int64()))
    assert arr.null_count == 0
    _assert_same(mapper(arr.to_pandas()), mapper(arr))


def test_multicategorical_sliced_list_arrow_matches_pandas() -> None:
    # A head slice keeps offsets[0] == 0 but ``arr.values`` still spans the
    # full child buffer; the native guard must defer this to the pandas
    # fallback rather than build a mismatched tensor.
    mapper = MultiCategoricalTensorMapper(
        category_counts=[[1, 2], [2, 1], [3, 1]], sep=None)
    arr = pa.array([[1, 2], [2, 3], [1], [1, 1]], type=pa.list_(pa.int64()))
    sliced = arr.slice(0, 2)  # head slice: values buffer not trimmed
    ser = pd.Series([[1, 2], [2, 3]])
    _assert_same(mapper(ser), mapper(sliced))


def test_sequence_list_arrow_matches_pandas() -> None:
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0], None])
    arr = pa.array([[1.0, 2.0], [3.0, 4.0], None], type=pa.list_(pa.float64()))
    _assert_same(mapper(ser), mapper(arr))
    # LargeListArray takes the same native path.
    large = pa.array([[1.0, 2.0], [3.0, 4.0], None],
                     type=pa.large_list(pa.float64()))
    _assert_same(mapper(ser), mapper(large))


def test_sequence_truncation_arrow_matches_pandas() -> None:
    # Rows longer than ``max_length`` keep the prefix; shorter rows pad w/ NaN.
    mapper = SequenceTensorMapper(min_length=1, max_length=2)
    ser = pd.Series([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    arr = pa.array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]],
                   type=pa.list_(pa.float64()))
    _assert_same(mapper(ser), mapper(arr))


def test_sequence_sliced_list_arrow_matches_pandas() -> None:
    # Head slice: native guard must defer to the pandas fallback (see the
    # multicategorical sliced test for the underlying pyarrow behavior).
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    arr = pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                   type=pa.list_(pa.float64()))
    sliced = arr.slice(0, 2)
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0]])
    _assert_same(mapper(ser), mapper(sliced))
