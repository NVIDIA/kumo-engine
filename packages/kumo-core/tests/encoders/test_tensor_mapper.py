# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Image,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_geometric.testing import withCUDA

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, ImageTensorMapper, MultiCategoricalTensorMapper, NumericalTensorMapper, SequenceTensorMapper, TensorMapper, TextTensorMapper, TimestampTensorMapper, UnsupportedTensorMapper


@withCUDA
def test_image_tensor_mapper(device: torch.device) -> None:
    mapper = ImageTensorMapper()
    assert str(mapper) == 'ImageTensorMapper()'

    ser = pd.Series([1, 2, 3])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.size() == (3, 0)


@withCUDA
def test_unsupported_tensor_mapper(device: torch.device) -> None:
    mapper = UnsupportedTensorMapper()
    assert str(mapper) == 'UnsupportedTensorMapper()'

    ser = pd.Series([1, 2, 3])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.size() == (3, 0)


@withCUDA
def test_unsupported_tensor_post_mapper(device: torch.device) -> None:
    mapper = UnsupportedTensorMapper()
    out = mapper.post_forward(pa.array([0, 0, 0]), device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.size() == (3, 0)

    with pytest.raises(RuntimeError, match="cannot map empty tensor"):
        mapper.backward(out)


@withCUDA
def test_numerical_tensor_mapper(device: torch.device) -> None:
    mapper = NumericalTensorMapper()
    assert str(mapper) == 'NumericalTensorMapper()'

    ser = pd.Series([1, 2, 3])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.equal(torch.tensor([[1.0], [2.0], [3.0]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out, check_dtype=False)

    ser = pd.Series([], dtype=float)
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.size() == (0, 1)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out, check_dtype=False)

    # Numerical tensor mappers support multi-dimensional tensors in backward:
    tensor = torch.tensor([[1, 2], [3, 4]], device=device)
    out = mapper.backward(tensor)
    pd.testing.assert_series_equal(pd.Series([[1, 2], [3, 4]]), out)


@withCUDA
def test_numerical_tensor_post_mapper(device: torch.device) -> None:
    mapper = NumericalTensorMapper()
    assert str(mapper) == 'NumericalTensorMapper()'
    arr = pa.array([1, 2, 3])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.equal(torch.tensor([[1.0], [2.0], [3.0]], device=device))

    arr = pa.array([])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.size() == (0, 1)


@withCUDA
def test_numerical_tensor_mapper_na(device: torch.device) -> None:
    mapper = NumericalTensorMapper()

    ser = pd.Series([1, float('NaN'), 3])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.isnan().equal(
        torch.tensor([[False], [True], [False]], device=device))
    assert out[~out.isnan()].equal(torch.tensor([1.0, 3.0], device=device))

    ser = pd.Series([1, None, 3], dtype=pd.Int64Dtype())
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    assert out.isnan().equal(
        torch.tensor([[False], [True], [False]], device=device))
    assert out[~out.isnan()].equal(torch.tensor([1.0, 3.0], device=device))


@withCUDA
def test_numerical_tensor_post_mapper_na(device: torch.device) -> None:
    mapper = NumericalTensorMapper()

    arr = pa.array([1, 2, float('NaN')])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.float
    # equal of NaNs is always false, so we compare separately
    assert out[:2, :].equal(torch.tensor([[1.0], [2.0]], device=device))
    assert out[2, 0].isnan().item()


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([['A', 2], ['B', 1]], id='ordered'),
    pytest.param([['B', 1], ['A', 2]], id='unordered'),
])
def test_categorical_tensor_mapper_str(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = CategoricalTensorMapper(category_counts=category_counts)
    assert str(mapper) == 'CategoricalTensorMapper()'

    ser = pd.Series(['A', 'B', 'C', None])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[0], [1], [-1], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series(['A', 'B', None, None]), out)

    ser = pd.Series([], dtype=str)
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, 1)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([[0, 2], [1, 1]], id='int_ordered'),
    pytest.param([[1, 1], [0, 2]], id='int_unordered'),
    pytest.param([[0.0, 2], [1.0, 1]], id='float_ordered'),
    pytest.param([[1.0, 1], [0.0, 2]], id='float_unordered'),
])
def test_categorical_tensor_mapper_number(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = CategoricalTensorMapper(category_counts=category_counts)
    assert str(mapper) == 'CategoricalTensorMapper()'

    ser = pd.Series([0, 1, 2])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[0], [1], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series([0, 1, None]), out)

    ser = pd.Series([0, 1, 2, None])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[0], [1], [-1], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series([0, 1, None, None]), out)

    ser = pd.Series([0, 1, 2, None], dtype=pd.Int64Dtype())
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[0], [1], [-1], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series([0, 1, None, None]), out)

    ser = pd.Series([], dtype=mapper.mapping.dtype)
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, 1)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([[False, 2], [True, 1]], id='ordered'),
    pytest.param([[True, 1], [False, 2]], id='unordered'),
])
def test_categorical_tensor_mapper_bool(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = CategoricalTensorMapper(category_counts=category_counts)
    assert str(mapper) == 'CategoricalTensorMapper()'

    ser = pd.Series([True, False, None], dtype=pd.BooleanDtype())
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[1], [0], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([True, False, None], dtype=pd.BooleanDtype()), out)

    ser = pd.Series([], dtype=pd.BooleanDtype())
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, 1)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_categorical_tensor_mapper_empty(device: torch.device) -> None:
    mapper = CategoricalTensorMapper(category_counts=[])

    ser = pd.Series(['a', None])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[-1], [-1]], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series([None, None]), out)


@withCUDA
def test_categorical_tensor_post_mapper(device: torch.device) -> None:
    mapper = CategoricalTensorMapper(category_counts=[['A', 2], ['B', 1]])

    arr = pa.array([0, 1, -1, -1])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.equal(torch.tensor([[0], [1], [-1], [-1]], device=device))

    arr = pa.array([])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, 1)


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([['A', 2], ['B', 1]], id='ordered'),
    pytest.param([['B', 1], ['A', 2]], id='unordered'),
])
def test_multi_categorical_tensor_mapper_sep(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = MultiCategoricalTensorMapper(category_counts=category_counts,
                                          sep='|')
    assert str(mapper) == 'MultiCategoricalTensorMapper(sep=|)'

    ser = pd.Series([None, 'A|B', 'B|C|', 'A', '', None])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 6
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(
        torch.tensor([-1, 0, 1, 1, -1, 0, -1], device=device))
    assert out.offset.equal(torch.tensor([0, 1, 3, 5, 6, 6, 7], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series(['', 'A|B', 'B', 'A', '', '']),
        out,
    )

    ser = pd.Series([], dtype=str)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([['A', 2], ['B', 1]], id='ordered'),
    pytest.param([['B', 1], ['A', 2]], id='unordered'),
])
def test_multi_categorical_tensor_mapper_no_sep(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = MultiCategoricalTensorMapper(category_counts=category_counts,
                                          sep=None)
    assert str(mapper) == 'MultiCategoricalTensorMapper(sep=None)'

    ser = pd.Series([None, 'A', 'B', 'C', '', None])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 6
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([-1, 0, 1, -1, -1], device=device))
    assert out.offset.equal(torch.tensor([0, 1, 2, 3, 4, 4, 5], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series(['', 'A', 'B', '', '', '']), out)

    ser = pd.Series([], dtype=str)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
@pytest.mark.parametrize('category_counts', [
    pytest.param([['A', 2], ['B', 1]], id='ordered'),
    pytest.param([['B', 1], ['A', 2]], id='unordered'),
])
def test_multi_categorical_tensor_mapper_list(
    device: torch.device,
    category_counts: list[list[Any]],
) -> None:
    mapper = MultiCategoricalTensorMapper(category_counts=category_counts,
                                          sep=None)
    assert str(mapper) == 'MultiCategoricalTensorMapper(sep=None)'

    ser = pd.Series([
        None,
        ['A', 'B'],
        ['B', 'C', '', None],
        ['A'],
        [''],
        [None],
        [],
        None,
        ['A', 'A'],
    ])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 9
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(
        torch.tensor([-1, 0, 1, 1, -1, -1, 0, -1, -1, 0, 0], device=device))
    assert out.offset.equal(
        torch.tensor([0, 1, 3, 6, 7, 7, 8, 8, 9, 11], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([[], ['A', 'B'], ['B'], ['A'], [], [], [], [], ['A', 'A']]),
        out,
    )

    ser = pd.Series([], dtype=object)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_multi_categorical_tensor_post_mapper(device: torch.device) -> None:
    mapper = MultiCategoricalTensorMapper(category_counts=[['B', 1], ['A', 2]],
                                          sep=None)
    assert str(mapper) == 'MultiCategoricalTensorMapper(sep=None)'

    arr = pa.array([[0, 1], [1, -1, -1], [0], [], [-1], [], [-1], [0, 0]])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 8
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(
        torch.tensor([0, 1, 1, -1, -1, 0, -1, -1, 0, 0], device=device))
    assert out.offset.equal(
        torch.tensor([0, 2, 5, 6, 6, 7, 7, 8, 10], device=device))

    arr = pa.array([])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))


@withCUDA
def test_multi_categorical_tensor_mapper_empty(device: torch.device) -> None:
    mapper = MultiCategoricalTensorMapper(category_counts=[], sep='|')

    ser = pd.Series(['a|b', None])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 2
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([-1, -1, -1], device=device))
    assert out.offset.equal(torch.tensor([0, 2, 3], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(pd.Series(['', '']), out)


@withCUDA
def test_sequence_tensor_mapper_list(device: torch.device) -> None:
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    assert str(mapper) == 'SequenceTensorMapper(min_length=2, max_length=2)'

    ser = pd.Series([None, [1, 2], [2, 3]])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 3
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.allclose(
        torch.tensor([
            [float('NaN'), float('NaN')],
            [1.0, 2.0],
            [2.0, 3.0],
        ], device=device), equal_nan=True)
    assert out.offset.equal(torch.tensor([0, 2], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([[float('NaN'), float('NaN')], [1.0, 2.0], [2.0, 3.0]]),
        out,
    )

    ser = pd.Series([], dtype=object)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.size() == (0, 2)
    assert out.offset.equal(torch.tensor([0, 2], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_nested_sequence_tensor_mapper_list(device: torch.device) -> None:
    mapper = SequenceTensorMapper(min_length=0, max_length=3)
    assert str(mapper) == 'SequenceTensorMapper(min_length=0, max_length=3)'

    ser = pd.Series([None, [1, 2, 3], [], [4, 5], [6, 7, 8, 9]])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 5
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.allclose(
        torch.tensor([
            [float('NaN'), float('NaN'),
             float('NaN')],
            [1.0, 2.0, 3.0],
            [float('NaN'), float('NaN'),
             float('NaN')],
            [4.0, 5.0, float('NaN')],
            [6.0, 7.0, 8.0],
        ], device=device), equal_nan=True)
    assert out.offset.equal(torch.tensor([0, 3], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([
            [float('NaN'), float('NaN'),
             float('NaN')],
            [1, 2, 3],
            [float('NaN'), float('NaN'),
             float('NaN')],
            [4, 5, float('NaN')],
            [6, 7, 8],
        ]), out)


@withCUDA
def test_sequence_tensor_mapper_str(device: torch.device) -> None:
    mapper = SequenceTensorMapper(min_length=2, max_length=2)
    assert str(mapper) == 'SequenceTensorMapper(min_length=2, max_length=2)'

    ser = pd.Series([None, '[1, 2]', '[2, 3]'])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 3
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.allclose(
        torch.tensor([
            [float('NaN'), float('NaN')],
            [1.0, 2.0],
            [2.0, 3.0],
        ], device=device), equal_nan=True)
    assert out.offset.equal(torch.tensor([0, 2], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series(['[nan, nan]', '[1.0, 2.0]', '[2.0, 3.0]']),
        out,
    )

    ser = pd.Series([], dtype=str)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.size() == (0, 2)
    assert out.offset.equal(torch.tensor([0, 2], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_nested_sequence_tensor_mapper_str(device: torch.device) -> None:
    mapper = SequenceTensorMapper(min_length=0, max_length=3)
    assert str(mapper) == 'SequenceTensorMapper(min_length=0, max_length=3)'

    ser = pd.Series([None, '[1, 2, 3]', '[]', '[4, 5]', '[6, 7, 8, 9]'])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 5
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.allclose(
        torch.tensor([
            [float('NaN'), float('NaN'),
             float('NaN')],
            [1.0, 2.0, 3.0],
            [float('NaN'), float('NaN'),
             float('NaN')],
            [4.0, 5.0, float('NaN')],
            [6.0, 7.0, 8.0],
        ], device=device), equal_nan=True)
    assert out.offset.equal(torch.tensor([0, 3], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([
            '[nan, nan, nan]',
            '[1.0, 2.0, 3.0]',
            '[nan, nan, nan]',
            '[4.0, 5.0, nan]',
            '[6.0, 7.0, 8.0]',
        ]), out)


@withCUDA
def test_nested_sequence_tensor_post_mapper(device: torch.device) -> None:
    mapper = SequenceTensorMapper(min_length=0, max_length=3)
    assert str(mapper) == 'SequenceTensorMapper(min_length=0, max_length=3)'

    arr = pa.array([[1.0, 2.0, 3.0], [], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 4
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.nan_to_num(-1.0).equal(
        torch.tensor([
            [1.0, 2.0, 3.0],
            [-1.0, -1.0, -1.0],
            [4.0, 5.0, -1.0],
            [6.0, 7.0, 8.0],
        ], device=device))
    assert out.offset.equal(torch.tensor([0, 3], device=device))

    mapper = SequenceTensorMapper(min_length=0, max_length=2)
    arr = pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 3
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.equal(
        torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device))
    assert out.offset.equal(torch.tensor([0, 2], device=device))

    arr = pa.array([])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiEmbeddingTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.float
    assert out.values.equal(torch.empty(0, 2, device=device))
    assert out.offset.equal(torch.tensor([0, 2], device=device))


@withCUDA
def test_timestamp_tensor_mapper_datetime(device: torch.device) -> None:
    mapper = TimestampTensorMapper()
    assert str(mapper) == 'TimestampTensorMapper()'

    ser = pd.Series([
        pd.Timestamp('2018-01-05'),
        None,
        pd.Timestamp.min,
        pd.Timestamp.max,
    ])
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    out = out[:, :mapper.index['unix']]
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (4, 15)
    assert out[0].tolist() == [
        0, 0, 2018, 4, 4, 4, 31, 365, 53, 0, 0, 10, 4, 0, 7
    ]
    assert out[1].tolist() == [-1] * 15
    assert out[2].tolist() == [
        0, 0, 1678, 5, 0, 0, 31, 365, 53, 0, 0, 415, -415, 10, 10
    ]
    assert out[3].tolist() == [
        47, 23, 2262, 4, 10, 100, 30, 365, 53, 3, 14, -505, 505, 8, 8
    ]

    out = mapper.backward(out)
    # We don't mind if extreme values get mildly malformed
    pd.testing.assert_series_equal(ser[:2], out[:2])

    ser = pd.Series([], dtype='datetime64[ns]')
    out = mapper(ser, device=device)
    out = out[:, :mapper.index['unix']]
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, 15)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_timestamp_tensor_mapper_str(device: torch.device) -> None:
    mapper = TimestampTensorMapper()
    assert str(mapper) == 'TimestampTensorMapper()'

    ser = pd.Series(['2018-01-05', ''])
    out = mapper(ser, device=device)
    out = out[:, :mapper.index['unix']]
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (2, 15)
    assert out[0].tolist() == [
        0, 0, 2018, 4, 4, 4, 31, 365, 53, 0, 0, 10, 4, 0, 7
    ]
    assert out[1].tolist() == [-1] * 15

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([pd.Timestamp('2018-01-05'), None]),
        out,
    )


@withCUDA
def test_timestamp_tensor_mapper_holidays(device: torch.device) -> None:
    mapper = TimestampTensorMapper()

    ser = pd.Series([
        pd.Timestamp('1960-01-01'),
        pd.Timestamp('1969-12-31'),
        pd.Timestamp('1970-01-01'),  # Holiday.
        pd.Timestamp('1970-01-02'),
        pd.Timestamp('2005-12-31'),
        pd.Timestamp('2006-01-01'),
        pd.Timestamp('2006-01-02'),  # Holiday falls on a Monday
        pd.Timestamp('1970-11-09'),
        pd.Timestamp('1970-11-11'),  # Holiday.
        pd.Timestamp('1970-11-14'),
        pd.Timestamp('2042-12-31'),
        pd.Timestamp('2043-01-01'),  # Holiday.
        pd.Timestamp('2043-01-03'),
        pd.Timestamp('2055-01-01'),
    ])
    out = mapper(ser, device=device)
    out = out[:, :mapper.index['unix']]
    assert out[:, -4:].tolist() == [
        [0, 0, 7, 7],
        [1, 6, 7, 1],
        [0, 0, 7, 7],
        [45, 1, 10, 7],
        [2, 5, 7, 1],
        [1, 6, 7, 1],
        [0, 0, 7, 7],
        [2, 28, 9, 2],
        [0, 0, 9, 9],
        [12, 3, 8, 9],
        [1, 6, 7, 1],
        [0, 0, 7, 7],
        [16, 2, 0, 7],
        [0, 0, 7, 7],
    ]


@withCUDA
def test_timestamp_tensor_post_mapper_datetime(device: torch.device) -> None:
    mapper = TimestampTensorMapper()
    assert str(mapper) == 'TimestampTensorMapper()'

    num_elems = len(TimestampTensorMapper.index)
    arr = pa.array([[0] * num_elems, [-1] * num_elems])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (2, num_elems)
    assert out[0].tolist() == [0] * num_elems
    assert out[1].tolist() == [-1] * num_elems

    ser = pd.Series([], dtype='datetime64[ns]')
    out = mapper(ser, device=device)
    assert isinstance(out, Tensor)
    assert out.device == device
    assert out.dtype == torch.long
    assert out.size() == (0, num_elems)


@withCUDA
def test_glove_tensor_mapper(device: torch.device) -> None:
    mapper = TextTensorMapper()
    assert str(mapper) == 'TextTensorMapper()'

    # TODO Tokenizer cannot handle multi-line sentences for now.
    ser = pd.Series(
        ['invalid_token', 'get', 'Get old', 'get\nold', 'get,old', '', None])
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 7
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(
        torch.tensor([0, 0, 1, 0, 1, 0, 1], device=device))
    assert out.offset.equal(
        torch.tensor([0, 0, 1, 3, 5, 7, 7, 7], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(
        pd.Series([None, 'get', 'get old', 'get old', 'get old', None, None]),
        out)

    ser = pd.Series([], dtype=str)
    out = mapper(ser, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))

    out = mapper.backward(out)
    pd.testing.assert_series_equal(ser, out)


@withCUDA
def test_glove_tensor_post_mapper(device: torch.device) -> None:
    mapper = TextTensorMapper()
    assert str(mapper) == 'TextTensorMapper()'

    arr = pa.array([[0, 1], [1, -1, -1], [0], [], [-1], [], [-1], [0, 0]])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 8
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(
        torch.tensor([0, 1, 1, -1, -1, 0, -1, -1, 0, 0], device=device))
    assert out.offset.equal(
        torch.tensor([0, 2, 5, 6, 6, 7, 7, 8, 10], device=device))

    arr = pa.array([])
    out = mapper.post_forward(arr, device=device)
    assert isinstance(out, MultiNestedTensor)
    assert out.num_rows == 0
    assert out.num_cols == 1
    assert out.device == device
    assert out.dtype == torch.long
    assert out.values.equal(torch.tensor([], dtype=torch.long, device=device))
    assert out.offset.equal(torch.tensor([0], device=device))

    texts = [
        'invalid_token', 'get', 'Get old', 'get\nold', 'get,old', '', None
    ]
    out_post_forward = mapper.post_forward(pa.array(texts), device=device)
    out_forward = mapper(pd.Series(texts), device=device)
    assert out_post_forward.values.equal(out_forward.values)
    assert out_post_forward.offset.equal(out_forward.offset)


class UnsupportedEncoder(Encoder):
    @property
    def required_stats(self) -> set[ColStatType]:
        return set()

    @property
    def supported_stypes(self) -> set[Stype]:
        return set()


def test_from_encoder() -> None:
    tensor_mapper = TensorMapper.from_encoder(Null())
    assert isinstance(tensor_mapper, UnsupportedTensorMapper)

    tensor_mapper = TensorMapper.from_encoder(Image())
    assert isinstance(tensor_mapper, ImageTensorMapper)

    tensor_mapper = TensorMapper.from_encoder(Numerical())
    assert isinstance(tensor_mapper, NumericalTensorMapper)

    tensor_mapper = TensorMapper.from_encoder(MaxLogNumerical())
    assert isinstance(tensor_mapper, NumericalTensorMapper)

    tensor_mapper = TensorMapper.from_encoder(MinLogNumerical())
    assert isinstance(tensor_mapper, NumericalTensorMapper)

    encoder = MultiCategorical()
    encoder._stats = {
        ColStatType.MULTI_CATEGORY_COUNTS: [['A', 2], ['B', 1]],
        ColStatType.MULTI_CATEGORIES_SEPARATOR: None,
    }
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, MultiCategoricalTensorMapper)
    assert tensor_mapper.mapping.tolist() == ['A', 'B']

    encoder = Index()
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS: [['A', 2], ['B', 1]],
    }
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, CategoricalTensorMapper)
    assert tensor_mapper.mapping.tolist() == ['A', 'B']

    encoder = Hash(num_components=1)
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS: [['A', 2], ['B', 1]],
    }
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, CategoricalTensorMapper)
    assert tensor_mapper.mapping.tolist() == ['A', 'B']

    encoder = NumericalList()
    encoder._stats = {
        ColStatType.SEQUENCE_MIN_LENGTH: 2,
        ColStatType.SEQUENCE_MAX_LENGTH: 2,
    }
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, SequenceTensorMapper)
    assert tensor_mapper.min_length == 2
    assert tensor_mapper.max_length == 2

    encoder = Datetime()
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, TimestampTensorMapper)

    encoder = TextEmbedding('glove.test-10d')
    tensor_mapper = TensorMapper.from_encoder(encoder)
    assert isinstance(tensor_mapper, TextTensorMapper)

    with pytest.raises(ValueError, match='Unsupported encoder'):
        TensorMapper.from_encoder(UnsupportedEncoder())
