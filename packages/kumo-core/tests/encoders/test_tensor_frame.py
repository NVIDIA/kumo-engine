# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrow tensor frames preserve per-column encoding and row counts."""

import pandas as pd
import pyarrow as pa
import pytest
from kumo_core.encoders.tensor_frame import combine_feats, get_tf
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
    UnsupportedTensorMapper,
)
from kumo_core.types.features import TensorData
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


def _assert_tensor_same(a: TensorData, b: TensorData) -> None:
    assert type(a) is type(b)
    if isinstance(a, (MultiNestedTensor, MultiEmbeddingTensor)):
        assert isinstance(b, (MultiNestedTensor, MultiEmbeddingTensor))
        assert a.offset.equal(b.offset)
        if a.values.is_floating_point():
            assert a.values.allclose(b.values, equal_nan=True)
        else:
            assert a.values.equal(b.values)
    else:
        assert isinstance(b, Tensor)
        assert a.shape == b.shape
        if a.is_floating_point():
            assert a.allclose(b, equal_nan=True)
        else:
            assert a.equal(b)


def _assert_tf_same(a: TensorFrame, b: TensorFrame) -> None:
    assert a.num_rows == b.num_rows
    assert a.col_names_dict == b.col_names_dict
    assert a.feat_dict.keys() == b.feat_dict.keys()
    for stype in a.feat_dict:
        _assert_tensor_same(a.feat_dict[stype], b.feat_dict[stype])


def _mapper_dict() -> dict[str, TensorMapper]:
    return {
        "num": NumericalTensorMapper(),
        "cat": CategoricalTensorMapper(category_counts=[["A", 2], ["B", 1]]),
        "multicat": MultiCategoricalTensorMapper(
            category_counts=[[1, 2], [2, 1], [3, 1]], sep=None
        ),
        "seq": SequenceTensorMapper(min_length=2, max_length=2),
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["A", "B", None, "C"],  # 'C' is out-of-vocab -> -1
            "multicat": [[1, 2], [2, 3], [1], [1, 1]],
            "seq": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        }
    )


@pytest.mark.parametrize("as_table", [False, True])
def test_get_tf_matches_column_mappers(as_table: bool) -> None:
    """Arrow frames match encoding each Arrow column separately."""
    df = _frame()
    mappers = _mapper_dict()
    expected = TensorFrame(
        combine_feats(
            {mapper.stype: [mapper(pa.array(df[name], from_pandas=True))] for name, mapper in mappers.items()}
        ),
        {mapper.stype: [name] for name, mapper in mappers.items()},
        num_rows=len(df),
    )
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    arrow = (
        pa.Table.from_batches([batch.slice(0, 2), batch.slice(2)])
        if as_table
        else batch
    )

    _assert_tf_same(get_tf(arrow, _mapper_dict()), expected)


@pytest.mark.parametrize("as_table", [False, True])
@pytest.mark.parametrize("num_rows", [0, 3])
@pytest.mark.parametrize("unsupported", [False, True])
def test_get_tf_without_features(
    as_table: bool,
    num_rows: int,
    unsupported: bool,
) -> None:
    """Empty feature schemas retain rows and skip absent unsupported columns."""
    batch = pa.record_batch({"unused": pa.nulls(num_rows)})
    arrow = pa.Table.from_batches([batch]) if as_table else batch
    mappers: dict[str, TensorMapper] = (
        {"absent": UnsupportedTensorMapper()} if unsupported else {}
    )

    result = get_tf(arrow, mappers)

    assert result.num_rows == num_rows
    assert result.feat_dict == {}
    assert result.col_names_dict == {}


@pytest.mark.parametrize("as_table", [False, True])
def test_get_tf_empty_features(as_table: bool) -> None:
    """An empty batch retains each supported feature's tensor shape."""
    batch = pa.record_batch({name: pa.nulls(0) for name in _mapper_dict()})
    arrow = pa.Table.from_batches([batch]) if as_table else batch

    result = get_tf(arrow, _mapper_dict())

    assert result.num_rows == 0
    assert result.col_names_dict == {
        mapper.stype: [name] for name, mapper in _mapper_dict().items()
    }


def test_get_tf_uses_mapper_column_order() -> None:
    """Multiple numerical columns follow mapper order rather than Arrow order."""
    batch = pa.record_batch({"first": [1.0], "second": [2.0]})
    mappers: dict[str, TensorMapper] = {
        "second": NumericalTensorMapper(),
        "first": NumericalTensorMapper(),
    }

    result = get_tf(batch, mappers)

    assert result.col_names_dict[Stype.numerical] == ["second", "first"]
    features = result.feat_dict[Stype.numerical]
    assert isinstance(features, Tensor)
    assert features.tolist() == [[2.0, 1.0]]
