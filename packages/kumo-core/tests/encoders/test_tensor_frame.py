# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden test: ``get_tf`` builds an identical TensorFrame from a pandas
``DataFrame`` and the equivalent pyarrow ``RecordBatch`` / ``Table``.
"""
import pandas as pd
import pyarrow as pa
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, NumericalTensorMapper, SequenceTensorMapper
from kumo_core.encoders.tensor_frame import get_tf


def _assert_tensor_same(a, b) -> None:
    assert type(a) is type(b)
    if isinstance(a, (MultiNestedTensor, MultiEmbeddingTensor)):
        assert a.offset.equal(b.offset)
        if a.values.is_floating_point():
            assert a.values.allclose(b.values, equal_nan=True)
        else:
            assert a.values.equal(b.values)
    else:
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


def _mapper_dict() -> dict:
    return {
        'num':
        NumericalTensorMapper(),
        'cat':
        CategoricalTensorMapper(category_counts=[['A', 2], ['B', 1]]),
        'multicat':
        MultiCategoricalTensorMapper(category_counts=[[1, 2], [2, 1], [3, 1]],
                                     sep=None),
        'seq':
        SequenceTensorMapper(min_length=2, max_length=2),
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        'num': [1.0, 2.0, None, 4.0],
        'cat': ['A', 'B', None, 'C'],  # 'C' is out-of-vocab -> -1
        'multicat': [[1, 2], [2, 3], [1], [1, 1]],
        'seq': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    })


def test_get_tf_arrow_record_batch_matches_pandas() -> None:
    df = _frame()
    rb = pa.RecordBatch.from_pandas(df, preserve_index=False)
    _assert_tf_same(get_tf(df, _mapper_dict()), get_tf(rb, _mapper_dict()))


def test_get_tf_arrow_table_matches_pandas() -> None:
    df = _frame()
    table = pa.Table.from_pandas(df, preserve_index=False)
    _assert_tf_same(get_tf(df, _mapper_dict()), get_tf(table, _mapper_dict()))
