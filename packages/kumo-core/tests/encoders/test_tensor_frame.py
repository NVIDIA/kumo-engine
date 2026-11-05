# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrow tensor frames preserve per-column encoding and row counts."""

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_frame import combine_feats, get_tf
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
    TextTensorMapper,
    TimestampTensorMapper,
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
            {
                mapper.stype: [mapper(pa.array(df[name], from_pandas=True))]
                for name, mapper in mappers.items()
            }
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


def test_get_tf_multiple_columns_per_stype() -> None:
    """Preserve grouped feature values across all supported semantic types."""
    mappers: dict[str, TensorMapper] = {}
    mappers["numerical_1"] = NumericalTensorMapper()
    mappers["numerical_2"] = NumericalTensorMapper()
    mappers["categorical_1"] = CategoricalTensorMapper(
        category_counts=[["A", 3], ["B", 2], ["C", 1]]
    )
    mappers["categorical_2"] = CategoricalTensorMapper(
        category_counts=[["A", 2], ["B", 1]]
    )
    # categorical_3 has been specified to use a Null encoder,
    # it will not be in the mapped `TensorFrame`
    mappers["categorical_3"] = UnsupportedTensorMapper()
    mappers["multicategorical_1"] = MultiCategoricalTensorMapper(
        category_counts=[["A", 3], ["B", 2], ["C", 1]], sep="|"
    )
    mappers["multicategorical_2"] = MultiCategoricalTensorMapper(
        category_counts=[["A", 2], ["B", 1]], sep="|"
    )
    # multicategorical_3 has been specified to use a Null encoder, it
    # will not be in the mapped `TensorFrame`
    mappers["multicategorical_3"] = UnsupportedTensorMapper()
    mappers["time_1"] = TimestampTensorMapper()
    mappers["time_2"] = TimestampTensorMapper()
    mappers["text_1"] = TextTensorMapper()
    mappers["text_2"] = TextTensorMapper()
    mappers["sequence_1"] = SequenceTensorMapper(1, 1)
    mappers["sequence_2"] = SequenceTensorMapper(2, 2)
    # sequence_3 has been specified to use a Null encoder, it will not be
    # in the mapped `TensorFrame`
    mappers["sequence_3"] = UnsupportedTensorMapper()

    # Test conversion to `TensorFrame`. In particular, we test that columns get
    # correctly stacked within each semantic type group:
    df = pd.DataFrame(
        {
            "numerical_1": [0.0, 2.0, 4.0, 8.0],
            "numerical_2": [1.0, 3.0, 5.0, 7.0],
            "categorical_1": ["A", "B", "C", "D"],
            "categorical_2": ["A", "B", "A", "B"],
            "categorical_3": ["A", "B", "A", "B"],
            "multicategorical_1": ["A", "B", "A|B", "A|B|C"],
            "multicategorical_2": ["A", "A|B", "C", "B"],
            "multicategorical_3": ["A", "A|B", "C", "B"],
            "time_1": ["1990-01-01", "1990-01-02", "1990-01-03", "1990-01-04"],
            "time_2": ["1991-01-01", "1991-01-02", "1991-01-03", "1991-01-04"],
            "text_1": ["hello", "a test", "welcome", "goodbye"],
            "text_2": ["this is a text", "wadjhd", "graph", "network"],
            "sequence_1": [[1.0], [2.0], [3.0], [4.0]],
            "sequence_2": ["[1,2]", "[3,4]", "[5,6]", "[7,8]"],
            "sequence_3": ["[1,2]", "[3,4]", "[5,6]", "[7,8]"],
        }
    )

    tf = get_tf(pa.Table.from_pandas(df, preserve_index=False), mappers)
    assert isinstance(tf, TensorFrame)
    feat_dict = tf.feat_dict
    col_names_dict = tf.col_names_dict

    assert len(feat_dict) == 6, "Should be equal to the number of stypes"
    assert col_names_dict[Stype.numerical.value] == ["numerical_1", "numerical_2"]
    feat = feat_dict[Stype.numerical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.allclose(
        feat,
        torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [8.0, 7.0],
            ]
        ),
    )

    assert col_names_dict[Stype.categorical.value] == ["categorical_1", "categorical_2"]
    feat = feat_dict[Stype.categorical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.equal(
        feat,
        torch.tensor(
            [
                [0, 0],
                [1, 1],
                [2, 0],
                [-1, 1],
            ]
        ),
    )

    assert col_names_dict[Stype.multicategorical.value] == [
        "multicategorical_1",
        "multicategorical_2",
    ]
    feat = feat_dict[Stype.multicategorical.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 2
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 2, 4, 7]))
    assert torch.equal(feat[:, 0].values, torch.tensor([0, 1, 0, 1, 0, 1, 2]))
    assert torch.equal(feat[:, 1].offset, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(feat[:, 1].values, torch.tensor([0, 0, 1, -1, 1]))

    assert col_names_dict[Stype.timestamp.value] == ["time_1", "time_2"]
    feat = feat_dict[Stype.timestamp.value]
    assert isinstance(feat, torch.Tensor)
    expected = torch.tensor(
        [
            [
                [0, 0, 1990, 0, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 631152000],
                [0, 0, 1991, 1, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 662688000],
            ],
            [
                [0, 0, 1990, 1, 1, 1, 31, 365, 53, 0, 0, 13, 1, 0, 7, 631238400],
                [0, 0, 1991, 2, 1, 1, 31, 365, 53, 0, 0, 19, 1, 0, 7, 662774400],
            ],
            [
                [0, 0, 1990, 2, 2, 2, 31, 365, 53, 0, 0, 12, 2, 0, 7, 631324800],
                [0, 0, 1991, 3, 2, 2, 31, 365, 53, 0, 0, 18, 2, 0, 7, 662860800],
            ],
            [
                [0, 0, 1990, 3, 3, 3, 31, 365, 53, 0, 0, 11, 3, 0, 7, 631411200],
                [0, 0, 1991, 4, 3, 3, 31, 365, 53, 0, 0, 17, 3, 0, 7, 662947200],
            ],
        ]
    )
    assert torch.equal(feat, expected)

    assert col_names_dict[Stype.text.value] == ["text_1", "text_2"]
    feat = feat_dict[Stype.text.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 2
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(
        feat[:, 0].values,
        torch.tensor([2, 4, 5, 6, 7]),
    )
    assert torch.equal(feat[:, 1].offset, torch.tensor([0, 4, 4, 5, 6]))
    assert torch.equal(
        feat[:, 1].values,
        torch.tensor([8, 9, 4, 10, 11, 12]),
    )

    assert col_names_dict[Stype.sequence.value] == ["sequence_1", "sequence_2"]
    feat = feat_dict[Stype.sequence.value]
    assert isinstance(feat, MultiEmbeddingTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 2
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1]))
    assert torch.allclose(
        feat[:, 0].values,
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    assert torch.equal(feat[:, 1].offset, torch.tensor([0, 2]))
    assert torch.allclose(
        feat[:, 1].values,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )

    # Test no Null columns are included in the `TensorFrame`:
    for col_names in col_names_dict.values():
        assert "categorical_3" not in col_names
        assert "multicategorical_3" not in col_names
        assert "sequence_3" not in col_names


def test_get_tf_single_column_per_stype() -> None:
    """Preserve grouped feature values across all supported semantic types."""
    mappers: dict[str, TensorMapper] = {}
    mappers["numerical_1"] = NumericalTensorMapper()
    mappers["categorical_1"] = CategoricalTensorMapper(
        category_counts=[["A", 3], ["B", 2], ["C", 1]]
    )
    mappers["multicategorical_1"] = MultiCategoricalTensorMapper(
        category_counts=[["A", 3], ["B", 2], ["C", 1]], sep="|"
    )
    mappers["time_1"] = TimestampTensorMapper()
    mappers["text_1"] = TextTensorMapper()
    mappers["sequence_1"] = SequenceTensorMapper(1, 1)

    # Test conversion to `TensorFrame`. In particular, we test that columns get
    # correctly stacked within each semantic type group:
    df = pd.DataFrame(
        {
            "numerical_1": [0.0, 2.0, 4.0, 8.0],
            "categorical_1": ["A", "B", "C", "D"],
            "multicategorical_1": ["A", "B", "A|B", "A|B|C"],
            "time_1": ["1990-01-01", "1990-01-02", "1990-01-03", "1990-01-04"],
            "text_1": ["hello", "a test", "welcome", "goodbye"],
            "sequence_1": [[1.0], [2.0], [3.0], [4.0]],
        }
    )

    tf = get_tf(pa.Table.from_pandas(df, preserve_index=False), mappers)
    assert isinstance(tf, TensorFrame)
    feat_dict = tf.feat_dict
    col_names_dict = tf.col_names_dict

    assert len(feat_dict) == 6, "Should be equal to the number of stypes"
    assert col_names_dict[Stype.numerical.value] == ["numerical_1"]
    feat = feat_dict[Stype.numerical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.allclose(
        feat,
        torch.tensor(
            [
                [0.0],
                [2.0],
                [4.0],
                [8.0],
            ]
        ),
    )

    assert col_names_dict[Stype.categorical.value] == ["categorical_1"]
    feat = feat_dict[Stype.categorical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.equal(
        feat,
        torch.tensor(
            [
                [0],
                [1],
                [2],
                [-1],
            ]
        ),
    )

    assert col_names_dict[Stype.multicategorical.value] == [
        "multicategorical_1",
    ]
    feat = feat_dict[Stype.multicategorical.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 2, 4, 7]))
    assert torch.equal(feat[:, 0].values, torch.tensor([0, 1, 0, 1, 0, 1, 2]))

    assert col_names_dict[Stype.timestamp.value] == ["time_1"]
    feat = feat_dict[Stype.timestamp.value]
    assert isinstance(feat, torch.Tensor)
    expected = torch.tensor(
        [
            [[0, 0, 1990, 0, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 631152000]],
            [[0, 0, 1990, 1, 1, 1, 31, 365, 53, 0, 0, 13, 1, 0, 7, 631238400]],
            [[0, 0, 1990, 2, 2, 2, 31, 365, 53, 0, 0, 12, 2, 0, 7, 631324800]],
            [[0, 0, 1990, 3, 3, 3, 31, 365, 53, 0, 0, 11, 3, 0, 7, 631411200]],
        ]
    )
    assert torch.equal(feat, expected)

    assert col_names_dict[Stype.text.value] == ["text_1"]
    feat = feat_dict[Stype.text.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(
        feat[:, 0].values,
        torch.tensor([2, 4, 5, 6, 7]),
    )

    assert col_names_dict[Stype.sequence.value] == ["sequence_1"]
    feat = feat_dict[Stype.sequence.value]
    assert isinstance(feat, MultiEmbeddingTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1]))
    assert torch.allclose(
        feat[:, 0].values,
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )


def test_get_tf_preserves_empty_lists() -> None:
    """Raw Arrow inputs retain null numerics and empty multicategorical rows."""
    mappers: dict[str, TensorMapper] = {
        "num": NumericalTensorMapper(),
        "multi": MultiCategoricalTensorMapper([[1, 2], [2, 1]], sep=None),
    }
    raw = {
        "num": pa.array([1.0, 2.0, None]),
        "multi": pa.array([[1], [], [2]], type=pa.list_(pa.int64())),
    }

    result = get_tf(pa.record_batch(raw), mappers)

    assert result.num_rows == 3
    assert result.feat_dict[Stype.numerical].flatten().tolist()[:2] == [1.0, 2.0]
    assert result.feat_dict[Stype.numerical][2].isnan().all()
    multi = result.feat_dict[Stype.multicategorical]
    assert isinstance(multi, MultiNestedTensor)
    assert multi.offset.tolist() == [0, 1, 1, 2]
    assert multi.values.tolist() == [0, 1]
