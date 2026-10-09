# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest
import torch
from kumoapi.typing import Stype
from torch_frame.data import (
    MultiEmbeddingTensor,
    MultiNestedTensor,
    TensorFrame,
)

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, NumericalTensorMapper, SequenceTensorMapper, TextTensorMapper, TimestampTensorMapper, UnsupportedTensorMapper
from kumo_core.encoders.tensor_mappers import TensorMappers


def test_tensor_mappers_with_only_null_encoders() -> None:
    store = TensorMappers()
    store['TABLE']['col1'] = UnsupportedTensorMapper()
    store['TABLE']['col2'] = UnsupportedTensorMapper()
    df = pd.DataFrame({
        'col1': ['A', 'B', 'C', 'D'],
        'col2': ['A', 'B', 'A|B', 'A|B|C'],
    })
    tf = store.convert_to_tensor_frame('TABLE', df)
    assert isinstance(tf, TensorFrame)
    assert tf.is_empty


# TODO(akihironitta): Use stype instead of str once PyTorch
# addresses https://github.com/pytorch/pytorch/issues/111603
def test_tensor_mappers() -> None:
    store = TensorMappers()
    store['TABLE']['numerical_1'] = NumericalTensorMapper()
    store['TABLE']['numerical_2'] = NumericalTensorMapper()
    store['TABLE']['categorical_1'] = CategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]])
    store['TABLE']['categorical_2'] = CategoricalTensorMapper(
        category_counts=[['A', 2], ['B', 1]])
    # categorical_3 has been specified to use a Null encoder,
    # it will not be in the mapped `TensorFrame`
    store['TABLE']['categorical_3'] = UnsupportedTensorMapper()
    store['TABLE']['multicategorical_1'] = MultiCategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]], sep='|')
    store['TABLE']['multicategorical_2'] = MultiCategoricalTensorMapper(
        category_counts=[['A', 2], ['B', 1]], sep='|')
    # multicategorical_3 has been specified to use a Null encoder, it
    # will not be in the mapped `TensorFrame`
    store['TABLE']['multicategorical_3'] = UnsupportedTensorMapper()
    store['TABLE']['time_1'] = TimestampTensorMapper()
    store['TABLE']['time_2'] = TimestampTensorMapper()
    store['TABLE']['text_1'] = TextTensorMapper()
    store['TABLE']['text_2'] = TextTensorMapper()
    store['TABLE']['sequence_1'] = SequenceTensorMapper(1, 1)
    store['TABLE']['sequence_2'] = SequenceTensorMapper(2, 2)
    # sequence_3 has been specified to use a Null encoder, it will not be
    # in the mapped `TensorFrame`
    store['TABLE']['sequence_3'] = UnsupportedTensorMapper()

    # Test conversion to `TensorFrame`. In particular, we test that columns get
    # correctly stacked within each semantic type group:
    df = pd.DataFrame({
        'numerical_1': [0., 2., 4., 8.],
        'numerical_2': [1., 3., 5., 7.],
        'categorical_1': ['A', 'B', 'C', 'D'],
        'categorical_2': ['A', 'B', 'A', 'B'],
        'categorical_3': ['A', 'B', 'A', 'B'],
        'multicategorical_1': ['A', 'B', 'A|B', 'A|B|C'],
        'multicategorical_2': ['A', 'A|B', 'C', 'B'],
        'multicategorical_3': ['A', 'A|B', 'C', 'B'],
        'time_1': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'],
        'time_2': ['1991-01-01', '1991-01-02', '1991-01-03', '1991-01-04'],
        'text_1': ['hello', 'a test', 'welcome', 'goodbye'],
        'text_2': ['this is a text', 'wadjhd', 'graph', 'network'],
        'sequence_1': [[1.0], [2.0], [3.0], [4.0]],
        'sequence_2': ['[1,2]', '[3,4]', '[5,6]', '[7,8]'],
        'sequence_3': ['[1,2]', '[3,4]', '[5,6]', '[7,8]'],
    })

    tf = store.convert_to_tensor_frame('TABLE', df)
    assert isinstance(tf, TensorFrame)
    feat_dict = tf.feat_dict
    col_names_dict = tf.col_names_dict

    assert len(feat_dict) == 6, "Should be equal to the number of stypes"
    assert (col_names_dict[Stype.numerical.value] == [
        'numerical_1', 'numerical_2'
    ])
    feat = feat_dict[Stype.numerical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.allclose(
        feat,
        torch.tensor([
            [0., 1.],
            [2., 3.],
            [4., 5.],
            [8., 7.],
        ]),
    )

    assert col_names_dict[Stype.categorical.value] == [
        'categorical_1', 'categorical_2'
    ]
    feat = feat_dict[Stype.categorical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.equal(
        feat,
        torch.tensor([
            [0, 0],
            [1, 1],
            [2, 0],
            [-1, 1],
        ]),
    )

    assert col_names_dict[Stype.multicategorical.value] == [
        'multicategorical_1', 'multicategorical_2'
    ]
    feat = feat_dict[Stype.multicategorical.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 2
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 2, 4, 7]))
    assert torch.equal(feat[:, 0].values, torch.tensor([0, 1, 0, 1, 0, 1, 2]))
    assert torch.equal(feat[:, 1].offset, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(feat[:, 1].values, torch.tensor([0, 0, 1, -1, 1]))

    assert col_names_dict[Stype.timestamp.value] == ['time_1', 'time_2']
    feat = feat_dict[Stype.timestamp.value]
    assert isinstance(feat, torch.Tensor)
    # minute/hour/year/day_of_week/day_of_month/day_of_year/days_per_month/days_per_year/weeks_per_year/month/week_of_year/unix_seconds  # noqa: E501
    expected = torch.tensor([
        [[0, 0, 1990, 0, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 631152000],
         [0, 0, 1991, 1, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 662688000]],
        [[0, 0, 1990, 1, 1, 1, 31, 365, 53, 0, 0, 13, 1, 0, 7, 631238400],
         [0, 0, 1991, 2, 1, 1, 31, 365, 53, 0, 0, 19, 1, 0, 7, 662774400]],
        [[0, 0, 1990, 2, 2, 2, 31, 365, 53, 0, 0, 12, 2, 0, 7, 631324800],
         [0, 0, 1991, 3, 2, 2, 31, 365, 53, 0, 0, 18, 2, 0, 7, 662860800]],
        [[0, 0, 1990, 3, 3, 3, 31, 365, 53, 0, 0, 11, 3, 0, 7, 631411200],
         [0, 0, 1991, 4, 3, 3, 31, 365, 53, 0, 0, 17, 3, 0, 7, 662947200]],
    ])
    assert torch.equal(feat, expected)

    assert col_names_dict[Stype.text.value] == ['text_1', 'text_2']
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

    assert col_names_dict[Stype.sequence.value] == ['sequence_1', 'sequence_2']
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
        assert 'categorical_3' not in col_names
        assert 'multicategorical_3' not in col_names
        assert 'sequence_3' not in col_names


# TODO(akihironitta): Use stype instead of str once PyTorch
# addresses https://github.com/pytorch/pytorch/issues/111603
def test_null_encoders_with_mulitple_tables() -> None:
    """This test makes sure that TensorMappers can distinguish the same column
    names in different tables.
    """
    store = TensorMappers()
    store['TABLE1']['numerical'] = NumericalTensorMapper()
    store['TABLE1']['categorical'] = UnsupportedTensorMapper()
    store['TABLE2']['numerical'] = UnsupportedTensorMapper()
    store['TABLE2']['categorical'] = CategoricalTensorMapper(
        category_counts=[['A', 2], ['B', 1]])

    # Test conversion to `TensorFrame`. In particular, we test that columns get
    # correctly stacked within each semantic type group:
    df1 = pd.DataFrame({
        'numerical': [0., 2., 4., 8.],
        'categorical': ['A', 'B', 'C', 'A'],
    })
    df2 = pd.DataFrame({
        'numerical': [0., 2., 4., 8.],
        'categorical': ['B', 'B', 'A', 'A'],
    })

    tf1 = store.convert_to_tensor_frame('TABLE1', df1)
    tf2 = store.convert_to_tensor_frame('TABLE2', df2)
    assert isinstance(tf1, TensorFrame)
    assert isinstance(tf2, TensorFrame)
    # Test no Null columns are included in the `TensorFrame`:
    assert (Stype.numerical.name in tf1.feat_dict
            and Stype.categorical.name not in tf1.feat_dict)
    assert (Stype.numerical.name not in tf2.feat_dict
            and Stype.categorical.name in tf2.feat_dict)


# TODO(akihironitta): Use stype instead of str once PyTorch
# addresses https://github.com/pytorch/pytorch/issues/111603
def test_tensor_mappers_single_column_per_stype() -> None:
    store = TensorMappers()
    store['TABLE']['numerical_1'] = NumericalTensorMapper()
    store['TABLE']['categorical_1'] = CategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]])
    store['TABLE']['multicategorical_1'] = MultiCategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]], sep='|')
    store['TABLE']['time_1'] = TimestampTensorMapper()
    store['TABLE']['text_1'] = TextTensorMapper()
    store['TABLE']['sequence_1'] = SequenceTensorMapper(1, 1)

    # Test conversion to `TensorFrame`. In particular, we test that columns get
    # correctly stacked within each semantic type group:
    df = pd.DataFrame({
        'numerical_1': [0., 2., 4., 8.],
        'categorical_1': ['A', 'B', 'C', 'D'],
        'multicategorical_1': ['A', 'B', 'A|B', 'A|B|C'],
        'time_1': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'],
        'text_1': ['hello', 'a test', 'welcome', 'goodbye'],
        'sequence_1': [[1.0], [2.0], [3.0], [4.0]],
    })

    tf = store.convert_to_tensor_frame('TABLE', df)
    assert isinstance(tf, TensorFrame)
    feat_dict = tf.feat_dict
    col_names_dict = tf.col_names_dict

    assert len(feat_dict) == 6, "Should be equal to the number of stypes"
    assert col_names_dict[Stype.numerical.value] == ['numerical_1']
    feat = feat_dict[Stype.numerical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.allclose(
        feat,
        torch.tensor([
            [0.],
            [2.],
            [4.],
            [8.],
        ]),
    )

    assert col_names_dict[Stype.categorical.value] == ['categorical_1']
    feat = feat_dict[Stype.categorical.value]
    assert isinstance(feat, torch.Tensor)
    assert torch.equal(
        feat,
        torch.tensor([
            [0],
            [1],
            [2],
            [-1],
        ]),
    )

    assert col_names_dict[Stype.multicategorical.value] == [
        'multicategorical_1',
    ]
    feat = feat_dict[Stype.multicategorical.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 2, 4, 7]))
    assert torch.equal(feat[:, 0].values, torch.tensor([0, 1, 0, 1, 0, 1, 2]))

    assert col_names_dict[Stype.timestamp.value] == ['time_1']
    feat = feat_dict[Stype.timestamp.value]
    assert isinstance(feat, torch.Tensor)
    # minute/hour/year/day_of_week/day_of_month/day_of_year/days_per_month/days_per_year/weeks_per_year/month/week_of_year/unix_seconds  # noqa: E501
    expected = torch.tensor([
        [[0, 0, 1990, 0, 0, 0, 31, 365, 53, 0, 0, 0, 0, 7, 7, 631152000]],
        [[0, 0, 1990, 1, 1, 1, 31, 365, 53, 0, 0, 13, 1, 0, 7, 631238400]],
        [[0, 0, 1990, 2, 2, 2, 31, 365, 53, 0, 0, 12, 2, 0, 7, 631324800]],
        [[0, 0, 1990, 3, 3, 3, 31, 365, 53, 0, 0, 11, 3, 0, 7, 631411200]],
    ])
    assert torch.equal(feat, expected)

    assert col_names_dict[Stype.text.value] == ['text_1']
    feat = feat_dict[Stype.text.value]
    assert isinstance(feat, MultiNestedTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(
        feat[:, 0].values,
        torch.tensor([2, 4, 5, 6, 7]),
    )

    assert col_names_dict[Stype.sequence.value] == ['sequence_1']
    feat = feat_dict[Stype.sequence.value]
    assert isinstance(feat, MultiEmbeddingTensor)
    assert feat.num_rows == len(feat) == 4
    assert feat.num_cols == 1
    assert torch.equal(feat[:, 0].offset, torch.tensor([0, 1]))
    assert torch.allclose(
        feat[:, 0].values,
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )


def _build_multi_stype_store_and_df() -> tuple[TensorMappers, pd.DataFrame]:
    """Builds the same multi-stype fixture used by ``test_tensor_mappers``."""
    store = TensorMappers()
    store['TABLE']['numerical_1'] = NumericalTensorMapper()
    store['TABLE']['numerical_2'] = NumericalTensorMapper()
    store['TABLE']['categorical_1'] = CategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]])
    store['TABLE']['categorical_2'] = CategoricalTensorMapper(
        category_counts=[['A', 2], ['B', 1]])
    store['TABLE']['multicategorical_1'] = MultiCategoricalTensorMapper(
        category_counts=[['A', 3], ['B', 2], ['C', 1]], sep='|')
    store['TABLE']['multicategorical_2'] = MultiCategoricalTensorMapper(
        category_counts=[['A', 2], ['B', 1]], sep='|')
    store['TABLE']['time_1'] = TimestampTensorMapper()
    store['TABLE']['time_2'] = TimestampTensorMapper()
    store['TABLE']['text_1'] = TextTensorMapper()
    store['TABLE']['text_2'] = TextTensorMapper()
    store['TABLE']['sequence_1'] = SequenceTensorMapper(1, 1)
    store['TABLE']['sequence_2'] = SequenceTensorMapper(2, 2)

    # Include None / "" / unknown-category values so the vectorized path
    # gets exercised on the same missing/unknown branches as the baseline:
    # NaN in numerical, "" and unseen 'D' in categorical.
    df = pd.DataFrame({
        'numerical_1': [0., None, 4., 8.],
        'numerical_2': [1., 3., 5., 7.],
        'categorical_1': ['A', 'B', 'C', 'D'],
        'categorical_2': ['A', '', 'A', 'B'],
        'multicategorical_1': ['A', 'B', 'A|B', 'A|B|C'],
        'multicategorical_2': ['A', 'A|B', 'C', 'B'],
        'time_1': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'],
        'time_2': ['1991-01-01', '1991-01-02', '1991-01-03', '1991-01-04'],
        'text_1': ['hello', 'a test', 'welcome', 'goodbye'],
        'text_2': ['this is a text', 'wadjhd', 'graph', 'network'],
        'sequence_1': [[1.0], [2.0], [3.0], [4.0]],
        'sequence_2': ['[1,2]', '[3,4]', '[5,6]', '[7,8]'],
    })
    return store, df


def test_convert_to_tensor_frame_defers_per_column_h2d() -> None:
    """``_to_tf`` should call each ``mapper.forward`` with ``device=None`` and
    issue a single transfer per stype after concatenation, rather than one
    transfer per column.
    """
    store, df = _build_multi_stype_store_and_df()

    forwarded_devices: list[torch.device | None] = []

    # Wrap each mapper instance's bound ``forward`` to capture the device the
    # caller passes through. Patching at the base class wouldn't work because
    # every concrete TensorMapper overrides ``forward``.
    for mapper in store['TABLE'].values():
        real = mapper.forward

        def _spy(ser, *, device=None, _real=real):
            forwarded_devices.append(device)
            return _real(ser, device=device)

        mapper.forward = _spy  # type: ignore[method-assign]

    target = torch.device('cpu')
    tf = store.convert_to_tensor_frame('TABLE', df, device=target)

    # All per-column forwards must have been deferred (device=None), regardless
    # of the device the caller asked for. This is the load-bearing behavior
    # change: per-column transfers are no longer issued.
    assert len(forwarded_devices) > 0
    assert all(d is None for d in forwarded_devices), forwarded_devices

    # Per-stype tensors must end up on the requested device (i.e., the single
    # post-cat transfer actually happened).
    for stype, feat in tf.feat_dict.items():
        if isinstance(feat, torch.Tensor):
            assert feat.device == target, stype
        elif isinstance(feat, (MultiNestedTensor, MultiEmbeddingTensor)):
            assert feat.values.device == target, stype
            assert feat.offset.device == target, stype
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected feat type for stype={stype}")


def test_convert_to_tensor_frame_device_none_matches_explicit_cpu() -> None:
    """Calling with ``device=None`` and ``device=torch.device('cpu')`` must
    produce numerically identical TensorFrames. Guards against the deferred
    transfer changing values.
    """
    store, df = _build_multi_stype_store_and_df()

    tf_default = store.convert_to_tensor_frame('TABLE', df, device=None)
    tf_cpu = store.convert_to_tensor_frame(
        'TABLE',
        df,
        device=torch.device('cpu'),
    )

    assert set(tf_default.feat_dict.keys()) == set(tf_cpu.feat_dict.keys())
    for stype in tf_default.feat_dict:
        a = tf_default.feat_dict[stype]
        b = tf_cpu.feat_dict[stype]
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype
            torch.testing.assert_close(a, b, equal_nan=True)
        else:
            assert type(a) is type(b)
            torch.testing.assert_close(a.values, b.values, equal_nan=True)
            assert torch.equal(a.offset, b.offset)


def test_convert_to_tensor_frame_numpy_dict_matches_dataframe() -> None:
    """``convert_to_tensor_frame(..., vectorized=True)`` over a
    ``dict[str, np.ndarray]`` must be numerically identical to the pandas
    path across every stype.
    """
    store, df = _build_multi_stype_store_and_df()

    np_dict = {col: df[col].to_numpy() for col in df.columns}

    tf_pd = store.convert_to_tensor_frame('TABLE', df)
    tf_np = store.convert_to_tensor_frame('TABLE', np_dict, vectorized=True)

    assert set(tf_pd.feat_dict.keys()) == set(tf_np.feat_dict.keys())
    assert tf_pd.col_names_dict == tf_np.col_names_dict
    assert tf_pd.num_rows == tf_np.num_rows
    for stype in tf_pd.feat_dict:
        a = tf_pd.feat_dict[stype]
        b = tf_np.feat_dict[stype]
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype
            torch.testing.assert_close(a, b, equal_nan=True,
                                       msg=f"stype={stype}")
        else:
            assert type(a) is type(b)
            torch.testing.assert_close(a.values, b.values, equal_nan=True,
                                       msg=str(stype))
            assert torch.equal(a.offset, b.offset), stype


def test_to_tf_vectorized_matches_to_tf() -> None:
    """``_to_tf_vectorized`` must produce a TensorFrame numerically identical
    to ``_to_tf`` on the same numpy-dict input, across every stype in the
    multi-stype fixture. Guards against drift between the two implementations
    when one is touched in isolation.
    """
    from kumo_core.encoders.tensor_mappers import _to_tf, _to_tf_vectorized

    store, df = _build_multi_stype_store_and_df()
    np_dict = {col: df[col].to_numpy() for col in df.columns}

    tf_baseline = _to_tf(df=np_dict, mappers=store['TABLE'])
    tf_vec = _to_tf_vectorized(df=np_dict, mappers=store['TABLE'])

    assert set(tf_baseline.feat_dict.keys()) == set(tf_vec.feat_dict.keys())
    assert tf_baseline.col_names_dict == tf_vec.col_names_dict
    assert tf_baseline.num_rows == tf_vec.num_rows
    for stype in tf_baseline.feat_dict:
        a = tf_baseline.feat_dict[stype]
        b = tf_vec.feat_dict[stype]
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype, stype
            torch.testing.assert_close(a, b, equal_nan=True,
                                       msg=f"stype={stype}")
        else:
            assert type(a) is type(b)
            torch.testing.assert_close(a.values, b.values, equal_nan=True,
                                       msg=str(stype))
            assert torch.equal(a.offset, b.offset), stype


def test_to_tf_vectorized_empty_input() -> None:
    """``_to_tf_vectorized`` handles a zero-row NumPy batch without raising."""
    import numpy as np

    from kumo_core.encoders.tensor_mappers import _to_tf_vectorized

    store, _ = _build_multi_stype_store_and_df()
    empty_dict = {
        'numerical_1': np.array([], dtype=np.float64),
        'numerical_2': np.array([], dtype=np.float64),
        'categorical_1': np.array([], dtype=object),
        'categorical_2': np.array([], dtype=object),
        'multicategorical_1': np.array([], dtype=object),
        'multicategorical_2': np.array([], dtype=object),
        'time_1': np.array([], dtype='datetime64[ns]'),
        'time_2': np.array([], dtype='datetime64[ns]'),
        'text_1': np.array([], dtype=object),
        'text_2': np.array([], dtype=object),
        'sequence_1': np.array([], dtype=object),
        'sequence_2': np.array([], dtype=object),
    }
    tf = _to_tf_vectorized(df=empty_dict, mappers=store['TABLE'])
    assert tf.num_rows == 0
    # Numerical / categorical stypes produce (0, num_cols) tensors.
    feat_num = tf.feat_dict[Stype.numerical.value]
    assert isinstance(feat_num, torch.Tensor)
    assert feat_num.shape == (0, 2)
    feat_cat = tf.feat_dict[Stype.categorical.value]
    assert isinstance(feat_cat, torch.Tensor)
    assert feat_cat.shape == (0, 2)


def test_convert_to_tensor_frame_routes_to_vectorized(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``convert_to_tensor_frame`` must dispatch on the ``vectorized`` flag:
    ``vectorized=True`` calls :func:`_to_tf_vectorized`, the default calls
    :func:`_to_tf`. We patch both module-level functions and inspect which
    one ran.
    """
    import kumo_core.encoders.tensor_mappers as tm_module

    store, df = _build_multi_stype_store_and_df()
    np_dict = {col: df[col].to_numpy() for col in df.columns}

    real_baseline = tm_module._to_tf
    real_vectorized = tm_module._to_tf_vectorized
    calls: list[str] = []

    def spy_baseline(*args, **kwargs):
        calls.append('baseline')
        return real_baseline(*args, **kwargs)

    def spy_vectorized(*args, **kwargs):
        calls.append('vectorized')
        return real_vectorized(*args, **kwargs)

    monkeypatch.setattr(tm_module, '_to_tf', spy_baseline)
    monkeypatch.setattr(tm_module, '_to_tf_vectorized', spy_vectorized)
    store.convert_to_tensor_frame('TABLE', np_dict, vectorized=True)
    store.convert_to_tensor_frame('TABLE', df)

    assert calls == ['vectorized', 'baseline'], calls


@pytest.mark.parametrize('cats', [
    ['A', 'B', 'C'],
    pd.Index(['A', 'B', 'C']),
    pd.Index(['A', 'B', 'C'], dtype='string'),
])
def test_mapper_map_numpy_matches_series(cats) -> None:
    """``Mapper.map`` numpy fast path must produce the same int64 codes as
    the pd.Series / CategoricalDtype path, including missing-value handling
    (unknown values → ``-1``).
    """
    import numpy as np

    from kumo_core.encoders.index_mapping import Mapper

    m = Mapper(cats)

    queries = np.array(['A', 'C', 'Z', 'B', 'A'])
    out_np = m.map(queries)
    out_pd = m.map(pd.Series(queries))
    assert torch.equal(out_np, out_pd)
    # Sanity: 'Z' is unknown and should be -1.
    assert out_np.tolist() == [0, 2, -1, 1, 0]


@pytest.mark.parametrize(('index', 'expected'), [
    (pd.Index([3, 1, 2]), [0, 1, -1]),
    (pd.Index([3, 1, 2], dtype=object), [0, 1, -1]),
    (pd.Index([3, 1, 2], dtype='Int64'), [0, 1, -1]),
    (pd.CategoricalIndex([3, 1, 2], categories=[1, 2, 3]), [0, 1, -1]),
    (pd.RangeIndex(1, 4), [2, 0, -1]),
])
def test_mapper_numpy_preserves_index_positions(index, expected) -> None:
    import numpy as np

    from kumo_core.encoders.index_mapping import Mapper

    assert Mapper(index).map(np.array([3, 1, 4])).tolist() == expected
