# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.encoders.selection import compute_missing_stats_
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.train.formatting import format_embeddings, format_prediction
from kumo_core.types import ColStatType, TaskType
from kumoapi.encoder import Index, MultiCategorical
from torch import tensor


def test_format_prediction_regression():
    df = pd.DataFrame({"USER_ID": ['1', '2', '3']})
    pred = tensor([0.3, 0.56, 0.14])
    prob = None
    encoder = NumericalTargetEncoder(offset=0., scale=1.)
    formatted_df = format_prediction(
        df=df,
        target_col_name='TARGET',
        target_encoder=encoder,
        task_type=TaskType.REGRESSION,
        pred=pred,
        prob=prob,
    )
    expected_df = pd.DataFrame({
        "USER_ID": ['1', '2', '3'],
        "TARGET_PRED": pred.numpy(),
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(formatted_df.reset_index(drop=True),
                                  expected_df)


def _assert_format_prediction_forecasting_quantiles(out_timestamps):
    df = pd.DataFrame({
        "USER_ID": ['1', '2'],
        "TIMESTAMP": pd.Timestamp("2022-04-11"),
    })
    pred = tensor([[10.0, 11.0], [20.0, 21.0]])
    quantiles = tensor([
        [[9.0], [10.0]],
        [[19.0], [20.0]],
    ])
    timestamps = pd.to_datetime(['2022-04-12', '2022-04-13'] * 2)

    formatted_df = format_prediction(
        df=df,
        target_col_name='TARGET',
        target_encoder=NumericalTargetEncoder(offset=0., scale=1.),
        task_type=TaskType.FORECASTING,
        pred=pred,
        prob=None,
        quantiles=quantiles,
        quantile_names=['0.5'],
        time_col_name='TIMESTAMP',
        out_timestamps=out_timestamps,
    ).reset_index(drop=True)

    expected_df = pd.DataFrame({
        "USER_ID": ['1', '1', '2', '2'],
        "TIMESTAMP": timestamps,
        "TARGET_PRED": [10.0, 11.0, 20.0, 21.0],
        "q_0.5": [9.0, 10.0, 19.0, 20.0],
        "ANCHOR_TIMESTAMP": pd.Timestamp('2022-04-11'),
    })

    pd.testing.assert_frame_equal(formatted_df, expected_df, check_dtype=False)


def test_format_prediction_forecasting_quantiles_detaches_timestamps():
    timestamps = pd.to_datetime(['2022-04-12', '2022-04-13'] * 2)
    out_timestamps = torch.tensor(timestamps.astype('int64'),
                                  dtype=torch.float64, requires_grad=True)

    _assert_format_prediction_forecasting_quantiles(out_timestamps)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA is not available')
def test_format_prediction_forecasting_quantiles_moves_timestamps_to_cpu():
    timestamps = pd.to_datetime(['2022-04-12', '2022-04-13'] * 2)
    out_timestamps = tensor(timestamps.astype('int64'), device='cuda')

    _assert_format_prediction_forecasting_quantiles(out_timestamps)


def test_forecast_rejects_anchor_time_column_collision():
    df = pd.DataFrame({'ANCHOR_TIMESTAMP': [pd.Timestamp('2022-04-11')]})
    original = df.copy()
    with pytest.raises(ValueError, match='Rename the time column'):
        format_prediction(
            df=df, target_col_name='TARGET',
            target_encoder=NumericalTargetEncoder(offset=0., scale=1.),
            task_type=TaskType.FORECASTING, pred=tensor([[10.0, 11.0]]),
            time_col_name='ANCHOR_TIMESTAMP')
    pd.testing.assert_frame_equal(df, original)


def test_format_prediction_binary_classification():
    # Binary classification has pred tensor(bool dtype), prob tensor of shape
    # N*2 and an Index target encoder with 2 categories
    df = pd.DataFrame({
        "USER_ID": ['1', '2', '3'],
    })
    pred = tensor([False, True, False])
    prob = tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    encoder_cfg = Index()
    compute_missing_stats_(encoder_cfg, pd.Series([False, True]))
    encoder = TargetEncoder.from_encoder(
        TaskType.BINARY_CLASSIFICATION,
        encoder_cfg,
    )
    formatted_df = format_prediction(
        df=df,
        target_col_name='TARGET',
        target_encoder=encoder,
        task_type=TaskType.BINARY_CLASSIFICATION,
        pred=pred,
        prob=prob,
    ).reset_index(drop=True)
    expected_df = pd.DataFrame({
        "USER_ID": ['1', '2', '3'],
        "TARGET_PRED": pred.numpy(),
        "False_PROB": prob[:, 0].numpy(),
        "True_PROB": prob[:, 1].numpy(),
    }).reset_index(drop=True)
    expected_df["TARGET_PRED"] = expected_df["TARGET_PRED"].astype(
        pd.BooleanDtype())
    pd.testing.assert_frame_equal(formatted_df, expected_df)


def test_format_prediction_multilabel():
    # Test with string dtype for entities
    df = pd.DataFrame({
        "USER_ID": ['1', '2'],
        "TIMESTAMP": ["2022-04-11", "2022-04-11"]
    })
    # Assume there are 3 categories in the dataset: {0, 1, 2} - string type
    encoder_config = MultiCategorical()
    compute_missing_stats_(encoder_config, pd.Series(['0', '1,2']))
    encoder = TargetEncoder.from_encoder(
        TaskType.MULTILABEL_CLASSIFICATION,
        encoder_config,
    )
    pred = tensor([[0, 1, 1], [1, 0, 0]])
    # Since it's a multi-label prediction, the probabilities don't need to be
    # normalized.
    prob = tensor([[0.1, 0.9, 0.6], [0.9, 0.1, 0.0]])
    df = format_prediction(
        df=df,
        target_col_name='TARGET',
        target_encoder=encoder,
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        pred=pred,
        prob=prob,
    )
    # In the long format and there are 3 total classes, we get 3
    # entries per entity = 6 rows.
    expected_df = pd.DataFrame({
        "USER_ID": ['1', '1', '1', '2', '2', '2'],
        "TIMESTAMP": ["2022-04-11"] * 6,
        "PREDICTED": [0, 1, 1, 1, 0, 0],
        "CLASS": ["0", "1", "2", "0", "1", "2"],
        "SCORE":
        np.array([0.1, 0.9, 0.6, 0.9, 0.1, 0.0], dtype=np.float32),
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df)


def test_format_prediction_multilabel_array():
    # Test with string dtype for entities
    df = pd.DataFrame({
        "USER_ID": ["1", "2"],
        "TIMESTAMP": ["2022-04-11", "2022-04-11"]
    })
    encoder = MultiCategorical(
        _stats={
            ColStatType.MULTI_CATEGORY_COUNTS: [["0", 1], ["1", 1], ["2", 1]],
            ColStatType.MULTI_CATEGORIES_SEPARATOR: None,
        })
    encoder = TargetEncoder.from_encoder(
        TaskType.MULTILABEL_CLASSIFICATION,
        encoder,
    )

    # NOTE tensor_mapper.forward to disables _is_string_dtype, but since this
    # test case doesn't have forward calls, we need to manually disable it.
    encoder.tensor_mapper._is_string_dtype = False

    pred = tensor([[0, 1, 1], [1, 0, 0]])
    # probabilities don't need to be normalized as it's a multi-label task.
    prob = tensor([[0.1, 0.9, 0.6], [0.9, 0.1, 0.0]])
    df = format_prediction(
        df=df,
        target_col_name="TARGET",
        target_encoder=encoder,
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        pred=pred,
        prob=prob,
        array_targets=True,
    )
    # In the long format and there are 3 total classes, we get 3
    # entries per entity = 6 rows.
    expected_df = pd.DataFrame({
        "USER_ID": ["1", "1", "1", "2", "2", "2"],
        "TIMESTAMP": ["2022-04-11"] * 6,
        "PREDICTED": [0, 1, 1, 1, 0, 0],
        "CLASS": ["0", "1", "2", "0", "1", "2"],
        "SCORE":
        np.array([0.1, 0.9, 0.6, 0.9, 0.1, 0.0], dtype=np.float32),
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df)


def test_format_prediction_multi_class_top_k():
    # Test with integer dtype for entities
    df = pd.DataFrame({
        "USER_ID": [1, 2],
        "TIMESTAMP": ["2022-04-11", "2022-04-11"]
    })
    encoder_cfg = Index()
    # Assume that there are 3 classes in the dataset: {0, 1, 2}
    compute_missing_stats_(encoder_cfg, pd.Series([1, 0, 2]))
    encoder = TargetEncoder.from_encoder(
        TaskType.MULTICLASS_CLASSIFICATION,
        encoder_cfg,
    )
    pred = tensor([1, 0])
    prob = tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]])

    df = format_prediction(
        df=df,
        target_col_name='TARGET',
        target_encoder=encoder,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        pred=pred,
        prob=prob,
        top_k=2,
    ).reset_index(drop=True)
    expected_df = pd.DataFrame({
        "USER_ID": [1, 1, 2, 2],
        "TIMESTAMP": ["2022-04-11", "2022-04-11"] * 2,
        "PREDICTED": [1, 0, 1, 0],
        "CLASS": [1, 0, 0, 1],
        "SCORE":
        np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32),
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected_df)


def test_format_embeddings_array():
    meta_df = pd.DataFrame({
        "USER_ID": [1, 2],
        "TIMESTAMP": ["2022-04-11", "2022-04-11"]
    })

    emb = np.array([[0.35686827, 0.8628754], [0.2126852, 0.8948473]])

    df = format_embeddings(emb, meta_df)

    assert df.shape == (2, 3)
    assert df.columns.tolist() == ['USER_ID', 'TIMESTAMP', 'EMBEDDING']
    df_emb = df['EMBEDDING'].tolist()
    for i in range(len(emb)):
        row = emb[i]
        assert np.array_equal(row, df_emb[i])
