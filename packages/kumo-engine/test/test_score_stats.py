# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from kumo_core.types import TaskType

from kumo.engine.trainer._provenance import holdout_baseline


def _close_to_ranks(summary, values):
    values = np.sort(values)
    assert summary['count'] == len(values)
    assert summary['min'] == values[0] and summary['max'] == values[-1]
    assert summary['mean'] == pytest.approx(values.mean())
    ranks = np.searchsorted(values, summary['percentiles']) / len(values)
    assert np.abs(ranks - np.arange(1, 100) / 100).max() < .02


def test_long_format_holdouts_take_each_predictions_top_score(tmp_path):
    rng = np.random.default_rng(0)
    entities, k = 400, 3
    score = -np.sort(-rng.random((entities, k)), axis=1)
    frame = pd.DataFrame({
        'ENTITY': np.repeat(np.arange(entities), k),
        'CLASS': [f'c{i}' for i in range(entities * k)],
        'SCORE': score.ravel(),
        'LABEL': np.tile([True, False, False], entities),
        'PREDICTED': np.tile([True, False, False], entities),
    })
    frame['TARGET'] = np.repeat(frame['CLASS'].to_numpy()[::k], k)
    path = tmp_path / 'test_df.parquet'
    frame.to_parquet(path, row_group_size=7)

    for task_type in (TaskType.MULTILABEL_CLASSIFICATION,
                      TaskType.MULTICLASS_CLASSIFICATION):
        holdout = holdout_baseline(path, task_type, truncated=None)
        assert holdout['rows'] == entities
        _close_to_ranks(holdout['columns']['top_score'], score[:, 0])
        assert holdout['truncated'] is None


def test_binary_and_regression_holdouts(tmp_path):
    rng = np.random.default_rng(1)
    prob = rng.random(1000)
    pd.DataFrame({
        'TARGET': (rng.random(1000) < .3).astype(int),
        'True_PROB': prob,
    }).to_parquet(tmp_path / 'binary.parquet')
    pd.DataFrame({
        'TARGET': [1., 2., 5.],
        'TARGET_PRED': [1., 3., 4.]
    }).to_parquet(tmp_path / 'regression.parquet')

    binary = holdout_baseline(tmp_path / 'binary.parquet',
                              TaskType.BINARY_CLASSIFICATION, truncated=False)
    regression = holdout_baseline(tmp_path / 'regression.parquet',
                                  TaskType.REGRESSION, truncated=False)

    assert binary['rows'] == 1000
    assert binary['label_positive_rate'] == pytest.approx(
        pd.read_parquet(tmp_path / 'binary.parquet')['TARGET'].mean())
    _close_to_ranks(binary['columns']['True_PROB'], prob)
    assert regression['rows'] == 3
    assert regression['columns']['TARGET']['max'] == 5.
    assert regression['columns']['TARGET_PRED']['mean'] == pytest.approx(8 / 3)
