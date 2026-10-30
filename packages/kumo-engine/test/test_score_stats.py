# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from kumo.engine._score_stats import parquet_stats, top_score_stats


def _close_to_ranks(summary, values):
    values = np.sort(values)
    assert summary['count'] == len(values)
    assert summary['min'] == values[0] and summary['max'] == values[-1]
    assert summary['mean'] == pytest.approx(values.mean())
    ranks = np.searchsorted(values, summary['percentiles']) / len(values)
    assert np.abs(ranks - np.arange(1, 100) / 100).max() < .02


def test_long_format_outputs_take_each_predictions_top_score(tmp_path):
    rng = np.random.default_rng(0)
    entities, k = 400, 3
    score = rng.random((entities, k))
    frame = pd.DataFrame({
        'ENTITY': np.repeat(np.arange(entities), k),
        'CLASS': [f'c{i}' for i in range(entities * k)],
        'SCORE': score.ravel(),
        'LABEL': np.tile([True, False, False], entities),
        'PREDICTED': np.tile([True, False, False], entities),
    })
    parts = [tmp_path / 'part-0.parquet', tmp_path / 'part-1.parquet']
    frame.iloc[:601].to_parquet(parts[0])
    frame.iloc[601:].to_parquet(parts[1])

    _close_to_ranks(top_score_stats(parts), score.max(axis=1))


def test_non_finite_scores_are_not_counted(tmp_path):
    pd.DataFrame({
        'True_PROB': [.2, np.nan, np.inf, .4, None, -np.inf],
        'TARGET': [1, 0, 1, 0, 1, 0],
    }).to_parquet(tmp_path / 'binary.parquet')

    summary = parquet_stats([tmp_path / 'binary.parquet'],
                            ['True_PROB'])['True_PROB']

    assert summary['count'] == 2
    assert summary['min'] == .2 and summary['max'] == .4
