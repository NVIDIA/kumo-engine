# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.types import TaskType

from kumo.engine._score_stats import output_counts


def _write(frame, tmp_path, cut):
    paths = []
    for index, part in enumerate((frame.iloc[:cut], frame.iloc[cut:])):
        paths.append(str(tmp_path / f'part-{index}.parquet'))
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False),
                       paths[-1], row_group_size=7)
    return paths


def _expected(name, axis, edges, values):
    counts = Counts({axis: tuple(edges)})
    counts.histogram(name, axis, values)
    counts.total(f'{name}/sum', values[np.isfinite(values)].sum())
    counts.extent(name, values)
    return counts


def _assert_same(counted, expected):
    assert counted.edges == expected.edges
    assert counted.extents == expected.extents
    assert counted.arrays.keys() == expected.arrays.keys()
    for name, array in expected.arrays.items():
        assert counted.arrays[name].tolist() == pytest.approx(array.tolist())


def test_scores_count_as_score_counts_does_on_edges_and_outside(tmp_path):
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.random(100), PROBABILITY_EDGES,
                             [-0.5, 1.5, np.nan, np.inf]])
    frame = pd.DataFrame({'True_PROB': scores, 'TARGET_PRED': scores > 0.5})
    paths = _write(frame, tmp_path, 60)

    counted = output_counts(paths, TaskType.BINARY_CLASSIFICATION,
                            {'score': PROBABILITY_EDGES})

    expected = _expected('True_PROB', 'score', PROBABILITY_EDGES, scores)
    expected.total('predicted_positive', (scores > 0.5).sum())
    _assert_same(counted, expected)


def test_a_target_counts_on_the_holdout_bins(tmp_path):
    edges = (0.0, 10.0, 20.0)
    values = np.array([-1.0, 0.0, 5.0, 10.0, 19.9, 20.0, 35.0, np.nan])
    paths = _write(pd.DataFrame({'TARGET_PRED': values}), tmp_path, 3)

    counted = output_counts(paths, TaskType.REGRESSION, {'target': edges})

    _assert_same(counted, _expected('TARGET_PRED', 'target', edges, values))


def test_top_scores_count_per_entity_and_anchor_across_files(tmp_path):
    rng = np.random.default_rng(1)
    entity = np.repeat(np.arange(200), 6)
    frame = pd.DataFrame({
        'ENTITY': entity,
        'ANCHOR_TIMESTAMP': pd.to_datetime(
            np.tile(np.repeat(['2024-01-01', '2024-02-01'], 3), 200)),
        'CLASS': rng.integers(0, 50, entity.size),
        'SCORE': rng.random(entity.size),
    })
    paths = _write(frame, tmp_path, 601)
    tops = frame.groupby(['ENTITY', 'ANCHOR_TIMESTAMP'])['SCORE'].max()
    edges = tuple(np.quantile(tops, np.linspace(0, 1, 101)).tolist())

    confidence = output_counts(paths, TaskType.MULTICLASS_CLASSIFICATION,
                               {'score': PROBABILITY_EDGES})
    link = output_counts(paths, TaskType.TEMPORAL_LINK_PREDICTION,
                         {'top_score': edges})

    _assert_same(confidence, _expected('confidence', 'score',
                                       PROBABILITY_EDGES, tops.to_numpy()))
    _assert_same(link, _expected('top_score', 'top_score', edges,
                                 tops.to_numpy()))


def test_nothing_is_counted_without_the_scores_bins(tmp_path):
    paths = _write(pd.DataFrame({'ENTITY': [1, 2], 'SCORE': [.1, .2]}),
                   tmp_path, 1)

    assert output_counts(paths, TaskType.STATIC_LINK_PREDICTION, {}) is None
    assert output_counts(paths, TaskType.REGRESSION, None) is None
