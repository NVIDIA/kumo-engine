# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``TrainTableGenerationProgress`` construction.

These guard a Python-3.14 / pydantic-2 migration bug: polling train-table
generation progress crashed because the per-timeframe stats loaded from the
job-dir JSON arrive as plain dicts with *string* dates, and pydantic v2 (unlike
v1) strictly validates them against the ``pd.Timestamp`` fields of
``TemporalNodeTimeFrameStats`` / ``TemporalLinkTimeFrameStats`` when those
dataclasses are nested inside the pydantic-v2 ``TrainTableGenerationProgress``
dataclass, raising ``is_instance_of`` validation errors.
"""
import json
import os

import pandas as pd

from kumo.stats.train_table_stats import (
    TemporalLinkTimeFrameStats,
    TemporalNodeTimeFrameStats,
    TrainTableGenerationProgress,
    train_table_generation_stats_from_job_dir,
)


def _node_timeframe_dict(index: int) -> dict:
    """A per-timeframe entry exactly as it appears in the job-dir JSON: a plain
    dict whose date fields are ISO strings (not ``pd.Timestamp``)."""
    return dict(
        min_date='2020-01-01',
        max_date='2020-02-01',
        anchor_date='2020-01-15',
        target_distribution={'a': 1},
        index=index,
        num_train_rows=10,
        num_entities=5,
        num_targets=5,
    )


def _link_timeframe_dict(index: int) -> dict:
    return dict(
        min_date='2020-01-01',
        max_date='2020-02-01',
        anchor_date='2020-01-15',
        num_train_rows=10,
        num_entities=5,
        num_targets=5,
        num_links=12,
        index=index,
        node_degree_stats=dict(
            avg_entity_degree=1.5,
            min_entity_degree=1,
            max_entity_degree=3,
            avg_target_degree=2.0,
            min_target_degree=1,
            max_target_degree=4,
        ),
    )


def test_progress_from_string_date_dicts_node() -> None:
    # Mirrors the value produced by ``TemporalNodeTrainTableStats(**stats_dict)``
    # whose ``stats_per_timeframe`` (loaded from JSON) is a list of plain dicts.
    iteration_data = [_node_timeframe_dict(0), _node_timeframe_dict(1)]
    progress = TrainTableGenerationProgress(
        num_expected_iterations=2,
        num_finished_iterations=2,
        iteration_data=iteration_data,
    )
    assert progress.num_expected_iterations == 2
    assert progress.num_finished_iterations == 2
    assert progress.iteration_data is not None
    assert len(progress.iteration_data) == 2
    first = progress.iteration_data[0]
    assert isinstance(first, TemporalNodeTimeFrameStats)
    # String dates must be coerced to ``pd.Timestamp``.
    assert isinstance(first.min_date, pd.Timestamp)
    assert first.min_date == pd.Timestamp('2020-01-01')
    assert isinstance(first.anchor_date, pd.Timestamp)


def test_progress_from_string_date_dicts_link() -> None:
    iteration_data = [_link_timeframe_dict(0)]
    progress = TrainTableGenerationProgress(
        num_expected_iterations=1,
        num_finished_iterations=1,
        iteration_data=iteration_data,
    )
    assert progress.iteration_data is not None
    first = progress.iteration_data[0]
    assert isinstance(first, TemporalLinkTimeFrameStats)
    assert isinstance(first.min_date, pd.Timestamp)
    assert first.num_links == 12
    # Nested optional dataclass is coerced from a dict too.
    assert first.node_degree_stats is not None
    assert first.node_degree_stats.max_target_degree == 4


def test_progress_passes_through_dataclass_instances() -> None:
    # When ``iteration_data`` already holds constructed dataclass instances
    # (valid case), validation must remain behavior-preserving.
    inst = TemporalNodeTimeFrameStats(
        min_date='2020-03-01',
        max_date='2020-04-01',
        anchor_date='2020-03-15',
        target_distribution={'b': 1},
        index=0,
    )
    assert isinstance(inst.min_date, pd.Timestamp)  # __post_init__ coercion
    progress = TrainTableGenerationProgress(
        num_expected_iterations=1,
        num_finished_iterations=1,
        iteration_data=[inst],
    )
    assert progress.iteration_data is not None
    assert isinstance(progress.iteration_data[0].min_date, pd.Timestamp)
    assert progress.iteration_data[0].min_date == pd.Timestamp('2020-03-01')


def test_generation_stats_from_job_dir_temporal_node(tmp_path) -> None:
    """End-to-end: write a synthetic ``pquery_table_stats.json`` and load it,
    reproducing the prod ``get_progress`` polling path."""
    stats_dict = dict(
        stats_type='temporal_node_pred',
        min_date='2020-01-01',
        max_date='2020-12-01',
        anchor_date='2020-06-01',
        num_entities=100,
        num_targets=50,
        num_train_rows=200,
        num_timeframes=3,
        num_timeframes_post_filter=2,
        avg_entities_per_timeframe=33,
        num_target_rows=200,
        target_distribution={'a': 1},
        scoring_table_column_stats={},
        node_degree_stats=None,
        stats_per_timeframe=[_node_timeframe_dict(0), _node_timeframe_dict(1)],
    )
    job_dir = str(tmp_path)
    out_dir = os.path.join(job_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'pquery_table_stats.json'), 'w') as f:
        json.dump(stats_dict, f)

    progress = train_table_generation_stats_from_job_dir(job_dir)
    assert progress is not None
    assert progress.num_expected_iterations == 3
    assert progress.num_finished_iterations == 2
    assert progress.iteration_data is not None
    assert len(progress.iteration_data) == 2
    assert all(
        isinstance(tf, TemporalNodeTimeFrameStats)
        for tf in progress.iteration_data)
    assert all(
        isinstance(tf.min_date, pd.Timestamp)
        for tf in progress.iteration_data)
