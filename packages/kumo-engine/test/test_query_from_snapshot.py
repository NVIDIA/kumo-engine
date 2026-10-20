# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pandas as pd
import pytest
from kumoapi.task import TaskType

import kumo.engine as kumo


def _scores(trainer, table, snapshot):
    frame = trainer.predict(table, snapshot=snapshot,
                            binary_classification_threshold=0.5).frame
    return frame.sort_values('ENTITY').reset_index(drop=True)


def test_snapshot_bound_query_predicts_like_graph_bound(fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    query = kumo.PredictiveQuery.from_snapshot(snapshot, graph_table.query)
    expected = _scores(trainer, graph_table, snapshot)

    assert query.get_task_type() == TaskType.BINARY_CLASSIFICATION
    table = query.generate_prediction_table(
        snapshot, plan=trainer.training_table_generation_plan)
    assert table.timestamps == graph_table.timestamps
    pd.testing.assert_frame_equal(_scores(trainer, table, snapshot), expected)

    anchor = pd.Timestamp(graph_table.timestamps[0])
    pd.DataFrame({
        'ENTITY1': range(50),
        'TIMESTAMP': [anchor] * 50,
    }).to_parquet(tmp_path / 'rows.parquet')
    supplied = kumo.PredictionTable.from_source_table(
        kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet'], pq=query,
        snapshot=snapshot)
    pd.testing.assert_frame_equal(_scores(trainer, supplied, snapshot),
                                  expected)


def test_snapshot_bound_query_refuses_other_graph(fitted):
    _, graph_table, snapshot = fitted
    query = kumo.PredictiveQuery.from_snapshot(snapshot, graph_table.query)
    newer = replace(snapshot, id=snapshot.id.split('@')[0] + '@newer')
    other = replace(snapshot, id='other@' + snapshot.id.split('@')[1])

    query._require_matching_snapshot(newer)
    plain = replace(snapshot, id='plain')
    kumo.PredictiveQuery.from_snapshot(plain, '')._require_matching_snapshot(
        plain)
    with pytest.raises(ValueError, match='was not materialized'):
        query.generate_prediction_table(other)
