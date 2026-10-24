# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pandas as pd
import pytest
from kumoapi.task import TaskType

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


def _scores(trainer, table, snapshot):
    frame = trainer.predict(table, snapshot=snapshot,
                            binary_classification_threshold=0.5).frame
    return frame.sort_values('ENTITY').reset_index(drop=True)


def test_snapshot_bound_query_predicts_like_graph_bound(fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    assert trainer._training_query == graph_table.query
    query = kumo.PredictiveQuery.from_snapshot(snapshot, trainer._training_query)
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


def test_predict_scores_a_snapshot_with_the_saved_query(fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    expected = _scores(trainer, graph_table, snapshot)

    def scores(**options):
        return trainer.predict(
            snapshot, binary_classification_threshold=0.5,
            **options).frame.sort_values('ENTITY').reset_index(drop=True)

    pd.testing.assert_frame_equal(scores(), expected)
    pd.DataFrame({
        'ENTITY1': range(50),
        'TIMESTAMP': [pd.Timestamp(graph_table.timestamps[0])] * 50,
    }).to_parquet(tmp_path / 'rows.parquet')
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']
    pd.testing.assert_frame_equal(scores(entities=rows), expected)
    pd.testing.assert_frame_equal(scores(table=graph_table), expected)
    legacy = trainer.predict(prediction_table=graph_table, snapshot=snapshot,
                             binary_classification_threshold=0.5).frame
    pd.testing.assert_frame_equal(
        legacy.sort_values('ENTITY').reset_index(drop=True), expected)
    with pytest.raises(ValueError, match='not both'):
        scores(entities=rows, prediction_time=graph_table.timestamps[0])
    other = replace(snapshot, id='other@' + snapshot.id.split('@')[1])
    with pytest.warns(UserWarning, match='trained on a snapshot of graph'):
        scored = trainer.predict(
            other, binary_classification_threshold=0.5).frame
    pd.testing.assert_frame_equal(
        scored.sort_values('ENTITY').reset_index(drop=True), expected)


def test_exported_snapshot_scores_in_a_fresh_scratch_without_a_graph(
        fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    expected = _scores(trainer, graph_table, snapshot)
    exports = kumo.FileConnector(root_dir=str(tmp_path / 'exports'))
    location = snapshot.export(kumo.SnapshotOutputConfig(exports, 'snapshots'))
    bundle = trainer._training_result.bundle_dir
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'fresh')
    try:
        loaded = kumo.GraphSnapshot.load_from_connector(exports, path=location)
        assert loaded.id == snapshot.id
        assert loaded.index_path.is_relative_to(tmp_path / 'fresh')
        model = kumo.Trainer.load(bundle)
        scored = model.result.predict(
            loaded, binary_classification_threshold=0.5).frame
        pd.testing.assert_frame_equal(
            scored.sort_values('ENTITY').reset_index(drop=True), expected)
        query = kumo.PredictiveQuery.from_snapshot(loaded, model._training_query)
        table = query.generate_prediction_table(
            loaded, plan=model.training_table_generation_plan)
        pd.testing.assert_frame_equal(_scores(model, table, loaded), expected)
    finally:
        kumo.configure(scratch_dir=previous)


def test_snapshot_bound_query_checks_snapshot_graph(fitted):
    _, graph_table, snapshot = fitted
    query = kumo.PredictiveQuery.from_snapshot(snapshot, graph_table.query)
    newer = replace(snapshot, id=snapshot.id.split('@')[0] + '@newer')
    other = replace(snapshot, id='other@' + snapshot.id.split('@')[1])
    plain = replace(snapshot, id='plain')

    query.suggest_model_plan(newer)
    kumo.PredictiveQuery.from_snapshot(plain, graph_table.query) \
        .suggest_model_plan(plain)
    with pytest.raises(ValueError, match='was not materialized'):
        query.suggest_model_plan(other)
