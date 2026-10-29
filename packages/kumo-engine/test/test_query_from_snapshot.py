# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pandas as pd
import pytest

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


def _scores(trainer, table, snapshot):
    frame = trainer.predict(table, snapshot=snapshot,
                            binary_classification_threshold=0.5).frame
    return frame.sort_values('ENTITY').reset_index(drop=True)


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
    pd.testing.assert_frame_equal(scores(prediction_table=graph_table), expected)
    legacy = trainer.predict(prediction_table=graph_table, snapshot=snapshot,
                             binary_classification_threshold=0.5).frame
    pd.testing.assert_frame_equal(
        legacy.sort_values('ENTITY').reset_index(drop=True), expected)
    with pytest.raises(ValueError, match='not both'):
        scores(entities=rows, prediction_time=graph_table.timestamps[0])
    # Same schema under another graph id: scored, with an info log only.
    other = replace(snapshot, id='other@' + snapshot.id.split('@')[1])
    scored = trainer.predict(other, binary_classification_threshold=0.5).frame
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
            scored.sort_values('ENTITY').reset_index(drop=True), expected,
            atol=1e-5)
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.mark.parametrize('scope', [{
    'entities': object()
}, {
    'prediction_time': '2024-01-01'
}])
def test_a_prediction_table_takes_no_entities_or_time(scope):
    from kumoapi.model_plan import ModelPlan

    with pytest.raises(ValueError, match='not both'):
        kumo.Trainer(ModelPlan()).predict(snapshot=object(),
                                          prediction_table=object(), **scope)


@pytest.mark.parametrize('call', [
    lambda trainer, snapshot: trainer.predict(object(), snapshot=snapshot,
                                              prediction_table=object()),
    lambda trainer, snapshot: trainer.predict(snapshot, snapshot=snapshot),
])
def test_an_argument_given_twice_is_refused(fitted, call):
    trainer, _, snapshot = fitted
    with pytest.raises(TypeError, match='two'):
        call(trainer, snapshot)


def test_each_fit_result_scores_with_its_own_model(fitted):
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.pquery.predictive_query import PredictiveQuery

    fixture_trainer, prediction_table, snapshot = fitted
    table = PredictiveQuery._from_snapshot(
        snapshot,
        fixture_trainer._training_query).generate_training_table(snapshot)
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    first = trainer.fit(table, snapshot=snapshot)
    second = trainer.fit(table, snapshot=snapshot)
    assert first.bundle_id != second.bundle_id

    scores = [
        _scores(result, prediction_table, snapshot)
        for result in (first, second)
    ]
    assert not scores[0].equals(scores[1])
    for result, score in zip((first, second), scores):
        own = kumo.Trainer.load(result.bundle_dir)
        pd.testing.assert_frame_equal(
            score, _scores(own, prediction_table, snapshot))


def test_entities_missing_from_the_snapshot_are_recorded_as_skipped(
        fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    anchor = pd.Timestamp(graph_table.timestamps[0])
    pd.DataFrame({
        'ENTITY1': [*range(10), *range(1000, 1004)],
        'TIMESTAMP': [anchor] * 14,
    }).to_parquet(tmp_path / 'rows.parquet')
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']

    with pytest.warns(UserWarning, match=r'^Prediction skipped 4 of 14 '
                      r'requested entities \(28\.6%\): not in snapshot$'):
        result = trainer.predict(snapshot, entities=rows,
                                 binary_classification_threshold=0.5)
    assert sorted(result.frame['ENTITY']) == list(range(10))
    assert (result.requested_rows, result.scored_rows,
            result.skipped_rows) == (14, 10, 4)
    table = result.manifest.prediction_table
    assert sorted(table.skipped_sample) == [1000, 1001, 1002, 1003]
    assert table.skip_reason == 'not in snapshot'

    from kumo.engine._tracking import prediction_record
    logged = {summary.name: summary.value
              for summary in prediction_record(result.manifest).summaries}
    assert logged['data/skipped_rows'] == 4
    assert logged['data/skipped_fraction'] == pytest.approx(4 / 14)

    full = trainer.predict(snapshot, binary_classification_threshold=0.5)
    assert full.skipped_rows == 0
    assert full.requested_rows == full.scored_rows == 50
    assert full.manifest.prediction_table.skipped_sample == []
