# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    skipped = pq.read_table(table.skipped_location).to_pydict()
    assert sorted(skipped['ENTITY1']) == [1000, 1001, 1002, 1003]
    assert set(skipped['condition']) == {'unknown_entity'}

    from kumo_core.summaries import Note, Scalar

    from kumo.engine._tracking import prediction_record
    summaries = prediction_record(result.manifest).summaries
    logged = {s.name: s.value for s in summaries if isinstance(s, Scalar)}
    assert logged['pred/skipped_share'] == pytest.approx(4 / 14)
    assert '4 requested rows were not scored: not in snapshot (1000, 1001, ' \
        '1002, 1003); full list: skipped/.' in [
            s.text for s in summaries if isinstance(s, Note)]

    full = trainer.predict(snapshot, binary_classification_threshold=0.5)
    assert full.skipped_rows == 0
    assert full.requested_rows == full.scored_rows == 50
    assert full.manifest.prediction_table.skipped_sample == []
    assert full.manifest.prediction_table.skipped_location is None


class _Recorder:
    def __init__(self):
        self.records = []

    def log_prediction_run(self, record):
        self.records.append(record)


def test_mlops_stores_and_tracks_every_skipped_row(fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    ids = [*range(10), *range(1000, 1004), None, None]
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array(ids, pa.int64()),
            'TIMESTAMP': [pd.Timestamp(graph_table.timestamps[0])] * 16,
        }), tmp_path / 'rows.parquet', row_group_size=4)
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']
    tracker = _Recorder()

    with pytest.warns(UserWarning, match='skipped 6 of 16'):
        result = trainer.predict(
            snapshot, entities=rows, binary_classification_threshold=0.5,
            mlops=kumo.MLOps(str(tmp_path / 'root'), tracking=tracker))

    stored = tmp_path / 'root' / 'predictions' / result.id / 'skipped'
    assert (stored / '_SUCCESS').is_file()
    assert result.manifest.prediction_table.skipped_location == str(stored)
    skipped = pq.read_table(stored).to_pydict()
    assert sorted(zip(skipped['ENTITY1'], skipped['condition']),
                  key=str) == sorted(
                      [*((id, 'unknown_entity') for id in range(1000, 1004)),
                       (None, 'null_entity'), (None, 'null_entity')], key=str)
    [record] = tracker.records
    assert pq.read_table(record.files['skipped']).num_rows == 6


def test_a_run_that_scores_nothing_publishes_nothing(fitted, tmp_path):
    trainer, graph_table, snapshot = fitted
    pd.DataFrame({
        'ENTITY1': range(1000, 1004),
        'TIMESTAMP': [pd.Timestamp(graph_table.timestamps[0])] * 4,
    }).to_parquet(tmp_path / 'rows.parquet')
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']
    tracker = _Recorder()

    with pytest.raises(ValueError, match='None of the 4'):
        trainer.predict(
            snapshot, entities=rows, binary_classification_threshold=0.5,
            mlops=kumo.MLOps(str(tmp_path / 'root'), tracking=tracker))
    assert not (tmp_path / 'root' / 'predictions').exists()
    assert tracker.records == []


@pytest.mark.parametrize('ids, times, message', [
    (range(1000, 1014), None, 'not in snapshot (sample: 1000, 1001, 1002, …)'),
    ([None] * 14, None, 'null entity id'),
    (range(14), pd.NaT, 'null anchor time'),
    ([1000, 1001, *[None] * 12], None,
     'null entity id (12); not in snapshot (2) (sample: 1000, 1001)'),
])
def test_predict_refuses_rows_that_none_can_be_scored(fitted, tmp_path, ids,
                                                      times, message):
    trainer, graph_table, snapshot = fitted
    anchor = pd.Timestamp(graph_table.timestamps[0]) if times is None else times
    pd.DataFrame({
        'ENTITY1': pd.array(list(ids), dtype='Int64'),
        'TIMESTAMP': pd.Series([anchor] * 14, dtype='datetime64[ns]'),
    }).to_parquet(tmp_path / 'rows.parquet')
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']

    with pytest.raises(ValueError) as raised:
        trainer.predict(snapshot, entities=rows,
                        binary_classification_threshold=0.5)
    assert str(raised.value) == (
        f'None of the 14 requested entities can be scored: {message}')


def test_training_rows_that_none_can_be_kept_keep_their_error(
        fitted, tmp_path):
    from diskgraph import DiskgraphDataError
    from kumoapi.train import TrainingTableSpec

    from kumo.engine.pquery import PredictiveQuery, TrainingTable

    trainer, graph_table, snapshot = fitted
    pd.DataFrame({
        'ENTITY1': range(1000, 1014),
        'TIMESTAMP': [pd.Timestamp(graph_table.timestamps[0])] * 14,
        'TARGET1': [True] * 14,
    }).to_parquet(tmp_path / 'rows.parquet')
    rows = kumo.FileConnector(root_dir=str(tmp_path))['rows.parquet']
    query = PredictiveQuery._from_snapshot(snapshot, trainer._training_query)
    with pytest.raises(DiskgraphDataError) as raised:
        TrainingTable.from_source_table(rows, TrainingTableSpec(), pq=query,
                                        snapshot=snapshot)
    assert str(raised.value) == (
        'custom-TT source kept no rows (14 read, 14 skipped), so the run '
        'would train on nothing')


def test_skipped_samples_count_rows_from_the_chunks_first_row_group(tmp_path):
    from kumo.engine.pquery.prediction_table import _sample_ids

    pq.write_table(pa.table({'ENTITY1': [10, 11, 12, 13, 14]}),
                   tmp_path / 'rows.parquet', row_group_size=2)
    conditions = [{'column': 'ENTITY1',
                   'samples': ['rows.parquet row group 1 row 2']}]
    assert _sample_ids(conditions, tmp_path / 'rows.parquet') == (14, )
