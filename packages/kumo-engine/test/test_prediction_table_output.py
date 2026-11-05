# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pandas as pd
import pytest
from kumoapi.jobs import PredictionArtifactType, WriteMode

import kumo.engine as kumo
from kumo.engine.trainer.output_config import PredictionOutputConfig


class TableSink:
    def __init__(self):
        self.tables = {}

    def write_files(self, connector, table_name, files,
                    write_mode=WriteMode.OVERWRITE, project_id=None, *,
                    export_id=None, extra_columns=None):
        frame = pd.concat(pd.read_parquet(f) for f in files).assign(
            **(extra_columns or {}))
        if write_mode is WriteMode.APPEND and table_name in self.tables:
            frame = pd.concat([self.tables[table_name], frame])
        self.tables[table_name] = frame.reset_index(drop=True)
        return table_name


def _output(sink, **options):
    return PredictionOutputConfig(
        SimpleNamespace(source_type='SNOWFLAKE', sink_backend=sink),
        PredictionArtifactType.PREDICTIONS, 'PREDS', **options)


@pytest.mark.parametrize('mlops', [False, True])
def test_table_outputs_overwrite_unless_appending_with_their_ids(
        fitted, tmp_path, mlops):
    trainer, table, snapshot = fitted
    sink = TableSink()
    cfg = kumo.MLOps(str(tmp_path / 'root')) if mlops else None

    def predict(**options):
        return trainer.predict(table, snapshot=snapshot, mlops=cfg,
                               output_config=_output(sink, **options),
                               binary_classification_threshold=.5)

    last = [predict() for _ in range(2)][-1]
    assert list(sink.tables['PREDS'].columns) == list(last.frame.columns)
    assert len(sink.tables['PREDS']) == last.num_rows
    last.export(_output(sink))
    assert len(sink.tables['PREDS']) == last.num_rows

    runs = [predict(write_mode=WriteMode.APPEND) for _ in range(2)]
    written = sink.tables['PREDS'].iloc[last.num_rows:]
    assert written.groupby('PREDICTION_ID').size().to_dict() == {
        run.id: run.num_rows
        for run in runs
    }
    assert set(written['BUNDLE_ID']) == {trainer.result.id}


@pytest.mark.parametrize('mlops', [False, True])
def test_file_outputs_are_written_at_exactly_path(fitted, tmp_path, mlops):
    trainer, table, snapshot = fitted
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    output = PredictionOutputConfig(connector, output_paths={'predictions':
                                                             'preds/'})
    cfg = kumo.MLOps(str(tmp_path / 'root')) if mlops else None
    run = trainer.predict(table, snapshot=snapshot, output_config=output,
                          mlops=cfg, binary_classification_threshold=.5)
    written = pd.read_parquet(tmp_path / 'preds')
    assert list(written.columns) == list(run.frame.columns)
    assert len(written) == run.num_rows
    run.export(PredictionOutputConfig(connector, path='exported.csv'))
    assert (tmp_path / 'exported.csv').is_file()

    appended = PredictionOutputConfig(connector, output_paths={
        'predictions': 'other/'
    }, write_mode=WriteMode.APPEND)
    with pytest.raises(NotImplementedError, match='APPEND'):
        trainer.predict(table, snapshot=snapshot, output_config=appended,
                        binary_classification_threshold=.5)


def test_path_names_a_file_destination_or_a_table():
    files = kumo.FileConnector(root_dir='.')
    assert PredictionOutputConfig(files, path='scores/').destinations() == {
        'predictions': 'scores/'}
    table = SimpleNamespace(source_type='SNOWFLAKE', sink_backend=None)
    config = PredictionOutputConfig(table, path='CHURN_SCORES')
    assert config.destinations() == {'predictions': 'CHURN_SCORES'}
    assert config.write_mode is WriteMode.OVERWRITE
    assert PredictionOutputConfig(files, path='scores/').write_mode is None
    with pytest.raises(ValueError, match='only one form'):
        PredictionOutputConfig(files, path='scores/',
                               output_paths={'predictions': 'x/'})
