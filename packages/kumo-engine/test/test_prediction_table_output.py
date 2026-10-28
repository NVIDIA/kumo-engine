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


def test_table_outputs_append_runs_with_their_ids(fitted):
    trainer, table, snapshot = fitted
    sink = TableSink()
    runs = [
        trainer.predict(table, snapshot=snapshot, output_config=_output(sink),
                        binary_classification_threshold=.5) for _ in range(2)
    ]

    written = sink.tables['PREDS']
    assert written.groupby('PREDICTION_ID').size().to_dict() == {
        run.id: run.num_rows
        for run in runs
    }
    assert set(written['BUNDLE_ID']) == {trainer.result.id}

    last = trainer.predict(
        table, snapshot=snapshot, binary_classification_threshold=.5,
        output_config=_output(sink, write_mode=WriteMode.OVERWRITE))
    assert set(sink.tables['PREDS']['PREDICTION_ID']) == {last.id}
    assert len(sink.tables['PREDS']) == last.num_rows


def test_file_outputs_default_to_a_run_directory_without_id_columns(
        fitted, tmp_path):
    trainer, table, snapshot = fitted
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    output = PredictionOutputConfig(connector, output_paths={'predictions':
                                                             'preds/'})
    assert output.write_mode is None
    assert _output(TableSink()).write_mode is WriteMode.APPEND

    run = trainer.predict(table, snapshot=snapshot, output_config=output,
                          binary_classification_threshold=.5)
    columns = set(pd.read_parquet(tmp_path / 'preds' / run.id).columns)
    assert not columns & {'PREDICTION_ID', 'BUNDLE_ID'}

    appended = PredictionOutputConfig(connector, output_paths={
        'predictions': 'other/'
    }, write_mode=WriteMode.APPEND)
    with pytest.raises(NotImplementedError, match='APPEND'):
        trainer.predict(table, snapshot=snapshot, output_config=appended,
                        binary_classification_threshold=.5)
