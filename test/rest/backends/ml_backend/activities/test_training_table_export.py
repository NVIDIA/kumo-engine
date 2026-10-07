# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import shutil
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.training import export
from kumo_core.types import PredictionArtifactType, TableStatType, TaskType
from kumoapi.data_source import DataSourceType
from kumoapi.jobs import TrainingTableArtifactType
from kumoapi.source_table import S3SourceTable

from rest.backends.data_backend.activities import custom_table_validation
from rest.backends.data_backend.activities.custom_train_table import (
    _validate_supplied_train_table,
    missing_source_column_errors,
)
from rest.backends.ml_backend.activities import parquet_io_activities as pio
from rest.backends.ml_backend.workflows.artifact_export_workflow import (
    ArtifactExportWorkflow,
)
from rest.config.artifact_export import ArtifactExportInput
from rest.config.batch_prediction_job import OutputArtifactStats
from rest.config.file_type import FileType

_TABLE = TrainingTableArtifactType.FULL_TRAIN_TABLE
_WORKFLOW = 'rest.backends.ml_backend.workflows.artifact_export_workflow.workflow'
_STAGES = {'train': [[1, 2], [3]], 'val': [[4]], 'test': [[5, 6]]}


def _part(entities):
    return pa.table({
        'ENTITY1': pa.array(entities, pa.int64()),
        'TARGET1': pa.array([float(e) for e in entities], pa.float64()),
        'TIMESTAMP': pa.array([e * 10 for e in entities], pa.int64()),
        'exclude_target_keys': pa.array([[e] for e in entities],
                                        pa.list_(pa.int64())),
    })


@pytest.fixture
def layout(tmp_path):
    staging = write_v3_bundle(str(tmp_path / 'built'), 'tt-1')
    stages = []
    for key, parts in _STAGES.items():
        names = []
        for index, entities in enumerate(parts):
            name = f'{key}-{index}.parquet'
            pq.write_table(_part(entities), osp.join(staging, name))
            names.append(name)
        stages.append({'key': key, 'parts': names})
    bundle = str(tmp_path / 'root' / '_bundles' / 'tt-1')
    shutil.copytree(staging, bundle)
    spec = osp.join(bundle, export.INGEST_SPEC)
    with open(spec) as handle:
        content = json.load(handle)
    content['split_stages'] = stages
    with open(spec, 'w') as handle:
        json.dump(content, handle)
    return str(tmp_path / 'root'), bundle


def _rows(path):
    table = pq.read_table(path)
    return sorted(
        zip(table['ENTITY1'].to_pylist(), table['SPLIT'].to_pylist()))


def _export_dir(root):
    return osp.join(root, '_exports', 'tt-1')


def _prepare(bundle):
    return pio.prepare_training_table_export.__wrapped__(
        osp.join(bundle, 'tt.parquet'))


def test_hosted_export_rows_equal_engine_export(layout, tmp_path):
    root, bundle = layout
    out = _prepare(bundle)
    engine = export.export_train_table(bundle, str(tmp_path / 'engine'))
    assert osp.dirname(out) == _export_dir(root)
    assert osp.basename(out).startswith('tt-export-v1-')
    assert pa.concat_tables([pq.read_table(f) for f in engine]).equals(
        pa.concat_tables(
            [pq.read_table(osp.join(out, f)) for f in sorted(os.listdir(out))]))
    assert _rows(out) == [(1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 2)]
    assert 'exclude_target_keys' not in pq.read_table(out).column_names


def test_second_call_reuses_pointer(layout):
    root, bundle = layout
    out = _prepare(bundle)
    with open(osp.join(_export_dir(root), 'tt-export-v1.json')) as handle:
        assert json.load(handle) == {'path': out}
    os.remove(osp.join(bundle, 'train-0.parquet'))
    assert _prepare(bundle) == out
    assert len(os.listdir(_export_dir(root))) == 2


def test_existing_pointer_wins_without_writing(layout):
    root, bundle = layout
    winner = osp.join(root, 'elsewhere')
    os.makedirs(_export_dir(root))
    with open(osp.join(_export_dir(root), 'tt-export-v1.json'), 'w') as handle:
        json.dump({'path': winner}, handle)
    assert _prepare(bundle) == winner
    assert os.listdir(_export_dir(root)) == ['tt-export-v1.json']


def test_bundle_outside_bundles_dir_passes_through(layout, tmp_path):
    root, bundle = layout
    moved = str(tmp_path / 'other' / 'tt-1')
    shutil.copytree(bundle, moved)
    path = osp.join(moved, 'tt.parquet')
    assert pio.exported_training_table(path) == path
    assert not osp.exists(osp.join(tmp_path, 'other', '_exports'))


def test_path_without_spec_passes_through(tmp_path):
    legacy = tmp_path / 'root' / '_bundles' / 'old' / 'tt.parquet'
    legacy.mkdir(parents=True)
    assert pio.exported_training_table(str(legacy)) == str(legacy)
    assert not (tmp_path / 'root' / '_exports').exists()


def _supplied(tmp_path, table, drop_row=False):
    table = table.append_column('WEIGHT', pa.array([1.0] * table.num_rows))
    if drop_row:
        table = table.slice(1)
    path = tmp_path / 'supplied'
    path.mkdir(exist_ok=True)
    pq.write_table(table, str(path / 'part-0.parquet'))
    return str(path)


def _validate(bundle, supplied, split_col_name='SPLIT'):
    return _validate_supplied_train_table(
        osp.join(bundle, 'tt.parquet'), supplied, 'WEIGHT',
        TaskType.BINARY_CLASSIFICATION, split_col_name=split_col_name)


def test_validation_compares_against_export_rows(layout, tmp_path):
    _, bundle = layout
    table = pq.read_table(_prepare(bundle))
    assert _validate(bundle, _supplied(tmp_path, table)).ok


def test_validation_rejects_missing_export_row(layout, tmp_path):
    _, bundle = layout
    table = pq.read_table(_prepare(bundle))
    response = _validate(bundle, _supplied(tmp_path, table, drop_row=True))
    assert not response.ok
    assert 'Row count mismatch' in [e.title for e in response.errors]


def test_validation_without_split_compares_against_tt_parquet(
        layout, tmp_path):
    _, bundle = layout
    tt = pq.read_table(osp.join(bundle, 'tt.parquet'))
    assert tt.num_rows != 6
    assert _validate(bundle, _supplied(tmp_path, tt), None).ok

    seeds = pq.read_table(_prepare(bundle)).drop_columns(['SPLIT'])
    response = _validate(bundle, _supplied(tmp_path, seeds), None)
    assert 'Row count mismatch' in [e.title for e in response.errors]


def test_validation_honors_configured_split_column(layout, tmp_path):
    _, bundle = layout
    table = pq.read_table(_prepare(bundle)).rename_columns({'SPLIT': 'stage'})
    supplied = _supplied(tmp_path, table)
    assert _validate(bundle, supplied, 'stage').ok
    assert missing_source_column_errors(
        table.column_names + ['WEIGHT'], osp.join(bundle, 'tt.parquet'),
        'WEIGHT', 'stage') == []


def test_unrelated_split_column_keeps_tt_parquet_reference(layout, tmp_path):
    _, bundle = layout
    tt = pq.read_table(osp.join(bundle, 'tt.parquet'))
    tt = tt.append_column('SPLIT', pa.array([0] * tt.num_rows, pa.int64()))
    assert _validate(bundle, _supplied(tmp_path, tt), None).ok


async def _extensive(bundle, split_col, num_rows):
    connector = MagicMock()
    connector.get_table_stats.return_value = {
        TableStatType.NUM_ROWS: SimpleNamespace(value=num_rows)
    }
    names = ['ENTITY1', 'TIMESTAMP', 'TARGET1', 'WEIGHT'] + (
        [split_col] if split_col else [])
    metadata = SimpleNamespace(
        cols=[SimpleNamespace(name=name) for name in names])
    source = S3SourceTable(source={'mode': 'relative', 'path': 'weighted'},
                           connector_id='connector_' + 'a' * 32,
                           file_type=FileType.PARQUET,
                           data_source_type=DataSourceType.S3)
    module = 'rest.backends.data_backend.activities.custom_table_validation'
    with patch(f'{module}.get_connector_from_spec_async',
               new=AsyncMock(return_value=connector)), \
            patch(f'{module}._get_source_table_metadata',
                  new=SimpleNamespace(coroutine=AsyncMock(
                      return_value=metadata))):
        return await (
            custom_table_validation.validate_custom_train_table_activity(
                MagicMock(), source, 'WEIGHT',
                osp.join(bundle, 'tt.parquet'), True, split_col))


@pytest.mark.asyncio
async def test_extensive_validation_counts_exported_rows(layout):
    _, bundle = layout
    tt_rows = pq.read_table(osp.join(bundle, 'tt.parquet')).num_rows
    assert tt_rows != 6
    assert (await _extensive(bundle, 'stage', 6)).ok
    assert not (await _extensive(bundle, 'stage', tt_rows)).ok
    assert (await _extensive(bundle, None, tt_rows)).ok


def _stats(kind, path):
    return OutputArtifactStats(artifact_type=kind, num_rows=1, num_cols=1,
                               size_bytes=1, output_path=path)


@pytest.mark.asyncio
async def test_workflow_exports_training_table_before_stats_and_write():
    exported = '/root/_exports/tt-1/tt-export-v1-abc'
    calls = []

    async def execute_activity(activity, *, args, **_kwargs):
        calls.append((activity.__name__, args))
        if activity is pio.prepare_training_table_export:
            return exported
        if activity is pio.get_artifact_stats:
            return {_TABLE: _stats(_TABLE, args[0][_TABLE])}
        return []

    input = ArtifactExportInput(
        artifact_paths={_TABLE: '/root/_bundles/tt-1/tt.parquet'},
        output_config={})
    with patch(f'{_WORKFLOW}.execute_activity', new=execute_activity):
        await ArtifactExportWorkflow()._run(input)

    assert [name for name, _ in calls] == [
        'prepare_training_table_export', 'get_artifact_stats',
        'write_artifacts_to_customer'
    ]
    assert calls[0][1] == ['/root/_bundles/tt-1/tt.parquet']
    assert calls[1][1][0] == {_TABLE: exported}
    assert calls[2][1][0].artifact_paths == {_TABLE: exported}
    assert input.artifact_paths == {_TABLE: '/root/_bundles/tt-1/tt.parquet'}


@pytest.mark.asyncio
async def test_workflow_skips_prediction_only_inputs():
    kind = PredictionArtifactType.PREDICTIONS
    execute = AsyncMock(side_effect=[{kind: _stats(kind, '/p')}, []])
    input = ArtifactExportInput(artifact_paths={kind: '/p'}, output_config={})
    with patch(f'{_WORKFLOW}.execute_activity', new=execute):
        await ArtifactExportWorkflow()._run(input)
    assert pio.prepare_training_table_export not in [
        c.args[0] for c in execute.call_args_list
    ]
