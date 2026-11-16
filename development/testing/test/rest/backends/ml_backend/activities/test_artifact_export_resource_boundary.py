# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
from asyncio import CancelledError
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, Mock, call, patch

import pandas as pd
import pytest
from kumo_core.types import PredictionArtifactType
from kumoapi.jobs import TrainingTableArtifactType
from pandas.testing import assert_frame_equal

from kumo.connector.connector import ConnectorError
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.file_connector import FileConnector
from rest.backends.ml_backend.activities.parquet_io_activities import (
    _DisposableCSVConverter,
    _write_artifacts_to_customer,
)
from rest.backends.ml_backend.artifact_export import (
    OutputStats,
    _convert_parquet_to_local_csv,
)
from rest.backends.ml_backend.workflows.artifact_export_workflow import (
    ArtifactExportWorkflow,
)
from rest.config.artifact_export import (
    ArtifactExportInput,
    ArtifactOutTable,
    DatabricksManagedVolumeStage,
    FileOutputConfig,
    NamedOutputConfig,
)
from rest.config.batch_prediction_job import OutputArtifactStats
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.jobs import WriteMode
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import ArtifactExportError

_ARTIFACT = PredictionArtifactType.PREDICTIONS
_PROJECT_ID = ProjectID('project_' + 'a' * 32)


def _reference(index: int) -> ConnectorExecutionRef:
    return ConnectorExecutionRef(
        project_id=_PROJECT_ID,
        connector_id=ConnectorID(f'connector_{index + 1:032x}'),
    )


def _stats(path='/internal/predictions'):
    return OutputArtifactStats(
        artifact_type=_ARTIFACT,
        num_rows=10,
        num_cols=3,
        size_bytes=240,
        output_path=path,
    )


def _named_output(index, mode, *, stage=True):
    stage_config = (DatabricksManagedVolumeStage(
        path=('/Volumes/catalog/schema/_kumo_staging/'
              f'artifact-export/output-{index}'),
        files=[
            '/Volumes/catalog/schema/_kumo_staging/'
            f'artifact-export/output-{index}/part.parquet'
        ],
    ) if stage else None)
    return NamedOutputConfig(
        connector=_reference(index),
        out_table=ArtifactOutTable(
            table_name=f'predictions_{index}',
            write_mode=mode,
        ),
        managed_volume_stage=stage_config,
    )


def _input(outputs, path='/internal/predictions'):
    return ArtifactExportInput(
        artifact_paths={_ARTIFACT: path},
        output_config={_ARTIFACT: outputs},
    )


def test_disposable_converter_protocol_contains_only_safe_paths() -> None:
    executor = Mock()
    future = executor.submit.return_value
    future.result.return_value = None

    with patch(
            'rest.backends.ml_backend.activities.parquet_io_activities.'
            'HealingProcessPoolExecutor', return_value=executor) as factory:
        with _DisposableCSVConverter() as converter:
            converter('/internal/artifact/part-000.parquet',
                      '/tmp/private-stage/part-000.csv')
            converter('/internal/artifact/part-001.parquet',
                      '/tmp/private-stage/part-001.csv')

    factory.assert_called_once_with(1)
    assert executor.submit.call_args_list == [
        call(
            _convert_parquet_to_local_csv,
            '/internal/artifact/part-000.parquet',
            '/tmp/private-stage/part-000.csv',
        ),
        call(
            _convert_parquet_to_local_csv,
            '/internal/artifact/part-001.parquet',
            '/tmp/private-stage/part-001.csv',
        ),
    ]
    executor.terminate.assert_called_once_with()


def test_disposable_converter_kills_child_on_cancellation() -> None:
    executor = Mock()
    future = executor.submit.return_value
    future.result.side_effect = TimeoutError

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'HealingProcessPoolExecutor', return_value=executor),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'activity.in_activity', return_value=True),
            patch('rest.backends.ml_backend.activities.parquet_io_activities.'
                  'activity.heartbeat') as heartbeat,
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'activity.is_cancelled', return_value=True),
            pytest.raises(CancelledError, match='CSV export cancelled'),
    ):
        with _DisposableCSVConverter() as converter:
            converter('/internal/artifact/part.parquet',
                      '/tmp/private-stage/part.csv')

    future.result.assert_called_once_with(timeout=1.0)
    heartbeat.assert_called_once_with()
    executor.terminate.assert_called_once_with()


def test_disposable_converter_kills_child_on_conversion_error() -> None:
    executor = Mock()
    executor.submit.return_value.result.side_effect = RuntimeError('failed')

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'HealingProcessPoolExecutor', return_value=executor),
            pytest.raises(RuntimeError, match='failed'),
    ):
        with _DisposableCSVConverter() as converter:
            converter('/internal/artifact/part.parquet',
                      '/tmp/private-stage/part.csv')

    executor.terminate.assert_called_once_with()


def test_real_disposable_converter_preserves_csv_and_reaps_child(
        tmp_path) -> None:
    """Exercise repeated Pandas/PyArrow work in one disposable process."""
    input_path = tmp_path / 'input.parquet'
    expected = pd.DataFrame({
        'id': range(2048),
        'embedding': [[1, 2, 3]] * 2048,
    })
    expected.to_parquet(input_path, index=False)
    expected['embedding'] = '1 2 3'

    existing_children = {
        child.pid
        for child in multiprocessing.active_children()
    }
    output_paths = [tmp_path / f'output-{index}.csv' for index in range(4)]
    with _DisposableCSVConverter() as converter:
        for output_path in output_paths:
            converter(os.fspath(input_path), os.fspath(output_path))

    leaked_children = [
        child for child in multiprocessing.active_children()
        if child.pid not in existing_children
    ]
    assert leaked_children == []
    for output_path in output_paths:
        actual = pd.read_csv(output_path)
        assert_frame_equal(actual, expected)


def test_bp_s3_artifact_is_copied_directly_under_customer_scope() -> None:
    project_id = ProjectID('project_' + 'a' * 32)
    connector_id = ConnectorID('connector_' + 'b' * 32)
    reference = ConnectorExecutionRef(project_id=project_id,
                                      connector_id=connector_id)
    output = FileOutputConfig(
        connector=reference,
        output_path='s3://customer-output/predictions/',
    )
    input = ArtifactExportInput(
        artifact_paths={
            PredictionArtifactType.PREDICTIONS:
            's3://internal-artifacts/predictions/'
        },
        output_config={PredictionArtifactType.PREDICTIONS: [output]},
    )
    state = {'customer_scope': False}
    connector = Mock(spec=FileConnector)

    @contextmanager
    def customer_scope(*_args):
        state['customer_scope'] = True
        try:
            yield
        finally:
            state['customer_scope'] = False

    connector.filesystem_credential_scope.side_effect = customer_scope

    def write_customer(source_path, *_args, **_kwargs):
        assert state['customer_scope']
        assert source_path == 's3://internal-artifacts/predictions/'
        assert _kwargs['artifact_files'] == [
            's3://internal-artifacts/predictions/part-000.parquet',
            's3://internal-artifacts/predictions/part-001.parquet',
        ]
        return OutputStats(size_bytes=1,
                           display_path='s3://customer-output/predictions/')

    converter = Mock()
    converter.__enter__ = Mock(return_value=Mock())
    converter.__exit__ = Mock(return_value=False)
    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'get_connector_from_spec', return_value=connector),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'resolve_file_output_path',
                return_value='s3://customer-output/predictions/'),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                '_resolve_selected_prediction_artifacts',
                return_value=(
                    {
                        PredictionArtifactType.PREDICTIONS:
                        's3://internal-artifacts/predictions/'
                    },
                    {
                        PredictionArtifactType.PREDICTIONS: [
                            's3://internal-artifacts/predictions/'
                            'part-000.parquet',
                            's3://internal-artifacts/predictions/'
                            'part-001.parquet',
                        ]
                    },
                    None,
                )),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'fsutil.copy',
                side_effect=AssertionError('BP output must not be staged')),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'fsutil.copy_file',
                side_effect=AssertionError('BP parts must not be staged')),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'tempfile.TemporaryDirectory',
                side_effect=AssertionError('BP output must not allocate a '
                                           'whole-artifact staging root')),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'write_artifact_to_s3', side_effect=write_customer),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                '_DisposableCSVConverter', return_value=converter),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'parquet_util.get_parquet_metadata',
                side_effect=AssertionError(
                    'export must reuse initial artifact metadata'),
            ) as read_metadata,
    ):
        result = _write_artifacts_to_customer(
            input,
            artifact_stats={
                PredictionArtifactType.PREDICTIONS:
                OutputArtifactStats(
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    num_rows=1,
                    num_cols=2,
                    size_bytes=8,
                    output_path='s3://internal-artifacts/predictions/',
                )
            },
            resolved_connectors={PredictionArtifactType.PREDICTIONS: [Mock()]},
        )

    read_metadata.assert_not_called()
    connector.close.assert_called_once_with()
    assert result[0].output_path == 's3://customer-output/predictions/'
    assert not state['customer_scope']


def test_legacy_training_table_export_retains_remote_staging() -> None:
    artifact_type = TrainingTableArtifactType.FULL_TRAIN_TABLE
    output = FileOutputConfig(
        connector=_reference(0),
        output_path='s3://customer-output/training-table/',
    )
    input = ArtifactExportInput(
        artifact_paths={
            artifact_type: 's3://internal-artifacts/training-table.parquet'
        },
        output_config={artifact_type: [output]},
    )
    connector = Mock(spec=FileConnector)
    connector.filesystem_credential_scope.return_value = contextmanager(
        lambda: (yield))()

    def write_customer(source_path, *_args, **_kwargs):
        assert os.path.isabs(source_path)
        assert not source_path.startswith('s3://')
        return OutputStats(
            size_bytes=1,
            display_path='s3://customer-output/training-table/',
        )

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'get_connector_from_spec', return_value=connector),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'resolve_file_output_path',
                return_value='s3://customer-output/training-table/'),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'fsutil.is_local', return_value=False),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'fsutil.copy') as copy_internal,
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'write_artifact_to_s3', side_effect=write_customer),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'parquet_util.get_parquet_metadata',
                side_effect=AssertionError(
                    'export must reuse initial artifact metadata')),
    ):
        result = _write_artifacts_to_customer(
            input,
            artifact_stats={
                artifact_type:
                OutputArtifactStats(
                    artifact_type=artifact_type,
                    num_rows=1,
                    num_cols=2,
                    size_bytes=8,
                    output_path=(
                        's3://internal-artifacts/training-table.parquet'),
                )
            },
            resolved_connectors={artifact_type: [Mock()]},
        )

    copy_internal.assert_called_once()
    connector.close.assert_called_once_with()
    assert result[0].output_path == 's3://customer-output/training-table/'


@pytest.mark.parametrize('second_mode', [WriteMode.OVERWRITE, WriteMode.APPEND])
def test_later_fanout_failure_preserves_completed_write_and_closes_connectors(
        second_mode) -> None:
    outputs = [
        _named_output(index, mode)
        for index, mode in enumerate([WriteMode.OVERWRITE, second_mode])
    ]
    connectors = [Mock(spec=DatabricksConnector) for _ in outputs]
    committed = OutputStats(size_bytes=240, display_path='predictions_0')

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'get_connector_from_spec',
                side_effect=connectors,
            ),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'write_artifact_to_databricks',
                side_effect=[
                    committed,
                    ConnectorError('second destination failed'),
                ],
            ) as write,
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'parquet_io_semaphore.acquire',
                side_effect=AssertionError(
                    'warehouse loads must not take the parquet IO slot'),
            ),
            pytest.raises(ConnectorError, match='second destination failed'),
    ):
        _write_artifacts_to_customer(
            _input(outputs),
            artifact_stats={_ARTIFACT: _stats()},
            resolved_connectors={_ARTIFACT: [Mock(), Mock()]},
        )

    assert write.call_count == 2
    for connector in connectors:
        connector.close.assert_called_once_with()


def test_databricks_export_rejects_internal_artifact_without_direct_stage(
) -> None:
    path = 's3://internal-artifacts/predictions/'
    connector = Mock(spec=DatabricksConnector)

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'get_connector_from_spec',
                return_value=connector,
            ),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'write_artifact_to_databricks', ) as write,
            pytest.raises(
                ArtifactExportError,
                match='producer-uploaded Unity Catalog volume',
            ) as exc_info,
    ):
        _write_artifacts_to_customer(
            _input(
                [_named_output(0, WriteMode.OVERWRITE, stage=False)],
                path,
            ),
            artifact_stats={_ARTIFACT: _stats(path)},
            resolved_connectors={_ARTIFACT: [Mock()]},
        )

    assert exc_info.value.non_retryable
    write.assert_not_called()
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('write_modes', 'expected_attempts'),
    [
        ([WriteMode.OVERWRITE, WriteMode.OVERWRITE], 3),
        ([WriteMode.OVERWRITE, WriteMode.APPEND], 1),
    ],
)
async def test_fanout_retry_policy_is_safe_for_every_destination(
    write_modes,
    expected_attempts,
) -> None:
    outputs = [
        _named_output(index, mode, stage=False)
        for index, mode in enumerate(write_modes)
    ]
    stats = _stats()
    execute = AsyncMock(side_effect=[
        {
            _ARTIFACT: stats
        },
        [stats],
    ])

    with patch(
            'rest.backends.ml_backend.workflows.artifact_export_workflow.'
            'workflow.execute_activity',
            new=execute,
    ):
        assert await ArtifactExportWorkflow()._run(_input(outputs)) == [stats]

    export_call = execute.call_args_list[1]
    assert export_call.kwargs['retry_policy'].maximum_attempts == (
        expected_attempts)
    one_attempt_timeout = timedelta(
        hours=2,
        seconds=(stats.size_bytes * len(outputs) / (2 << 20)),
    )
    assert export_call.kwargs['schedule_to_close_timeout'] == (
        one_attempt_timeout * (3 if expected_attempts == 3 else 1))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'artifact_types',
    [
        [PredictionArtifactType.PREDICTIONS],
        [PredictionArtifactType.EMBEDDINGS],
        [
            PredictionArtifactType.PREDICTIONS,
            PredictionArtifactType.EMBEDDINGS,
        ],
    ],
)
async def test_local_download_returns_stats_for_every_artifact(
        artifact_types) -> None:
    artifact_paths = {
        artifact_type: f'/internal/{artifact_type.value.lower()}'
        for artifact_type in artifact_types
    }
    stats = {
        artifact_type:
        OutputArtifactStats(
            artifact_type=artifact_type,
            num_rows=10,
            num_cols=3,
            size_bytes=240,
            output_path=artifact_paths[artifact_type],
        )
        for artifact_type in artifact_types
    }
    execute = AsyncMock(side_effect=[stats, []])
    input = ArtifactExportInput(
        artifact_paths=artifact_paths,
        output_config={},
    )

    with patch(
            'rest.backends.ml_backend.workflows.artifact_export_workflow.'
            'workflow.execute_activity', new=execute):
        output = await ArtifactExportWorkflow()._run(input)

    assert [item.artifact_type for item in output] == artifact_types
    assert [item.output_path for item in output] == [
        f'N/A (customer can download {artifact_type.value.lower()})'
        for artifact_type in artifact_types
    ]


@pytest.mark.asyncio
async def test_embeddings_only_export_never_opens_predictions() -> None:
    embedding_path = '/internal/embeddings'
    embedding_stats = OutputArtifactStats(
        artifact_type=PredictionArtifactType.EMBEDDINGS,
        num_rows=147,
        num_cols=3,
        size_bytes=4096,
        output_path=embedding_path,
    )
    calls = []

    async def execute_activity(_activity, *, args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            artifact_paths = args[0]
            assert artifact_paths == {
                PredictionArtifactType.EMBEDDINGS: embedding_path,
            }
            assert PredictionArtifactType.PREDICTIONS not in artifact_paths
            return {PredictionArtifactType.EMBEDDINGS: embedding_stats}
        assert len(calls) == 2
        export_input, stats = args
        assert set(export_input.artifact_paths) == {
            PredictionArtifactType.EMBEDDINGS,
        }
        assert set(stats) == {PredictionArtifactType.EMBEDDINGS}
        return []

    input = ArtifactExportInput(
        artifact_paths={
            PredictionArtifactType.EMBEDDINGS: embedding_path,
        },
        output_config={},
    )
    with patch(
            'rest.backends.ml_backend.workflows.artifact_export_workflow.'
            'workflow.execute_activity', new=execute_activity):
        output = await ArtifactExportWorkflow()._run(input)

    assert len(calls) == 2
    assert output == [embedding_stats]
    assert output[0].num_rows == 147
