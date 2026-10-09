# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from kumo_core.types import PredictionArtifactType

from kumo.train.artifact_manifest import PredictionManifestRef
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.workflows import \
    batch_prediction_job_workflow as module
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    DatabricksVolumeShard,
    PredictionShardResult,
)
from rest.config.ml_backend import JobState
from rest.config.prediction_output import DatabricksPredictionOutput
from statedb.config.common import ConnectorID
from statedb.utils.errors import PredictionOutputProcessingError


@pytest.mark.parametrize(
    ('num_shards', 'max_durable_gpu_slots', 'expected'),
    [(4, None, 4), (5, 4, 4), (0, None, 1), (1, None, 1)],
)
def test_diskgraph_bp_resource_hint(
    num_shards,
    max_durable_gpu_slots,
    expected,
):
    assert module._prediction_resource_hint(
        num_shards, max_durable_gpu_slots).min_gpu_slots == expected


def _uploads(producer_abis):
    stage = '/Volumes/catalog/schema/_kumo_staging/artifact-export/job'
    return [
        DatabricksVolumeShard(
            output_index=0,
            artifact_type=PredictionArtifactType.PREDICTIONS,
            stage_path=stage,
            staged_files=[f'{stage}/shard-{index}.parquet'],
            num_rows=10 + index,
            num_cols=3,
            size_bytes=100 + index,
            producer_abi=abi,
        ) for index, abi in enumerate(producer_abis)
    ]


def test_aggregate_databricks_uploads_combines_workers_once_per_output():
    stage = '/Volumes/catalog/schema/_kumo_staging/artifact-export/job'
    uploads = _uploads(['kumo-pred-output-v3'] * 2)

    stages, stats = module._aggregate_databricks_uploads(
        uploads,
        expected_shards=2,
    )

    assert stages[0].files == [
        f'{stage}/shard-0.parquet',
        f'{stage}/shard-1.parquet',
    ]
    assert stats[PredictionArtifactType.PREDICTIONS].num_rows == 21
    assert stats[PredictionArtifactType.PREDICTIONS].size_bytes == 201


@pytest.mark.parametrize('producer_abis,accepted', [
    (['kumo-pred-output-v2'], True),
    (['kumo-pred-output-v2', 'kumo-pred-output-v3'], False),
    (['kumo-pred-output-v2', 'kumo-pred-output-v2'], False),
])
def test_aggregate_databricks_uploads_requires_shard_count_abi(
        producer_abis, accepted):
    uploads = _uploads(producer_abis)
    if accepted:
        module._aggregate_databricks_uploads(uploads,
                                             expected_shards=len(uploads))
    else:
        with pytest.raises(PredictionOutputProcessingError,
                           match='version change.*Re-run'):
            module._aggregate_databricks_uploads(
                uploads, expected_shards=len(uploads))
        # Pre-patch histories replay without the check.
        module._aggregate_databricks_uploads(uploads,
                                             expected_shards=len(uploads),
                                             check_producer_abi=False)


@pytest.mark.asyncio
async def test_diskgraph_bp_dispatch_uses_long_job_deadline(monkeypatch):
    job_input = SimpleNamespace(
        id='batchpredictionjob-test',
        training_job_id='trainingjob-test',
        predictive_query=SimpleNamespace(
            name='query', graph=SimpleNamespace(name='prediction-graph')),
        graph_data_compute_input=SimpleNamespace(graph_key='prediction-graph'),
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
        config=SimpleNamespace(num_parallel_workers_limit=1, outputs=[]),
    )
    monkeypatch.setattr(
        module, 'deploy_env', lambda: SimpleNamespace(
            on_local=False,
            data_compute=SimpleNamespace(k8s_pool=None),
        ))
    monkeypatch.setattr(module.workflow, 'now', lambda: datetime.now())
    monkeypatch.setattr(
        module.workflow,
        'patched',
        lambda name: name == module.PREDICTION_ARTIFACT_OWNERSHIP_PATCH,
    )
    monkeypatch.setattr(module.workflow, 'execute_activity', AsyncMock())
    run_pooled = AsyncMock(return_value=PredictionShardResult(
        state=JobState.FAILED))
    monkeypatch.setattr(module, 'run_pooled_activity', run_pooled)

    workflow = module.BatchPredictionJobWorkflow()
    workflow._save_output = AsyncMock()  # type: ignore[method-assign]
    await workflow._run(
        cast(BatchPredictionJobInput, job_input),
        cast(MLJobSettings, SimpleNamespace()),
        '/tmp/prediction-config.json',
    )

    dispatch = run_pooled.await_args_list[0].kwargs
    assert dispatch['schedule_to_close_timeout'] == timedelta(days=21)
    assert dispatch['heartbeat_timeout'] == timedelta(minutes=5)
    assert dispatch['args'][-2:] == [True, True]


@pytest.mark.asyncio
async def test_direct_databricks_stats_run_before_copy_and_stage_cleanup(
        monkeypatch):
    connector_id = ConnectorID('connector_' + '2' * 32)
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(connector_id),
        table_name='catalog__kumo__schema__kumo__predictions',
    )
    stage = '/Volumes/catalog/schema/_kumo_staging/artifact-export/job'
    upload = DatabricksVolumeShard(
        output_index=0,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        stage_path=stage,
        staged_files=[f'{stage}/part.parquet'],
        num_rows=2,
        num_cols=2,
        size_bytes=100,
    )
    job_input = SimpleNamespace(
        id='batchpredictionjob-direct',
        training_job_id='trainingjob-test',
        predictive_query=SimpleNamespace(
            name='query',
            graph=SimpleNamespace(name='prediction-graph'),
            target='target',
        ),
        graph_data_compute_input=SimpleNamespace(graph_key='prediction-graph'),
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
        output_connectors={connector_id: SimpleNamespace()},
        config=SimpleNamespace(
            num_parallel_workers_limit=1,
            outputs=[output],
        ),
    )
    events = []

    async def execute_activity(activity, *, args=None, **_kwargs):
        if activity is module.DataBackendActivities.start_bp_stats_activity:
            events.append(('stats', args[4].path))
        elif activity is module.cleanup_databricks_volume_outputs:
            events.append(('cleanup', None))

    async def execute_child_workflow(*_args, **_kwargs):
        events.append(('copy', None))
        return []

    monkeypatch.setattr(module, 'deploy_env',
                        lambda: SimpleNamespace(on_local=False))
    monkeypatch.setattr(module, 'DataBackendContext',
                        lambda *_args: SimpleNamespace())
    monkeypatch.setattr(module.workflow, 'now', lambda: datetime.now())
    monkeypatch.setattr(module.workflow, 'patched', lambda _name: True)
    monkeypatch.setattr(module.workflow, 'uuid4',
                        lambda: SimpleNamespace(hex='abcdef'))
    monkeypatch.setattr(module.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(module.workflow, 'execute_child_workflow',
                        execute_child_workflow)
    monkeypatch.setattr(
        module,
        'run_pooled_activity',
        AsyncMock(return_value=PredictionShardResult(
            state=JobState.SUCCEEDED,
            databricks_uploads=[upload],
        )),
    )
    monkeypatch.setattr(module.ArtifactExportInput, 'from_bp_input',
                        lambda *_args, **_kwargs: SimpleNamespace())

    workflow = module.BatchPredictionJobWorkflow()
    workflow._save_output = AsyncMock()  # type: ignore[method-assign]
    result = await workflow._run(
        cast(BatchPredictionJobInput, job_input),
        cast(MLJobSettings, SimpleNamespace()),
        '/tmp/prediction-config.json',
        module.ResourceHint(min_gpu_slots=1),
    )

    assert events == [('stats', stage), ('copy', None), ('cleanup', None)]
    assert result.status == module.JobStatus.DONE
    assert all(call.args[0].status != module.JobStatus.DONE
               for call in workflow._save_output.await_args_list)


@pytest.mark.asyncio
async def test_durable_output_selects_only_accepted_shard_manifests(
        monkeypatch):
    refs = [
        PredictionManifestRef(
            path=f'/attempt-{index}/_artifact_manifest.json',
            sha256=str(index + 1) * 64,
            size_bytes=100,
            job_id='batchpredictionjob-durable',
            attempt_id=f'attempt-{index}',
            runtime_shard_id=index,
        ) for index in range(2)
    ]
    job_input = SimpleNamespace(
        id='batchpredictionjob-durable',
        training_job_id='trainingjob-test',
        predictive_query=SimpleNamespace(
            name='query',
            graph=SimpleNamespace(name='prediction-graph'),
            target='target',
        ),
        graph_data_compute_input=SimpleNamespace(graph_key='prediction-graph'),
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
        config=SimpleNamespace(
            num_parallel_workers_limit=2,
            outputs=[
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.PREDICTIONS)
            ],
        ),
    )
    selection = SimpleNamespace(
        pointer_path='/selected/_selected_artifacts.json')
    finalizer_args = None
    finalizer_options = None

    async def execute_activity(activity, *, args=None, **kwargs):
        nonlocal finalizer_args, finalizer_options
        if activity is module.finalize_prediction_job_output:
            finalizer_args = args
            finalizer_options = kwargs
            return selection
        return None

    artifact_export_args = None

    async def execute_child_workflow(*_args, args=None, **_kwargs):
        nonlocal artifact_export_args
        artifact_export_args = args
        return []

    export_input = SimpleNamespace()

    def from_bp_input(*args, **kwargs):
        assert args[1] == {}
        assert kwargs['prediction_selection'] is selection
        return export_input

    monkeypatch.setattr(module, 'deploy_env',
                        lambda: SimpleNamespace(on_local=False))
    monkeypatch.setattr(module.workflow, 'now', lambda: datetime.now())
    monkeypatch.setattr(module.workflow, 'patched', lambda _name: True)
    monkeypatch.setattr(module.workflow, 'uuid4',
                        lambda: SimpleNamespace(hex='selection'))
    monkeypatch.setattr(module.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(module.workflow, 'execute_child_workflow',
                        execute_child_workflow)
    monkeypatch.setattr(
        module,
        'run_pooled_activity',
        AsyncMock(side_effect=[
            PredictionShardResult(state=JobState.SUCCEEDED,
                                  artifact_manifest=refs[0]),
            PredictionShardResult(state=JobState.SUCCEEDED,
                                  artifact_manifest=refs[1]),
        ]),
    )
    monkeypatch.setattr(module.ArtifactExportInput, 'from_bp_input',
                        from_bp_input)

    workflow = module.BatchPredictionJobWorkflow()
    workflow._save_output = AsyncMock()  # type: ignore[method-assign]
    await workflow._run(
        cast(BatchPredictionJobInput, job_input),
        cast(MLJobSettings, SimpleNamespace()),
        '/tmp/prediction-config.json',
        module.ResourceHint(min_gpu_slots=2),
    )

    assert finalizer_args is not None
    assert finalizer_args[4] == refs
    assert finalizer_args[5] == 2
    assert finalizer_options is not None
    assert finalizer_options['schedule_to_close_timeout'] == timedelta(hours=6)
    assert finalizer_options['heartbeat_timeout'] == timedelta(minutes=5)
    assert artifact_export_args == [export_input]
