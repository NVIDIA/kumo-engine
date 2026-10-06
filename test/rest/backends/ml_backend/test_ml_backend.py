# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_source as data_source_api
import pytest
from kumo_core.job_config import TrainDirOverrides
from kumo_core.model_plan import (
    ModelPlan,
    OptimizationPlan,
    TrainingJobPlan,
    TrainingTableGenerationPlan,
)
from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import FPID
from kumoapi.jobs import S3PredictionTableSource
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType
from kumoapi.source_table import S3RelativeSource
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr

from io_utils import fsutil
from kumo.config.connector import ConnectorConfig
from kumo.config.graph import (
    ColumnConfig,
    GraphConfig,
    PrimaryKeyConfig,
    TableConfig,
)
from kumo.config.predictive_query import (
    PredictiveQueryConfigBase,
    PredictiveQueryV2Config,
)
from kumo.home import HomeDirManager
from kumo.testing.rest import fake_admin_request_context
from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobConfig,
    GenerateTrainTableJobOutput,
)
from rest.backends.ml_backend.job_queue import JobConcurrencyLimitError
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobInput,
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
)
from rest.config.credentials import AWSFederatedCredentials
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import TrainingJobInput
from rest.config.mlops import (
    JobRegistryQuery,
    MLOpsQueryConfig,
    MLOpsStatsQuery,
    MLOpsStatType,
)
from rest.config.prediction_output import (
    PredictionStorageType,
    S3PredictionOutput,
)
from rest.config.training_job import (
    TrainingJobOutput,
    TrainingJobResponse,
    TrainingJobResults,
    TrainingSpec,
)
from rest.deps.authentication import RequestContext, UserAuthorization
from rest.deps.context import _current_user_ctx
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphIDv1,
    JobEventLogEntry,
    JobFailureInfo,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
    UserID,
    to_graph_id,
)
from statedb.config.predictive_query import FilterOverrideArgs
from statedb.config.timestamp import now_utc
from statedb.db.identities import ConnectorService, IdentityService
from statedb.db.ml_jobs import (
    GenerateTrainTableJobConfig,
    JobInvariantViolationError,
)
from statedb.db.services import StateServices
from temporalio_commons.dataclass_converter import KumoPayloadConverter

_PROJECT_ID = ProjectID.generate()
_CONNECTOR_ID = ConnectorID.generate()


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.mark.parametrize(
    ('principal', 'visible'),
    [
        (UserAuthorization.proxied(), False),
        (UserAuthorization.disabled(), True),
        (UserAuthorization(
            user_id=UserID('user'),
            access_token='token',
            permissions=(),
        ), False),
    ],
    ids=('proxied', 'auth-disabled', 'unassigned'),
)
def test_state_service_job_visibility_uses_explicit_principal(
        principal, visible) -> None:
    token = _current_user_ctx.set(principal)
    try:
        assert MLJobBackend._state_service_job_visible(
            object(),
            ProjectID('project_' + 'a' * 32),
        ) is visible
    finally:
        _current_user_ctx.reset(token)


def _make_graph_config() -> GraphConfig:
    return GraphConfig(
        name="graph",
        data_source_dict={
            str(_CONNECTOR_ID): ConnectorConfig(name=str(_CONNECTOR_ID))
        },
        tables=[
            TableConfig(
                table_name='ACCOUNT',
                source_name='ACCOUNT',
                data_source=str(_CONNECTOR_ID),
                cols=[
                    ColumnConfig(
                        name="K_SYMBOL",
                        dtype=Dtype.string,
                        stype=Stype.categorical,
                    ),
                ],
                pkey=PrimaryKeyConfig(
                    name="ACCOUNT_ID",
                    dtype=Dtype.string,
                    stype=Stype.ID,
                ),
            ),
        ],
    )


def _make_graph_data_compute_input(
    project_id: ProjectID = _PROJECT_ID, ) -> GraphDataComputeInput:
    """Build a v2 ``GraphDataComputeInput`` directly.

    Constructs the v2 ``TableDefinition`` shape natively instead of going
    through ``build_graph_data_compute_input`` (which calls the
    ``@blocking_func`` ``TableDefinition._build_from_inputs`` and rejects
    calls from a running asyncio event loop, breaking these async tests).
    Keys the
    tables dict with v2 ``TableIDv2`` derived from the
    ``TableDefinition.fpid`` so nothing here depends on v1 IDs.
    """
    from kumoapi import source_table as source_table_api
    from kumoapi.data_source import DataSourceType

    from rest.config.table_v2 import Column, S3SourceTable, TableDefinition
    from statedb.config.common import TableIDv2, TableName

    connector_id = _CONNECTOR_ID

    table_def = TableDefinition(
        cols=[
            Column(name='K_SYMBOL', stype=Stype.categorical,
                   dtype=Dtype.string),
        ],
        source_table=S3SourceTable(
            connector_id=str(connector_id),
            source={
                'mode': 'relative',
                'path': 'ACCOUNT'
            },
            file_type=source_table_api.FileType.PARQUET,
            data_source_type=DataSourceType.S3,
        ),
        pkey='ACCOUNT_ID',
    )
    table_id = TableIDv2.compose(pquery_id=None,
                                 table_name=TableName('ACCOUNT'),
                                 fpid=table_def.fpid)

    connector_ref = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=project_id,
    )

    from rest.config.graph_v2 import GraphDefinition
    graph_definition = GraphDefinition(
        tables={TableName('ACCOUNT'): table_def},
        col_groups=[],
    )
    return GraphDataComputeInput(
        graph_display_name='graph',
        graph_key='graph',
        tables={table_id: table_def},
        base_tables={},
        connectors={connector_id: connector_ref},
        graph_definition=graph_definition,
        table_aliases={table_id: TableName('ACCOUNT')},
    )


def _make_predictive_query(
) -> Tuple[PredictiveQueryID, PredictiveQueryV2Config]:
    pq_name = "testq"
    pq_id = PredictiveQueryID(pq_name)
    graph = _make_graph_config()

    pq_config = PredictiveQueryConfigBase(
        graph=graph,
        name=pq_name,
        entity="ACCOUNT.ACCOUNT_ID",
        target="ACCOUNT.K_SYMBOL",
        query_string="PREDICT ACCOUNT.K_SYMBOL FOR EACH ACCOUNT.ACCOUNT_ID",
    ).to_predictive_query_v2_config()
    return pq_id, pq_config


def _make_training_spec(
        advanced_options: Optional[ModelPlan] = None) -> TrainingSpec:
    pq_id, pq = _make_predictive_query()
    return TrainingSpec(
        predictive_query=pq,
        advanced_options=advanced_options or ModelPlan(),
        predictive_query_id=pq_id,
        graph_id=GraphIDv1("graph"),
        split=pq.split,
    )


def _make_batch_pred_input(
    tmp_path: pathlib.Path,
    training_id: TrainingJobID,
    training_spec: TrainingSpec,
    *,
    project_id: ProjectID = _PROJECT_ID,
    output_connector_id: ConnectorID = _CONNECTOR_ID,
) -> BatchPredictionJobInput:
    graph_data_compute_input = _make_graph_data_compute_input(project_id)
    output_connector_ref = ConnectorExecutionRef(
        connector_id=output_connector_id,
        project_id=project_id,
    )
    bp_job_id = BatchPredictionJobID.make_random_id()
    return BatchPredictionJobInput(
        id=bp_job_id,
        project_id=project_id,
        predictive_query=training_spec.predictive_query,
        output_connectors={output_connector_id: output_connector_ref},
        start_time=datetime.now(),
        graph_data_compute_input=graph_data_compute_input,
        config=BatchPredictionJobConfig(
            predictive_query_id=training_spec.predictive_query_id,
            training_job_id=training_id,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            binary_classification_threshold=None,
            num_classes_to_return=None,
            prediction_time=None,
            outputs=[
                S3PredictionOutput(
                    storage_type=PredictionStorageType.S3,
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    file_path="predictions/",
                    connector_id=str(output_connector_id),
                )
            ],
            graph_id=training_spec.graph_id,
            generate_pred_table_job_id=(
                GeneratePredictionTableJobID.as_child_of(bp_job_id)),
            pquery_def_fpid=None,
        ),
    )


class _TestS3WorkloadExchanger:
    def exchange(self, config, _credentials):
        if not isinstance(config, data_source_api.S3IdentityConfig):
            raise TypeError(type(config))
        return AWSFederatedCredentials(
            access_key_id=SecretStr('temporary-access-key'),
            secret_access_key=SecretStr('temporary-secret-key'),
            session_token=SecretStr('temporary-session-token'),
        )


async def _provision_batch_output_connector(
    state_services: StateServices, ) -> tuple[ProjectID, ConnectorID]:
    """Create a valid project-owned S3 output connector for an ML job."""
    group, project = await state_services.groups.ensure_default_resources()
    suffix = str(ConnectorID.generate()).removeprefix('connector_')
    principal = UserAuthorization.disabled()
    exchanger = _TestS3WorkloadExchanger()
    identities = IdentityService(
        state_services.state_db,
        state_services.connector_authorizations,
        identity_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    connectors = ConnectorService(
        state_services.state_db,
        state_services.connector_authorizations,
        connector_validator=lambda *_: None,
        workload_exchanger=exchanger,
    )
    identity = await identities.create(
        data_source_api.CreateIdentityArgs(
            display_name=f'ml-output-{suffix}',
            config=data_source_api.S3IdentityConfig(
                authentication=data_source_api.S3AssumeRoleAuthentication(
                    partition='aws',
                    role_arn=(
                        f'arn:aws:iam::123456789012:role/ml-output-{suffix}'),
                ), ),
        ),
        principal,
    )
    await identities.replace_groups(
        identity.id,
        data_source_api.ReplaceIdentityGroupsArgs(group_ids=[str(group.id)]),
        principal,
    )
    args = data_source_api.CreateConnectorArgs(
        identity_id=identity.id,
        name=f'ml-output-{suffix}',
        config=data_source_api.S3ConnectorResourceConfig(
            target=data_source_api.S3RootedConnectorTarget(
                bucket='kumo-test-batch-output',
                prefix='predictions',
                region='us-west-2',
            ), ),
    )
    authorizations = await connectors.authorize_create(project.id, args,
                                                       principal)
    connector = await connectors.create_from_activity(project.id, args,
                                                      principal.user_id,
                                                      set(authorizations))
    return project.id, ConnectorID(connector.id)


def test_training_job_input_persists_only_connector_execution_refs() -> None:
    training_spec = _make_training_spec()
    graph_input = _make_graph_data_compute_input()
    connector_id = next(iter(graph_input.connectors))
    training_spec.predictive_query.graph.data_source_dict[str(
        connector_id)] = ConnectorConfig(
            name='resolved-config-must-not-enter-history')

    job_input = TrainingJobInput(
        id=TrainingJobID.make_random_id(),
        spec=training_spec,
        start_time=datetime.now(),
        generate_train_table_job_id=GenerateTrainTableJobID.make_random_id(),
        graph_data_compute_input=graph_input,
    )
    payload = KumoPayloadConverter().to_payloads([job_input])[0]

    assert not hasattr(job_input.spec, 'credentials')
    assert job_input.spec.predictive_query.graph.data_source_dict == {}
    assert b'credentials' not in payload.data.lower()
    assert (b'resolved-config-must-not-enter-history' not in payload.data)
    assert list(job_input.graph_data_compute_input.connectors.values()) == [
        graph_input.connectors[connector_id]
    ]


def test_batch_job_input_persists_graph_and_output_connector_refs(
    tmp_path: pathlib.Path, ) -> None:
    job_input = _make_batch_pred_input(
        tmp_path,
        TrainingJobID.make_random_id(),
        _make_training_spec(),
    )
    payload = KumoPayloadConverter().to_payloads([job_input])[0]

    assert job_input.predictive_query.graph.data_source_dict == {}
    assert all(
        isinstance(value, ConnectorExecutionRef)
        for value in job_input.graph_data_compute_input.connectors.values())
    assert all(
        isinstance(value, ConnectorExecutionRef)
        for value in job_input.output_connectors.values())
    assert b'credentials' not in payload.data.lower()


class _AuthorizedHydrationReached(RuntimeError):
    pass


@pytest.mark.parametrize('job_kind', ['training', 'prediction'])
def test_pool_target_table_job_shapes_use_authorized_activity_hydration(
    job_kind: str,
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """Pool target-table inputs carry refs and hydrate only in the activity."""
    from rest.backends.ml_backend.activities import generate_target_table

    training_id = TrainingJobID.make_random_id()
    training_spec = _make_training_spec()
    if job_kind == 'training':
        job_input = TrainingJobInput(
            id=training_id,
            spec=training_spec,
            start_time=datetime.now(),
            generate_train_table_job_id=(
                GenerateTrainTableJobID.as_child_of(training_id)),
            graph_data_compute_input=_make_graph_data_compute_input(),
        )

        def invoke():
            return generate_target_table.generate_training_table(
                job_input, MagicMock())
    else:
        job_input = _make_batch_pred_input(
            tmp_path,
            training_id,
            training_spec,
        )

        def invoke():
            return generate_target_table.generate_prediction_table(
                job_input,
                MagicMock(),
                home_dir=str(tmp_path / 'prediction-job'),
            )

    payload = KumoPayloadConverter().to_payloads([job_input])[0]
    assert b'credentials' not in payload.data.lower()
    assert all(
        isinstance(reference, ConnectorExecutionRef)
        for reference in job_input.graph_data_compute_input.connectors.values())

    observed_references = None

    def stop_at_authorized_hydration(_pquery_config, references):
        nonlocal observed_references
        observed_references = references
        raise _AuthorizedHydrationReached

    monkeypatch.setattr(generate_target_table,
                        'hydrate_predictive_query_for_activity',
                        stop_at_authorized_hydration)
    monkeypatch.setattr(generate_target_table,
                        'get_diskgraph_index_path_for_job',
                        lambda _job: '/tmp/index')
    monkeypatch.setattr(generate_target_table,
                        'save_job_storage_path_to_state_db',
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generate_target_table, 'get_job_base_dir',
                        lambda: str(tmp_path))
    monkeypatch.setattr(generate_target_table, 'get_job_dir',
                        lambda _root, _job_id: str(tmp_path / 'training-job'))

    with pytest.raises(_AuthorizedHydrationReached):
        invoke()
    assert observed_references is job_input.graph_data_compute_input.connectors


@pytest.mark.asyncio
async def test_get_running_job_ids(ml_backend: MLJobBackend,
                                   tmp_path: pathlib.Path,
                                   state_services: StateServices):
    ml_job_service = ml_backend._require_ml_job_service()
    project_id, output_connector_id = await _provision_batch_output_connector(
        state_services)
    training_spec = _make_training_spec()
    graph_data_compute_input = _make_graph_data_compute_input(project_id)

    training_id = await ml_backend.start_training(
        pquery_def_fpid=None, graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec, project_id=project_id)

    # since job ids are randomized if its in the db it was added successfully
    db_job = (await ml_job_service.select_jobs({'job_id': str(training_id)}))[0]
    assert db_job
    assert db_job.pquery_id == 'testq'
    assert (db_job.pquery_str ==
            'PREDICT ACCOUNT.K_SYMBOL FOR EACH ACCOUNT.ACCOUNT_ID')
    assert not db_job.generator_advanced_options
    await ml_backend.mock_training_job_complete(training_id)
    bp_input = _make_batch_pred_input(
        tmp_path,
        training_id,
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )
    await ml_backend.start_batch_prediction(bp_input, project_id=project_id)

    db_job = (await ml_job_service.select_jobs({'job_id': str(bp_input.id)}))[0]
    assert db_job
    assert db_job.pquery_id == 'testq'
    assert not db_job.bp_filters
    assert db_job.trained_job_id == str(training_id)

    ids = await ml_backend.get_running_batch_prediction_job_ids(project_ids=None
                                                                )
    assert len(ids) == 1
    assert ids[0] == bp_input.id

    await ml_backend.mock_bp_job_complete(bp_input.id)
    ids = await ml_backend.get_running_batch_prediction_job_ids(project_ids=None
                                                                )
    assert len(ids) == 0


@pytest.mark.parametrize('reuse_child', [False, True])
@pytest.mark.asyncio
async def test_rejected_batch_prediction_does_not_poison_queue(
    ml_backend: MLJobBackend,
    tmp_path: pathlib.Path,
    state_services: StateServices,
    reuse_child: bool,
) -> None:
    project_id, output_connector_id = await _provision_batch_output_connector(
        state_services)
    training_spec = _make_training_spec()
    job_input = _make_batch_pred_input(
        tmp_path,
        TrainingJobID.make_random_id(),
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )
    job_input.config.num_parallel_workers_limit = (
        ml_backend._settings.job_queue_settings.max_concurrent_gpu_worker_slots
        + 1)
    child_id = job_input.generate_pred_table_job_id
    assert child_id is not None
    service = ml_backend._require_ml_job_service()
    if reuse_child:
        await service.create_generate_pred_table_job(
            id=child_id,
            config=GeneratePredTableJobConfig(
                pquery_id=job_input.config.predictive_query_id,
                plan=job_input.config.pred_table_generation_plan,
                graph_id=job_input.config.graph_id,
                graph_data_version=job_input.config.graph_data_version,
            ),
            project_id=project_id,
        )

    with pytest.raises(JobConcurrencyLimitError, match="requires"):
        await ml_backend.start_batch_prediction(
            job_input,
            project_id=project_id,
        )

    assert await service.get_job_if_exists_generic(job_input.id) is None
    assert ((await service.get_job_if_exists_generic(child_id)
             is not None) == reuse_child)
    queue_status = await ml_backend._job_queue.report_status()
    assert job_input.id not in queue_status.queued_jobs

    if reuse_child:
        return
    retry_input = _make_batch_pred_input(
        tmp_path,
        TrainingJobID.make_random_id(),
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )
    retry_input.config.num_parallel_workers_limit = 1
    await ml_backend.start_batch_prediction(retry_input, project_id=project_id)
    retry = await service.get_job_if_exists_generic(retry_input.id)
    assert retry is not None
    assert retry.execution.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_start_training_with_reserved_job_id_is_idempotent(
    ml_backend: MLJobBackend, ):
    training_spec = _make_training_spec()
    graph_data_compute_input = _make_graph_data_compute_input()
    reserved_job_id = TrainingJobID.make_random_id()
    ml_job_service = ml_backend._require_ml_job_service()

    training_ids_before = set(await ml_job_service.list_job_ids_by_type(
        JobType.TRAINING_JOB))
    train_table_ids_before = set(await ml_job_service.list_job_ids_by_type(
        JobType.GENERATE_TRAIN_TABLE_JOB))

    first_id = await ml_backend.start_training(
        pquery_def_fpid=None,
        graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec,
        project_id=_PROJECT_ID,
        reserved_job_id=reserved_job_id,
    )
    second_id = await ml_backend.start_training(
        pquery_def_fpid=None,
        graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec,
        project_id=_PROJECT_ID,
        reserved_job_id=reserved_job_id,
    )

    assert first_id == reserved_job_id
    assert second_id == reserved_job_id

    training_ids = set(await
                       ml_job_service.list_job_ids_by_type(JobType.TRAINING_JOB
                                                           ))
    train_table_ids = set(await ml_job_service.list_job_ids_by_type(
        JobType.GENERATE_TRAIN_TABLE_JOB))
    assert training_ids - training_ids_before == {reserved_job_id}
    assert train_table_ids - train_table_ids_before == {
        GenerateTrainTableJobID.as_child_of(reserved_job_id)
    }


@pytest.mark.asyncio
async def test_start_training_rejects_cross_project_connector_reference(
    ml_backend: MLJobBackend, ):
    other_project = ProjectID.generate()
    graph_input = _make_graph_data_compute_input(other_project)

    with pytest.raises(ValueError, match="another project"):
        await ml_backend.start_training(
            pquery_def_fpid=None,
            graph_data_compute_input=graph_input,
            training_spec=_make_training_spec(),
            project_id=_PROJECT_ID,
        )


@pytest.mark.asyncio
async def test_training_idempotent_replay_cannot_change_project(
    ml_backend: MLJobBackend, ):
    reserved_job_id = TrainingJobID.make_random_id()
    training_spec = _make_training_spec()
    await ml_backend.start_training(
        pquery_def_fpid=None,
        graph_data_compute_input=_make_graph_data_compute_input(_PROJECT_ID),
        training_spec=training_spec,
        project_id=_PROJECT_ID,
        reserved_job_id=reserved_job_id,
    )

    other_project = ProjectID.generate()
    with pytest.raises(ValueError, match="different project"):
        await ml_backend.start_training(
            pquery_def_fpid=None,
            graph_data_compute_input=_make_graph_data_compute_input(
                other_project),
            training_spec=training_spec,
            project_id=other_project,
            reserved_job_id=reserved_job_id,
        )


@pytest.mark.asyncio
async def test_batch_prediction_rejects_cross_project_output_reference(
    ml_backend: MLJobBackend,
    tmp_path: pathlib.Path,
):
    training_spec = _make_training_spec()
    job_input = _make_batch_pred_input(tmp_path, TrainingJobID.make_random_id(),
                                       training_spec)
    connector_id = next(iter(job_input.output_connectors))
    job_input.output_connectors[connector_id] = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=ProjectID.generate(),
    )

    with pytest.raises(ValueError, match="another project"):
        await ml_backend.start_batch_prediction(
            job_input,
            project_id=_PROJECT_ID,
        )


def _with_custom_prediction_source(job_input: BatchPredictionJobInput,
                                   source_connector_id: ConnectorID,
                                   reference: Any) -> BatchPredictionJobInput:
    job_input.config.generate_pred_table_job_id = None
    job_input.config.custom_pred_table_source = S3PredictionTableSource(
        connector_id=str(source_connector_id),
        source=S3RelativeSource(path="prediction/input.parquet"),
    )
    job_input.custom_pred_table_connector = reference
    return job_input


@pytest.mark.parametrize(
    ('tamper', 'error_type', 'message'),
    [
        ('extra-output', ValueError, 'exactly match'),
        ('graph-project', ValueError, 'different project'),
        ('output-key', ValueError, 'key does not match'),
        ('graph-key', ValueError, 'key does not match'),
        ('graph-type', TypeError, 'Graph connectors'),
        ('output-type', TypeError, 'Output connectors'),
        ('missing-output', ValueError, 'missing its authorized'),
        ('custom-id', ValueError, 'does not match'),
        ('custom-project', ValueError, 'different project'),
        ('missing-custom-ref', ValueError, 'provided together'),
    ],
)
def test_batch_prediction_input_rejects_tampered_connector_refs(
        tmp_path, tamper, error_type, message):
    job_input = _make_batch_pred_input(
        tmp_path,
        TrainingJobID.make_random_id(),
        _make_training_spec(),
    )
    graph_id = next(iter(job_input.graph_data_compute_input.connectors))
    output_id = next(iter(job_input.output_connectors))
    if tamper == 'extra-output':
        extra_id = ConnectorID.generate()
        job_input.output_connectors[extra_id] = ConnectorExecutionRef(
            connector_id=extra_id, project_id=job_input.project_id)
    elif tamper == 'graph-project':
        job_input.graph_data_compute_input.connectors[graph_id] = (
            ConnectorExecutionRef(connector_id=graph_id,
                                  project_id=ProjectID.generate()))
    elif tamper == 'output-key':
        job_input.output_connectors[output_id] = ConnectorExecutionRef(
            connector_id=ConnectorID.generate(),
            project_id=job_input.project_id)
    elif tamper == 'graph-key':
        job_input.graph_data_compute_input.connectors[graph_id] = (
            ConnectorExecutionRef(connector_id=ConnectorID.generate(),
                                  project_id=job_input.project_id))
    elif tamper == 'graph-type':
        job_input.graph_data_compute_input.connectors[graph_id] = cast(
            Any, object())
    elif tamper == 'output-type':
        job_input.output_connectors[output_id] = cast(Any, object())
    elif tamper == 'missing-output':
        job_input.output_connectors.clear()
    else:
        source_id = ConnectorID.generate()
        reference = {
            'custom-id':
            ConnectorExecutionRef(connector_id=ConnectorID.generate(),
                                  project_id=job_input.project_id),
            'custom-project':
            ConnectorExecutionRef(connector_id=source_id,
                                  project_id=ProjectID.generate()),
            'missing-custom-ref':
            None,
        }[tamper]
        _with_custom_prediction_source(job_input, source_id, reference)

    with pytest.raises(error_type, match=message):
        job_input.validate_connector_execution_refs()


@pytest.mark.asyncio
async def test_start_generate_pred_table_with_reserved_job_id_is_idempotent(
    ml_backend: MLJobBackend,
    tmp_path: pathlib.Path,
):
    training_spec = _make_training_spec()
    training_id = TrainingJobID.make_random_id()
    first_input = _make_batch_pred_input(tmp_path, training_id, training_spec)
    first_input.config.pquery_def_fpid = FPID(
        "0123456789abcdef0123456789abcdef")
    gen_pred_job_id = first_input.generate_pred_table_job_id
    assert gen_pred_job_id
    first_input.id = BatchPredictionJobID(gen_pred_job_id)

    second_input = _make_batch_pred_input(tmp_path, training_id, training_spec)
    second_input.id = BatchPredictionJobID(gen_pred_job_id)
    second_input.config.generate_pred_table_job_id = gen_pred_job_id
    second_input.config.pquery_def_fpid = first_input.config.pquery_def_fpid
    ml_job_service = ml_backend._require_ml_job_service()

    assert await ml_job_service.get_generate_pred_table_job(gen_pred_job_id
                                                            ) is None

    first_id = await ml_backend.start_generate_pred_table(
        first_input, project_id=_PROJECT_ID)
    first_entry = await ml_job_service.get_generate_pred_table_job(
        gen_pred_job_id)
    assert first_entry is not None

    second_id = await ml_backend.start_generate_pred_table(
        second_input, project_id=_PROJECT_ID)

    assert first_id == gen_pred_job_id
    assert second_id == gen_pred_job_id

    second_entry = await ml_job_service.get_generate_pred_table_job(
        gen_pred_job_id)
    assert second_entry is not None
    assert second_entry.timestamps.created_at == first_entry.timestamps.created_at


@pytest.mark.asyncio
async def test_ml_db_entries(ml_backend: MLJobBackend, tmp_path: pathlib.Path,
                             state_services: StateServices):
    ml_job_service = ml_backend._require_ml_job_service()
    project_id, output_connector_id = await _provision_batch_output_connector(
        state_services)
    advanced_options = ModelPlan(
        training_job=TrainingJobPlan(num_experiments=3),  # type: ignore
        optimization=OptimizationPlan(max_steps_per_epoch=3))  # type: ignore
    training_spec = _make_training_spec(advanced_options)

    # add pquery advanced options and autotrainer to check that only
    # pquery options are stored in db
    training_spec.predictive_query.forecast_length = 1
    training_spec.predictive_query.timeframe_step = 10
    graph_data_compute_input = _make_graph_data_compute_input(project_id)

    training_id = await ml_backend.start_training(
        pquery_def_fpid=None, graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec, project_id=project_id)

    # since job ids are randomized if its in the db it was added successfully
    db_job = (await ml_job_service.select_jobs({'job_id': str(training_id)}))[0]
    assert db_job
    assert db_job.pquery_id == 'testq'
    assert (db_job.pquery_str ==
            'PREDICT ACCOUNT.K_SYMBOL FOR EACH ACCOUNT.ACCOUNT_ID')
    assert (db_job.generator_advanced_options ==
            'forecast_length: 1\ntimeframe_step: 10\n')

    await ml_backend.mock_training_job_complete(training_id)
    bp_input = _make_batch_pred_input(
        tmp_path,
        training_id,
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )

    # add a BP filter to make sure it is stored in the db
    bp_input.config.filter_override_args = FilterOverrideArgs(
        entity_filter_override='ACCOUNT.ACCOUNT_ID < 500')
    await ml_backend.start_batch_prediction(bp_input, project_id=project_id)

    db_job = (await ml_job_service.select_jobs({'job_id': str(bp_input.id)}))[0]
    assert db_job
    assert db_job.pquery_id == 'testq'
    assert (db_job.bp_filters ==
            'entity_filter_override: ACCOUNT.ACCOUNT_ID < 500\n')
    assert db_job.trained_job_id == str(training_id)
    assert (db_job.pquery_str ==
            'PREDICT ACCOUNT.K_SYMBOL FOR EACH ACCOUNT.ACCOUNT_ID')


@pytest.mark.asyncio
async def test_override_model(ml_backend: MLJobBackend, tmp_path: pathlib.Path,
                              ctx: RequestContext):
    training_spec = _make_training_spec()
    graph_data_compute_input = _make_graph_data_compute_input()
    training_id = await ml_backend.start_training(
        pquery_def_fpid=None, graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec, project_id=_PROJECT_ID)

    # TODO(kiriti): Assert that the sqlite update was successful

    invalid_overrides = TrainDirOverrides(
        str(tmp_path / "missing-train-dir"),
        str(tmp_path / "missing-inference-dir"),
    )
    valid_overrides = TrainDirOverrides(str(tmp_path), str(tmp_path))

    # Test invalid training job ID
    with pytest.raises(RuntimeError, match="does not exist"):
        await ml_backend.override_model(TrainingJobID("invalidID"),
                                        valid_overrides)

    # Test invalid TrainDirOverrides
    with pytest.raises(ValueError, match="override path does not exist"):
        await ml_backend.override_model(training_id, invalid_overrides)

    # Test valid TrainDirOverrides
    await ml_backend.override_model(training_id, valid_overrides)
    assert fsutil.exists(
        HomeDirManager(
            await
            ml_backend.get_job_home_dir(training_id)).train_dir_overrides_file)
    job = await ml_backend.get_training_job(ctx, training_id)
    assert job is not None
    assert job.execution.status == JobStatus.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize('job_kind', ['batch-prediction', 'training'])
async def test_sanitize_job_output_stacktraces(ml_backend: MLJobBackend,
                                               job_kind: str):
    ml_job_service = ml_backend._require_ml_job_service()
    if job_kind == 'batch-prediction':
        id = BatchPredictionJobID.make_random_id()
        await ml_job_service.create_batch_prediction_job(
            id=id,
            now=now_utc(),
            config=BatchPredictionJobConfig(
                PredictiveQueryID('some_pquery'),
                TrainingJobID.make_random_id(),
                TaskType.REGRESSION,
                generate_pred_table_job_id=(
                    GeneratePredictionTableJobID.as_child_of(id)),
                graph_id=GraphIDv1('test_graph'),
            ),
            project_id=_PROJECT_ID,
        )
        entry = (await
                 ml_job_service.get_batch_prediction_jobs([id],
                                                          project_ids=None))[0]
        entry.execution = BatchPredictionJobOutput(
            status=JobStatus.FAILED,
            log=JobEventLogEntry.failed(),
            failure_info=JobFailureInfo(message='message',
                                        stacktrace='stacktrace'),
        )
    else:
        id, _ = await ml_job_service.create_training_job(
            pquery_id=PredictiveQueryID('dummy_pquery'),
            pquery_def_fpid=None,
            graph_id=to_graph_id('dummy_graph'),
            now=now_utc(),
            advanced_options=ModelPlan(),
            project_id=_PROJECT_ID,
        )
        entry = await ml_job_service.get_training_job(id)
        assert entry is not None
        entry.execution = TrainingJobOutput(
            status=JobStatus.FAILED,
            log=JobEventLogEntry.failed(),
            failure_info=JobFailureInfo(message='message',
                                        stacktrace='stacktrace'),
        )
    await ml_job_service.modify_job_if_open_generic(entry)

    def stacktrace(job) -> str:
        failure = (job.failure_info if job_kind == 'batch-prediction' else
                   job.execution.failure_info)
        assert failure is not None
        return failure.stacktrace

    get_one = cast(
        Any,
        ml_backend.get_batch_prediction_job
        if job_kind == 'batch-prediction' else ml_backend.get_training_job,
    )
    get_many = cast(
        Any,
        ml_backend.get_batch_prediction_jobs
        if job_kind == 'batch-prediction' else ml_backend.get_training_jobs,
    )
    user_ctx = fake_admin_request_context()
    user_job = await get_one(user_ctx, id)
    jobs = await get_many(user_ctx, {id})

    assert user_job is not None
    assert stacktrace(user_job) == ''
    assert len(jobs) == 1
    job = jobs[id]
    assert job is not None
    assert stacktrace(job) == ''


@pytest.mark.asyncio
async def test_training_responses_preserve_execution_start_time(
    ml_backend: MLJobBackend,
) -> None:
    ml_job_service = ml_backend._require_ml_job_service()
    job_id, _ = await ml_job_service.create_training_job(
        pquery_id=PredictiveQueryID('dummy_pquery'),
        pquery_def_fpid=None,
        graph_id=to_graph_id('dummy_graph'),
        now=now_utc(),
        advanced_options=ModelPlan(),
        project_id=_PROJECT_ID,
    )
    job = await ml_job_service.get_training_job(job_id)
    assert job is not None
    expected_start_time = now_utc()
    job.execution.start_time = expected_start_time

    # The public entry points set the principal context var before delegating
    # to the response builders; mirror that when calling them directly.
    ctx = fake_admin_request_context()
    token = _current_user_ctx.set(ctx.user_authorization)
    try:
        response = ml_backend._training_job_response_from_entry(
            ctx, job_id, job)
        assert response is not None
        assert response.execution.start_time == expected_start_time
        assert job.to_generic(ctx).metadata.start_time == expected_start_time
    finally:
        _current_user_ctx.reset(token)


@pytest.mark.asyncio
async def test_write_time_invariant_gate_training_done_without_results(
    ml_backend: MLJobBackend, ):
    """A DONE training job missing ``results`` must not be persisted.

    Restores the pydantic-v1 write-time guard that the v2 migration
    downgraded to a warning. Covers both terminal-persist chokepoints:
    the typed ``save_training_job_output`` and the generic
    ``modify_job_if_open_generic`` path.
    """
    ml_job_service = ml_backend._require_ml_job_service()
    id, _ = await ml_job_service.create_training_job(
        pquery_id=PredictiveQueryID('dummy_pquery'),
        pquery_def_fpid=None,
        graph_id=to_graph_id('dummy_graph'),
        now=now_utc(),
        advanced_options=ModelPlan(),
        project_id=_PROJECT_ID,
    )

    broken = TrainingJobOutput(
        status=JobStatus.DONE,
        log=JobEventLogEntry.done(),
        results=None,  # <- invariant violation for a DONE job
    )

    # (a) typed terminal-output persist path raises.
    with pytest.raises(JobInvariantViolationError, match="results"):
        await ml_job_service.save_training_job_output(id, broken)

    # (a') generic persist chokepoint raises too.
    entry = await ml_job_service.get_training_job(id)
    assert entry is not None
    entry.execution = broken
    with pytest.raises(JobInvariantViolationError, match="results"):
        await ml_job_service.modify_job_if_open_generic(entry)

    # The broken record never persisted: the job is still non-terminal.
    entry = await ml_job_service.get_training_job(id)
    assert entry is not None
    assert entry.execution.status != JobStatus.DONE

    # (b) a valid terminal job persists fine.
    valid = TrainingJobOutput(
        status=JobStatus.DONE,
        log=JobEventLogEntry.done(),
        results=TrainingJobResults(total_training_time_sec=1,
                                   total_iterations=1),
    )
    await ml_job_service.save_training_job_output(id, valid)
    entry = await ml_job_service.get_training_job(id)
    assert entry is not None
    assert entry.execution.status == JobStatus.DONE
    assert entry.execution.results is not None


@pytest.mark.asyncio
async def test_write_time_invariant_gate_bp_failed_without_failure_info(
    ml_backend: MLJobBackend, ):
    """A FAILED batch-prediction job missing ``failure_info`` must not persist.

    Exercises the batch-prediction branch of ``_save_job_entry_if_open``
    reached via ``modify_job_if_open_generic``.
    """
    ml_job_service = ml_backend._require_ml_job_service()
    id = BatchPredictionJobID.make_random_id()
    await ml_job_service.create_batch_prediction_job(
        id=id,
        now=now_utc(),
        config=BatchPredictionJobConfig(
            PredictiveQueryID('some_pquery'), TrainingJobID.make_random_id(),
            TaskType.REGRESSION, generate_pred_table_job_id=(
                GeneratePredictionTableJobID.as_child_of(id)),
            graph_id=GraphIDv1('test_graph')),
        project_id=_PROJECT_ID,
    )
    entry = (await
             ml_job_service.get_batch_prediction_jobs([id],
                                                      project_ids=None))[0]
    entry.execution = BatchPredictionJobOutput(
        status=JobStatus.FAILED,
        log=JobEventLogEntry.failed(),
        failure_info=None,  # <- invariant violation for a FAILED job
    )
    with pytest.raises(JobInvariantViolationError, match="failure_info"):
        await ml_job_service.modify_job_if_open_generic(entry)

    # A valid FAILED job (failure_info present) persists.
    entry = (await
             ml_job_service.get_batch_prediction_jobs([id],
                                                      project_ids=None))[0]
    entry.execution = BatchPredictionJobOutput(
        status=JobStatus.FAILED,
        log=JobEventLogEntry.failed(),
        failure_info=JobFailureInfo(message="boom", stacktrace=""),
    )
    await ml_job_service.modify_job_if_open_generic(entry)
    entry = (await
             ml_job_service.get_batch_prediction_jobs([id],
                                                      project_ids=None))[0]
    assert entry.execution.status == JobStatus.FAILED
    assert entry.execution.failure_info is not None


def test_read_path_incomplete_terminal_output_does_not_raise():
    """Re-validating an incomplete terminal output stays warn-only.

    The ``mode='after'`` validator (which v2 re-runs on read/response paths)
    must NOT raise; only the write chokepoint enforces. Constructing a DONE
    output without ``results`` -- and re-validating it by nesting it into a
    parent model on the response path -- must succeed.
    """
    incomplete = TrainingJobOutput(
        status=JobStatus.DONE,
        log=JobEventLogEntry.done(),
        results=None,
    )
    # Construction (which runs the mode='after' validator) did not raise.
    assert incomplete.status == JobStatus.DONE
    assert incomplete.terminal_invariant_violations()  # diagnostic still flags

    # Re-wrapping into a parent model (response path) re-runs the validator
    # under pydantic v2; this must stay warn-only and not raise.
    response = TrainingJobResponse(
        id=TrainingJobID.make_random_id(),
        execution=incomplete,
        project_id=ProjectID.generate(),
    )
    assert response.execution.results is None


@pytest.mark.asyncio
async def test_mlops_query(
    ml_backend: MLJobBackend,
    tmp_path: pathlib.Path,
    state_services: StateServices,
):
    project_id, output_connector_id = await _provision_batch_output_connector(
        state_services)
    # start some jobs to have db entries
    advanced_options = ModelPlan.from_default(TaskType.REGRESSION,
                                              QueryType.TEMPORAL,
                                              has_train_table_time_col=True,
                                              run_mode=RunMode.DEBUG)
    training_spec = _make_training_spec(advanced_options)

    # add pquery advanced options and autotrainer to check that only
    # pquery options are stored in db
    training_spec.predictive_query.forecast_length = 1
    training_spec.predictive_query.timeframe_step = 10
    graph_data_compute_input = _make_graph_data_compute_input(project_id)

    training_id = await ml_backend.start_training(
        pquery_def_fpid=None, graph_data_compute_input=graph_data_compute_input,
        training_spec=training_spec, project_id=project_id)

    await ml_backend.mock_training_job_complete(training_id)
    bp_input = _make_batch_pred_input(
        tmp_path,
        training_id,
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )
    await ml_backend.start_batch_prediction(bp_input, project_id=project_id)

    await ml_backend.mock_training_job_complete(training_id)

    bp_input_2 = _make_batch_pred_input(
        tmp_path,
        training_id,
        training_spec,
        project_id=project_id,
        output_connector_id=output_connector_id,
    )
    await ml_backend.start_batch_prediction(bp_input_2, project_id=project_id)

    # assemble and execute test queries
    job_query = JobRegistryQuery(
        job_filters={'job_type': JobType.BATCH_PREDICTION_JOB.value})
    stat_query = MLOpsStatsQuery(stat_request=[
        MLOpsStatType.PREDICTION_COUNT,
        MLOpsStatType.START_TIME,
        MLOpsStatType.TRAINING_VERSION,
        MLOpsStatType.TABLE_STATS,
        MLOpsStatType.TRAINED_JOB_ID,
        MLOpsStatType.COL_PSI_SCORES,
    ])
    query = MLOpsQueryConfig(job_query=job_query, stat_query=stat_query)
    stats = await ml_backend.get_mlops_jobs_and_db_stats(query)

    assert stats
    assert training_id not in stats.train_stats.keys()
    assert bp_input.id in stats.bp_stats.keys()
    assert bp_input_2.id in stats.bp_stats.keys()
    assert (stats.bp_stats[bp_input_2.id][MLOpsStatType.TRAINED_JOB_ID] ==
            training_id)

    # test query with invalid filter dropped
    job_query = JobRegistryQuery(job_filters={
        'job_type': JobType.BATCH_PREDICTION_JOB.value,
        'fake_col': 'xyz'
    })

    query = MLOpsQueryConfig(job_query=job_query, stat_query=stat_query)
    stats2 = await ml_backend.get_mlops_jobs_and_db_stats(query)
    # should be exact same as result with only the valid filter
    assert stats2 == stats

    # test query with invalid filter dropped
    job_query = JobRegistryQuery(job_filters={})
    query = MLOpsQueryConfig(job_query=job_query, stat_query=stat_query)
    stats3 = await ml_backend.get_mlops_jobs_and_db_stats(query)
    assert stats3
    # invalid filter ignored -> nonzero results returned
    assert stats3.bp_stats

    # filter on a value that won't ever appear -> zero results
    job_query = JobRegistryQuery(job_filters={'job_id': 'xyz'})
    stat_query = MLOpsStatsQuery(stat_request=[MLOpsStatType.PREDICTION_COUNT])
    query = MLOpsQueryConfig(job_query=job_query, stat_query=stat_query)
    stats = await ml_backend.get_mlops_jobs_and_db_stats(query)
    assert stats
    assert len(stats.train_stats.keys()) == 0
    assert len(stats.bp_stats.keys()) == 0


# Substring written by finalize_job's safety-net when it cannot recover a real
# error. The point-of-failure terminalization in the parent workflows exists so
# this generic text never masks a real validation error.
_GENERIC_TERMINAL_MSG = 'did not report a terminal status'


@pytest.mark.asyncio
@pytest.mark.parametrize('job_kind', ['batch-prediction', 'training'])
async def test_validation_failure_early_return_surfaces_real_error(
    ml_backend: MLJobBackend,
    job_kind: str,
):
    """Both parent workflows freeze their real validation error."""
    from temporalio.client import WorkflowExecutionStatus

    from kumo.validation.validation import ValidationError, ValidationResponse

    ml_job_service = ml_backend._require_ml_job_service()
    title, detail = {
        'batch-prediction': (
            'Pred Table Mismatch',
            'Prediction table is incompatible with the graph',
        ),
        'training': (
            'Custom Train Table Unsupported',
            'Custom training table is not supported here',
        ),
    }[job_kind]
    validation_response = ValidationResponse(
        errors=[ValidationError(title, detail)], warnings=[], info_items=[])

    if job_kind == 'batch-prediction':
        id = BatchPredictionJobID.make_random_id()
        await ml_job_service.create_batch_prediction_job(
            id=id,
            now=now_utc(),
            config=BatchPredictionJobConfig(
                PredictiveQueryID('some_pquery'),
                TrainingJobID.make_random_id(),
                TaskType.REGRESSION,
                generate_pred_table_job_id=(
                    GeneratePredictionTableJobID.as_child_of(id)),
                graph_id=GraphIDv1('test_graph'),
            ),
            project_id=_PROJECT_ID,
        )
        job_output = BatchPredictionJobOutput.failed(
            validation=validation_response,
            workflow_end_state=WorkflowExecutionStatus.COMPLETED,
        )
        entry = (await
                 ml_job_service.get_batch_prediction_jobs([id],
                                                          project_ids=None))[0]
    else:
        id, _ = await ml_job_service.create_training_job(
            pquery_id=PredictiveQueryID('dummy_pquery'),
            pquery_def_fpid=None,
            graph_id=to_graph_id('dummy_graph'),
            now=now_utc(),
            advanced_options=ModelPlan(),
            project_id=_PROJECT_ID,
        )
        job_output = TrainingJobOutput.failed(
            validation=validation_response,
            workflow_end_state=WorkflowExecutionStatus.COMPLETED,
        )
        entry = await ml_job_service.get_training_job(id)
        assert entry is not None

    entry = cast(Any, entry)
    entry.execution = job_output
    await ml_job_service.modify_job_if_open_generic(entry)

    if job_kind == 'batch-prediction':
        entry = (await ml_job_service.get_batch_prediction_jobs(
            [cast(BatchPredictionJobID, id)], project_ids=None))[0]
    else:
        entry = await ml_job_service.get_training_job(cast(TrainingJobID, id))
        assert entry is not None
    assert entry.execution.status == JobStatus.FAILED
    assert entry.execution.status.is_terminal
    assert entry.execution.workflow_end_state is not None
    assert entry.execution.failure_info is not None
    msg = entry.execution.failure_info.message
    assert title in msg
    assert _GENERIC_TERMINAL_MSG not in msg


@pytest.mark.asyncio
async def test_bp_parent_step5_failed_child_preserves_real_error(
    ml_backend: MLJobBackend, ):
    """The child freezes the BP row terminal FAILED with the real error; a
    later success-shaped OK overwrite must be rejected by the open-gate, so the
    real error survives and finalize never sees a non-terminal row to mask."""
    from temporalio.client import WorkflowExecutionStatus

    ml_job_service = ml_backend._require_ml_job_service()
    id = BatchPredictionJobID.make_random_id()
    await ml_job_service.create_batch_prediction_job(
        id=id,
        now=now_utc(),
        config=BatchPredictionJobConfig(
            PredictiveQueryID('some_pquery'), TrainingJobID.make_random_id(),
            TaskType.REGRESSION, generate_pred_table_job_id=(
                GeneratePredictionTableJobID.as_child_of(id)),
            graph_id=GraphIDv1('test_graph')),
        project_id=_PROJECT_ID,
    )

    # What the child writes on its graceful shard-failure path.
    real_err = ('kumo-pred subprocess exited non-zero for job '
                f'{id} on shard(s) [0]; see pool worker logs for details')
    child_output = BatchPredictionJobOutput.failed(
        err_msg=real_err, workflow_end_state=WorkflowExecutionStatus.COMPLETED)
    entry = (await
             ml_job_service.get_batch_prediction_jobs([id],
                                                      project_ids=None))[0]
    entry.execution = child_output
    await ml_job_service.modify_job_if_open_generic(entry)

    # The parent's success-shaped overwrite must be rejected by the open-gate.
    ok_overwrite = BatchPredictionJobOutput(
        status=JobStatus.DONE, log=JobEventLogEntry.done(),
        results=BatchPredictionJobResults(key_cols=[]),
        workflow_end_state=WorkflowExecutionStatus.COMPLETED)
    await ml_job_service.save_batch_prediction_job_output(id, ok_overwrite)

    entry = (await
             ml_job_service.get_batch_prediction_jobs([id],
                                                      project_ids=None))[0]
    assert entry.execution.status == JobStatus.FAILED
    assert entry.execution.status.is_terminal
    assert entry.execution.workflow_end_state is not None
    assert entry.execution.failure_info is not None
    msg = entry.execution.failure_info.message
    assert 'kumo-pred subprocess exited non-zero' in msg
    assert _GENERIC_TERMINAL_MSG not in msg


@pytest.mark.asyncio
@pytest.mark.parametrize('job_kind', ['batch-prediction', 'training'])
async def test_parent_finally_finalizes_on_pre_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
    job_kind: str,
):
    """Both parents finalize rows even when initialization fails."""
    from rest.backends.ml_backend.workflows import (
        base_training_workflow,
        batch_prediction_workflow,
    )

    if job_kind == 'batch-prediction':
        workflow_module = batch_prediction_workflow
        workflow_type = batch_prediction_workflow.BatchPredictionParentWorkflow
        job_id = BatchPredictionJobID.make_random_id()
    else:
        workflow_module = base_training_workflow
        workflow_type = base_training_workflow.BaseTrainingWorkflow
        job_id = TrainingJobID.make_random_id()

    finalized: list = []

    async def _spy_dequeue(job_id):
        finalized.append(job_id)

    async def _run_passthrough(self, coro):
        # The real ``run_until_canceled`` needs a Temporal workflow context;
        # outside the sandbox we just await the wrapped coroutine so the
        # pre-enqueue exception propagates exactly as it would in production.
        return await coro

    async def _raise_pre_enqueue(self, job_input, ml_job_settings):
        # Simulate init_job_generic/enqueue_job failing before the job is
        # enqueued (e.g. persist_job_inputs activity exhausts its retries).
        raise RuntimeError('init failed before enqueue')

    monkeypatch.setattr(workflow_module, 'dequeue_job', _spy_dequeue)
    monkeypatch.setattr(workflow_type, 'run_until_canceled', _run_passthrough)
    monkeypatch.setattr(workflow_type, '_run', _raise_pre_enqueue)

    class _StubInput:
        id = job_id

    wf = cast(Any, workflow_type())
    with pytest.raises(RuntimeError, match='init failed before enqueue'):
        await wf.run(_StubInput(), ml_job_settings=None)

    assert finalized == [job_id]


def _build_ml_backend_activities(ml_backend: MLJobBackend):
    """Construct an ``MLBackendActivities`` bound to the test ML backend.

    Mirrors ``rest/backends/ml_backend/worker.py`` wiring but over the
    in-process StateDB the ``ml_backend`` fixture already opened. Slack
    notification is replaced with a no-op so the finalize tests stay hermetic
    and exercise only the terminalization logic.
    """
    from rest.backends.data_backend.context import DataBackendContext
    from rest.backends.ml_backend.activities import ml_backend_activities
    from rest.backends.ml_backend.job_queue import StateDbJobQueue
    from rest.backends.ml_backend.settings import MLJobQueueSettings
    from rest.settings import DataBackendSettings
    from service_framework.deploy_env_config import deploy_env

    state_db = ml_backend._state_db
    assert state_db is not None
    job_queue_settings = MLJobQueueSettings.parse_from_env()
    activities = ml_backend_activities.MLBackendActivities(
        StateDbJobQueue(state_db, job_queue_settings),
        ml_backend,
        ml_backend._require_ml_job_service(),
        DataBackendContext(deploy_env(),
                           DataBackendSettings(disable_async_ingestion=True)),
        job_queue_settings,
        slack_url=None,
    )

    async def _noop_failure_notification(*args, **kwargs):
        return None

    activities._send_failure_notification = _noop_failure_notification
    return activities


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('validation_outcome', 'expected_message', 'uses_generic'),
    [
        ('missing', _GENERIC_TERMINAL_MSG, True),
        ('unreadable', _GENERIC_TERMINAL_MSG, True),
        ('failed', 'Graph Refresh Failed', False),
    ],
    ids=('missing-validation', 'unreadable-validation', 'failed-validation'),
)
async def test_finalize_job_terminalizes_nonterminal_rows(
    ml_backend: MLJobBackend,
    monkeypatch: pytest.MonkeyPatch,
    validation_outcome: str,
    expected_message: str,
    uses_generic: bool,
):
    """Finalization uses a real validation error when one can be read."""
    from kumo.validation.validation import ValidationError, ValidationResponse

    ml_job_service = ml_backend._require_ml_job_service()
    id, _ = await ml_job_service.create_training_job(
        pquery_id=PredictiveQueryID('dummy_pquery'),
        pquery_def_fpid=None,
        graph_id=to_graph_id('dummy_graph'),
        now=now_utc(),
        advanced_options=ModelPlan(),
        project_id=_PROJECT_ID,
    )

    activities = _build_ml_backend_activities(ml_backend)
    if validation_outcome == 'failed':
        response = ValidationResponse(
            errors=[
                ValidationError(
                    'Graph Refresh Failed',
                    'Table ingestion produced no rows',
                )
            ],
            warnings=[],
            info_items=[],
        )
        validation_read = AsyncMock(return_value=response)
    elif validation_outcome == 'unreadable':
        validation_read = AsyncMock(
            side_effect=ValueError('corrupt validation json'))
    else:
        validation_read = AsyncMock(return_value=None)
    monkeypatch.setattr(ml_backend, 'get_job_validation_response',
                        validation_read)

    await activities.finalize_job(id)

    entry = await ml_job_service.get_training_job(id)
    assert entry is not None
    assert entry.execution.status == JobStatus.FAILED
    assert entry.execution.status.is_terminal
    assert entry.execution.workflow_end_state is not None
    assert entry.execution.failure_info is not None
    msg = entry.execution.failure_info.message
    assert expected_message in msg
    assert (_GENERIC_TERMINAL_MSG in msg) is uses_generic
    assert entry.execution.log.detail == f'TRAINING_JOB failed: {msg}'


@pytest.mark.asyncio
async def test_a_registered_custom_table_survives_the_job_record(
    ml_backend: MLJobBackend, ):
    """`TrainingTable.from_source_table` was a no-op against the cloud.

    The SDK reconstructs a `TrainingTable` from a job ID alone and asks the
    job record what was registered. The record could not answer:
    `GenerateTrainTableJobConfig` had no field for it, the statedb call took
    the argument and wrote nothing durable, and the resource mapper rebuilt
    `config` without it. So the rows were read into Kumo storage, the job
    completed, and the training run that followed read `custom_train_table`
    as `None` and generated its own table -- reporting success on rows the
    caller never supplied. This pins the whole chain: config -> row ->
    resource.
    """
    from kumoapi.jobs import CustomTrainingTable
    from kumoapi.source_table import FileType, S3SourceTable
    from kumoapi.train import TrainingTableSpec

    registered = CustomTrainingTable(
        source_table=S3SourceTable(
            connector_id=str(_CONNECTOR_ID),
            source=S3RelativeSource(path='labels'),
            file_type=FileType.PARQUET,
        ),
        table_mod_spec=TrainingTableSpec(weight_col='weight',
                                         split_col='SPLIT'),
    )
    ml_job_service = ml_backend._require_ml_job_service()

    id = await ml_job_service.create_generate_train_table_job(
        config=GenerateTrainTableJobConfig(
            pquery_id=PredictiveQueryID("p"),
            plan=TrainingTableGenerationPlan(),
            graph_id=to_graph_id('dummy_graph'),
            custom_train_table=registered,
        ),
        project_id=_PROJECT_ID,
    )

    entry = await ml_job_service.get_generate_train_table_job(
        id, project_ids=None)
    assert entry is not None
    # Durable: read back out of the job row, not out of the caller's object.
    assert entry.config.custom_train_table == registered
    assert entry.custom_train_table_used is True

    resources = await ml_backend.get_generate_train_table_jobs(
        {id}, project_ids=None)
    recovered = resources[0].config.custom_train_table
    assert recovered is not None
    assert recovered.table_mod_spec.weight_col == 'weight'
    assert recovered.table_mod_spec.split_col == 'SPLIT'
    assert resources[0].custom_train_table_used is True


@pytest.mark.asyncio
async def test_a_generated_job_says_it_registered_nothing(
    ml_backend: MLJobBackend, ):
    """The other half of the same answer: a generating job carries no custom
    table and says so, so a reader can tell that apart from a record that
    cannot say -- which is what the SDK refuses on.
    """
    ml_job_service = ml_backend._require_ml_job_service()

    id = await ml_job_service.create_generate_train_table_job(
        config=GenerateTrainTableJobConfig(
            pquery_id=PredictiveQueryID("p"),
            plan=TrainingTableGenerationPlan(),
            graph_id=to_graph_id('dummy_graph'),
        ),
        project_id=_PROJECT_ID,
    )

    resources = await ml_backend.get_generate_train_table_jobs(
        {id}, project_ids=None)
    assert resources[0].config.custom_train_table is None
    assert resources[0].custom_train_table_used is False


@pytest.mark.asyncio
async def test_gen_train_table_public_end_time_is_execution_completion_time(
    ml_backend: MLJobBackend, ):
    """A terminal gen-train-table job's public ``end_time`` must be the real
    completion time (``execution.last_updated_at``), not the stale DB-row
    ``timestamps.updated_at`` (finding 024).

    ``timestamps.updated_at`` is set once at creation and never bumped for this
    job type, so reading it as ``end_time`` reports ``end_time == created_at``
    and a bogus 0-second duration for a job that actually ran. The correct
    source is ``execution.last_updated_at`` -- the same value the sibling
    ``to_generic()`` path and the training/prediction routers use.
    """
    ml_job_service = ml_backend._require_ml_job_service()
    created = now_utc() - timedelta(hours=1)
    completion = created + timedelta(seconds=13)
    id = await ml_job_service.create_generate_train_table_job(
        config=GenerateTrainTableJobConfig(
            pquery_id=PredictiveQueryID("p"),
            plan=TrainingTableGenerationPlan(),
            graph_id=to_graph_id('dummy_graph'),
        ),
        now=created,
        project_id=_PROJECT_ID,
    )
    # The terminal-completion write stamps execution.last_updated_at from
    # now_utc(); job.timestamps is left untouched (stays at created_at).
    with patch("statedb.db.ml_jobs.now_utc", return_value=completion):
        await ml_job_service.save_generate_train_table_job_output(
            id, GenerateTrainTableJobOutput.failed(err_msg="boom"))

    # The stale DB timestamp really is still created_at, distinct from the
    # completion time -- so selecting the wrong field is observable.
    entry = await ml_job_service.get_generate_train_table_job(
        id, project_ids=None)
    assert entry is not None
    assert entry.timestamps.updated_at == created
    assert entry.execution.last_updated_at == completion

    resources = await ml_backend.get_generate_train_table_jobs({id},
                                                               project_ids=None)
    assert len(resources) == 1
    report = resources[0].job_status_report
    assert report.status.is_terminal
    # A job terminalized before admission has a zero-length execution interval.
    assert report.start_time == completion
    # The fix: end_time is the true completion time, not the stale created_at.
    # Pre-fix this asserted-against value (created) is what was reported, giving
    # a 0-second span.
    assert report.end_time == completion
    assert report.end_time != created
