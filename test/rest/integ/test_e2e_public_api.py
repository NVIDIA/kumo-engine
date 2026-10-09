# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import hashlib
import os
from copy import deepcopy
from http import HTTPStatus
from test.rest.integ.fixture_connector_args import (
    bind_test_client_project,
    duplicate_fixture_sources,
    provision_fixture_connectors,
)
from test.rest.integ.public_response_reader import (
    flatten_public_parquet_refs,
    read_public_response_bytes,
)
from typing import List, Optional, Tuple

import kumoapi.data_snapshot as snapshot_interfaces
import kumoapi.jobs as api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from kumo_core.types import TaskType
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.data_source import DEFAULT_PROJECT_ID
from kumoapi.graph import GraphResource
from kumoapi.model_plan import RunMode
from kumoapi.source_table import S3RelativeSource, S3SourceTable

from io_utils import fsutil
from io_utils.json import load_json_async
from kumo.home import HomeDirManager
from kumo.stats.train_table_stats import SplitTrainTableStats
from kumo.testing.rest import (
    get_and_expect_error,
    get_and_expect_success,
    post_and_expect_success,
    randomize_fixture_ids_v2,
)
from kumo.validation import ValidationResponse
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.workflows import GenerateTrainTableResult
from rest.config.api_command import FixtureDefinition
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.graph_v2 import ColumnKey, ColumnKeyGroup, GraphDefinition
from rest.config.predictive_query_v2 import (
    PQueryMetadataResponse,
    PQueryResource,
)
from rest.config.status import EmptyResponse, IDResponse
from rest.public.routers.pquery import _public_pquery_from_internal
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2
from rest.routers.gen_pred_table import GeneratePredictionTableResponse
from rest.routers.gen_train_table import GenerateTrainTableResponse
from rest.testing.integration import (
    wait_for_prediction_job_to_finish,
    wait_for_training_job_to_finish,
)
from statedb.config.common import (
    BatchPredictionJobID,
    DataSnapshot,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphID,
    GraphIDv2,
    JobStatus,
    PredictiveQueryID,
    TrainingJobID,
    to_graph_id,
)
from temporalio_commons import TemporalClient

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in ``test/rest/integ/conftest.py``. Without ``_pool_subprocesses`` the
# pool controller / worker subprocesses never spawn, so ``place_snapshot_activity``
# (routed via ``DiskgraphIngestWorkflow`` -> ``pool.wf_coordinator``) hangs on a
# no-worker AutoRPC queue and the workflow times out.
# This suite covers public-API e2e, not storage-backend coverage —
# only test_diskgraph_e2e.py needs the [s3] parametrize.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


async def _wait_for_graph_snapshot_ingest_done(
    public_api_client: TestClient,
    snapshot_id: GraphSnapshotID,
    *,
    timeout_seconds: int = 300,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_snapshot = None
    while asyncio.get_running_loop().time() < deadline:
        last_snapshot = get_and_expect_success(
            public_api_client, f'/v1/graphsnapshots/{snapshot_id}', {},
            snapshot_interfaces.GraphSnapshotResource)
        stage = last_snapshot.stages.get(
            snapshot_interfaces.GraphSnapshotStage.INGEST)
        if stage is not None and stage.status.is_terminal:
            assert stage.status == snapshot_interfaces.JobStatus.DONE, (
                f'Graph snapshot {snapshot_id} ingest ended with '
                f'{stage.status}: {last_snapshot}')
            return
        await asyncio.sleep(1)
    raise TimeoutError(f'Graph snapshot {snapshot_id} ingest did not finish in '
                       f'{timeout_seconds}s; last_snapshot={last_snapshot}')


def _get_registered_pquery_graph_id(
    public_api_client: TestClient,
    pquery_id: str,
) -> GraphID:
    graph = get_and_expect_success(
        public_api_client,
        f'/v1/graphs/{pquery_id}_graph',
        {},
        GraphResource,
    )
    return GraphIDv2(graph.id)


@pytest_asyncio.fixture(scope='function')
async def setup_connector(
    public_api_client: TestClient,
    pquery_db_dump: FixtureDefinition,
):
    randomized = randomize_fixture_ids_v2(pquery_db_dump)
    # ``setup_connector`` and the test body share this function-scoped
    # fixture. Canonicalize it in place so the project connectors provisioned
    # here and the graph registered by the test carry the same durable IDs.
    pquery_db_dump.tables[:] = randomized.tables
    pquery_db_dump.graphs[:] = randomized.graphs
    pquery_db_dump.queries[:] = randomized.queries
    pquery_db_dump.predictions = randomized.predictions
    project_id, connectors = await provision_fixture_connectors(
        public_api_client, pquery_db_dump, api_prefix='/v1')
    bind_test_client_project(public_api_client, project_id)
    assert len(connectors) == 1
    public_api_client.fixture_duplicate_source_paths = (
        await duplicate_fixture_sources(connectors[0], pquery_db_dump))
    yield


def _update_table_names(pquery_res: PQueryResource, table_name_prefix: str):
    for name in pquery_res.graph.tables.keys():
        pquery_res.query_string = (pquery_res.query_string.replace(
            f'{name}.', f'{table_name_prefix}{name}.'))
    pquery_res.graph.tables = {
        f'{table_name_prefix}{k}': v
        for k, v in pquery_res.graph.tables.items()
    }
    pquery_res.graph.col_groups = [
        ColumnKeyGroup(columns=[
            ColumnKey(f'{table_name_prefix}{col.table_name}', col.col_name)
            for col in col_group.columns
        ]) for col_group in pquery_res.graph.col_groups
    ]


def _graph_id_for_pquery_resource_graph(args: PQueryResource) -> GraphIDv2:
    graph_fpid = GraphDefinition(**dataclasses.asdict(args.graph)).fpid
    return GraphIDv2.public(graph_fpid)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_connector")
async def test_e2e_flow_decoupled(
    client: TestClient,
    public_api_client: TestClient,
    temporal_client: TemporalClient,
    ml_backend: MLJobBackend,
    pquery_db_dump: FixtureDefinition,
):
    for args in await fixture_to_pquery_resources_v2.coroutine(pquery_db_dump):
        _update_table_names(args, 'test_')
        for col_key in args.graph.col_groups[0].columns:
            table_name = col_key.table_name
            table_def = args.graph.tables[col_key.table_name]
            if table_def.pkey and table_def.pkey != col_key.col_name:
                single_col_key = ColumnKey(table_name, table_def.pkey)
                args.graph.col_groups.append(
                    ColumnKeyGroup(columns=(single_col_key, )))
                break
        post_and_expect_success(public_api_client,
                                '/v1/predictive_queries/?dialect=PQLANG_V2',
                                _public_pquery_from_internal(args),
                                EmptyResponse)
        assert (pquery_id := args.name) is not None
        graph_id = post_and_expect_success(
            public_api_client, f'/v1/graphs?name_alias={pquery_id}_graph',
            args.graph, IDResponse).id
        expected_graph_fpid = GraphDefinition(
            **dataclasses.asdict(args.graph)).fpid
        assert graph_id == f'graph-{expected_graph_fpid}'
        snapshot_id = GraphSnapshotID(
            post_and_expect_success(public_api_client,
                                    f'/v1/graphsnapshots?graph_id={graph_id}',
                                    {}, IDResponse).id)
        await _wait_for_graph_snapshot_ingest_done(public_api_client,
                                                   snapshot_id)
        task_type = _get_task_type_v1_compat(
            client,
            pquery_id,
            project_id=DEFAULT_PROJECT_ID,
        )
        training_job_id = await _test_training_flow_decoupled_v1_compat(
            client, public_api_client, temporal_client, args, snapshot_id)
        args.graph.col_groups = args.graph.col_groups[:-1]
        graph_id2 = post_and_expect_success(public_api_client, '/v1/graphs',
                                            args.graph, IDResponse).id
        assert graph_id2 != graph_id
        snapshot_id2 = GraphSnapshotID(
            post_and_expect_success(public_api_client,
                                    f'/v1/graphsnapshots?graph_id={graph_id2}',
                                    {}, IDResponse).id)
        await _wait_for_graph_snapshot_ingest_done(public_api_client,
                                                   snapshot_id2)
        await _test_prediction_flow_decoupled_v1_compat(client,
                                                        public_api_client,
                                                        temporal_client, args,
                                                        task_type, snapshot_id2,
                                                        training_job_id)
        await _test_prediction_over_modified_graph_v1_compat(
            public_api_client, temporal_client, ml_backend, args, task_type,
            training_job_id)
        if task_type not in {
                TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION
        }:
            await _test_prediction_no_outputs_v1_compat(public_api_client,
                                                        temporal_client, args,
                                                        task_type, snapshot_id,
                                                        training_job_id,
                                                        ml_backend)


async def _test_training_flow_decoupled_v1_compat(
        client: TestClient, public_api_client: TestClient,
        temporal_client: TemporalClient, args: PQueryResource,
        snapshot_id: GraphSnapshotID) -> TrainingJobID:
    assert (pquery_id := args.name) is not None
    graph = get_and_expect_success(public_api_client,
                                   f'/v1/graphs/{pquery_id}_graph', {},
                                   GraphResource)
    payload = plan_api.SuggestModelPlanRequest(
        query_string=args.query_string,
        graph_id=graph.id,
        run_mode=RunMode.FAST,
    )
    train_table_plan = post_and_expect_success(
        public_api_client,
        ('/v1/predictive_queries/train_table_generation_plan?'
         'dialect=PQLANG_V2'), payload, api.TrainingTableGenerationPlan)
    train_table_job_config = api.GenerateTrainTableRequest({}, pquery_id,
                                                           train_table_plan,
                                                           snapshot_id)
    train_table_job_id = post_and_expect_success(public_api_client,
                                                 '/v1/gentraintable_jobs',
                                                 train_table_job_config,
                                                 IDResponse).id
    train_table_job = get_and_expect_success(
        public_api_client, f'/v1/gentraintable_jobs/{train_table_job_id}', {},
        api.GenerateTrainTableJobResource)
    assert train_table_job.job_id == train_table_job_id
    assert train_table_job.config.plan == train_table_job_config.plan
    assert train_table_job.config.graph_snapshot_id == snapshot_id
    train_table_job_internal = get_and_expect_success(
        client, f'/v0/gentraintable_jobs/{train_table_job_id}', {
            'project_id': DEFAULT_PROJECT_ID
        }, GenerateTrainTableResponse).resource
    assert train_table_job_internal.job_id == train_table_job_id
    assert train_table_job_internal.config.plan == train_table_job_config.plan
    assert train_table_job_internal.config.graph_snapshot_id == snapshot_id
    _get_task_type_v1_compat(
        client,
        pquery_id,
        project_id=DEFAULT_PROJECT_ID,
    )
    frozen_pq_res = get_and_expect_success(
        public_api_client,
        f'/v1/predictive_queries/{train_table_job.config.pquery_id}', {},
        pquery_api.PQueryResource)
    pq_res = get_and_expect_success(public_api_client,
                                    f'/v1/predictive_queries/{pquery_id}', {},
                                    pquery_api.PQueryResource)
    assert frozen_pq_res == pq_res
    assert train_table_job.job_status_report.status in [
        JobStatus.QUEUED, JobStatus.RUNNING
    ]
    # The small financial-dataset train-table job can materialize in
    # milliseconds, so it may race to DONE between job creation and these
    # checks. Re-fetch the status and only assert the "not ready" 422
    # behavior while the job is still in progress to avoid flakiness; once
    # the job is DONE, ``table_data``/``split_masks`` legitimately return 200
    # and are covered by the assertions later in this flow.
    train_table_job = get_and_expect_success(
        public_api_client, f'/v1/gentraintable_jobs/{train_table_job_id}', {},
        api.GenerateTrainTableJobResource)
    if train_table_job.job_status_report.status in [
            JobStatus.QUEUED, JobStatus.RUNNING
    ]:
        # The result may become readable after the status fetch above but
        # before either request below. Accept that successful race; otherwise
        # retain the exact not-ready status and actionable-error assertions.
        for suffix, error_fragment in [
            ('table_data', 'Unable to get training table data URL for job '),
            ('split_masks', 'Unable to get training table split masks for job'),
        ]:
            response = public_api_client.get(
                f'/v1/gentraintable_jobs/{train_table_job_id}/{suffix}',
                params={'skip_cache': 'true'},
                follow_redirects=True,
            )
            assert response.status_code in [
                HTTPStatus.OK, HTTPStatus.UNPROCESSABLE_ENTITY
            ]
            if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                assert error_fragment in response.text
    model_plan = post_and_expect_success(
        public_api_client,
        '/v1/predictive_queries/model_plan?dialect=PQLANG_V2', payload,
        plan_api.ModelPlanInfo).model_plan
    assert not isinstance(model_plan.neighbor_sampling.num_neighbors,
                          plan_api.MissingType)
    num_neighbors_config = model_plan.neighbor_sampling.num_neighbors[0]
    for i in range(1, num_neighbors_config.num_hops() + 1):
        setattr(num_neighbors_config, f'hop{i}', plan_api.HopConfig(-1))
    train_table_job_id2 = train_table_job_id
    training_job_config = api.TrainingJobRequest({}, pquery_id, model_plan,
                                                 None, train_table_job_id2)
    training_job_id = post_and_expect_success(public_api_client,
                                              '/v1/training_jobs',
                                              training_job_config,
                                              IDResponse).id
    training_job = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{training_job_id}', {},
        api.TrainingJobResource)
    assert training_job.job_id == training_job_id
    assert training_job.job_status_report.status in [
        JobStatus.QUEUED, JobStatus.RUNNING
    ]
    await wait_for_training_job_to_finish(TrainingJobID(training_job_id),
                                          temporal_client)
    training_job = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{training_job_id}', {},
        api.TrainingJobResource)
    train_table_job2 = get_and_expect_success(
        public_api_client, f'/v1/gentraintable_jobs/{train_table_job_id2}', {},
        api.GenerateTrainTableJobResource)
    assert training_job.job_status_report.status == JobStatus.DONE
    assert train_table_job2.job_status_report.status == JobStatus.DONE
    train_table_wf_id = GenerateTrainTableJobID(train_table_job_id2).wf_id
    train_table_wf_result: GenerateTrainTableResult = (
        await temporal_client.result(train_table_wf_id))
    # TT-NG returns the selected physical parquet inside its immutable bundle.
    # It no longer re-materializes the split into ``attributes.json`` /
    # ``{stage}_mask.pt``; the per-stage seed parts are their TT-NG analog.
    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
    )
    staging_dir = ttng_bundle_dir_for_train_table(
        train_table_wf_result.train_table_path)
    assert fsutil.exists(os.path.join(staging_dir, 'manifest.json'))
    assert fsutil.exists(os.path.join(staging_dir, 'tt.parquet'))
    assert fsutil.exists(os.path.join(staging_dir, 'tt_ingest_spec.json'))
    for stage in ['train', 'val', 'test']:
        assert fsutil.exists(os.path.join(staging_dir, f'{stage}_seeds')), (
            f"missing {stage}_seeds in TT-NG bundle: {staging_dir}")
    train_table_job_internal = get_and_expect_success(
        client, f'/v0/gentraintable_jobs/{train_table_job_id}', {
            'project_id': DEFAULT_PROJECT_ID
        }, GenerateTrainTableResponse).resource
    assert train_table_job_internal.job_status_report.status == JobStatus.DONE
    train_table_stats_internal = get_and_expect_success(
        client, f'/v0/gentraintable_jobs/{train_table_job_id}/split_stats',
        {'project_id': DEFAULT_PROJECT_ID}, SplitTrainTableStats)
    assert train_table_stats_internal.train_table_stats.num_train_rows > 0
    table_data_refs = get_and_expect_success(
        public_api_client,
        f'/v1/gentraintable_jobs/{train_table_job_id2}/table_data', {},
        List[str])
    public_parts = flatten_public_parquet_refs(table_data_refs)
    internal_parts = flatten_public_parquet_refs(
        [os.path.join(staging_dir, 'tt.parquet')])
    assert public_parts and internal_parts

    def _content_hashes(paths: List[str]) -> List[str]:
        return sorted(
            hashlib.sha256(read_public_response_bytes(path)).hexdigest()
            for path in paths)

    # Compare every public partition with every durable internal partition.
    # Sorting makes the assertion independent of object-store listing order.
    assert _content_hashes(public_parts) == _content_hashes(internal_parts)
    return TrainingJobID(training_job_id)


def _get_task_type_v1_compat(
    client: TestClient,
    pquery_id: str,
    *,
    project_id: str,
) -> TaskType:
    return get_and_expect_success(
        client, f'/v1/predictive_queries/{pquery_id}/metadata', {
            'project_id': project_id
        }, PQueryMetadataResponse).task_type


def _get_binary_threshold_v1_compat(task_type: TaskType) -> float | None:
    return 0.5 if task_type == TaskType.BINARY_CLASSIFICATION else None


def _get_num_classes_v1_compat(task_type: TaskType) -> int | None:
    return 5 if task_type.is_link_pred else None


async def _test_prediction_no_outputs_v1_compat(
    public_api_client: TestClient,
    temporal_client: TemporalClient,
    args: PQueryResource,
    task_type: TaskType,
    snapshot_id: GraphSnapshotID,
    training_job_id: TrainingJobID,
    ml_backend: MLJobBackend,
) -> None:
    assert (pquery_id := args.name) is not None
    pred_table_job_id = post_and_expect_success(
        public_api_client, '/v1/genpredtable_jobs',
        api.GeneratePredictionTableRequest(
            {}, pquery_id=pquery_id, plan=api.PredictionTableGenerationPlan(),
            graph_snapshot_id=snapshot_id), IDResponse).id
    pred_table_job = get_and_expect_success(
        public_api_client, f'/v1/genpredtable_jobs/{pred_table_job_id}', {},
        api.GeneratePredictionTableJobResource)
    assert pred_table_job.config.graph_snapshot_id == snapshot_id
    assert PredictiveQueryID.is_public(pred_table_job.config.pquery_id)
    assert pred_table_job.job_status_report.status.is_submitted
    bp_job_config = api.BatchPredictionRequest(
        {},
        model_training_job_id=training_job_id,
        predict_options=api.BatchPredictionOptions(
            binary_classification_threshold=_get_binary_threshold_v1_compat(
                task_type),
            num_classes_to_return=_get_num_classes_v1_compat(task_type)),
        outputs=[
            api.LocalPredictionOutput(
                artifact_type=api.PredictionArtifactType.PREDICTIONS)
        ],
        graph_snapshot_id=snapshot_id,
        pred_table_job_id=pred_table_job_id,
    )
    bp_job_id = post_and_expect_success(
        public_api_client, '/v1/prediction_jobs', bp_job_config,
        Tuple[Optional[str], ValidationResponse])[0]
    assert bp_job_id
    await wait_for_prediction_job_to_finish(BatchPredictionJobID(bp_job_id),
                                            temporal_client)
    job_home_dir = await ml_backend.get_job_home_dir(
        BatchPredictionJobID(bp_job_id))
    bp_job_input = await load_json_async(
        HomeDirManager(job_home_dir).prediction_job_input_file,
        BatchPredictionJobInput)
    config = bp_job_input.config
    if task_type.is_link_pred:
        assert config.num_classes_to_return is not None
    else:
        assert config.num_classes_to_return is None


async def _test_prediction_flow_decoupled_v1_compat(
    client: TestClient, public_api_client: TestClient,
    temporal_client: TemporalClient, args: PQueryResource, task_type: TaskType,
    snapshot_id: GraphSnapshotID, training_job_id: TrainingJobID
) -> tuple[BatchPredictionJobID, GeneratePredictionTableJobID]:
    assert (pquery_id := args.name) is not None
    pred_table_job_id = post_and_expect_success(
        public_api_client, '/v1/genpredtable_jobs',
        api.GeneratePredictionTableRequest(
            {}, pquery_id=pquery_id, plan=api.PredictionTableGenerationPlan(),
            graph_snapshot_id=snapshot_id), IDResponse).id
    pred_table_job = get_and_expect_success(
        public_api_client, f'/v1/genpredtable_jobs/{pred_table_job_id}', {},
        api.GeneratePredictionTableJobResource)
    assert pred_table_job.config.graph_snapshot_id == snapshot_id
    assert PredictiveQueryID.is_public(pred_table_job.config.pquery_id)
    frozen_pq_res = get_and_expect_success(
        public_api_client,
        f'/v1/predictive_queries/{pred_table_job.config.pquery_id}', {},
        pquery_api.PQueryResource)
    pq_res = get_and_expect_success(public_api_client,
                                    f'/v1/predictive_queries/{pquery_id}', {},
                                    pquery_api.PQueryResource)
    assert frozen_pq_res != pq_res
    assert len(frozen_pq_res.graph.col_groups) < len(pq_res.graph.col_groups)
    assert GraphDefinition(**dataclasses.asdict(frozen_pq_res.graph)
                           ).fpid == DataSnapshot.GRAPH.parse_id(snapshot_id)[0]
    assert pred_table_job.job_status_report.status.is_submitted
    post_and_expect_success(
        public_api_client, f'/v1/genpredtable_jobs/{pred_table_job_id}/cancel',
        {}, EmptyResponse)
    pred_table_job = get_and_expect_success(
        public_api_client, f'/v1/genpredtable_jobs/{pred_table_job_id}', {},
        api.GeneratePredictionTableJobResource)
    assert pred_table_job.job_status_report.status == JobStatus.CANCELLED
    canceled_pred_table_job_id = pred_table_job_id
    pred_table_job_id = post_and_expect_success(
        public_api_client, '/v1/genpredtable_jobs',
        api.GeneratePredictionTableRequest(
            {}, pquery_id=pquery_id, plan=api.PredictionTableGenerationPlan(),
            graph_snapshot_id=snapshot_id), IDResponse).id
    bp_job_config = api.BatchPredictionRequest(
        {},
        model_training_job_id=training_job_id,
        predict_options=api.BatchPredictionOptions(
            binary_classification_threshold=_get_binary_threshold_v1_compat(
                task_type),
            num_classes_to_return=_get_num_classes_v1_compat(task_type)),
        outputs=[
            api.LocalPredictionOutput(
                artifact_type=api.PredictionArtifactType.PREDICTIONS)
        ],
        graph_snapshot_id=snapshot_id,
        pred_table_job_id=pred_table_job_id,
    )
    bp_job_id = post_and_expect_success(
        public_api_client, '/v1/prediction_jobs', bp_job_config,
        Tuple[Optional[str], ValidationResponse])[0]
    assert bp_job_id
    await wait_for_prediction_job_to_finish(BatchPredictionJobID(bp_job_id),
                                            temporal_client)
    bp_job = get_and_expect_success(public_api_client,
                                    f'/v1/prediction_jobs/{bp_job_id}', {},
                                    api.BatchPredictionJobResource)
    assert bp_job.job_status_report.end_time
    assert bp_job.job_status_report.status == JobStatus.DONE
    assert bp_job.config == bp_job_config
    assert bp_job.result and bp_job.result.num_entities_predicted > 0
    pred_table_job = get_and_expect_success(
        public_api_client, f'/v1/genpredtable_jobs/{pred_table_job_id}', {},
        api.GeneratePredictionTableJobResource)
    assert pred_table_job.job_status_report.status == JobStatus.DONE
    pred_table_path = get_and_expect_success(
        public_api_client,
        f'/v1/genpredtable_jobs/{pred_table_job_id}/table_data', {},
        List[str])[0]
    pred_df = pd.read_parquet(pred_table_path)
    # TT-NG serving pred-table schema (produced from the gen bundle by
    # _postprocess_bp_batch in diskgraph_batch_pred.py): slot-0 ENTITY1->ENTITY,
    # arange row-index appended, TARGET dropped (targets are what LP predicts;
    # the RHS candidate pool is seeded at ingest, never a pred-table column).
    # Static array-target LP additionally drops TIMESTAMP (GE final_cols =
    # {ENTITY, TARGET}); all other tasks keep TIMESTAMP.
    expected_columns = {"ENTITY", "arange"}
    if task_type != TaskType.STATIC_LINK_PREDICTION:
        expected_columns.add("TIMESTAMP")
    assert set(pred_df.columns.to_list()) == expected_columns
    assert pred_df.size > 0
    get_and_expect_error(
        public_api_client, '/v1/genpredtable_jobs/bad-id/table_data', {},
        HTTPStatus.NOT_FOUND,
        r'Generate prediction table job bad-id is not found\.')
    get_and_expect_error(
        public_api_client, '/v1/genpredtable_jobs/bad_id/table_data', {},
        HTTPStatus.UNPROCESSABLE_ENTITY,
        'identifier may only contain alphanumeric characters and dashes')
    get_and_expect_error(
        public_api_client,
        f'/v1/genpredtable_jobs/{canceled_pred_table_job_id}/table_data', {},
        HTTPStatus.UNPROCESSABLE_ENTITY,
        f'job {canceled_pred_table_job_id} in status: '
        f'{JobStatus.CANCELLED.value}')
    pred_table_job_internal = get_and_expect_success(
        client, f'/v0/genpredtable_jobs/{pred_table_job_id}', {
            'project_id': DEFAULT_PROJECT_ID
        }, GeneratePredictionTableResponse).resource
    assert pred_table_job_internal.job_status_report.status == JobStatus.DONE
    return (BatchPredictionJobID(bp_job_id),
            GeneratePredictionTableJobID(pred_table_job_id))


async def _make_modified_graph_snapshot_v1_compat(
        original_graph: GraphDefinition,
        public_api_client: TestClient) -> Tuple[GraphResource, GraphSnapshotID]:
    modified_graph = deepcopy(original_graph)
    for table_definition in modified_graph.tables.values():
        assert isinstance(table_definition.source_table, S3SourceTable)
        source = table_definition.source_table.source
        assert isinstance(source, S3RelativeSource)
        duplicate_paths = public_api_client.fixture_duplicate_source_paths
        table_definition.source_table = dataclasses.replace(
            table_definition.source_table,
            source=dataclasses.replace(
                source,
                path=duplicate_paths[source.path],
            ),
        )
    # The duplicate upload paths deliberately have no cached exact stats yet.
    # Their first graph snapshot below performs the ingest; graph creation
    # therefore must not require the pre-ingest time-column stats.
    modified_graph_id_str = post_and_expect_success(
        public_api_client,
        '/v1/graphs?skip_validation=true',
        modified_graph,
        IDResponse,
    ).id
    modified_graph_id = to_graph_id(modified_graph_id_str)
    assert isinstance(modified_graph_id, GraphIDv2)
    modified_graph_res = get_and_expect_success(
        public_api_client, f'/v1/graphs/{modified_graph_id}', {}, GraphResource)
    snapshot_id = post_and_expect_success(
        public_api_client, f'/v1/graphsnapshots?graph_id={modified_graph_id}',
        {}, IDResponse).id
    await _wait_for_graph_snapshot_ingest_done(public_api_client,
                                               GraphSnapshotID(snapshot_id))
    graph_fpid, _ = DataSnapshot.GRAPH.parse_id(snapshot_id)
    assert graph_fpid == modified_graph_id.fpid
    return modified_graph_res, GraphSnapshotID(snapshot_id)


async def _test_prediction_over_modified_graph_v1_compat(
    public_api_client: TestClient,
    temporal_client: TemporalClient,
    ml_backend: MLJobBackend,
    args: PQueryResource,
    task_type: TaskType,
    training_job_id: TrainingJobID,
):
    _, modified_graph_snapshot_id = (await
                                     _make_modified_graph_snapshot_v1_compat(
                                         args.graph, public_api_client))
    bp_job_config = api.BatchPredictionRequest(
        {},
        model_training_job_id=training_job_id,
        predict_options=api.BatchPredictionOptions(
            binary_classification_threshold=_get_binary_threshold_v1_compat(
                task_type),
            num_classes_to_return=_get_num_classes_v1_compat(task_type)),
        outputs=[
            api.LocalPredictionOutput(
                artifact_type=api.PredictionArtifactType.PREDICTIONS)
        ],
        graph_snapshot_id=modified_graph_snapshot_id,
    )
    bp_job_id = post_and_expect_success(
        public_api_client, '/v1/prediction_jobs', bp_job_config,
        Tuple[Optional[str], ValidationResponse])[0]
    assert bp_job_id
    bp_job = get_and_expect_success(public_api_client,
                                    f'/v1/prediction_jobs/{bp_job_id}', {},
                                    api.BatchPredictionJobResource)
    assert bp_job.config.graph_snapshot_id == modified_graph_snapshot_id
    # Wait for the job to finish before reading its persisted inputs: the
    # prediction_job_input.json file is written by the workflow's async
    # persist_job_inputs activity, so reading it immediately after POST races
    # that write (mirrors the wait-then-read pattern used elsewhere).
    await wait_for_prediction_job_to_finish(BatchPredictionJobID(bp_job_id),
                                            temporal_client)
    job_home_dir = await ml_backend.get_job_home_dir(
        BatchPredictionJobID(bp_job_id))
    bp_job_input = await load_json_async(
        HomeDirManager(job_home_dir).prediction_job_input_file,
        BatchPredictionJobInput)
    assert isinstance(bp_job_input.graph_id, GraphIDv2)
    graph_fpid, graph_data_ver = DataSnapshot.GRAPH.parse_id(
        modified_graph_snapshot_id)
    assert GraphIDv2.parse(bp_job_input.graph_id)[1] == graph_fpid
    assert (bp_job_input.graph_version_info
            and bp_job_input.graph_version_info.graph_data_version
            == graph_data_ver)
    bp_job = get_and_expect_success(public_api_client,
                                    f'/v1/prediction_jobs/{bp_job_id}', {},
                                    api.BatchPredictionJobResource)
    assert bp_job.job_status_report.end_time
    assert bp_job.job_status_report.status == JobStatus.DONE
    assert bp_job.result and bp_job.result.num_entities_predicted > 0
