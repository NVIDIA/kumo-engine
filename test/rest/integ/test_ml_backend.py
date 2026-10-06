# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import os
import pathlib
from queue import Queue
from threading import Event, Thread
from typing import List
from uuid import uuid4

import pytest
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumo_core.utils.time import now_utc
from kumoapi.data_source import DEFAULT_PROJECT_ID
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType
from temporalio import workflow
from temporalio.client import Client

from io_utils import fsutil
from io_utils.json import load_json_async, save_json
from kumo.config.model_plan_yaml import model_plan_from_yaml
from kumo.data import get_persist_dir
from kumo.testing.data import load_financial_csv
from kumo.validation.validation import ValidationResponse
from rest.backends.ml_backend.job_queue import (
    JobConcurrencyLimitError,
    StateDbJobQueue,
)
from rest.backends.ml_backend.ml_job_backend import (
    MLJobBackend,
    _training_workflow_id,
)
from rest.backends.ml_backend.settings import (
    MLBackendSettings,
    MLJobQueueSettings,
)
from rest.backends.ml_backend.workflows.utils import dequeue_job, enqueue_job
from rest.config.connector import FileConnectorResourceConfig
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import TrainingSpec
from rest.constants import ML_WORKFLOW_TASK_QUEUE
from rest.deps.authentication import RequestContext
from rest.testing.integration import run_server_until_stop
from rest.utils.pquery import make_predictive_config_with_graph
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from statedb.config.common import (
    ConnectorID,
    GenerateTrainTableJobID,
    GraphIDv1,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TableIDv1,
    TableName,
    TrainingJobID,
    to_graph_id,
)
from statedb.db import StateServices, open_state_db_from_deploy_config
from statedb.db.auth import AuthPrincipal
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from temporalio_commons import TemporalClient


async def _file_graph_runtime(
    graph_config,
    state_services: StateServices | None = None,
    principal: AuthPrincipal | None = None,
) -> tuple[object, GraphDataComputeInput]:
    """Build the ID-keyed runtime graph and its connector references."""
    from kumo.graph import Graph
    from rest.driver.graph_logic import _add_tables_to_graph_v2
    from rest.utils.graph import export_kumo_graph_to_fixture

    if state_services is None:
        project_id = ProjectID(DEFAULT_PROJECT_ID)
    else:
        _, project = await state_services.groups.ensure_default_resources()
        project_id = project.id
    connector_configs = {}
    connector_specs = {}
    if state_services is not None and len(graph_config.data_source_dict) != 1:
        raise ValueError(
            'StateDB-backed local graph fixtures require one project upload '
            'source; use supported external connectors for multiple sources')
    for name, connector in graph_config.data_source_dict.items():
        root_dir = connector.root_dir
        if state_services is None:
            connector_id = ConnectorID.generate()
        else:
            upload_root = os.path.join(
                deploy_env().data_plane_storage_root,
                'uploaded-datasets',
                str(project_id),
            )
            await fsutil.copy.coroutine(
                root_dir,
                upload_root,
                overwrite=True,
                raise_on_exists=False,
            )
            connector_id = (
                await
                state_services.connectors.register_project_upload_connector(
                    project_id, upload_root))
            root_dir = upload_root
        connector_configs[name] = (
            connector_id,
            FileConnectorResourceConfig(name=name, root_dir=root_dir),
        )
        if state_services is not None:
            connector_specs[connector_id] = await (
                state_services.connectors.execution_ref_for_project(
                    principal,
                    project_id,
                    connector_id,
                    CONNECTOR_GRAPH_ACTIONS,
                ))
        else:
            connector_specs[connector_id] = ConnectorExecutionRef(
                connector_id=connector_id,
                project_id=project_id,
            )
    graph_definition = export_kumo_graph_to_fixture(graph_config,
                                                    connector_configs)
    runtime_graph = Graph(graph_config.name)
    for connector_id, reference in connector_specs.items():
        persisted_config = connector_configs[next(
            name for name, value in connector_configs.items()
            if value[0] == connector_id)][1]
        runtime_graph.connect(
            FileConnectorResourceConfig(
                name=str(reference.connector_id),
                root_dir=persisted_config.root_dir,
            ).as_connector())
    _add_tables_to_graph_v2(
        runtime_graph,
        graph_definition.tables,
        graph_definition.col_groups,
        ignore_errors=True,
    )
    tables = {}
    aliases = {}
    for table_name, table in graph_definition.tables.items():
        table_id = TableIDv1.compose(table.get_connector_id(), table_name)
        tables[table_id] = table
        aliases[table_id] = TableName(table_name)
    graph_input = GraphDataComputeInput(
        graph_display_name=graph_config.name,
        graph_key=graph_config.name,
        tables=tables,
        base_tables={},
        connectors=connector_specs,
        graph_definition=graph_definition,
        table_aliases=aliases,
    )
    return runtime_graph, graph_input


@pytest.mark.asyncio
async def test_serialize_advanced_options(tmp_path):
    """Test that advanced_optns get serialized correctly in TrainingJobInput"""
    graph = load_financial_csv()
    await graph.infer_stypes.coroutine()
    await graph.infer_links.coroutine()
    runtime_graph, graph_input = await _file_graph_runtime(graph)
    pqid = PredictiveQueryID("p")
    advanced_options = model_plan_from_yaml("")
    pquery_config = await make_predictive_config_with_graph.coroutine(
        name=pqid,
        query_yaml="""
PREDICT TRANS.K_SYMBOL = 'DUCHOD'
FOR EACH TRANS.TRANS_ID
""",
        train_options=advanced_options.training_table_generation,
        train_options_provided=False,
        graph=runtime_graph,
    )

    spec = TrainingSpec(
        predictive_query=pquery_config,
        advanced_options=advanced_options,
        predictive_query_id=PredictiveQueryID(pqid),
        graph_id=GraphIDv1("g"),
        split=pquery_config.split,
        stop_after_materialization=True,
    )
    assert isinstance(spec.advanced_options, ModelPlan)
    training_job_input = TrainingJobInput(
        id=TrainingJobID('id'),
        generate_train_table_job_id=GenerateTrainTableJobID('child-job-id'),
        graph_data_compute_input=graph_input,
        spec=spec,
        start_time=now_utc(),
    )
    assert isinstance(training_job_input.spec.advanced_options, ModelPlan)
    await save_json.coroutine(training_job_input,
                              os.path.join(tmp_path, 'job_input.json'))
    training_job_input = await load_json_async(
        os.path.join(tmp_path, 'job_input.json'), TrainingJobInput)
    assert isinstance(training_job_input.spec.advanced_options, ModelPlan)


@pytest.mark.asyncio
async def test_serialize_model_plan(tmp_path):
    """Test that model plans get serialized correctly in TrainingJobInput"""
    graph = load_financial_csv()
    await graph.infer_stypes.coroutine()
    await graph.infer_links.coroutine()
    runtime_graph, graph_input = await _file_graph_runtime(graph)
    pqid = PredictiveQueryID("p")
    model_planner_yaml = ("""training_table_generation:
  split: RandomSplit([0.8, 0.1, 0.1])
  train_start_offset: 360
  train_end_offset: 0
  timeframe_step: 90
  entity_candidate:
  entity_candidate_aggregation:
  task_path_override:
  train_table_path_override:
training_job:
  num_experiments: 2
  metrics:
  - auroc
  - auprc
  tune_metric: auroc
  refit_trainval: false
  refit_full: false
  disable_explain: false
  manual_seed:
column_processing:
  encoder_overrides:
neighbor_sampling:
  max_target_neighbors_per_entity:
  - inferred
  num_neighbors:
  - [12, 12]
optimization:
  max_epochs: 8
  min_steps_per_epoch: 30
  max_steps_per_epoch: 1000
  max_val_steps: 1000
  max_test_steps: 2000
  loss:
  - binary_cross_entropy
  base_lr:
  - 0.0001
  - 0.0005
  - 0.001
  - 0.005
  - 0.01
  - 0.05
  weight_decay:
  - 0.0
  - 5.0e-08
  - 5.0e-07
  - 5.0e-06
  batch_size:
  - 512
  - 1024
  early_stopping:
  - min_delta: 0.0
    patience: 3
  lr_scheduler:
  - name: cosine_with_warmup_restarts
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: constant_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: linear_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: cosine_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  majority_sampling_ratio:
  -
  weight_mode:
  - sample
model_architecture:
  channels:
  - 64
  - 128
  - 256
  num_pre_message_passing_layers:
  - 0
  - 1
  - 2
  num_post_message_passing_layers:
  - 1
  - 2
  aggregation:
  - [sum, mean, max]
  - [sum, mean, min, max, std]
  activation:
  - relu
  - leaky_relu
  - elu
  - gelu
  normalization:
  - layer_norm
  use_seq_id:
  - false
""")
    advanced_options = model_plan_from_yaml(model_planner_yaml)
    pquery_config = await make_predictive_config_with_graph.coroutine(
        name=pqid,
        query_yaml="""
PREDICT TRANS.K_SYMBOL = 'DUCHOD'
FOR EACH TRANS.TRANS_ID
""",
        train_options=advanced_options.training_table_generation,
        train_options_provided=bool(model_planner_yaml),
        graph=runtime_graph,
    )

    spec = TrainingSpec(
        predictive_query=pquery_config,
        advanced_options=advanced_options,
        predictive_query_id=PredictiveQueryID(pqid),
        graph_id=GraphIDv1("g"),
        stop_after_materialization=True,
        split=pquery_config.split,
    )
    assert isinstance(spec.advanced_options, ModelPlan)
    training_job_input = TrainingJobInput(
        id=TrainingJobID('id'),
        generate_train_table_job_id=GenerateTrainTableJobID('gen-train-table'),
        graph_data_compute_input=graph_input,
        spec=spec,
        start_time=now_utc(),
    )
    assert isinstance(training_job_input.spec.advanced_options, ModelPlan)
    await save_json.coroutine(training_job_input,
                              os.path.join(tmp_path, 'job_input.json'))
    training_job_input = await load_json_async(
        os.path.join(tmp_path, 'job_input.json'), TrainingJobInput)
    assert isinstance(training_job_input.spec.advanced_options, ModelPlan)


# DiskGraph materialization routes through DiskgraphIngestWorkflow, which uses
# the pool controller/worker for snapshot placement. Pin this test to the local
# pool backend so it does not wait on an unserved pool AutoRPC queue in CI.
@pytest.mark.usefixtures("_pool_subprocesses")
@pytest.mark.parametrize("storage_backend", ["local"], indirect=True)
@pytest.mark.asyncio
async def test_make_materialization_info(ml_backend: MLJobBackend,
                                         temporal_client: TemporalClient,
                                         state_services: StateServices,
                                         ctx: RequestContext):
    """Test that the materialization info is available after materialization.

    TODO(zack): This test was added in order to test the K8s materialization
    path, because this path is currently not covered by test_e2e. However, this
    adds 3m of test latency. Once test_e2e is updated to test the K8s path, we
    should merge this test with test_e2e in order to eliminate the latency.
    """
    # Clear the data persist dir to force re ingestion of all tables.
    # Use fsutil.rm (recursive, scheme-aware) since data_persist_dir may be
    # an s3:// URI in CI; stdlib shutil.rmtree is local-only.
    fsutil.rm(get_persist_dir(), ignore_errors=True)

    graph = load_financial_csv()
    await graph.infer_stypes.coroutine()
    await graph.infer_links.coroutine()
    runtime_graph, graph_data_compute_input = await _file_graph_runtime(
        graph, state_services, ctx.user_authorization)
    pqid = PredictiveQueryID("p")
    pquery_config = await make_predictive_config_with_graph.coroutine(
        name=pqid,
        query_yaml="""
PREDICT TRANS.K_SYMBOL = 'DUCHOD'
FOR EACH TRANS.TRANS_ID
""",
        train_options=model_plan_from_yaml("").training_table_generation,
        train_options_provided=False,
        graph=runtime_graph,
    )

    advanced_options = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.DEBUG,
    )

    project_id = next(iter(
        graph_data_compute_input.connectors.values())).project_id

    spec = TrainingSpec(
        predictive_query=pquery_config,
        advanced_options=advanced_options,
        predictive_query_id=PredictiveQueryID(pqid),
        graph_id=GraphIDv1("g"),
        stop_after_materialization=True,
        split=pquery_config.split,
    )
    training_job_id = await ml_backend.start_training(
        pquery_def_fpid=None, graph_data_compute_input=graph_data_compute_input,
        training_spec=spec, project_id=project_id)
    await temporal_client.result(_training_workflow_id(training_job_id),
                                 disable_backoff=True)
    info = await ml_backend.get_materialization_info(training_job_id)

    assert info is not None
    assert info.start_time
    assert info.end_time > info.start_time
    assert info.id == training_job_id
    assert info.graph_id
    assert info.predictive_query_id
    assert info.code_version
    assert info.temporal_workflow_id
    assert len(info.table_stats) == len(runtime_graph)

    stat_map = {stat.table_id: stat for stat in info.table_stats}
    for table in runtime_graph:
        assert table.name in stat_map
        table_stat = stat_map[table.name]
        assert table_stat.code_version
        assert table_stat.connector_id
        assert table_stat.data_version
        # Ingestion should begin after the training job is launched
        assert table_stat.ingest_start_time > info.start_time
        assert table_stat.ingest_end_time > table_stat.ingest_start_time
        assert table_stat.num_rows
        if table.pkey is not None:
            assert table_stat.pkey == table.pkey.name
        assert table_stat.size_bytes_source
        assert table_stat.source_table_name
        assert table_stat.temporal_workflow_id
        if table.time_col is not None:
            time_col = table_stat.time_col
            assert time_col is not None
            assert time_col.name == table.time_col.name
            assert time_col.min
            assert time_col.max > time_col.min
        assert len(table_stat.columns) == len(table)
        col_map = {col.name: col for col in table_stat.columns}
        for col in table:
            assert col.name in col_map
            col_stat = col_map[col.name]
            assert col_stat.dtype
            assert col_stat.stype
            if table.name == "TRANS" and col.name == "ACCOUNT":
                assert col_stat.num_empty == 50


@workflow.defn(name='TrainingWorkflow')
class MockTrainingWorkflow:
    def __init__(self):
        self._finish = False

    @workflow.signal
    async def finish_wf(self):
        self._finish = True

    @workflow.run
    async def run(self, job_id: TrainingJobID) -> ValidationResponse:
        await enqueue_job(job_id)
        try:
            await workflow.wait_condition(lambda: self._finish)
            return ValidationResponse()
        finally:
            # Workflow-owned finalization, matching production: dequeue_job ->
            # finalize_job runs on THIS worker (its own StateDB) and, because
            # this mock never wrote a terminal status, applies the safety-net
            # terminalization (status -> FAILED, frozen) and releases the queue
            # slot so the next queued job is admitted.  We deliberately do NOT
            # write DONE via save_training_job_output here: that is an AutoRPC
            # call to the MLJobOutputStore service, which in this multi-worker
            # integ setup can land on a different worker (different StateDB)
            # and fail to find the job.  The test only asserts terminal-ness
            # and FIFO admission, which the safety-net path satisfies.
            await dequeue_job(job_id)


async def _mock_launch_jobs(client: Client, state_services: StateServices,
                            num_jobs: int = 1) -> List[TrainingJobID]:
    job_ids = []
    for _ in range(num_jobs):
        job_id, _ = await state_services.ml_jobs.create_training_job(
            pquery_id=PredictiveQueryID('dummy_pquery'), pquery_def_fpid=None,
            graph_id=to_graph_id('dummy_graph'), now=now_utc(),
            advanced_options=ModelPlan(),
            project_id=ProjectID(DEFAULT_PROJECT_ID))
        await client.start_workflow(
            MockTrainingWorkflow.run,
            args=[job_id],
            id=_training_workflow_id(job_id),
            task_queue=ML_WORKFLOW_TASK_QUEUE,
        )
        job_ids.append(job_id)
    return job_ids


@pytest.mark.asyncio
async def test_ml_backend_with_job_queue(tmp_path: pathlib.Path,
                                         ctx: RequestContext,
                                         monkeypatch: pytest.MonkeyPatch):
    deploy_env_config = dataclasses.replace(
        DeployEnvConfig.for_test(),
        # Use a unique instance_id so this test's ML backend worker doesn't
        # collide with the default one from test setup.
        instance_id=uuid4().hex,
    )
    state_root = tmp_path.absolute().as_posix()
    state_db = await open_state_db_from_deploy_config(
        deploy_env_config,
        state_root=state_root,
    )
    monkeypatch.setenv('DB_STORAGE_ROOT', state_root)
    state_services = StateServices(state_db)
    temporal_client = TemporalClient(deploy_env_config)
    client = await temporal_client.get()

    job_id1, job_id2 = await _mock_launch_jobs(client, state_services,
                                               num_jobs=2)
    entry2 = await state_services.ml_jobs.get_training_job(job_id2)
    assert entry2 is not None
    entry2.execution.status = JobStatus.RUNNING
    await state_services.ml_jobs.modify_job_if_open_generic(entry2)

    # Override queue settings for the ML job backend.
    monkeypatch.setenv('ML_MAX_CONCURRENT_JOBS_TOTAL', '1')
    monkeypatch.setenv('ML_MAX_GPU_WORKER_SLOTS', '1')
    monkeypatch.setenv('ML_JOB_QUEUE_SIZE', '1')

    start_event = Event()
    stop_event = Event()
    start_exceptions = Queue[BaseException]()
    worker_daemon_thread = Thread(
        target=run_server_until_stop,
        daemon=True,
        args=[
            start_event,
            stop_event,
            deploy_env_config,
            start_exceptions,
            {},
            [],  # activities override
            [MockTrainingWorkflow],  # workflows override
        ])
    print('Starting worker thread')
    worker_daemon_thread.start()
    print('ML Job backend workers have started!')
    start_event.wait()
    if not start_exceptions.empty():
        raise RuntimeError(
            'Failed to start temporal worker') from start_exceptions.get()
    print('Running test against ML Job Backend worker.')
    # In-process MLJobBackend over the same StateDB the e2e workers use (no
    # AutoRPC client). Used for read accessors only; the ml-workflow-worker
    # started by run_server_until_stop owns the lifecycle (reconcile +
    # admission/drain) the queue assertions below depend on.
    ml_backend = MLJobBackend(MLBackendSettings.parse_from_env(),
                              temporal_client, state_db)
    await ml_backend.start(read_only=True)
    print('ML Job backend started.')

    # Verify child jobs are initially QUEUED
    child_jobs = await ml_backend.get_generate_train_table_jobs(
        set(
            GenerateTrainTableJobID.as_child_of(id)
            for id in [job_id1, job_id2]),
        project_ids=None,
    )
    assert len(child_jobs) == 2
    for job in child_jobs:
        assert job.job_status_report.status == JobStatus.QUEUED

    # DB-backed admission queue over the same StateDB the worker uses.
    settings = MLJobQueueSettings(1, 1, True, 1)
    job_queue = StateDbJobQueue(state_db, settings)

    # Status is derived from the DB: job2 RUNNING (1 slot), job1 QUEUED.
    queue_status = await job_queue.report_status()
    assert queue_status.job_slots_occupied == 1
    assert queue_status.gpu_slots_occupied == 1
    assert queue_status.running_jobs == [job_id2]
    assert queue_status.queued_jobs == [job_id1]

    # Unable to enqueue more jobs since ML_JOB_QUEUE_SIZE is 1: a fresh
    # QUEUED job submitted while the queue is already full is rejected.
    # Create only the job *row* (no workflow), so nothing later re-submits it
    # and races the terminal-state assertions below.
    job_id3, job_id3_child = await state_services.ml_jobs.create_training_job(
        pquery_id=PredictiveQueryID('dummy_pquery'), pquery_def_fpid=None,
        graph_id=to_graph_id('dummy_graph'), now=now_utc(),
        advanced_options=ModelPlan(), project_id=ProjectID(DEFAULT_PROJECT_ID))
    with pytest.raises(JobConcurrencyLimitError):
        await job_queue.submit_job(job_id3)

    # Verify job1 should be QUEUED while job2 is RUNNING
    job1 = await ml_backend.get_training_job(ctx, job_id1)
    assert job1 and job1.execution.status == JobStatus.QUEUED
    job2 = await ml_backend.get_training_job(ctx, job_id2)
    assert job2 and job2.execution.status == JobStatus.RUNNING

    # Allow job2 to complete (and will be removed from queue).
    wf_hdl2 = await temporal_client.get_workflow_handle(
        _training_workflow_id(job_id2))
    await wf_hdl2.signal(MockTrainingWorkflow.finish_wf)
    await wf_hdl2.result()

    # After "removing" running job2, job1 should be running
    job1 = await ml_backend.get_training_job(ctx, job_id1)
    assert job1 and job1.execution.status == JobStatus.RUNNING

    # Allow job1 to complete, then there should be no running or queued job.
    wf_hdl1 = await temporal_client.get_workflow_handle(
        _training_workflow_id(job_id1))
    await wf_hdl1.signal(MockTrainingWorkflow.finish_wf)
    await wf_hdl1.result()
    # job3 (and its auto-created child gen-train-table job) were created above
    # but never admitted; delete both so the queue is empty for the terminal
    # assertion (deleting only the parent would orphan the child, which the
    # queue would then count as a top-level QUEUED job).
    await state_services.ml_jobs.delete_job_generic(job_id3, force=True)
    await state_services.ml_jobs.delete_job_generic(job_id3_child, force=True)
    queue_status = await job_queue.report_status()
    assert queue_status.job_slots_occupied == 0
    assert queue_status.gpu_slots_occupied == 0
    assert queue_status.running_jobs == []
    assert queue_status.queued_jobs == []

    # Wait until both job1 and job2 becomes terminal
    while True:
        jobs = await ml_backend.get_training_jobs(ctx, set([job_id1, job_id2]))
        assert len(jobs) == 2
        assert job_id1 in jobs
        assert job_id2 in jobs
        if any(not job.execution.status.is_terminal for job in jobs.values()):
            await asyncio.sleep(1)
        else:
            break  # both jobs are terminal
    # Verify child jobs are terminal as well
    child_jobs = await ml_backend.get_generate_train_table_jobs(
        set(GenerateTrainTableJobID.as_child_of(id) for id in jobs),
        project_ids=None,
    )
    assert len(child_jobs) == 2
    for job in child_jobs:
        assert job.job_status_report.status.is_terminal

    print('Shutting down ML Job backend worker thread')
    stop_event.set()
    print('Stop event sent')
    worker_daemon_thread.join()
