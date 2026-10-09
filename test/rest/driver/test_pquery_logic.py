# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.pquery import QueryType
from kumoapi.typing import Dtype, Stype

from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import SourceTableNotFoundError
from kumo.connector.file_connector import FileConnector
from kumo.graph.column import Column as GraphColumn
from kumo.graph.graph import Graph
from kumo.validation import ValidationException, ValidationResponse
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ValidatedPredictiveQuerySnapshot,
)
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import PQueryDefinition
from rest.config.predictive_query_v2 import PQueryValidationArgs
from rest.config.table_v2 import Column, CustomSQLSourceTable, TableDefinition
from rest.driver import pquery_logic
from rest.driver.graph_logic import canonicalize_graph_custom_sql_tables
from rest.driver.pquery_logic import PQueryLogic
from rest.settings import PredictiveQuerySettings
from statedb.config.common import (
    ConnectorID,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import TAG_KEY_PQUERY_ID, JobTagValue
from statedb.config.timestamp import get_current_timestamps
from statedb.rest import connector_authorization as connector_auth

_PQUERY_RESOURCE_PATH = (Path(__file__).resolve().parents[1] / "test_data" /
                         "pquery_resource.json")

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_TEST_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)


def _validated_snapshot(graph) -> ValidatedPredictiveQuerySnapshot:
    config = object.__new__(PredictiveQueryV2Config)
    for config_field in fields(PredictiveQueryV2Config):
        setattr(config, config_field.name, None)
    config.graph = graph
    snapshot = object.__new__(ValidatedPredictiveQuerySnapshot)
    snapshot.config = config
    snapshot.validation_response = ValidationResponse()
    return snapshot


@pytest.mark.asyncio
async def test_default_model_plan_delegates_all_target_paths_to_core(
        monkeypatch):
    target_paths = [
        [("orders", "user_id", "users")],
        [("orders", "product_id", "products")],
    ]
    pquery = SimpleNamespace(
        graph=SimpleNamespace(config=object),
        entity_table_names=["users", "products"],
        target_to_entity_paths=target_paths,
        get_default_split_async=AsyncMock(return_value=object()),
        is_forecasting=False,
        validated_predictive_query=SimpleNamespace(top_k=None),
    )
    parsed = SimpleNamespace(
        predictive_query=pquery,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
    )
    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=SimpleNamespace(),
        project_id=_TEST_PROJECT_ID,
    )
    logic.parse_predictive_query_v2 = AsyncMock(
        return_value=(parsed, ValidationResponse()))
    captured = {}

    def fake_from_default(cls, **kwargs):
        del cls
        captured.update(kwargs)
        return pquery_logic.ModelPlan()

    monkeypatch.setattr(pquery_logic.ModelPlan, "from_default",
                        classmethod(fake_from_default))
    monkeypatch.setattr(pquery_logic, "construct_object_string_from_config",
                        lambda _split: "RandomSplit([0.8, 0.2])")
    request = pquery_logic.DefaultModelPlanRequest(
        query_string="query",
        graph_id="",
        graph_def=GraphDefinition(tables={}, col_groups=[]),
        run_mode=pquery_logic.RunMode.FAST.value,
    )

    await logic.get_model_plan_info_with_graph_def(
        request,
        PredictiveQuerySettings(),
    )

    assert captured["target_to_entity_paths"] is target_paths


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def test_real_connector_ids_from_graph_definition_returns_project_connectors():
    pquery = json.loads(_PQUERY_RESOURCE_PATH.read_text(encoding="utf-8"))
    for table in pquery["graph"]["tables"].values():
        table["source_table"]["connector_id"] = str(_TEST_CONNECTOR_ID)
    graph_def = GraphDefinition(**pquery["graph"])

    assert PQueryLogic._real_connector_ids_from_graph_definition(graph_def) == {
        _TEST_CONNECTOR_ID
    }


@pytest.mark.asyncio
async def test_predictive_query_args_are_validated_before_reconstruction(
    monkeypatch, ):
    reference = ConnectorExecutionRef(_TEST_CONNECTOR_ID, _TEST_PROJECT_ID)
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=reference), )
    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=SimpleNamespace(connectors=connector_service),
        project_id=_TEST_PROJECT_ID,
    )
    expected = MagicMock()
    build_pquery = AsyncMock(return_value=expected)
    monkeypatch.setattr(
        pquery_logic,
        "_build_request_predictive_query",
        SimpleNamespace(coroutine=build_pquery),
    )
    request_table = SimpleNamespace(
        name="connector_table",
        data_source=str(_TEST_CONNECTOR_ID),
        data_version=None,
    )
    validated_table = SimpleNamespace(
        name="connector_table",
        data_source=str(_TEST_CONNECTOR_ID),
        data_version="data-version",
    )
    execute_activity = AsyncMock(return_value=_validated_snapshot(
        SimpleNamespace(data_source_dict={}, tables=[validated_table])))
    monkeypatch.setattr(
        pquery_logic.DataBackend,
        "_execute_single_activity",
        execute_activity,
    )
    request_graph_config = SimpleNamespace(
        data_source_dict={str(_TEST_CONNECTOR_ID): MagicMock()},
        tables=[request_table],
    )
    monkeypatch.setattr(
        pquery_logic,
        "credential_free_graph_config_for_activity",
        MagicMock(return_value=request_graph_config),
    )
    bind_graph = MagicMock(return_value="workflow-safe-graph")
    monkeypatch.setattr(pquery_logic, "bind_graph_config_to_execution_refs",
                        bind_graph)
    monkeypatch.setattr(
        logic,
        "_real_connector_ids_from_graph_definition",
        MagicMock(return_value={_TEST_CONNECTOR_ID}),
    )
    request_graph = MagicMock()
    args = SimpleNamespace(graph=MagicMock())
    settings = MagicMock()

    snapshot = await logic._validate_predictive_query_args(
        args,
        settings,
        graph=request_graph,
    )

    assert snapshot.config.graph.data_source_dict == (
        request_graph_config.data_source_dict)
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None,
        _TEST_PROJECT_ID,
        _TEST_CONNECTOR_ID,
        connector_auth.CONNECTOR_PQUERY_VALIDATION_ACTIONS,
    )
    bind_graph.assert_called_once_with(
        request_graph_config,
        {_TEST_CONNECTOR_ID: reference},
    )
    execute_activity.assert_awaited_once_with(
        logic._data_backend_ctx,
        pquery_logic.validate_predictive_query_activity,
        [
            args,
            "workflow-safe-graph",
            settings,
            {
                _TEST_CONNECTOR_ID: reference
            },
            False,
        ],
        f"validate-pquery:{_TEST_PROJECT_ID}",
    )

    connector = MagicMock()
    graph = MagicMock(data_source_dict={str(_TEST_CONNECTOR_ID): connector})
    monkeypatch.setattr(pquery_logic.Graph, "from_config",
                        MagicMock(return_value=graph))
    result = await logic._create_pq_object_from_args_and_config(
        snapshot.config,
        settings,
        validated_snapshot=snapshot,
    )

    assert result is expected
    connector.require_explicit_credentials.assert_called_once_with()
    request_config = build_pquery.await_args.args[0]
    assert request_config is not snapshot.config
    assert request_config.graph.tables[0].data_version == "data-version"
    build_pquery.assert_awaited_once_with(
        request_config,
        graph=graph,
        pquery_settings=settings,
        skip_validation=False,
    )


@pytest.mark.parametrize(
    'connector_family',
    ['snowflake', 'bigquery'],
    ids=['warehouse-sf', 'warehouse-bq'],
)
def test_predictive_query_activity_returns_credential_free_snapshot(
        monkeypatch, connector_family):
    reference = ConnectorExecutionRef(_TEST_CONNECTOR_ID, _TEST_PROJECT_ID)
    workflow_graph = SimpleNamespace(data_source_dict={}, tables=[])
    graph = MagicMock()
    graph.tables = [
        SimpleNamespace(name='warehouse.events', data_version='version-1'),
    ]
    canonical_graph = SimpleNamespace(
        data_source_dict={str(_TEST_CONNECTOR_ID): object()},
        tables=[
            SimpleNamespace(
                name='warehouse.events',
                data_source=str(_TEST_CONNECTOR_ID),
                data_version='version-1',
            )
        ],
    )
    canonical_config = object.__new__(PredictiveQueryV2Config)
    for config_field in fields(PredictiveQueryV2Config):
        setattr(canonical_config, config_field.name, None)
    canonical_config.graph = canonical_graph
    graph.config.return_value = canonical_graph
    validation_response = ValidationResponse()
    pquery = SimpleNamespace(
        graph=graph,
        get_validate_response=lambda: validation_response,
    )
    secret = f'{connector_family}-identity-secret'

    @contextmanager
    def authorized_graph(graph_config, references):
        assert graph_config.data_source_dict == {}
        assert references == {_TEST_CONNECTOR_ID: reference}
        # The credential-bearing graph exists only inside this scope. It is
        # deliberately never returned by the activity.
        graph.identity_credential = secret
        yield graph

    monkeypatch.setattr(pquery_logic, 'authorized_graph_for_activity',
                        authorized_graph)
    from_config = MagicMock(return_value=pquery)
    monkeypatch.setattr(pquery_logic.PredictiveQueryV2, 'from_config',
                        from_config)
    make_config = MagicMock(return_value=canonical_config)
    monkeypatch.setattr(pquery_logic, 'make_predictive_config_with_graph',
                        make_config)
    args = SimpleNamespace(
        name='warehouse-query',
        query_string='PREDICT events.label FOR EACH events.entity_id',
        model_plan='',
        train_table_options=None,
        list_distinct_target_is_array=None,
        lp_bp_layout=None,
    )

    result = pquery_logic.validate_predictive_query_activity(
        args,
        workflow_graph,
        PredictiveQuerySettings(),
        {_TEST_CONNECTOR_ID: reference},
    )

    assert isinstance(result, ValidatedPredictiveQuerySnapshot)
    assert result.config is not canonical_config
    assert result.config.graph.data_source_dict == {}
    assert result.config.graph.tables[0].data_version == 'version-1'
    assert result.validation_response is validation_response
    assert secret not in repr(result)
    graph.validate.assert_called_once_with(raise_on_error=True,
                                           ignore_encoders=True)
    make_config.assert_called_once()
    from_config.assert_called_once()


def test_predictive_query_activity_safely_translates_model_plan_validation(
        monkeypatch):
    graph = MagicMock()

    @contextmanager
    def authorized_graph(_graph_config, _references):
        yield graph

    monkeypatch.setattr(pquery_logic, 'authorized_graph_for_activity',
                        authorized_graph)
    secret = 'api_key=do-not-reflect-this-plan-value'
    args = PQueryValidationArgs(
        name='unsafe-model-plan',
        query_string='PREDICT events.label FOR EACH events.entity_id',
        graph=GraphDefinition(tables={}, col_groups=[]),
        model_plan=('training_table_generation:\n'
                    f'  split: InvalidSplit({secret})\n'),
    )

    with pytest.raises(ValidationException) as caught:
        pquery_logic.validate_predictive_query_activity(
            args,
            SimpleNamespace(data_source_dict={}),
            PredictiveQuerySettings(),
            {},
        )

    assert caught.value.http_code == 422
    assert str(caught.value) == (
        'Invalid model plan: Check its field names, values, and expression '
        'syntax.')
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_request_pquery_reconstruction_never_runs_graph_validation(monkeypatch):
    graph = MagicMock()
    graph.validate.side_effect = AssertionError('request-side provider I/O')
    expected = object()
    reconstruct = MagicMock(return_value=expected)
    monkeypatch.setattr(pquery_logic.PredictiveQueryV2,
                        '_from_validated_config', reconstruct)
    config = object()
    settings = PredictiveQuerySettings()

    result = pquery_logic._build_request_predictive_query(
        config,
        graph,
        settings,
        False,
    )

    assert result is expected
    reconstruct.assert_called_once()
    assert reconstruct.call_args.args == (config, )
    assert reconstruct.call_args.kwargs['graph'] is graph
    assert reconstruct.call_args.kwargs['pquery_settings'] is not None
    graph.validate.assert_not_called()


@pytest.mark.asyncio
async def test_require_successful_training_job_resolves_v2_tagged_job():
    """Regression: a v2 pquery is stored only as a ``PQueryDefinition`` and has
    no cached ``current_training_job_id`` (it is synthesized as None); its
    training jobs live in job tags. A DONE tagged job must be returned, not
    rejected with "needs to be trained".
    """
    # A v2 pquery: only a PQueryDefinition (+ alias), no
    # predictive_queries_map entry -- exactly what create_predictive_query_v2
    # produces.
    pq_def = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name='mypquery',
        query_str='PREDICT COUNT(t.*) FOR EACH t.user_id',
        graph_fpid=FPID('a' * 32),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )

    # A successful training job exists, linked to the pquery via job tags.
    training_job_id = TrainingJobID('trainingjob-done')
    pquery_service = SimpleNamespace(
        get_definition_by_id=AsyncMock(return_value=pq_def), )
    ml_job_service = SimpleNamespace(
        list_job_tags=AsyncMock(return_value={training_job_id: object()}),
        list_training_job_rows=AsyncMock(
            return_value=[SimpleNamespace(id=training_job_id)]),
    )
    state_services = SimpleNamespace(
        pqueries=pquery_service,
        ml_jobs=ml_job_service,
    )

    done_job = SimpleNamespace(execution=SimpleNamespace(status=JobStatus.DONE))
    ml_backend = MagicMock()
    ml_backend.get_training_job = AsyncMock(return_value=done_job)
    ml_backend.get_job_home_dir = AsyncMock(return_value='/tmp/jobhome')

    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )
    job, home_mgr = await logic._require_successful_training_job(
        MagicMock(), PredictiveQueryID('mypquery'), ml_backend)

    assert job is done_job
    assert home_mgr.home_dir == '/tmp/jobhome'
    # The job resolved must be the one tagged to the pquery.
    assert ml_backend.get_training_job.await_args.args[1] == training_job_id
    pquery_service.get_definition_by_id.assert_awaited_once_with(
        PredictiveQueryID('mypquery'))
    ml_job_service.list_job_tags.assert_awaited_once_with(
        JobType.TRAINING_JOB,
        {str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(pq_def.fpid))},
    )


@pytest.mark.asyncio
async def test_training_job_ids_for_pquery_uses_ml_job_service_tags():
    fpid = FPID("a" * 32)
    training_job_id = TrainingJobID("trainingjob-done")
    ml_job_service = SimpleNamespace(
        list_job_tags=AsyncMock(return_value={training_job_id: object()}),
        list_training_job_rows=AsyncMock(
            return_value=[SimpleNamespace(id=training_job_id)]),
    )
    state_services = SimpleNamespace(ml_jobs=ml_job_service)
    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    assert await logic._training_job_ids_for_pquery(fpid) == [training_job_id]
    ml_job_service.list_job_tags.assert_awaited_once_with(
        JobType.TRAINING_JOB,
        {str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(fpid))},
    )
    ml_job_service.list_training_job_rows.assert_awaited_once_with(
        job_ids={training_job_id},
        project_ids={_TEST_PROJECT_ID},
        sort_created_at="asc",
    )


@pytest.mark.asyncio
async def test_delete_predictive_query_with_state_services_deletes_definition():
    graph_fpid = FPID("a" * 32)
    pq_def = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="saved_query",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    training_job_id = TrainingJobID("trainingjob-running")
    pquery_service = SimpleNamespace(
        get_definition_by_id=AsyncMock(return_value=pq_def),
        delete_alias=AsyncMock(),
        delete_definition=AsyncMock(return_value=True),
    )
    graph_service = SimpleNamespace(
        decrement_or_delete_graph_definition=AsyncMock(return_value=True), )
    ml_job_service = SimpleNamespace(
        list_job_tags=AsyncMock(return_value={training_job_id: object()}),
        list_training_job_rows=AsyncMock(
            return_value=[SimpleNamespace(id=training_job_id)]),
    )
    state_services = SimpleNamespace(
        pqueries=pquery_service,
        graphs=graph_service,
        ml_jobs=ml_job_service,
    )
    ml_backend = MagicMock()
    ml_backend.get_training_job = AsyncMock(return_value=SimpleNamespace(
        execution=SimpleNamespace(status=JobStatus.RUNNING), ))
    ml_backend.cancel_training_job = AsyncMock(return_value=True)

    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    deleted = await logic.delete_predictive_query(
        MagicMock(),
        PredictiveQueryID("saved_query"),
        ml_backend,
    )

    assert deleted is True
    pquery_service.get_definition_by_id.assert_awaited_once_with(
        PredictiveQueryID("saved_query"))
    ml_job_service.list_job_tags.assert_awaited_once_with(
        JobType.TRAINING_JOB,
        {str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(pq_def.fpid))},
    )
    ml_backend.cancel_training_job.assert_awaited_once_with(training_job_id)
    graph_service.decrement_or_delete_graph_definition.assert_awaited_once_with(
        _TEST_PROJECT_ID, graph_fpid)
    pquery_service.delete_alias.assert_awaited_once_with("saved_query")
    pquery_service.delete_definition.assert_awaited_once_with(pq_def.fpid)


def _stub_create_full_pquery_def_v2_dependencies(
    monkeypatch,
    *,
    config_lp_bp_layout='separate_lhs_rhs',
    pquery_lp_bp_layout='separate_lhs_rhs',
):
    class FakeGraphLogic:
        def __init__(
            self,
            data_backend_ctx,
            *,
            state_services=None,
            project_id=None,
        ):
            del data_backend_ctx, state_services, project_id

        async def create_full_graph_def_v2(self, graph_def):
            return MagicMock(name='graph')

    async def fake_validate_predictive_query_args(self, *args, **kwargs):
        return MagicMock(name='validated-snapshot')

    async def fake_parse_predictive_query_v2(self, *args, **kwargs):
        pquery = SimpleNamespace(
            get_task_type=lambda: None,
            query_type=None,
            lp_bp_layout=pquery_lp_bp_layout,
        )
        parsed = SimpleNamespace(
            config=SimpleNamespace(lp_bp_layout=config_lp_bp_layout),
            predictive_query=pquery,
        )
        return parsed, ValidationResponse()

    monkeypatch.setattr(pquery_logic.graph_logic, 'GraphLogic', FakeGraphLogic)
    monkeypatch.setattr(PQueryLogic, '_validate_predictive_query_args',
                        fake_validate_predictive_query_args)
    monkeypatch.setattr(PQueryLogic, 'parse_predictive_query_v2',
                        fake_parse_predictive_query_v2)


def _pquery_resource():
    return SimpleNamespace(
        graph=GraphDefinition(tables={}, col_groups=[]),
        name='saved_query',
        query_string=(
            'PREDICT responses.is_sensitive FOR EACH responses.response_id'),
        list_distinct_target_is_array=None,
        lp_bp_layout='hstack',
    )


@pytest.mark.asyncio
async def test_create_full_pquery_def_v2_uses_validated_config_lp_bp_layout(
    monkeypatch, ):
    _stub_create_full_pquery_def_v2_dependencies(monkeypatch)

    fpid = FPID('b' * 32)
    pquery_service = SimpleNamespace(
        add_definition=AsyncMock(return_value=fpid),
        set_alias=AsyncMock(),
    )
    state_services = SimpleNamespace(pqueries=pquery_service)
    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    await logic.create_full_pquery_def_v2(
        _pquery_resource(),
        pquery_settings=MagicMock(),
    )

    pquery_def = pquery_service.add_definition.await_args.args[0]
    assert pquery_def.lp_bp_layout == 'separate_lhs_rhs'
    pquery_service.set_alias.assert_awaited_once_with(
        'saved_query',
        fpid,
    )


@pytest.mark.asyncio
async def test_create_full_pquery_def_v2_rejects_lp_bp_layout_divergence(
    monkeypatch, ):
    _stub_create_full_pquery_def_v2_dependencies(
        monkeypatch,
        config_lp_bp_layout='separate_lhs_rhs',
        pquery_lp_bp_layout='hstack',
    )

    pquery_service = SimpleNamespace(
        add_definition=AsyncMock(return_value=FPID('b' * 32)),
        set_alias=AsyncMock(),
    )
    state_services = SimpleNamespace(pqueries=pquery_service)
    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    with pytest.raises(ValueError, match='lp_bp_layout does not match'):
        await logic.create_full_pquery_def_v2(
            _pquery_resource(),
            pquery_settings=MagicMock(),
        )

    pquery_service.add_definition.assert_not_awaited()


def _custom_sql_graph_definition() -> GraphDefinition:
    sql_def = TableDefinition(
        source_table=CustomSQLSourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            connector_type='snowflake',
            sql='SELECT response_id, is_sensitive FROM responses',
            # Provisional (non-canonical) synthetic name: canonicalization
            # recomputes it, which changes the table (and graph) fingerprint.
            table='custom_sql_provisional',
            referenced_tables=[],
        ),
        cols=[
            Column(name='response_id', stype=Stype.ID, dtype=Dtype.string),
            Column(name='is_sensitive', stype=Stype.categorical,
                   dtype=Dtype.string),
        ],
        pkey='response_id',
    )
    return GraphDefinition(tables={'responses': sql_def}, col_groups=[])


@pytest.mark.asyncio
async def test_build_pquery_def_v2_uses_canonical_graph_fpid(monkeypatch):
    """A pquery's ``graph_fpid`` is the CANONICAL graph fingerprint (matching
    what the graph-create step persists), not the pre-canonical one the client
    sent. A Custom SQL table's synthetic name is recomputed server-side, so
    without canonicalizing here the pquery would reference an unsaved graph
    fpid."""
    original_graph = _custom_sql_graph_definition()
    state_services = SimpleNamespace(
        pqueries=SimpleNamespace(),
        connectors=SimpleNamespace(get_for_project=AsyncMock(
            return_value=SimpleNamespace(config=SimpleNamespace(
                type=SimpleNamespace(value='snowflake'))))),
    )
    # Expected canonical fingerprint, resolved by the same shared helper.
    canonical_graph = await canonicalize_graph_custom_sql_tables(
        original_graph, _TEST_PROJECT_ID, state_services, None)
    assert canonical_graph.fpid != original_graph.fpid

    async def fake_validate_predictive_query_args(self, *args, **kwargs):
        return MagicMock(name='validated-snapshot')

    async def fake_parse_predictive_query_v2(self, *args, **kwargs):
        pquery = SimpleNamespace(get_task_type=lambda: None, query_type=None,
                                 lp_bp_layout='separate_lhs_rhs')
        parsed = SimpleNamespace(
            config=SimpleNamespace(lp_bp_layout='separate_lhs_rhs'),
            predictive_query=pquery)
        return parsed, ValidationResponse()

    monkeypatch.setattr(PQueryLogic, '_validate_predictive_query_args',
                        fake_validate_predictive_query_args)
    monkeypatch.setattr(PQueryLogic, 'parse_predictive_query_v2',
                        fake_parse_predictive_query_v2)

    logic = PQueryLogic(
        data_backend_ctx=MagicMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )
    pquery_resource = SimpleNamespace(
        graph=original_graph,
        name='saved_query',
        query_string='PREDICT responses.is_sensitive FOR EACH '
        'responses.response_id',
        list_distinct_target_is_array=None,
        lp_bp_layout='separate_lhs_rhs',
    )

    pq_def = await logic.build_pquery_def_v2(
        pquery_resource=pquery_resource,
        pquery_settings=MagicMock(),
        graph=MagicMock(name='graph'),
    )

    # The pquery references the canonical graph fpid, never the provisional one.
    assert pq_def.graph_fpid == canonical_graph.fpid
    assert pq_def.graph_fpid != original_graph.fpid


@pytest.mark.parametrize('source_kind', ['file', 'warehouse'])
def test_predictive_query_activity_preserves_missing_source_error(
        monkeypatch, source_kind):
    connector = CompositeFileConnector(
        'csv',
        root_dir=str(
            Path(__file__).resolve().parents[2] / 'csv_data' / 'movielens'),
    )
    graph = Graph(name='missing_source_graph')
    graph.connect(connector)
    graph.add(
        'events',
        pkey='userId',
        source_name='tag_small',
        cols=[
            GraphColumn('userId', dtype=Dtype.int64, stype=Stype.ID),
            GraphColumn('tag', dtype=Dtype.string, stype=Stype.categorical),
            GraphColumn('timestamp', dtype=Dtype.time, stype=Stype.timestamp),
        ],
        time_col='timestamp',
    )
    config = graph.config()
    config.tables[0].source_name = 'missing_source_table'
    config.tables[0].data_version = None

    if source_kind == 'warehouse':

        def missing_warehouse_table(self, table_name):
            raise SourceTableNotFoundError(table_name, 'warehouse')

        monkeypatch.setattr(FileConnector, '_get_data_version',
                            missing_warehouse_table)

    @contextmanager
    def authorized_graph(_config, _references):
        yield Graph.from_config(config)

    monkeypatch.setattr(pquery_logic, 'authorized_graph_for_activity',
                        authorized_graph)
    args = PQueryValidationArgs(
        name='missing-source-table',
        query_string='PREDICT events.tag FOR EACH events.userId',
        graph=GraphDefinition(tables={}, col_groups=[]),
    )

    with pytest.raises(SourceTableNotFoundError) as error:
        pquery_logic.validate_predictive_query_activity(
            args,
            SimpleNamespace(data_source_dict={}),
            PredictiveQuerySettings(),
            {},
        )

    assert error.value.http_code == 422
    assert 'missing_source_table' in str(error.value)
