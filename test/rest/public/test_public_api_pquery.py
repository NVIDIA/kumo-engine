# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import kumoapi.data_source as data_source_api
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID
from pydantic import SecretStr
from temporalio import activity

from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.graph import Graph
from kumo.pquery.predictive_query import PredictiveQueryV2
from rest.backends.data_backend.data_backend import DataBackend
from rest.deps.authentication import UserAuthorization
from rest.driver import pquery_logic
from rest.driver.pquery_logic import PQueryLogic
from statedb.config.common import ProjectID

# Converted owner for Phase 1b pquery/model-plan coverage. Keep this file
# API-visible only: public_api_client plus raw HTTP JSON assertions.
_PQUERY_RESOURCE_PATH = (Path(__file__).resolve().parents[1] / "test_data" /
                         "pquery_resource.json")
_FINANCIAL_DATA_PATH = (Path(__file__).resolve().parents[2] / "csv_data" /
                        "FINANCIAL")


def _project_id(client: TestClient) -> str:
    del client
    return DEFAULT_PROJECT_ID


def _load_pquery_resource(
    client: TestClient,
    *,
    named: bool = True,
) -> dict:
    pquery = json.loads(_PQUERY_RESOURCE_PATH.read_text(encoding="utf-8"))
    pquery["desc"] = ""
    if not named:
        pquery["name"] = None
    project_id = _project_id(client)
    for table in pquery["graph"]["tables"].values():
        source_filename = table["source_table"]["source"]["path"]
        source_path = _FINANCIAL_DATA_PATH / source_filename
        source_table_name = source_path.stem
        start = client.post(
            f"/v1/projects/{project_id}/connectors/uploads/start",
            json={
                "source_table_name": source_table_name,
                "file_type": "CSV",
                "num_parts": 1,
            },
        )
        assert start.status_code == HTTPStatus.OK, start.text
        upload = start.json()
        part_url = upload["presigned_part_urls"]["1"]
        part = client.put(part_url, content=source_path.read_bytes())
        assert part.status_code == HTTPStatus.OK, part.text
        complete = client.post(
            f"/v1/projects/{project_id}/connectors/uploads/complete",
            json={
                "source_table_name":
                source_table_name,
                "file_type":
                "CSV",
                "temp_upload_path":
                upload["temp_upload_path"],
                "upload_id":
                upload["upload_id"],
                "parts_metadata": [{
                    "part_number": 1,
                    "etag": part.json()["etag"],
                }],
                "validate_data":
                False,
            },
        )
        assert complete.status_code == HTTPStatus.OK, complete.text
        table["source_table"]["source"] = {
            "mode": "relative",
            "path": source_table_name,
        }
    response = client.get(
        f"/v1/projects/{project_id}/connectors/uploads/connector")
    assert response.status_code == HTTPStatus.OK, response.text
    connector_id = response.json()["connector_id"]
    for table in pquery["graph"]["tables"].values():
        table["source_table"]["connector_id"] = connector_id
    return pquery


def _response_json(response):
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def _post_json(
    client: TestClient,
    path: str,
    payload: dict,
    *,
    params: dict | None = None,
) -> dict:
    request_params = dict(params or {})
    request_params.setdefault("project_id", _project_id(client))
    return _response_json(client.post(path, params=request_params,
                                      json=payload))


def _get_json(
    client: TestClient,
    path: str,
    *,
    params: dict | None = None,
) -> dict | list:
    request_params = dict(params or {})
    request_params.setdefault("project_id", _project_id(client))
    return _response_json(client.get(path, params=request_params))


def _assert_error(response, status_code: HTTPStatus,
                  expected_text: str) -> None:
    assert response.status_code == status_code, response.text
    assert expected_text in response.text


def _create_pquery(client: TestClient, pquery: dict) -> str:
    response = _post_json(
        client,
        "/v1/predictive_queries/",
        pquery,
        params={"dialect": "PQLANG_V2"},
    )
    pquery_id = response["id"]
    assert pquery_id.startswith("pquery-")
    return pquery_id


def _create_graph(client: TestClient, pquery: dict) -> str:
    response = _post_json(
        client,
        "/v1/graphs",
        pquery["graph"],
        params={"skip_validation": "true"},
    )
    graph_id = response["id"]
    assert graph_id.startswith("graph-")
    return graph_id


def _assert_pquery_matches(actual: dict, expected: dict) -> None:
    assert actual["name"] == expected["name"]
    assert actual["query_string"] == expected["query_string"]
    assert actual["desc"] == expected["desc"]
    assert set(actual["graph"]["tables"]) == set(expected["graph"]["tables"])
    assert actual["graph"]["col_groups"] == expected["graph"]["col_groups"]


def _assert_single_listed_pquery(
    pqueries: list[dict],
    *,
    expected_name: str,
    expected_query_string: str,
) -> None:
    assert len(pqueries) == 1
    pquery = pqueries[0]
    assert pquery["name"] == expected_name
    assert pquery["query_string"] == expected_query_string
    assert pquery["graph"]["tables"]


def _model_plan_request(pquery: dict, graph_id: str, run_mode: str) -> dict:
    return {
        "query_string": pquery["query_string"],
        "graph_id": graph_id,
        "run_mode": run_mode,
    }


def _assert_fast_model_plan(plan_response: dict) -> dict:
    model_plan = plan_response["model_plan"]
    training_job = model_plan["training_job"]
    assert training_job["num_experiments"] == 2
    assert training_job["metrics"] == ["auroc", "ap", "auprc"]
    assert training_job["tune_metric"] == "auroc"
    return model_plan


def _google_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode('utf-8')


async def _create_identity_bound_warehouse_connector(
    state_services,
    monkeypatch,
    provider: str,
) -> str:
    monkeypatch.setenv('DISABLE_AUTH', 'true')
    admin = UserAuthorization.disabled()
    project = await state_services.projects.get_project(
        ProjectID(DEFAULT_PROJECT_ID), admin)
    monkeypatch.setattr(state_services.identities, '_identity_validator',
                        lambda *_args: None)
    monkeypatch.setattr(state_services.connectors, '_connector_validator',
                        lambda *_args: None)
    if provider == 'snowflake':
        identity_args = data_source_api.CreateIdentityArgs(
            display_name='PQuery Snowflake identity',
            config=data_source_api.SnowflakeIdentityConfig(
                account='organization-account',
                username='KUMO_READER',
                authentication=(
                    data_source_api.SnowflakeAuthenticationKind.PASSWORD),
            ),
            credentials=data_source_api.SnowflakePasswordCredentials(
                password=SecretStr('snowflake-route-secret')),
        )
        resource_config = data_source_api.SnowflakeConnectorResourceConfig(
            role='KUMO_READER',
            warehouse='COMPUTE_WH',
            database='PRODUCTION',
            schema_name='ANALYTICS',
        )
    else:
        identity_args = data_source_api.CreateIdentityArgs(
            display_name='PQuery BigQuery identity',
            config=data_source_api.BigQueryIdentityConfig(
                authentication=(
                    data_source_api.GoogleServiceAccountKeyAuthentication(
                        project_id='service-project',
                        client_email=(
                            'kumo@service-project.iam.gserviceaccount.com'),
                        client_id='1234567890',
                        auth_uri='https://accounts.google.com/o/oauth2/auth',
                        token_uri='https://oauth2.googleapis.com/token',
                    )), ),
            credentials=data_source_api.GoogleServiceAccountKeyCredentials(
                private_key_id=SecretStr('bigquery-key-id'),
                private_key=SecretStr(_google_private_key_pem()),
            ),
        )
        resource_config = data_source_api.BigQueryConnectorResourceConfig(
            project_id='data-project',
            dataset_id='events',
            location='US',
            billing_project_id='billing-project',
        )
    identity = await state_services.identities.create(identity_args, admin)
    await state_services.identities.replace_groups(
        identity.id,
        data_source_api.ReplaceIdentityGroupsArgs(
            group_ids=[str(project.group_id)]),
        admin,
    )
    create_args = data_source_api.CreateConnectorArgs(
        identity_id=str(identity.id),
        name=f'PQuery {provider} connector',
        config=resource_config,
    )
    authorizations = await state_services.connectors.authorize_create(
        ProjectID(DEFAULT_PROJECT_ID), create_args, admin)
    connector = await state_services.connectors.create_from_activity(
        ProjectID(DEFAULT_PROJECT_ID),
        create_args,
        admin.user_id,
        set(authorizations),
    )
    return str(connector.id)


def _warehouse_pquery(provider: str, connector_id: str) -> dict:
    if provider == 'snowflake':
        source = {
            'snowflake_connector_id': connector_id,
            'database': 'PRODUCTION',
            'schema_name': 'ANALYTICS',
            'table': 'events',
            'data_source_type': 'SNOWFLAKE',
        }
    else:
        source = {
            'bigquery_connector_id': connector_id,
            'project_id': 'data-project',
            'dataset_id': 'events',
            'table_name': 'events',
            'data_source_type': 'BIGQUERY',
        }
    return {
        'name': f'validate_{provider}',
        'query_string':
        ("PREDICT events.label = 'yes' FOR EACH events.entity_id"),
        'desc': '',
        'graph': {
            'tables': {
                'events': {
                    'cols': [{
                        'name': 'entity_id',
                        'stype': 'ID',
                        'dtype': 'int',
                    }, {
                        'name': 'label',
                        'stype': 'categorical',
                        'dtype': 'string',
                    }],
                    'source_table':
                    source,
                    'pkey':
                    'entity_id',
                    'time_col':
                    None,
                },
            },
            'col_groups': [],
        },
    }


@pytest.mark.parametrize(
    ('provider', 'prepopulated_version'),
    [
        ('snowflake', False),
        ('snowflake', True),
        ('bigquery', False),
        ('bigquery', True),
    ],
    ids=[
        'sf-missing-version', 'sf-pinned-version', 'bq-missing-version',
        'bq-pinned-version'
    ],
)
def test_validate_pquery_route_uses_identity_bound_activity_once(
    public_api_client: TestClient,
    state_services,
    monkeypatch,
    provider: str,
    prepopulated_version: bool,
) -> None:
    connector_id = asyncio.run(
        _create_identity_bound_warehouse_connector(state_services, monkeypatch,
                                                   provider))
    connector_type = (SnowflakeConnector
                      if provider == 'snowflake' else BigQueryConnector)

    def require_activity_boundary():
        assert activity.in_activity(), 'request-side provider/PQuery I/O'

    open_provider = MagicMock(side_effect=lambda *_args, **_kwargs:
                              (require_activity_boundary(), object())[1])
    close_provider = MagicMock(
        side_effect=lambda *_args, **_kwargs: require_activity_boundary())
    get_data_version = MagicMock(side_effect=lambda *_args, **_kwargs: (
        require_activity_boundary(), f'{provider}-version')[1])
    monkeypatch.setattr(connector_type, '_open', open_provider)
    monkeypatch.setattr(connector_type, '_close', close_provider)
    monkeypatch.setattr(connector_type, 'validate_permissions',
                        lambda _self: None)
    monkeypatch.setattr(connector_type, '_get_data_version', get_data_version)

    activity_calls = 0
    validate_args = PQueryLogic._validate_predictive_query_args

    async def track_validate_args(self, *args, **kwargs):
        nonlocal activity_calls
        activity_calls += 1
        return await validate_args(self, *args, **kwargs)

    monkeypatch.setattr(PQueryLogic, '_validate_predictive_query_args',
                        track_validate_args)

    make_graph = PQueryLogic._make_graph_from_graph_def_v2

    async def make_versioned_graph(self, *args, **kwargs):
        graph = await make_graph(self, *args, **kwargs)
        if prepopulated_version:
            for table in graph.tables:
                table._data_version = f'{provider}-pinned-version'
        return graph

    monkeypatch.setattr(PQueryLogic, '_make_graph_from_graph_def_v2',
                        make_versioned_graph)

    make_config = pquery_logic.make_predictive_config_with_graph

    def activity_only_make_config(*args, **kwargs):
        require_activity_boundary()
        return make_config(*args, **kwargs)

    monkeypatch.setattr(pquery_logic, 'make_predictive_config_with_graph',
                        activity_only_make_config)
    graph_validate = Graph.validate

    def activity_only_graph_validate(self, *args, **kwargs):
        require_activity_boundary()
        return graph_validate.__get__(self, Graph)(*args, **kwargs)

    monkeypatch.setattr(Graph, 'validate', activity_only_graph_validate)
    pquery_validate = PredictiveQueryV2.validate

    def activity_only_pquery_validate(self, *args, **kwargs):
        require_activity_boundary()
        return pquery_validate.__get__(self, PredictiveQueryV2)(*args, **kwargs)

    monkeypatch.setattr(PredictiveQueryV2, 'validate',
                        activity_only_pquery_validate)

    execute_activity = DataBackend._execute_single_activity
    provider_secret = ('snowflake-route-secret'
                       if provider == 'snowflake' else 'bigquery-key-id')

    async def assert_secret_free_activity(*args, **kwargs):
        assert provider_secret not in repr(args)
        result = await execute_activity(*args, **kwargs)
        assert provider_secret not in repr(result)
        return result

    monkeypatch.setattr(DataBackend, '_execute_single_activity',
                        assert_secret_free_activity)

    response = public_api_client.post(
        '/v1/predictive_queries/validate',
        params={
            'project_id': DEFAULT_PROJECT_ID,
            'dialect': 'PQLANG_V2',
        },
        json=_warehouse_pquery(provider, connector_id),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    assert not response.json()['errors']
    assert activity_calls == 1
    expected_provider_reads = 0 if prepopulated_version else 1
    assert open_provider.call_count == expected_provider_reads
    assert get_data_version.call_count == expected_provider_reads
    assert close_provider.call_count == expected_provider_reads


def test_missing_file_source_has_typed_create_and_curated_validate_responses(
        public_api_client: TestClient) -> None:
    pquery = _load_pquery_resource(public_api_client)
    table = next(iter(pquery['graph']['tables'].values()))
    table['source_table']['source']['path'] = 'missing_source_table'
    params = {
        'project_id': _project_id(public_api_client),
        'dialect': 'PQLANG_V2',
    }

    create = public_api_client.post('/v1/predictive_queries/', params=params,
                                    json=pquery)
    assert create.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert create.json()['error_type'] == 'SourceTableNotFoundError'
    assert 'missing_source_table' in create.json()['error']

    validate = public_api_client.post('/v1/predictive_queries/validate',
                                      params=params, json=pquery)
    assert validate.status_code == HTTPStatus.OK, validate.text
    assert validate.json()['errors']
    serialized_errors = json.dumps(validate.json()['errors'])
    assert 'missing_source_table' in serialized_errors
    assert 'ActivityError' not in serialized_errors
    assert 'RPCError' not in serialized_errors


@pytest.mark.parametrize("named_pquery", [True, False])
def test_pquery_api(public_api_client: TestClient, named_pquery: bool):
    pquery = _load_pquery_resource(public_api_client, named=named_pquery)
    pquery_id = _create_pquery(public_api_client, pquery)

    _assert_pquery_matches(
        _get_json(public_api_client, f"/v1/predictive_queries/{pquery_id}"),
        pquery,
    )
    if named_pquery:
        assert pquery["name"]
        _assert_pquery_matches(
            _get_json(public_api_client,
                      f"/v1/predictive_queries/{pquery['name']}"),
            pquery,
        )
        _assert_single_listed_pquery(
            _get_json(public_api_client, "/v1/predictive_queries/"),
            expected_name=pquery["name"],
            expected_query_string=pquery["query_string"],
        )
        _assert_single_listed_pquery(
            _get_json(
                public_api_client,
                "/v1/predictive_queries/",
                params={"name_pattern": pquery["name"]},
            ),
            expected_name=pquery["name"],
            expected_query_string=pquery["query_string"],
        )
        _assert_single_listed_pquery(
            _get_json(
                public_api_client,
                "/v1/predictive_queries/",
                params={"name_pattern": ".*"},
            ),
            expected_name=pquery["name"],
            expected_query_string=pquery["query_string"],
        )
        assert _get_json(
            public_api_client,
            "/v1/predictive_queries/",
            params={"name_pattern": "nonexistquery"},
        ) == []

    graph_id = _create_graph(public_api_client, pquery)
    fast_request = _model_plan_request(pquery, graph_id, "fast")
    _assert_fast_model_plan(
        _post_json(
            public_api_client,
            "/v1/predictive_queries/model_plan",
            fast_request,
            params={"dialect": "PQLANG_V2"},
        ))

    best_request = copy.deepcopy(fast_request)
    best_request["run_mode"] = "best"
    best_plan = _post_json(
        public_api_client,
        "/v1/predictive_queries/model_plan",
        best_request,
        params={"dialect": "PQLANG_V2"},
    )
    assert best_plan["model_plan"]["training_job"]["num_experiments"] == 8

    train_table_plan = _post_json(
        public_api_client,
        "/v1/predictive_queries/train_table_generation_plan",
        best_request,
        params={"dialect": "PQLANG_V2"},
    )
    assert train_table_plan["split"] == "TemporalSplit(ratio=[0.8, 0.1, 0.1])"

    if named_pquery:
        delete_response = public_api_client.delete(
            f"/v1/predictive_queries/{pquery['name']}",
            params={"project_id": _project_id(public_api_client)},
        )
        assert delete_response.status_code == HTTPStatus.OK, delete_response.text
        _assert_error(
            public_api_client.get(
                f"/v1/predictive_queries/{pquery['name']}",
                params={"project_id": _project_id(public_api_client)},
            ),
            HTTPStatus.NOT_FOUND,
            "does not exist",
        )
        assert _get_json(
            public_api_client,
            "/v1/predictive_queries/",
            params={"name_pattern": pquery["name"]},
        ) == []
    else:
        _assert_error(
            public_api_client.delete(
                f"/v1/predictive_queries/{pquery_id}",
                params={"project_id": _project_id(public_api_client)},
            ),
            HTTPStatus.FORBIDDEN,
            "Not allowed to delete a frozen unnamed pquery definition",
        )


def test_infer_task_type_semantic_pql_error_returns_422(
        public_api_client: TestClient):
    pquery = _load_pquery_resource(public_api_client)
    graph_id = _create_graph(public_api_client, pquery)
    params = {
        "project_id": _project_id(public_api_client),
        "graph_id": graph_id,
    }

    response = public_api_client.post(
        "/v1/predictive_queries/infer_task_type",
        params={
            **params, "pquery_string": pquery["query_string"]
        },
    )
    assert response.status_code == HTTPStatus.OK, response.text
    task_type, reason = response.json()
    assert task_type == "binary_classification"
    assert reason

    response = public_api_client.post(
        "/v1/predictive_queries/infer_task_type",
        params={
            **params,
            "pquery_string":
            "PREDICT TRANS.DOES_NOT_EXIST FOR EACH TRANS.TRANS_ID",
        },
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
        response.text)
    body = response.json()
    assert body["error_type"] == "ValidationException"
    assert "Unknown column name" in body["error"]


@pytest.mark.parametrize('path', [
    '/v1/predictive_queries/',
    '/v1/predictive_queries/validate',
])
def test_nested_graph_pydantic_error_is_clean_422(public_api_client, path):
    pquery = _load_pquery_resource(public_api_client)
    columns = pquery['graph']['col_groups'][0]['columns']
    columns.append(copy.deepcopy(columns[0]))

    response = public_api_client.post(
        path,
        params={'project_id': _project_id(public_api_client)},
        json=pquery,
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert 'All columns must be unique' in response.text
    assert 'input_value' not in response.text
    assert 'errors.pydantic.dev' not in response.text


def test_pquery_api_model_plan_weight_tt(public_api_client: TestClient):
    pquery = _load_pquery_resource(public_api_client)
    _create_pquery(public_api_client, pquery)
    graph_id = _create_graph(public_api_client, pquery)

    request = _model_plan_request(pquery, graph_id, "fast")
    request["train_table_spec"] = {"weight_col": "TRANS.AMOUNT"}
    weighted_plan = _post_json(
        public_api_client,
        "/v1/predictive_queries/model_plan",
        request,
        params={"dialect": "PQLANG_V2"},
    )
    model_plan = _assert_fast_model_plan(weighted_plan)
    assert model_plan["optimization"]["weight_mode"] == ["weighted_loss"]

    unweighted_request = copy.deepcopy(request)
    unweighted_request["train_table_spec"] = None
    unweighted_plan = _post_json(
        public_api_client,
        "/v1/predictive_queries/model_plan",
        unweighted_request,
        params={"dialect": "PQLANG_V2"},
    )
    model_plan = _assert_fast_model_plan(unweighted_plan)
    assert model_plan["optimization"]["weight_mode"] == [None]
