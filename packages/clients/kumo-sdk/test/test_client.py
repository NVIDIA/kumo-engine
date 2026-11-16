# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import FrozenInstanceError
from typing import Dict, List

import pytest
import requests
from kumoapi.data_snapshot import TableSnapshotID
from kumoapi.data_source import (
    BigQueryIdentityConfig,
    ConnectorResponse,
    CreateConnectorArgs,
    GCSIdentityConfig,
    GoogleServiceAccountKeyAuthentication,
    GoogleWorkloadIdentityAuthentication,
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
)
from kumoapi.graph import GraphDefinition
from kumoapi.jobs import (
    ArtifactExportRequest,
    LocalPredictionOutput,
    PredictionArtifactType,
)
from kumoapi.source_table import (
    FileType,
    S3RelativeSource,
    S3SourceTable,
    TableStats,
)
from kumoapi.table import TableDefinition
from pydantic import ValidationError

from kumoai_studio.client.client import KumoClient
from kumoai_studio.client.utils import parse_response
from kumoai_studio.exceptions import HTTPException

GOOD_KEY = "MOCK_GOOD_API_KEY"


def test_artifact_export_client_preserves_explicit_request_id(requests_mock):
    base_url = 'https://kumo.ai'
    requests_mock.post(f'{base_url}/v1/artifact',
                       json={'job_id': 'export-job'})
    client = KumoClient(base_url, 'api-key')
    request_id = '8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d'
    request = ArtifactExportRequest(
        request_id=request_id,
        job_id='batch-job',
        prediction_output=LocalPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS),
    )

    assert client.artifact_export_api.create(
        request, project_id='project-a') == 'export-job'
    assert requests_mock.last_request.json()['request_id'] == request_id


def test_table_snapshot_exposes_public_row_and_size_statistics(requests_mock):
    base_url = 'https://kumo.ai'
    snapshot_id = TableSnapshotID('table-fingerprint@data-version')
    requests_mock.get(
        f'{base_url}/v1/tablesnapshots/{snapshot_id}',
        json={
            'id': snapshot_id,
            'stats': {
                'size_bytes': 128,
                'num_rows': 5,
            },
            'stages': {},
        },
    )
    client = KumoClient(base_url, 'api-key')

    resource = client.table_api.get_snapshot(snapshot_id,
                                             project_id='project-sales')

    assert resource.stats is not None
    assert resource.stats == TableStats(size_bytes=128, num_rows=5)
    assert resource.stats.num_rows == 5
    assert requests_mock.last_request.qs == {
        'project_id': ['project-sales'],
    }


def test_create_table_requires_and_sends_its_name(requests_mock):
    base_url = 'https://kumo.ai'
    requests_mock.post(f'{base_url}/v1/tables', json={'id': 'table-001'})
    client = KumoClient(base_url, 'api-key')
    definition = TableDefinition(
        cols=[],
        source_table=S3SourceTable(
            connector_id='connector-events',
            source=S3RelativeSource(path='events.parquet'),
            file_type=FileType.PARQUET,
        ),
    )

    with pytest.raises(TypeError, match='name_alias'):
        client.table_api.create_table(  # ty: ignore[missing-argument]
            definition)

    assert client.table_api.create_table(
        definition,
        name_alias='customer-events',
        force_rename=True,
        project_id='project-sales',
    ) == 'table-001'
    assert requests_mock.last_request.qs == {
        'force_rename': ['true'],
        'name_alias': ['customer-events'],
        'project_id': ['project-sales'],
    }


def test_create_graph_threads_server_validation_choice(requests_mock):
    base_url = 'https://kumo.ai'
    requests_mock.post(f'{base_url}/v1/graphs', json={'id': 'graph-id'})
    client = KumoClient(base_url, 'api-key')

    result = client.graph_api.create_graph(
        GraphDefinition(tables={}, col_groups=[]),
        project_id='project-a',
        skip_validation=True,
    )

    assert result == 'graph-id'
    assert requests_mock.last_request.qs == {
        'force_rename': ['false'],
        'project_id': ['project-a'],
        'skip_validation': ['true'],
    }


@pytest.fixture
def mock_auth(requests_mock):
    r"""Matcher for mock authentication to return a 200 status code on a
    valid API key, 404 otherwise.
    """
    def matcher(req):
        res = requests.Response()
        if req.headers.get("X-API-Key") != GOOD_KEY:
            res.status_code = 404
        else:
            res.status_code = 200
        return res

    requests_mock._adapter.add_matcher(matcher)
    yield


@pytest.mark.parametrize(
    "api_key, ok",
    [("MOCK_BAD_API_KEY", False), (GOOD_KEY, True)],
)
def test_client_creation(api_key: str, ok: bool, mock_auth):
    base_url = "http://kumo.ai"
    api_key = api_key
    client = KumoClient(base_url, api_key)
    if ok:
        client.authenticate()
    else:
        with pytest.raises(ValueError):
            client.authenticate()


@pytest.mark.parametrize(
    "verify_ssl, expected_verify",
    [(True, True), (False, False), (None, True)],  # None tests default
)
def test_client_ssl_verification_setting(verify_ssl, expected_verify,
                                         requests_mock):
    """Test that verify_ssl parameter is properly stored and used."""
    base_url = "https://kumo.ai"
    api_key = "test_key"

    requests_mock.get(f"{base_url}/v1/authentication", status_code=204)

    if verify_ssl is None:
        client = KumoClient(base_url, api_key)  # Test default
    else:
        client = KumoClient(base_url, api_key, verify_ssl=verify_ssl)

    assert client._verify_ssl == expected_verify

    # Trigger a request and verify the verify parameter was passed correctly
    client.authenticate()
    assert requests_mock.last_request.verify == expected_verify


def test_list_identities_returns_immutable_safe_metadata_for_project(
        requests_mock):
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/projects/project-sales/identities",
        json=[{
            "id": "identity-databricks",
            "slug": "sales-service-account",
            "display_name": "Sales service account",
            "description": "",
            "timestamps": {
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            "credential_status": {
                "configured": True,
                "last_rotated_at": "2026-01-01T00:00:00Z",
            },
            "config": {
                "type": "databricks",
                "host": "https://workspace.example",
                "client_id": "service-principal",
            },
        }],
    )
    client = KumoClient(base_url, "api-key")

    identities = client.identity_api.list("project-sales")

    assert [identity.id for identity in identities] == ["identity-databricks"]
    assert identities[0].slug == "sales-service-account"
    assert identities[0].name == "Sales service account"
    assert identities[0].type.value == "databricks"
    assert requests_mock.last_request.qs == {}
    assert not hasattr(identities[0], "credentials")
    with pytest.raises(FrozenInstanceError):
        setattr(identities[0], "name", "mutated")


def test_google_identity_deserialization_exposes_both_safe_modes_only(
        requests_mock):
    base_url = "https://kumo.ai"
    responses = []
    for provider in ("bigquery", "gcs"):
        for mode in ("service_account_key", "workload_identity"):
            authentication = ({
                "mode":
                "service_account_key",
                "project_id":
                "identity-project",
                "client_email":
                (f"{provider}@identity-project.iam.gserviceaccount.com"),
                "client_id":
                f"{provider}-client-id",
                "auth_uri":
                "https://accounts.google.com/o/oauth2/auth",
                "token_uri":
                "https://oauth2.googleapis.com/token",
                "universe_domain":
                "googleapis.com",
            } if mode == "service_account_key" else {
                "mode":
                "workload_identity",
                "service_account_email":
                (f"{provider}@identity-project.iam.gserviceaccount.com"),
            })
            responses.append({
                "id": f"identity-{provider}-{mode}",
                "slug": f"{provider}-{mode.replace('_', '-')}",
                "display_name": f"{provider} {mode}",
                "description": "",
                "timestamps": {
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
                "credential_status": {
                    "configured":
                    True,
                    "last_rotated_at": ("2026-01-01T00:00:00Z" if mode
                                        == "service_account_key" else None),
                },
                "config": {
                    "type": provider,
                    "authentication": authentication,
                },
            })
    requests_mock.get(
        f"{base_url}/v1/projects/project-google/identities",
        json=responses,
    )
    client = KumoClient(base_url, "api-key")

    identities = client.identity_api.list("project-google")

    assert len(identities) == 4
    for identity in identities:
        config = identity.config
        expected_config_type = (BigQueryIdentityConfig if identity.type.value
                                == "bigquery" else GCSIdentityConfig)
        assert isinstance(config, expected_config_type)
        mode = config.authentication.mode
        expected_auth_type = (GoogleServiceAccountKeyAuthentication
                              if mode == "service_account_key" else
                              GoogleWorkloadIdentityAuthentication)
        assert isinstance(config.authentication, expected_auth_type)
        assert config.authentication.mode == mode
        assert not hasattr(identity, "credentials")
        assert "private_key" not in repr(identity)
        assert "access_token" not in repr(identity)


def test_slug_lookup_clients_use_explicit_authorized_routes(requests_mock):
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/projects/by-slug/fraud-detection",
        json={
            "id": "project-fraud",
            "slug": "fraud-detection",
            "display_name": "Fraud Detection",
            "group_id": "group-risk",
        },
    )
    requests_mock.get(
        f"{base_url}/v1/groups/by-slug/risk-team",
        json={
            "id": "group-risk",
            "slug": "risk-team",
            "display_name": "Risk Team",
        },
    )
    requests_mock.get(
        f"{base_url}/v1/projects/project-fraud/identities/by-slug/prod-aws",
        json={
            "id": "identity-aws",
            "slug": "prod-aws",
            "display_name": "Production AWS",
            "description": "",
            "timestamps": {
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            "credential_status": {
                "configured": False,
                "last_rotated_at": None,
            },
            "config": {
                "type": "s3",
                "authentication": {
                    "mode": "ambient"
                },
            },
        },
    )
    client = KumoClient(base_url, "api-key")

    project = client.projects_api.get_project_by_slug("fraud-detection")
    group = client.groups_api.get_group_by_slug("risk-team")
    identity = client.identity_api.get_by_slug("project-fraud", "prod-aws")

    assert project.slug == "fraud-detection"
    assert group.slug == "risk-team"
    assert identity.slug == "prod-aws"


def test_group_client_requires_canonical_id_response_field(requests_mock):
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/groups",
        json=[{
            "group_id": "group-risk",
            "slug": "risk-team",
            "display_name": "Risk Team",
        }],
    )
    client = KumoClient(base_url, "api-key")

    with pytest.raises(KeyError, match="id"):
        client.groups_api.list_groups()


def test_create_project_sends_optional_explicit_slug(requests_mock):
    base_url = "https://kumo.ai"
    matcher = requests_mock.post(
        f"{base_url}/v1/projects",
        json={
            "id": "project-fraud",
            "slug": "fraud-detection",
            "display_name": "Fraud Detection",
            "group_id": "group-risk",
        },
    )
    client = KumoClient(base_url, "api-key")

    project = client.projects_api.create_project(
        "Fraud Detection",
        "group-risk",
        slug="fraud-detection",
    )

    assert project.slug == "fraud-detection"
    assert matcher.last_request.json()["slug"] == "fraud-detection"


_SPA_HTML = (
    '<!doctype html> <html lang="en"> <head> <meta charset="utf-8"/> '
    '<title>Kumo</title> </head> <body> <div id="root"></div> </body> </html>')


def test_non_json_html_200_raises_actionable_error(requests_mock):
    """A 200 whose body is an HTML page (misrouted URL landing on the web UI)
    must raise an actionable error, not an opaque JSONDecodeError.
    """
    base_url = "https://kumo.ai"
    url = f"{base_url}/v1/projects/default"
    requests_mock.get(
        url,
        status_code=200,
        text=_SPA_HTML,
        headers={'content-type': 'text/html'},
    )
    client = KumoClient(base_url, "api-key")

    with pytest.raises(HTTPException) as error:
        client.projects_api.get_project("default")

    detail = error.value.detail
    assert error.value.status_code == 200
    # The error must name the URL, status, content-type, and an HTML/URL hint.
    assert url in detail
    assert "HTTP 200" in detail
    assert "text/html" in detail
    assert "web UI" in detail
    assert "/api" in detail
    # A snippet of the body must be included to aid diagnosis.
    assert "<!doctype html>" in detail


def test_valid_json_served_as_html_content_type_still_parses(requests_mock):
    """The Kumo REST API serves valid JSON with a ``text/html`` content-type
    on some deployments; the fix must never reject on content-type alone.
    """
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/projects/default",
        status_code=200,
        json={
            "id": "project-default",
            "slug": "default",
            "display_name": "Default",
            "group_id": "group-a",
        },
        headers={'content-type': 'text/html; charset=UTF-8'},
    )
    client = KumoClient(base_url, "api-key")

    project = client.projects_api.get_project("default")

    assert project.id == "project-default"
    assert project.slug == "default"


@pytest.mark.parametrize('response_type', [
    pytest.param(GraphDefinition, id='case-3'),
    pytest.param(List[str], id='list'),
    pytest.param(Dict[str, int], id='dict'),
])
def test_parse_response_non_json_html_raises_actionable_error(
        requests_mock, response_type):
    url = 'https://kumo.ai/v1/parse-response'
    requests_mock.get(
        url,
        status_code=200,
        text=_SPA_HTML,
        headers={'content-type': 'text/html'},
    )

    with pytest.raises(HTTPException) as error:
        parse_response(response_type, requests.get(url))

    assert error.value.status_code == 200
    assert url in error.value.detail
    assert 'web UI' in error.value.detail


def test_parse_response_preserves_schema_mismatch_error(requests_mock):
    url = 'https://kumo.ai/v1/parse-response'
    requests_mock.get(
        url,
        status_code=200,
        json={'unexpected': True},
        headers={'content-type': 'text/html'},
    )

    with pytest.raises(ValidationError):
        parse_response(GraphDefinition, requests.get(url))


@pytest.mark.parametrize('detail', [
    'Ambient S3 workload identity is disabled.',
    ('No available s3 identity could access this target. Ask an administrator '
     'to grant access or pass an explicit identity.'),
])
def test_omitted_s3_identity_preserves_sanitized_resolution_error(
        requests_mock, detail):
    base_url = 'https://kumo.ai'
    requests_mock.post(
        f'{base_url}/v1/projects/project-risk/connectors',
        status_code=422,
        json={'detail': detail},
    )
    client = KumoClient(base_url, 'api-key')
    args = CreateConnectorArgs(
        name='s3',
        config=S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()),
    )

    with pytest.raises(HTTPException) as error:
        client.connector_api.create_if_not_exist(args, 'project-risk')

    assert error.value.status_code == 422
    assert error.value.detail == detail


def test_explicit_missing_s3_identity_preserves_not_found(requests_mock):
    base_url = 'https://kumo.ai'
    detail = "Identity with id='identity_missing' does not exist"
    requests_mock.post(
        f'{base_url}/v1/projects/project-risk/connectors',
        status_code=404,
        json={'detail': detail},
    )
    client = KumoClient(base_url, 'api-key')
    args = CreateConnectorArgs(
        name='s3',
        identity_id='identity_missing',
        config=S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget()),
    )

    with pytest.raises(HTTPException) as error:
        client.connector_api.create_if_not_exist(args, 'project-risk')

    assert error.value.status_code == 404
    assert error.value.detail == detail


def test_get_project_connector_parses_identity_reference(requests_mock):
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/projects/project-sales/connectors/connector-sales",
        json={
            "id": "connector-sales",
            "project_id": "project-sales",
            "identity_id": "identity-snowflake",
            "name": "sales",
            "timestamps": {
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            "config": {
                "type": "snowflake",
                "warehouse": "COMPUTE_WH",
                "database": "ANALYTICS",
                "schema_name": "SALES",
            },
        },
    )
    client = KumoClient(base_url, "api-key")

    connector = client.connector_api.get("connector-sales", "project-sales")

    assert isinstance(connector, ConnectorResponse)
    assert str(connector.identity_id) == "identity-snowflake"
    assert connector.project_id == "project-sales"
    assert requests_mock.last_request.qs == {}


def test_list_connectors_is_exclusively_project_scoped(requests_mock):
    base_url = "https://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/projects/project-sales/connectors",
        json=[
            {
                "id": "connector-sales",
                "project_id": "project-sales",
                "identity_id": "identity-bigquery",
                "name": "sales",
                "timestamps": {
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
                "config": {
                    "type": "bigquery",
                    "project_id": "analytics",
                    "dataset_id": "sales",
                    "location": "US",
                },
            },
        ],
    )
    client = KumoClient(base_url, "api-key")

    connectors = client.connector_api.list(project_id="project-sales")

    assert isinstance(connectors[0], ConnectorResponse)
    assert connectors[0].project_id == "project-sales"
    assert requests_mock.last_request.qs == {}


def test_source_table_config_preserves_nested_path_components(requests_mock):
    base_url = 'https://kumo.ai'
    path = ('/v1/projects/project-sales/connectors/connector-events/'
            'source-table-config')
    requests_mock.get(
        f'{base_url}{path}',
        json={
            'source_table': {
                'connector_id': 'connector-events',
                'source': {
                    'mode': 'relative',
                    'path': 'nested/2026/events data.parquet',
                },
                'file_type': 'PARQUET',
                'data_source_type': 'S3',
            },
        },
    )
    client = KumoClient(base_url, 'api-key')

    response = client.source_table_api.get_table_config(
        'connector-events',
        'nested/2026/events data.parquet',
        'project-sales',
    )

    assert isinstance(response.source_table, S3SourceTable)
    assert isinstance(response.source_table.source, S3RelativeSource)
    assert response.source_table.source.path == (
        'nested/2026/events data.parquet')
    assert requests_mock.last_request.path == path
    assert requests_mock.last_request.qs == {
        'table_name': ['nested/2026/events data.parquet']
    }


def test_source_table_error_exposes_structured_server_detail(requests_mock):
    base_url = 'https://kumo.ai'
    path = ('/v1/projects/project-sales/connectors/connector-events/'
            'source-table-config')
    requests_mock.get(
        f'{base_url}{path}',
        status_code=422,
        json={'detail': 'Invalid table missing.parquet'},
        headers={'X-Request-ID': 'request-123'},
    )
    client = KumoClient(base_url, 'api-key')

    with pytest.raises(HTTPException) as error:
        client.source_table_api.get_table_config(
            'connector-events',
            'missing.parquet',
            'project-sales',
        )

    assert error.value.status_code == 422
    assert error.value.detail == 'Invalid table missing.parquet'
    assert error.value.detail in str(error.value)
    assert error.value.headers is not None
    assert error.value.headers['X-Request-ID'] == 'request-123'


@pytest.mark.integration
def test_client_creation_integ():
    base_url = os.environ["KUMO_API_ENDPOINT"]
    api_key = os.environ["KUMO_API_KEY"]
    client = KumoClient(base_url, api_key)
    client.authenticate()


@pytest.mark.parametrize(
    "configured_url",
    ["https://customer.example.com/api", "https://customer.example.com/api/"],
)
def test_versioned_requests_normalize_trailing_slash(requests_mock,
                                                     configured_url):
    canonical_url = "https://customer.example.com/api"
    requests_mock.get(f"{canonical_url}/v1/authentication", status_code=204)
    requests_mock.get(f"{canonical_url}/v1/groups", json=[])

    client = KumoClient(configured_url, "api-key")
    client.authenticate()

    assert client._url == canonical_url
    assert client.groups_api.list_groups() == []
    assert [request.url for request in requests_mock.request_history] == [
        f"{canonical_url}/v1/authentication",
        f"{canonical_url}/v1/groups",
    ]


@pytest.mark.parametrize(
    "configured_url",
    [
        "https://customer.example.com/api?signature=a%2Fb/",
        "https://customer.example.com/api#deployment/",
    ],
)
def test_query_or_fragment_base_is_preserved_verbatim(configured_url):
    assert KumoClient(configured_url, "api-key")._url == configured_url


def test_authenticate_connection_error_message(requests_mock):
    """Server unreachable must not be reported as an auth failure."""
    base_url = "http://localhost:65535"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        exc=requests.exceptions.ConnectionError(
            "Connection refused: [Errno 111]"),
    )
    client = KumoClient(base_url, "any-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    message = str(exc.value)
    assert "Could not connect" in message
    assert base_url in message
    # Must not blame the API key when the real issue is connectivity.
    assert "valid API key" not in message


def test_authenticate_auth_error_message(requests_mock):
    """A 401 must still be reported as an authentication failure."""
    base_url = "http://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        status_code=401,
        text="unauthorized",
    )
    client = KumoClient(base_url, "bad-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    assert "authentication failed" in str(exc.value).lower()
    assert base_url in str(exc.value)


def test_authenticate_server_error_message(requests_mock):
    """A 5xx must be reported as a server error, not an auth error."""
    base_url = "http://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        status_code=500,
        text="boom",
    )
    client = KumoClient(base_url, "any-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    message = str(exc.value)
    assert "HTTP 500" in message
    # Don't blame credentials for a 500.
    assert "valid API key" not in message
    assert "credentials" not in message.lower()


@pytest.mark.parametrize("status_code", [401, 403])
def test_authenticate_auth_status_codes_blame_credentials(
        status_code, requests_mock):
    """401 / 403 must be classified as authentication failures.

    The Kumo backend only ever raises 401 (missing / invalid / expired /
    blocked API key) or 403 (insufficient permissions) for auth failures.
    These are the only status codes that should ask the user to check
    their credentials.
    """
    base_url = "http://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        status_code=status_code,
        text="nope",
    )
    client = KumoClient(base_url, "bad-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    message = str(exc.value)
    assert "authentication failed" in message.lower()
    assert f"HTTP {status_code}" in message
    assert base_url in message


@pytest.mark.parametrize("status_code", [404, 500, 502, 503])
def test_authenticate_non_auth_status_codes_do_not_blame_credentials(
        status_code, requests_mock):
    """Non-auth HTTP errors (including 404) must be classified as server
    errors, never as credential problems.

    A 404 from the Kumo backend signals "endpoint not found" (wrong URL,
    version skew, proxy misroute) — never an invalid API key — so we
    must not mislead the user into rotating credentials.
    """
    base_url = "http://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        status_code=status_code,
        text="boom",
    )
    client = KumoClient(base_url, "any-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    message = str(exc.value)
    assert f"HTTP {status_code}" in message
    # Server errors must not blame credentials.
    assert "authentication failed" not in message.lower()
    assert "valid API key" not in message
    assert "credentials" not in message.lower()


def test_authenticate_timeout_message(requests_mock):
    """A timeout must be reported as a connectivity issue, not an auth
    failure.
    """
    base_url = "http://kumo.ai"
    requests_mock.get(
        f"{base_url}/v1/authentication",
        exc=requests.exceptions.ConnectTimeout("slowloris"),
    )
    client = KumoClient(base_url, "any-key")
    with pytest.raises(ValueError) as exc:
        client.authenticate()

    message = str(exc.value)
    assert "Timed out" in message
    assert "valid API key" not in message
    assert "credentials" not in message.lower()


def test_kumo_client_satisfies_the_transport_protocol(protocol_members):
    r"""The typed API classes depend on `Transport`, not on `KumoClient`.

    That inversion is what breaks the import cycle, so it only holds while
    `KumoClient` keeps providing the whole transport surface.
    """
    from kumoai_studio.client.client import KumoClient
    from kumoai_studio.client.transport import Transport

    missing = [
        attr for attr in protocol_members(Transport)
        if not hasattr(KumoClient, attr)
    ]
    assert not missing, f'KumoClient is missing {missing}'


@pytest.mark.parametrize('module_name', [
    'connector',
    'graph',
    'groups',
    'identity',
    'jobs',
    'pquery',
    'projects',
    'source_table',
    'table',
])
def test_api_modules_do_not_import_the_client(module_name):
    r"""Guards the cycle: `client.py` imports these at module scope, so none
    of them may import it back.
    """
    import importlib

    module = importlib.import_module(f'kumoai_studio.client.{module_name}')
    assert 'KumoClient' not in vars(module)


def test_client_resolves_its_api_classes_at_module_scope():
    r"""The API properties no longer need function-local imports."""
    from kumoai_studio.client import client as client_module

    for name in ('TableAPI', 'GraphAPI', 'PQueryAPI', 'TrainingJobAPI',
                 'BatchPredictionJobAPI', 'ArtifactExportJobAPI'):
        assert name in vars(client_module), name
