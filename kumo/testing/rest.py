# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""
This module contains helper functions required for writing unit tests
against the kumo REST api.
"""
import json
import os
import re
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from http import HTTPStatus
from typing import Any, Optional, TypeVar
from uuid import uuid4

import kumoapi.data_source as data_source_api
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kumo_core.types.schema import FPID
from pydantic import SecretStr
from requests import Response

from io_utils.databricks import DatabricksConnectionInfo
from io_utils.json import from_json, to_json, trusted_custom_pydantic_encoder
from kumo.utils.dataclass import parse_dataclass
from rest.config.api_command import FixtureDefinition
from rest.config.authentication import RequestsResponse
from rest.config.connector import (
    BigQueryConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    SnowflakeConnectorResourceConfig,
)
from rest.config.connector_db import _api_external_config
from rest.config.credentials import KeyPair, UsernamePassword
from rest.config.group import AddGroupMemberRequest, CreateKumoGroupRequest
from rest.config.ml_backend import (
    MaterializationTaskInfo,
    MaterializedTableInfo,
)
from rest.config.status import PatchResponse
from rest.deps.authentication import (
    AUTHORIZATION_API_KEY,
    Permissions,
    RequestContext,
    UserAuthorization,
)
from rest.deps.context import _current_user_ctx
from rest.utils.routers import ValidateNameResponse
from statedb.config.common import (
    BatchPredictionJobID,
    CallToAction,
    ConnectorID,
    GraphIDv1,
    PredictiveQueryID,
    TrainingJobID,
    UserID,
)
from statedb.rest.group import GroupMemberEntry, KumoGroupDBEntry

# The maximum number of characters to display from the HTTP body
# in a pytest failure message.
_RESPONSE_TEXT_DISPLAY_LIMIT = 1000
_JSON_HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}
TEST_REST_ADMIN_USER_ID = 'pytest-rest-admin'

ResponseType = TypeVar('ResponseType')


@dataclass(frozen=True)
class IdentityBindingSetup:
    identity_id: str
    group_id: str


@dataclass(frozen=True)
class IdentityBackedConnectorSetup(IdentityBindingSetup):
    connector_id: str
    project_id: str


@contextmanager
def disabled_user_authorization_context() -> Generator[UserAuthorization]:
    """Supply the explicit auth-disabled admin principal to direct tests."""
    principal = UserAuthorization.disabled()
    token = _current_user_ctx.set(principal)
    try:
        yield principal
    finally:
        _current_user_ctx.reset(token)


@contextmanager
def fixture_admin_user_authorization_context() -> Generator[UserAuthorization]:
    """Supply the fixture admin whose native memberships persist in StateDB."""
    principal = UserAuthorization(
        user_id=UserID(TEST_REST_ADMIN_USER_ID),
        access_token=AUTHORIZATION_API_KEY,
        permissions=[
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
            Permissions.ADMIN.compose(),
        ],
        user_name='Pytest REST administrator',
    )
    token = _current_user_ctx.set(principal)
    try:
        yield principal
    finally:
        _current_user_ctx.reset(token)


def databricks_test_identity(
) -> tuple[data_source_api.DatabricksIdentityConfig,
           data_source_api.DatabricksOAuthCredentials]:
    info = DatabricksConnectionInfo.get()
    client_id = info.credentials.get('client_id')
    client_secret = info.credentials.get('client_secret')
    if client_id is None or client_secret is None:
        raise RuntimeError(
            'Identity tests require Databricks OAuth credentials')
    return (
        data_source_api.DatabricksIdentityConfig(
            host=info.host,
            client_id=client_id.get_secret_value(),
        ),
        data_source_api.DatabricksOAuthCredentials(
            client_secret=client_secret, ),
    )


def snowflake_test_identity(
    credentials: UsernamePassword | KeyPair,
    *,
    account: str = 'xva19026',
) -> tuple[data_source_api.SnowflakeIdentityConfig,
           data_source_api.IdentityCredentials]:
    if isinstance(credentials, UsernamePassword):
        return (
            data_source_api.SnowflakeIdentityConfig(
                account=account,
                username=credentials.username.get_secret_value(),
                authentication=(
                    data_source_api.SnowflakeAuthenticationKind.PASSWORD),
            ),
            data_source_api.SnowflakePasswordCredentials(
                password=credentials.password),
        )
    if credentials.user is None or credentials.private_key is None:
        raise ValueError('Snowflake key-pair credentials require user and key')
    return (
        data_source_api.SnowflakeIdentityConfig(
            account=account,
            username=credentials.user.get_secret_value(),
            authentication=(
                data_source_api.SnowflakeAuthenticationKind.KEY_PAIR),
        ),
        data_source_api.SnowflakeKeyPairCredentials(
            private_key=credentials.private_key,
            private_key_passphrase=credentials.private_key_passphrase,
        ),
    )


def bigquery_test_identity(
) -> tuple[data_source_api.BigQueryIdentityConfig,
           Optional[data_source_api.IdentityCredentials]]:
    workload_service_account = os.getenv('GCP_TEST_SERVICE_ACCOUNT')
    if workload_service_account:
        return (
            data_source_api.BigQueryIdentityConfig(
                authentication=(
                    data_source_api.GoogleWorkloadIdentityAuthentication(
                        service_account_email=workload_service_account, )), ),
            None,
        )

    from kumo.connector.bigquery_connector import (
        ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH,
        ENV_GOOGLE_CREDENTIALS_JSON,
    )

    path = (os.getenv(ENV_GOOGLE_CREDENTIALS_JSON)
            or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            or ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH)
    with open(path, encoding='utf-8') as credentials_file:
        payload = json.load(credentials_file)
    if payload.get('type') != 'service_account':
        raise ValueError('BigQuery test credentials must be a service account')
    return (
        data_source_api.BigQueryIdentityConfig(
            authentication=(
                data_source_api.GoogleServiceAccountKeyAuthentication(
                    project_id=payload['project_id'],
                    client_email=payload['client_email'],
                    client_id=payload['client_id'],
                    auth_uri=payload['auth_uri'],
                    token_uri=payload['token_uri'],
                    universe_domain=payload.get('universe_domain',
                                                'googleapis.com'),
                )), ),
        data_source_api.GoogleServiceAccountKeyCredentials(
            private_key_id=SecretStr(payload['private_key_id']),
            private_key=SecretStr(payload['private_key']),
        ),
    )


def invalid_bigquery_key_test_identity_config(
) -> data_source_api.BigQueryIdentityConfig:
    """Build key-auth metadata for testing invalid private-key rejection."""
    return data_source_api.BigQueryIdentityConfig(
        authentication=data_source_api.GoogleServiceAccountKeyAuthentication(
            project_id='kumo-engineering',
            client_email=('invalid-bigquery-test@'
                          'kumo-engineering.iam.gserviceaccount.com'),
            client_id='0',
            auth_uri='https://accounts.google.com/o/oauth2/auth',
            token_uri='https://oauth2.googleapis.com/token',
            universe_domain='googleapis.com',
        ), )


# When running unit tests, we don't pass the token into the header, we
# just check the overrided dependency. The authorization_middleware request
# doesn't have an access token in its header
def inject_access_token(test_token: RequestsResponse) -> dict[str, str]:
    """
    Create a header with the access token present for use in testing
    live authentication
    """
    return {'authorization': f'Bearer {test_token.get("access_token")}'}


def _combine_headers(headers: Mapping[str, str],
                     extra_values: Mapping[str, str]) -> Mapping[str, str]:
    x = dict(headers.items())
    x.update(extra_values)
    return x


def get_and_expect_success(client: TestClient, path: str, params: dict[str,
                                                                       str],
                           response_type: type[ResponseType],
                           headers: Mapping[str, str] = {},
                           skip_cache: bool = False) -> ResponseType:
    """Send a get request and parse the response into a Pydantic dataclass.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        params (Dict[str, str]): a dict of url parameters to send
        response_type (Type[ResponseType]): please refer to parse_dataclass()

    Returns:
        ResponseType: the parsed response
    """
    request_params = dict(params)
    if skip_cache:
        request_params['skip_cache'] = 'true'
    response = client.get(path, params=request_params, headers=headers,
                          follow_redirects=True)
    _check_status_code(response, HTTPStatus.OK)
    return parse_dataclass(response_type, response)


def poll_get_and_expect_success(
    client: TestClient,
    path: str,
    params: dict[str, str],
    response_type: type[ResponseType],
    response_token_attr: str = 'response_token',
    sleep_time: float = 5,
    max_time: Optional[int] = None,
    headers: Mapping[str, str] = {},
    skip_cache: bool = True,
) -> ResponseType:
    """
    Polls an API endpoint that uses a token-based long-running operation pattern.

    Many APIs respond with a token indicating that the operation is still in progress or that
    more data is available. This function repeatedly calls the endpoint until the token becomes
    empty (''), indicating completion of the operation or retrieval of all data.

    Args:
        client (TestClient): The client for sending the HTTP request
        path (str): The HTTP endpoint to call.
        params (Dict[str, str]): A dictionary of URL parameters to send with the request
        response_type (Type[ResponseType]): The Pydantic model type to parse the response into
        response_token_attr (str, optional): The attribute name of the response token. Defaults to 'response_token'.
        sleep_time (float, optional): The time in seconds to sleep between poll attempts. Defaults to 5.
        max_time (int, optional): The maximum time in seconds to poll the endpoint; only enforced if set. Defaults to None.
        headers (Mapping[str, str], optional): The HTTP headers to send with the request. Defaults to {}.
        skip_cache (bool, optional): Whether to bypass any caching mechanisms. Defaults to True.

    Returns:
        ResponseType: The final parsed response after the token becomes empty,
                     indicating the operation is complete or all data has been retrieved.

    Raises:
        AssertionError: If the response does not contain the specified token attribute

    """
    response = get_and_expect_success(client, path, params, response_type,
                                      headers, skip_cache)
    response_token = getattr(response, response_token_attr, None)
    assert response_token is not None, f"Response does not have a response token {response_token_attr}"
    params['response_token'] = response_token
    start_time = time.time()
    while response_token != '' and (max_time is None
                                    or time.time() - start_time < max_time):
        time.sleep(sleep_time)
        response = get_and_expect_success(client, path, params, response_type,
                                          headers, skip_cache)
        response_token = getattr(response, response_token_attr)
    return response


def poll_get_and_expect_error(client: TestClient, path: str, params: dict[str,
                                                                          str],
                              response_type: type[ResponseType],
                              status_code: HTTPStatus, error_pattern: str,
                              response_token_attr: str = 'response_token',
                              sleep_time: int = 5, headers: Mapping[str,
                                                                    str] = {},
                              skip_cache: bool = True):
    """
    Polls an API endpoint that uses a token-based long-running operation pattern,
    expecting a specific error to occur.

    This function is similar to poll_get_and_expect_success but is designed for testing error cases.
    It repeatedly calls the endpoint until the token becomes empty (''), and then verifies that
    the expected error status code and error message pattern are present in the response.

    Args:
        client (TestClient): The client for sending the HTTP request
        path (str): The HTTP endpoint to call
        params (Dict[str, str]): A dictionary of URL parameters to send with the request
        response_type (Type[ResponseType]): The Pydantic model type to parse the response into
        status_code (HTTPStatus): The expected HTTP error status code (e.g., HTTPStatus.BAD_REQUEST)
        error_pattern (str): A regex pattern that should match the expected error message
        response_token_attr (str, optional): The attribute name of the response token. Defaults to 'response_token'
        sleep_time (int, optional): The time in seconds to sleep between poll attempts. Defaults to 5
        headers (Mapping[str, str], optional): The HTTP headers to send with the request. Defaults to {}
        skip_cache (bool, optional): Whether to bypass any caching mechanisms. Defaults to True

    Raises:
        AssertionError: If the response does not contain the specified token attribute
        AssertionError: If the final response status code doesn't match the expected status_code
        AssertionError: If the error message doesn't match the expected error_pattern
    """
    if skip_cache:
        params['skip_cache'] = 'true'
    response_raw = client.get(path, params=params, headers=headers,
                              follow_redirects=True)
    response = parse_dataclass(response_type, response_raw)
    response_token = getattr(response, response_token_attr, None)
    assert response_token is not None, f"Response does not have a response token {response_token_attr}"
    params['response_token'] = response_token
    while response_token != '':
        time.sleep(sleep_time)
        response_raw = client.get(path, params=params, headers=headers,
                                  follow_redirects=True)
        response = parse_dataclass(response_type, response_raw)
        response_token = getattr(response, response_token_attr)
    _check_status_code(response_raw, status_code)
    _check_error_pattern(error_pattern, response_raw)


def patch_and_expect_success(
    client: TestClient,
    path: str,
    params: dict[str, str],
    headers: Mapping[str, str] = {},
) -> bool:
    """Send a PATCH request and parse the response into a Pydantic dataclass.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        params (Dict[str, str]): a dict of url parameters to send

    Returns:
        bool: whether the resource was modified or not
    """
    response = client.patch(path, params=params, headers=headers,
                            follow_redirects=True)
    _check_status_code(response, HTTPStatus.OK)
    return parse_dataclass(PatchResponse, response).resource_updated


def post_and_expect_success(client: TestClient, path: str, payload: Any,
                            response_type: type[ResponseType],
                            headers: Mapping[str, str] = {},
                            skip_cache: bool = False,
                            expected_status_code: HTTPStatus = HTTPStatus.OK,
                            **params) -> ResponseType:
    """Send a post request and parse the response into a Pydantic dataclass.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        payload (dataclass): the payload to include in the request
        response_type (Type[ResponseType]): refer to parse_dataclass()

    Returns:
        ResponseType: the parsed response
    """
    request_params = dict(params)
    if skip_cache:
        request_params['skip_cache'] = 'true'
    response = client.post(
        path,
        data=_dump_dataclass(payload),
        headers=_combine_headers(headers, _JSON_HEADERS),
        params=request_params,
        follow_redirects=True,
    )
    _check_status_code(response, expected_status_code)
    return parse_dataclass(response_type, response)


def put_and_expect_success(
    client: TestClient,
    path: str,
    payload: Any,
    response_type: type[ResponseType],
    headers: Mapping[str, str] = {},
) -> ResponseType:
    """Send a put request and parse the response into a Pydantic dataclass.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        payload (dataclass): the payload to include in the request
        response_type (Type[ResponseType]): refer to parse_dataclass()

    Returns:
        ResponseType: the parsed response
    """
    response = client.put(path, data=_dump_dataclass(payload),
                          headers=_combine_headers(headers, _JSON_HEADERS),
                          follow_redirects=True)
    _check_status_code(response, HTTPStatus.OK)
    return parse_dataclass(response_type, response)


def delete_and_expect_success(
    client: TestClient,
    path: str,
    headers: Mapping[str, str] = {},
):
    """Send a delete request and check for a successful status code.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
    """
    response = client.delete(path, headers=headers, follow_redirects=True)
    _check_status_code(response, HTTPStatus.OK)


def get_and_expect_error(
    client: TestClient,
    path: str,
    params: dict[str, str],
    status_code: HTTPStatus,
    error_pattern: str,
    headers: Mapping[str, str] = {},
    skip_cache: bool = True,
):
    """Send a get request and check that the error matches the given pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        params (Dict[str, str]): a dict of url parameters to send
        error_pattern (str): a regex that is expected to match the error
            message
    """
    if skip_cache:
        params['skip_cache'] = 'true'
    response = client.get(path, params=params, headers=headers,
                          follow_redirects=True)
    _check_status_code(response, status_code)
    _check_error_pattern(error_pattern, response)


def post_and_expect_error(
    client: TestClient,
    path: str,
    payload: Any,
    status_code: HTTPStatus,
    error_pattern: str,
    headers: Mapping[str, str] = {},
):
    """Send a post request and check that the error matches the given pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        payload (dataclass): the payload to include in the request
        error_pattern (str): a regex that is expected to match the error
            message
    """
    response = client.post(path, data=_dump_dataclass(payload),
                           headers=_combine_headers(headers, _JSON_HEADERS),
                           follow_redirects=True)
    _check_status_code(response, status_code)
    _check_error_pattern(error_pattern, response)


def put_and_expect_error(
    client: TestClient,
    path: str,
    payload: Any,
    status_code: HTTPStatus,
    error_pattern: str,
    headers: Mapping[str, str] = {},
):
    """Send a put request and check that the error matches the given pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        payload (dataclass): the payload to include in the request
        error_pattern (str): a regex that is expected to match the error
            message
    """
    response = client.put(path, data=_dump_dataclass(payload),
                          headers=_combine_headers(headers, _JSON_HEADERS),
                          follow_redirects=True)
    _check_status_code(response, status_code)
    _check_error_pattern(error_pattern, response)


def patch_and_expect_error(
    client: TestClient,
    path: str,
    params: dict[str, str],
    status_code: HTTPStatus,
    error_pattern: str,
    headers: Mapping[str, str] = {},
):
    """Send a PATCH request and check that the error matches the given pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        params (Dict[str, str]): a dict of url parameters to send
        error_pattern (str): a regex that is expected to match the error
            message
    """
    response = client.patch(path, params=params, headers=headers,
                            follow_redirects=True)
    _check_status_code(response, status_code)
    _check_error_pattern(error_pattern, response)


def delete_and_expect_error(
    client: TestClient,
    path: str,
    status_code: HTTPStatus,
    error_pattern: str,
    headers: Mapping[str, str] = {},
):
    """Send a delete request and check that the error matches the given
    pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        error_pattern (str): a regex that is expected to match the error
            message
    """
    response = client.delete(path, headers=headers, follow_redirects=True)
    _check_status_code(response, status_code)
    _check_error_pattern(error_pattern, response)


def post_and_expect_field_matches(
    client: TestClient,
    path: str,
    payload: Any,
    response_type: type[ResponseType],
    field_key: str,
    error_pattern: str,
    headers: Mapping[str, str] = {},
) -> ResponseType:
    """Send a post request and check that the warning matches the given
    pattern.

    Args:
        client (TestClient): the client for sending the HTTP request
        path (str): the HTTP endpoint to call
        payload (dataclass): the payload to include in the request
        error_pattern (str): a regex that is expected to match the error
            message
    """
    response = client.post(path, data=_dump_dataclass(payload),
                           headers=_combine_headers(headers, _JSON_HEADERS),
                           follow_redirects=True)
    _check_status_code(response, HTTPStatus.OK)
    response = parse_dataclass(response_type, response)

    assert response is not None
    _check_field_pattern(error_pattern, str(response.__dict__.get(field_key)))

    return response


def _check_status_code(response: Response, expected_status_code: HTTPStatus):
    import pytest
    if response.status_code != expected_status_code:
        pytest.fail(
            f"Expected HTTP status code = {expected_status_code}, but got "
            f"{response.status_code}. Response headers: {response.headers} "
            f"Response body: '{response.text[:_RESPONSE_TEXT_DISPLAY_LIMIT]}'")


def _check_error_pattern(error_pattern: str, response: Response):
    import pytest
    if not re.search(error_pattern, response.text):
        pytest.fail(f"Expected error message matching '{error_pattern}' but "
                    f"got: {response.text[:_RESPONSE_TEXT_DISPLAY_LIMIT]}")


def _check_field_pattern(pattern: str, text: str):
    import pytest
    if not re.search(pattern, text):
        pytest.fail(f"Expected field value matching '{pattern}' but "
                    f"got: {text[:_RESPONSE_TEXT_DISPLAY_LIMIT]}")


def _dump_dataclass(payload: Any) -> str:
    return json.dumps(
        payload, default=partial(trusted_custom_pydantic_encoder,
                                 insecure=True))


def configure_test_client_token(
    app: FastAPI,
    test_token: RequestsResponse,
    call_with_access_token: bool = True,
) -> None:
    del app, test_token, call_with_access_token


def assert_has_call_to_action(call_to_actions: list[CallToAction],
                              message: str):
    assert any(message in cta.message for cta in call_to_actions), (
        f"No call to action containing '{message}' was "
        f"found. Got: {call_to_actions}")


def validate_name_test_helper(client: TestClient, name: str, endpoint: str,
                              should_succeed: bool):
    response = get_and_expect_success(client, endpoint, {"name": name},
                                      ValidateNameResponse)
    if not should_succeed:
        assert response.is_valid is False
        assert response.error_msg is not None
        assert len(response.error_msg) > 0
    else:
        assert response.is_valid is True
        assert response.error_msg is None


def _remap_fixture_ids_v2(
    dump: FixtureDefinition,
    id_map: Mapping[str, str],
) -> FixtureDefinition:
    """Return a fixture with IDs and dependent graph fingerprints remapped."""
    json_dump = to_json(dump)
    for old_id, new_id in id_map.items():
        json_dump = json_dump.replace(str(old_id), str(new_id))
    remapped = from_json(json_dump, FixtureDefinition)

    old_to_new_graph_fpid: dict[FPID, FPID] = {
        old_g.fpid: new_g.fpid
        for old_g, new_g in zip(dump.graphs, remapped.graphs)
    }
    queries_remapped = [
        replace(
            q,
            graph_fpid=old_to_new_graph_fpid.get(q.graph_fpid, q.graph_fpid),
        ) for q in remapped.queries
    ]
    return replace(remapped, queries=queries_remapped)


def remap_fixture_connector_ids_v2(
    dump: FixtureDefinition,
    connector_id_map: Mapping[str, str],
) -> FixtureDefinition:
    """Remap fixture connector references to IDs returned by connector create.

    Project connector IDs are server-generated. Warehouse integration fixtures
    use a temporary canonical ID before connector creation, then must replace it
    with the created resource ID everywhere, including graph fingerprints and
    predictive-query graph references.
    """
    canonical_map = {
        str(old_id): str(ConnectorID(str(new_id)))
        for old_id, new_id in connector_id_map.items()
    }
    return _remap_fixture_ids_v2(dump, canonical_map)


def randomize_fixture_ids_v2(dump: FixtureDefinition) -> FixtureDefinition:
    """
    Randomize connector references and predictive-query names consistently.

    V2 fixtures deliberately do not contain connector definitions: identities,
    credentials, and connector lifecycle are environment-owned. Connector IDs
    are therefore discovered from the graph's table definitions and replaced
    everywhere in the credential-free fixture payload.
    """
    id_map = {}

    def raw_connector_id(table) -> str:
        """Read a fixture connector reference before enforcing its ID shape.

        Historical JSON fixtures use human-readable connector aliases.  Those
        aliases are import-only data: randomization must replace them with a
        durable project connector ID before the fixture is parsed or used by
        any runtime connector boundary.
        """
        source_table = table.source_table
        for field_name in (
                'snowflake_connector_id',
                'databricks_connector_id',
                'bigquery_connector_id',
                'connector_id',
        ):
            value = getattr(source_table, field_name, None)
            if value is not None:
                return str(value)
        raise ValueError(
            f'Fixture table has no connector reference: {source_table!r}')

    connector_ids = {
        raw_connector_id(table)
        for graph in dump.graphs
        for table in graph.tables.values()
    }
    for old_id in connector_ids:
        new_id = ConnectorID(f'connector_{uuid4().hex}')
        id_map[old_id] = new_id

    for pq in dump.queries:
        old_id = pq.pquery_name
        new_id = PredictiveQueryID(old_id + str(hash(uuid4()))[:5])
        id_map[old_id] = new_id

    return _remap_fixture_ids_v2(dump, id_map)


def create_identity_binding(
    client: TestClient,
    *,
    identity_config: data_source_api.IdentityConfig,
    credentials: Optional[data_source_api.IdentityCredentials],
) -> IdentityBindingSetup:
    """Create an admin-owned identity and bind it to a native test group."""
    suffix = uuid4().hex
    group = post_and_expect_success(
        client,
        '/v0/groups',
        CreateKumoGroupRequest(display_name=f'test-group-{suffix}'),
        KumoGroupDBEntry,
        expected_status_code=HTTPStatus.CREATED,
    )
    post_and_expect_success(
        client,
        f'/v0/groups/{group.id}/members',
        AddGroupMemberRequest(principal_id=TEST_REST_ADMIN_USER_ID),
        GroupMemberEntry,
        expected_status_code=HTTPStatus.CREATED,
    )
    identities = get_and_expect_success(
        client,
        '/v0/identities',
        {},
        list[data_source_api.IdentityResponse],
    )
    identity = next(
        (candidate
         for candidate in identities if candidate.config == identity_config),
        None,
    )
    if identity is None:
        identity = post_and_expect_success(
            client,
            '/v0/identities',
            data_source_api.CreateIdentityArgs(
                display_name=f'test-identity-{suffix}',
                config=identity_config,
                credentials=credentials,
            ),
            data_source_api.IdentityResponse,
        )
    mapped_groups = get_and_expect_success(
        client,
        f'/v0/identities/{identity.id}/groups',
        {},
        data_source_api.IdentityGroupsResponse,
    )
    put_and_expect_success(
        client,
        f'/v0/identities/{identity.id}/groups',
        data_source_api.ReplaceIdentityGroupsArgs(group_ids=[
            *[str(group_id) for group_id in mapped_groups.group_ids],
            str(group.id),
        ]),
        data_source_api.IdentityGroupsResponse,
    )
    return IdentityBindingSetup(
        identity_id=identity.id,
        group_id=str(group.id),
    )


def create_identity_backed_connector(
    client: TestClient,
    *,
    identity_config: data_source_api.IdentityConfig,
    credentials: Optional[data_source_api.IdentityCredentials],
    connector_config,
) -> IdentityBackedConnectorSetup:
    """Create an identity binding followed by a project-owned target."""
    binding = create_identity_binding(
        client,
        identity_config=identity_config,
        credentials=credentials,
    )
    project_id = create_project_for_group(client, binding.group_id)
    connector_name = connector_config.name
    connector = post_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/',
        data_source_api.CreateConnectorArgs(
            identity_id=binding.identity_id,
            name=connector_name,
            config=_api_external_config(connector_config),
        ),
        data_source_api.ConnectorResponse,
    )
    return IdentityBackedConnectorSetup(
        connector_id=str(connector.id),
        identity_id=binding.identity_id,
        group_id=binding.group_id,
        project_id=project_id,
    )


def create_project_for_group(client: TestClient, group_id: str) -> str:
    """Create a non-private project for an identity-binding test group."""
    suffix = uuid4().hex
    response = client.post(
        f'/v0/groups/{group_id}/projects',
        json={
            'name': f'test-project-{suffix}',
            'description': 'project connector test project',
            'is_private': False,
        },
    )
    _check_status_code(response, HTTPStatus.CREATED)
    return str(response.json()['id'])


def fake_materialization_info(
    training_job_id: TrainingJobID | None,
    batch_prediction_job_id: BatchPredictionJobID | None,
    fake_table_stats: list[MaterializedTableInfo] | None = None,
) -> MaterializationTaskInfo:
    return MaterializationTaskInfo(
        table_stats=fake_table_stats or [],
        code_version="1.2.3",
        start_time=datetime.now(),
        end_time=datetime.now(),
        job_id=batch_prediction_job_id or training_job_id,
        graph_id=GraphIDv1("fake_graph"),
        predictive_query_id=PredictiveQueryID("fake_pq"),
        temporal_workflow_id="temporal_id",
    )


def financial_snowflake_config(
        connector_name: str) -> SnowflakeConnectorResourceConfig:
    return SnowflakeConnectorResourceConfig(name=connector_name,
                                            warehouse='WH_XS', database='KUMO',
                                            schema_name='FINANCIAL_PYTEST')


def writable_snowflake_config(
        connector_name: str) -> SnowflakeConnectorResourceConfig:
    return SnowflakeConnectorResourceConfig(
        name=connector_name, warehouse='WH_XS', database='KUMO',
        schema_name='TEST_SNOWFLAKE_WRITE_PANDAS')


def writable_bigquery_connector(
        connector_name: str) -> BigQueryConnectorResourceConfig:
    return BigQueryConnectorResourceConfig(name=connector_name,
                                           project_id='kumo-engineering',
                                           dataset_id='test', location='US')


def hm_databricks_config(
        connector_name: str) -> DatabricksConnectorResourceConfig:
    return DatabricksConnectorResourceConfig(
        name=connector_name,
        warehouse_id='82d859e4a6b283a0',
        catalog='hive_mapped_uc',
        schema_name='default',
    )


def hm_bigquery_config(connector_name: str) -> BigQueryConnectorResourceConfig:
    r"""Connector config for the hnm_integ_test BigQuery dataset.

    Points at ``kumo-engineering.hnm_integ_test``, which carries the
    permanent ``customers`` / ``articles`` / ``transactions`` tables
    used by the BigQuery ingestion e2e tests. Analog of
    ``hm_databricks_config``.
    """
    return BigQueryConnectorResourceConfig(name=connector_name,
                                           project_id='kumo-engineering',
                                           dataset_id='hnm_integ_test',
                                           location='US')


def fake_admin_request_context() -> RequestContext:
    """Returns an administrator request context."""
    return RequestContext(user_authorization=UserAuthorization(
        user_id=UserID('user@kumo.ai'),
        access_token="",
        permissions=[
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
            Permissions.ADMIN.compose()
        ],
    ))
