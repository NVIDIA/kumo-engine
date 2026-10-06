# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression tests for kumo-ai/kumo#29794.

The table-metadata validators in ``rest/driver/table_logic.py`` run inside
data-backend single activities, and ``_raise_safe_single_activity_failure``
masks any activity failure whose type is not in
``_SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES`` as the generic "Data backend
operation failed. Please retry or contact Kumo support." The validators
raise the allowlisted ``ValidationException`` for pure user-input errors so
the hand-written actionable message survives the masking wrapper.

These tests drive the production public HTTP application through the real
Temporal single-activity path (local Temporal test server, real activity
worker, local file connector reading the HNM_TINY fixture data); nothing
between the HTTP request and the validator raise site is mocked. They fail
against the pre-fix bare-``ValueError`` raise sites, where the caller saw an
HTTP 500 InternalError (``POST /v1/tables/infer_metadata``) or a generic
``errors[]`` entry (``POST /v1/tables/validate``) instead of the message.
"""

import dataclasses
import json
from functools import partial
from http import HTTPStatus

from fastapi.testclient import TestClient
from kumoapi import table as table_api
from kumoapi.typing import Dtype, Stype

from io_utils.json import trusted_custom_pydantic_encoder
from rest.driver.error_constants import INVALID_PRIMARY_KEY_TITLE
from statedb.config.table import PKEY_PRESENT_IN_COLS

_GENERIC_MASKED_MESSAGE = (
    'Data backend operation failed. Please retry or contact Kumo support.')
_JSON_HEADERS = {'Content-Type': 'application/json'}


def _post(client: TestClient, path: str, payload, project_id: str):
    body = json.dumps(
        payload, default=partial(trusted_custom_pydantic_encoder,
                                 insecure=True))
    return client.post(path, content=body, headers=_JSON_HEADERS,
                       params={'project_id': project_id})


def test_infer_metadata_unknown_column_returns_actionable_422(
    public_api_client: TestClient,
    table_definition_v2: table_api.TableDefinition,
    default_project_id: str,
) -> None:
    """A column missing from the source table is a user error: the caller
    must get the actionable 422, not the masked InternalError 500."""
    request = table_api.TableMetadataRequest(
        cols=[
            table_api.ColumnMetadataRequest(
                name='ghost',
                stype=Stype.numerical,
                dtype=Dtype.float64,
            ),
        ],
        source_table=table_definition_v2.source_table,
        pkey=None,
    )
    response = _post(public_api_client, '/v1/tables/infer_metadata',
                     dataclasses.asdict(request), default_project_id)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
        response.text)
    body = response.json()
    assert body['error_type'] == 'ValidationException'
    assert 'Column ghost does not exist in table transactions' in body['error']
    assert _GENERIC_MASKED_MESSAGE not in response.text


def _validate(client: TestClient, definition: table_api.TableDefinition,
              project_id: str) -> dict:
    response = _post(client, '/v1/tables/validate',
                     {'table_definition': dataclasses.asdict(definition)},
                     project_id)
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def test_validate_surfaces_missing_pkey_message(
    public_api_client: TestClient,
    table_definition_v2: table_api.TableDefinition,
    default_project_id: str,
) -> None:
    definition = dataclasses.replace(table_definition_v2, pkey='missing_pkey')
    body = _validate(public_api_client, definition, default_project_id)

    assert body['errors'], body
    message = body['errors'][0]['message']
    assert message == f'{INVALID_PRIMARY_KEY_TITLE}: {PKEY_PRESENT_IN_COLS}'
    assert _GENERIC_MASKED_MESSAGE not in message


def test_validate_surfaces_dtype_mismatch_message(
    public_api_client: TestClient,
    table_definition_v2: table_api.TableDefinition,
    default_project_id: str,
) -> None:
    definition = dataclasses.replace(
        table_definition_v2,
        cols=[
            table_api.Column(
                name='t_dat',
                stype=Stype.timestamp,
                dtype=Dtype.time,
                timestamp_format=None,
            ),
            table_api.Column(
                name='customer_id',
                stype=Stype.numerical,
                dtype=Dtype.int64,
                timestamp_format=None,
            ),
        ],
    )
    body = _validate(public_api_client, definition, default_project_id)

    assert body['errors'], body
    message = body['errors'][0]['message']
    assert "Data type of column 'customer_id'" in message
    assert 'was selected to be Int64 but should be' in message
    assert _GENERIC_MASKED_MESSAGE not in message
