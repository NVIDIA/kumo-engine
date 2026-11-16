# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity-backed GCS connector helpers for integration tests."""

from __future__ import annotations

import json
import os

import kumoapi.data_source as data_source_api
from fastapi.testclient import TestClient
from kumoapi.source_table import SourceTableConfigResponse, SourceTableType
from pydantic import SecretStr

from kumo.testing.rest import (
    IdentityBackedConnectorSetup,
    create_identity_backed_connector,
    get_and_expect_success,
    post_and_expect_success,
)
from rest.config.connector import GCSConnectorResourceConfig


def load_gcs_test_identity() -> tuple[
    data_source_api.GCSIdentityConfig,
    data_source_api.IdentityCredentials | None,
]:
    """Load CI's workload identity, with key credentials as a local fallback.

    GitHub Actions uses the durable workload-identity configuration. Temporal
    activities re-resolve it through the deployment's federated ADC source;
    no private key is stored in the identity model.
    """
    workload_service_account = os.getenv('GCP_TEST_SERVICE_ACCOUNT')
    if workload_service_account:
        return (
            data_source_api.GCSIdentityConfig(
                authentication=(
                    data_source_api.GoogleWorkloadIdentityAuthentication(
                        service_account_email=workload_service_account, )), ),
            None,
        )

    with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
              encoding='utf-8') as credential_file:
        document = json.load(credential_file)
    if document.get('type') != 'service_account':
        raise ValueError(
            'GCS integration credentials must be a service account')

    config = data_source_api.GCSIdentityConfig(
        authentication=data_source_api.GoogleServiceAccountKeyAuthentication(
            project_id=document['project_id'],
            client_email=document['client_email'],
            client_id=document['client_id'],
            auth_uri=document['auth_uri'],
            token_uri=document['token_uri'],
            universe_domain=document.get('universe_domain', 'googleapis.com'),
        ), )
    credentials = data_source_api.GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr(document['private_key_id']),
        private_key=SecretStr(document['private_key']),
    )
    return config, credentials


def _target(root_dir: str) -> tuple[str, str]:
    if not root_dir.startswith('gs://'):
        raise ValueError('GCS integration target must use gs://')
    bucket, separator, prefix = root_dir[5:].partition('/')
    if not bucket:
        raise ValueError('GCS integration target requires a bucket')
    return bucket, prefix.strip('/') if separator else ''


def create_identity_backed_gcs_connector(
    client: TestClient,
    *,
    name: str,
    root_dir: str,
    identity_config: data_source_api.GCSIdentityConfig,
    identity_credentials: data_source_api.IdentityCredentials | None,
) -> IdentityBackedConnectorSetup:
    """Create the first GCS connector together with its identity and project."""

    bucket, prefix = _target(root_dir)
    return create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=identity_credentials,
        connector_config=GCSConnectorResourceConfig(
            name=name,
            bucket=bucket,
            prefix=prefix,
        ),
    )


def create_gcs_connector_in_project(
    client: TestClient,
    *,
    name: str,
    root_dir: str,
    project_id: str,
    identity_id: str,
) -> str:
    """Create another logical GCS target using the project's bound identity."""

    bucket, prefix = _target(root_dir)
    connector = post_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/',
        data_source_api.CreateConnectorArgs(
            identity_id=identity_id,
            name=name,
            config=data_source_api.GCSConnectorResourceConfig(
                bucket=bucket,
                prefix=prefix,
            ),
        ),
        data_source_api.ConnectorResponse,
    )
    return str(connector.id)


def get_gcs_source_table(
    client: TestClient,
    *,
    project_id: str,
    connector_id: str,
    source_table_name: str,
) -> SourceTableType:
    """Read the typed source coordinate through the authorized activity."""

    response = get_and_expect_success(
        client,
        (f'/v0/projects/{project_id}/connectors/{connector_id}/'
         'source-table-config'),
        {'table_name': source_table_name},
        SourceTableConfigResponse,
    )
    return response.source_table
