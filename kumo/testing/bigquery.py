# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum

import pytest

from io_utils import fsutil
from kumo.connector.bigquery_connector import (
    ENV_GCS_KUMO_BUCKET,
    ENV_GOOGLE_CREDENTIALS_JSON,
)
from service_framework.env_var import EnvVar
from tools.ci_test_profiles import TestProfile, profile_for

ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT = 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'
ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT = 'GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT'
ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT = 'GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT'
ENV_BIGQUERY_NO_TABLE_CREDENTIALS = 'GCP_BIGQUERY_NO_TABLE_CREDENTIALS'
ENV_BIGQUERY_NO_JOB_CREDENTIALS = 'GCP_BIGQUERY_NO_JOB_CREDENTIALS'

_CREDENTIAL_ENV_VARS = (
    'GOOGLE_APPLICATION_CREDENTIALS',
    ENV_GOOGLE_CREDENTIALS_JSON,
    ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
    ENV_GCS_KUMO_BUCKET.name,
    'GCP_PROJECT_ID',
)


class BigQueryIdentity(Enum):
    DEFAULT = 'default'
    NO_TABLE = 'no_table'
    NO_JOB = 'no_job'


class CredentialKind(Enum):
    FEDERATED = 'external_account'
    USER = 'authorized_user'
    IMPERSONATED = 'impersonated_service_account'
    KEY = 'service_account'


def configured_profile() -> TestProfile | None:
    name = os.environ.get('KUMO_TEST_PROFILE')
    return profile_for(name) if name else None


def credential_kind(path: str) -> CredentialKind:
    if not fsutil.exists(path) or fsutil.isdir(path):
        raise RuntimeError('Configured BigQuery credential file does not exist')
    try:
        document = json.loads(fsutil.read_text(path))
        if not isinstance(document, dict):
            raise ValueError('credential document must be a JSON object')
        return CredentialKind(document.get('type'))
    except (ValueError, TypeError) as error:
        raise RuntimeError('BigQuery credential file has an invalid or '
                           'unsupported credential type') from error


def resolve_credentials(identity: BigQueryIdentity) -> tuple[str, str | None]:
    """Resolve explicit inputs without changing the environment or files."""
    if not isinstance(identity, BigQueryIdentity):
        raise TypeError('BigQuery credentials require a BigQueryIdentity')
    path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not path:
        raise RuntimeError('BigQuery tests require explicit '
                           'GOOGLE_APPLICATION_CREDENTIALS')
    kind = credential_kind(path)
    profile = configured_profile()
    require_wif = (profile is not None and 'gcp' in profile.capabilities) or EnvVar(
        'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', bool, default_value=False).parse_value()
    if require_wif and kind != CredentialKind.FEDERATED:
        raise RuntimeError('Workload exchange BigQuery tests require '
                           'external_account ADC')
    if identity == BigQueryIdentity.DEFAULT:
        return path, None

    if kind == CredentialKind.KEY:
        variable = {
            BigQueryIdentity.NO_TABLE: ENV_BIGQUERY_NO_TABLE_CREDENTIALS,
            BigQueryIdentity.NO_JOB: ENV_BIGQUERY_NO_JOB_CREDENTIALS,
        }[identity]
        restricted_path = os.environ.get(variable)
        if not restricted_path:
            raise RuntimeError(f'{variable} must identify the restricted '
                               'BigQuery service-account credential file')
        if credential_kind(restricted_path) != CredentialKind.KEY:
            raise RuntimeError(f'{variable} must contain service_account '
                               'credentials')
        return restricted_path, None

    variable = {
        BigQueryIdentity.NO_TABLE: ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
        BigQueryIdentity.NO_JOB: ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
    }[identity]
    target = os.environ.get(variable)
    if not target:
        raise RuntimeError(f'{variable} must be configured for '
                           'the restricted BigQuery test identity')
    return path, target


@contextmanager
def setup_bigquery_credentials(
    identity: BigQueryIdentity = BigQueryIdentity.DEFAULT,
) -> Generator[None, None, None]:
    """Scope explicit ADC to a test or session; input files remain read-only."""
    path, target = resolve_credentials(identity)
    with pytest.MonkeyPatch.context() as environment:
        for name in _CREDENTIAL_ENV_VARS:
            previous = os.environ.get(name)
            environment.setenv(name, previous or '')
            if previous is None:
                environment.delenv(name)

        environment.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
        environment.setenv(ENV_GOOGLE_CREDENTIALS_JSON, path)
        if target:
            environment.setenv(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT, target)
        else:
            environment.delenv(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                               raising=False)
        environment.setenv(
            ENV_GCS_KUMO_BUCKET.name,
            os.environ.get(ENV_GCS_KUMO_BUCKET.name) or 'kumo-test-bucket')
        project = (os.environ.get('GCP_PROJECT_ID')
                   or os.environ.get('GOOGLE_CLOUD_PROJECT'))
        if project:
            environment.setenv('GCP_PROJECT_ID', project)
        yield


@pytest.fixture(scope='session', autouse=True)
def configure_profile_credentials(request: pytest.FixtureRequest):
    """Prepare ADC before session resources and worker processes start."""
    profile = configured_profile()
    selected = any(item.get_closest_marker('bigquery_credentials') is not None
                   and item.get_closest_marker('skip') is None
                   for item in request.session.items)
    if not selected and (profile is None or 'gcp' not in profile.capabilities):
        yield
        return
    with setup_bigquery_credentials():
        yield


@pytest.fixture(autouse=True)
def configure_bigquery_credentials(
    request: pytest.FixtureRequest,
    configure_profile_credentials,
):
    marker = request.node.get_closest_marker('bigquery_credentials')
    if marker is None:
        yield
        return
    identity, = marker.args
    if not isinstance(identity, BigQueryIdentity):
        raise TypeError('bigquery_credentials requires a BigQueryIdentity')
    with setup_bigquery_credentials(identity):
        yield


def withBigQueryCredentials(
    func: Callable | None = None,
    *,
    identity: BigQueryIdentity = BigQueryIdentity.DEFAULT,
):
    """Declare credentials without performing auth or wrapping the test body."""
    def decorator(inner_func: Callable):
        return pytest.mark.requires_credentials('gcp')(
            pytest.mark.bigquery_credentials(identity)(inner_func))

    return decorator if func is None else decorator(func)
