# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from typing import Optional

import pytest

from io_utils import fsutil
from io_utils.secrets import get_secrets
from kumo.connector.bigquery_connector import (
    ENV_GCS_KUMO_BUCKET,
    ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH,
    ENV_GOOGLE_CREDENTIALS_JSON,
)
from tools.ci_test_profiles import TestProfile, profile_for

DEFAULT_BIGQUERY_KEY = 'kumo-test-bigquery-service-account-key'
FAIL_TABLE_PERMISSION_BIGQUERY_KEY = 'kumo-test-permission-fail-bigquery-service-account-key-2'
FAIL_PROJECT_PERMISSION_BIGQUERY_KEY = 'kumo-test-permission-fail-bigquery-service-account-key'

ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT = 'GOOGLE_IMPERSONATE_SERVICE_ACCOUNT'
ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT = 'GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT'
ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT = 'GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT'

_IMPERSONATION_ENV_BY_KEY = {
    FAIL_TABLE_PERMISSION_BIGQUERY_KEY: ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
    FAIL_PROJECT_PERMISSION_BIGQUERY_KEY: ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
}
_CREDENTIAL_ENV_VARS = (
    'GOOGLE_APPLICATION_CREDENTIALS',
    ENV_GOOGLE_CREDENTIALS_JSON,
    ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
    ENV_GCS_KUMO_BUCKET.name,
    'GCP_PROJECT_ID',
)


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


@contextmanager
def setup_bigquery_credentials(
    service_account_key: Optional[str] = None,
) -> Generator[None, None, None]:
    """Scope test credentials, including transitional stored-key fallback.

    An ADC file's type determines whether restricted tests use impersonation.
    A service-account key instead needs the existing per-identity secret. The
    fallback remains until hosted workflows and local setup migrate together;
    it never overwrites input files or writes shared home/repository files.
    """
    key = service_account_key or DEFAULT_BIGQUERY_KEY
    with pytest.MonkeyPatch.context() as environment:
        # Register all touched variables even when absent, including values a
        # decorated test may overwrite before raising or skipping.
        for name in _CREDENTIAL_ENV_VARS:
            previous = os.environ.get(name)
            environment.setenv(name, previous or '')
            if previous is None:
                environment.delenv(name)

        profile = configured_profile()
        profiled_gcp = profile is not None and 'gcp' in profile.capabilities
        path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if profiled_gcp and not path:
            raise RuntimeError('GCP test profiles require explicit '
                               'GOOGLE_APPLICATION_CREDENTIALS')
        if not profiled_gcp:
            path = path or os.environ.get(ENV_GOOGLE_CREDENTIALS_JSON)
        if (not profiled_gcp and not path
                and fsutil.exists(ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH)):
            path = ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH
        kind = credential_kind(path) if path else None
        if profiled_gcp and kind != CredentialKind.FEDERATED:
            raise RuntimeError('GCP test profiles require external_account ADC')
        target_variable = _IMPERSONATION_ENV_BY_KEY.get(key)
        target = None
        if kind is not None and kind != CredentialKind.KEY and target_variable:
            target = os.environ.get(target_variable)
            if not target:
                raise RuntimeError(f'{target_variable} must be configured for '
                                   'the restricted BigQuery test identity')

        temporary_dir = None
        try:
            if path is None or (kind == CredentialKind.KEY and target_variable):
                try:
                    credentials = get_secrets(key)
                except Exception:
                    pytest.skip('No BigQuery credentials found')
                temporary_dir = fsutil.mkdtemp()
                path = os.path.join(temporary_dir, 'credentials.json')
                fsutil.write_text(path, json.dumps(credentials))
                if credential_kind(path) != CredentialKind.KEY:
                    raise RuntimeError('Stored BigQuery test credentials must '
                                       'be service_account credentials')

            environment.setenv('GOOGLE_APPLICATION_CREDENTIALS', path)
            environment.setenv(ENV_GOOGLE_CREDENTIALS_JSON, path)
            if target:
                environment.setenv(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                                   target)
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
        finally:
            if temporary_dir is not None:
                fsutil.rm(temporary_dir)


@pytest.fixture(scope='session', autouse=True)
def configure_profile_credentials():
    """Prepare ADC before session resources and worker processes start."""
    profile = configured_profile()
    if profile is None or 'gcp' not in profile.capabilities:
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
    if configured_profile() is None or marker is None:
        yield
        return
    key, = marker.args
    if key is not None and not isinstance(key, str):
        raise TypeError('bigquery_credentials requires a string identity key')
    with setup_bigquery_credentials(key):
        yield


def withBigQueryCredentials(func: Optional[Callable] = None,
                            service_account_key: Optional[str] = None):
    def decorator(inner_func: Callable):
        marked = pytest.mark.requires_credentials('gcp')(
            pytest.mark.bigquery_credentials(service_account_key)(inner_func))
        if configured_profile() is not None:
            return marked
        # Temporary unprofiled path: existing workflows retain phase-one
        # behavior until all invocations use the profile launcher.
        if asyncio.iscoroutinefunction(inner_func):

            async def async_wrapper(*args, **kwargs):
                with setup_bigquery_credentials(service_account_key):
                    return await inner_func(*args, **kwargs)

            wrapper = async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with setup_bigquery_credentials(service_account_key):
                    return inner_func(*args, **kwargs)

            wrapper = sync_wrapper
        return functools.wraps(marked)(wrapper)

    return decorator if func is None else decorator(func)
