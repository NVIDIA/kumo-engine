# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import json
import os
from collections.abc import Callable
from typing import Optional

from io_utils import fsutil, logging
from io_utils.secrets import get_secrets
from kumo.connector.bigquery_connector import (
    ENV_GCS_KUMO_BUCKET,
    ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH,
    ENV_GOOGLE_CREDENTIALS_JSON,
)
from service_framework.env_var import EnvVar

logger = logging.getLogger(__name__)

DEFAULT_BIGQUERY_KEY = "kumo-test-bigquery-service-account-key"

# Missing required table read permission
FAIL_TABLE_PERMISSION_BIGQUERY_KEY = "kumo-test-permission-fail-bigquery-service-account-key-2"
# Missing required jobs.create permission
FAIL_PROJECT_PERMISSION_BIGQUERY_KEY = "kumo-test-permission-fail-bigquery-service-account-key"

# Keyless CI maps the two legacy negative-test identities to deliberately
# restricted service accounts.  The names remain keyed by the old secret
# identifiers so callers do not need to know how credentials are sourced.
ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT = "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT"
ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT = (
    "GCP_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT")
ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT = "GCP_BIGQUERY_NO_JOB_SERVICE_ACCOUNT"

_IMPERSONATION_ENV_BY_KEY = {
    FAIL_TABLE_PERMISSION_BIGQUERY_KEY: ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
    FAIL_PROJECT_PERMISSION_BIGQUERY_KEY: ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT,
}


def _configure_application_default_credentials(
        service_account_key: str) -> bool:
    """Configure an existing ADC file for BigQuery connector tests.

    GitHub's OIDC authentication action writes an ``external_account`` ADC
    file and exports ``GOOGLE_APPLICATION_CREDENTIALS``.  Negative permission
    tests select a restricted service account through IAM impersonation rather
    than loading a second private key.
    """
    # GCP test jobs enable workload exchange explicitly, independently of the
    # runner. Reuse that configuration to require federated test credentials.
    require_wif = EnvVar('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', bool,
                         default_value=False).parse_value()
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path or not fsutil.exists(credentials_path):
        if require_wif:
            raise RuntimeError(
                'Workload exchange BigQuery tests require Workload Identity '
                'Federation ADC configured through '
                'GOOGLE_APPLICATION_CREDENTIALS')
        return False

    if require_wif:
        credentials = json.loads(fsutil.read_text(credentials_path))
        if credentials.get('type') != 'external_account':
            raise RuntimeError(
                'Workload exchange BigQuery tests require external_account ADC; '
                'stored service-account keys and user credentials are not '
                'supported with workload exchange enabled')

    impersonation_env = _IMPERSONATION_ENV_BY_KEY.get(service_account_key)
    if impersonation_env is None:
        os.environ.pop(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT, None)
    else:
        target = os.getenv(impersonation_env)
        if not target:
            raise RuntimeError(
                f"{impersonation_env} must be configured for the restricted "
                "BigQuery permission test identity")
        os.environ[ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] = target

    os.environ[ENV_GOOGLE_CREDENTIALS_JSON] = credentials_path
    project_id = (os.getenv('GCP_PROJECT_ID')
                  or os.getenv('GOOGLE_CLOUD_PROJECT'))
    if project_id:
        os.environ['GCP_PROJECT_ID'] = project_id
    logger.info("Using Application Default Credentials for BigQuery tests")
    return True


def setup_bigquery_credentials(service_account_key: Optional[str] = None):
    # Set env var for GCS kumo bucket for testing.
    os.environ[ENV_GCS_KUMO_BUCKET.name] = 'kumo-test-bucket'
    service_account_key = service_account_key or DEFAULT_BIGQUERY_KEY

    # Prefer short-lived ADC supplied by Workload Identity Federation.  This
    # keeps workload exchange tests from reading stored service-account keys.
    if _configure_application_default_credentials(service_account_key):
        return

    try:
        # Get the GCloud secrets from AWS secrets manager and store it in the
        # default file. Workload exchange requires federated ADC above and
        # never reaches this existing fallback.
        secrets = get_secrets(service_account_key)
        file_path = os.path.join(os.getcwd(), ".google_credential.json")
        with open(file_path, 'w') as f:
            json.dump(secrets, f)

        with open(ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH, 'w') as f:
            # This is a hack. allows big query connector
            # to fall back to default credentials
            json.dump(secrets, f)
        os.environ[ENV_GOOGLE_CREDENTIALS_JSON] = file_path
        # Also set the standard ADC env var so any google-cloud-* client
        # built without explicit credentials (e.g. fsutil's GCS client
        # used by integration tests to verify EXPORT DATA output) picks
        # up the same service account.
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path
    except Exception as e:
        # Use built-in credentials if this env doesn't have access to read the
        # secrets from AWS
        logger.exception(e)


def withBigQueryCredentials(func: Optional[Callable] = None,
                            service_account_key: Optional[str] = None):
    import pytest

    def decorator(inner_func: Callable):
        def _setup():
            setup_bigquery_credentials(service_account_key)
            if (not os.getenv(ENV_GOOGLE_CREDENTIALS_JSON) and
                    not os.path.exists(ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH)):
                pytest.skip("No BigQuery credentials found")

        if asyncio.iscoroutinefunction(inner_func):

            async def async_wrapper(*args, **kwargs):
                previous_target = os.environ.get(
                    ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT)
                try:
                    _setup()
                    return await inner_func(*args, **kwargs)
                finally:
                    if previous_target is None:
                        os.environ.pop(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                                       None)
                    else:
                        os.environ[ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] = (
                            previous_target)

            wrapper = async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                previous_target = os.environ.get(
                    ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT)
                try:
                    _setup()
                    return inner_func(*args, **kwargs)
                finally:
                    if previous_target is None:
                        os.environ.pop(ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT,
                                       None)
                    else:
                        os.environ[ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] = (
                            previous_target)

            wrapper = sync_wrapper

        return functools.wraps(inner_func)(wrapper)

    if func is None:
        return decorator
    else:
        return decorator(func)
