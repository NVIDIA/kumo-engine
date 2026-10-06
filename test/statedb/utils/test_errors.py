# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider classifier trust-boundary tests."""
import pytest
from azure.core.exceptions import AzureError
from botocore.exceptions import ClientError
from databricks.sql.exc import ServerOperationError
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import RefreshError
from snowflake.connector.errors import ProgrammingError
from temporalio.api.failure.v1 import Failure
from temporalio.converter import (
    DefaultFailureConverter,
    DefaultPayloadConverter,
)
from temporalio.exceptions import ApplicationError

from common_utils.error_diagnostics import (
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from kumo.connector.connector import ConnectorError
from statedb.utils.errors import (
    ConnectorValidationDiagnosticCode,
    ConnectorValidationDiagnosticError,
    ProviderBackendError,
    connector_validation_diagnostic_value,
)

# These imports require the databricks SDK to be installed. Skip the whole
# module cleanly if it isn't (mirrors how the connector tests treat the SDK).
pytest.importorskip("databricks.sdk.errors.platform")

from databricks.sdk.errors.platform import (  # noqa: E402
    NotFound,
    PermissionDenied,
    Unauthenticated,
)

_SAFE_QUERY_ID = '01234567-89ab-cdef-0123-456789abcdef'


class _NestedThirdPartyError(Exception):
    pass


def _temporal_failure(public_error):
    application_error = ApplicationError(
        public_error.message,
        public_error.to_dict(),
        type=type(public_error).__name__,
        non_retryable=public_error.non_retryable,
    )
    failure = Failure()
    DefaultFailureConverter().to_failure(
        application_error, DefaultPayloadConverter(), failure)
    return application_error, failure


@pytest.mark.parametrize(
    ('error', 'lead', 'expected', 'absent'),
    [
        (
            PermissionDenied(''),
            None,
            ('Files API', 'files', 'READ VOLUME', 'WRITE VOLUME'),
            ('PermissionDenied', ),
        ),
        (
            Unauthenticated(''),
            None,
            ('Files API', 'READ VOLUME'),
            (),
        ),
        (
            NotFound(''),
            None,
            ('could not find', 'USE CATALOG', 'USE SCHEMA'),
            ('NotFound', ),
        ),
        (
            PermissionDenied('backend detail'),
            'Kumo could not access the volume.',
            ('Kumo could not access the volume.', 'Files API'),
            ('backend detail', ),
        ),
        (
            PermissionDenied(
                'grant CREATE VOLUME on schema -- misleading raw text'),
            None,
            ('Files API', ),
            ('misleading raw text', ),
        ),
    ],
    ids=['denied', 'unauthenticated', 'not_found', 'lead', 'untrusted_text'],
)
def test_databricks_uc_files_api_classification_contract(
        error, lead, expected, absent):
    """Files API guidance is type-based and never includes provider text."""
    kwargs = {'error': error}
    if lead is not None:
        kwargs['message'] = lead
    message = ProviderBackendError(**kwargs).get_user_message()
    assert all(value in message for value in expected)
    assert all(value not in message for value in absent)


def test_snowflake_classifier_allows_only_safe_driver_fields(caplog):
    canary = 'snowflake-provider-body password=sf-secret-canary'
    provider_error = ProgrammingError(
        msg=canary,
        errno=390111,
        sqlstate='08001',
        sfqid=_SAFE_QUERY_ID,
        query="SELECT 'sf-secret-canary'",
        send_telemetry=False,
    )

    with caplog.at_level('ERROR'):
        error = ProviderBackendError(
            message='Snowflake connector operation failed.',
            error=provider_error,
        )

    rendered = error.get_user_message()
    assert canary not in rendered
    assert 'sf-secret-canary' not in rendered
    assert canary not in caplog.text
    assert 'sf-secret-canary' not in caplog.text
    assert 'code=390111' in rendered
    assert 'sqlstate=08001' in rendered
    assert f'query_id={_SAFE_QUERY_ID}' in rendered
    assert not exception_message_should_be_logged_at_error(provider_error)
    assert safe_exception_message_for_log(provider_error) == (
        'Provider operation failed')

    caplog.clear()
    with caplog.at_level('DEBUG'):
        ProviderBackendError(
            message='Snowflake connector operation failed.',
            error=provider_error,
        )
    assert canary in caplog.text


def test_snowflake_classifier_rejects_poisoned_driver_fields(caplog):
    poisoned_sqlstate = 'sqlstate-secret-canary\nforged-log-line'
    poisoned_query_id = 'query-id-secret-canary\nforged-log-line'
    provider_error = ProgrammingError(
        msg='provider body',
        errno=390111,
        sqlstate=poisoned_sqlstate,
        sfqid=poisoned_query_id,
        send_telemetry=False,
    )

    with caplog.at_level('ERROR'):
        error = ProviderBackendError(
            message='Snowflake connector operation failed.',
            error=provider_error,
        )

    rendered = f'{error!s} {error!r} {error.to_dict()} {caplog.text}'
    assert poisoned_sqlstate not in rendered
    assert poisoned_query_id not in rendered
    assert 'secret-canary' not in rendered
    assert 'code=390111' in rendered


@pytest.mark.parametrize(
    ('provider_error', 'expected'),
    [
        (
            ProgrammingError(
                msg='snowflake-provider-outer-canary',
                errno=390111,
                sqlstate='08001',
                send_telemetry=False,
            ),
            'code=390111',
        ),
        (
            ServerOperationError('databricks-provider-outer-canary'),
            'Databricks could not complete',
        ),
        (GoogleAPIError(
            'google-provider-outer-canary gs://private-bucket/secret/path'),
         'Google Cloud provider'),
        (RefreshError('google-auth-provider-outer-canary'),
         'Google Cloud provider'),
        (
            ClientError(
                {
                    'Error': {
                        'Code': 'AccessDenied',
                        'Message': 'aws-provider-outer-canary',
                    }
                },
                'GetObject',
            ),
            'AWS could not complete',
        ),
        (AzureError('azure-provider-outer-canary'), 'Azure could not complete'),
    ],
    # Avoid the repository's credential-gated ``snowflake`` pytest marker
    # heuristic: this classifier test is hermetic and must run in default CI.
    ids=['sf_driver', 'databricks', 'google', 'google_auth', 'aws', 'azure'],
)
def test_provider_classifier_stops_nested_cause_chain(
        provider_error, expected, caplog):
    """A classified provider is a terminal trust boundary."""
    nested_canary = 'nested-third-party-body password=nested-secret-canary'
    provider_error.__cause__ = _NestedThirdPartyError(nested_canary)

    with caplog.at_level('ERROR'):
        public_error = ConnectorError(
            'Connector operation failed.',
            error=provider_error,
        )
    try:
        raise public_error from None
    except ConnectorError as raised_error:
        public_error = raised_error
    assert public_error.__cause__ is None
    assert public_error.__suppress_context__ is True

    application_error, failure = _temporal_failure(public_error)
    rendered = ' '.join((
        str(public_error),
        repr(public_error),
        str(public_error.to_dict()),
        caplog.text,
        str(application_error),
        repr(application_error),
        str(failure),
    ))

    assert nested_canary not in rendered
    assert 'nested-secret-canary' not in rendered
    assert 'provider-outer-canary' not in rendered
    assert 'private-bucket' not in rendered
    assert 'Connector operation failed' in rendered
    assert expected in rendered
    assert type(provider_error).__name__ in caplog.text


@pytest.mark.asyncio
async def test_workflow_interceptor_drops_nested_provider_cause_from_history():
    """Workflow PublicException conversion must sever untrusted causes."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from temporalio_commons.worker_interceptors import (
        KumoWorkflowInboundInterceptor,
    )

    canary = 'WORKFLOW_CAUSE_SECRET_CANARY password=workflow-secret'
    provider_error = ProgrammingError(
        msg=canary,
        errno=390111,
        sqlstate='08001',
        query="SELECT 'WORKFLOW_CAUSE_SECRET_CANARY'",
        send_telemetry=False,
    )
    try:
        raise ConnectorError(
            'Snowflake connector operation failed.',
            error=provider_error,
        ) from provider_error
    except ConnectorError as connector_error:
        next_interceptor = SimpleNamespace(
            execute_workflow=AsyncMock(side_effect=connector_error),
        )

    interceptor = KumoWorkflowInboundInterceptor(next_interceptor)
    with pytest.raises(ApplicationError) as exc_info:
        await interceptor.execute_workflow(SimpleNamespace())

    application_error = exc_info.value
    assert application_error.__cause__ is None
    assert application_error.__suppress_context__ is True

    failure = Failure()
    DefaultFailureConverter().to_failure(
        application_error, DefaultPayloadConverter(), failure)
    rendered = ' '.join((
        str(application_error),
        repr(application_error),
        str(failure),
    ))
    assert canary not in rendered
    assert 'workflow-secret' not in rendered


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        # Standalone diagnostic error (e.g. the Databricks samples case).
        (ConnectorValidationDiagnosticError(
            ConnectorValidationDiagnosticCode.DATABRICKS_SAMPLE_TABLE_PERMISSIONS
        ), 'databricks_sample_table_permissions'),
        # Attribute form: a detailed ConnectorError carrying a safe category.
        (ConnectorError(
            'Schema s does not exist under catalog c.',
            diagnostic_code=ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND,
        ), 'target_not_found'),
        # Uncategorized connector failure and an arbitrary exception: no code.
        (ConnectorError('Some connector failure.'), None),
        (RuntimeError('permission denied: secret'), None),
        # A raw, non-enum attribute value is never surfaced as a code.
        (ConnectorError('boom', diagnostic_code='target_not_found'), None),
    ],
)
def test_connector_validation_diagnostic_value_normalizes_both_shapes(
        error, expected):
    assert connector_validation_diagnostic_value(error) == expected


def test_connector_error_preserves_message_and_carries_diagnostic_code():
    error = ConnectorError(
        'Schema s does not exist under catalog c.',
        diagnostic_code=ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND,
    )
    # The detailed public message is unchanged for non-masked callers (ingest)...
    assert str(error) == 'Schema s does not exist under catalog c.'
    # ...while the safe category rides alongside for the masking boundaries.
    assert error.diagnostic_code == (
        ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND)
