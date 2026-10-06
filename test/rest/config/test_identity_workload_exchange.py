# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)
from google.auth import compute_engine, external_account
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2 import credentials as oauth2_credentials
from kumoapi.data_source import (
    ADLSIdentityConfig,
    AzureCloud,
    BigQueryIdentityConfig,
    GCSIdentityConfig,
    GoogleAmbientAuthentication,
    GoogleServiceAccountKeyAuthentication,
    GoogleServiceAccountKeyCredentials,
    GoogleWorkloadIdentityAuthentication,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AddressingStyle,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3ConnectionConfig,
    S3IdentityConfig,
)
from pydantic import SecretStr

from io_utils.fsutil.credentials import credential_scope, pyarrow_filesystem
from io_utils.fsutil.s3_filesystem import (
    ThreadLocalBotoSession,
    head_object_with_unsigned_fallback,
)
from rest.config.credentials import (
    AWSAccessKeyCredentials,
    AWSAmbientCredentials,
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    GoogleCloudCredentials,
)
from rest.config.data_backend import ResolvedConnectorSpec
from rest.config.identity import (
    IdentityAuthenticationCategory,
    IdentityAuthenticationCategoryError,
    NativeWorkloadIdentityExchanger,
    WorkloadExchangeConfigurationError,
    _gcloud_cli_user_credentials,
    ambient_adls_identity_config,
    compose_resolved_connector_spec,
    identity_key,
    validate_google_key_material,
    validate_identity_authentication,
    validate_resolved_connector,
)
from service_framework._raw_deploy_config import (
    AmbientGCPWorkloadConfig,
    AmbientS3WorkloadConfig,
)
from statedb.config.common import ConnectorID, IdentityID
from temporalio_commons import dataclass_converter


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _s3_config() -> S3IdentityConfig:
    return S3IdentityConfig(
        authentication=S3AssumeRoleAuthentication(
            partition='aws',
            role_arn='arn:aws:iam::123456789012:role/customer-kumo',
        ), )


def _google_config(provider: str):
    authentication = GoogleWorkloadIdentityAuthentication(
        service_account_email=('customer-kumo@example.iam.gserviceaccount.com'))
    if provider == 'bigquery':
        return BigQueryIdentityConfig(authentication=authentication)
    return GCSIdentityConfig(authentication=authentication)


def _google_key_config(provider: str):
    authentication = GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email='reader@service-project.iam.gserviceaccount.com',
        client_id='1234567890',
        auth_uri='https://accounts.google.com/o/oauth2/auth',
        token_uri='https://oauth2.googleapis.com/token',
    )
    if provider == 'bigquery':
        return BigQueryIdentityConfig(authentication=authentication)
    return GCSIdentityConfig(authentication=authentication)


def _google_ambient_config(provider: str):
    authentication = GoogleAmbientAuthentication()
    if provider == 'bigquery':
        return BigQueryIdentityConfig(authentication=authentication)
    return GCSIdentityConfig(authentication=authentication)


_AMBIENT_GCP_EMAIL = 'kumo-workload@example-project.iam.gserviceaccount.com'


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_ambient_gcp_uses_source_directly_without_impersonation(
        monkeypatch, provider) -> None:
    source = Mock(spec=compute_engine.Credentials)
    source.service_account_email = _AMBIENT_GCP_EMAIL
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (source, 'example-project'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        lambda: Mock(approved_service_account_emails=(_AMBIENT_GCP_EMAIL, )),
    )

    result = NativeWorkloadIdentityExchanger().exchange(
        _google_ambient_config(provider), None)

    assert isinstance(result, GoogleCloudCredentials)
    assert result.provider == provider
    # The pod's OWN credential is returned - not an impersonated credential.
    assert result.credential is source
    source.refresh.assert_called_once()


def test_ambient_gcp_rejects_unapproved_service_account(monkeypatch) -> None:
    source = Mock(spec=compute_engine.Credentials)
    source.service_account_email = 'rogue@evil.iam.gserviceaccount.com'
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (source, 'p'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        lambda: Mock(approved_service_account_emails=(_AMBIENT_GCP_EMAIL, )),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='not approved'):
        NativeWorkloadIdentityExchanger().exchange(
            _google_ambient_config('gcs'), None)


def test_ambient_gcp_rejects_disallowed_source(monkeypatch) -> None:
    # A static / developer credential (not a workload chain) must be rejected.
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (Mock(), 'p'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        lambda: Mock(approved_service_account_emails=(_AMBIENT_GCP_EMAIL, )),
    )

    with pytest.raises(IdentityAuthenticationCategoryError) as exc:
        NativeWorkloadIdentityExchanger().exchange(
            _google_ambient_config('gcs'), None)
    assert exc.value.category == (
        IdentityAuthenticationCategory.DISALLOWED_SOURCE)


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_ambient_gcp_accepts_user_adc_on_devserver(monkeypatch,
                                                   provider) -> None:
    source = oauth2_credentials.Credentials(
        token='developer-access-token',
        refresh_token='developer-refresh-token',
        token_uri='https://oauth2.googleapis.com/token',
        client_id='developer-client-id',
        client_secret='developer-client-secret',
        account='developer@example.com',
    )
    refresh = Mock()
    monkeypatch.setattr(source, 'refresh', refresh)
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
    monkeypatch.setenv('KUMO_DEV_GCP_USER_CREDENTIALS_ENABLED', 'true')
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (source, 'developer-project'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        Mock(side_effect=WorkloadExchangeConfigurationError('disabled')),
    )

    result = NativeWorkloadIdentityExchanger().exchange(
        _google_ambient_config(provider), None)

    assert result.provider == provider
    assert result.credential is source
    assert result.project == 'developer-project'
    refresh.assert_called_once()


def test_ambient_gcp_devserver_falls_back_to_active_gcloud_profile(
        monkeypatch) -> None:
    source = oauth2_credentials.Credentials(token='gcloud-access-token')
    refresh = Mock()
    monkeypatch.setattr(source, 'refresh', refresh)
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
    monkeypatch.setenv('KUMO_DEV_GCP_USER_CREDENTIALS_ENABLED', 'true')
    monkeypatch.setattr('google.auth.default',
                        Mock(side_effect=RuntimeError('ADC unavailable')))
    gcloud_credentials = Mock(return_value=(source, 'gcloud-project'))
    monkeypatch.setattr(
        'rest.config.identity._gcloud_cli_user_credentials',
        gcloud_credentials,
    )
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        Mock(side_effect=WorkloadExchangeConfigurationError('disabled')),
    )

    result = NativeWorkloadIdentityExchanger().exchange(
        _google_ambient_config('bigquery'), None)

    assert result.credential is source
    assert result.project == 'gcloud-project'
    gcloud_credentials.assert_called_once_with(
        ['https://www.googleapis.com/auth/cloud-platform'])
    refresh.assert_called_once()


def test_gcloud_profile_adapter_refreshes_with_google_auth_contract(
        monkeypatch) -> None:
    def run(command, **kwargs):
        del kwargs
        if command[1:3] == ['auth', 'print-access-token']:
            return Mock(returncode=0,
                        stdout='{"token":"short-lived-access-token"}')
        assert command[1:4] == ['config', 'get-value', 'project']
        return Mock(returncode=0, stdout='example-project\n')

    monkeypatch.setattr('subprocess.run', run)

    credential, project = _gcloud_cli_user_credentials(
        ['https://www.googleapis.com/auth/cloud-platform'])
    credential.refresh(Mock())

    assert project == 'example-project'
    assert credential.token == 'short-lived-access-token'
    assert credential.expiry is not None


@pytest.mark.parametrize(('deploy_stage', 'enabled'), [
    ('prod', 'true'),
    ('dev_local', 'false'),
])
def test_ambient_gcp_rejects_user_adc_outside_enabled_devserver(
        monkeypatch, deploy_stage, enabled) -> None:
    source = oauth2_credentials.Credentials(token='developer-access-token')
    monkeypatch.setenv('DEPLOY_STAGE', deploy_stage)
    monkeypatch.setenv('KUMO_DEV_GCP_USER_CREDENTIALS_ENABLED', enabled)
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (source, 'developer-project'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        lambda: Mock(approved_service_account_emails=(_AMBIENT_GCP_EMAIL, )),
    )

    with pytest.raises(IdentityAuthenticationCategoryError) as exc:
        NativeWorkloadIdentityExchanger().exchange(
            _google_ambient_config('bigquery'), None)

    assert exc.value.category == (
        IdentityAuthenticationCategory.DISALLOWED_SOURCE)


def test_ambient_gcp_devserver_does_not_bypass_workload_allowlist(
        monkeypatch) -> None:
    source = Mock(spec=compute_engine.Credentials)
    source.service_account_email = _AMBIENT_GCP_EMAIL
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
    monkeypatch.setenv('KUMO_DEV_GCP_USER_CREDENTIALS_ENABLED', 'true')
    monkeypatch.setattr('google.auth.default', lambda scopes=None:
                        (source, 'workload-project'))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_gcp_workload_config',
        Mock(side_effect=WorkloadExchangeConfigurationError('disabled')),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='workload identity is disabled'):
        NativeWorkloadIdentityExchanger().exchange(
            _google_ambient_config('bigquery'), None)


def test_ambient_gcp_launch_config_validates_emails() -> None:
    config = AmbientGCPWorkloadConfig(
        approved_service_account_emails=(_AMBIENT_GCP_EMAIL, ))
    assert config.approved_service_account_emails == (_AMBIENT_GCP_EMAIL, )
    with pytest.raises(ValueError, match='must not be empty'):
        AmbientGCPWorkloadConfig(approved_service_account_emails=())
    with pytest.raises(ValueError, match='duplicates'):
        AmbientGCPWorkloadConfig(
            approved_service_account_emails=(_AMBIENT_GCP_EMAIL,
                                             _AMBIENT_GCP_EMAIL))
    with pytest.raises(ValueError, match='Google service-account'):
        AmbientGCPWorkloadConfig(
            approved_service_account_emails=('not-an-email', ))


def test_ambient_gcp_identity_key() -> None:
    assert identity_key(_google_ambient_config('gcs')) == 'gcs:ambient'
    assert identity_key(
        _google_ambient_config('bigquery')) == 'bigquery:ambient'


class _TestExternalAccountCredentials(external_account.Credentials):
    """Actual external-account base type with inert token retrieval."""
    def __init__(self) -> None:
        super().__init__(
            audience=('//iam.googleapis.com/projects/123/locations/global/'
                      'workloadIdentityPools/kumo/providers/runtime'),
            subject_token_type=('urn:ietf:params:oauth:token-type:jwt'),
            token_url='https://sts.googleapis.com/v1/token',
            credential_source={'file': '/var/run/kumo/subject-token'},
        )

    def retrieve_subject_token(self, request) -> str:
        del request
        return 'test-subject-token'


def _ambient_session(method: str = 'container-role',
                     token: str | None = 'temporary-token',
                     role_name: str = 'kumo-runtime'):
    frozen = Mock(access_key='temporary-access', secret_key='temporary-secret',
                  token=token)
    source = Mock(method=method)
    source.get_frozen_credentials.return_value = frozen
    sts = Mock()
    sts.get_caller_identity.return_value = {
        'Arn': ('arn:aws:sts::123456789012:assumed-role/'
                f'{role_name}/session')
    }
    session = Mock(spec=boto3.session.Session)
    session.get_credentials.return_value = source
    session.client.return_value = sts
    return session, source, sts


@pytest.mark.parametrize('method', [
    'assume-role-with-web-identity',
    'container-role',
    'iam-role',
])
def test_ambient_s3_attests_only_approved_workload_roles(monkeypatch,
                                                         method) -> None:
    session, _, sts = _ambient_session(method)
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )

    result = NativeWorkloadIdentityExchanger().exchange(
        S3IdentityConfig(authentication=S3AmbientAuthentication()), None)

    assert isinstance(result, AWSAmbientCredentials)
    assert result.session is session
    assert result.approved_role_arn == (
        'arn:aws:iam::123456789012:role/kumo-runtime')
    sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_concurrent_exchange_attests_once(monkeypatch) -> None:
    session, _, sts = _ambient_session()
    session_constructor = Mock(return_value=session)
    monkeypatch.setattr('boto3.session.Session', session_constructor)
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )
    exchanger = NativeWorkloadIdentityExchanger()
    config = S3IdentityConfig(authentication=S3AmbientAuthentication())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: exchanger.exchange(config, None), range(8)))

    assert all(result is results[0] for result in results)
    session_constructor.assert_called_once_with()
    sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_failed_leader_does_not_block_follower(monkeypatch) -> None:
    rejected, _, rejected_sts = _ambient_session(role_name='other-runtime')
    approved, _, approved_sts = _ambient_session()
    failure_started = threading.Event()
    release_failure = threading.Event()

    def reject_role():
        failure_started.set()
        assert release_failure.wait(1)
        return {
            'Arn': ('arn:aws:sts::123456789012:assumed-role/'
                    'other-runtime/session')
        }

    rejected_sts.get_caller_identity.side_effect = reject_role
    session_constructor = Mock(side_effect=[rejected, approved])
    monkeypatch.setattr('boto3.session.Session', session_constructor)
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )
    exchanger = NativeWorkloadIdentityExchanger()
    config = S3IdentityConfig(authentication=S3AmbientAuthentication())

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(exchanger.exchange, config, None)
        assert failure_started.wait(1)
        follower = executor.submit(exchanger.exchange, config, None)
        release_failure.set()
        with pytest.raises(WorkloadExchangeConfigurationError,
                           match='not approved'):
            leader.result()
        result = follower.result()

    assert result.session is approved
    assert exchanger.exchange(config, None) is result
    assert session_constructor.call_count == 2
    rejected_sts.get_caller_identity.assert_called_once_with()
    approved_sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_cache_reattests_after_ttl(monkeypatch) -> None:
    first, _, first_sts = _ambient_session()
    second, _, second_sts = _ambient_session()
    session_constructor = Mock(side_effect=[first, second])
    clock = [100.0]
    monkeypatch.setattr('boto3.session.Session', session_constructor)
    monkeypatch.setattr('rest.config.identity.time.monotonic', lambda: clock[0])
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )
    exchanger = NativeWorkloadIdentityExchanger()
    config = S3IdentityConfig(authentication=S3AmbientAuthentication())

    assert exchanger.exchange(config, None).session is first
    clock[0] = 159.999
    assert exchanger.exchange(config, None).session is first
    clock[0] = 160.0
    assert exchanger.exchange(config, None).session is second

    assert session_constructor.call_count == 2
    first_sts.get_caller_identity.assert_called_once_with()
    second_sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_cache_is_shared_across_configs_and_bound_to_role_policy(
        monkeypatch) -> None:
    first, _, _ = _ambient_session()
    second, _, second_sts = _ambient_session(role_name='replacement-runtime')
    session_constructor = Mock(side_effect=[first, second])
    approved_roles = [
        ('arn:aws:iam::123456789012:role/kumo-runtime', ),
    ]
    monkeypatch.setattr('boto3.session.Session', session_constructor)
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=approved_roles[0]),
    )
    exchanger = NativeWorkloadIdentityExchanger()
    west = S3IdentityConfig(
        authentication=S3AmbientAuthentication(),
        connection=S3ConnectionConfig(region='us-west-2'),
    )
    east = S3IdentityConfig(
        authentication=S3AmbientAuthentication(),
        connection=S3ConnectionConfig(region='us-east-1'),
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda config: exchanger.exchange(config, None),
                [west, east] * 4,
            ))

    assert all(result.session is first for result in results)
    assert session_constructor.call_count == 1
    approved_roles[0] = ('arn:aws:iam::123456789012:role/replacement-runtime', )
    result = exchanger.exchange(east, None)

    assert result.session is second
    assert result.approved_role_arn == approved_roles[0][0]
    assert session_constructor.call_count == 2
    second_sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_rejects_unapproved_attested_role(monkeypatch) -> None:
    session, _, sts = _ambient_session('container-role')
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/other-runtime', )),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='not approved'):
        NativeWorkloadIdentityExchanger().exchange(
            S3IdentityConfig(authentication=S3AmbientAuthentication()), None)

    sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_rejects_same_name_from_path_qualified_allowlist(
        monkeypatch) -> None:
    session, _, sts = _ambient_session('iam-role')
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/approved/kumo-runtime', )),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='not approved'):
        NativeWorkloadIdentityExchanger().exchange(
            S3IdentityConfig(authentication=S3AmbientAuthentication()), None)

    sts.get_caller_identity.assert_called_once_with()


def test_ambient_s3_launch_config_rejects_role_paths_and_duplicates() -> None:
    with pytest.raises(ValueError, match='pathless IAM role ARNs'):
        AmbientS3WorkloadConfig(approved_role_arns=(
            'arn:aws:iam::123456789012:role/approved/kumo-runtime', ))

    role = 'arn:aws:iam::123456789012:role/kumo-runtime'
    with pytest.raises(ValueError, match='must not contain duplicates'):
        AmbientS3WorkloadConfig(approved_role_arns=(role, role))


def test_ambient_s3_launch_config_accepts_exact_pathless_role() -> None:
    role = 'arn:aws:iam::123456789012:role/kumo-runtime'
    config = AmbientS3WorkloadConfig(approved_role_arns=(role, ))

    assert config.approved_role_arns == (role, )


@pytest.mark.parametrize('method', ['env', 'shared-credentials-file', 'sso'])
def test_ambient_s3_rejects_static_and_developer_provider_chains(
        monkeypatch, method) -> None:
    session, _, _ = _ambient_session(method)
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='web-identity, container-role, or instance-role'):
        NativeWorkloadIdentityExchanger().exchange(
            S3IdentityConfig(authentication=S3AmbientAuthentication()), None)


def test_ambient_s3_requires_temporary_session_token(monkeypatch) -> None:
    session, _, _ = _ambient_session(token=None)
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))
    monkeypatch.setattr(
        'rest.config.identity.require_ambient_s3_workload_config',
        lambda: Mock(approved_role_arns=(
            'arn:aws:iam::123456789012:role/kumo-runtime', )),
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='temporary session credentials'):
        NativeWorkloadIdentityExchanger().exchange(
            S3IdentityConfig(authentication=S3AmbientAuthentication()), None)


def test_ambient_session_is_exact_for_boto_and_refreshed_for_each_pyarrow_use(
        monkeypatch) -> None:
    session, source, s3 = _ambient_session()
    s3.head_bucket.return_value = {'BucketRegion': 'us-west-2'}
    credentials = AWSAmbientCredentials(
        session=session,
        approved_role_arn='arn:aws:iam::123456789012:role/kumo-runtime',
    )
    pyarrow_s3 = Mock(side_effect=[object(), object()])
    monkeypatch.setattr('pyarrow.fs.S3FileSystem', pyarrow_s3)
    conflicting_default = Mock(side_effect=AssertionError(
        'independent AWS default chain must not be consulted'))
    monkeypatch.setattr(ThreadLocalBotoSession, 'core_session',
                        conflicting_default)

    with credential_scope(credentials.as_dict(), require_explicit=True):
        assert ThreadLocalBotoSession().session() is session
        first = pyarrow_filesystem('s3://company-data/a.parquet')
    with credential_scope(credentials.as_dict(), require_explicit=True):
        second = pyarrow_filesystem('s3://company-data/b.parquet')

    assert first is not second
    assert source.get_frozen_credentials.call_count == 2
    assert s3.head_bucket.call_count == 2
    assert all(call.kwargs['region'] == 'us-west-2'
               for call in pyarrow_s3.call_args_list)
    conflicting_default.assert_not_called()
    assert pyarrow_s3.call_count == 2


def test_ambient_credentials_are_not_temporal_serializable_or_repr_visible(
) -> None:
    session, _, _ = _ambient_session()
    credentials = AWSAmbientCredentials(
        session=session,
        approved_role_arn='arn:aws:iam::123456789012:role/kumo-runtime',
    )

    with pytest.raises(TypeError, match='activity-local'):
        dataclass_converter.DataclassJsonPayloadConverter().to_payload(
            credentials)
    rendered = repr(credentials)
    assert 'temporary-access' not in rendered
    assert 'temporary-secret' not in rendered
    assert 'kumo-runtime' not in rendered


def test_explicit_ambient_scope_never_falls_back_to_unsigned_s3(
        monkeypatch) -> None:
    client = Mock()
    client.head_object.side_effect = NoCredentialsError()
    unsigned = Mock()
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.threadsafe_boto.s3_client',
        Mock(return_value=client),
    )
    monkeypatch.setattr('io_utils.fsutil.s3_filesystem._unsigned_s3_client',
                        unsigned)
    session, _, _ = _ambient_session()
    credentials = AWSAmbientCredentials(
        session=session,
        approved_role_arn='arn:aws:iam::123456789012:role/kumo-runtime',
    )

    with credential_scope(credentials.as_dict(), require_explicit=True):
        with pytest.raises(NoCredentialsError):
            head_object_with_unsigned_fallback('s3://company-data/a.parquet')

    unsigned.assert_not_called()


def test_workload_exchange_must_be_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.delenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', raising=False)

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='is not configured'):
        NativeWorkloadIdentityExchanger().exchange(_s3_config(), None)


def test_aws_exchange_rejects_static_or_developer_source_credentials(
        monkeypatch) -> None:
    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', 'true')
    session = Mock()
    session.get_credentials.return_value = Mock(method='env')
    constructor = Mock(return_value=session)
    monkeypatch.setattr('boto3.session.Session', constructor)

    with pytest.raises(WorkloadExchangeConfigurationError, match='static keys'):
        NativeWorkloadIdentityExchanger().exchange(_s3_config(), None)

    session.client.assert_not_called()


def test_aws_exchange_assumes_only_the_selected_customer_role(
        monkeypatch) -> None:
    monkeypatch.setenv('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED', 'true')
    sts = Mock()
    sts.assume_role.return_value = {
        'Credentials': {
            'AccessKeyId': 'temporary-access-key',
            'SecretAccessKey': 'temporary-secret',
            'SessionToken': 'temporary-token',
        }
    }
    session = Mock()
    session.get_credentials.return_value = Mock(
        method='assume-role-with-web-identity')
    session.client.return_value = sts
    monkeypatch.setattr('boto3.session.Session', Mock(return_value=session))

    result = NativeWorkloadIdentityExchanger().exchange(
        _s3_config(),
        S3AssumeRoleCredentials(external_id=SecretStr('customer-external-id')),
    )

    assert isinstance(result, AWSFederatedCredentials)
    assert result.access_key_id.get_secret_value() == 'temporary-access-key'
    session.client.assert_called_once_with('sts')
    sts.assume_role.assert_called_once_with(
        RoleArn='arn:aws:iam::123456789012:role/customer-kumo',
        RoleSessionName='kumo-project-connector',
        ExternalId='customer-external-id',
    )


def test_s3_access_key_resolution_does_not_use_aws_sts(monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    session_constructor = Mock()
    monkeypatch.setattr('boto3.session.Session', session_constructor)
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://127.0.0.1:9000',
            region='us-east-1',
            addressing_style=S3AddressingStyle.PATH,
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('minio-access-key'),
        secret_access_key=SecretStr('minio-secret-key'),
    )

    resolved = NativeWorkloadIdentityExchanger().exchange(config, credentials)

    assert isinstance(resolved, AWSAccessKeyCredentials)
    assert resolved.region == 'us-east-1'
    assert resolved.access_key_id.get_secret_value() == 'minio-access-key'
    session_constructor.assert_not_called()


def test_s3_access_key_validation_makes_one_bounded_signed_request(
        monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://127.0.0.1:9000',
            region='us-east-1',
            addressing_style=S3AddressingStyle.PATH,
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('minio-access-key'),
        secret_access_key=SecretStr('minio-secret-key'),
    )
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr('boto3.client', constructor)

    validate_identity_authentication(
        config,
        credentials,
        NativeWorkloadIdentityExchanger(),
    )

    client.list_buckets.assert_called_once_with()
    options = constructor.call_args.kwargs
    assert options['endpoint_url'] == 'http://127.0.0.1:9000'
    assert options['region_name'] == 'us-east-1'
    assert options['aws_access_key_id'] == 'minio-access-key'
    assert options['aws_secret_access_key'] == 'minio-secret-key'
    assert options['config'].connect_timeout == 3
    assert options['config'].read_timeout == 3
    assert options['config'].retries['total_max_attempts'] == 1


def test_s3_access_key_validation_rejects_provider_failure(monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://127.0.0.1:9000',
            region='us-east-1',
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('invalid-access-key'),
        secret_access_key=SecretStr('invalid-secret-key'),
    )
    poison = 'provider-response-must-not-leak'
    client = Mock()
    client.list_buckets.side_effect = ClientError(
        {'Error': {
            'Code': 'SignatureDoesNotMatch',
            'Message': poison,
        }},
        'ListBuckets',
    )
    monkeypatch.setattr('boto3.client', Mock(return_value=client))

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='S3 access key or connection settings') as caught:
        validate_identity_authentication(
            config,
            credentials,
            NativeWorkloadIdentityExchanger(),
        )

    assert caught.value.category == (
        IdentityAuthenticationCategory.S3_VALIDATION_FAILED)
    assert poison not in str(caught.value)
    assert caught.value.__cause__ is None


def test_s3_access_key_validation_sanitizes_client_construction_failure(
        monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://unavailable.internal:9000',
            region='us-east-1',
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('invalid-access-key'),
        secret_access_key=SecretStr('invalid-secret-key'),
    )
    poison = 'http://provider-detail.internal:9000'
    monkeypatch.setattr(
        'boto3.client',
        Mock(side_effect=EndpointConnectionError(endpoint_url=poison)),
    )

    with pytest.raises(IdentityAuthenticationCategoryError) as caught:
        validate_identity_authentication(
            config,
            credentials,
            NativeWorkloadIdentityExchanger(),
        )

    assert caught.value.category == (
        IdentityAuthenticationCategory.S3_VALIDATION_FAILED)
    assert poison not in str(caught.value)
    assert caught.value.__cause__ is None


def test_s3_access_key_validation_accepts_permission_limited_principal(
        monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://127.0.0.1:9000',
            region='us-east-1',
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('read-only-access-key'),
        secret_access_key=SecretStr('read-only-secret-key'),
    )
    client = Mock()
    client.list_buckets.side_effect = ClientError(
        {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'not authorized to list every bucket',
            }
        },
        'ListBuckets',
    )
    monkeypatch.setattr('boto3.client', Mock(return_value=client))

    validate_identity_authentication(
        config,
        credentials,
        NativeWorkloadIdentityExchanger(),
    )


def test_ambient_s3_connector_keeps_refreshable_session_as_only_auth(
        monkeypatch) -> None:
    identity_id = IdentityID.generate()
    connector_id = ConnectorID.generate()
    session, _, _ = _ambient_session()
    ambient = AWSAmbientCredentials(
        session=session,
        approved_role_arn='arn:aws:iam::123456789012:role/kumo-runtime',
    )
    monkeypatch.setattr(
        'rest.config.identity.credentials_for_identity',
        Mock(return_value=ambient),
    )
    identity = Mock(
        id=identity_id,
        config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
        credentials_rotated_at=None,
    )
    native_connector = Mock()
    native_connector.config.return_value = {'type': 's3'}
    connector_config = Mock(type=Mock(value='s3'))
    connector_config.name = str(connector_id)
    connector_config.as_connector.return_value = native_connector
    connector = Mock(
        id=connector_id,
        identity_id=identity_id,
        config=connector_config,
    )

    resolved = compose_resolved_connector_spec(
        identity,
        connector,
        None,
        Mock(),
    )

    assert isinstance(resolved.credentials, AWSAmbientCredentials)
    assert resolved.credentials.connection is None
    assert 's3_connection' not in resolved.credentials.as_dict()
    native_connector.set_credentials.assert_called_once_with(
        resolved.credentials.as_dict())


@pytest.mark.parametrize(
    ('authentication', 'runtime_credentials'),
    [
        (
            S3AccessKeyAuthentication(),
            AWSAccessKeyCredentials(
                access_key_id=SecretStr('plain-aws-access-key'),
                secret_access_key=SecretStr('plain-aws-secret-key'),
            ),
        ),
        (
            _s3_config().authentication,
            AWSFederatedCredentials(
                access_key_id=SecretStr('federated-aws-access-key'),
                secret_access_key=SecretStr('federated-aws-secret-key'),
                session_token=SecretStr('federated-aws-session-token'),
            ),
        ),
    ],
)
def test_plain_aws_connector_does_not_attach_s3_connection(
        monkeypatch, authentication, runtime_credentials) -> None:
    identity_id = IdentityID.generate()
    connector_id = ConnectorID.generate()
    monkeypatch.setattr(
        'rest.config.identity.credentials_for_identity',
        Mock(return_value=runtime_credentials),
    )
    identity = Mock(
        id=identity_id,
        config=S3IdentityConfig(authentication=authentication),
        credentials_rotated_at=None,
    )
    native_connector = Mock()
    native_connector.config.return_value = {'type': 's3'}
    connector_config = Mock(type=Mock(value='s3'))
    connector_config.name = str(connector_id)
    connector_config.region = 'us-west-2'
    connector_config.as_connector.return_value = native_connector
    connector = Mock(
        id=connector_id,
        identity_id=identity_id,
        config=connector_config,
    )

    resolved = compose_resolved_connector_spec(
        identity,
        connector,
        None,
        Mock(),
    )

    assert resolved.credentials.region == 'us-west-2'
    assert resolved.credentials.connection is None
    assert 's3_connection' not in resolved.credentials.as_dict()
    native_connector.set_credentials.assert_called_once_with(
        resolved.credentials.as_dict())


def test_s3_http_endpoint_is_rejected_outside_local_modes(monkeypatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', 'prod')
    config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=S3ConnectionConfig(
            endpoint_url='http://minio.internal:9000',
            region='us-east-1',
        ),
    )
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('minio-access-key'),
        secret_access_key=SecretStr('minio-secret-key'),
    )

    with pytest.raises(WorkloadExchangeConfigurationError, match='HTTPS'):
        NativeWorkloadIdentityExchanger().exchange(config, credentials)


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_gcp_exchange_rejects_service_account_or_user_source_credentials(
        monkeypatch, provider) -> None:
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setattr('google.auth.default',
                        Mock(return_value=(object(), 'source-project')))
    impersonate = Mock()
    monkeypatch.setattr('google.auth.impersonated_credentials.Credentials',
                        impersonate)
    config = _google_config(provider)

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='not an approved workload credential') as caught:
        NativeWorkloadIdentityExchanger().exchange(config, None)

    assert caught.value.category == IdentityAuthenticationCategory.DISALLOWED_SOURCE
    impersonate.assert_not_called()


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_gcp_exchange_disabled_is_a_safe_category(monkeypatch,
                                                  provider) -> None:
    monkeypatch.delenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', raising=False)

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='disabled for this deployment') as caught:
        NativeWorkloadIdentityExchanger().exchange(_google_config(provider),
                                                   None)

    assert caught.value.category == IdentityAuthenticationCategory.EXCHANGE_DISABLED


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_gcp_source_failure_is_sanitized_by_category(monkeypatch,
                                                     provider) -> None:
    poison = 'poison-source-token'
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setattr('google.auth.default',
                        Mock(side_effect=RefreshError(poison)))

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='workload credential is unavailable') as caught:
        NativeWorkloadIdentityExchanger().exchange(_google_config(provider),
                                                   None)

    assert caught.value.category == IdentityAuthenticationCategory.SOURCE_UNAVAILABLE
    assert poison not in str(caught.value)


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_gcp_impersonation_failure_is_sanitized_by_category(
        monkeypatch, provider) -> None:
    poison = 'poison-impersonation-response'
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    source = compute_engine.Credentials()
    monkeypatch.setattr('google.auth.default',
                        Mock(return_value=(source, 'source-project')))
    exchanged = Mock()
    exchanged.refresh.side_effect = RefreshError(poison)
    monkeypatch.setattr(
        'google.auth.impersonated_credentials.Credentials',
        Mock(return_value=exchanged),
    )

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='impersonation was denied') as caught:
        NativeWorkloadIdentityExchanger().exchange(_google_config(provider),
                                                   None)

    assert caught.value.category == IdentityAuthenticationCategory.IMPERSONATION_DENIED
    assert poison not in str(caught.value)


@pytest.mark.parametrize(
    ('provider', 'scope'),
    [
        ('bigquery', 'https://www.googleapis.com/auth/cloud-platform'),
        ('gcs', 'https://www.googleapis.com/auth/devstorage.read_write'),
    ],
)
def test_gcp_exchange_impersonates_selected_service_account_from_workload_source(
        monkeypatch, provider, scope) -> None:
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    source = compute_engine.Credentials()
    monkeypatch.setattr(
        'google.auth.default',
        Mock(return_value=(source, 'deployment-source-project')),
    )
    exchanged = Mock()
    impersonate = Mock(return_value=exchanged)
    monkeypatch.setattr('google.auth.impersonated_credentials.Credentials',
                        impersonate)
    config = _google_config(provider)

    result = NativeWorkloadIdentityExchanger().exchange(config, None)

    assert isinstance(result, GoogleCloudCredentials)
    assert result.provider == provider
    assert result.credential is exchanged
    assert result.project == 'deployment-source-project'
    impersonate.assert_called_once_with(
        source_credentials=source,
        target_principal='customer-kumo@example.iam.gserviceaccount.com',
        target_scopes=[scope],
    )
    exchanged.refresh.assert_called_once()


@pytest.mark.parametrize(
    ('provider', 'scope'),
    [
        ('bigquery', 'https://www.googleapis.com/auth/cloud-platform'),
        ('gcs', 'https://www.googleapis.com/auth/devstorage.read_write'),
    ],
)
def test_gcp_exchange_accepts_external_account_workload_source(
        monkeypatch, provider, scope) -> None:
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')
    source = _TestExternalAccountCredentials()
    monkeypatch.setattr(
        'google.auth.default',
        Mock(return_value=(source, 'external-source-project')),
    )
    exchanged = Mock()
    impersonate = Mock(return_value=exchanged)
    monkeypatch.setattr('google.auth.impersonated_credentials.Credentials',
                        impersonate)

    result = NativeWorkloadIdentityExchanger().exchange(
        _google_config(provider), None)

    assert isinstance(source, external_account.Credentials)
    assert isinstance(result, GoogleCloudCredentials)
    assert result.provider == provider
    assert result.credential is exchanged
    assert result.project == 'external-source-project'
    impersonate.assert_called_once_with(
        source_credentials=source,
        target_principal='customer-kumo@example.iam.gserviceaccount.com',
        target_scopes=[scope],
    )
    exchanged.refresh.assert_called_once()


def test_google_runtime_credentials_are_activity_local_and_secret_safe(
) -> None:
    credential = Mock(token='poison-access-token')
    credentials = GoogleCloudCredentials(
        provider='gcs',
        credential=credential,
        project='deployment-project',
    )

    with pytest.raises(TypeError, match='activity-local'):
        dataclass_converter.DataclassJsonPayloadConverter().to_payload(
            credentials)
    assert 'poison-access-token' not in repr(credentials)


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_google_workload_validation_refreshes_exactly_once(provider) -> None:
    credential = Mock()

    def exchange(config, credentials):
        del config, credentials
        credential.refresh(Request())
        return GoogleCloudCredentials(
            provider=provider,
            credential=credential,
            project='deployment-source-project',
        )

    exchanger = Mock()
    exchanger.exchange.side_effect = exchange

    validate_identity_authentication(_google_config(provider), None, exchanger)

    credential.refresh.assert_called_once()


@pytest.mark.parametrize('provider', ['bigquery', 'gcs'])
def test_google_key_refresh_failure_uses_stored_key_category(
        monkeypatch, provider) -> None:
    poison = 'poison-refresh-response'
    credential = Mock()
    credential.refresh.side_effect = RefreshError(poison)
    monkeypatch.setattr(
        'rest.config.identity.credentials_for_identity',
        Mock(return_value=GoogleCloudCredentials(
            provider=provider,
            credential=credential,
            project='service-project',
        )),
    )
    credentials = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('key-id'),
        private_key=SecretStr('not-used'),
    )

    with pytest.raises(IdentityAuthenticationCategoryError,
                       match='service-account key is invalid') as caught:
        validate_identity_authentication(
            _google_key_config(provider),
            credentials,
            Mock(),
        )

    assert caught.value.category == IdentityAuthenticationCategory.STORED_KEY_INVALID
    assert poison not in str(caught.value)
    assert caught.value.__cause__ is None


def test_google_key_rejects_noncanonical_endpoint_before_provider_call(
        monkeypatch) -> None:
    authentication = GoogleServiceAccountKeyAuthentication(
        project_id='service-project',
        client_email='reader@service-project.iam.gserviceaccount.com',
        client_id='1234567890',
        auth_uri='https://accounts.google.com/o/oauth2/auth',
        token_uri='https://oauth2.googleapis.com/token',
    )
    object.__setattr__(authentication, 'token_uri',
                       'http://169.254.169.254/poison-token')
    provider_constructor = Mock()
    monkeypatch.setattr(
        'google.oauth2.service_account.Credentials.from_service_account_info',
        provider_constructor,
    )

    with pytest.raises(ValueError, match='not canonical'):
        validate_google_key_material(
            authentication,
            GoogleServiceAccountKeyCredentials(
                private_key_id=SecretStr('key-id'),
                private_key=SecretStr('not-reached'),
            ),
        )

    provider_constructor.assert_not_called()


def test_azure_exchange_requires_federated_token_file(monkeypatch) -> None:
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.delenv('AZURE_FEDERATED_TOKEN_FILE', raising=False)
    config = ADLSIdentityConfig(
        tenant_id='customer-tenant',
        client_id='customer-application',
        cloud=AzureCloud.PUBLIC,
    )

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='AZURE_FEDERATED_TOKEN_FILE'):
        NativeWorkloadIdentityExchanger().exchange(config, None)


@pytest.mark.parametrize(('authority', 'cloud'), [
    ('https://login.microsoftonline.com/', AzureCloud.PUBLIC),
    ('https://login.microsoftonline.us/', AzureCloud.US_GOVERNMENT),
    ('https://login.chinacloudapi.cn/', AzureCloud.CHINA),
])
def test_ambient_adls_config_uses_injected_workload_identity(
        monkeypatch, authority, cloud) -> None:
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('AZURE_TENANT_ID', 'tenant-id')
    monkeypatch.setenv('AZURE_CLIENT_ID', 'client-id')
    monkeypatch.setenv('AZURE_AUTHORITY_HOST', authority)

    assert ambient_adls_identity_config() == ADLSIdentityConfig(
        tenant_id='tenant-id',
        client_id='client-id',
        cloud=cloud,
    )


def test_ambient_adls_config_is_absent_when_exchange_is_disabled(
        monkeypatch) -> None:
    monkeypatch.delenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', raising=False)

    assert ambient_adls_identity_config() is None


def test_ambient_adls_config_requires_complete_aks_injection(
        monkeypatch) -> None:
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('AZURE_TENANT_ID', 'tenant-id')
    monkeypatch.delenv('AZURE_CLIENT_ID', raising=False)
    monkeypatch.setenv('AZURE_AUTHORITY_HOST',
                       'https://login.microsoftonline.com/')

    with pytest.raises(WorkloadExchangeConfigurationError,
                       match='AZURE_CLIENT_ID'):
        ambient_adls_identity_config()


@pytest.mark.parametrize(('cloud', 'authority', 'scope'), [
    (AzureCloud.PUBLIC, 'login.microsoftonline.com',
     'https://storage.azure.com/.default'),
    (AzureCloud.US_GOVERNMENT, 'login.microsoftonline.us',
     'https://storage.azure.us/.default'),
    (AzureCloud.CHINA, 'login.chinacloudapi.cn',
     'https://storage.azure.cn/.default'),
])
def test_azure_exchange_targets_selected_customer_application(
        monkeypatch, cloud, authority, scope) -> None:
    monkeypatch.setenv('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED', 'true')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', '/var/run/kumo/token')
    credential = Mock()
    constructor = Mock(return_value=credential)
    monkeypatch.setattr('azure.identity.WorkloadIdentityCredential',
                        constructor)
    config = ADLSIdentityConfig(
        tenant_id='customer-tenant',
        client_id='customer-application',
        cloud=cloud,
    )

    result = NativeWorkloadIdentityExchanger().exchange(config, None)

    assert isinstance(result, AzureFederatedCredentials)
    assert result.credential is credential
    constructor.assert_called_once_with(
        tenant_id='customer-tenant',
        client_id='customer-application',
        token_file_path='/var/run/kumo/token',
        authority=authority,
    )
    credential.get_token.assert_called_once_with(scope)


def test_warehouse_target_validation_relies_on_open_permission_check_once(
        monkeypatch) -> None:
    connector = Mock()
    monkeypatch.setattr(
        'kumo.connector.connector.Connector.from_config',
        Mock(return_value=connector),
    )
    credentials = AWSFederatedCredentials(
        access_key_id=SecretStr('temporary-access-key'),
        secret_access_key=SecretStr('temporary-secret'),
        session_token=SecretStr('temporary-token'),
    )

    validate_resolved_connector(
        ResolvedConnectorSpec(
            connector_id=ConnectorID.generate(),
            connector_config=object(),
            credentials=credentials,
        ))

    connector.require_explicit_credentials.assert_called_once_with()
    connector.set_credentials.assert_called_once_with(credentials.as_dict())
    connector.open.assert_called_once_with()
    connector.validate_permissions.assert_not_called()
    connector.close.assert_called_once_with()


def test_google_key_mapping_preserves_hosted_normalization(monkeypatch):
    from rest.config.identity import _google_service_account_info

    authentication = _google_key_config('bigquery').authentication
    private_key = '  synthetic\\nPEM\n  '
    credentials = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr(' key-id '),
        private_key=SecretStr(private_key),
    )
    parser = Mock()
    monkeypatch.setattr(
        'rest.config.identity.serialization.load_pem_private_key', parser)
    ambient = Mock(side_effect=AssertionError('key auth must not use ADC'))
    monkeypatch.setattr('google.auth.default', ambient)

    info = _google_service_account_info(authentication, credentials)
    parser.assert_called_once_with(private_key.encode('utf-8'), password=None)
    assert info == {
        'type': 'service_account',
        'project_id': authentication.project_id,
        'client_email': authentication.client_email,
        'client_id': authentication.client_id,
        'auth_uri': authentication.auth_uri,
        'token_uri': authentication.token_uri,
        'universe_domain': authentication.universe_domain,
        'private_key_id': 'key-id',
        'private_key': private_key,
    }
    ambient.assert_not_called()
