# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from dataclasses import replace
from enum import Enum
from typing import Optional, Protocol
from urllib.parse import urlsplit, urlunsplit

import google.auth.credentials
from cryptography.hazmat.primitives import serialization
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from kumoapi.data_source import (
    ADLSIdentityConfig,
    AzureCloud,
    BigQueryIdentityConfig,
    DatabricksIdentityConfig,
    DatabricksOAuthCredentials,
    GCSIdentityConfig,
    GoogleAmbientAuthentication,
    GoogleServiceAccountKeyAuthentication,
    GoogleServiceAccountKeyCredentials,
    GoogleWorkloadIdentityAuthentication,
    IdentityConfig,
    IdentityCredentials,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3IdentityConfig,
    SnowflakeAuthenticationKind,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
)
from pydantic import SecretStr
from snowflake.connector import connect

from io_utils.fsutil.s3_connection import (
    AWS_WORKLOAD_CREDENTIAL_METHODS,
    S3Connection,
)
from rest.config.connector_db import ExternalConnectorDBEntry
from rest.config.credentials import (
    AWSAccessKeyCredentials,
    AWSAmbientCredentials,
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    Credentials,
    DatabricksCredentials,
    GoogleCloudCredentials,
    KeyPair,
    UsernamePassword,
)
from rest.config.data_backend import ResolvedConnectorSpec
from rest.config.encryption import CredentialsEncryptor
from service_framework.deploy_stage import DeployStage, deploy_stage
from service_framework.developer_credentials import (
    dev_gcp_user_credentials_enabled,
)
from statedb.rest.identity import IdentityDBEntry


class WorkloadExchangeConfigurationError(RuntimeError):
    pass


class IdentityAuthenticationCategory(str, Enum):
    EXCHANGE_DISABLED = 'exchange_disabled'
    SOURCE_UNAVAILABLE = 'source_unavailable'
    DISALLOWED_SOURCE = 'disallowed_source'
    IMPERSONATION_DENIED = 'impersonation_denied'
    S3_VALIDATION_FAILED = 's3_validation_failed'
    STORED_KEY_INVALID = 'stored_key_invalid'


_IDENTITY_AUTHENTICATION_CATEGORY_MESSAGES = {
    IdentityAuthenticationCategory.EXCHANGE_DISABLED:
    'Google workload exchange is disabled for this deployment.',
    IdentityAuthenticationCategory.SOURCE_UNAVAILABLE:
    'The deployment Google workload credential is unavailable.',
    IdentityAuthenticationCategory.DISALLOWED_SOURCE:
    'The deployment Google credential source is not an approved workload '
    'credential.',
    IdentityAuthenticationCategory.IMPERSONATION_DENIED:
    'Google service-account impersonation was denied or unavailable.',
    IdentityAuthenticationCategory.S3_VALIDATION_FAILED:
    'The stored S3 access key or connection settings are invalid.',
    IdentityAuthenticationCategory.STORED_KEY_INVALID:
    'The stored Google service-account key is invalid.',
}


class IdentityAuthenticationCategoryError(WorkloadExchangeConfigurationError):
    """Trusted, fixed authentication category safe for public boundaries."""
    def __init__(self, category: IdentityAuthenticationCategory) -> None:
        self.category = category
        super().__init__(_IDENTITY_AUTHENTICATION_CATEGORY_MESSAGES[category])


_GOOGLE_CLOUD_PLATFORM_SCOPE = (
    'https://www.googleapis.com/auth/cloud-platform')
_GOOGLE_GCS_SCOPE = ('https://www.googleapis.com/auth/devstorage.read_write')
_GOOGLE_SERVICE_ACCOUNT_AUTH_URI = ('https://accounts.google.com/o/oauth2/auth')
_GOOGLE_SERVICE_ACCOUNT_TOKEN_URI = 'https://oauth2.googleapis.com/token'
_GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN = 'googleapis.com'
# Coalesce one connector fanout while bounding stale role attestation.
_AMBIENT_AWS_ATTESTATION_TTL_SECONDS = 60.0


def is_ambient_s3_identity(config: IdentityConfig) -> bool:
    return (isinstance(config, S3IdentityConfig)
            and isinstance(config.authentication, S3AmbientAuthentication))


def require_ambient_s3_workload_config():
    """Return the explicit launch policy or fail closed when absent."""
    from service_framework.deploy_env_config import deploy_env

    config = deploy_env().ambient_s3
    if config is None:
        raise WorkloadExchangeConfigurationError(
            'Ambient S3 workload identity is disabled')
    return config


def is_ambient_gcp_identity(config: IdentityConfig) -> bool:
    return (isinstance(config, (BigQueryIdentityConfig, GCSIdentityConfig))
            and isinstance(config.authentication, GoogleAmbientAuthentication))


def require_ambient_gcp_workload_config():
    """Return the explicit ambient-GCP launch policy or fail closed."""
    from service_framework.deploy_env_config import deploy_env

    config = deploy_env().ambient_gcp
    if config is None:
        raise WorkloadExchangeConfigurationError(
            'Ambient GCP workload identity is disabled')
    return config


def _approved_ambient_role(caller_arn: str,
                           approved_role_arns: tuple[str, ...]) -> str:
    """Map an STS assumed-role ARN to exactly one approved IAM role ARN."""
    parts = caller_arn.split(':', 5)
    if len(parts) != 6 or parts[0] != 'arn':
        raise WorkloadExchangeConfigurationError(
            'AWS workload caller did not return a valid ARN')
    partition, service, account_id, resource = (parts[1], parts[2], parts[4],
                                                parts[5])
    if service == 'iam' and resource.startswith('role/'):
        if caller_arn in approved_role_arns:
            return caller_arn
        raise WorkloadExchangeConfigurationError(
            'AWS workload role is not approved for ambient S3')
    if service != 'sts' or not resource.startswith('assumed-role/'):
        raise WorkloadExchangeConfigurationError(
            'AWS workload caller is not an IAM role session')
    role_parts = resource.split('/')
    if len(role_parts) != 3 or not role_parts[1] or not role_parts[2]:
        raise WorkloadExchangeConfigurationError(
            'AWS workload assumed-role ARN is malformed')
    role_name = role_parts[1]
    iam_role_arn = (f'arn:{partition}:iam::{account_id}:role/{role_name}')
    if iam_role_arn not in approved_role_arns:
        raise WorkloadExchangeConfigurationError(
            'AWS workload role is not approved for ambient S3')
    return iam_role_arn


def _ambient_aws_credentials(
    approved_role_arns: tuple[str, ...]
) -> AWSAmbientCredentials:
    """Resolve and attest only an AWS workload-role credential provider."""
    import boto3

    session = boto3.session.Session()
    source = session.get_credentials()
    source_method = getattr(source, 'method', None)
    if source is None or source_method not in AWS_WORKLOAD_CREDENTIAL_METHODS:
        raise WorkloadExchangeConfigurationError(
            'Ambient S3 requires web-identity, container-role, or '
            'instance-role credentials')
    caller = session.client('sts').get_caller_identity()
    caller_arn = caller.get('Arn') if isinstance(caller, dict) else None
    if not isinstance(caller_arn, str):
        raise WorkloadExchangeConfigurationError(
            'AWS workload role attestation did not return an ARN')
    approved_role_arn = _approved_ambient_role(caller_arn,
                                               approved_role_arns)
    frozen = source.get_frozen_credentials()
    if not (frozen.access_key and frozen.secret_key and frozen.token):
        raise WorkloadExchangeConfigurationError(
            'AWS workload role did not provide temporary session credentials')
    return AWSAmbientCredentials(
        session=session,
        approved_role_arn=approved_role_arn,
    )


def _gcloud_cli_user_credentials(
    scopes: list[str]
) -> tuple['google.auth.credentials.Credentials', Optional[str]]:
    """Build refreshable dev credentials from the active gcloud profile.

    ``gcloud auth login`` and ``gcloud auth application-default login`` use
    separate stores.  Devserver accepts both: ADC is resolved normally, while
    this adapter asks gcloud to mint a short-lived access token on refresh. It
    never reads gcloud's private credential database or persists the token.
    """
    import json
    import subprocess
    from datetime import datetime, timedelta, timezone

    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials as UserCredentials

    def refresh_handler(request, scopes=None):
        del request, scopes
        try:
            result = subprocess.run(
                [
                    'gcloud', 'auth', 'print-access-token', '--quiet',
                    '--format=json'
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            document = json.loads(
                result.stdout) if result.returncode == 0 else {}
            token = document.get('token') if isinstance(document,
                                                        dict) else None
            if not isinstance(token, str) or not token:
                raise ValueError('missing token')
        except Exception:
            raise RefreshError(
                'The active gcloud developer login is unavailable. Run '
                '`gcloud auth login`.') from None
        # gcloud user access tokens normally live much longer than this. A
        # deliberately short local cache keeps long-running Temporal jobs
        # refreshable without relying on undocumented gcloud expiry metadata.
        # google-auth's refresh-handler contract currently expects naive UTC.
        expiry = (datetime.now(timezone.utc).replace(tzinfo=None) +
                  timedelta(minutes=5))
        return token, expiry

    try:
        project_result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project', '--quiet'],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        project = project_result.stdout.strip()
        if project_result.returncode != 0 or not project or project == '(unset)':
            project = None
    except Exception:
        project = None

    return UserCredentials(
        token=None,
        scopes=scopes,
        refresh_handler=refresh_handler,
    ), project


def _ambient_google_credentials(
        config: IdentityConfig) -> 'GoogleCloudCredentials':
    """Resolve the deployment's ambient Google credential.

    Workload deployments accept only a compute or external-account credential
    whose resolved service account is explicitly allowlisted.  A devserver may
    additionally use refreshable Google user Application Default Credentials,
    allowing the same ambient identity product flow to run against resources
    available to the local developer.  The latter is guarded independently by
    both an explicit devserver flag and the development deploy stage.
    """
    import google.auth
    from google.auth import compute_engine, external_account
    from google.oauth2 import credentials as oauth2_credentials

    allow_user_credentials = dev_gcp_user_credentials_enabled()
    policy = None
    try:
        policy = require_ambient_gcp_workload_config()
    except WorkloadExchangeConfigurationError:
        if not allow_user_credentials:
            raise

    scopes = [_GOOGLE_CLOUD_PLATFORM_SCOPE]
    try:
        source, project = google.auth.default(scopes=scopes)
    except Exception:
        if not allow_user_credentials:
            raise IdentityAuthenticationCategoryError(
                IdentityAuthenticationCategory.SOURCE_UNAVAILABLE) from None
        try:
            source, project = _gcloud_cli_user_credentials(scopes)
        except Exception:
            raise IdentityAuthenticationCategoryError(
                IdentityAuthenticationCategory.SOURCE_UNAVAILABLE) from None
    workload_source = isinstance(
        source, (external_account.Credentials, compute_engine.Credentials))
    user_source = isinstance(source, oauth2_credentials.Credentials)
    if not workload_source and not (allow_user_credentials and user_source):
        raise IdentityAuthenticationCategoryError(
            IdentityAuthenticationCategory.DISALLOWED_SOURCE)
    if workload_source and policy is None:
        raise WorkloadExchangeConfigurationError(
            'Ambient GCP workload identity is disabled')
    try:
        # Refreshing attests that either credential chain is currently usable.
        # For workload credentials it also populates the service-account email
        # used for the allowlist check below.
        source.refresh(Request())
    except Exception:
        raise IdentityAuthenticationCategoryError(
            IdentityAuthenticationCategory.SOURCE_UNAVAILABLE) from None
    if user_source:
        return GoogleCloudCredentials(
            provider=_google_provider(config),
            credential=source,
            project=project,
            ambient=True,
        )

    assert policy is not None
    email = getattr(source, 'service_account_email', None)
    if not isinstance(email, str) or '@' not in email:
        raise WorkloadExchangeConfigurationError(
            'Ambient GCP workload identity did not resolve a concrete '
            'service-account email')
    email = email.strip().lower()
    approved = {
        item.strip().lower()
        for item in policy.approved_service_account_emails
    }
    if email not in approved:
        raise WorkloadExchangeConfigurationError(
            'Ambient GCP service account is not approved for this deployment')
    return GoogleCloudCredentials(
        provider=_google_provider(config),
        credential=source,
        project=project,
        ambient=True,
    )


_AZURE_CLOUDS = {
    'public': (
        'dfs.core.windows.net',
        'https://storage.azure.com/.default',
        'AZURE_PUBLIC_CLOUD',
    ),
    'us_government': (
        'dfs.core.usgovcloudapi.net',
        'https://storage.azure.us/.default',
        'AZURE_GOVERNMENT',
    ),
    'china': (
        'dfs.core.chinacloudapi.cn',
        'https://storage.azure.cn/.default',
        'AZURE_CHINA',
    ),
}


def _azure_cloud_settings(cloud: str) -> tuple[str, str, str]:
    normalized = cloud.strip().casefold().replace('-', '_').replace(' ', '_')
    aliases = {
        'azurepublic': 'public',
        'azure_public': 'public',
        'government': 'us_government',
        'usgovernment': 'us_government',
        'azureusgovernment': 'us_government',
        'azure_us_government': 'us_government',
        'azurechina': 'china',
        'azure_china': 'china',
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return _AZURE_CLOUDS[normalized]
    except KeyError as error:
        raise ValueError(f'Unsupported Azure cloud: {cloud}') from error


class WorkloadIdentityExchanger(Protocol):
    def exchange(
        self,
        config: IdentityConfig,
        credentials: Optional[IdentityCredentials],
    ) -> Credentials:
        ...


def _require_exchange_enabled(provider: str) -> None:
    variable = f'KUMO_{provider.upper()}_WORKLOAD_EXCHANGE_ENABLED'
    if os.getenv(variable, '').lower() not in ('1', 'true', 'yes'):
        raise WorkloadExchangeConfigurationError(
            f'{provider.upper()} workload exchange is not configured; set '
            f'{variable}=true and configure the deployment workload issuer')


def ambient_adls_identity_config() -> Optional[ADLSIdentityConfig]:
    try:
        _require_exchange_enabled('azure')
    except WorkloadExchangeConfigurationError:
        return None

    tenant_id = os.getenv('AZURE_TENANT_ID', '').strip()
    client_id = os.getenv('AZURE_CLIENT_ID', '').strip()
    authority = os.getenv('AZURE_AUTHORITY_HOST', '').strip()
    missing = [
        name for name, value in (
            ('AZURE_TENANT_ID', tenant_id),
            ('AZURE_CLIENT_ID', client_id),
            ('AZURE_AUTHORITY_HOST', authority),
        ) if not value
    ]
    if missing:
        raise WorkloadExchangeConfigurationError(
            'Azure workload exchange is enabled but AKS did not inject '
            + ', '.join(missing))

    authority_host = urlsplit(
        authority if '://' in authority else f'https://{authority}').hostname
    clouds = {
        'login.microsoftonline.com': AzureCloud.PUBLIC,
        'login.microsoftonline.us': AzureCloud.US_GOVERNMENT,
        'login.chinacloudapi.cn': AzureCloud.CHINA,
    }
    try:
        cloud = clouds[authority_host or '']
    except KeyError:
        raise WorkloadExchangeConfigurationError(
            f'Unsupported Azure workload authority: {authority_host!r}'
        ) from None
    return ADLSIdentityConfig(
        tenant_id=tenant_id,
        client_id=client_id,
        cloud=cloud,
    )


def _google_provider(config: IdentityConfig) -> str:
    if isinstance(config, BigQueryIdentityConfig):
        return 'bigquery'
    if isinstance(config, GCSIdentityConfig):
        return 'gcs'
    raise TypeError('Identity is not a Google provider')


def _google_scope(config: IdentityConfig) -> str:
    return (_GOOGLE_CLOUD_PLATFORM_SCOPE if isinstance(
        config, BigQueryIdentityConfig) else _GOOGLE_GCS_SCOPE)


def _google_service_account_info(
    authentication: GoogleServiceAccountKeyAuthentication,
    credentials: GoogleServiceAccountKeyCredentials,
) -> dict[str, str]:
    canonical = {
        'auth_uri': _GOOGLE_SERVICE_ACCOUNT_AUTH_URI,
        'token_uri': _GOOGLE_SERVICE_ACCOUNT_TOKEN_URI,
        'universe_domain': _GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN,
    }
    for field_name, expected in canonical.items():
        if getattr(authentication, field_name).strip() != expected:
            raise ValueError(
                f'Google service-account {field_name} is not canonical')
    private_key_id = credentials.private_key_id.get_secret_value().strip()
    private_key = credentials.private_key.get_secret_value()
    if not private_key_id:
        raise ValueError('Google service-account key ID is required')
    try:
        serialization.load_pem_private_key(private_key.encode('utf-8'),
                                           password=None)
    except (TypeError, ValueError):
        raise ValueError(
            'Google service-account private key is not valid PEM') from None
    return {
        'type': 'service_account',
        'project_id': authentication.project_id,
        'private_key_id': private_key_id,
        'private_key': private_key,
        'client_email': authentication.client_email,
        'client_id': authentication.client_id,
        'auth_uri': authentication.auth_uri,
        'token_uri': authentication.token_uri,
        'universe_domain': authentication.universe_domain,
    }


def validate_google_key_material(
    authentication: GoogleServiceAccountKeyAuthentication,
    credentials: GoogleServiceAccountKeyCredentials,
) -> None:
    """Validate the complete decomposed key without exposing secret detail."""
    info = _google_service_account_info(authentication, credentials)
    service_account.Credentials.from_service_account_info(
        info,
        scopes=[_GOOGLE_CLOUD_PLATFORM_SCOPE],
    )


def _validate_s3_access_key(
    config: S3IdentityConfig,
    submitted: S3AccessKeyCredentials,
    runtime: AWSAccessKeyCredentials,
) -> None:
    """Authenticate one bounded request without requiring bucket access."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    connection_config = config.connection
    endpoint_url = (connection_config.endpoint_url
                    if connection_config else None)
    connection = S3Connection(
        endpoint_url=endpoint_url,
        region=((connection_config.region if connection_config else None)
                or runtime.region),
        addressing_style=(connection_config.addressing_style.value
                          if connection_config else 'auto'),
        credential_secret_name='',
        access_key=runtime.access_key_id.get_secret_value(),
        secret_key=runtime.secret_access_key.get_secret_value(),
        session_token=(runtime.session_token.get_secret_value()
                       if runtime.session_token else None),
        ca_bundle_path=None,
        send_bucket_owner_full_control=endpoint_url is None,
        owner='identity-validation',
        ca_certificate=(submitted.ca_certificate.get_secret_value()
                        if submitted.ca_certificate else None),
    )
    request_config = connection.boto_config.merge(
        Config(
            connect_timeout=3,
            read_timeout=3,
            retries={
                'mode': 'standard',
                'total_max_attempts': 1,
            },
        ))
    try:
        client = boto3.client(
            's3',
            aws_access_key_id=connection.access_key,
            aws_secret_access_key=connection.secret_key,
            aws_session_token=connection.session_token,
            endpoint_url=connection.endpoint_url,
            region_name=connection.region,
            verify=connection.boto_verify,
            config=request_config,
        )
        client.list_buckets()
    except ClientError as error:
        code = error.response.get('Error', {}).get('Code')
        # A recognized principal may lack the account-wide ListAllMyBuckets
        # permission. Invalid keys, signatures, regions, and tokens use
        # distinct error codes and must fail validation.
        if code == 'AccessDenied':
            return
        raise IdentityAuthenticationCategoryError(
            IdentityAuthenticationCategory.S3_VALIDATION_FAILED) from None
    except Exception:
        # Endpoint, TLS/CA, timeout, and transport errors are intentionally
        # collapsed so provider responses cannot disclose submitted secrets.
        raise IdentityAuthenticationCategoryError(
            IdentityAuthenticationCategory.S3_VALIDATION_FAILED) from None


class NativeWorkloadIdentityExchanger:
    """Exchange the explicitly configured deployment workload identity."""
    def __init__(self) -> None:
        self._ambient_aws_lock = threading.Lock()
        self._ambient_aws_cache: Optional[tuple[
            tuple[str, ...],
            float,
            AWSAmbientCredentials,
        ]] = None

    def _cached_ambient_aws_credentials(self) -> AWSAmbientCredentials:
        policy = require_ambient_s3_workload_config()
        key = tuple(policy.approved_role_arns)
        now = time.monotonic()
        cached = self._ambient_aws_cache
        if cached is not None and cached[0] == key and now < cached[1]:
            return cached[2]

        with self._ambient_aws_lock:
            now = time.monotonic()
            cached = self._ambient_aws_cache
            if cached is not None and cached[0] == key and now < cached[1]:
                return cached[2]
            credentials = _ambient_aws_credentials(key)
            self._ambient_aws_cache = (
                key,
                time.monotonic() + _AMBIENT_AWS_ATTESTATION_TTL_SECONDS,
                credentials,
            )
            return credentials

    def exchange(
        self,
        config: IdentityConfig,
        credentials: Optional[IdentityCredentials],
    ) -> Credentials:
        if isinstance(config, S3IdentityConfig):
            _validate_s3_connection_policy(config)
            if isinstance(config.authentication, S3AmbientAuthentication):
                if credentials is not None:
                    raise WorkloadExchangeConfigurationError(
                        'Ambient S3 identities cannot store credentials')
                return self._cached_ambient_aws_credentials()
            if isinstance(config.authentication, S3AccessKeyAuthentication):
                if not isinstance(credentials, S3AccessKeyCredentials):
                    raise WorkloadExchangeConfigurationError(
                        'S3 access-key credentials are required')
                return AWSAccessKeyCredentials(
                    access_key_id=credentials.access_key_id,
                    secret_access_key=credentials.secret_access_key,
                    session_token=credentials.session_token,
                    region=(config.connection.region
                            if config.connection else None),
                )
            if not isinstance(config.authentication,
                              S3AssumeRoleAuthentication):
                raise TypeError('Unsupported S3 identity authentication mode')
            _require_exchange_enabled('aws')
            import boto3

            session = boto3.session.Session()
            source = session.get_credentials()
            source_method = getattr(source, 'method', None)
            if source is None or source_method not in {
                    'assume-role-with-web-identity',
                    'container-role',
                    'iam-role',
            }:
                raise WorkloadExchangeConfigurationError(
                    'AWS workload exchange requires an instance, container, '
                    'or web-identity role as its source; static keys and '
                    'developer credentials are not permitted')
            args = {
                'RoleArn': config.authentication.role_arn,
                'RoleSessionName': 'kumo-project-connector',
            }
            if isinstance(credentials, S3AssumeRoleCredentials):
                if credentials.external_id:
                    args[
                        'ExternalId'] = credentials.external_id.get_secret_value(
                        )
            response = session.client('sts').assume_role(**args)
            exchanged = response['Credentials']
            return AWSFederatedCredentials(
                access_key_id=SecretStr(exchanged['AccessKeyId']),
                secret_access_key=SecretStr(exchanged['SecretAccessKey']),
                session_token=SecretStr(exchanged['SessionToken']),
            )
        if isinstance(config, (BigQueryIdentityConfig, GCSIdentityConfig)):
            authentication = config.authentication
            if isinstance(authentication, GoogleAmbientAuthentication):
                if credentials is not None:
                    raise WorkloadExchangeConfigurationError(
                        'Google ambient identities cannot store credentials')
                # Ambient GCP: use the deployment's own Workload Identity
                # service account DIRECTLY, with no impersonation and no key.
                return _ambient_google_credentials(config)
            if not isinstance(authentication,
                              GoogleWorkloadIdentityAuthentication):
                raise TypeError(
                    'Google key identities do not use workload exchange')
            if credentials is not None:
                raise WorkloadExchangeConfigurationError(
                    'Google workload identities cannot store credentials')
            try:
                _require_exchange_enabled('gcp')
            except WorkloadExchangeConfigurationError:
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.EXCHANGE_DISABLED) from None
            import google.auth
            from google.auth import (
                compute_engine,
                external_account,
                impersonated_credentials,
            )

            try:
                source, project = google.auth.default(
                    scopes=[_GOOGLE_CLOUD_PLATFORM_SCOPE])
            except Exception:
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.SOURCE_UNAVAILABLE) from None
            source_types = (
                external_account.Credentials,
                compute_engine.Credentials,
            )
            if not isinstance(source, source_types):
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.DISALLOWED_SOURCE)
            try:
                exchanged = impersonated_credentials.Credentials(
                    source_credentials=source,
                    target_principal=authentication.service_account_email,
                    target_scopes=[_google_scope(config)],
                )
                exchanged.refresh(Request())
            except Exception:
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.IMPERSONATION_DENIED
                ) from None
            return GoogleCloudCredentials(
                provider=_google_provider(config),
                credential=exchanged,
                project=project,
            )
        if isinstance(config, ADLSIdentityConfig):
            _require_exchange_enabled('azure')
            from azure.identity import (
                AzureAuthorityHosts,
                WorkloadIdentityCredential,
            )

            token_file_path = os.getenv('AZURE_FEDERATED_TOKEN_FILE')
            if not token_file_path:
                raise WorkloadExchangeConfigurationError(
                    'Azure workload exchange requires '
                    'AZURE_FEDERATED_TOKEN_FILE')
            endpoint_suffix, token_scope, authority_name = (
                _azure_cloud_settings(config.cloud))
            exchanged = WorkloadIdentityCredential(
                tenant_id=config.tenant_id,
                client_id=config.client_id,
                token_file_path=token_file_path,
                authority=getattr(AzureAuthorityHosts, authority_name),
            )
            exchanged.get_token(token_scope)
            return AzureFederatedCredentials(exchanged,
                                             endpoint_suffix=endpoint_suffix)
        raise TypeError(
            f'Identity {type(config)} does not use workload exchange')


def _normalized_host(host: str) -> str:
    value = host.strip()
    if '://' not in value:
        value = f'https://{value}'
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                       parsed.path.rstrip('/'), '', ''))


def _validate_s3_connection_policy(config: S3IdentityConfig) -> None:
    connection = config.connection
    if connection is None or connection.endpoint_url is None:
        return
    if (urlsplit(connection.endpoint_url).scheme == 'http'
            and deploy_stage() not in {
                DeployStage.DEV,
                DeployStage.DEV_LOCAL,
                DeployStage.TEST,
            }):
        raise WorkloadExchangeConfigurationError(
            'S3-compatible endpoints must use HTTPS in this deployment')


def identity_key(
    config: IdentityConfig,
    credentials: Optional[IdentityCredentials] = None,
) -> str:
    if isinstance(config, DatabricksIdentityConfig):
        return (f'databricks:{_normalized_host(config.host)}:'
                f'{config.client_id.strip().lower()}')
    if isinstance(config, SnowflakeIdentityConfig):
        return (f'snowflake:{config.account.strip().lower()}:'
                f'{config.username.strip().upper()}:'
                f'{config.authentication.value}')
    if isinstance(config, BigQueryIdentityConfig):
        authentication = config.authentication
        if isinstance(authentication, GoogleServiceAccountKeyAuthentication):
            return ('bigquery:service_account_key:'
                    f'{authentication.client_email.strip().lower()}')
        if isinstance(authentication, GoogleWorkloadIdentityAuthentication):
            return ('bigquery:workload_identity:'
                    f'{authentication.service_account_email.strip().lower()}')
        if isinstance(authentication, GoogleAmbientAuthentication):
            return 'bigquery:ambient'
        raise TypeError('Unsupported BigQuery authentication mode')
    if isinstance(config, S3IdentityConfig):
        if isinstance(config.authentication, S3AmbientAuthentication):
            return 's3:ambient'
        if isinstance(config.authentication, S3AssumeRoleAuthentication):
            return (f's3:{config.authentication.partition.strip().lower()}:'
                    f'{config.authentication.role_arn.strip().lower()}')
        if isinstance(config.authentication, S3AccessKeyAuthentication):
            if not isinstance(credentials, S3AccessKeyCredentials):
                raise ValueError(
                    'S3 access-key credentials are required for identity key')
            endpoint_url = (config.connection.endpoint_url
                            if config.connection
                            and config.connection.endpoint_url else '')
            access_key_id = credentials.access_key_id.get_secret_value().strip()
            digest = CredentialsEncryptor().s3_access_key_fingerprint(
                endpoint_url,
                access_key_id,
            )
            return f's3:access_key:{digest}'
        raise TypeError('Unsupported S3 identity authentication mode')
    if isinstance(config, GCSIdentityConfig):
        authentication = config.authentication
        if isinstance(authentication, GoogleServiceAccountKeyAuthentication):
            return ('gcs:service_account_key:'
                    f'{authentication.client_email.strip().lower()}')
        if isinstance(authentication, GoogleWorkloadIdentityAuthentication):
            return ('gcs:workload_identity:'
                    f'{authentication.service_account_email.strip().lower()}')
        if isinstance(authentication, GoogleAmbientAuthentication):
            return 'gcs:ambient'
        raise TypeError('Unsupported GCS authentication mode')
    if isinstance(config, ADLSIdentityConfig):
        return (f'adls:{config.tenant_id.strip().lower()}:'
                f'{config.client_id.strip().lower()}:'
                f'{config.cloud.strip().lower()}')
    raise TypeError(f'Unsupported identity config: {type(config)}')


def credentials_for_identity(
    config: IdentityConfig,
    credentials: Optional[IdentityCredentials],
    exchanger: WorkloadIdentityExchanger,
) -> Credentials:
    if (isinstance(config, DatabricksIdentityConfig)
            and isinstance(credentials, DatabricksOAuthCredentials)):
        return DatabricksCredentials(
            client_id=SecretStr(config.client_id),
            client_secret=credentials.client_secret,
            pat=SecretStr(''),
        )
    if isinstance(config, SnowflakeIdentityConfig):
        if (config.authentication == SnowflakeAuthenticationKind.PASSWORD
                and isinstance(credentials, SnowflakePasswordCredentials)):
            return UsernamePassword(
                username=SecretStr(config.username),
                password=credentials.password,
            )
        if (config.authentication == SnowflakeAuthenticationKind.KEY_PAIR
                and isinstance(credentials, SnowflakeKeyPairCredentials)):
            return KeyPair(
                user=SecretStr(config.username),
                private_key=credentials.private_key,
                private_key_passphrase=credentials.private_key_passphrase,
            )
    if isinstance(config, (BigQueryIdentityConfig, GCSIdentityConfig)):
        authentication = config.authentication
        if (isinstance(authentication, GoogleServiceAccountKeyAuthentication)
                and isinstance(credentials,
                               GoogleServiceAccountKeyCredentials)):
            try:
                info = _google_service_account_info(authentication, credentials)
                google_credentials = (
                    service_account.Credentials.from_service_account_info(
                        info,
                        scopes=[_google_scope(config)],
                    ))
            except Exception:
                raise IdentityAuthenticationCategoryError(
                    IdentityAuthenticationCategory.STORED_KEY_INVALID) from None
            return GoogleCloudCredentials(
                provider=_google_provider(config),
                credential=google_credentials,
                project=authentication.project_id,
            )
        if (isinstance(
                authentication,
            (GoogleWorkloadIdentityAuthentication, GoogleAmbientAuthentication))
                and credentials is None):
            return exchanger.exchange(config, credentials)
        raise ValueError('Google identity config and credentials do not match')
    if isinstance(config, (S3IdentityConfig, ADLSIdentityConfig)):
        return exchanger.exchange(config, credentials)
    raise ValueError('Identity config and credentials do not match')


def compose_resolved_connector_spec(
    identity: IdentityDBEntry,
    connector: ExternalConnectorDBEntry,
    credentials: Optional[IdentityCredentials],
    exchanger: WorkloadIdentityExchanger,
) -> ResolvedConnectorSpec:
    if connector.identity_id != identity.id:
        raise ValueError('Connector does not reference the supplied identity')
    if connector.config.type.value != identity.config.type.value:
        raise ValueError('Connector and identity types do not match')
    if str(connector.id) != connector.config.name:
        raise ValueError('Connector ID and runtime config name must match')

    runtime_credentials = credentials_for_identity(identity.config, credentials,
                                                   exchanger)
    if isinstance(runtime_credentials,
                  (AWSFederatedCredentials, AWSAmbientCredentials,
                   AWSAccessKeyCredentials)):
        runtime_credentials = replace(
            runtime_credentials,
            region=getattr(connector.config, 'region', None),
        )
        connection_config = (identity.config.connection if isinstance(
            identity.config, S3IdentityConfig) else None)
        endpoint_url = (connection_config.endpoint_url
                        if connection_config else None)
        region = ((connection_config.region if connection_config else None)
                  or runtime_credentials.region)
        if connection_config is not None and isinstance(
                runtime_credentials,
                (AWSFederatedCredentials, AWSAccessKeyCredentials)):
            access_key = runtime_credentials.access_key_id.get_secret_value()
            secret_key = (
                runtime_credentials.secret_access_key.get_secret_value())
            session_token = (runtime_credentials.session_token.get_secret_value()
                             if runtime_credentials.session_token else None)
            ca_certificate = None
            if isinstance(credentials, S3AccessKeyCredentials):
                ca_certificate = (
                    credentials.ca_certificate.get_secret_value()
                    if credentials.ca_certificate else None)
            resolved_connection = S3Connection(
                endpoint_url=endpoint_url,
                region=region,
                addressing_style=(connection_config.addressing_style.value
                                  if connection_config else 'auto'),
                credential_secret_name='',
                access_key=access_key,
                secret_key=secret_key,
                session_token=session_token,
                ca_bundle_path=None,
                send_bucket_owner_full_control=endpoint_url is None,
                owner=f'connector:{identity.id}',
                rotation_id=(identity.credentials_rotated_at.isoformat()
                             if identity.credentials_rotated_at else 'static'),
                ca_certificate=ca_certificate,
            )
            runtime_credentials = replace(
                runtime_credentials,
                connection=resolved_connection,
            )
    elif isinstance(runtime_credentials, GoogleCloudCredentials):
        if runtime_credentials.provider == 'gcs':
            runtime_credentials = replace(
                runtime_credentials,
                requester_pays_project=getattr(connector.config,
                                               'requester_pays_project', None),
            )
        elif runtime_credentials.provider == 'bigquery':
            runtime_credentials = replace(
                runtime_credentials,
                billing_project_id=getattr(connector.config,
                                           'billing_project_id', None),
            )
    elif isinstance(identity.config, ADLSIdentityConfig):
        expected_endpoint, _, _ = _azure_cloud_settings(identity.config.cloud)
        target = getattr(connector.config, 'target', None)
        actual_endpoint = getattr(target, 'endpoint_suffix', None)
        if (actual_endpoint is not None
                and actual_endpoint != expected_endpoint):
            raise ValueError(
                f'ADLS endpoint {actual_endpoint!r} does not match identity '
                f'cloud {identity.config.cloud!r}')
    if isinstance(identity.config, DatabricksIdentityConfig):
        native_connector = connector.config.as_connector(
            identity.config.host,
            require_explicit_credentials=True,
        )
    elif isinstance(identity.config, SnowflakeIdentityConfig):
        native_connector = connector.config.as_connector(
            identity.config.account,
            require_explicit_credentials=True,
        )
    else:
        native_connector = connector.config.as_connector(
            require_explicit_credentials=True)
    native_connector.set_credentials(runtime_credentials.as_dict())
    return ResolvedConnectorSpec(
        connector_id=connector.id,
        connector_config=native_connector.config(),
        credentials=runtime_credentials,
    )


def validate_identity_authentication(
    config: IdentityConfig,
    credentials: Optional[IdentityCredentials],
    exchanger: WorkloadIdentityExchanger,
) -> None:
    runtime_credentials = credentials_for_identity(config, credentials,
                                                   exchanger)
    if isinstance(config, S3IdentityConfig):
        if isinstance(config.authentication, S3AccessKeyAuthentication):
            assert isinstance(credentials, S3AccessKeyCredentials)
            assert isinstance(runtime_credentials, AWSAccessKeyCredentials)
            _validate_s3_access_key(config, credentials, runtime_credentials)
        # Assume-role exchange and ambient-role attestation already make a
        # bounded STS request while obtaining the runtime credentials.
        return
    if isinstance(config, DatabricksIdentityConfig):
        from kumo.connector.databricks_connector import (
            validate_databricks_auth,
        )
        validate_databricks_auth(config.host, runtime_credentials.as_dict())
        return
    if isinstance(config, SnowflakeIdentityConfig):
        from kumo.connector import SnowflakeConnector
        from kumo.connector.snowflake_connector import (
            get_snowflake_connection_string,
        )

        connector = SnowflakeConnector(
            name='identity_validation',
            account=config.account,
            warehouse='',
            database='',
            schema_name='',
        )
        connection_args = connector.get_user_pass_or_key(
            'Snowflake identity',
            'SNOWFLAKE_USER',
            'SNOWFLAKE_PASSWORD',
            runtime_credentials.as_dict(),
            key_encoding=serialization.Encoding.DER,
            env_var_private_key='SNOWFLAKE_PRIVATE_KEY',
        )
        connection = connect(
            **connection_args,
            account=config.account,
            login_timeout=5,
            application=get_snowflake_connection_string(),
        )
        connection.close()
        return
    if isinstance(config, (BigQueryIdentityConfig, GCSIdentityConfig)):
        assert isinstance(runtime_credentials, GoogleCloudCredentials)
        if isinstance(config.authentication,
                      (GoogleWorkloadIdentityAuthentication,
                       GoogleAmbientAuthentication)):
            # The workload exchanger refreshes the impersonated (or ambient)
            # credential as part of exchange, which both validates it and
            # obtains the provider token. Do not perform a second provider
            # request here.
            return
        try:
            runtime_credentials.credential.refresh(Request())
        except Exception:
            raise IdentityAuthenticationCategoryError(
                IdentityAuthenticationCategory.STORED_KEY_INVALID) from None
        return
    # Workload exchange already obtains and validates a provider token.


def validate_resolved_connector(spec: ResolvedConnectorSpec) -> None:
    from kumo.connector.composite_file_connector import CompositeFileConnector
    from kumo.connector.connector import Connector

    connector = Connector.from_config(spec.connector_config)
    if spec.credentials is None:
        raise ValueError('External connector credentials are required')
    connector.require_explicit_credentials()
    connector.set_credentials(spec.credentials.as_dict())
    if isinstance(connector, CompositeFileConnector):
        # Object-store prefixes are virtual and may legitimately contain no
        # objects, so FileConnector._open()'s existence check is not a valid
        # permission probe.  Listing the exact configured root is both the
        # minimum read check and sufficient for an empty prefix.  Do this once
        # and never create a marker/probe object.
        connector.validate_permissions()
        return
    try:
        connector.open()
    finally:
        connector.close()
