# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

from async_utils.sync import blocking_func
from io_utils import logging
from monitoring.tracing import TraceOptions, trace
from service_framework._raw_deploy_config import SecretBackend
from service_framework.deploy_env_config import deploy_env

logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions - unified error types for all cloud providers
# ============================================================================


class SecretException(Exception):
    """Base exception for secret operations."""


class SecretNotFoundException(SecretException):
    """Secret does not exist in the secret manager."""


class SecretAccessException(SecretException):
    """Failed to access secret manager (auth, network, permissions, etc.)."""


def _parse_secret_string(name: str, raw: str) -> Dict[str, str]:
    """Parse a raw secret string into a dict.

    If the string is JSON, parse it. Otherwise wrap as ``{name: raw}``.
    """
    if raw.startswith('{'):
        return json.loads(raw)
    return {name: raw}


class SecretProvider(ABC):
    """Abstract interface for secret management across cloud providers."""
    @abstractmethod
    def get_secret(self, name: str) -> Dict[str, str]:
        """Fetch a secret by name.

        Args:
            name: The secret name/identifier.

        Returns:
            Dictionary containing the secret key-value pairs.
        """

    @abstractmethod
    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        """Store a secret.

        Args:
            name: The secret name/identifier.
            value: Dictionary of key-value pairs to store.

        Returns:
            Response from the secret manager.
        """


class AWSSecretProvider(SecretProvider):
    """AWS Secrets Manager implementation."""
    def __init__(self):
        from service_framework.aws_region import AWS_ACCOUNT_ID, AWS_REGION
        self.region = AWS_REGION
        self.account_id = AWS_ACCOUNT_ID

    def _to_arn(self, name: str) -> str:
        """Pass-through. Bare names resolve in the calling AWS account;
        explicit ARNs are honored as-is."""
        return name

    def get_secret(self, name: str) -> Dict[str, str]:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        # Check env var override first
        secret = os.getenv(name)

        if secret is None:
            logger.debug('Fetching secret %s from AWS Secrets Manager', name)
            try:
                client = boto3.client(service_name='secretsmanager',
                                      region_name=self.region)

                secret_arn = self._to_arn(name)
                get_secret_value_response = client.get_secret_value(
                    SecretId=secret_arn)

                if get_secret_value_response['SecretString'] is None:
                    raise SecretAccessException(
                        'AWS Secrets Manager response did not include '
                        'SecretString')
                secret = get_secret_value_response['SecretString']
                logger.debug('Successfully fetched secret %s from AWS', name)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ResourceNotFoundException':
                    raise SecretNotFoundException(
                        f'Secret {name} not found in AWS') from None
                raise SecretAccessException(
                    f'Unable to access AWS secret {name}. Verify Secrets '
                    'Manager permissions and credentials.') from None
            except NoCredentialsError:
                raise SecretAccessException(
                    'AWS credentials are unavailable for secret access.'
                ) from None
        else:
            logger.debug('Using env var override for secret %s', name)

        return _parse_secret_string(name, secret)

    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        logger.debug('Storing secret %s in AWS Secrets Manager', name)
        secret_arn = self._to_arn(name)
        secret_string = json.dumps(value)

        try:
            client = boto3.client(service_name='secretsmanager',
                                  region_name=self.region)

            response = client.put_secret_value(SecretId=secret_arn,
                                               SecretString=secret_string)

            if response['Name'] != name:
                raise SecretAccessException(
                    f'put_secret response returned invalid name: '
                    f'{response["Name"]} (requested: {name})')
            logger.debug('Successfully stored secret %s in AWS', name)
            return response
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFoundException':
                raise SecretNotFoundException(
                    f'Secret {name} not found in AWS') from None
            raise SecretAccessException(
                f'Unable to store AWS secret {name}. Verify Secrets Manager '
                'permissions and credentials.') from None
        except NoCredentialsError:
            raise SecretAccessException(
                'AWS credentials are unavailable for secret storage.') from None


class GCPSecretProvider(SecretProvider):
    """Google Cloud Secret Manager implementation."""
    def __init__(self, project_id: Optional[str] = None):
        """Initialize GCP Secret Provider.

        Args:
            project_id: GCP project ID. If None, uses default from environment.
        """
        self.project_id = project_id or os.environ.get('GCP_PROJECT_ID')
        if not self.project_id:
            # Try to get from google.auth (works with ADC and Workload Identity)
            try:
                import google.auth
                _, self.project_id = google.auth.default()
            except Exception:
                pass
        if not self.project_id:
            # Fallback: query GCP metadata server (works on GKE/GCE/Cloud Run)
            try:
                import urllib.request
                req = urllib.request.Request(
                    'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                    headers={'Metadata-Flavor': 'Google'})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    self.project_id = resp.read().decode('utf-8')
            except Exception:
                pass
        if not self.project_id:
            raise RuntimeError(
                'Could not determine GCP project ID. Set GCP_PROJECT_ID env var '
                'or ensure Application Default Credentials are configured.')

    def _get_client(self):
        """Get Secret Manager client."""
        from google.cloud import secretmanager
        return secretmanager.SecretManagerServiceClient()

    def _get_secret_path(self, name: str, version: str = 'latest') -> str:
        """Build the full secret resource path."""
        return f'projects/{self.project_id}/secrets/{name}/versions/{version}'

    def get_secret(self, name: str) -> Dict[str, str]:
        from google.api_core.exceptions import GoogleAPIError, NotFound

        # Check env var override first
        secret = os.getenv(name)

        if secret is None:
            logger.debug(
                'Fetching secret %s from GCP Secret Manager '
                '(project: %s)', name, self.project_id)
            try:
                client = self._get_client()
                secret_path = self._get_secret_path(name)
                response = client.access_secret_version(
                    request={'name': secret_path})
                secret = response.payload.data.decode('UTF-8')
                logger.debug('Successfully fetched secret %s from GCP', name)
            except NotFound:
                raise SecretNotFoundException(
                    f'Secret {name} not found in GCP') from None
            except GoogleAPIError:
                raise SecretAccessException(
                    f'Unable to access GCP secret {name}. Verify Secret '
                    'Manager permissions and credentials.') from None
            except Exception:
                raise SecretAccessException(
                    f'Unable to access GCP secret {name}. Verify Secret '
                    'Manager configuration.') from None
        else:
            logger.debug('Using env var override for secret %s', name)

        return _parse_secret_string(name, secret)

    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        from google.api_core.exceptions import GoogleAPIError, NotFound

        logger.debug('Storing secret %s in GCP Secret Manager '
                     '(project: %s)', name, self.project_id)
        client = self._get_client()
        parent = f'projects/{self.project_id}'
        secret_string = json.dumps(value)

        try:
            # Check if secret exists, create if not
            try:
                client.get_secret(request={'name': f'{parent}/secrets/{name}'})
            except NotFound:
                # Secret doesn't exist, create it
                logger.debug('Secret %s does not exist, creating it', name)
                client.create_secret(
                    request={
                        'parent': parent,
                        'secret_id': name,
                        'secret': {
                            'replication': {
                                'automatic': {}
                            }
                        },
                    })

            # Add new version
            response = client.add_secret_version(
                request={
                    'parent': f'{parent}/secrets/{name}',
                    'payload': {
                        'data': secret_string.encode('UTF-8')
                    },
                })

            logger.debug('Successfully stored secret %s in GCP', name)
            return {'name': name, 'version': response.name}
        except GoogleAPIError:
            raise SecretAccessException(
                f'Unable to store GCP secret {name}. Verify Secret Manager '
                'permissions and credentials.') from None
        except Exception:
            raise SecretAccessException(
                f'Unable to store GCP secret {name}. Verify Secret Manager '
                'configuration.') from None


class EnvVarSecretProvider(SecretProvider):
    """Read secrets from environment variables only.

    Used in environments where secrets are injected as env vars by the
    platform (e.g. a host that injects Snowflake credentials as env vars).
    """
    def get_secret(self, name: str) -> Dict[str, str]:
        secret = os.getenv(name)
        if secret is None:
            raise SecretNotFoundException(
                f'Secret {name} not found in environment variables')
        return _parse_secret_string(name, secret)

    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        raise SecretAccessException(
            'EnvVarSecretProvider is read-only; cannot store secrets')


_DEFAULT_SECRET_MOUNT_PATH = '/var/run/secrets/kumo'


class LocalFileSecretProvider(SecretProvider):
    """Read secrets from files mounted by Kubernetes.

    When external-secrets-operator (or native k8s Secret volumes) mounts
    secrets, each key becomes a file under the mount path.  For example,
    a secret named ``db-password`` is read from
    ``{base_path}/db-password``.
    """
    def __init__(self) -> None:
        cfg = deploy_env().secrets
        self._base_path = cfg.mount_path or _DEFAULT_SECRET_MOUNT_PATH

    def get_secret(self, name: str) -> Dict[str, str]:
        # Check env var override first (consistent with other providers)
        secret = os.getenv(name)
        if secret is not None:
            logger.debug('Using env var override for secret %s', name)
            return _parse_secret_string(name, secret)

        secret_path = os.path.join(self._base_path, name)
        logger.debug('Reading secret %s from %s', name, secret_path)
        try:
            with open(secret_path) as f:
                secret = f.read()
        except FileNotFoundError:
            raise SecretNotFoundException(
                f'Secret {name} not found at {secret_path}') from None
        except OSError:
            raise SecretAccessException(
                f'Unable to read secret file {secret_path}.') from None
        return _parse_secret_string(name, secret)

    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        raise SecretAccessException(
            'LocalFileSecretProvider is read-only; cannot store secrets')


class ChainSecretProvider(SecretProvider):
    """Composite provider that tries a chain of backends in order.

    On ``get_secret``, tries each provider in sequence.  If a provider
    raises ``SecretNotFoundException``, the next provider is tried.
    Other exceptions (``SecretAccessException``, network errors, etc.)
    propagate immediately — they indicate a reachable-but-broken backend,
    not a missing secret.

    ``put_secret`` always delegates to the primary (first) provider.
    Fallbacks are read-only by design — if the primary is read-only
    (e.g. ENV_VAR), writes will raise ``SecretAccessException``.
    """
    def __init__(self, providers: Sequence[SecretProvider]) -> None:
        if not providers:
            raise ValueError(
                'ChainSecretProvider requires at least one provider')
        self._providers = list(providers)

    def get_secret(self, name: str) -> Dict[str, str]:
        last_exc: Optional[SecretNotFoundException] = None
        for provider in self._providers:
            try:
                return provider.get_secret(name)
            except SecretNotFoundException as e:
                logger.debug('Secret %s not found in %s, trying next provider',
                             name,
                             type(provider).__name__)
                last_exc = e
        raise last_exc  # type: ignore[misc]

    def put_secret(self, name: str, value: Dict[str, str]) -> Dict:
        return self._providers[0].put_secret(name, value)


# Provider singleton
_secret_provider: Optional[SecretProvider] = None

_PROVIDER_FACTORY: Dict[SecretBackend, type] = {
    SecretBackend.AWS_SM: AWSSecretProvider,
    SecretBackend.GCP_SM: GCPSecretProvider,
    SecretBackend.ENV_VAR: EnvVarSecretProvider,
    SecretBackend.K8S_MOUNT: LocalFileSecretProvider,
}


def _build_secret_provider(backend: SecretBackend) -> SecretProvider:
    """Instantiate a single SecretProvider for the given backend."""
    factory = _PROVIDER_FACTORY.get(backend)
    if factory is None:
        raise ValueError(f'No secret provider for backend: {backend}')
    return factory()


def _get_provider() -> SecretProvider:
    """Get the secret provider chain based on the v2 deploy config."""
    global _secret_provider
    if _secret_provider is None:
        secrets_cfg = deploy_env().secrets
        primary = _build_secret_provider(secrets_cfg.backend)
        fallbacks = [_build_secret_provider(fb) for fb in secrets_cfg.fallbacks]
        chain = [primary] + fallbacks
        logger.info('Secret provider chain: %s',
                    ' -> '.join(type(p).__name__ for p in chain))
        if len(chain) == 1:
            _secret_provider = primary
        else:
            _secret_provider = ChainSecretProvider(chain)
    return _secret_provider


# ============================================================================
# Public API - cloud-agnostic functions
# ============================================================================


# NOTE: Can't mark @blocking_func because we need to get secrets when we build
# the REST API app. Caller is responsible for marking itself as @blocking
@trace(TraceOptions(trace_params=True, record_exceptions=False))
def get_secrets(name: str) -> Dict[str, str]:
    """Fetch a secret by name.

    Selects the secret provider based on ``SecretBackend`` in the v2
    deploy config (AWS_SM, GCP_SM, ENV_VAR, or K8S_MOUNT).

    Args:
        name: The secret name/identifier.

    Returns:
        Dictionary containing the secret key-value pairs.
    """
    return _get_provider().get_secret(name)


@blocking_func(tracing_options=TraceOptions(record_exceptions=False))
def put_secrets(name: str, value: Dict[str, str]) -> Dict:
    """Store a secret.

    Selects the secret provider based on ``SecretBackend`` in the v2
    deploy config. Note: ENV_VAR backend is read-only.

    Args:
        name: The secret name/identifier.
        value: Dictionary of key-value pairs to store.

    Returns:
        Response from the secret manager.
    """
    return _get_provider().put_secret(name, value)
