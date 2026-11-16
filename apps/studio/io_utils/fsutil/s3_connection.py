# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolved S3-compatible connections with strict owner boundaries."""

import atexit
import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping, Optional, TypedDict

from botocore.config import Config
from kumo_s3.session import arrow_connection_options

from io_utils.secrets import get_secrets
from service_framework._raw_deploy_config import S3AddressingStyle
from service_framework.deploy_env_config import deploy_env

AWS_WORKLOAD_CREDENTIAL_METHODS = frozenset({
    'assume-role-with-web-identity',
    'container-role',
    'iam-role',
})

# Environment selectors that let an AWS SDK child reconstruct the renewable
# workload provider already attested by the control plane. Instance-role auth
# has no positive selector; it works when metadata access is left enabled.
AWS_WORKLOAD_CREDENTIAL_ENVIRONMENT_KEYS = frozenset({
    'AWS_WEB_IDENTITY_TOKEN_FILE',
    'AWS_ROLE_ARN',
    'AWS_ROLE_SESSION_NAME',
    'AWS_CONTAINER_CREDENTIALS_FULL_URI',
    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE',
})

# The Rust SDK checks shared AWS files before workload providers, while boto3
# checks web identity before shared files. Point both file inputs at an empty
# file so the child cannot select a profile that was not the attested provider.
AWS_WORKLOAD_CREDENTIAL_PROFILE_ISOLATION_ENVIRONMENT = {
    'AWS_CONFIG_FILE': os.devnull,
    'AWS_SHARED_CREDENTIALS_FILE': os.devnull,
}


@dataclass(frozen=True, repr=False)
class S3Connection:
    endpoint_url: Optional[str]
    region: Optional[str]
    addressing_style: S3AddressingStyle | str
    credential_secret_name: str
    access_key: str
    secret_key: str
    session_token: Optional[str]
    ca_bundle_path: Optional[str]
    send_bucket_owner_full_control: bool
    owner: str = 'deployment'
    rotation_id: str = 'startup'
    ca_certificate: Optional[str] = None

    @property
    def identity(self) -> tuple[str, str, str, str, str, Optional[str], bool]:
        """Non-secret cache identity, including owner and rotation."""
        return (
            self.owner,
            self.rotation_id,
            self.endpoint_url or '',
            self.region or '',
            _addressing_style_value(self.addressing_style),
            self.credential_secret_name,
            self.send_bucket_owner_full_control,
        )

    @property
    def boto_config(self) -> Config:
        return Config(
            signature_version='s3v4',
            s3={
                'addressing_style':
                _addressing_style_value(self.addressing_style)
            },
        )

    @property
    def boto_verify(self) -> bool | str:
        return self.ca_path or True

    @property
    def ca_path(self) -> Optional[str]:
        if self.ca_certificate is not None:
            return _materialize_ca(self.owner, self.rotation_id,
                                   self.ca_certificate)
        return self.ca_bundle_path

    def pyarrow_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            'access_key': self.access_key,
            'secret_key': self.secret_key,
            'session_token': self.session_token,
            'anonymous': False,
            'region': self.region,
        }
        options.update(
            arrow_connection_options(
                self.endpoint_url,
                _addressing_style_value(self.addressing_style), self.ca_path))
        return options


S3AddressingStyleValue = Literal['auto', 'path', 'virtual']


def _addressing_style_value(
        style: S3AddressingStyle | str) -> S3AddressingStyleValue:
    if style == S3AddressingStyle.AUTO or style == 'auto':
        return 'auto'
    if style == S3AddressingStyle.PATH or style == 'path':
        return 'path'
    if style == S3AddressingStyle.VIRTUAL or style == 'virtual':
        return 'virtual'
    raise ValueError(f'Unsupported S3 addressing style: {style!r}')


_CA_DIRECTORY = tempfile.mkdtemp(prefix='kumo-s3-ca-')
os.chmod(_CA_DIRECTORY, 0o700)
_CA_PATHS: dict[tuple[str, str, str], str] = {}
_CA_LOCK = threading.Lock()


def _cleanup_ca_directory() -> None:
    for path in tuple(_CA_PATHS.values()):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(_CA_DIRECTORY)
    except OSError:
        pass


atexit.register(_cleanup_ca_directory)


def _materialize_ca(owner: str, rotation_id: str, certificate: str) -> str:
    """Materialize validated CA data in a process-private, read-only file.

    Files remain available to in-flight clients until process exit. Rotation
    creates a distinct path and cache identity; shutdown removes every file.
    """
    digest = hashlib.sha256(certificate.encode('utf-8')).hexdigest()
    key = owner, rotation_id, digest
    with _CA_LOCK:
        existing = _CA_PATHS.get(key)
        if existing is not None:
            return existing
        descriptor, path = tempfile.mkstemp(dir=_CA_DIRECTORY, prefix='ca-',
                                            suffix='.pem')
        try:
            os.fchmod(descriptor, 0o400)
            os.write(descriptor, certificate.encode('utf-8'))
        finally:
            os.close(descriptor)
        _CA_PATHS[key] = path
        return path


def _required_secret_value(secret: Mapping[str, str], key: str,
                           secret_name: str) -> str:
    value = secret.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f'S3 credential secret {secret_name!r} is missing required key '
            f'{key!r}')
    return value


@lru_cache(maxsize=1)
def deployment_s3_connection() -> Optional[S3Connection]:
    """Resolve the deployment connection and its mounted credential secret."""
    config = deploy_env().storage.s3
    if config is None:
        return None
    secret = get_secrets(config.credential_secret_name)
    session_token = secret.get('aws_session_token')
    if session_token is not None and (not isinstance(session_token, str)
                                      or not session_token.strip()):
        raise RuntimeError(
            f'S3 credential secret {config.credential_secret_name!r} has an '
            'empty aws_session_token')
    return S3Connection(
        endpoint_url=config.endpoint_url,
        region=config.region,
        addressing_style=config.addressing_style,
        credential_secret_name=config.credential_secret_name,
        access_key=_required_secret_value(secret, 'aws_access_key_id',
                                          config.credential_secret_name),
        secret_key=_required_secret_value(secret, 'aws_secret_access_key',
                                          config.credential_secret_name),
        session_token=session_token,
        ca_bundle_path=config.ca_bundle_path,
        send_bucket_owner_full_control=(config.send_bucket_owner_full_control),
    )


def active_s3_connection() -> Optional[S3Connection]:
    """Return deployment settings unless an explicit S3 identity is active."""
    # Local import breaks the credentials -> resolved-connection type cycle.
    from io_utils.fsutil.credentials import credentials_for_provider
    scope = credentials_for_provider('s3')
    if scope is not None:
        connection = (scope.credentials or {}).get('s3_connection')
        if connection is None:
            return None
        if not isinstance(connection, S3Connection):
            raise RuntimeError('Invalid resolved S3 connector connection')
        return connection
    return deployment_s3_connection()


S3_INHERITED_ENVIRONMENT_KEYS = frozenset({
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
    'AWS_SECURITY_TOKEN',
    'AWS_PROFILE',
    'AWS_DEFAULT_PROFILE',
    'AWS_SHARED_CREDENTIALS_FILE',
    'AWS_CONFIG_FILE',
    'AWS_WEB_IDENTITY_TOKEN_FILE',
    'AWS_ROLE_ARN',
    'AWS_ROLE_SESSION_NAME',
    'AWS_CONTAINER_CREDENTIALS_FULL_URI',
    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE',
    'AWS_ENDPOINT_URL',
    'AWS_ENDPOINT_URL_S3',
    'AWS_CA_BUNDLE',
    'AWS_S3_ADDRESSING_STYLE',
    'S3_ACCESS_KEY',
    'S3_SECRET_KEY',
    'S3_SESSION_TOKEN',
    'S3_REGION',
    'S3_ENDPOINT',
    'S3_VIRTUAL_HOSTED',
})


def scrubbed_s3_environment() -> dict[str, None]:
    """Remove every inherited credential-chain and endpoint selector."""
    return {name: None for name in S3_INHERITED_ENVIRONMENT_KEYS}


def s3_connection_environment(connection: S3Connection) -> dict[str, str]:
    """Build child-only settings for one already-authorized connection."""
    if not connection.access_key or not connection.secret_key:
        raise RuntimeError('Resolved S3 connection has no static credentials')
    environment = {
        'AWS_ACCESS_KEY_ID': connection.access_key,
        'AWS_SECRET_ACCESS_KEY': connection.secret_key,
        'AWS_REGION': connection.region,
        'AWS_DEFAULT_REGION': connection.region,
        'AWS_ENDPOINT_URL': connection.endpoint_url,
        'AWS_ENDPOINT_URL_S3': connection.endpoint_url,
        'AWS_EC2_METADATA_DISABLED': 'true',
        # DiskGraph's object_store client reads the S3-prefixed aliases.
        'S3_ACCESS_KEY': connection.access_key,
        'S3_SECRET_KEY': connection.secret_key,
        'S3_REGION': connection.region,
        'S3_ENDPOINT': connection.endpoint_url,
    }
    if connection.session_token:
        environment['AWS_SESSION_TOKEN'] = connection.session_token
        environment['S3_SESSION_TOKEN'] = connection.session_token
    if _addressing_style_value(connection.addressing_style) != 'auto':
        virtual = str(
            _addressing_style_value(connection.addressing_style) ==
            'virtual').lower()
        environment['S3_VIRTUAL_HOSTED'] = virtual
        environment['AWS_S3_ADDRESSING_STYLE'] = (_addressing_style_value(
            connection.addressing_style))
    if connection.ca_path:
        environment['AWS_CA_BUNDLE'] = connection.ca_path
        environment['SSL_CERT_FILE'] = connection.ca_path
    return environment


def s3_subprocess_environment() -> dict[str, str]:
    """Build child-only settings for native deployment-storage clients."""
    connection = deployment_s3_connection()
    return s3_connection_environment(connection) if connection else {}


class BucketOwnerWriteArgs(TypedDict, total=False):
    ACL: Literal['bucket-owner-full-control']


def bucket_owner_write_args() -> BucketOwnerWriteArgs:
    connection = active_s3_connection()
    if connection is None or connection.send_bucket_owner_full_control:
        return {'ACL': 'bucket-owner-full-control'}
    return {}
