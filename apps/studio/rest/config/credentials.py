# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, Literal, Optional, Union

from pydantic import SecretStr, model_validator

from io_utils import before_validator_dict, kumo_dataclass
from io_utils.encryption import EncryptedStr
from io_utils.fsutil.s3_connection import S3Connection


class AsDictMixin:
    __temporal_serializable__: ClassVar[bool] = False

    def as_dict(self) -> Dict[str, EncryptedStr]:
        return {k: v for k, v in asdict(self).items() if v}


def _make_root_validator(expected_type: str):
    # v1 @root_validator(pre=True) -> @model_validator(mode='before'); the
    # factory still returns a per-type validator bound onto each class.
    @model_validator(mode='before')
    @classmethod
    def _convert_secrets(cls, values):
        values = before_validator_dict(values)
        type = values.pop('_type', None)
        if type is not None and type != expected_type:
            raise ValueError(
                f'Unexpected credential type: {type} vs {expected_type}')

        new_values = {}
        for field, value in values.items():
            if isinstance(value, str):
                new_values[field] = EncryptedStr(value)
            elif isinstance(value, SecretStr):
                new_values[field] = EncryptedStr(value)
            else:
                new_values[field] = value
        new_values['_type'] = expected_type
        return new_values

    return _convert_secrets


@kumo_dataclass(frozen=True)
class UsernamePassword(AsDictMixin):
    username: SecretStr
    password: SecretStr
    _type: Literal['username_password'] = 'username_password'

    _convert = _make_root_validator('username_password')

    def as_dict(self) -> Dict[str, EncryptedStr]:
        return {
            'user': EncryptedStr(self.username),
            'password': EncryptedStr(self.password)
        }


@kumo_dataclass(frozen=True)
class DatabricksCredentials(AsDictMixin):
    client_id: SecretStr
    client_secret: SecretStr
    pat: SecretStr
    _type: Literal['databricks'] = 'databricks'

    _convert = _make_root_validator('databricks')


@kumo_dataclass(frozen=True)
class KeyPair(AsDictMixin):
    """
    Private key pair for snowflake authentication.
    """
    _type: Literal['keypair'] = 'keypair'

    private_key: Optional[SecretStr] = None
    private_key_passphrase: Optional[SecretStr] = None
    user: Optional[SecretStr] = None

    _convert = _make_root_validator('keypair')


@dataclass(frozen=True, repr=False)
class AWSFederatedCredentials:
    __temporal_serializable__: ClassVar[bool] = False
    access_key_id: SecretStr
    secret_access_key: SecretStr
    session_token: SecretStr
    region: Optional[str] = None
    connection: Optional[S3Connection] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'provider': 's3',
            'aws_access_key_id': self.access_key_id.get_secret_value(),
            'aws_secret_access_key': self.secret_access_key.get_secret_value(),
            'aws_session_token': self.session_token.get_secret_value(),
            'region': self.region,
        }
        if self.connection is not None:
            result['s3_connection'] = self.connection
        return result


@dataclass(frozen=True, repr=False)
class AWSAmbientCredentials:
    """Process-local marker for one attested refreshable AWS role session."""
    __temporal_serializable__: ClassVar[bool] = False
    session: Any
    approved_role_arn: str
    region: Optional[str] = None
    connection: Optional[S3Connection] = None

    def as_dict(self) -> Dict[str, Any]:
        # Deliberately carry the refreshable provider session, never frozen
        # access keys. ResolvedConnectorSpec prevents this process-local value
        # from being accepted by Temporal's payload converter.
        result = {
            'provider': 's3',
            'ambient_session': self.session,
            'region': self.region,
        }
        if self.connection is not None:
            result['s3_connection'] = self.connection
        return result


@dataclass(frozen=True, repr=False)
class AWSAccessKeyCredentials:
    __temporal_serializable__: ClassVar[bool] = False
    access_key_id: SecretStr
    secret_access_key: SecretStr
    session_token: Optional[SecretStr] = None
    region: Optional[str] = None
    connection: Optional[S3Connection] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'provider': 's3',
            'aws_access_key_id': self.access_key_id.get_secret_value(),
            'aws_secret_access_key': self.secret_access_key.get_secret_value(),
            'aws_session_token': (self.session_token.get_secret_value()
                                  if self.session_token else None),
            'region': self.region,
        }
        if self.connection is not None:
            result['s3_connection'] = self.connection
        return result


@dataclass(frozen=True, repr=False)
class GoogleCloudCredentials:
    """Activity-local Google credential for BigQuery or GCS.

    Google credential objects contain refresh tokens and signed assertions and
    are intentionally neither representable nor Temporal-serializable.
    """
    __temporal_serializable__: ClassVar[bool] = False
    provider: Literal['bigquery', 'gcs']
    credential: Any
    project: Optional[str] = None
    requester_pays_project: Optional[str] = None
    billing_project_id: Optional[str] = None
    ambient: bool = False

    def as_dict(self) -> Dict[str, Any]:
        credential = self.credential
        if (self.provider == 'bigquery' and self.billing_project_id
                and hasattr(credential, 'with_quota_project')):
            credential = credential.with_quota_project(self.billing_project_id)
        result = {
            'provider': self.provider,
            'credential': credential,
            'project': self.project,
            'requester_pays_project': self.requester_pays_project,
            'billing_project_id': self.billing_project_id,
        }
        if self.ambient:
            result['ambient'] = True
        return result


@dataclass(frozen=True, repr=False)
class AzureFederatedCredentials:
    __temporal_serializable__: ClassVar[bool] = False
    credential: Any
    endpoint_suffix: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            'provider': 'adls',
            'credential': self.credential,
            'endpoint_suffix': self.endpoint_suffix,
        }


Credentials = Union[UsernamePassword, DatabricksCredentials, KeyPair,
                    AWSFederatedCredentials, AWSAmbientCredentials,
                    AWSAccessKeyCredentials,
                    GoogleCloudCredentials, AzureFederatedCredentials]
