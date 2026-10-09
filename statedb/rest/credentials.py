# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union

from kumoapi.data_source import IdentityCredentialKind

from io_utils import kumo_dataclass


@kumo_dataclass(frozen=True)
class EncryptedDatabricksOAuthCredentials:
    client_secret: str
    kind: Literal[IdentityCredentialKind.DATABRICKS_OAUTH] = (
        IdentityCredentialKind.DATABRICKS_OAUTH)


@kumo_dataclass(frozen=True)
class EncryptedSnowflakePasswordCredentials:
    password: str
    kind: Literal[IdentityCredentialKind.SNOWFLAKE_PASSWORD] = (
        IdentityCredentialKind.SNOWFLAKE_PASSWORD)


@kumo_dataclass(frozen=True)
class EncryptedSnowflakeKeyPairCredentials:
    private_key: str
    private_key_passphrase: Optional[str] = None
    kind: Literal[IdentityCredentialKind.SNOWFLAKE_KEY_PAIR] = (
        IdentityCredentialKind.SNOWFLAKE_KEY_PAIR)


@kumo_dataclass(frozen=True)
class EncryptedGoogleServiceAccountKeyCredentials:
    private_key_id: str
    private_key: str
    kind: Literal[IdentityCredentialKind.GOOGLE_SERVICE_ACCOUNT_KEY] = (
        IdentityCredentialKind.GOOGLE_SERVICE_ACCOUNT_KEY)


@kumo_dataclass(frozen=True)
class EncryptedS3AssumeRoleCredentials:
    external_id: Optional[str] = None
    kind: Literal[IdentityCredentialKind.S3_ASSUME_ROLE] = (
        IdentityCredentialKind.S3_ASSUME_ROLE)


@kumo_dataclass(frozen=True)
class EncryptedS3AccessKeyCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    ca_certificate: Optional[str] = None
    kind: Literal[IdentityCredentialKind.S3_ACCESS_KEY] = (
        IdentityCredentialKind.S3_ACCESS_KEY)


EncryptedIdentityCredentials = Union[
    EncryptedDatabricksOAuthCredentials,
    EncryptedSnowflakePasswordCredentials,
    EncryptedSnowflakeKeyPairCredentials,
    EncryptedGoogleServiceAccountKeyCredentials,
    EncryptedS3AssumeRoleCredentials,
    EncryptedS3AccessKeyCredentials,
]
