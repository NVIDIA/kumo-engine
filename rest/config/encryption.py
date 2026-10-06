# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import hmac
import os
from base64 import b64decode, b64encode, urlsafe_b64decode, urlsafe_b64encode
from copy import deepcopy
from typing import Optional

from cryptography.fernet import Fernet
from kumoapi.data_source import (
    DatabricksOAuthCredentials,
    GoogleServiceAccountKeyCredentials,
    IdentityCredentials,
    S3AccessKeyCredentials,
    S3AssumeRoleCredentials,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
)
from pydantic import SecretStr

from kumo.utils.singleton import Singleton
from rest.config.firstboot_secrets import get_secret_prefer_firstboot
from service_framework.deploy_stage import DeployStage
from statedb.rest.credentials import (
    EncryptedDatabricksOAuthCredentials,
    EncryptedGoogleServiceAccountKeyCredentials,
    EncryptedIdentityCredentials,
    EncryptedS3AccessKeyCredentials,
    EncryptedS3AssumeRoleCredentials,
    EncryptedSnowflakeKeyPairCredentials,
    EncryptedSnowflakePasswordCredentials,
)
from statedb.rest.group import ExternalIdentityProviderID

# Constant, default encryption key that's only meant for dev/test
_DEFAULT_ENCRYPTION_KEY = urlsafe_b64encode(
    hashlib.md5("kumo-default-encryption-key".encode()).hexdigest().encode())
_IDP_LOOKUP_KDF_DOMAIN = b'kumo:idp-lookup-key:v1'
_IDP_GROUP_LOOKUP_DOMAIN = b'kumo:idp-group-id:v2'
_S3_ACCESS_KEY_LOOKUP_DOMAIN = b'kumo:s3-access-key-id:v1'
_DEFAULT_FERNET_SECRET_NAME = 'kumo-secret-key'


def _derive_idp_lookup_key(encryption_key: bytes) -> bytes:
    raw_key = urlsafe_b64decode(encryption_key)
    return hmac.new(raw_key, _IDP_LOOKUP_KDF_DOMAIN, hashlib.sha256).digest()


def _keyed_lookup_digest(value: str, key: bytes, domain: bytes) -> str:
    payload = domain + b'\0' + value.encode('utf-8')
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


class CredentialsEncryptor(metaclass=Singleton):
    def __init__(self, deploy_mode: Optional[DeployStage] = None) -> None:
        # Resolve deployment mode from the canonical config when callers do
        # not provide it, so REST, workers, and StateDB select the same key.
        if deploy_mode is None:
            from service_framework.deploy_env_config import deploy_env

            deploy_mode = deploy_env().deploy_mode
        is_non_prod = deploy_mode != DeployStage.PROD
        # Default, unsecure encryption key for dev/test purpose only.
        if is_non_prod:
            encryption_key: bytes = _DEFAULT_ENCRYPTION_KEY
        else:
            secret_name = os.getenv('FERNET_SECRET_NAME',
                                    _DEFAULT_FERNET_SECRET_NAME)
            secrets = get_secret_prefer_firstboot(secret_name)
            encryption_key: bytes = secrets[
                'credentials_encryption_key'].encode()
        try:
            self._cipher = Fernet(encryption_key)
        except ValueError as err:
            raise ValueError('Invalid credentials encryption key') from err
        self._idp_lookup_key = _derive_idp_lookup_key(encryption_key)

    def encrypt_identity_credentials(
        self,
        credentials: IdentityCredentials,
    ) -> EncryptedIdentityCredentials:
        return encrypt_identity_credentials(self._cipher, credentials)

    def decrypt_identity_credentials(
        self,
        encrypted_credentials: EncryptedIdentityCredentials,
    ) -> IdentityCredentials:
        return decrypt_identity_credentials(self._cipher, encrypted_credentials)

    def s3_access_key_fingerprint(
        self,
        endpoint_url: str,
        access_key_id: str,
    ) -> str:
        lookup_value = f'{endpoint_url}\0{access_key_id}'
        return _keyed_lookup_digest(
            lookup_value,
            self._idp_lookup_key,
            _S3_ACCESS_KEY_LOOKUP_DOMAIN,
        )


def encrypt_identity_credentials(
    cipher: Fernet,
    credentials: IdentityCredentials,
) -> EncryptedIdentityCredentials:
    if isinstance(credentials, DatabricksOAuthCredentials):
        return EncryptedDatabricksOAuthCredentials(
            client_secret=_encrypt_string(
                cipher, credentials.client_secret.get_secret_value()))
    if isinstance(credentials, SnowflakePasswordCredentials):
        return EncryptedSnowflakePasswordCredentials(password=_encrypt_string(
            cipher, credentials.password.get_secret_value()))
    if isinstance(credentials, SnowflakeKeyPairCredentials):
        return EncryptedSnowflakeKeyPairCredentials(
            private_key=_encrypt_string(
                cipher, credentials.private_key.get_secret_value()),
            private_key_passphrase=(_encrypt_string(
                cipher,
                credentials.private_key_passphrase.get_secret_value(),
            ) if credentials.private_key_passphrase else None),
        )
    if isinstance(credentials, GoogleServiceAccountKeyCredentials):
        return EncryptedGoogleServiceAccountKeyCredentials(
            private_key_id=_encrypt_string(
                cipher, credentials.private_key_id.get_secret_value()),
            private_key=_encrypt_string(
                cipher, credentials.private_key.get_secret_value()),
        )
    if isinstance(credentials, S3AssumeRoleCredentials):
        return EncryptedS3AssumeRoleCredentials(
            external_id=(_encrypt_string(
                cipher, credentials.external_id.get_secret_value())
                         if credentials.external_id else None), )
    if isinstance(credentials, S3AccessKeyCredentials):
        return EncryptedS3AccessKeyCredentials(
            access_key_id=_encrypt_string(
                cipher, credentials.access_key_id.get_secret_value()),
            secret_access_key=_encrypt_string(
                cipher, credentials.secret_access_key.get_secret_value()),
            session_token=(_encrypt_string(
                cipher, credentials.session_token.get_secret_value())
                           if credentials.session_token else None),
            ca_certificate=(_encrypt_string(
                cipher, credentials.ca_certificate.get_secret_value())
                            if credentials.ca_certificate else None),
        )
    raise RuntimeError(f'Unknown identity credential type: {type(credentials)}')


def decrypt_identity_credentials(
    cipher: Fernet,
    encrypted_credentials: EncryptedIdentityCredentials,
) -> IdentityCredentials:
    credentials = deepcopy(encrypted_credentials)
    if isinstance(credentials, EncryptedDatabricksOAuthCredentials):
        return DatabricksOAuthCredentials(client_secret=SecretStr(
            _decrypt_string(cipher, credentials.client_secret)))
    if isinstance(credentials, EncryptedSnowflakePasswordCredentials):
        return SnowflakePasswordCredentials(
            password=SecretStr(_decrypt_string(cipher, credentials.password)))
    if isinstance(credentials, EncryptedSnowflakeKeyPairCredentials):
        return SnowflakeKeyPairCredentials(
            private_key=SecretStr(
                _decrypt_string(cipher, credentials.private_key)),
            private_key_passphrase=(SecretStr(
                _decrypt_string(cipher, credentials.private_key_passphrase))
                                    if credentials.private_key_passphrase else
                                    None),
        )
    if isinstance(credentials, EncryptedGoogleServiceAccountKeyCredentials):
        return GoogleServiceAccountKeyCredentials(
            private_key_id=SecretStr(
                _decrypt_string(cipher, credentials.private_key_id)),
            private_key=SecretStr(
                _decrypt_string(cipher, credentials.private_key)),
        )
    if isinstance(credentials, EncryptedS3AssumeRoleCredentials):
        return S3AssumeRoleCredentials(
            external_id=(SecretStr(
                _decrypt_string(cipher, credentials.external_id))
                         if credentials.external_id else None), )
    if isinstance(credentials, EncryptedS3AccessKeyCredentials):
        return S3AccessKeyCredentials(
            access_key_id=SecretStr(
                _decrypt_string(cipher, credentials.access_key_id)),
            secret_access_key=SecretStr(
                _decrypt_string(cipher, credentials.secret_access_key)),
            session_token=(SecretStr(
                _decrypt_string(cipher, credentials.session_token))
                           if credentials.session_token else None),
            ca_certificate=(SecretStr(
                _decrypt_string(cipher, credentials.ca_certificate))
                            if credentials.ca_certificate else None),
        )
    raise RuntimeError('Unknown encrypted identity credential type: '
                       f'{type(credentials)}')


def _encrypt_string(cipher: Fernet, value: str) -> str:
    return b64encode(cipher.encrypt(value.encode('utf-8'))).decode('utf-8')


def _decrypt_string(cipher: Fernet, encrypted: str) -> str:
    return cipher.decrypt(b64decode(encrypted)).decode('utf-8')


# ---------------------------------------------------------------------------
# IDP ID helpers
# ---------------------------------------------------------------------------


def hash_idp_group_id(
    identity_provider_id: ExternalIdentityProviderID,
    external_group_id: str,
) -> str:
    """Hash one canonical provider/group pair for equality lookups."""
    if not identity_provider_id or not external_group_id:
        return ""
    if '\0' in identity_provider_id.value or '\0' in external_group_id:
        raise ValueError('Identity provider and external group IDs cannot '
                         'contain NUL characters')
    lookup_value = f'{identity_provider_id.value}\0{external_group_id}'
    return _keyed_lookup_digest(
        lookup_value,
        CredentialsEncryptor()._idp_lookup_key,
        _IDP_GROUP_LOOKUP_DOMAIN,
    )


def encrypt_idp_id(idp_id: str) -> str:
    """Fernet-encrypt an IDP group ID for at-rest storage."""
    if not idp_id:
        return ""
    cipher = CredentialsEncryptor()._cipher
    return _encrypt_string(cipher, idp_id)


def decrypt_idp_id(encrypted: str) -> str:
    """Decrypt a Fernet-encrypted IDP group ID for display."""
    if not encrypted:
        return ""
    cipher = CredentialsEncryptor()._cipher
    return _decrypt_string(cipher, encrypted)


def encrypt_secret_string(value: str) -> str:
    """Fernet-encrypt a generic secret string for at-rest storage."""
    if not value:
        return ""
    cipher = CredentialsEncryptor()._cipher
    return _encrypt_string(cipher, value)


def decrypt_secret_string(encrypted: str) -> str:
    """Decrypt a generic Fernet-encrypted secret string."""
    if not encrypted:
        return ""
    cipher = CredentialsEncryptor()._cipher
    return _decrypt_string(cipher, encrypted)
