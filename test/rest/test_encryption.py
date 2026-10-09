# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from typing import Dict

import pytest
from cryptography.fernet import Fernet
from kumoapi.data_source import (
    DatabricksOAuthCredentials,
    GoogleServiceAccountKeyCredentials,
    S3AccessKeyCredentials,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
)
from pydantic import SecretStr, ValidationError

from io_utils import kumo_dataclass
from io_utils.encryption import EncryptedStr, GlobalEncryptor
from io_utils.json import from_json_dict, to_json_dict
from io_utils.secrets import SecretAccessException
from rest.config.credentials import (
    Credentials,
    DatabricksCredentials,
    UsernamePassword,
)
from rest.config.encryption import (
    _IDP_GROUP_LOOKUP_DOMAIN,
    CredentialsEncryptor,
    _derive_idp_lookup_key,
    _keyed_lookup_digest,
    hash_idp_group_id,
)
from service_framework.deploy_env_config import _reset_deploy_env, deploy_env
from service_framework.deploy_stage import DeployStage

# isort: off
from statedb.rest.credentials import (
    EncryptedGoogleServiceAccountKeyCredentials,
    EncryptedS3AccessKeyCredentials,
)
# isort: on
from statedb.rest.group import ExternalIdentityProviderID

_TEST_CA = """-----BEGIN CERTIFICATE-----
MIIBSjCB/aADAgECAhRiPITwqOT0gz4XKMTFMkS41Z2f7jAFBgMrZXAwGzEZMBcG
A1UEAwwQa3Vtby1hcGktdGVzdC1jYTAeFw0yNjA4MTIxOTM3MDNaFw0yNjA4MTMx
OTM3MDNaMBsxGTAXBgNVBAMMEGt1bW8tYXBpLXRlc3QtY2EwKjAFBgMrZXADIQBv
iURRF3Kk370JkN7pUmmxmRXKOToEyBp1t/09UPGNsaNTMFEwHQYDVR0OBBYEFCkf
DW84FLpOgv59tsdFJSoUlLnjMB8GA1UdIwQYMBaAFCkfDW84FLpOgv59tsdFJSoU
lLnjMA8GA1UdEwEB/wQFMAMBAf8wBQYDK2VwA0EAVDph44XtYxnNfHE+ISJET136
I1mq8yFZ7bQkQ9MEleKKvXO13frbEvWZeBaX34PR+ZymWlEmuxiDRsIYTxA+DQ==
-----END CERTIFICATE-----"""


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@kumo_dataclass
class NewUsernamePassword:
    username: EncryptedStr
    password: EncryptedStr


@kumo_dataclass
class CredentialsHolder:
    credentials: Credentials
    credentials_dict: Dict[str, Credentials]


def test_credentials_union_serde(monkeypatch: pytest.MonkeyPatch):
    GlobalEncryptor._instance = None

    username_pw = UsernamePassword(username=SecretStr('kumo-dev'),
                                   password=SecretStr('ramen'))
    databricks = DatabricksCredentials(client_id=SecretStr('client'),
                                       client_secret=SecretStr('secret'),
                                       pat=SecretStr('pat'))
    creds = CredentialsHolder(
        credentials=username_pw, credentials_dict={
            'user': username_pw,
            'db': databricks
        })

    creds_json = to_json_dict(creds)
    # Verify fields in json dict.
    assert creds_json['credentials']['_type'] == username_pw._type
    assert creds_json['credentials']['username'].startswith('__encrypted__')
    assert creds_json['credentials']['password'].startswith('__encrypted__')
    creds_dict = creds_json['credentials_dict']
    assert creds_dict['user']['_type'] == username_pw._type
    assert creds_dict['user']['username'].startswith('__encrypted__')
    assert creds_dict['user']['password'].startswith('__encrypted__')
    assert creds_dict['db']['_type'] == databricks._type
    assert creds_dict['db']['client_id'].startswith('__encrypted__')
    assert creds_dict['db']['client_secret'].startswith('__encrypted__')
    assert creds_dict['db']['pat'].startswith('__encrypted__')

    # Round-trip json serde with encryption.
    assert from_json_dict(creds_json, CredentialsHolder) == creds

    # Backward compatibility (without "_type" discriminator)
    creds_json_old = deepcopy(creds_json)
    creds_json_old['credentials'].pop('_type')
    creds_json_old['credentials_dict']['user'].pop('_type')
    creds_json_old['credentials_dict']['db'].pop('_type')
    assert from_json_dict(creds_json_old, CredentialsHolder) == creds

    # Corrupted encrypted value.
    creds_json_bad = deepcopy(creds_json)
    creds_json_bad['credentials']['username'] = '__encrypted__xxxxxxyyyyzzzz'
    with pytest.raises(ValidationError) as ex_info:
        from_json_dict(creds_json_bad, CredentialsHolder)
    assert 'Failed to decrypt secret' in ex_info.value.errors()[0]['msg']

    # Different decryption key.
    monkeypatch.setenv('FERNET_SECRET_NAME', 'kumo-pytest-secret-key')
    GlobalEncryptor._instance = None
    with pytest.raises(ValidationError) as ex_info:
        from_json_dict(creds_json_bad, CredentialsHolder)
    assert 'Failed to decrypt secret' in ex_info.value.errors()[0]['msg']


def test_encrypted_str():
    username_pw = UsernamePassword(username=SecretStr('kumo-dev'),
                                   password=SecretStr('ramen'))

    print(f'username type is : {type(username_pw.username)}')

    insecure = to_json_dict(username_pw, insecure=True)
    print(insecure)
    from_insecure = from_json_dict(insecure, UsernamePassword)
    print(f'from_insecure username type is : {type(from_insecure.username)}')
    assert from_insecure == username_pw
    from_insecure2 = from_json_dict(insecure, NewUsernamePassword)
    assert from_insecure2.username.get_secret_value() == 'kumo-dev'
    assert from_insecure2.password.get_secret_value() == 'ramen'

    secure = to_json_dict(username_pw, insecure=False)
    print(secure)
    from_secure = from_json_dict(secure, UsernamePassword)
    print(f'from_secure username type is : {type(from_secure.username)}')
    assert from_secure == username_pw
    from_secure2 = from_json_dict(secure, NewUsernamePassword)
    assert from_secure2.username.get_secret_value() == 'kumo-dev'
    assert from_secure2.password.get_secret_value() == 'ramen'


def test_encrypted_str_revalidation_and_masking(
        monkeypatch: pytest.MonkeyPatch):
    GlobalEncryptor._instance = None

    # Construction from a raw str, an EncryptedStr instance, and a parent
    # SecretStr instance must all succeed (the unwrap branches were
    # unreachable when the str schema wrapped _validate as an "after"
    # validator -- a SecretStr instance was rejected before _validate ran).
    holder = NewUsernamePassword(username='kumo-dev', password='ramen')
    assert holder.username.get_secret_value() == 'kumo-dev'

    from_secret = NewUsernamePassword(username=SecretStr('alice'),
                                      password=EncryptedStr('bob'))
    assert from_secret.username.get_secret_value() == 'alice'
    assert from_secret.password.get_secret_value() == 'bob'

    # Model-from-model re-validation preserves the secret value.
    revalidated = NewUsernamePassword(username=holder.username,
                                      password=holder.password)
    assert revalidated.username.get_secret_value() == 'kumo-dev'
    assert revalidated.password.get_secret_value() == 'ramen'

    # json serialization masks the plaintext (never leaks the secret value).
    dumped = holder.__pydantic_serializer__.to_json(holder)
    assert b'kumo-dev' not in dumped
    assert b'ramen' not in dumped
    assert b'**********' in dumped


def test_encryption_default_key():
    CredentialsEncryptor.clear()
    encryptor = CredentialsEncryptor()
    password = SnowflakePasswordCredentials(password=SecretStr('ramen'))
    encrypted = encryptor.encrypt_identity_credentials(password)
    assert encryptor.decrypt_identity_credentials(encrypted) == password
    # make sure Singleton works
    assert CredentialsEncryptor().decrypt_identity_credentials(
        encrypted) == password

    key_pair = SnowflakeKeyPairCredentials(
        private_key=SecretStr('my_key'),
        private_key_passphrase=SecretStr('password'),
    )
    encrypted = encryptor.encrypt_identity_credentials(key_pair)
    assert encryptor.decrypt_identity_credentials(encrypted) == key_pair
    # make sure Singleton works
    assert CredentialsEncryptor().decrypt_identity_credentials(
        encrypted) == key_pair


def test_no_arg_encryptor_matches_rest_seeded_encryptor(
        monkeypatch: pytest.MonkeyPatch):
    """A no-arg CredentialsEncryptor (worker/StateDB seeding) must resolve
    the same key as explicit REST-style construction from deploy_env()."""
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')
    _reset_deploy_env()
    CredentialsEncryptor.clear()
    try:
        config = deploy_env()
        rest_seeded = CredentialsEncryptor(deploy_mode=config.deploy_mode)
        password = SnowflakePasswordCredentials(password=SecretStr('ramen'))
        encrypted = rest_seeded.encrypt_identity_credentials(password)

        CredentialsEncryptor.clear()
        worker_seeded = CredentialsEncryptor()
        assert worker_seeded is not rest_seeded
        assert worker_seeded.decrypt_identity_credentials(encrypted) == password
        assert worker_seeded._idp_lookup_key == rest_seeded._idp_lookup_key
    finally:
        CredentialsEncryptor.clear()
        _reset_deploy_env()


def test_invalid_credentials_encryption_keys_are_sanitized(monkeypatch, caplog):
    invalid_key = 'known-invalid-fernet-key-material'
    monkeypatch.setattr(
        'rest.config.encryption.get_secret_prefer_firstboot',
        lambda _secret_name: {'credentials_encryption_key': invalid_key},
    )
    CredentialsEncryptor.clear()
    with pytest.raises(ValueError) as rest_error:
        CredentialsEncryptor(deploy_mode=DeployStage.PROD)
    CredentialsEncryptor.clear()

    monkeypatch.setattr(
        'io_utils.encryption.deploy_env',
        lambda: SimpleNamespace(deploy_mode=DeployStage.PROD),
    )
    monkeypatch.setattr(
        'io_utils.encryption.get_secrets',
        lambda _secret_name: {'credentials_encryption_key': invalid_key},
    )
    GlobalEncryptor._instance = None
    try:
        with pytest.raises(ValueError) as global_error:
            GlobalEncryptor()
    finally:
        GlobalEncryptor._instance = None

    for error in (rest_error.value, global_error.value):
        assert str(error) == 'Invalid credentials encryption key'
        assert isinstance(error.__cause__, ValueError)
        assert invalid_key not in str(error)
        assert invalid_key not in repr(error)
    assert invalid_key not in caplog.text


def test_idp_group_lookup_digest_is_provider_keyed_and_domain_separated():
    value = 'direct_connect\0engineering'
    key_a = _derive_idp_lookup_key(Fernet.generate_key())
    key_b = _derive_idp_lookup_key(Fernet.generate_key())

    digest = _keyed_lookup_digest(value, key_a, _IDP_GROUP_LOOKUP_DOMAIN)

    assert digest == _keyed_lookup_digest(value, key_a,
                                          _IDP_GROUP_LOOKUP_DOMAIN)
    assert digest != _keyed_lookup_digest(value, key_b,
                                          _IDP_GROUP_LOOKUP_DOMAIN)
    assert digest != _keyed_lookup_digest(value, key_a,
                                          b'kumo:other-lookup-domain:v1')
    assert digest != sha256(value.encode()).hexdigest()


def test_idp_group_lookup_hash_is_deterministic_and_group_scoped():
    CredentialsEncryptor.clear()
    CredentialsEncryptor()
    try:
        engineering = hash_idp_group_id(
            ExternalIdentityProviderID.DIRECT_CONNECT, 'engineering')
        assert engineering == hash_idp_group_id(
            ExternalIdentityProviderID.DIRECT_CONNECT, 'engineering')
        assert engineering != hash_idp_group_id(
            ExternalIdentityProviderID.DIRECT_CONNECT, 'data-science')
    finally:
        CredentialsEncryptor.clear()


def test_idp_group_lookup_hash_rejects_ambiguous_nul_input():
    with pytest.raises(ValueError, match='NUL'):
        hash_idp_group_id(ExternalIdentityProviderID.DIRECT_CONNECT,
                          'engineering\0privileged')


def test_encryption_aws_sm_key(monkeypatch: pytest.MonkeyPatch):
    requested_secret_names = []

    def unavailable_secret(secret_name):
        requested_secret_names.append(secret_name)
        raise SecretAccessException('secret unavailable')

    monkeypatch.setattr(
        'rest.config.encryption.get_secret_prefer_firstboot',
        unavailable_secret,
    )
    CredentialsEncryptor.clear()
    with pytest.raises(SecretAccessException):
        CredentialsEncryptor(deploy_mode=DeployStage.PROD)
    assert requested_secret_names == ['kumo-secret-key']

    CredentialsEncryptor.clear()
    encryptor = CredentialsEncryptor(deploy_mode=DeployStage.TEST)
    credentials = DatabricksOAuthCredentials(client_secret=SecretStr('ramen'))
    encrypted = encryptor.encrypt_identity_credentials(credentials)
    assert encryptor.decrypt_identity_credentials(encrypted) == credentials


def test_encryption_google_service_account_key_creds():
    CredentialsEncryptor.clear()
    encryptor = CredentialsEncryptor()
    creds = GoogleServiceAccountKeyCredentials(
        private_key_id=SecretStr('kumo-dev'),
        private_key=SecretStr('password'),
    )
    encrypted = encryptor.encrypt_identity_credentials(creds)
    assert isinstance(encrypted, EncryptedGoogleServiceAccountKeyCredentials)
    assert encryptor.decrypt_identity_credentials(encrypted) == creds


def test_encryption_s3_access_key_credentials():
    CredentialsEncryptor.clear()
    encryptor = CredentialsEncryptor()
    credentials = S3AccessKeyCredentials(
        access_key_id=SecretStr('minio-access-key'),
        secret_access_key=SecretStr('minio-secret-key'),
        session_token=SecretStr('minio-session-token'),
        ca_certificate=SecretStr(_TEST_CA),
    )

    encrypted = encryptor.encrypt_identity_credentials(credentials)

    assert isinstance(encrypted, EncryptedS3AccessKeyCredentials)
    assert encryptor.decrypt_identity_credentials(encrypted) == credentials
    for secret in (
        'minio-access-key',
        'minio-secret-key',
        'minio-session-token',
        _TEST_CA,
    ):
        assert secret not in repr(encrypted)
