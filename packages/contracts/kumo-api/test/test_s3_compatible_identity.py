# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr, ValidationError

from kumoapi.data_source import (
    CreateIdentityArgs,
    RotateIdentityCredentialsArgs,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AddressingStyle,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3ConnectionConfig,
    S3IdentityConfig,
)
from kumoapi.json_serde import from_json, to_json, to_json_dict

_TEST_CA = """-----BEGIN CERTIFICATE-----
MIIBSjCB/aADAgECAhRiPITwqOT0gz4XKMTFMkS41Z2f7jAFBgMrZXAwGzEZMBcG
A1UEAwwQa3Vtby1hcGktdGVzdC1jYTAeFw0yNjA4MTIxOTM3MDNaFw0yNjA4MTMx
OTM3MDNaMBsxGTAXBgNVBAMMEGt1bW8tYXBpLXRlc3QtY2EwKjAFBgMrZXADIQBv
iURRF3Kk370JkN7pUmmxmRXKOToEyBp1t/09UPGNsaNTMFEwHQYDVR0OBBYEFCkf
DW84FLpOgv59tsdFJSoUlLnjMB8GA1UdIwQYMBaAFCkfDW84FLpOgv59tsdFJSoU
lLnjMA8GA1UdEwEB/wQFMAMBAf8wBQYDK2VwA0EAVDph44XtYxnNfHE+ISJET136
I1mq8yFZ7bQkQ9MEleKKvXO13frbEvWZeBaX34PR+ZymWlEmuxiDRsIYTxA+DQ==
-----END CERTIFICATE-----"""


def _connection() -> S3ConnectionConfig:
    return S3ConnectionConfig(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
    )


def _credentials() -> S3AccessKeyCredentials:
    return S3AccessKeyCredentials(
        access_key_id=SecretStr('customer-access-key'),
        secret_access_key=SecretStr('customer-secret-key'),
        session_token=SecretStr('optional-session-token'),
        ca_certificate=SecretStr(_TEST_CA),
    )


def test_existing_aws_identity_connection_defaults_are_unchanged() -> None:
    ambient = S3IdentityConfig(authentication=S3AmbientAuthentication())
    role = S3IdentityConfig(authentication=S3AssumeRoleAuthentication(
        partition='aws',
        role_arn='arn:aws:iam::123456789012:role/kumo-reader',
    ))

    assert ambient.connection is None
    assert role.connection is None


def test_s3_compatible_identity_roundtrips_and_redacts_all_secrets() -> None:
    payload = CreateIdentityArgs(
        display_name='Private MinIO',
        config=S3IdentityConfig(
            authentication=S3AccessKeyAuthentication(),
            connection=_connection(),
        ),
        credentials=_credentials(),
    )

    safe = to_json(payload)
    insecure = to_json(payload, insecure=True)

    assert from_json(insecure, CreateIdentityArgs) == payload
    assert to_json_dict(payload)['config']['connection'] == {
        'endpoint_url': 'https://minio.example.test:9000',
        'region': 'us-east-1',
        'addressing_style': 'path',
    }
    for secret in ('customer-access-key', 'customer-secret-key',
                   'optional-session-token', 'BEGIN CERTIFICATE'):
        assert secret not in safe
        assert secret not in repr(payload.credentials)


def test_s3_access_key_rotation_uses_standard_secret_serialization() -> None:
    payload = RotateIdentityCredentialsArgs(credentials=_credentials())

    safe = to_json(payload)
    insecure = to_json(payload, insecure=True)

    assert from_json(insecure, RotateIdentityCredentialsArgs) == payload
    for secret in ('customer-access-key', 'customer-secret-key',
                   'optional-session-token', 'BEGIN CERTIFICATE'):
        assert secret not in safe
        assert secret in insecure


def test_s3_secret_validation_errors_redact_input_material() -> None:
    secret_material = 'private-key-material-must-not-leak'

    with pytest.raises(ValidationError) as error:
        from_json(
            {
                'kind': 's3_access_key',
                'access_key_id': 'customer-access-key',
                'secret_access_key': 'customer-secret-key',
                'ca_certificate': f'not-a-ca-certificate:{secret_material}',
            },
            S3AccessKeyCredentials,
        )

    rendered_error = str(error.value)
    assert secret_material not in rendered_error
    assert 'customer-access-key' not in rendered_error
    assert 'customer-secret-key' not in rendered_error


def test_s3_credentials_schema_marks_every_value_write_only() -> None:
    from pydantic import TypeAdapter
    schema = TypeAdapter(S3AccessKeyCredentials).json_schema()

    for name in ('access_key_id', 'secret_access_key', 'session_token',
                 'ca_certificate'):
        field_schema = schema['properties'][name]
        string_schema = next(
            (variant for variant in field_schema.get('anyOf', [])
             if variant.get('type') == 'string'),
            field_schema,
        )
        assert string_schema['format'] == 'password'
        assert string_schema['writeOnly'] is True


def test_s3_connection_normalizes_origin_region_and_addressing() -> None:
    connection = S3ConnectionConfig(
        endpoint_url='HTTPS://MINIO.EXAMPLE.TEST:9000/',
        region=' us-east-1 ',
        addressing_style=S3AddressingStyle.PATH,
    )

    assert connection.endpoint_url == 'https://minio.example.test:9000'
    assert connection.region == 'us-east-1'
    assert connection.addressing_style == S3AddressingStyle.PATH


@pytest.mark.parametrize('endpoint', [
    'minio.example.test:9000',
    'ftp://minio.example.test',
    'https://user@minio.example.test',
    'https://minio.example.test/api',
    'https://minio.example.test?console=true',
    'https://minio.example.test#fragment',
    ' https://minio.example.test',
    'https://minio.example.test:invalid',
])
def test_s3_connection_rejects_unsafe_endpoint(endpoint: str) -> None:
    with pytest.raises((ValidationError, ValueError), match='endpoint|port'):
        S3ConnectionConfig(endpoint_url=endpoint, region='us-east-1')


def test_s3_custom_endpoint_requires_signing_region() -> None:
    with pytest.raises((ValidationError, ValueError), match='signing region'):
        S3ConnectionConfig(endpoint_url='https://minio.example.test:9000')


@pytest.mark.parametrize('field', [
    'access_key_id',
    'secret_access_key',
    'session_token',
    'ca_certificate',
])
def test_s3_access_key_credentials_reject_blank_material(field: str, ) -> None:
    credentials = _credentials()

    with pytest.raises((ValidationError, ValueError),
                       match='must not be blank'):
        replace(credentials, **{field: SecretStr('  ')})


def test_s3_access_key_credentials_reject_non_certificate_pem() -> None:
    credentials = _credentials()
    private_key = rsa.generate_private_key(public_exponent=65537,
                                           key_size=2048)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    with pytest.raises((ValidationError, ValueError),
                       match='PEM certificates'):
        replace(
            credentials,
            ca_certificate=SecretStr(private_key_pem),
        )


def test_s3_authentication_modes_require_matching_credentials() -> None:
    access_config = S3IdentityConfig(
        authentication=S3AccessKeyAuthentication(),
        connection=_connection(),
    )
    role_config = S3IdentityConfig(authentication=S3AssumeRoleAuthentication(
        partition='aws',
        role_arn='arn:aws:iam::123456789012:role/kumo-reader',
    ))

    with pytest.raises((ValidationError, ValueError),
                       match='config and credential types'):
        CreateIdentityArgs(display_name='wrong', config=access_config,
                           credentials=S3AssumeRoleCredentials())
    with pytest.raises((ValidationError, ValueError),
                       match='config and credential types'):
        CreateIdentityArgs(display_name='wrong', config=role_config,
                           credentials=_credentials())
