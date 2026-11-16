# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for io_utils.secrets — provider dispatch and chain fallback."""
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPIError

from io_utils.secrets import (
    AWSSecretProvider,
    ChainSecretProvider,
    EnvVarSecretProvider,
    GCPSecretProvider,
    SecretAccessException,
    SecretNotFoundException,
    SecretProvider,
    _build_secret_provider,
    _parse_secret_string,
)
from service_framework._raw_deploy_config import SecretBackend

# ============================================================================
# Helpers
# ============================================================================


class FakeProvider(SecretProvider):
    """In-memory provider for testing."""
    def __init__(self, secrets=None):
        self._secrets = dict(secrets or {})

    def get_secret(self, name):
        if name not in self._secrets:
            raise SecretNotFoundException(f'{name} not found')
        return self._secrets[name]

    def put_secret(self, name, value):
        self._secrets[name] = value
        return {'name': name}


class BrokenProvider(SecretProvider):
    """Provider that always raises SecretAccessException."""
    def get_secret(self, name):
        raise SecretAccessException('backend unreachable')

    def put_secret(self, name, value):
        raise SecretAccessException('backend unreachable')


# ============================================================================
# _parse_secret_string
# ============================================================================


class TestParseSecretString:
    def test_json_string(self):
        result = _parse_secret_string('db', '{"user": "admin", "pass": "pw"}')
        assert result == {'user': 'admin', 'pass': 'pw'}

    def test_plain_string(self):
        result = _parse_secret_string('api_key', 'sk-123')
        assert result == {'api_key': 'sk-123'}


def test_aws_invalid_response_redacts_secret_binary(monkeypatch, caplog):
    secret_binary = b'known-secret-binary-material'
    client = MagicMock()
    client.get_secret_value.return_value = {
        'SecretString': None,
        'SecretBinary': secret_binary,
    }
    monkeypatch.setattr(boto3, 'client', lambda **_kwargs: client)
    monkeypatch.delenv('test-secret', raising=False)
    provider = object.__new__(AWSSecretProvider)
    provider.region = 'us-east-1'

    with pytest.raises(SecretAccessException) as error:
        provider.get_secret('test-secret')

    assert str(error.value) == (
        'AWS Secrets Manager response did not include SecretString')
    assert secret_binary.decode() not in str(error.value)
    assert secret_binary.decode() not in repr(error.value)
    assert secret_binary.decode() not in caplog.text


def test_aws_provider_error_is_sanitized_at_secret_boundary(
        monkeypatch, caplog):
    canary = 'aws-provider-body password=secret-manager-canary'
    client = MagicMock()
    client.get_secret_value.side_effect = ClientError(
        {'Error': {
            'Code': 'AccessDenied',
            'Message': canary,
        }},
        'GetSecretValue',
    )
    monkeypatch.setattr(boto3, 'client', lambda **_kwargs: client)
    monkeypatch.delenv('test-secret', raising=False)
    provider = object.__new__(AWSSecretProvider)
    provider.region = 'us-east-1'

    with pytest.raises(SecretAccessException) as error:
        provider.get_secret('test-secret')

    rendered = f'{error.value!s} {error.value!r} {caplog.text}'
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True
    assert canary not in rendered
    assert 'secret-manager-canary' not in rendered


def test_gcp_provider_error_is_sanitized_at_secret_boundary(
        monkeypatch, caplog):
    canary = 'gcp-provider-body token=secret-manager-canary'
    client = MagicMock()
    client.access_secret_version.side_effect = GoogleAPIError(canary)
    provider = object.__new__(GCPSecretProvider)
    provider.project_id = 'safe-project'
    monkeypatch.setattr(provider, '_get_client', lambda: client)
    monkeypatch.delenv('test-secret', raising=False)

    with pytest.raises(SecretAccessException) as error:
        provider.get_secret('test-secret')

    rendered = f'{error.value!s} {error.value!r} {caplog.text}'
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True
    assert canary not in rendered
    assert 'secret-manager-canary' not in rendered


# ============================================================================
# ChainSecretProvider
# ============================================================================


class TestChainSecretProvider:
    def test_single_provider_hit(self):
        p = FakeProvider({'key': {'val': '1'}})
        chain = ChainSecretProvider([p])
        assert chain.get_secret('key') == {'val': '1'}

    def test_single_provider_miss(self):
        p = FakeProvider()
        chain = ChainSecretProvider([p])
        with pytest.raises(SecretNotFoundException):
            chain.get_secret('missing')

    def test_fallback_on_not_found(self):
        primary = FakeProvider()  # empty
        fallback = FakeProvider({'key': {'val': 'from-fallback'}})
        chain = ChainSecretProvider([primary, fallback])
        assert chain.get_secret('key') == {'val': 'from-fallback'}

    def test_primary_wins_over_fallback(self):
        primary = FakeProvider({'key': {'val': 'from-primary'}})
        fallback = FakeProvider({'key': {'val': 'from-fallback'}})
        chain = ChainSecretProvider([primary, fallback])
        assert chain.get_secret('key') == {'val': 'from-primary'}

    def test_access_error_propagates_immediately(self):
        """SecretAccessException should NOT be caught — it means the
        backend is reachable but broken, not that the secret is missing."""
        broken = BrokenProvider()
        fallback = FakeProvider({'key': {'val': 'ok'}})
        chain = ChainSecretProvider([broken, fallback])
        with pytest.raises(SecretAccessException):
            chain.get_secret('key')

    def test_all_miss_raises_last_not_found(self):
        p1 = FakeProvider()
        p2 = FakeProvider()
        chain = ChainSecretProvider([p1, p2])
        with pytest.raises(SecretNotFoundException):
            chain.get_secret('missing')

    def test_put_delegates_to_primary(self):
        primary = FakeProvider()
        fallback = FakeProvider()
        chain = ChainSecretProvider([primary, fallback])
        chain.put_secret('key', {'val': '1'})
        assert primary.get_secret('key') == {'val': '1'}
        with pytest.raises(SecretNotFoundException):
            fallback.get_secret('key')

    def test_put_with_readonly_primary_raises(self):
        """If primary is read-only, put_secret raises even if a fallback
        supports writes.  This is by design — fallbacks are read-only."""
        readonly_primary = EnvVarSecretProvider()
        writable_fallback = FakeProvider()
        chain = ChainSecretProvider([readonly_primary, writable_fallback])
        with pytest.raises(SecretAccessException, match='read-only'):
            chain.put_secret('key', {'val': '1'})

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match='at least one provider'):
            ChainSecretProvider([])


# ============================================================================
# EnvVarSecretProvider
# ============================================================================


class TestEnvVarSecretProvider:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv('MY_SECRET', 'secret-value')
        provider = EnvVarSecretProvider()
        assert provider.get_secret('MY_SECRET') == {'MY_SECRET': 'secret-value'}

    def test_reads_json_from_env(self, monkeypatch):
        monkeypatch.setenv('MY_SECRET', '{"user": "admin"}')
        provider = EnvVarSecretProvider()
        assert provider.get_secret('MY_SECRET') == {'user': 'admin'}

    def test_missing_raises_not_found(self, monkeypatch):
        monkeypatch.delenv('NONEXISTENT', raising=False)
        provider = EnvVarSecretProvider()
        with pytest.raises(SecretNotFoundException):
            provider.get_secret('NONEXISTENT')

    def test_put_raises(self):
        provider = EnvVarSecretProvider()
        with pytest.raises(SecretAccessException, match='read-only'):
            provider.put_secret('key', {'val': '1'})


# ============================================================================
# _build_secret_provider
# ============================================================================


class TestBuildProvider:
    def test_env_var_backend(self):
        provider = _build_secret_provider(SecretBackend.ENV_VAR)
        assert isinstance(provider, EnvVarSecretProvider)
