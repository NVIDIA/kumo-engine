# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
import urllib.request
from typing import List

import pandas as pd
import pytest
from cachetools import TTLCache
from kumoapi.typing import Dtype

import kumo.connector.utils as connector_utils
from kumo.connector.utils import (
    cached_batch_fetch,
    cast_col_to_str,
    cast_df_to_dtypes,
    get_temporary_aws_credential,
)


def test_get_temporary_aws_credential_uses_boto3_web_identity(
        tmp_path, monkeypatch):
    token_path = tmp_path / 'token'
    token_path.write_text('oidc-token')

    class FakeStsClient:
        def assume_role_with_web_identity(self, **kwargs):
            assert kwargs == {
                'RoleArn': 'arn:aws:iam::123456789012:role/kumo-test',
                'RoleSessionName': 'session',
                'WebIdentityToken': 'oidc-token',
                'DurationSeconds': 3600,
            }
            return {
                'Credentials': {
                    'AccessKeyId': 'access-key',
                    'SecretAccessKey': 'secret-key',
                    'SessionToken': 'session-token',
                }
            }

    def fake_client(service_name, region_name=None):
        assert service_name == 'sts'
        assert region_name == 'us-west-2'
        return FakeStsClient()

    monkeypatch.setenv('AWS_ROLE_ARN',
                       'arn:aws:iam::123456789012:role/kumo-test')
    monkeypatch.setattr(connector_utils, '_LOCAL_WEB_IDENTITY_TOKEN_PATH',
                        str(token_path))
    monkeypatch.setitem(sys.modules, 'boto3',
                        types.SimpleNamespace(client=fake_client))

    assert get_temporary_aws_credential('s3://bucket/path') == {
        'AccessKeyId': 'access-key',
        'SecretAccessKey': 'secret-key',
        'SessionToken': 'session-token',
    }


def test_web_identity_failure_never_logs_token_or_provider_body(
        tmp_path, monkeypatch, caplog):
    token = 'oidc-token-canary-do-not-log'
    provider_body = 'aws-provider-body secret-access-key-canary'
    token_path = tmp_path / 'token'
    token_path.write_text(token)

    class FakeStsClient:
        def assume_role_with_web_identity(self, **_kwargs):
            raise RuntimeError(f'{provider_body} token={token}')

    class FakeSession:
        def get_credentials(self):
            return None

    fake_boto3 = types.SimpleNamespace(
        client=lambda *_args, **_kwargs: FakeStsClient(),
        Session=FakeSession,
    )
    monkeypatch.setenv('AWS_ROLE_ARN',
                       'arn:aws:iam::123456789012:role/canary-role')
    monkeypatch.delenv('AWS_TEST_ROLE_ARN', raising=False)
    monkeypatch.setattr(connector_utils, '_LOCAL_WEB_IDENTITY_TOKEN_PATH',
                        str(token_path))
    monkeypatch.setitem(sys.modules, 'boto3', fake_boto3)

    with caplog.at_level('WARNING'):
        assert get_temporary_aws_credential('s3://bucket/path') == {}

    assert token not in caplog.text
    assert provider_body not in caplog.text
    assert 'RuntimeError' in caplog.text


def _disable_irsa_and_test_role(tmp_path, monkeypatch):
    """Force the ambient-credential fallback: no IRSA token file and no
    AWS_TEST_ROLE_ARN so both earlier branches return nothing."""
    monkeypatch.setattr(connector_utils, '_LOCAL_WEB_IDENTITY_TOKEN_PATH',
                        str(tmp_path / 'missing-token'))
    monkeypatch.delenv('AWS_TEST_ROLE_ARN', raising=False)


def test_get_temporary_aws_credential_ambient_creds_with_token(
        tmp_path, monkeypatch):
    """Singlehost fallback: when boto3's default chain yields temp creds with
    a session token, surface them for the credentialed COPY."""
    _disable_irsa_and_test_role(tmp_path, monkeypatch)

    frozen = types.SimpleNamespace(
        access_key='ambient-key',
        secret_key='ambient-secret',
        token='ambient-token',
    )

    class FakeSession:
        def get_credentials(self):
            return types.SimpleNamespace(
                get_frozen_credentials=lambda: frozen)

    monkeypatch.setitem(sys.modules, 'boto3',
                        types.SimpleNamespace(Session=FakeSession))

    assert get_temporary_aws_credential('s3://bucket/path') == {
        'AccessKeyId': 'ambient-key',
        'SecretAccessKey': 'ambient-secret',
        'SessionToken': 'ambient-token',
    }


def test_get_temporary_aws_credential_ambient_creds_without_token(
        tmp_path, monkeypatch):
    """Static instance creds (no session token) are unusable for Snowflake's
    AWS_TOKEN clause, so the fallback skips them and returns {}."""
    _disable_irsa_and_test_role(tmp_path, monkeypatch)

    frozen = types.SimpleNamespace(
        access_key='static-key',
        secret_key='static-secret',
        token=None,
    )

    class FakeSession:
        def get_credentials(self):
            return types.SimpleNamespace(
                get_frozen_credentials=lambda: frozen)

    monkeypatch.setitem(sys.modules, 'boto3',
                        types.SimpleNamespace(Session=FakeSession))

    assert get_temporary_aws_credential('s3://bucket/path') == {}


def test_get_temporary_aws_credential_passes_env_session_token(monkeypatch):
    class FakeStsClient:
        def assume_role(self, **kwargs):
            assert kwargs == {
                'RoleArn': 'arn:aws:iam::123456789012:role/kumo-test',
                'RoleSessionName': 'kumo-test-session',
                'DurationSeconds': 3600,
            }
            return {
                'Credentials': {
                    'AccessKeyId': 'assumed-access-key',
                    'SecretAccessKey': 'assumed-secret-key',
                    'SessionToken': 'assumed-session-token',
                }
            }

    def fake_client(service_name, **kwargs):
        assert service_name == 'sts'
        assert kwargs == {
            'aws_access_key_id': 'env-access-key',
            'aws_secret_access_key': 'env-secret-key',
            'aws_session_token': 'env-session-token',
        }
        return FakeStsClient()

    def fail_imds(*args, **kwargs):
        raise TimeoutError('no imds in ci')

    monkeypatch.setattr(connector_utils, '_LOCAL_WEB_IDENTITY_TOKEN_PATH',
                        '/does/not/exist')
    monkeypatch.setenv('AWS_TEST_ROLE_ARN',
                       'arn:aws:iam::123456789012:role/kumo-test')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'env-access-key')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'env-secret-key')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'env-session-token')
    monkeypatch.setattr(urllib.request, 'urlopen', fail_imds)
    monkeypatch.setitem(sys.modules, 'boto3',
                        types.SimpleNamespace(client=fake_client))

    assert get_temporary_aws_credential('s3://bucket/path') == {
        'AccessKeyId': 'assumed-access-key',
        'SecretAccessKey': 'assumed-secret-key',
        'SessionToken': 'assumed-session-token',
    }


def test_cached_batch_fetch():
    hour_cache = TTLCache(maxsize=10, ttl=3600)

    # Empty cache
    test_entity_names = ['1', '2', '3']
    cache_key = lambda x: f'test/{x}'

    cache_miss_counter = 0

    def test_fetch_fn(entity_names: List[str]) -> List[int]:
        nonlocal cache_miss_counter
        cache_miss_counter += len(entity_names)
        return [int(x) for x in entity_names]

    cached_output = cached_batch_fetch(test_entity_names, hour_cache,
                                       cache_key=cache_key,
                                       fetch_fn=test_fetch_fn)

    assert len(cached_output) == len(test_entity_names)
    assert cached_output == [1, 2, 3]
    # All 3 input values should have missed the cache
    assert cache_miss_counter == 3

    # Semi-filled cache with different ordering
    cache_miss_counter = 0
    expanded_test_entity_names = ['4', '1', '3', '5', '2']
    expanded_cached_output = cached_batch_fetch(expanded_test_entity_names,
                                                hour_cache,
                                                cache_key=cache_key,
                                                fetch_fn=test_fetch_fn)

    assert len(expanded_cached_output) == len(expanded_test_entity_names)
    assert expanded_cached_output == [4, 1, 3, 5, 2]
    # Only 2 input values should have missed the cache
    assert cache_miss_counter == 2

    # Fully filled cache with sorted ordering and repeats
    cache_miss_counter = 0
    sorted_expanded_test_entity_names = ['1', '2', '2', '3', '4', '5']
    expanded_cached_output_2 = cached_batch_fetch(
        sorted_expanded_test_entity_names, hour_cache, cache_key=cache_key,
        fetch_fn=test_fetch_fn)
    assert len(expanded_cached_output_2) == len(
        sorted_expanded_test_entity_names)
    assert expanded_cached_output_2 == [1, 2, 2, 3, 4, 5]
    # No values should miss the cache
    assert cache_miss_counter == 0


def test_cast_cols_to_str():
    ser = pd.Series([
        1680913671,
        1680013671,
        123.1,
        pd.NA,
    ])
    col = cast_col_to_str(ser)
    assert col[0] == '1680913671'
    assert col[1] == '1680013671'
    assert col[2] == '123.1'
    assert pd.isna(col[3])


@pytest.mark.parametrize('dtype,values,expected', [
    pytest.param(
        Dtype.time,
        [1680913671, 1680013671],
        ['2023-04-08 00:27:51', '2023-03-28 14:27:51'],
        id='epoch-seconds',
    ),
    pytest.param(
        Dtype.date,
        ['2023-04-08', '2023-03-28'],
        ['2023-04-08', '2023-03-28'],
        id='string-dates',
    ),
    pytest.param(
        Dtype.date,
        [20230408, 20230328],
        ['2023-04-08', '2023-03-28'],
        id='compact-dates',
    ),
    pytest.param(
        Dtype.time,
        [20230408120530, 20230328143045],
        ['2023-04-08 12:05:30', '2023-03-28 14:30:45'],
        id='compact-datetimes',
    ),
    pytest.param(
        Dtype.time,
        ['2023-04-08T12:05:30Z', '2023-03-28T14:30:45+02:00'],
        ['2023-04-08 12:05:30', '2023-03-28 12:30:45'],
        id='iso-offset-timestamps',
    ),
    pytest.param(
        Dtype.date,
        ['2023-04-08', 'not-a-date'],
        ['2023-04-08', pd.NaT],
        id='unparseable-value',
    ),
])
def test_cast_df_to_dtypes_temporal(dtype, values, expected):
    result = cast_df_to_dtypes(
        pd.DataFrame({'temporal': values}),
        {'temporal': dtype},
    )

    assert pd.api.types.is_datetime64_any_dtype(result['temporal'])
    pd.testing.assert_series_equal(
        result['temporal'],
        pd.Series(expected, name='temporal', dtype='datetime64[ns]'),
        check_dtype=False,
    )


def test_cast_df_to_dtypes_materializes_arrow_backed_temporal_column():
    temporal = pd.concat([
        pd.Series(['2023-04-08'], dtype='string[pyarrow]'),
        pd.Series(['2023-03-28'], dtype='string[pyarrow]'),
    ], ignore_index=True)
    assert temporal.array._pa_array.num_chunks == 2

    result = cast_df_to_dtypes(
        pd.DataFrame({'temporal': temporal}),
        {'temporal': Dtype.date},
    )

    assert pd.api.types.is_datetime64_any_dtype(result['temporal'])
    pd.testing.assert_series_equal(
        result['temporal'],
        pd.Series(
            ['2023-04-08', '2023-03-28'],
            name='temporal',
            dtype='datetime64[ns]',
        ),
        check_dtype=False,
    )
