# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from io_utils.databricks import DatabricksConnectionInfo, validate_host
from io_utils.fsutil.uri import LocalURI, ParseURI
from service_framework.deploy_env import (
    DATABRICKS_HOST,
    DATABRICKS_WAREHOUSE_ID,
)


def test_db_connection_info_host():
    info = DatabricksConnectionInfo(host='host',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    info = DatabricksConnectionInfo(host='https://host/',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    info = DatabricksConnectionInfo(
                                    host='https://host/login.html',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'
    # Test http scheme is converted to https
    info = DatabricksConnectionInfo(host='http://host',
                                    cluster_id='cluster',
                                    warehouse_id='warehouse',
                                    catalog='catalog', credentials={})
    assert info.host == 'https://host'


def test_connection_info_load(monkeypatch):
    monkeypatch.setenv('DATABRICKS_HOST', 'https://example.invalid')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'synthetic-pat')
    monkeypatch.setenv('DATABRICKS_WAREHOUSE_ID', 'synthetic-warehouse')
    monkeypatch.setenv('DATABRICKS_CLUSTER_ID', '')
    info = DatabricksConnectionInfo.from_environment()
    assert info.host == validate_host(DATABRICKS_HOST.parse_value())
    assert info.credentials.get('pat') is not None
    assert info.warehouse_id == DATABRICKS_WAREHOUSE_ID.parse_value()
    assert not info.cluster_id
    assert not info.catalog


def test_is_auth_error():
    """_is_auth_error matches invalid_client errors and nothing else."""
    from io_utils.fsutil.impl_ucv import _is_auth_error
    assert _is_auth_error(ValueError("invalid_client: Client authentication failed"))
    assert _is_auth_error(Exception("invalid_client"))
    assert not _is_auth_error(ValueError("some other error"))
    assert not _is_auth_error(RuntimeError("invalid_request"))


def test_uc_context_reconnect_clears_cached_state():
    """reconnect() clears both the cached client and connection_info."""
    from io_utils.fsutil.impl_ucv import UCContext
    ctx = UCContext()
    ctx._local.client = object()
    ctx._connection_info = DatabricksConnectionInfo(
        host='https://h', cluster_id='c', warehouse_id='w',
        catalog='', credentials={})
    ctx.reconnect()
    assert ctx._local.client is None
    assert ctx._connection_info is None


def test_retry_policy_reconnects_on_auth_failure(caplog, monkeypatch):
    """retry_policy invokes uc_ctx.reconnect() on invalid_client errors,
    then retries the operation."""
    from io_utils.fsutil import impl_ucv
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _delay: None)

    call_count = 0
    canary = 'invalid_client: token=ucv-auth-secret-canary'

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError(canary)
        return "success"

    with patch.object(impl_ucv.uc_ctx, 'reconnect') as mock_reconnect:
        result = operation()

    assert result == "success"
    assert call_count == 2
    mock_reconnect.assert_called_once()
    assert canary not in caplog.text
    assert 'ucv-auth-secret-canary' not in caplog.text


def test_upload_retry_log_never_exposes_provider_error(monkeypatch, caplog):
    from io_utils.fsutil import impl_ucv

    canary = 'databricks-provider-body token=ucv-upload-secret-canary'
    files = MagicMock()
    files.upload_from.side_effect = [RuntimeError(canary), None]
    client = MagicMock(files=files)
    monkeypatch.setattr(impl_ucv, '_check_size_match', lambda **_kwargs: None)
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _delay: None)
    local = LocalURI(path='/tmp/local-artifact')
    remote = ParseURI('dbfs:/Volumes/catalog/schema/volume/artifact')

    with patch.object(type(impl_ucv.uc_ctx), 'client', new_callable=PropertyMock,
                      return_value=client):
        impl_ucv._upload(local=local, remote=remote)

    assert files.upload_from.call_count == 2
    assert canary not in caplog.text
    assert 'ucv-upload-secret-canary' not in caplog.text


def test_retry_policy_does_not_reconnect_on_other_errors(monkeypatch):
    """retry_policy does NOT call reconnect for non-auth errors (e.g.
    transient RuntimeError). It still retries via the base retry."""
    from io_utils.fsutil import impl_ucv
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _delay: None)

    call_count = 0

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")
        return "success"

    with patch.object(impl_ucv.uc_ctx, 'reconnect') as mock_reconnect:
        result = operation()

    assert result == "success"
    assert call_count == 2
    mock_reconnect.assert_not_called()


def test_retry_policy_gives_up_after_max_tries(monkeypatch):
    """If auth failure persists across all retries, retry_policy still
    raises after exhausting attempts."""
    from io_utils.fsutil import impl_ucv
    monkeypatch.setattr('io_utils.retry.time.sleep', lambda _delay: None)

    call_count = 0

    @impl_ucv.retry_policy()
    def operation():
        nonlocal call_count
        call_count += 1
        raise ValueError("invalid_client: Client authentication failed")

    with patch.object(impl_ucv.uc_ctx, 'reconnect'):
        with pytest.raises(ValueError, match="invalid_client"):
            operation()

    assert call_count == 3
