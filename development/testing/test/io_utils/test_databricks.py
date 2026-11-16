# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from io_utils import databricks, fsutil


@pytest.mark.parametrize(
    ('failure_type', 'expected_type'),
    [(RuntimeError, RuntimeError), (ValueError, ValueError)],
)
def test_databricks_parquet_head_sanitizes_provider_failure(
        monkeypatch, caplog, failure_type, expected_type):
    canary = 'databricks-provider-body token=parquet-secret-canary'
    monkeypatch.setattr(fsutil, 'isdir', lambda _path: False)
    monkeypatch.setattr(
        databricks,
        '_execute_query',
        lambda _query: (_ for _ in ()).throw(failure_type(canary)),
    )

    with pytest.raises(expected_type) as exc_info:
        databricks.get_parquet_head_databricks('dbfs:/Volumes/safe/table')

    error = exc_info.value
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert canary not in str(error)
    assert canary not in repr(error)
    assert canary not in caplog.text


@pytest.mark.parametrize('allowed_path', [None, '/tmp/prediction-output'])
def test_sql_connection_limits_put_local_file_access(allowed_path):
    client = databricks.DatabricksClient(
        databricks.DatabricksConnectionInfo(
            host='https://workspace.example',
            cluster_id='',
            warehouse_id='warehouse-id',
            catalog='catalog',
            credentials={'pat': SecretStr('token')},
        ))
    with patch('databricks.sql.connect', return_value=object()) as connect:
        client.create_sql_connection(
            schema='schema',
            staging_allowed_local_path=allowed_path,
        )

    kwargs = connect.call_args.kwargs
    if allowed_path is None:
        assert 'staging_allowed_local_path' not in kwargs
    else:
        assert kwargs['staging_allowed_local_path'] == allowed_path
