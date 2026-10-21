# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Missing result files must not hide invalid stage or session errors."""
from types import SimpleNamespace

import pytest
from kumo_snowflake._ml_job_stage import StageStore, _stage_object_missing


@pytest.mark.parametrize('message', [
    'Stage does not exist or not authorized',
    'Session not found',
    'Object not found',
])
def test_stage_or_session_error_is_not_a_missing_file(message):
    error = RuntimeError(message)
    assert not _stage_object_missing(error)

    def get_stream(*args, **kwargs):
        raise error

    store = StageStore(
        SimpleNamespace(sql=lambda sql: None,
                        file=SimpleNamespace(get_stream=get_stream)),
        '@DB.PUBLIC.STAGE')
    with pytest.raises(RuntimeError, match=message):
        store.get_json('result.json', missing_ok=True)


@pytest.mark.parametrize('attribute', ['errno', 'sql_error_code'])
def test_connector_and_snowpark_missing_file_codes(attribute):
    error = RuntimeError('the file does not exist')
    setattr(error, attribute, 253006)
    assert _stage_object_missing(error)
