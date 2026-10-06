# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the in-process DiskGraph ingest error boundary."""
from __future__ import annotations

from importlib import import_module

import pytest

from statedb.utils.errors import DiskgraphError

_materialize = import_module(
    'rest.backends.ml_backend.activities.materialize_training_task')
_ensure_extension = _materialize._ttng_ensure_train_table_extension
_NATIVE_MESSAGE = 'cannot open /mnt/localssd/idx/train_tables/x for read: ENOENT'
_TRAIN_TABLE_PATH = (
    's3://bucket/jobs/job/train_table.ttng_staging/'
    '_bundles/bundle-abc/tt.parquet')


class _FakeDiskgraphSystemError(Exception):
    pass


class _FakeDiskgraphDataError(Exception):
    pass


class _FakeDiskgraphConfigError(Exception):
    pass


@pytest.fixture
def native_error_boundary(monkeypatch):
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NATIVE_ERRORS',
                        (_FakeDiskgraphSystemError, ))
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NON_RETRYABLE_ERRORS', ())

    def _raise_native(**_kwargs):
        raise _FakeDiskgraphSystemError(_NATIVE_MESSAGE)

    monkeypatch.setattr(
        'kumo.data.generic.diskgraph_staging.ensure_stage_ingested',
        _raise_native,
    )


def _call() -> DiskgraphError:
    with pytest.raises(DiskgraphError) as caught:
        _ensure_extension(
            train_table_path=_TRAIN_TABLE_PATH,
            diskgraph_index_path='/mnt/localssd/idx',
            ttng_tt_id='tt-abc123',
        )
    return caught.value


def test_native_error_is_converted_and_keeps_its_cause(
        native_error_boundary) -> None:
    error = _call()
    assert isinstance(error.__cause__, _FakeDiskgraphSystemError)


def test_converted_error_carries_native_and_boundary_context(
        native_error_boundary) -> None:
    error = _call()
    assert _NATIVE_MESSAGE in error.message
    assert _NATIVE_MESSAGE in error.get_user_message()
    assert 'train_tables extension' in error.message
    assert 'tt-abc123' in error.message
    assert '_bundles/bundle-abc' in error.message


def test_system_error_remains_retryable(
        native_error_boundary) -> None:
    assert _call().non_retryable is False


@pytest.mark.parametrize('error_type',
                         [_FakeDiskgraphDataError, _FakeDiskgraphConfigError])
def test_deterministic_native_error_is_terminal(monkeypatch,
                                                error_type) -> None:
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NATIVE_ERRORS', (error_type, ))
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NON_RETRYABLE_ERRORS',
                        (error_type, ))

    def _raise_native(**_kwargs):
        raise error_type(_NATIVE_MESSAGE)

    monkeypatch.setattr(
        'kumo.data.generic.diskgraph_staging.ensure_stage_ingested',
        _raise_native,
    )

    assert _call().non_retryable is True


def test_absent_diskgraph_wheel_leaves_mapping_inert(monkeypatch) -> None:
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NATIVE_ERRORS', ())
    monkeypatch.setattr(_materialize, '_DISKGRAPH_NON_RETRYABLE_ERRORS', ())

    def _raise_native(**_kwargs):
        raise _FakeDiskgraphSystemError(_NATIVE_MESSAGE)

    monkeypatch.setattr(
        'kumo.data.generic.diskgraph_staging.ensure_stage_ingested',
        _raise_native,
    )

    with pytest.raises(_FakeDiskgraphSystemError):
        _ensure_extension(
            train_table_path=_TRAIN_TABLE_PATH,
            diskgraph_index_path='/mnt/localssd/idx',
            ttng_tt_id='tt-abc123',
        )
