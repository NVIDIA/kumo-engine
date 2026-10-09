# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace

import pytest

split_mod = import_module('rest.backends.ml_backend.activities.split')

# ``execute_split`` is a plain ``@activity.defn`` now that it runs on the
# data-backend thread pool (no ``process_pool_activity_defn`` wrapper).
# ``inspect.unwrap`` walks any ``__wrapped__`` chain back to the raw,
# synchronous implementation so we can call it directly in a unit test
# without a Temporal activity context.
_execute_split = inspect.unwrap(split_mod.execute_split)


@contextmanager
def _patched_common(monkeypatch, *, calls, manifest):
    """Patch the heavy, environment-coupled dependencies of ``execute_split``.

    ``calls`` is mutated to record which load/save paths fired so each test can
    assert on the control flow.
    """
    class _FakeHomeDirManager:
        materialized_train_table_config_path = '/tmp/fake-config.json'

        def __init__(self, _home_dir):
            pass

    monkeypatch.setattr(split_mod, 'HomeDirManager', _FakeHomeDirManager)
    monkeypatch.setattr(split_mod, 'load_json',
                        lambda *a, **k: SimpleNamespace())

    def _save_data(path):
        calls['save'] = path

    fake_train_table = SimpleNamespace(save_data=_save_data)
    monkeypatch.setattr(split_mod.TrainingTable, 'from_config',
                        classmethod(lambda cls, cfg: fake_train_table))

    def _validate(manifest_path):
        calls['validate'] = manifest_path
        return manifest, {}, 0, b''

    monkeypatch.setattr(split_mod, '_validate_native_split_manifest', _validate)

    def _apply_stats(train_table, manifest_path, prevalidated_manifest=None):
        calls['apply_stats'] = (manifest_path, prevalidated_manifest)

    monkeypatch.setattr(split_mod, '_apply_native_split_stats', _apply_stats)

    yield fake_train_table


def _make_job_input():
    return SimpleNamespace(
        graph_key='gk',
        graph_data_version='gdv',
        generate_train_table_job_id='gid',
        materialized_train_table_data_path='/tmp/data',
        training_spec=None,
        connector_refs={},
        diskgraph_manifest_path='/tmp/manifest.json',
    )


def test_ttng_split_skips_load_and_save(monkeypatch):
    calls: dict = {}
    manifest = {'parsed_split_stats': {}}
    with _patched_common(monkeypatch, calls=calls, manifest=manifest):
        status = _execute_split(
            _make_job_input(),
            materialized_split_root_dir='/tmp/split-root',
            home_dir='/tmp/home',
        )

    # Heavy I/O must NOT happen on the TT-NG path.
    assert 'load' not in calls
    assert 'save' not in calls
    # Manifest validation + split stats DO happen.
    assert calls['validate'] == '/tmp/manifest.json'
    assert calls['apply_stats'] == ('/tmp/manifest.json', manifest)
    # Returns DONE with no materialized parquet.
    assert status.status == split_mod.JobStatus.DONE
    assert status.materialized_path is None


def test_ttng_split_failure_returns_failed(monkeypatch):
    calls: dict = {}
    with _patched_common(monkeypatch, calls=calls, manifest={}):

        def _boom(*a, **k):
            raise RuntimeError('bad manifest')

        monkeypatch.setattr(split_mod, '_validate_native_split_manifest', _boom)
        status = _execute_split(
            _make_job_input(),
            materialized_split_root_dir='/tmp/split-root',
            home_dir='/tmp/home',
        )

    assert status.status == split_mod.JobStatus.FAILED
    assert 'load' not in calls
    assert 'save' not in calls


def test_validate_native_split_manifest_extracts_payload(monkeypatch):
    manifest = {
        'split': {
            'authoritative': True
        },
        'parsed_split': {
            'row_count': 3,
            'stage_ids': b'\x00\x01\x02'
        },
    }
    monkeypatch.setattr(
        split_mod, 'get_service',
        lambda: SimpleNamespace(validate_manifest=lambda p: manifest))
    out_manifest, parsed, row_count, stage_ids = (
        split_mod._validate_native_split_manifest('/tmp/manifest.json'))
    assert out_manifest is manifest
    assert parsed == manifest['parsed_split']
    assert row_count == 3
    assert stage_ids == b'\x00\x01\x02'


def test_validate_native_split_manifest_rejects_non_authoritative(monkeypatch):
    manifest = {
        'split': {
            'authoritative': False
        },
        'parsed_split': {
            'row_count': 0,
            'stage_ids': b''
        },
    }
    monkeypatch.setattr(
        split_mod, 'get_service',
        lambda: SimpleNamespace(validate_manifest=lambda p: manifest))
    with pytest.raises(RuntimeError, match='non-authoritative'):
        split_mod._validate_native_split_manifest('/tmp/manifest.json')
