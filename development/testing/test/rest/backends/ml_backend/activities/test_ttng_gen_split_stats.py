# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from importlib import import_module

import pytest

mtt = import_module(
    'rest.backends.ml_backend.activities.materialize_training_task')


def _selected_train_table_path(tmp_path, job_id: str) -> str:
    bundle_dir = (tmp_path / 'jobs' / job_id / 'train_table.ttng_staging' /
                  '_bundles' / 'bundle-abc')
    bundle_dir.mkdir(parents=True)
    return str(bundle_dir / 'tt.parquet')


def _write_split_stats(staging_dir: str) -> None:
    os.makedirs(staging_dir, exist_ok=True)
    payload = {
        'version': 1,
        'kind': 'tt_split_stats',
        'train_table_stats': {
            'num_train_rows': 3,
            'num_entities': 3,
            'num_targets': 3,
            'target_distribution': {
                'count': {
                    '0': 2.0,
                    '1': 1.0
                },
                'percentage': {
                    '0': 66.6,
                    '1': 33.3
                },
            },
        },
        'val_table_stats': None,
        'test_table_stats': None,
    }
    with open(os.path.join(staging_dir, 'split_stats.json'), 'w') as f:
        json.dump(payload, f)


def test_gen_split_stats_written_to_job_dir_when_present(tmp_path):
    """C3: a persisted bundle WITH split_stats.json -> the gen-tt job dir gets
    split_train_table_stats.json (the file the /split_stats endpoint reads)."""
    from kumo_core.types import TaskType

    from kumo.home import HomeDirManager

    train_table_path = _selected_train_table_path(tmp_path, 'gen-tt-1')
    _write_split_stats(os.path.dirname(train_table_path))

    job_dir = str(tmp_path / 'jobs' / 'gen-tt-1')
    split_fpid = 'fp-abc'

    mtt._mirror_ttng_split_stats_to_job_dir(
        train_table_path=train_table_path,
        job_dir=job_dir,
        split_fpid=split_fpid,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    out_path = HomeDirManager(job_dir).split_train_table_stats(split_fpid)
    assert os.path.exists(out_path), (
        'C3 must write split_train_table_stats.json to the gen-tt job dir '
        'so the /split_stats endpoint returns non-null on TT-NG')
    with open(out_path) as f:
        written = json.load(f)
    # The endpoint deserializes this as SplitTrainTableStats.
    assert written['train_table_stats']['num_train_rows'] == 3


def test_gen_split_stats_noop_when_sidecar_absent(tmp_path):
    """A bundle without split_stats.json produces no job artifact."""
    from kumo_core.types import TaskType

    from kumo.home import HomeDirManager

    # The selected bundle exists but carries no native split-stats sidecar.
    train_table_path = _selected_train_table_path(tmp_path, 'gen-tt-1')

    job_dir = str(tmp_path / 'jobs' / 'gen-tt-1')
    split_fpid = 'fp-abc'

    mtt._mirror_ttng_split_stats_to_job_dir(
        train_table_path=train_table_path,
        job_dir=job_dir,
        split_fpid=split_fpid,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    out_path = HomeDirManager(job_dir).split_train_table_stats(split_fpid)
    assert not os.path.exists(out_path)


def test_gen_split_stats_noop_when_no_split(tmp_path, monkeypatch):
    """A no-split table does not require or look for a split sidecar."""
    def fail_on_exists(*args, **kwargs):
        del args, kwargs
        raise AssertionError('no-split tables must not inspect split artifacts')

    monkeypatch.setattr(mtt.fsutil, 'exists', fail_on_exists)

    mtt._mirror_ttng_split_stats_to_job_dir(
        train_table_path=str(tmp_path / 'train_table'),
        job_dir=str(tmp_path / 'job'),
        split_fpid=None,
        task_type=None,
    )


def _write_reconstructable_bundle(staging_dir: str) -> None:
    """Write an old-style bundle with stats/parquet but no split sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(staging_dir, exist_ok=True)
    # 4 train + 1 val + 1 test rows; static/random split via mask_counts.
    n = 6
    stats = {
        'version': 1,
        'kind': 'tt_stats',
        'num_total_rows': n,
        'payload': {
            'split_payload': {
                'mask_counts': {
                    'train': 4,
                    'val': 1,
                    'test': 1
                },
            },
        },
    }
    with open(os.path.join(staging_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    table = pa.table({
        'ENTITY1':
        pa.array(list(range(n)), type=pa.int64()),
        'TIMESTAMP':
        pa.array([1_700_000_000_000_000_000] * n, type=pa.int64()),
        'TARGET1':
        pa.array(['a', 'b', 'a', 'b', 'a', 'b']),
    })
    pq.write_table(table, os.path.join(staging_dir, 'tt.parquet'))


def test_missing_sidecar_does_not_read_reconstructable_bundle(
        tmp_path, monkeypatch):
    """An old reconstructable bundle must not trigger a parquet scan."""
    import pyarrow.parquet as pq
    from kumo_core.types import TaskType

    from kumo.home import HomeDirManager

    train_table_path = _selected_train_table_path(tmp_path, 'gen-tt-2')
    _write_reconstructable_bundle(os.path.dirname(train_table_path))

    job_dir = str(tmp_path / 'jobs' / 'gen-tt-2')
    split_fpid = 'fp-xyz'

    def fail_on_read(*args, **kwargs):
        del args, kwargs
        raise AssertionError('split-stats mirroring must not read tt.parquet')

    monkeypatch.setattr(mtt.fsutil, 'get_binary_handle', fail_on_read)
    monkeypatch.setattr(mtt.fsutil, 'read_text', fail_on_read)
    monkeypatch.setattr(pq, 'read_table', fail_on_read)

    mtt._mirror_ttng_split_stats_to_job_dir(
        train_table_path=train_table_path,
        job_dir=job_dir,
        split_fpid=split_fpid,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    out_path = HomeDirManager(job_dir).split_train_table_stats(split_fpid)
    assert not os.path.exists(out_path)


def test_malformed_sidecar_fails_without_writing_job_artifact(tmp_path):
    from kumo_core.types import TaskType

    from kumo.home import HomeDirManager

    train_table_path = _selected_train_table_path(tmp_path, 'gen-tt-3')
    staging_dir = os.path.dirname(train_table_path)
    with open(os.path.join(staging_dir, 'split_stats.json'), 'w') as f:
        json.dump({'version': 2, 'kind': 'tt_split_stats'}, f)

    job_dir = str(tmp_path / 'jobs' / 'gen-tt-3')
    split_fpid = 'fp-invalid'
    with pytest.raises(RuntimeError, match='version/kind mismatch'):
        mtt._mirror_ttng_split_stats_to_job_dir(
            train_table_path=train_table_path,
            job_dir=job_dir,
            split_fpid=split_fpid,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    out_path = HomeDirManager(job_dir).split_train_table_stats(split_fpid)
    assert not os.path.exists(out_path)
