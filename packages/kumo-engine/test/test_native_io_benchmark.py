# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for controlled-work training and prediction benchmarks."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from io_utils import fsutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from native_io_isolation import (  # noqa: E402
    clone_prepared,
    file_hash,
    file_tree,
    validate_clone,
    verify_baseline,
)
from native_io_tasks import (  # noqa: E402
    arguments,
    check_work,
    generate_data,
    logical_seed_hash,
    read_table,
)


def test_generated_workloads(tmp_path):
    roots = [tmp_path / name for name in ('one', 'two', 'hot', 'community')]
    counts = [generate_data(root, task, scale=1, width=4, seed=0)
              for root, task in zip(roots, ('binary', 'binary', 'regression',
                                            'temporal_link'))]
    assert counts == [counts[0]] * 4
    assert counts[0]['users'] == 128
    assert counts[0]['items'] == 256
    files = [file_tree(str(root)) for root in roots]
    assert [file_hash(path) for path in files[0]] == [
        file_hash(path) for path in files[1]]
    events = [read_table(str(root / 'events/part-000000.parquet'))
              for root in roots]
    item = [np.array(table['item_id']) for table in events]
    assert np.array_equal(item[0], item[1])
    assert not np.array_equal(item[0], item[2])
    assert not np.array_equal(item[0], item[3])
    assert (item[2] < 2).mean() > .75
    for table in events:
        assert {len(row) for row in table['features'].to_pylist()} == {4}
        assert table.num_rows > 0


def test_completed_work():
    state = dict(fits=1, completed_fits=1, optimizer_steps=8, train_batches=8,
                 predict_batches={'predict': 1, 'repeat': 1, 'reload': 1})
    check_work(state, 8)
    check_work({**state, 'optimizer_count': 2, 'optimizer_steps': 16}, 8)
    for key in ('fits', 'completed_fits', 'optimizer_steps', 'train_batches'):
        with pytest.raises(ValueError, match='fixed plan'):
            check_work({**state, key: 0}, 8)
    with pytest.raises(ValueError, match='fixed plan'):
        check_work({**state, 'predict_batches': {'predict': 1, 'repeat': 0}}, 8)
    with pytest.raises(ValueError, match='fixed plan'):
        check_work({**state, 'predict_batches': {}}, 8)


@pytest.mark.parametrize('option', [
    '--steps', '--scale', '--feature-width', '--threads', '--rayon-threads',
    '--batch-size',
])
def test_invalid_budget(option):
    with pytest.raises(SystemExit):
        arguments(['--task', 'binary', '--prepared-dir', '/tmp/prepared',
                   '--output', '/tmp/result.json', option, '0'])


def test_missing_outputs():
    with pytest.raises(SystemExit):
        arguments(['--task', 'binary', '--prepared-dir', '/tmp/prepared'])
    with pytest.raises(SystemExit):
        arguments(['--task', 'binary', '--prepared-dir', '/tmp/prepared',
                   '--prepare'])


def test_logical_seed_fingerprint():
    batch = HeteroData()
    batch['users'].n_id = torch.tensor([11, 12, 99])
    batch['train_table'].batch_size = 2
    batch['train_table'].time = torch.tensor([100, 200])
    batch['train_table'].y = torch.tensor([0., 1.])
    expected = logical_seed_hash(batch)
    assert len(expected) == 64
    assert logical_seed_hash(batch.clone()) == expected
    for store, name in [('users', 'n_id'), ('train_table', 'time'),
                        ('train_table', 'y')]:
        changed = batch.clone()
        changed[store][name][0] += 1
        assert logical_seed_hash(changed) != expected
    # Extra sampled neighbors do not change which training seeds were used.
    batch['users'].n_id[2] += 1
    assert logical_seed_hash(batch) == expected


def test_private_trial_copy(tmp_path):
    baseline, trial = tmp_path / 'baseline', tmp_path / 'trial'
    fsutil.makedirs(str(baseline / 'index'))
    data = str(baseline / 'index/data.bin')
    fsutil.write_text(data, 'original graph bytes')
    stage = str(baseline / 'index/stage.json')
    fsutil.write_text(stage, '{"path":"data.bin"}')
    original = dict(
        config_sha256='fixed', source='/separate/immutable/source',
        snapshot={'index_path': str(baseline / 'index')},
        training={'staging_dir': str(baseline / 'index'),
                  'train_table_path': data, 'manifest_path': stage},
        inputs=data, owned_files=[data, stage], index_files=[data, stage],
    )
    manifest_path = str(baseline / 'manifest.json')
    fsutil.write_text(manifest_path, json.dumps(original))
    before = file_hash(manifest_path)
    with pytest.raises(ValueError, match='exactly cover'):
        clone_prepared(str(baseline), str(tmp_path / 'partial'), {
            'index/data.bin': file_hash(data),
        })
    manifest = clone_prepared(str(baseline), str(trial))
    validate_clone(manifest, str(trial))
    assert manifest['config_sha256'] == 'fixed'
    assert manifest['source'] == original['source']
    assert manifest['training']['manifest_path'] == str(trial / 'index/stage.json')
    assert manifest['isolation']['source_manifest_sha256'] == before
    assert os.stat(data).st_ino != os.stat(trial / 'index/data.bin').st_ino
    fsutil.write_text(str(trial / 'index/data.bin'), 'trial may modify this')
    verify_baseline(manifest)
    assert fsutil.read_text(data) == 'original graph bytes'
    assert file_hash(manifest_path) == before
    fsutil.write_text(data, 'modified graph bytes')
    with pytest.raises(ValueError, match='baseline changed'):
        verify_baseline(manifest)
    fsutil.write_text(data, 'original graph bytes')
    fsutil.write_text(str(baseline / 'new-unexpected-file'), 'mutation')
    with pytest.raises(ValueError, match='baseline changed'):
        verify_baseline(manifest)


def test_clone_rejects_escape(tmp_path):
    baseline = tmp_path / 'baseline'
    fsutil.makedirs(str(baseline))
    outside = str(tmp_path / 'not-owned')
    fsutil.write_text(outside, 'unrelated')
    fsutil.write_text(str(baseline / 'manifest.json'), json.dumps({
        'owned_files': [outside],
    }))
    with pytest.raises(ValueError, match='escapes the baseline'):
        clone_prepared(str(baseline), str(tmp_path / 'trial'))
