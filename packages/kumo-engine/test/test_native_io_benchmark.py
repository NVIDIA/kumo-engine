# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for controlled-work training and prediction benchmarks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from native_io_tasks import (  # noqa: E402
    arguments,
    check_work,
    file_hash,
    file_tree,
    generate_data,
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
