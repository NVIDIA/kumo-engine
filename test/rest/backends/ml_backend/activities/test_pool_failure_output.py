# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pool.failure_output import MAX_CHARS, failure_output_path
from rest.backends.ml_backend.activities.pool_failure_output import (
    read_pool_failure_output,
)


def test_read_pool_failure_output_returns_and_logs_saved_output(
        tmp_path, caplog):
    config = str(tmp_path / 'trainingjob-1.json')
    path = failure_output_path(config, 'kumo-auto')
    with open(path, 'w') as handle:
        handle.write('Traceback (most recent call last):\n'
                     'ValueError: bad input\n')

    with caplog.at_level('ERROR'):
        output = read_pool_failure_output('trainingjob-1', [path])

    assert output == ('Traceback (most recent call last):\n'
                      'ValueError: bad input')
    assert any('trainingjob-1' in record.getMessage()
               and 'ValueError: bad input' in record.getMessage()
               for record in caplog.records)


def test_read_pool_failure_output_labels_each_shard(tmp_path):
    config = str(tmp_path / 'batchpredictionjob-1.json')
    paths = [
        failure_output_path(config, 'kumo-pred', index) for index in (0, 2)
    ]
    for index, path in zip((0, 2), paths):
        with open(path, 'w') as handle:
            handle.write(f'RuntimeError: shard {index}\n')

    output = read_pool_failure_output('batchpredictionjob-1', paths)

    assert output == (
        '[batchpredictionjob-1.json.kumo-pred.shard-00000.failure.txt]\n'
        'RuntimeError: shard 0\n'
        '[batchpredictionjob-1.json.kumo-pred.shard-00002.failure.txt]\n'
        'RuntimeError: shard 2')


def test_read_pool_failure_output_skips_missing_and_empty_files(tmp_path):
    empty = tmp_path / 'empty.failure.txt'
    empty.write_text('  \n')

    assert read_pool_failure_output(
        'job-1', [str(tmp_path / 'missing.failure.txt'),
                  str(empty)]) is None


def test_read_pool_failure_output_skips_unreadable_files(
        tmp_path, monkeypatch, caplog):
    path = tmp_path / 'job.failure.txt'
    path.write_text('RuntimeError: boom\n')

    def _fail_read(*_args, **_kwargs):
        raise OSError('storage unavailable')

    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.pool_failure_output.fsutil'
        '.read_text', _fail_read)

    with caplog.at_level('WARNING'):
        assert read_pool_failure_output('job-1', [str(path)]) is None
    assert any(
        'Could not read the saved pool failure output' in record.getMessage()
        for record in caplog.records)


def test_read_pool_failure_output_keeps_the_end_of_large_output(tmp_path):
    path = tmp_path / 'job.failure.txt'
    path.write_text('x' * (2 * MAX_CHARS) + '\nValueError: end')

    output = read_pool_failure_output('job-1', [str(path)])

    assert output is not None
    assert len(output) == MAX_CHARS
    assert output.endswith('ValueError: end')
