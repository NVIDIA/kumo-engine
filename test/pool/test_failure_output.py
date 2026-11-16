# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pool.failure_output import (
    MAX_CHARS,
    MAX_LINE_CHARS,
    MAX_SUMMARY_CHARS,
    OutputTail,
    failure_output_path,
    failure_summary,
)


def test_output_tail_keeps_last_non_empty_lines():
    tail = OutputTail(max_lines=2)
    for record in [b'first', b'', b'   ', b'second\r', b'third']:
        tail.add(record)

    assert tail.text() == 'second\nthird'


def test_output_tail_bounds_lines_and_total_size():
    tail = OutputTail(max_lines=100)
    tail.add(b'x' * (MAX_LINE_CHARS + 10))
    assert tail.text() == 'x' * MAX_LINE_CHARS

    for index in range(100):
        tail.add(f'{index:04d}'.encode() + b'y' * (MAX_LINE_CHARS - 4))
    text = tail.text()
    assert len(text) == MAX_CHARS
    assert text.endswith('0099' + 'y' * (MAX_LINE_CHARS - 4))


def test_output_tail_replaces_invalid_utf8():
    tail = OutputTail()
    tail.add(b'bad \xff byte')

    assert tail.text() == 'bad � byte'


def test_failure_output_path_is_next_to_the_shared_job_config():
    config = 'az://data/home/jobs/trainingjob-1.json'

    assert failure_output_path(config, 'kumo-auto') == (
        'az://data/home/jobs/trainingjob-1.json.kumo-auto.failure.txt')
    assert failure_output_path(
        config, 'kumo-pred',
        3) == ('az://data/home/jobs/trainingjob-1.json.kumo-pred.shard-00003'
               '.failure.txt')


def test_failure_summary_prefers_the_last_exception_line():
    output = '\n'.join([
        'Traceback (most recent call last):',
        '  File "train.py", line 3, in <module>',
        "kumo_core.errors.DataError: column 'amount' is missing",
        'INFO shutting down',
    ])

    assert failure_summary(output) == (
        "kumo_core.errors.DataError: column 'amount' is missing")


def test_failure_summary_falls_back_to_the_last_line():
    assert failure_summary('loading\nKilled\n\n') == 'Killed'
    assert failure_summary('  \n') is None


def test_failure_summary_is_bounded():
    summary = failure_summary('RuntimeError: ' + 'z' * (2 * MAX_SUMMARY_CHARS))

    assert summary is not None
    assert len(summary) == MAX_SUMMARY_CHARS
