# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep raw prediction reading complete under an explicit CPU resource budget."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/scripts/weekly_prediction_reader.py'


class Frame(list):
    pass


def database(monkeypatch, *, failure=False):
    calls = []
    frames = [
        Frame([dict(ENTITY=1, CLASS=2, SCORE=.8)]),
        Frame([dict(ENTITY=2, CLASS=1, SCORE=.3)])
    ]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append('closed')

        def execute(self, query):
            calls.append(query)
            return self

        def fetchone(self):
            return ('1.8 GiB', 1)

        def read_parquet(self, files):
            calls.append(('files', files))
            return self

        def order(self, order):
            calls.append(('order', order))
            return self

        def fetch_record_batch(self, rows):
            calls.append(('batch_rows', rows))
            if failure:
                raise RuntimeError('Retained output read failed')
            return [
                SimpleNamespace(to_pandas=lambda frame=frame: frame)
                for frame in frames
            ]

    def connect(*, config):
        calls.append(config)
        return Connection()

    monkeypatch.setitem(sys.modules, 'duckdb',
                        SimpleNamespace(connect=connect, __version__='1.5.5'))
    return calls, frames


def test_reader_preserves_native_stream_semantics_and_freezes_cpu_budget(
        tmp_path, monkeypatch):
    calls, frames = database(monkeypatch)
    reader = runpy.run_path(str(SCRIPT))
    scratch = tmp_path / 'sort'
    assert list(reader['ordered_batches'](['retained.parquet'],
                                          scratch)) == frames
    assert calls[0] == dict(memory_limit='2GB', threads=1,
                            temp_directory=str(scratch))
    assert ('order', 'ENTITY, SCORE DESC, CLASS ASC') in calls
    assert ('batch_rows', 65536) in calls
    receipt = json.loads((scratch / 'reader-receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['rows'] == 2
    assert receipt['actual_threads'] == 1 and calls[-1] == 'closed'


def test_reader_failure_closes_connection_and_preserves_failed_receipt(
        tmp_path, monkeypatch):
    calls, _ = database(monkeypatch, failure=True)
    with pytest.raises(RuntimeError):
        list(
            runpy.run_path(str(SCRIPT))['ordered_batches'](['retained.parquet'],
                                                           tmp_path / 'sort'))
    receipt = json.loads((tmp_path / 'sort/reader-receipt.json').read_text())
    assert receipt['status'] == 'FAIL' and receipt['rows'] == 0
    assert calls[-1] == 'closed'


def test_partial_consumer_never_claims_a_complete_read(tmp_path, monkeypatch):
    calls, _ = database(monkeypatch)
    batches = runpy.run_path(
        str(SCRIPT))['ordered_batches'](['retained.parquet'], tmp_path / 'sort')
    next(batches)
    batches.close()
    receipt = json.loads((tmp_path / 'sort/reader-receipt.json').read_text())
    assert receipt['status'] == 'STOPPED_BEFORE_COMPLETION' and receipt[
        'rows'] == 1
    assert calls[-1] == 'closed'


def test_existing_sort_evidence_cannot_be_reused_or_overwritten(
        tmp_path, monkeypatch):
    database(monkeypatch)
    scratch = tmp_path / 'preserved-sort'
    scratch.mkdir()
    marker = scratch / 'old-failure.json'
    marker.write_text('{"preserved":true}')
    with pytest.raises(FileExistsError):
        list(
            runpy.run_path(str(SCRIPT))['ordered_batches'](['retained.parquet'],
                                                           scratch))
    assert marker.read_text() == '{"preserved":true}'
