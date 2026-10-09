# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(kiriti): Make this test_json.py independent of anything Kumo.
import logging
import multiprocessing
import os
import stat
import threading
import time
from os import path as osp
from tempfile import TemporaryDirectory
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from kumoapi.typing import Dtype

from io_utils import fsutil
from io_utils.json import (
    escape_dataclass,
    from_json,
    json_safe_float,
    json_safe_float_array,
    load_json,
    save_json,
    save_json_atomic,
    to_json,
)
from kumo.config import ColumnConfig, TableMetadataConfig
from kumo.graph import Column, Table


@pytest.fixture
def test_dir() -> Generator[str, None, None]:
    with TemporaryDirectory() as d:
        yield d


def test_save_and_load(test_dir):
    col = Column(name='LOAN_ID', dtype=Dtype.int)
    json_path = osp.join(test_dir, 'test.yml')
    save_json(col.config(), json_path)

    assert col.config() == load_json(json_path, ColumnConfig)


def test_save_and_load_compressed(test_dir):
    col = Column(name='LOAN_ID', dtype=Dtype.int)
    json_path = osp.join(test_dir, 'test.json')
    save_json(col.config(), json_path, compressed=True)

    assert col.config() == load_json(json_path, ColumnConfig, compressed=True)


def test_save_json_default_keeps_direct_write(monkeypatch, test_dir):
    path = osp.join(test_dir, 'default.json')
    write = MagicMock()
    replace = MagicMock()
    monkeypatch.setattr(fsutil, 'write_text', write)
    monkeypatch.setattr(os, 'replace', replace)

    save_json({'value': 1}, path)

    write.assert_called_once()
    assert write.call_args.args[:2] == (path, to_json({'value': 1}))
    replace.assert_not_called()


def _publish_after_process_barrier(path, ready_path, allow_path):
    original_replace = os.replace

    def paused_replace(src, dst):
        with open(ready_path, 'w', encoding='utf-8') as ready_file:
            ready_file.write('ready')
        deadline = time.monotonic() + 10
        while not os.path.exists(allow_path):
            if time.monotonic() >= deadline:
                raise TimeoutError('parent did not release atomic replace')
            time.sleep(0.01)
        original_replace(src, dst)

    os.replace = paused_replace
    save_json_atomic({'value': 'new'}, path)


def test_save_json_atomic_keeps_old_value_visible_across_process_replace(
        test_dir):
    path = osp.join(test_dir, 'atomic.json')
    save_json_atomic({'value': 'old'}, path)
    context = multiprocessing.get_context('spawn')
    ready_path = osp.join(test_dir, 'replace.ready')
    allow_path = osp.join(test_dir, 'replace.allow')
    writer = context.Process(
        target=_publish_after_process_barrier,
        args=(path, ready_path, allow_path),
    )
    writer.start()
    deadline = time.monotonic() + 10
    while not os.path.exists(ready_path):
        assert writer.is_alive()
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert load_json(path, dict) == {'value': 'old'}
    with open(allow_path, 'w', encoding='utf-8') as allow_file:
        allow_file.write('allow')
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert writer.exitcode == 0
    assert load_json(path, dict) == {'value': 'new'}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_save_json_atomic_concurrent_readers_never_see_partial_document(
        test_dir):
    path = osp.join(test_dir, 'concurrent.json')
    save_json_atomic({'generation': 0, 'payload': '0' * 8192}, path)
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                value = load_json(path, dict)
                generation = value['generation']
                assert value['payload'] == str(generation) * 8192
            except Exception as error:
                errors.append(error)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    for generation in range(1, 33):
        save_json_atomic(
            {
                'generation': generation,
                'payload': str(generation) * 8192,
            }, path)
    stop.set()
    for thread in readers:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []


def test_save_json_atomic_failure_preserves_target_and_cleans_temp(
        monkeypatch, test_dir):
    path = osp.join(test_dir, 'atomic.json')
    save_json_atomic({'secret': 'old'}, path)

    def reject_replace(_src, _dst):
        raise PermissionError('do not expose temporary path')

    monkeypatch.setattr(os, 'replace', reject_replace)
    with pytest.raises(RuntimeError, match='Failed to atomically save') as exc:
        save_json_atomic({'secret': 'must-not-appear'}, path)

    assert 'must-not-appear' not in str(exc.value)
    assert '.kumo-json-' not in str(exc.value)
    assert load_json(path, dict) == {'secret': 'old'}
    assert sorted(os.listdir(test_dir)) == ['atomic.json']


@pytest.mark.parametrize('scheme,path', [
    ('s3', 's3://test-bucket/job/inputs/input.json'),
    ('gs', 'gs://test-bucket/job/inputs/input.json'),
])
def test_save_json_atomic_remote_uses_single_final_key_write(
        monkeypatch, scheme, path):
    write = MagicMock()
    mkstemp = MagicMock(side_effect=AssertionError(
        'remote publication must not create a temporary object'))
    monkeypatch.setattr(fsutil, 'makedirs', MagicMock())
    monkeypatch.setattr(fsutil, 'get_scheme', MagicMock(return_value=scheme))
    monkeypatch.setattr(fsutil, 'write_text', write)
    monkeypatch.setattr('io_utils.json.tempfile.mkstemp', mkstemp)

    save_json_atomic({'value': 1}, path)

    write.assert_called_once_with(path, to_json({'value': 1}),
                                  compression=False)
    mkstemp.assert_not_called()


@pytest.mark.parametrize('scheme', ['abfs', 'abfss'])
def test_save_json_atomic_adls_uses_provider_atomic_write(monkeypatch, scheme):
    path = (f'{scheme}://filesystem@account.dfs.core.windows.net/'
            'job/inputs/input.json')
    atomic_write = MagicMock()
    monkeypatch.setattr(fsutil, 'makedirs', MagicMock())
    monkeypatch.setattr(fsutil, 'get_scheme', MagicMock(return_value=scheme))
    monkeypatch.setattr(fsutil, 'atomic_write_text', atomic_write)

    save_json_atomic({'value': 1}, path)

    atomic_write.assert_called_once_with(path, to_json({'value': 1}),
                                         compression=False)


def test_save_json_atomic_compressed_roundtrip(test_dir):
    path = osp.join(test_dir, 'atomic.json.gz')
    save_json_atomic({'value': 1}, path, compressed=True)
    assert load_json(path, dict, compressed=True) == {'value': 1}


def test_save_json_atomic_rejects_symlink(test_dir):
    real_path = osp.join(test_dir, 'real.json')
    link_path = osp.join(test_dir, 'link.json')
    save_json({'value': 'old'}, real_path)
    os.symlink(real_path, link_path)

    with pytest.raises(RuntimeError, match='Failed to atomically save'):
        save_json_atomic({'value': 'new'}, link_path)

    assert load_json(real_path, dict) == {'value': 'old'}
    assert os.path.islink(link_path)


@pytest.mark.parametrize('scheme,path', [
    ('dbfs', 'dbfs:/Volumes/catalog/schema/volume/job/inputs/input.json'),
    ('unsupported', 'unsupported://input.json'),
])
def test_save_json_atomic_fails_closed_without_proven_atomic_replace(
        monkeypatch, scheme, path):
    write = MagicMock()
    monkeypatch.setattr(fsutil, 'get_scheme', MagicMock(return_value=scheme))
    monkeypatch.setattr(fsutil, 'makedirs', MagicMock())
    monkeypatch.setattr(fsutil, 'write_text', write)

    with pytest.raises(RuntimeError, match='Failed to atomically save'):
        save_json_atomic({'value': 1}, path)
    write.assert_not_called()


def test_escape_dataclass(create_graph):
    graph = create_graph(name='FINANCIAL')
    table = graph['LOAN']
    table.infer_metadata()
    table.infer_encoders()

    table_as_json = to_json(escape_dataclass(table.metadata_config()))
    table_from_json_config = from_json(table_as_json, TableMetadataConfig)
    recovered_table = Table.from_source_config(table_from_json_config,
                                               'FINANCIAL', table.data_source,
                                               table.connector)
    assert len(recovered_table.cols) == len(table.cols)


def test_json_safe_float():
    assert json_safe_float(float('inf')) is None
    assert json_safe_float(float('nan')) is None
    assert json_safe_float(4) == 4.0
    assert json_safe_float(np.array(4)) == 4.0
    assert json_safe_float_array([4, float('inf')]) == [4.0, None]


def test_json_load_no_such_key():
    path = "s3://kumo-public-datasets/something_not_exist"
    assert load_json(path, str, default="not_exists",
                     raise_on_missing=False) == "not_exists"

    with pytest.raises(FileNotFoundError):
        load_json(path, str, default="not_exists")


def test_json_load_missing_local_file_returns_default_without_read(
        monkeypatch, test_dir):
    missing_path = osp.join(test_dir, "missing.json")
    read_called = False

    def _raise_if_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        raise AssertionError("read_text should not be called for missing file")

    monkeypatch.setattr(fsutil, "read_text", _raise_if_read)
    assert load_json(missing_path, dict, default={"missing": True},
                     raise_on_missing=False) == {
                         "missing": True
                     }
    assert read_called is False


def test_json_load_zipfile():
    path = "s3://kumo-public-datasets/test_zip_file/" \
           "materialization_task_info.json.gz"
    # This public artifact predates the strict connector-ID schema. The test
    # covers compressed S3 JSON I/O, not legacy object-model compatibility.
    data = load_json(path, dict, compressed=True)
    assert data.get('table_stats')


def test_load_json_eof(test_dir, caplog):
    path = osp.join(test_dir, "test.json")
    obj: dict[int, str] = {20250606: "friday"}
    rogue_chars_cases = [
        "}", "{}", ",", "\\", "\"", "\u0394", "\xA0", "\a", "\a\n",
        "AAAAAAAAAA", "20250606"
    ]
    for rogue_chars in rogue_chars_cases:
        content = f"{to_json(obj)}{rogue_chars}"
        fsutil.write_text(path, content)

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            obj_rt = load_json(path, cls=dict[int, str])
            assert "json decode error" in caplog.records[0].message

        assert obj_rt == obj
