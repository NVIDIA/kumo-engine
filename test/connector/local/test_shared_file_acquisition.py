# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types.schema import IngestedTableFingerprint
from kumoapi.typing import Dtype

from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector


def _export(connector, columns, destination):
    fingerprint = IngestedTableFingerprint(connector_name=connector.name,
                                           source_name='data', cols=columns,
                                           pkey=None, time_col=None,
                                           end_time_col=None)
    with patch('kumo.data.utils.get_materialize_path',
               return_value=str(destination)):
        return connector.export_table('data', fingerprint, 'version')


def test_hosted_csv_preserves_configured_escape_nulls_and_exact_integers(
        tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.csv').write_text(
        'id;label\n9007199254740993;hello\\;world\n10;MISSING\n11;NA\n')
    connector = CSVConnector(name='csv', root_dir=str(source), sep=';',
                             escapechar='\\', na_values=['MISSING'])
    output = tmp_path / 'output'
    metrics = _export(connector, {
        'id': Dtype.int64,
        'label': Dtype.string
    }, output)
    result = pq.read_table(output)
    assert result.to_pydict() == {
        'id': [9007199254740993, 10, 11],
        'label': ['hello;world', None, 'NA'],
    }
    assert result.schema.field('id').type == pa.int64()
    assert metrics.num_rows == 3
    assert metrics.source_size != (source / 'data.csv').stat().st_size
    assert metrics.source_size == sum(p.stat().st_size
                                      for p in output.glob('*.parquet'))


def test_hosted_parquet_publishes_after_source_credentials_exit(
        tmp_path, monkeypatch):
    from kumo.connector import source_bridge

    source = tmp_path / 'source'
    source.mkdir()
    original = source / 'data.parquet'
    pq.write_table(pa.table({
        'id': [1, 2],
        'unselected': ['a', 'b']
    }), original, compression='gzip')
    connector = ParquetConnector(name='parquet', root_dir=str(source))
    active = False

    @contextmanager
    def scope(_):
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    publish = source_bridge.publish_source

    def checked_publish(staged, destination):
        assert not active
        assert Path(staged.paths[0]).read_bytes() == original.read_bytes()
        return publish(staged, destination)

    monkeypatch.setattr(connector, 'filesystem_credential_scope', scope)
    monkeypatch.setattr(source_bridge, 'publish_source', checked_publish)
    output = tmp_path / 'output'
    _export(connector, {'id': Dtype.int64}, output)
    assert pq.read_table(output).column_names == ['id', 'unselected']
    assert next(output.glob('*.parquet')).read_bytes() == original.read_bytes()


@pytest.mark.parametrize('nested', ['country=US', 'nested'])
def test_hosted_nested_parquet_keeps_physical_partition_columns(
        tmp_path, nested):
    source = tmp_path / 'source'
    parts = source / 'data' / nested
    parts.mkdir(parents=True)
    pq.write_table(
        pa.table({
            'id': [1, 2],
            'unused_struct': [{
                'value': 1
            }, {
                'value': 2
            }]
        }), parts / 'part.parquet')
    connector = ParquetConnector(name='parquet', root_dir=str(source))
    output = tmp_path / 'output'
    metrics = _export(connector, {'id': Dtype.int64}, output)
    result = pq.read_table(output)
    assert result.column('id').to_pylist() == [1, 2]
    assert result.column('unused_struct').to_pylist() == [{
        'value': 1
    }, {
        'value': 2
    }]
    if nested.startswith('country='):
        assert result.column('country').to_pylist() == ['US', 'US']
    assert metrics.num_rows == 2
    assert metrics.source_size == (parts / 'part.parquet').stat().st_size


@pytest.mark.parametrize('suffix', ['txt', 'csv.gz'])
@pytest.mark.parametrize('directory', [False, True])
def test_hosted_csv_retains_configured_source_suffix(tmp_path, suffix,
                                                     directory):
    source = tmp_path / 'source'
    source.mkdir()
    body = b'id,label\n1,alpha\n2,beta\n'
    part = source / 'data' / f'part.{suffix}' if directory else source / f'data.{suffix}'
    part.parent.mkdir(exist_ok=True)
    part.write_bytes(gzip.compress(body) if suffix.endswith('.gz') else body)
    connector = CSVConnector(name='csv', root_dir=str(source), suffix=suffix)
    output = tmp_path / 'output'
    _export(connector, {'id': Dtype.int64, 'label': Dtype.string}, output)
    assert pq.read_table(output).to_pydict() == {
        'id': [1, 2],
        'label': ['alpha', 'beta']
    }


def test_hosted_unknown_input_size_uses_existing_scoped_estimator(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pyarrow.fs import LocalFileSystem

    source = tmp_path / 'source'
    source.mkdir()
    pq.write_table(pa.table({'id': [1]}), source / 'data.parquet')
    filesystem = LocalFileSystem()
    active = False
    calls = []

    class UnknownSize:
        def get_file_info(self, path):
            assert active
            info = filesystem.get_file_info(path)
            return SimpleNamespace(path=info.path, type=info.type, size=None,
                                   mtime=info.mtime, mtime_ns=info.mtime_ns)

        def open_input_stream(self, path):
            assert active
            return filesystem.open_input_stream(path)

    @contextmanager
    def scope(_):
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    def estimate(name):
        assert active
        calls.append(name)
        return 17

    connector = ParquetConnector(name='parquet', root_dir=str(source))
    monkeypatch.setattr(connector, 'filesystem_credential_scope', scope)
    monkeypatch.setattr(connector, '_get_table_source_size', estimate)
    monkeypatch.setattr('io_utils.fsutil.pyarrow_filesystem',
                        lambda _: UnknownSize())
    read_schema = pq.read_schema

    def scoped_read_schema(path, *, filesystem):
        assert active
        assert isinstance(filesystem, UnknownSize)
        return read_schema(path)

    monkeypatch.setattr(pq, 'read_schema', scoped_read_schema)
    metrics = _export(connector, {'id': Dtype.int64}, tmp_path / 'output')
    assert metrics.source_size == 17
    assert calls == ['data']
    assert not active


def test_hosted_csv_does_not_infer_excluded_values(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.csv').write_text('id,excluded\n1,' + '9' * 80 + '\n')
    output = tmp_path / 'output'
    connector = CSVConnector(name='csv', root_dir=str(source))
    _export(connector, {'id': Dtype.int64}, output)
    assert pq.read_table(output).to_pydict() == {'id': [1]}
