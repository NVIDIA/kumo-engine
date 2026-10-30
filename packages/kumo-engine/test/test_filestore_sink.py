# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.artifact_export import ArtifactDestinationConflict
from kumo_core.chunked_fs import ChunkedFileHandler
from kumo_core.object_store_sink import FilesystemSink
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.plugin import FileStoreSink
from pyarrow.fs import PyFileSystem

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.connector import file_store, is_file_destination
from kumo.engine.trainer.output_config import (
    PredictionOutputConfig,
    TrainingTableOutputConfig,
)


class _Bytes:
    def __init__(self, data):
        self.data = data

    def read_at(self, offset, length):
        return self.data[offset:offset + length]

    def close(self):
        pass


class MemoryStore(ChunkedFileHandler):
    part_bytes = 4096

    def __init__(self):
        self.objects, self.spooled = {}, set()

    def get_type_name(self):
        return 'memory'

    def _size(self, path):
        data = self.objects.get(path)
        return None if data is None else len(data)

    def _list(self, directory):
        return {
            path: len(data)
            for path, data in self.objects.items()
            if path.startswith(directory + '/')
        }

    def _put(self, local, path):
        self.spooled.add(Path(local).parent.parent)
        self.objects[path] = Path(local).read_bytes()

    def _part(self, path, size):
        return _Bytes(self.objects[path])

    def _remove(self, path, *, tree):
        for name in list(self.objects):
            if name.startswith(path + '/') if tree else name == path:
                del self.objects[name]


class StageSink:
    r"""Not an object store: holds '@stage/...' paths through
    ``filesystem``; anything else is a table."""
    def __init__(self):
        self.store = MemoryStore()

    def holds_files(self, connector, path):
        return path.startswith('@')

    def filesystem(self, connector, path):
        assert self.holds_files(connector, path)
        return PyFileSystem(self.store), path.rstrip('/')


@pytest.fixture
def connector(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    yield SimpleNamespace(source_type='SNOWFLAKE', sink_backend=StageSink())
    kumo.configure(scratch_dir=previous)


def test_sinks_with_a_filesystem_hold_files(tmp_path, connector):
    local = kumo.FileConnector(root_dir=str(tmp_path))
    assert isinstance(connector.sink_backend, FileStoreSink)
    assert isinstance(local.sink_backend, FileStoreSink)
    assert is_file_destination(local, 'any/path')
    assert is_file_destination(connector, '@stage/p')
    assert not is_file_destination(connector, 'DB.SCHEMA.T')
    assert file_store(connector, 'DB.SCHEMA.T') is None
    assert not is_file_destination(SimpleNamespace(), 'x')


def test_output_configs_take_a_sinks_file_paths(connector):
    assert TrainingTableOutputConfig(connector, '@stage/tt').table_name == (
        '@stage/tt/')
    assert TrainingTableOutputConfig(connector, 'TT').table_name == 'TT'
    PredictionOutputConfig(connector,
                           output_paths={'predictions': '@stage/p/'})
    with pytest.raises(ValueError, match='output_paths requires'):
        PredictionOutputConfig(connector, output_paths={'predictions': 'T'})
    PredictionOutputConfig(connector, PredictionArtifactType.PREDICTIONS,
                           'PREDS')


def test_bundles_round_trip_through_a_filesystem_only_sink(
        tmp_path, connector):
    bundle = tmp_path / 'bundle'
    (bundle / 'sub').mkdir(parents=True)
    (bundle / '_provenance.json').write_text('{}')
    (bundle / 'sub' / 'weights.bin').write_bytes(os.urandom(10000))
    files = FilesystemSink(connector.sink_backend)
    published = files.publish_model(connector, str(bundle),
                                    path='@stage/models/m')
    assert published.uri == '@stage/models/m'
    store = connector.sink_backend.store
    assert list(store.objects)[-1] == '@stage/models/m/_provenance.json'
    assert any(
        name.startswith('@stage/models/m/sub/weights.bin.chunks/')
        and name.endswith('/_SUCCESS') for name in store.objects)
    files.fetch_model(connector, str(tmp_path / 'fetched'),
                      path='@stage/models/m')
    for name in ('_provenance.json', 'sub/weights.bin'):
        assert (tmp_path / 'fetched' / name).read_bytes() == (
            bundle / name).read_bytes()


def test_a_trainer_publishes_and_loads_through_a_filesystem_only_sink(
        fitted):
    trainer, *_ = fitted
    connector = SimpleNamespace(source_type='SNOWFLAKE',
                                sink_backend=StageSink())
    published = trainer.result.export(
        kumo.ModelOutputConfig(connector, path='@stage/models/m'))
    assert published.uri == '@stage/models/m'
    loaded = kumo.Trainer.load_from_connector(connector,
                                              path='@stage/models/m')
    assert loaded.result.id == trainer.result.id
    with pytest.raises(NotImplementedError, match='cannot publish'):
        trainer.result.export(kumo.ModelOutputConfig(connector, path='T'))


def test_predictions_export_through_a_filesystem_only_sink(fitted, connector):
    trainer, table, snapshot = fitted
    output = PredictionOutputConfig(
        connector, output_paths={'predictions': '@stage/preds/'})
    result = trainer.predict(table, snapshot=snapshot, output_config=output)
    store = connector.sink_backend.store
    assert any(name.startswith('@stage/preds/') for name in store.objects)
    result.export(
        PredictionOutputConfig(connector,
                               output_paths={'predictions': '@stage/again/'}))
    assert any(name.startswith('@stage/again/') for name in store.objects)
    with pytest.raises(NotImplementedError, match='requires'):
        PredictionOutputConfig(connector, PredictionArtifactType.PREDICTIONS,
                               'DB.S.T').writer('DB.S.T')


def test_parquet_writes_stay_plain_up_to_one_part(tmp_path, connector):
    files = FilesystemSink(connector.sink_backend)
    frame = pd.DataFrame({'x': range(10)})
    files.write_table(connector, '@stage/small.parquet', frame)
    files.write_table(connector, '@stage/big.parquet',
                      pd.DataFrame({'x': [os.urandom(50).hex() for _ in range(100)]}))
    store = connector.sink_backend.store
    assert '@stage/small.parquet' in store.objects
    assert any(
        name.startswith('@stage/big.parquet.chunks/')
        and name.endswith('/_SUCCESS') for name in store.objects)
    assert pd.read_parquet(
        '@stage/big.parquet',
        filesystem=PyFileSystem(store)).shape == (100, 1)

    part = tmp_path / 'part-0.parquet'
    frame.to_parquet(part)
    assert files.write_files(connector, '@stage/out/', [str(part)])
    assert pd.read_parquet('@stage/out/part-0.parquet',
                           filesystem=PyFileSystem(store)).equals(frame)


def test_publishing_an_identical_bundle_again_reuses_the_copy(
        tmp_path, connector, monkeypatch, caplog):
    import kumo_core.object_store_sink as sink

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / '_provenance.json').write_text('{"id": 1}')
    (bundle / 'weights.bin').write_bytes(os.urandom(10000))
    files = FilesystemSink(connector.sink_backend)
    files.publish_model(connector, str(bundle), path='@stage/models/m')
    upload = sink.upload_bundle
    monkeypatch.setattr(sink, 'upload_bundle',
                        lambda *a: pytest.fail('uploaded again'))
    with caplog.at_level('INFO', logger=sink.__name__):
        files.publish_model(connector, str(bundle), path='@stage/models/m')
    assert 'reused it' in caplog.text

    (bundle / '_provenance.json').write_text('{"id": 2}')
    uploads = []
    monkeypatch.setattr(sink, 'upload_bundle',
                        lambda *a: uploads.append(a) or upload(*a))
    files.publish_model(connector, str(bundle), path='@stage/models/m')
    assert len(uploads) == 1


def test_an_occupied_directory_is_refused_by_its_full_location(
        tmp_path, connector):
    files = FilesystemSink(connector.sink_backend)
    part = tmp_path / 'part-0.parquet'
    pd.DataFrame({'x': range(3)}).to_parquet(part)
    files.write_files(connector, '@stage/out/', [str(part)])

    files.preflight_write_files(connector, '@stage/new/')
    files.preflight_write_files(connector, '@stage/out/',
                                write_mode=WriteMode.OVERWRITE)
    with pytest.raises(ArtifactDestinationConflict, match="'@stage/out/'"):
        files.preflight_write_files(connector, '@stage/out/')
