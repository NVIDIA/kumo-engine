# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Snowflake stage seen through pyarrow, against an in-memory stage."""

import gc
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fakes import FakeStage
from kumo_snowflake import stage_fs
from kumo_snowflake.sink import SnowflakeSinkBackend
from kumoapi.plugin import FileStoreSink
from pyarrow.fs import FileSelector, FileType

ROOT = '@DB.SCHEMA.KUMO_ARTIFACTS/runs'
PART = 32


@pytest.fixture
def stage(monkeypatch, tmp_path):
    monkeypatch.setattr('kumo_snowflake.staging.SPCS_TOKEN_PATH',
                        str(tmp_path / 'absent'))
    monkeypatch.setattr(stage_fs, '_ranged_stages', {})
    monkeypatch.setattr(stage_fs.StageFileSystemHandler, 'part_bytes', PART)
    fake = FakeStage()
    monkeypatch.setattr(stage_fs, '_download_session',
                        lambda: SimpleNamespace(get=fake.http_get))
    return fake


@pytest.fixture
def sink(stage, monkeypatch):
    session = stage.session()
    sink = SnowflakeSinkBackend()

    @contextmanager
    def connected(connector):
        yield session

    monkeypatch.setattr(sink, '_session', connected)
    return sink


@pytest.fixture
def filesystem(sink):
    return sink.filesystem(object(), ROOT + '/')


def _write(fs, path, data):
    with fs.open_output_stream(path) as writer:
        writer.write(data)


def _read(fs, path):
    with fs.open_input_stream(path) as reader:
        return reader.readall()


def _chunks(names, path):
    assert all(name.startswith(f'{path}.chunks/') for name in names)
    return sorted(name.rsplit('/', 1)[1] for name in names)


def test_the_sink_resolves_a_stage_path(filesystem, stage):
    fs, root = filesystem
    assert root == ROOT
    _write(fs, f'{root}/model/weights.bin', b'weights')
    assert stage.objects == {'runs/model/weights.bin': b'weights'}
    assert stage.puts == [{
        'parallel': 1,
        'auto_compress': False,
        'source_compression': 'NONE',
        'overwrite': True
    }]
    assert _read(fs, f'{root}/model/weights.bin') == b'weights'


@pytest.mark.parametrize('path,held', [(ROOT, True), (ROOT + '/m/1', True),
                                       ('USER_SCORES', False),
                                       ('@~/runs', False),
                                       ('@%USERS/runs', False)])
def test_only_named_stage_paths_are_file_locations(path, held):
    sink = SnowflakeSinkBackend()
    assert isinstance(sink, FileStoreSink)
    assert sink.holds_files(object(), path) is held


def test_a_bundle_round_trips_through_a_stage_path(sink, stage, tmp_path):
    bundle = tmp_path / 'bundle'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'weights.pt').write_bytes(os.urandom(2 * PART + 3))
    (bundle / 'bundle.json').write_text('{}')
    published = sink.publish_model(object(), str(bundle),
                                   path=f'{ROOT}/models/b1')
    assert published.uri == f'{ROOT}/models/b1'
    fetched = sink.fetch_model(object(), str(tmp_path / 'fetched'),
                               path=f'{ROOT}/models/b1')
    assert fetched.version == f'{ROOT}/models/b1'
    for name in ('model/weights.pt', 'bundle.json'):
        assert (tmp_path / 'fetched' /
                name).read_bytes() == (bundle / name).read_bytes()
    with pytest.raises(ValueError, match='Nothing is published'):
        sink.fetch_model(object(), str(tmp_path / 'x'),
                         path=f'{ROOT}/models/none')


def test_parquet_files_are_written_to_a_stage_path(
        sink, stage, tmp_path):
    import pandas as pd
    parts = []
    for index in range(2):
        parts.append(tmp_path / f'part-{index}.parquet')
        pd.DataFrame({'ID': [index]}).to_parquet(parts[-1])
    written = sink.write_files(object(), f'{ROOT}/predictions/r1/',
                               [str(path) for path in parts], export_id='r1')
    assert written == f'{ROOT}/predictions/r1/'
    fs, root = sink.filesystem(object(), f'{ROOT}/predictions/r1')
    names = [i.base_name for i in fs.get_file_info(FileSelector(root))]
    assert names == [
        '_KUMO_EXPORT', '_SUCCESS', 'part-0.parquet', 'part-1.parquet'
    ]
    assert sink.write_table(object(), f'{ROOT}/tables/t1.parquet',
                            pd.DataFrame({'ID': [1]
                                          })) == (f'{ROOT}/tables/t1.parquet')


def test_a_user_stage_is_refused(filesystem):
    with pytest.raises(ValueError, match='fully-qualified'):
        stage_fs.stage_filesystem(None, '@~/runs')


def test_listing_reports_sizes_and_logical_files(filesystem, stage):
    fs, root = filesystem
    _write(fs, f'{root}/a/small.json', b'{}')
    _write(fs, f'{root}/a/big.bin', os.urandom(2 * PART + 1))
    infos = fs.get_file_info(FileSelector(f'{root}/a', recursive=True))
    assert [(i.base_name, i.size) for i in infos] == [('big.bin', 2 * PART + 1),
                                                      ('small.json', 2)]
    assert fs.get_file_info(f'{root}/a').type is FileType.Directory
    assert fs.get_file_info(f'{root}/a/small').type is FileType.NotFound


def test_removing_a_file_spares_names_it_prefixes(filesystem, stage):
    fs, root = filesystem
    for name in ('part-1', 'part-10', 'part-1.x', 'd/part-1'):
        _write(fs, f'{root}/{name}', name.encode())
    fs.delete_file(f'{root}/part-1')
    assert sorted(stage.objects) == [
        'runs/d/part-1', 'runs/part-1.x', 'runs/part-10'
    ]
    fs.delete_dir_contents(root)
    assert stage.objects == {}


def test_a_chunked_file_reads_exact_ranges_over_presigned_urls(
        filesystem, stage):
    fs, root = filesystem
    data = os.urandom(3 * PART)
    _write(fs, f'{root}/index.bin', data)
    assert _chunks(stage.objects, 'runs/index.bin') == [
        '00000', '00001', '00002', '_SUCCESS'
    ]
    with fs.open_input_file(f'{root}/index.bin') as reader:
        assert reader.read_at(PART, PART // 2) == data[PART // 2:PART * 3 // 2]
    assert _read(fs, f'{root}/index.bin') == data
    assert len(stage.ranges) > 1 and not stage.gets
    assert any(
        s.startswith('REMOVE @DB.SCHEMA.KUMO_ARTIFACTS/.kumo-preflight/')
        for s in stage.statements)
    assert not any('.kumo-preflight' in name for name in stage.objects)


def test_a_stage_without_ranged_plaintext_reads_each_file_once_by_get(
        sink, stage, caplog, tmp_path, monkeypatch):
    spool = tmp_path / 'spool'
    monkeypatch.setattr(stage_fs.StageFileSystemHandler, 'spool_dir',
                        str(spool))
    fs, root = sink.filesystem(object(), ROOT)
    stage.ignores_range = True
    data = os.urandom(2 * PART)
    _write(fs, f'{root}/a.bin', data)
    _write(fs, f'{root}/a.bin2', b'sibling')
    for _ in range(2):
        with fs.open_input_file(f'{root}/a.bin') as reader:
            assert reader.read_at(PART,
                                  PART // 2) == data[PART // 2:PART * 3 // 2]
    assert _read(fs, f'{root}/a.bin') == data
    assert _chunks(stage.gets, 'runs/a.bin') == ['00000', '00001']
    assert [
        r.levelname for r in caplog.records
        if 'SNOWFLAKE_SSE' in r.getMessage()
    ] == ['WARNING']
    del fs, reader
    gc.collect()
    assert not list(spool.iterdir())


def test_a_client_side_encrypted_stage_is_refused_on_read(filesystem, stage):
    fs, root = filesystem
    stage.ciphertext = True
    _write(fs, f'{root}/a.bin', b'0123456789')
    with pytest.raises(OSError, match="ENCRYPTION = \\(TYPE = 'SNOWFLAKE_SSE'\\)"):
        _read(fs, f'{root}/a.bin')
    assert stage.gets == []


def test_a_role_that_cannot_write_reads_by_get_without_blaming_encryption(
        filesystem, stage, caplog, monkeypatch):
    fs, root = filesystem
    _write(fs, f'{root}/a.bin', b'0123456789')

    def refused(*args, **kwargs):
        raise PermissionError('insufficient privileges')

    monkeypatch.setattr(stage, 'put', refused)
    with fs.open_input_file(f'{root}/a.bin') as reader:
        assert reader.read_at(4, 3) == b'3456'
    messages = [r.getMessage() for r in caplog.records]
    assert any('lack WRITE' in m for m in messages)
    assert not any('SNOWFLAKE_SSE' in m for m in messages)


def test_a_remove_that_matches_nothing_raises(filesystem, stage):
    fs, root = filesystem
    _write(fs, f'{root}/a.bin', b'old')
    stage.remove_pattern_ignored = True
    with pytest.raises(OSError, match='left'):
        _write(fs, f'{root}/a.bin', os.urandom(2 * PART))


def test_a_name_outside_the_stage_character_set_fails_at_write(filesystem):
    fs, root = filesystem
    with pytest.raises(ValueError):
        _write(fs, f'{root}/my table.json', b'{}')


def test_a_failed_read_names_no_signed_url(filesystem, stage):
    fs, root = filesystem
    _write(fs, f'{root}/a.bin', b'data')
    reader = fs.open_input_file(f'{root}/a.bin')
    del stage.objects['runs/a.bin']
    with pytest.raises(OSError) as error:
        reader.read_at(2, 0)
    assert 'signature' not in str(error.value)
    assert 'status=404' in str(error.value)


def test_a_registry_path_is_not_sent_to_the_stage(sink):
    with pytest.raises(ValueError, match='ModelRegistryLocation'):
        sink.publish_model(object(), '/tmp/none', path='models/m1')


def test_an_occupied_stage_directory_is_refused_before_scoring(
        sink, stage, tmp_path):
    import pandas as pd
    from kumo_core.artifact_export import ArtifactDestinationConflict
    from kumoapi.jobs import WriteMode

    target = f'{ROOT}/predictions/r1/'
    sink.preflight_write_files(object(), target)
    part = tmp_path / 'part-0.parquet'
    pd.DataFrame({'ID': [1]}).to_parquet(part)
    sink.write_files(object(), target, [str(part)], export_id='r1')

    for mode in (None, WriteMode.OVERWRITE):
        with pytest.raises(ArtifactDestinationConflict,
                           match='/predictions/r1/'):
            sink.preflight_write_files(object(), target, write_mode=mode)
    sink.preflight_write_files(object(), f'{ROOT}/predictions/r2/')
    sink.preflight_write_files(object(), 'SCORES')
