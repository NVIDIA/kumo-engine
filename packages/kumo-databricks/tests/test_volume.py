# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from databricks.sdk.errors import PermissionDenied
from fakes import FakeVolume, FakeWorkspaceClient
from kumo_databricks.sink import DatabricksSinkBackend
from kumo_databricks.volume import VolumeHandler
from pyarrow.fs import FileType, PyFileSystem

ROOT = '/Volumes/main/analytics/artifacts'
BODY = bytes(range(256)) * 10


def filesystem(objects=None, part_bytes=None):
    volume = FakeVolume(objects)
    handler = VolumeHandler(FakeWorkspaceClient(volume))
    if part_bytes is not None:
        handler.part_bytes = part_bytes
    return PyFileSystem(handler), volume


def write(fs, path, body, piece=700):
    with fs.open_output_stream(path) as writer:
        for start in range(0, len(body), piece):
            writer.write(body[start:start + piece])


def test_reads_cross_part_boundaries():
    fs, _ = filesystem(part_bytes=1000)
    write(fs, f'{ROOT}/run/big.bin', BODY)
    with fs.open_input_file(f'{ROOT}/run/big.bin') as reader:
        assert reader.read_at(300, 900) == BODY[900:1200]
        assert reader.read_at(10, 1200) == BODY[1200:1210]


def test_sized_writes_stay_plain_up_to_a_part():
    from kumo_core.object_store_sink import open_sized

    fs, volume = filesystem(part_bytes=1000)
    fs.handler.plain_bytes = 100
    for name, body in (('plain', BODY[:1000]), ('parts', BODY[:1500])):
        with open_sized(fs, f'{ROOT}/{name}', len(body)) as writer:
            writer.write(body)
        with fs.open_input_stream(f'{ROOT}/{name}') as reader:
            assert reader.readall() == body
    assert volume.objects[f'{ROOT}/plain'] == BODY[:1000]
    assert f'{ROOT}/parts' not in volume.objects
    assert any(path.startswith(f'{ROOT}/parts.chunks/')
               and path.endswith('/_SUCCESS') for path in volume.objects)


def test_a_failed_upload_raises_without_the_signed_url():
    from kumo_core.object_store_sink import open_sized

    fs, volume = filesystem()
    signed = 'https://bucket.example/part?X-Amz-Signature=secret'

    def drop(path, contents, *, overwrite):
        contents.read(100)
        raise ConnectionError(f'Max retries exceeded with url: {signed}')

    volume.upload = drop
    with pytest.raises(OSError) as raised:
        with open_sized(fs, f'{ROOT}/x', len(BODY)) as writer:
            for start in range(0, len(BODY), 64):
                writer.write(BODY[start:start + 64])
    assert 'Signature' not in str(raised.value)
    assert raised.value.__cause__ is None


def test_a_download_that_ignores_the_range_is_refused():
    fs, volume = filesystem({f'{ROOT}/a': BODY})
    volume.ignore_range = True
    with fs.open_input_file(f'{ROOT}/a') as reader:
        with pytest.raises(OSError, match='asked for bytes from 100'):
            reader.read_at(10, 100)


def test_directories_and_missing_paths():
    fs, volume = filesystem({
        f'{ROOT}/s/kumo_snapshot.json': b'{}',
        f'{ROOT}/s/index/sub/b': b'de',
    })
    kinds = [info.type for info in fs.get_file_info(
        [f'{ROOT}/s/kumo_snapshot.json', f'{ROOT}/s/index/', f'{ROOT}/s/no'])]
    assert kinds == [FileType.File, FileType.Directory, FileType.NotFound]
    with pytest.raises(FileNotFoundError):
        fs.open_input_stream(f'{ROOT}/s/no')
    fs.delete_dir_contents(f'{ROOT}/s/')
    assert volume.objects == {}


def test_refused_upload_raises_permission_error():
    fs, volume = filesystem()

    def refuse(path, contents, *, overwrite):
        raise PermissionDenied('no WRITE VOLUME')

    volume.upload = refuse
    with pytest.raises(PermissionError):
        write(fs, f'{ROOT}/x', b'data')


def test_sink_filesystem_only_takes_volume_paths():
    sink = DatabricksSinkBackend()
    volume = FakeVolume()
    sink._client = lambda connector: FakeWorkspaceClient(volume)
    connector = SimpleNamespace()
    fs, path = sink.filesystem(connector, f'{ROOT}/exports/')
    assert path == f'{ROOT}/exports'
    write(fs, f'{path}/x', b'1')
    assert volume.objects == {f'{path}/x': b'1'}
    assert sink.holds_files(connector, f'{ROOT}/exports/')
    for bad in ('s3://bucket/x', '/Volumes/main/analytics', 'users_scores',
                f'{ROOT}/../other/x', 'dbfs:/Volumes/main/analytics/v'):
        assert not sink.holds_files(connector, bad)
        with pytest.raises(ValueError, match='volume path'):
            sink.filesystem(connector, bad)


def test_bundle_publish_and_fetch_through_a_volume_path(tmp_path):
    sink = DatabricksSinkBackend()
    volume = FakeVolume()
    sink._client = lambda connector: FakeWorkspaceClient(volume)
    bundle = tmp_path / 'bundle'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'training-spec.json').write_text('{}')
    (bundle / 'model' / 'weights.pt').write_bytes(BODY)
    published = sink.publish_model(SimpleNamespace(), str(bundle),
                                   path=f'{ROOT}/models/b1')
    assert published.uri == f'{ROOT}/models/b1'
    assert volume.objects[f'{ROOT}/models/b1/model/weights.pt'] == BODY
    fetched = tmp_path / 'fetched'
    sink.fetch_model(SimpleNamespace(), str(fetched), path=f'{ROOT}/models/b1')
    assert (fetched / 'model' / 'weights.pt').read_bytes() == BODY
    with pytest.raises(ValueError, match='Nothing is published'):
        sink.fetch_model(SimpleNamespace(), str(fetched),
                         path=f'{ROOT}/models/none')


def test_predictions_and_frames_written_to_a_volume_path(tmp_path):
    sink = DatabricksSinkBackend()
    volume = FakeVolume()
    sink._client = lambda connector: FakeWorkspaceClient(volume)
    part = tmp_path / 'part-0.parquet'
    pq.write_table(pa.table({'users': [1, 2], 'score': [0.1, 0.9]}), part)
    written = sink.write_files(SimpleNamespace(), f'{ROOT}/predictions/',
                               [str(part)], export_id='run-1')
    assert written == f'{ROOT}/predictions/'
    names = {path.rsplit('/', 1)[-1] for path in volume.objects}
    assert {'part-0.parquet', '_SUCCESS'} <= names
    frame = pa.table({'users': [3]}).to_pandas()
    assert sink.write_table(SimpleNamespace(), f'{ROOT}/one',
                            frame) == f'{ROOT}/one.parquet'


def test_an_occupied_volume_directory_is_refused_before_scoring(tmp_path):
    from kumo_core.artifact_export import ArtifactDestinationConflict
    from kumoapi.jobs import WriteMode

    sink = DatabricksSinkBackend()
    volume = FakeVolume()
    sink._client = lambda connector: FakeWorkspaceClient(volume)
    target = f'{ROOT}/predictions/'
    sink.preflight_write_files(SimpleNamespace(), target)
    part = tmp_path / 'part-0.parquet'
    pq.write_table(pa.table({'users': [1]}), part)
    sink.write_files(SimpleNamespace(), target, [str(part)], export_id='r1')

    with pytest.raises(ArtifactDestinationConflict, match='/predictions/'):
        sink.preflight_write_files(SimpleNamespace(), target)
    sink.preflight_write_files(SimpleNamespace(), target,
                               write_mode=WriteMode.OVERWRITE)
    sink.preflight_write_files(SimpleNamespace(), f'{ROOT}/other/')
    sink.preflight_write_files(SimpleNamespace(), 'main.ml.scores')
