# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import io
import os
import threading

import pytest
from kumo_core import chunked_fs
from kumo_core.chunked_fs import ChunkedFileHandler
from kumo_core.object_store_sink import open_sized
from pyarrow.fs import FileSelector, FileType, PyFileSystem

PART = 16


class _Bytes:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read_at(self, offset, length):
        return self.data[offset:offset + length]

    def close(self):
        pass


class MemoryHandler(ChunkedFileHandler):
    part_bytes = PART

    def __init__(self):
        self.objects = {}
        self.spooled = []

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
        assert os.path.basename(local) == os.path.basename(path)
        self.spooled.append(os.path.dirname(os.path.dirname(local)))
        with open(local, 'rb') as reader:
            self.objects[path] = reader.read()

    def _part(self, path, size):
        return _Bytes(self.objects[path])

    def _remove(self, path, *, tree):
        for name in list(self.objects):
            if name.startswith(path + '/') if tree else name == path:
                del self.objects[name]


class _Upload(io.BytesIO):
    def __init__(self, handler, path):
        super().__init__()
        self.handler, self.path = handler, path

    def close(self):
        self.handler.objects[self.path] = self.getvalue()
        self.handler.streamed.append(self.path)
        super().close()


class StreamingHandler(MemoryHandler):
    plain_bytes = 6

    def __init__(self):
        super().__init__()
        self.streamed = []

    def _stream(self, path):
        return _Upload(self, path)


@pytest.fixture
def store():
    handler = MemoryHandler()
    return handler, PyFileSystem(handler)


@pytest.fixture
def streaming():
    handler = StreamingHandler()
    return handler, PyFileSystem(handler)


def _write(fs, path, data):
    with fs.open_output_stream(path) as writer:
        for start in range(0, len(data), 5):
            writer.write(data[start:start + 5])


def _read(fs, path):
    with fs.open_input_stream(path) as reader:
        return reader.readall()


def _is_plain(handler):
    plain = 'root/a.bin' in handler.objects
    assert any('.chunks/' in name for name in handler.objects) != plain
    return plain


def _generations(handler):
    parts = {}
    for name in sorted(handler.objects):
        directory, _, part = name.rpartition('/')
        parts.setdefault(directory, []).append(part)
    return parts


@pytest.mark.parametrize('size', [0, 3, PART])
def test_files_up_to_one_part_are_one_plain_object(store, size):
    handler, fs = store
    data = bytes(range(size))
    _write(fs, 'root/a.bin', data)
    assert handler.objects == {'root/a.bin': data}
    assert fs.get_file_info('root/a.bin').size == size
    assert _read(fs, 'root/a.bin') == data


@pytest.mark.parametrize('size', [2 * PART + 5, 3 * PART])
def test_larger_files_are_parts_committed_last_and_read_back_exactly(
        store, size):
    handler, fs = store
    data = os.urandom(size)
    _write(fs, 'root/a.bin', data)
    (generation, parts), = _generations(handler).items()
    assert generation.startswith('root/a.bin.chunks/')
    assert parts == ['00000', '00001', '00002', '_SUCCESS']
    info = fs.get_file_info('root/a.bin')
    assert (info.type, info.size) == (FileType.File, len(data))
    assert _read(fs, 'root/a.bin') == data
    with fs.open_input_file('root/a.bin') as reader:
        assert reader.size() == len(data)
        assert reader.read_at(10, PART - 4) == data[PART - 4:PART + 6]
        assert reader.read_at(0, 2 * PART) == b''
        assert reader.read_at(PART, PART) == data[PART:2 * PART]
        assert reader.read_at(100, PART + 1) == data[PART + 1:]


def test_an_uncommitted_chunked_write_is_not_a_file(store):
    handler, fs = store
    handler.objects['root/a.bin.chunks/1/00000'] = b'x' * PART
    assert fs.get_file_info('root/a.bin').type is not FileType.File
    with pytest.raises(FileNotFoundError):
        fs.open_input_file('root/a.bin')


@pytest.mark.parametrize('first,second', [(3, 2 * PART), (2 * PART, 3)])
def test_a_write_removes_the_other_form(store, first, second):
    handler, fs = store
    _write(fs, 'root/a.bin', b'1' * first)
    _write(fs, 'root/a.bin', b'2' * second)
    assert _read(fs, 'root/a.bin') == b'2' * second
    assert _is_plain(handler) == (second <= PART)


@pytest.mark.parametrize('sized', [False, True])
@pytest.mark.parametrize('streams', [False, True])
@pytest.mark.parametrize('old', [3, 2 * PART])
def test_a_failed_chunked_write_leaves_the_old_file(
        monkeypatch, streams, sized, old):
    handler = StreamingHandler() if streams else MemoryHandler()
    fs = PyFileSystem(handler)
    _write(fs, 'root/a.bin', b'o' * old)
    upload = '_stream' if streams else '_put'
    original = getattr(handler, upload)

    def second_part_fails(*args):
        if args[-1].endswith('/00001'):
            raise OSError('upload failed')
        return original(*args)

    monkeypatch.setattr(handler, upload, second_part_fails)
    with pytest.raises(OSError, match='upload failed'):
        with (open_sized(fs, 'root/a.bin', 3 * PART)
              if sized else fs.open_output_stream('root/a.bin')) as writer:
            writer.write(b'n' * (3 * PART))
    assert _read(fs, 'root/a.bin') == b'o' * old

    monkeypatch.setattr(handler, upload, original)
    _write(fs, 'root/a.bin', b'n' * (3 * PART))
    assert _read(fs, 'root/a.bin') == b'n' * (3 * PART)
    assert list(_generations(handler).values()) == [
        ['00000', '00001', '00002', '_SUCCESS']
    ]


def test_a_failed_plain_stream_is_aborted_not_committed(
        monkeypatch, streaming):
    handler, fs = streaming

    def fail(self, data):
        raise OSError('upload failed')

    def abort(self):
        aborted.append(self.path)
        io.BytesIO.close(self)

    aborted = []
    monkeypatch.setattr(_Upload, 'write', fail)
    monkeypatch.setattr(_Upload, 'abort', abort, raising=False)
    with pytest.raises(OSError, match='upload failed'):
        with open_sized(fs, 'root/a.bin', 3) as writer:
            writer.write(b'abc')
    assert aborted == ['root/a.bin']
    assert handler.objects == {}


def test_the_newest_committed_generation_is_read(store):
    handler, fs = store
    _write(fs, 'root/a.bin', b'1' * (2 * PART))
    old = dict(handler.objects)
    _write(fs, 'root/a.bin', b'2' * (2 * PART))
    handler.objects.update(old)
    handler._forget('root/a.bin')
    assert _read(fs, 'root/a.bin') == b'2' * (2 * PART)


def test_plain_wins_when_a_crash_left_both_forms(store):
    handler, fs = store
    _write(fs, 'root/a.bin', b'c' * (2 * PART))
    handler.objects['root/a.bin'] = b'plain'
    assert _read(fs, 'root/a.bin') == b'plain'
    infos = fs.get_file_info(FileSelector('root', recursive=True))
    assert [(i.path, i.size) for i in infos] == [('root/a.bin', 5)]


def test_listing_shows_logical_files_and_directories(store):
    _, fs = store
    _write(fs, 'root/top.bin', b'1')
    _write(fs, 'root/dir/big.bin', b'2' * (PART + 1))
    _write(fs, 'root/dir/sub/small.bin', b'3')
    recursive = fs.get_file_info(FileSelector('root', recursive=True))
    assert [(i.path, i.type, i.size) for i in recursive] == [
        ('root/dir', FileType.Directory, None),
        ('root/dir/big.bin', FileType.File, PART + 1),
        ('root/dir/sub', FileType.Directory, None),
        ('root/dir/sub/small.bin', FileType.File, 1),
        ('root/top.bin', FileType.File, 1),
    ]
    shallow = fs.get_file_info(FileSelector('root'))
    assert [i.path for i in shallow] == ['root/dir', 'root/top.bin']
    assert fs.get_file_info('root/dir').type is FileType.Directory
    assert fs.get_file_info('root/none').type is FileType.NotFound
    assert fs.get_file_info(FileSelector('nothing', allow_not_found=True)) == []
    with pytest.raises(FileNotFoundError):
        fs.get_file_info(FileSelector('nothing'))


def test_deletes(store):
    handler, fs = store
    _write(fs, 'root/a.bin', b'a' * (PART + 1))
    _write(fs, 'root/b.bin', b'b')
    fs.delete_file('root/a.bin')
    assert sorted(handler.objects) == ['root/b.bin']
    with pytest.raises(FileNotFoundError):
        fs.delete_file('root/a.bin')
    fs.delete_dir_contents('root')
    assert handler.objects == {}
    fs.delete_dir_contents('root', missing_dir_ok=True)
    with pytest.raises(FileNotFoundError):
        fs.delete_dir_contents('root')
    with pytest.raises(NotImplementedError):
        fs.move('root/a', 'root/b')


def test_parts_spool_under_the_spool_dir_and_are_removed(
        store, tmp_path, monkeypatch):
    handler, fs = store
    root = [tmp_path / 'scratch']
    monkeypatch.setattr(chunked_fs, '_spool_dir', None)
    chunked_fs.set_spool_dir(lambda: str(root[0]))
    _write(fs, 'root/a.bin', b'a' * (2 * PART + 1))
    root[0] = tmp_path / 'moved'
    _write(fs, 'root/b.bin', b'b')
    handler.spool_dir = str(tmp_path / 'own')
    _write(fs, 'root/c.bin', b'c')
    assert handler.spooled == [str(tmp_path / 'scratch')] * 4 + [
        str(tmp_path / 'moved'), str(tmp_path / 'own')
    ]
    assert not os.listdir(tmp_path / 'scratch')
    assert not os.listdir(tmp_path / 'own')


@pytest.mark.parametrize('size', [0, 6, 7, PART, 2 * PART + 5, 3 * PART])
def test_a_streaming_handler_streams_parts_and_spools_only_small_files(
        streaming, size):
    handler, fs = streaming
    _write(fs, 'root/a.bin', b'x' * (2 * PART))
    data = os.urandom(size)
    _write(fs, 'root/a.bin', data)
    assert _read(fs, 'root/a.bin') == data
    if size <= 6:
        assert handler.objects == {'root/a.bin': data}
        return
    parts = [f'{i:05d}' for i in range(-(-size // PART))]
    (generation, stored), = _generations(handler).items()
    assert stored == parts + ['_SUCCESS']
    assert handler.streamed[-len(parts):] == [
        f'{generation}/{part}' for part in parts
    ]
    assert len(handler.spooled) == 2
    with fs.open_input_file('root/a.bin') as reader:
        assert reader.read_at(4, PART - 2) == data[PART - 2:PART + 2]


def test_opening_a_file_again_reuses_its_layout(store):
    handler, fs = store
    _write(fs, 'root/a.bin', b'a' * (2 * PART))
    probes = []
    size, listing = handler._size, handler._list
    handler._size = lambda path: probes.append(path) or size(path)
    handler._list = lambda path: probes.append(path) or listing(path)
    for _ in range(3):
        assert _read(fs, 'root/a.bin') == b'a' * (2 * PART)
    assert len(probes) == 2
    _write(fs, 'root/a.bin', b'new')
    assert _read(fs, 'root/a.bin') == b'new'
    fs.delete_dir_contents('root/')
    with pytest.raises(FileNotFoundError):
        fs.open_input_file('root/a.bin')


def test_layout_cache_survives_concurrent_opens_and_writes(store):
    _, fs = store
    for i in range(20):
        _write(fs, f'root/{i}.bin', b'x' * (PART + 1))
    errors = []

    def work(i):
        try:
            for _ in range(50):
                _read(fs, f'root/{i}.bin')
                _write(fs, f'root/{i}.bin', b'x' * (PART + 1))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=work, args=(i, )) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


@pytest.mark.parametrize('size', [7, PART, PART + 1, 3 * PART])
def test_a_known_size_streams_without_spooling(streaming, size):
    handler, fs = streaming
    _write(fs, 'root/a.bin', b'x' * (2 * PART))
    handler.spooled.clear()
    data = os.urandom(size)
    with open_sized(fs, 'root/a.bin', size) as writer:
        for start in range(0, size, 5):
            writer.write(data[start:start + 5])
    assert _read(fs, 'root/a.bin') == data
    plain = _is_plain(handler)
    assert plain == (size <= PART)
    # Only the empty _SUCCESS marker of a chunked file is put from disk.
    assert len(handler.spooled) == (0 if plain else 1)


def test_reads_spanning_parts_fill_the_callers_buffer(store):
    _, fs = store
    data = os.urandom(3 * PART + 3)
    _write(fs, 'root/a.bin', data)
    with fs.open_input_file('root/a.bin') as reader:
        buffer = bytearray(2 * PART)
        reader.seek(PART - 3)
        assert reader.readinto(buffer) == 2 * PART
        assert bytes(buffer) == data[PART - 3:3 * PART - 3]
        assert reader.read_at(4, 2) == data[2:6]


@pytest.mark.parametrize('size', [1, 3 * PART])
def test_a_writer_that_fails_to_open_keeps_the_old_file(
        monkeypatch, tmp_path, streaming, size):
    handler, fs = streaming
    handler.spool_dir = str(tmp_path)
    _write(fs, 'root/a.bin', b'preserve-me')

    def fail(path):
        raise OSError('cannot open')

    monkeypatch.setattr(handler, '_stream', fail)
    with pytest.raises(OSError, match='cannot open'):
        chunked_fs._Writer(handler, 'root/a.bin', size=size)
    gc.collect()
    assert _read(fs, 'root/a.bin') == b'preserve-me'
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize('newer_commits_first', [True, False])
def test_overlapping_writers_on_one_path_never_break_each_other(
        store, newer_commits_first):
    handler, fs = store
    older = fs.open_output_stream('root/a.bin')
    older.write(b'o' * 19)
    newer = fs.open_output_stream('root/a.bin')
    newer.write(b'n' * 21)
    first, second = (newer, older) if newer_commits_first else (older, newer)
    first.close()
    older_rest = b'o' * 18 if newer_commits_first else b''
    if older_rest:
        older.write(older_rest)
    second.close()
    handler._forget('root/a.bin')
    assert _read(fs, 'root/a.bin') == b'n' * 21
    assert fs.get_file_info('root/a.bin').size == 21


def test_a_committed_generation_with_missing_parts_is_skipped(store):
    handler, fs = store
    _write(fs, 'root/a.bin', b'1' * (2 * PART))
    (good, _), = _generations(handler).items()
    newer = good.rsplit('/', 1)[0] + '/99999999999999999999-ffffffff'
    handler.objects[f'{newer}/00001'] = b'2' * PART
    handler.objects[f'{newer}/_SUCCESS'] = b''
    handler._forget('root/a.bin')
    assert _read(fs, 'root/a.bin') == b'1' * (2 * PART)
    for name in [n for n in handler.objects if n.startswith(good + '/')]:
        del handler.objects[name]
    handler._forget('root/a.bin')
    assert fs.get_file_info('root/a.bin').type is FileType.NotFound
