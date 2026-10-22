# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from kumo_core.chunked_fs import ChunkedFileHandler
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
        with open(local, 'rb') as reader:
            self.objects[path] = reader.read()

    def _part(self, path, size):
        return _Bytes(self.objects[path])

    def _remove(self, path, *, tree):
        for name in list(self.objects):
            if name.startswith(path + '/') if tree else name == path:
                del self.objects[name]


@pytest.fixture
def store():
    handler = MemoryHandler()
    return handler, PyFileSystem(handler)


def _write(fs, path, data):
    with fs.open_output_stream(path) as writer:
        for start in range(0, len(data), 5):
            writer.write(data[start:start + 5])


def _read(fs, path):
    with fs.open_input_stream(path) as reader:
        return reader.readall()


@pytest.mark.parametrize('size', [0, 3, PART])
def test_files_up_to_one_part_are_one_plain_object(store, size):
    handler, fs = store
    data = bytes(range(size))
    _write(fs, 'root/a.bin', data)
    assert handler.objects == {'root/a.bin': data}
    assert fs.get_file_info('root/a.bin').size == size
    assert _read(fs, 'root/a.bin') == data


def test_larger_files_are_parts_committed_last_and_read_back_exactly(store):
    handler, fs = store
    data = os.urandom(2 * PART + 5)
    _write(fs, 'root/a.bin', data)
    assert sorted(handler.objects) == [
        'root/a.bin.chunks/00000', 'root/a.bin.chunks/00001',
        'root/a.bin.chunks/00002', 'root/a.bin.chunks/_SUCCESS'
    ]
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
    handler.objects['root/a.bin.chunks/00000'] = b'x' * PART
    assert fs.get_file_info('root/a.bin').type is not FileType.File
    with pytest.raises(FileNotFoundError):
        fs.open_input_file('root/a.bin')


@pytest.mark.parametrize('first,second', [(3, 2 * PART), (2 * PART, 3)])
def test_a_write_removes_the_other_form(store, first, second):
    handler, fs = store
    _write(fs, 'root/a.bin', b'1' * first)
    _write(fs, 'root/a.bin', b'2' * second)
    assert _read(fs, 'root/a.bin') == b'2' * second
    plain = 'root/a.bin' in handler.objects
    assert plain == (second <= PART)
    assert any('.chunks/' in name for name in handler.objects) != plain


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
