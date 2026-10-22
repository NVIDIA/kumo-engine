# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A pyarrow filesystem over stores that only put, list, read and remove.

A file larger than :attr:`ChunkedFileHandler.part_bytes` is stored as
``<name>.chunks/00000 ... NNNNN`` followed by an empty ``_SUCCESS``; anything
smaller is one plain object. Readers see one logical file either way. A write
removes the other form; if a crash leaves both, the plain object wins.
"""

import io
import os
import posixpath
import shutil
import tempfile
from typing import Dict, List, Optional, Protocol, Tuple

import pyarrow as pa
from pyarrow.fs import FileInfo, FileSystemHandler, FileType

PART_BYTES = 256 << 20
CHUNKS = '.chunks'
DONE = '_SUCCESS'


class Part(Protocol):
    def read_at(self, offset: int, length: int) -> bytes:
        ...

    def close(self) -> None:
        ...


class ChunkedFileHandler(FileSystemHandler):
    r"""Subclasses supply the five store primitives below."""
    part_bytes = PART_BYTES

    def _size(self, path: str) -> Optional[int]:
        r"""Size of exactly the object at ``path``, or ``None``."""
        raise NotImplementedError

    def _list(self, directory: str) -> Dict[str, int]:
        r"""Every object below ``directory/``, by full path."""
        raise NotImplementedError

    def _put(self, local: str, path: str) -> None:
        r"""Uploads ``local``, whose basename matches ``path``'s."""
        raise NotImplementedError

    def _part(self, path: str, size: int) -> Part:
        raise NotImplementedError

    def _remove(self, path: str, *, tree: bool) -> None:
        r"""Removes the object at ``path``, or everything below it."""
        raise NotImplementedError

    def normalize_path(self, path):
        return path.rstrip('/')

    def _layout(self, path: str) -> Optional[List[Tuple[str, int]]]:
        size = self._size(path)
        if size is not None:
            return [(path, size)]
        return _logical_parts(self._list(path + CHUNKS), path + CHUNKS)

    def get_file_info(self, paths):
        infos = []
        for path in paths:
            layout = self._layout(path)
            if layout is not None:
                infos.append(
                    FileInfo(path, FileType.File,
                             size=sum(size for _, size in layout)))
            elif self._list(path):
                infos.append(FileInfo(path, FileType.Directory))
            else:
                infos.append(FileInfo(path, FileType.NotFound))
        return infos

    def get_file_info_selector(self, selector):
        base = selector.base_dir.rstrip('/')
        files = _logical_files(self._list(base))
        if not files and not selector.allow_not_found:
            raise FileNotFoundError(f'{base} does not exist.')
        infos: Dict[str, FileInfo] = {}
        for path, size in files.items():
            parts = path[len(base) + 1:].split('/')
            depths = len(parts) if selector.recursive else min(len(parts), 2)
            for depth in range(1, depths):
                directory = f"{base}/{'/'.join(parts[:depth])}"
                infos[directory] = FileInfo(directory, FileType.Directory)
            if selector.recursive or len(parts) == 1:
                infos[path] = FileInfo(path, FileType.File, size=size)
        return [infos[path] for path in sorted(infos)]

    def create_dir(self, path, recursive):
        pass

    def delete_dir(self, path):
        self._remove(path, tree=True)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        if not missing_dir_ok and not self._list(path):
            raise FileNotFoundError(f'{path} does not exist.')
        self._remove(path, tree=True)

    def delete_root_dir_contents(self):
        raise NotImplementedError('Refusing to delete the whole store.')

    def delete_file(self, path):
        plain = self._size(path) is not None
        chunked = bool(self._list(path + CHUNKS))
        if not plain and not chunked:
            raise FileNotFoundError(f'{path} does not exist.')
        if plain:
            self._remove(path, tree=False)
        if chunked:
            self._remove(path + CHUNKS, tree=True)

    def move(self, src, dest):
        raise NotImplementedError(f'{self.get_type_name()} cannot rename.')

    def copy_file(self, src, dest):
        raise NotImplementedError(f'{self.get_type_name()} cannot copy.')

    def open_append_stream(self, path, metadata):
        raise NotImplementedError(f'{self.get_type_name()} cannot append.')

    def open_input_stream(self, path):
        return self.open_input_file(path)

    def open_input_file(self, path):
        layout = self._layout(path)
        if layout is None:
            raise FileNotFoundError(f'{path} does not exist.')
        return pa.PythonFile(_Reader(self, layout), mode='r')

    def open_output_stream(self, path, metadata):
        return pa.PythonFile(_Writer(self, path), mode='w')


def _logical_parts(objects: Dict[str, int],
                   directory: str) -> Optional[List[Tuple[str, int]]]:
    if f'{directory}/{DONE}' not in objects:
        return None
    return sorted((path, size) for path, size in objects.items()
                  if posixpath.basename(path) != DONE)


def _logical_files(objects: Dict[str, int]) -> Dict[str, int]:
    r"""Collapses committed ``.chunks`` directories into their files."""
    files: Dict[str, int] = {}
    chunked: Dict[str, Dict[str, int]] = {}
    for path, size in objects.items():
        directory = posixpath.dirname(path)
        if directory.endswith(CHUNKS):
            chunked.setdefault(directory, {})[path] = size
        else:
            files[path] = size
    for directory, parts in chunked.items():
        name = directory[:-len(CHUNKS)]
        layout = _logical_parts(parts, directory)
        if layout is not None and name not in files:
            files[name] = sum(size for _, size in layout)
    return files


class _Reader(io.RawIOBase):
    def __init__(self, handler: ChunkedFileHandler,
                 layout: List[Tuple[str, int]]) -> None:
        self._handler = handler
        self._layout: List[Tuple[str, int, int]] = []
        start = 0
        for path, size in layout:
            self._layout.append((path, start, size))
            start += size
        self._size = start
        self._open: Dict[str, Part] = {}
        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self._position,
            io.SEEK_END: self._size
        }[whence]
        self._position = max(base + offset, 0)
        return self._position

    def read(self, size=-1):
        end = self._size if size is None or size < 0 else min(
            self._position + size, self._size)
        out = bytearray()
        for path, start, length in self._layout:
            if start + length <= self._position or start >= end:
                continue
            offset = max(self._position, start) - start
            want = min(end, start + length) - start - offset
            part = self._open.get(path)
            if part is None:
                part = self._open[path] = self._handler._part(path, length)
            chunk = part.read_at(offset, want)
            if len(chunk) != want:
                raise OSError(f'{path} returned {len(chunk)} of {want} bytes')
            out += chunk
        self._position += len(out)
        return bytes(out)

    def readall(self):
        return self.read()

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            for part in self._open.values():
                part.close()
        super().close()


class _Writer(io.RawIOBase):
    def __init__(self, handler: ChunkedFileHandler, path: str) -> None:
        self._handler = handler
        self._path = path
        self._dir = tempfile.mkdtemp(prefix='kumo-chunked-')
        self._spool = open(os.path.join(self._dir, 'spool'), 'wb')
        self._filled = 0
        self._parts = 0
        self._written = 0

    def writable(self):
        return True

    def tell(self):
        return self._written

    def write(self, data):
        view = memoryview(data).cast('B')
        total = len(view)
        while view:
            if self._filled == self._handler.part_bytes:
                self._flush_part()
            take = min(len(view), self._handler.part_bytes - self._filled)
            self._spool.write(view[:take])
            self._filled += take
            view = view[take:]
        self._written += total
        return total

    def _upload(self, target: str) -> None:
        self._spool.close()
        local = os.path.join(self._dir, posixpath.basename(target))
        os.replace(os.path.join(self._dir, 'spool'), local)
        try:
            self._handler._put(local, target)
        finally:
            os.remove(local)
        self._spool = open(os.path.join(self._dir, 'spool'), 'wb')
        self._filled = 0

    def _flush_part(self) -> None:
        chunks = self._path + CHUNKS
        if not self._parts:
            self._handler._remove(chunks, tree=True)
            if self._handler._size(self._path) is not None:
                self._handler._remove(self._path, tree=False)
        self._upload(f'{chunks}/{self._parts:05d}')
        self._parts += 1

    def close(self):
        if self.closed:
            return
        try:
            if self._parts:
                self._flush_part()
                self._upload(f'{self._path}{CHUNKS}/{DONE}')
            else:
                self._upload(self._path)
                if self._handler._list(self._path + CHUNKS):
                    self._handler._remove(self._path + CHUNKS, tree=True)
        finally:
            self._spool.close()
            shutil.rmtree(self._dir, ignore_errors=True)
            super().close()
