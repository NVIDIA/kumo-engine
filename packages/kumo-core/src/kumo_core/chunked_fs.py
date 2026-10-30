# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A pyarrow filesystem storing large files as ``<name>.chunks/`` parts."""

import contextlib
import io
import os
import posixpath
import shutil
import tempfile
import time
import uuid
from typing import (
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import pyarrow as pa
from pyarrow.fs import FileInfo, FileSystemHandler, FileType

PART_BYTES = 256 << 20
CHUNKS = '.chunks'
DONE = '_SUCCESS'
SIZE_KEY = 'kumo-size'
_spool_dir: Optional[Callable[[], str]] = None


def set_spool_dir(spool_dir: Optional[Callable[[], str]]) -> None:
    r"""Where writers spool parts unless a handler sets ``spool_dir``."""
    global _spool_dir
    _spool_dir = spool_dir


class Part(Protocol):
    def read_at(self, offset: int, length: int) -> bytes:
        ...

    def close(self) -> None:
        ...


def spool_root(handler: 'ChunkedFileHandler') -> Optional[str]:
    root = handler.spool_dir or (_spool_dir() if _spool_dir else None)
    if root is not None:
        os.makedirs(root, exist_ok=True)
    return root


class ChunkedFileHandler(FileSystemHandler):
    r"""Subclasses supply the store primitives below."""
    part_bytes = PART_BYTES
    spool_dir: Optional[str] = None
    plain_bytes = PART_BYTES

    def _stream(self, path: str) -> BinaryIO:
        r"""Optional: a writable committed by ``close``, dropped by ``abort``."""
        raise NotImplementedError

    def _size(self, path: str) -> Optional[int]:
        raise NotImplementedError

    def _list(self, directory: str) -> Dict[str, int]:
        raise NotImplementedError

    def _put(self, local: str, path: str) -> None:
        r"""Upload ``local``; its basename matches ``path``'s."""
        raise NotImplementedError

    def _part(self, path: str, size: int) -> Part:
        raise NotImplementedError

    def _remove(self, path: str, *, tree: bool) -> None:
        raise NotImplementedError

    def normalize_path(self, path):
        return path.rstrip('/')

    def _layout(self, path: str) -> Optional[List[Tuple[str, int]]]:
        cache = self.__dict__.setdefault('_layouts', {})
        layout = cache.get(path)
        if layout is not None:
            return layout
        size = self._size(path)
        layout = ([(path, size)] if size is not None else _logical_parts(
            self._list(path + CHUNKS), path + CHUNKS))
        if layout is not None:
            cache[path] = layout
        return layout

    def _forget(self, path: str) -> None:
        cache = self.__dict__.get('_layouts', {})
        path = path.rstrip('/')
        for name in list(cache):
            if name == path or name.startswith(path + '/'):
                cache.pop(name, None)

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
        path = path.rstrip('/')
        self._forget(path)
        self._remove(path, tree=True)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        path = path.rstrip('/')
        self._forget(path)
        if not missing_dir_ok and not self._list(path):
            raise FileNotFoundError(f'{path} does not exist.')
        self._remove(path, tree=True)

    def delete_root_dir_contents(self):
        raise NotImplementedError('Refusing to delete the whole store.')

    def delete_file(self, path):
        self._forget(path)
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
        self._forget(path)
        size = (metadata or {}).get(SIZE_KEY)
        return pa.PythonFile(
            _Writer(self, path, None if size is None else int(size)),
            mode='w')


def _logical_parts(objects: Dict[str, int],
                   directory: str) -> Optional[List[Tuple[str, int]]]:
    committed = [
        posixpath.dirname(path) for path in objects
        if posixpath.basename(path) == DONE
        and posixpath.dirname(posixpath.dirname(path)) == directory
    ]
    if not committed:
        return None
    generation = max(committed)
    return sorted((path, size) for path, size in objects.items()
                  if posixpath.dirname(path) == generation
                  and posixpath.basename(path) != DONE)


def _logical_files(objects: Dict[str, int]) -> Dict[str, int]:
    files: Dict[str, int] = {}
    chunked: Dict[str, Dict[str, int]] = {}
    for path, size in objects.items():
        directory = posixpath.dirname(posixpath.dirname(path))
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

    def _pieces(self, end: int):
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
            yield chunk

    def read(self, size=-1):
        end = self._size if size is None or size < 0 else min(
            self._position + size, self._size)
        if end <= self._position:
            return b''
        pieces = self._pieces(end)
        first = next(pieces)
        if len(first) == end - self._position:
            self._position = end
            return first
        out = bytearray(end - self._position)
        out[:len(first)] = first
        filled = len(first)
        for chunk in pieces:
            out[filled:filled + len(chunk)] = chunk
            filled += len(chunk)
        self._position = end
        return out

    def readall(self):
        return self.read()

    def readinto(self, buffer):
        view = memoryview(buffer).cast('B')
        end = min(self._position + len(view), self._size)
        filled = 0
        for chunk in self._pieces(end):
            view[filled:filled + len(chunk)] = chunk
            filled += len(chunk)
        self._position += filled
        return filled

    def close(self):
        if not self.closed:
            for part in self._open.values():
                part.close()
        super().close()


def _abandon(stream: Optional[BinaryIO], close: bool = False) -> None:
    if stream is None:
        return
    abort = getattr(stream, 'abort', stream.close if close else None)
    if abort is not None:
        with contextlib.suppress(Exception):
            abort()


class _Writer(io.RawIOBase):
    def __init__(self, handler: ChunkedFileHandler, path: str,
                 size: Optional[int] = None) -> None:
        self._handler = handler
        self._path = path
        self._chunks = (f'{path}{CHUNKS}/{time.time_ns():020d}-'
                        f'{uuid.uuid4().hex[:8]}')
        self._dir = tempfile.mkdtemp(prefix='kumo-chunked-',
                                     dir=spool_root(handler))
        self._spool = open(os.path.join(self._dir, 'spool'), 'wb')
        self._filled = 0
        self._parts = 0
        self._written = 0
        self._part: Optional[BinaryIO] = None
        self._streams = type(handler)._stream is not ChunkedFileHandler._stream
        self._limit = (min(handler.plain_bytes, handler.part_bytes)
                       if self._streams else handler.part_bytes)
        self._plain: Optional[BinaryIO] = None
        self._failed = False
        if self._streams and size is not None:
            if size <= handler.part_bytes:
                self._plain = handler._stream(path)
            else:
                self._part = self._open_part()

    def writable(self):
        return True

    def tell(self):
        return self._written

    def write(self, data):
        try:
            return self._write(memoryview(data).cast('B'))
        except BaseException:
            self._failed = True
            raise

    def _write(self, view: memoryview) -> int:
        total = len(view)
        if self._plain is not None:
            self._plain.write(view)
            self._written += total
            return total
        while view:
            if self._part is not None:
                if self._filled == self._handler.part_bytes:
                    self._next_stream()
                take = min(len(view), self._handler.part_bytes - self._filled)
                self._part.write(view[:take])
            else:
                if self._filled == self._limit:
                    if self._streams:
                        self._start_stream()
                        continue
                    self._flush_part()
                take = min(len(view), self._limit - self._filled)
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

    def _commit_parts(self) -> None:
        self._upload(f'{self._chunks}/{DONE}')
        if self._handler._size(self._path) is not None:
            self._handler._remove(self._path, tree=False)
        stale = {
            posixpath.dirname(path)
            for path in self._handler._list(self._path + CHUNKS)
        } - {self._chunks}
        for generation in stale:
            self._handler._remove(generation, tree=True)

    def _flush_part(self) -> None:
        self._upload(f'{self._chunks}/{self._parts:05d}')
        self._parts += 1

    def _open_part(self) -> BinaryIO:
        return self._handler._stream(f'{self._chunks}/{self._parts:05d}')

    def _start_stream(self) -> None:
        self._spool.close()
        spooled = os.path.join(self._dir, 'spool')
        self._part = self._open_part()
        with open(spooled, 'rb') as reader:
            shutil.copyfileobj(reader, self._part, 8 << 20)
        self._spool = open(spooled, 'wb')

    def _next_stream(self) -> None:
        assert self._part is not None
        self._part.close()
        self._parts += 1
        self._part = self._open_part()
        self._filled = 0

    def close(self):
        if self.closed:
            return
        try:
            if self._failed:
                _abandon(self._plain)
                _abandon(self._part, close=True)
            elif self._plain is not None:
                self._plain.close()
                self._plain = None
                if self._handler._list(self._path + CHUNKS):
                    self._handler._remove(self._path + CHUNKS, tree=True)
            elif self._part is not None:
                self._part.close()
                self._part = None
                self._commit_parts()
            elif self._parts:
                self._flush_part()
                self._commit_parts()
            else:
                self._upload(self._path)
                if self._handler._list(self._path + CHUNKS):
                    self._handler._remove(self._path + CHUNKS, tree=True)
        finally:
            self._handler._forget(self._path)
            self._spool.close()
            shutil.rmtree(self._dir, ignore_errors=True)
            super().close()
