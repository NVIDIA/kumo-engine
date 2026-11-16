# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from dataclasses import asdict, dataclass
from typing import Optional

from fsspec.registry import register_implementation
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import stringify_path

from io_utils import fsutil

_FSSPEC_FSUTIL_SCHEME = "fsutil"
_FSSPEC_FSUTIL_PREFIX = f"{_FSSPEC_FSUTIL_SCHEME}://"


def fsspec_mangle(path: str) -> str:
    return _FSSPEC_FSUTIL_PREFIX + path


def fsspec_demangle(path: str) -> str:
    return path.removeprefix(_FSSPEC_FSUTIL_PREFIX)


class FsutilFilesystem(AbstractFileSystem):
    """Implementation of a fsspec Filesystem backed by fsutil
    Whenever possible, prefer to access fsutil directly.
    This layer is lower performance and exists for compatibility only.
    """
    protocol = _FSSPEC_FSUTIL_SCHEME

    # @override  # disabled due to pyright bug
    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        assert isinstance(path, str)
        return fsspec_demangle(path)

    # @override  # disabled due to pyright bug
    def unstrip_protocol(self, name):
        assert isinstance(name, str)
        return fsspec_mangle(name)

    # @override  # disabled due to pyright bug
    def mkdir(self, path, create_parents=True, **kwargs):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)

        if create_parents:
            self.makedirs(path, exist_ok=False)
        else:
            parent = osp.dirname(path)
            if not fsutil.isdir(parent):
                raise ValueError(
                    f"{parent} does not exist while creating {path}")
            fsutil.makedirs(path, exist_ok=False)

    # @override  # disabled due to pyright bug
    def makedirs(self, path, exist_ok=False):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)

        fsutil.makedirs(path, exist_ok=exist_ok)

    # @override  # disabled due to pyright bug
    def _open(self, path, mode="rb", block_size=None, autocommit=True,
              cache_options=None, **kwargs):
        """Return raw bytes-mode file-like from the file-system"""
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)

        return FsutilBufferedFile(
            self,
            path,
            mode,
            block_size,  # type: ignore
            autocommit,
            cache_options=cache_options,
            **kwargs)

    # @override  # disabled due to pyright bug
    def rmdir(self, path):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)

        if fsutil.is_empty_dir(path):
            fsutil.rm(path)
        else:
            raise ValueError(f"{path} is not empty")

    # @override  # disabled due to pyright bug
    def isdir(self, path):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)
        return fsutil.isdir(path)

    # @override  # disabled due to pyright bug
    def ls(self, path, detail=True, **kwargs):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)

        # If path is a file, return info about that file (standard fsspec behavior)
        if not fsutil.isdir(path):
            names = [path] if fsutil.exists(path) else []
        else:
            names = fsutil.listdir(path)

        if not detail:
            return names

        @dataclass
        class FileInfo:
            name: str
            type: str
            size: Optional[int]

            @classmethod
            def populate(cls, name: str):
                isdir = fsutil.isdir(name)
                return cls(
                    name=name,
                    type="directory" if isdir else "file",
                    size=None if isdir else fsutil.filesize(name),
                )

        return [asdict(FileInfo.populate(name)) for name in names]

    # @override  # disabled due to pyright bug
    def rm_file(self, path):
        path = FsutilFilesystem._strip_protocol(path)
        assert isinstance(path, str)
        if fsutil.isdir(path):
            self.rmdir(path)
        fsutil.rm(path, ignore_errors=True)


class FsutilBufferedFile(AbstractBufferedFile):
    def __init__(self, fs, path, mode="rb", block_size="default",
                 autocommit=True, cache_type="readahead", cache_options=None,
                 size=None, **kwargs):
        super().__init__(fs, path, mode, block_size, autocommit, cache_type,
                         cache_options, size, **kwargs)
        self.temp_dir = fsutil.TemporaryDirectory()
        self.stage_path = osp.join(self.temp_dir.__enter__(), "temp_file")
        if "r" in mode or "a" in mode:
            fsutil.copy_file(self.path, self.stage_path)

    # @override  # disabled due to pyright bug
    def commit(self):
        """Move from temp to final destination"""
        fsutil.copy_file(self.stage_path, self.path, overwrite=True)
        self.temp_dir.__exit__(None, None, None)

    # @override  # disabled due to pyright bug
    def discard(self):
        """Throw away temporary file"""
        self.temp_dir.__exit__(None, None, None)

    # @override  # disabled due to pyright bug
    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        with open(self.stage_path, 'ab') as f:
            f.write(self.buffer.getvalue())

        if final and self.autocommit:
            self.commit()

    # @override  # disabled due to pyright bug
    def _initiate_upload(self):
        """Create remote file/upload"""

    # @override  # disabled due to pyright bug
    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        with open(self.stage_path, 'rb') as f:
            f.seek(start, 0)
            return f.read(end - start)


register_implementation(_FSSPEC_FSUTIL_SCHEME, FsutilFilesystem)
