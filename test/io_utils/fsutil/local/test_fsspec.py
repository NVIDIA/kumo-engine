# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import tempfile

import fsspec
import pytest

from io_utils import fsutil
from io_utils.fsutil import compat_fsspec
from io_utils.fsutil.compat_fsspec import fsspec_demangle, fsspec_mangle


def test_fs():
    fs1 = fsspec.filesystem(compat_fsspec._FSSPEC_FSUTIL_SCHEME)
    assert isinstance(fs1, compat_fsspec.FsutilFilesystem)


def test_monolith_training_registers_core_fsspec_fallback(monkeypatch):
    from kumo_core.io import filesystem

    import kumo.train  # noqa: F401  # registers the monolith adapter

    original = fsspec.get_filesystem_class

    def get_filesystem_class(protocol):
        if protocol == "abfss":
            raise ValueError("Protocol not known: abfss")
        if protocol == "gs":
            return object
        return original(protocol)

    monkeypatch.setattr(fsspec, "get_filesystem_class", get_filesystem_class)

    path = "abfss://container@account.dfs.core.windows.net/path"
    assert filesystem.fsspec_mangle(path) == fsspec_mangle(path)
    s3_path = "s3://bucket/key"
    mangled_s3_path = fsspec_mangle(s3_path)
    assert filesystem.fsspec_mangle(s3_path) == mangled_s3_path
    assert filesystem.fsspec_mangle(mangled_s3_path) == mangled_s3_path
    assert filesystem.fsspec_mangle("gs://bucket/key") == "gs://bucket/key"
    assert filesystem.fsspec_mangle("/tmp/local") == "/tmp/local"


@pytest.mark.parametrize("path", [
    "/mnt/absolute/path",
    "path/ending/with/"
    "relative/path",
    "s3://s3/path",
    "dbfs:/Volumes/ucv/path",
])
def test_mangler(path):
    mangled = fsspec_mangle(path)
    assert mangled.startswith(compat_fsspec._FSSPEC_FSUTIL_PREFIX)

    demangled1 = fsspec_demangle(mangled)
    assert demangled1 == path

    suffix2 = "even/more/path"
    mangled2 = osp.join(mangled, suffix2)
    assert mangled2 == fsspec_mangle(osp.join(path, suffix2))
    demangled2 = fsspec_demangle(mangled2)
    assert demangled2 == osp.join(path, suffix2)

    fs2, path2 = fsspec.core.url_to_fs(mangled)
    assert isinstance(fs2, compat_fsspec.FsutilFilesystem)
    assert path2 == path


def test_ls():
    with tempfile.TemporaryDirectory() as temp_dir:
        fsutil.touch(osp.join(temp_dir, "touched"))
        fsutil.write_text(osp.join(temp_dir, "text"), "content")
        subdir = osp.join(temp_dir, "subdir")
        fsutil.makedirs(subdir)
        fsutil.touch(osp.join(subdir, "touched"))
        fsutil.write_text(osp.join(subdir, "text"), "content")

        local_fs = fsspec.filesystem("file")
        local_res = local_fs.ls(temp_dir, detail=True)
        fsutil_fs = fsspec.filesystem(compat_fsspec._FSSPEC_FSUTIL_SCHEME)
        fsutil_res = fsutil_fs.ls(temp_dir)

        for local in local_res:
            local_name = local["name"]

            matches = [
                wrapped for wrapped in fsutil_res
                if wrapped["name"] == local_name
            ]
            assert len(matches) == 1
            assert matches[0]["type"] == local["type"]
            if local["type"] == "file":
                assert matches[0]["size"] == local["size"]


def test_open():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1 = osp.join(temp_dir, "text1")
        mangled1 = fsspec_mangle(file1)
        fs, demangled1 = fsspec.core.url_to_fs(mangled1)
        assert file1 == demangled1

        contents = ["line1\n", "line2\nline3\n"]

        with fs.open(demangled1, 'w') as f:
            for s in contents:
                f.write(s)

        with open(file1, 'r') as f:
            assert f.read() == ''.join(contents)

        with fs.open(demangled1, 'r') as f:
            assert f.read() == ''.join(contents)

        file2 = osp.join(temp_dir, "text2")
        with open(file2, 'w') as f:
            f.write(''.join(contents))
            f.write(''.join(contents))

        with fs.open(file2, 'r') as f:
            assert f.read() == (''.join(contents)) * 2

        binary_contents = [s.encode() for s in contents]
        file3 = osp.join(temp_dir, "bin3")
        with open(file3, 'wb') as f:
            for b in binary_contents:
                f.write(b)

        with fs.open(file3, 'ab') as f:
            for b in binary_contents:
                f.write(b)

        with fs.open(file3, 'rb') as f:
            assert f.read() == (b''.join(binary_contents)) * 2


def test_mkdir():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir1 = osp.join(temp_dir, "dir1")
        mangled1 = fsspec_mangle(dir1)
        fs, _ = fsspec.core.url_to_fs(mangled1)
        fs.mkdir(dir1)
        assert fsutil.is_empty_dir(dir1)

        dir2 = osp.join(temp_dir, "dir2/subdir")
        with pytest.raises(ValueError):
            fs.mkdir(dir2, create_parents=False)
        fs.makedirs(dir2)
        assert fsutil.is_empty_dir(dir2)


def test_rm():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1 = osp.join(temp_dir, "text1")
        with open(file1, 'w') as f:
            f.write("some text")
        assert fsutil.exists(file1)
        mangled1 = fsspec_mangle(file1)
        fs, _ = fsspec.core.url_to_fs(mangled1)
        fs.rm(file1)
        assert not fsutil.exists(file1)

        dir2 = osp.join(temp_dir, "dir2")
        fsutil.makedirs(dir2)
        fsutil.touch(osp.join(dir2, "some_file"))
        with pytest.raises(ValueError):
            fs.rm(dir2)
        fs.rm(dir2, recursive=True)
        assert not fsutil.exists(dir2)
