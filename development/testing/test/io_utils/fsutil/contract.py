# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import time
from io import BytesIO
from test.io_utils.fsutil.common import no_retry, random_name
from zipfile import ZipFile

import pytest

from io_utils import fsutil


def data_str(n_chars=128) -> str:
    s = (str(n_chars) + "data" * (n_chars // 4 + 1))[:n_chars]
    assert len(s) == n_chars
    return s


def data_bytes(n_bytes=128) -> bytes:
    b = data_str(n_bytes).encode()
    assert len(b) == n_bytes
    return b


def _expected_filesize(path: str, n_bytes: int) -> int:
    return n_bytes


class FilesystemContract:
    def test_exists_and_isdir(self, root: str):
        directory = osp.join(root, f"dir-{random_name()}")
        file = osp.join(directory, f"file-{random_name()}")

        assert not fsutil.exists(directory)
        assert not fsutil.isdir(directory)
        fsutil.makedirs(directory)
        if not (fsutil.is_s3(root) or fsutil.is_gcs(root)):
            assert fsutil.exists(directory)
            assert fsutil.isdir(directory)

        assert not fsutil.exists(file)
        assert not fsutil.isdir(file)
        fsutil.write_text(file, data_str())
        assert fsutil.exists(directory)
        assert fsutil.isdir(directory)
        assert fsutil.exists(file)
        assert not fsutil.isdir(file)

    def test_read_write_binary(self, root: str):
        data = data_bytes()

        f1 = osp.join(root, f"file-{random_name()}")
        fsutil.write_binary(f1, data)
        assert fsutil.read_binary(f1) == data

        f2 = osp.join(root, f"file-{random_name()}")
        fsutil.write_binary(f2, data, compression=True)
        assert fsutil.read_binary(f2) != data
        assert fsutil.read_binary(f2, compression=True) == data

    def test_read_write_text(self, root: str):
        data = data_str()

        f1 = osp.join(root, f"file-{random_name()}")
        fsutil.write_text(f1, data)
        assert fsutil.read_text(f1) == data

        f2 = osp.join(root, f"file-{random_name()}")
        fsutil.write_text(f2, data, compression=True)
        assert fsutil.read_binary(f2) != data.encode()
        assert fsutil.read_text(f2, compression=True) == data

    def test_get_binary_handle(self, root: str):
        f1 = osp.join(root, f"handle-{random_name()}")
        payload = data_bytes(96)

        with fsutil.api.get_binary_handle(f1, mode="wb") as handle:
            handle.write(payload)

        assert fsutil.read_binary(f1) == payload

        with fsutil.api.get_binary_handle(f1, mode="rb") as handle:
            assert handle.read() == payload

        f2 = osp.join(root, f"handle-{random_name()}")
        compressed_payload = data_bytes(128)

        with fsutil.api.get_binary_handle(
                f2,
                mode="wb",
                compression=True,
        ) as handle:
            handle.write(compressed_payload)

        assert fsutil.read_binary(f2, compression=True) == compressed_payload

        with fsutil.api.get_binary_handle(
                f2,
                mode="rb",
                compression=True,
        ) as handle:
            assert handle.read() == compressed_payload

    def test_get_handle(self, root: str):
        f1 = osp.join(root, f"handle-{random_name()}")
        payload = data_str(96)

        with fsutil.api.get_handle(f1, mode="w") as handle:
            handle.write(payload)

        assert fsutil.read_text(f1) == payload

        with fsutil.api.get_handle(f1, mode="r") as handle:
            assert handle.read() == payload

        f2 = osp.join(root, f"handle-{random_name()}")
        compressed_payload = data_str(120)

        with fsutil.api.get_handle(
                f2,
                mode="w",
                compression=True,
        ) as handle:
            handle.write(compressed_payload)

        assert fsutil.read_text(f2, compression=True) == compressed_payload

        with fsutil.api.get_handle(
                f2,
                mode="r",
                compression=True,
        ) as handle:
            assert handle.read() == compressed_payload

    @pytest.mark.skip("time 193.11s")
    def test_write_large(self, root: str):
        def test_size(n_bytes: int):
            f = osp.join(root, f"file-{random_name()}")
            data = data_bytes(n_bytes)
            fsutil.write_binary(f, data)
            rt_data = fsutil.read_binary(f)
            assert data == rt_data
            fsutil.rm(f)

        test_size(50 * 1024 * 1024 + 1)
        test_size(100 * 1024 * 1024)
        test_size(100 * 1024 * 1024 + 1)
        test_size(6 * 1024 * 1024 * 1024)

    def test_get_if_present(self, root: str):
        data = data_bytes()

        assert fsutil.api._get_if_present(None) is None
        f1 = osp.join(root, f"file-{random_name()}")
        assert fsutil.api._get_if_present(f1) is None
        fsutil.write_binary(f1, data)
        assert fsutil.api._get_if_present(f1) == data
        fsutil.rm(f1)
        assert fsutil.api._get_if_present(f1) is None

    def test_listdir(self, root: str):
        # Create test directory structure
        test_dir = osp.join(root, f"test_dir_{random_name()}")
        fsutil.makedirs(test_dir)

        assert len(fsutil.listdir(test_dir)) == 0

        # Create some files and subdirectories
        files = [
            "test1.txt",
            "test2.txt",
            "data.csv",
            "config.json",
            "subdir1/test3.txt",
            "subdir1/subdir2/data2.csv",
        ]

        for file_path in files:
            full_path = osp.join(test_dir, file_path)
            if "/" in file_path:  # This is a subdirectory file
                fsutil.makedirs(osp.dirname(full_path))
            fsutil.touch(full_path)

        # Test basic listing
        entries = fsutil.listdir(test_dir)
        assert len(entries) == 5
        assert all(
            osp.basename(entry) in [
                "test1.txt",
                "test2.txt",
                "data.csv",
                "config.json",
                "subdir1",
            ] for entry in entries)

        # Test include pattern
        txt_files = fsutil.listdir(test_dir, include=["*.txt"])
        assert len(txt_files) == 2
        assert all(entry.endswith(".txt") for entry in txt_files)

        # Test exclude pattern
        non_txt_files = fsutil.listdir(test_dir, exclude=["*.txt"])
        assert len(non_txt_files) == 3  # .csv, .json, directory
        assert all(not entry.endswith(".txt") for entry in non_txt_files)

        # Test suffix
        csv_files_and_dir = fsutil.listdir(test_dir, suffix=".csv")
        assert len(csv_files_and_dir) == 2  # a .csv and directory.
        assert any(entry.endswith(".csv") for entry in csv_files_and_dir)
        assert any(entry.endswith("subdir1") for entry in csv_files_and_dir)

        subdir1_listdir = fsutil.listdir(osp.join(test_dir, "subdir1"))
        assert any(entry.endswith("subdir1/test3.txt") for entry in subdir1_listdir)
        assert any(entry.endswith("subdir1/subdir2") for entry in subdir1_listdir)

        subdir2_listdir = fsutil.listdir(osp.join(test_dir, "subdir1/subdir2"))
        assert len(subdir2_listdir) == 1
        assert subdir2_listdir[0].endswith("data2.csv")

        # Test max_items
        listdir_output = fsutil.listdir(test_dir, max_items=2)
        assert len(listdir_output) == 2

        listdir_output = fsutil.listdir(test_dir, max_items=5)
        assert len(listdir_output) == 5

        listdir_output = fsutil.listdir(test_dir, max_items=7)
        assert len(listdir_output) == 5

        # Test max_items being out of range.
        with pytest.raises(ValueError):
            fsutil.listdir(test_dir, max_items=-1)
        with pytest.raises(ValueError):
            fsutil.listdir(test_dir, max_items=0)

    def test_listdir_detail(self, root: str):
        test_dir = osp.join(root, f"test_dir_{random_name()}")
        fsutil.makedirs(test_dir)
        fsutil.touch(osp.join(test_dir, "test1.txt"))
        fsutil.makedirs(osp.join(test_dir, "nested"))
        fsutil.touch(osp.join(test_dir, "nested", "test2.txt"))

        if any(
                check(root) for check in (fsutil.is_ucv, fsutil.is_adls,
                                          fsutil.is_s3, fsutil.is_gcs)):
            entries = fsutil.api._listdir_detail(test_dir)
            assert all(
                isinstance(entry, fsutil.api.PathDetail) for entry in entries)
            assert any(entry.path.endswith("test1.txt") for entry in entries)
            assert any(
                entry.path.endswith("nested") and entry.is_directory
                for entry in entries)
        else:
            with pytest.raises(NotImplementedError):
                fsutil.api._listdir_detail(test_dir)

    def test_contains_subdir(self, root: str):
        test_dir = osp.join(root, f"contains_subdir_{random_name()}")
        fsutil.makedirs(test_dir)

        # Flat directory has files only, so contains_subdir should be false.
        fsutil.touch(osp.join(test_dir, "top_level.txt"))
        assert not fsutil.contains_subdir(test_dir)

        # Nested child directory should flip contains_subdir to true.
        fsutil.makedirs(osp.join(test_dir, "nested"))
        fsutil.touch(osp.join(test_dir, "nested", "child.txt"))
        assert fsutil.contains_subdir(test_dir)

    def test_rm(self, root: str):
        n1 = f"file-{random_name()}"
        f1 = osp.join(root, n1)
        d2 = osp.join(root, f"dir-{random_name()}")
        f2 = osp.join(d2, f"file2-{random_name()}")
        f3 = osp.join(d2, f"file3-{random_name()}")

        # rm file
        assert not fsutil.exists(f1)
        fsutil.touch(f1)
        assert fsutil.exists(f1)
        fsutil.rm(f1)
        assert not fsutil.exists(f1)

        # rm dir
        fsutil.makedirs(d2)
        fsutil.touch(f2)
        fsutil.touch(f3)
        fsutil.rm(d2)
        assert not fsutil.exists(d2)
        assert not fsutil.exists(f2)
        assert not fsutil.exists(f3)

    def test_rm_glob(self, root: str):
        if fsutil.is_ucv(root):
            pytest.xfail("ucv does not implement glob rm")

        d2 = osp.join(root, f"dir-{random_name()}")
        f2 = osp.join(d2, f"file2-{random_name()}")
        f3 = osp.join(d2, f"file3-{random_name()}")

        # rm dir pattern
        fsutil.makedirs(d2)
        fsutil.touch(f2)
        fsutil.touch(f3)
        fsutil.rm(d2, include=['*absent_pattern*'])
        assert fsutil.exists(f2)
        assert fsutil.exists(f3)
        fsutil.rm(d2, include=['*file2-*'])
        assert not fsutil.exists(f2)
        assert fsutil.exists(f3)
        fsutil.touch(f2)
        fsutil.rm(d2, include=['*file3-*'])
        assert fsutil.exists(f2)
        assert not fsutil.exists(f3)
        fsutil.touch(f3)
        fsutil.rm(d2, include=['*'])
        assert not fsutil.exists(d2)
        assert not fsutil.exists(f2)
        assert not fsutil.exists(f3)

    def test_copy_file(self, root: str, temp_dir: str):
        data = data_str()

        # local file to remote file
        n1 = f"file-{random_name()}"
        l1 = osp.join(temp_dir, n1)
        fsutil.write_text(l1, data)
        r1 = osp.join(root, n1)
        fsutil.copy_file(l1, r1)
        assert fsutil.read_text(r1) == data

        # local file to remote dir
        n2 = f"file-{random_name()}"
        l2 = osp.join(temp_dir, n2)
        fsutil.write_text(l2, data)
        fsutil.copy_file(l2, root)
        r2 = osp.join(root, n2)
        assert fsutil.read_text(r2) == data

        # remote file to local file
        n3 = f"file-{random_name()}"
        r3 = osp.join(root, n3)
        fsutil.write_text(r3, data)
        l3 = osp.join(temp_dir, n3)
        fsutil.copy_file(r3, l3)
        assert fsutil.read_text(l3) == data

        # local file to remote dir
        n4 = f"file-{random_name()}"
        r4 = osp.join(root, n4)
        fsutil.write_text(r4, data)
        fsutil.copy_file(r4, temp_dir)
        l4 = osp.join(temp_dir, n4)
        assert fsutil.read_text(l4) == data

    def test_copy(self, root: str):
        data = data_str()
        source = osp.join(root, f"source-{random_name()}")
        destination = osp.join(root, f"destination-{random_name()}")
        fsutil.makedirs(osp.join(source, "nested"))
        fsutil.write_text(osp.join(source, "top.txt"), data)
        fsutil.write_text(osp.join(source, "nested", "child.txt"), data)

        fsutil.copy(source, destination)

        assert fsutil.read_text(osp.join(destination, "top.txt")) == data
        assert fsutil.read_text(osp.join(destination, "nested",
                                         "child.txt")) == data

    def test_copy_tree_threaded(self, root: str, temp_dir: str):
        """Parametrized over every backend (``root_local``, ``root_s3``,
        ``root_gcs``, ``root_adls``, ``root_ucv``). Confirms tree
        preservation, predicate exclusion, and per-task error surfacing
        end-to-end against each backend's real transport.
        """
        # Build a small staged tree locally with nested subdirs and a marker
        # file that we'll exclude from the upload.
        src_dir = osp.join(temp_dir, f"src-{random_name()}")
        fsutil.makedirs(src_dir)
        fsutil.write_binary(osp.join(src_dir, 'top.txt'), b'TOP')
        fsutil.write_binary(osp.join(src_dir, 'manifest.json'), b'COMMIT')
        sub1 = osp.join(src_dir, 'sub1')
        fsutil.makedirs(sub1)
        fsutil.write_binary(osp.join(sub1, 'a.parquet'), b'A')
        fsutil.write_binary(osp.join(sub1, 'b.parquet'), b'B')
        deep = osp.join(src_dir, 'sub2', 'deep')
        fsutil.makedirs(deep)
        fsutil.write_binary(osp.join(deep, 'leaf.txt'), b'LEAF')

        dst_dir = osp.join(root, f"dst-{random_name()}")

        # Predicate excludes the top-level commit marker by basename;
        # nested files with the same basename (none here) would also be
        # skipped — see the unit test in test_fsutil.py for top-level-
        # only behavior.
        fsutil.copy_tree_threaded(
            src_dir,
            dst_dir,
            exclude=lambda p: osp.basename(p) == 'manifest.json',
        )

        assert fsutil.read_binary(osp.join(dst_dir, 'top.txt')) == b'TOP'
        assert fsutil.read_binary(osp.join(dst_dir, 'sub1', 'a.parquet')) == b'A'
        assert fsutil.read_binary(osp.join(dst_dir, 'sub1', 'b.parquet')) == b'B'
        assert fsutil.read_binary(osp.join(dst_dir, 'sub2', 'deep',
                                           'leaf.txt')) == b'LEAF'
        # Excluded file was not transferred.
        assert not fsutil.exists(osp.join(dst_dir, 'manifest.json'))

    def test_filesize(self, root: str):
        for n_bytes in [0, 1, 127, 128, 129]:
            data = data_bytes(n_bytes)
            f1 = osp.join(root, f"file-{random_name()}")
            assert fsutil.filesize(f1) is None
            fsutil.write_binary(f1, data)
            assert fsutil.filesize(f1) == _expected_filesize(root, n_bytes)
            fsutil.rm(f1)
            assert fsutil.filesize(f1) is None

        d1 = osp.join(root, f"dir-{random_name()}")
        f2 = osp.join(d1, f"file-{random_name()}")
        assert fsutil.filesize(d1) is None
        assert fsutil.filesize(f2) is None

        fsutil.makedirs(d1)
        assert fsutil.filesize(d1) is None
        assert fsutil.filesize(f2) is None

        for encoding in [None, 'utf-8', 'utf-32', 'latin-1']:
            data = data_str()
            fsutil.write_text(f2, data, encoding=encoding)
            encoded_len = len(data.encode(encoding) if encoding else data.encode())
            assert fsutil.filesize(f2) == _expected_filesize(root, encoded_len)

    def test_dirsize(self, root: str):
        d1 = osp.join(root, f"dir-{random_name()}")
        f1 = osp.join(d1, f"file-{random_name()}")
        f2 = osp.join(d1, "nested", f"file-{random_name()}")

        fsutil.makedirs(d1)
        size0 = fsutil.dirsize(d1)

        n1 = 127
        n2 = 129
        e1 = _expected_filesize(root, n1)
        e2 = _expected_filesize(root, n2)

        fsutil.write_binary(f1, data_bytes(n1))
        size1 = fsutil.dirsize(d1)
        assert size1 == size0 + e1

        fsutil.makedirs(osp.dirname(f2))
        size2 = fsutil.dirsize(d1)
        assert size2 >= size1

        fsutil.write_binary(f2, data_bytes(n2))
        size3 = fsutil.dirsize(d1)
        assert size3 == size2 + e2

        fsutil.rm(f1)
        size4 = fsutil.dirsize(d1)
        assert size4 == size3 - e1

        fsutil.rm(f2)
        size5 = fsutil.dirsize(d1)
        assert size5 == size4 - e2

    def test_file_xor_dir(self, root: str):
        from azure.core.exceptions import \
            ResourceExistsError as AzureResourceExistsError
        from databricks.sdk.errors.platform import ResourceConflict
        df1 = osp.join(root, f"dirfile-{random_name()}")
        fsutil.makedirs(df1)

        # Divergent behavior when writing to a dir as though it's a file
        # TODO make this consistent
        if fsutil.is_local(root):
            with pytest.raises(IsADirectoryError):
                fsutil.write_text(df1, "blah")
        elif fsutil.is_ucv(root):
            with no_retry():
                with pytest.raises(ResourceConflict):
                    fsutil.write_text(df1, "blah")
        elif fsutil.is_adls(root):
            # Azure SDK assumption: ADLS has real hierarchical directories, so
            # writing a file at a path that already exists as a directory raises
            # 409 PathConflict (ResourceExistsError). S3/GCS silently succeed
            # because their "directories" are virtual key prefixes, not real objects.
            with pytest.raises(AzureResourceExistsError):
                fsutil.write_text(df1, "blah")
        else:
            fsutil.write_text(df1, "blah")

    def test_touch(self, root: str):
        f1 = osp.join(root, f"file-{random_name()}")
        assert not fsutil.exists(f1)
        fsutil.touch(f1)
        assert fsutil.exists(f1)
        fsutil.touch(f1)
        fsutil.touch(f1)
        assert fsutil.exists(f1)
        fsutil.rm(f1)
        assert not fsutil.exists(f1)
        fsutil.touch(f1)
        assert fsutil.exists(f1)

    def test_has_write_permission(self, root: str):
        target = osp.join(root, f"nested-{random_name()}", f"file-{random_name()}")
        assert fsutil.has_write_permission(target) is True

    def test_get_last_modified_time(self, root: str):
        missing_file = osp.join(root, f"missing-{random_name()}")
        default_ts = fsutil.get_last_modified_time(missing_file)
        assert default_ts <= 0

        touched_file = osp.join(root, f"file-{random_name()}.txt")
        fsutil.touch(touched_file)
        modified_ts = fsutil.get_last_modified_time(touched_file)
        assert modified_ts > default_ts
        # allow skew for backend clock differences
        assert modified_ts <= time.time() + 60

        dir_ts = fsutil.get_last_modified_time(root)
        if fsutil.is_ucv(root) or fsutil.is_adls(root):
            # UCV and ADLS directory timestamps come from an in-dir _SUCCESS file.
            assert dir_ts == default_ts
        else:
            assert dir_ts >= modified_ts

    def test_get_presigned_helper(self, root: str):
        from io_utils.fsutil.impl_adls import ADLSPresignedHelper
        from io_utils.fsutil.impl_gcs import GCSPresignedHelper
        from io_utils.fsutil.impl_local import LocalPresignedHelper
        from io_utils.fsutil.impl_s3 import S3PresignedHelper
        helper = fsutil.get_presigned_helper(root)
        expected_type = None
        if fsutil.is_s3(root):
            expected_type = S3PresignedHelper
        elif fsutil.is_gcs(root):
            expected_type = GCSPresignedHelper
        elif fsutil.is_adls(root):
            expected_type = ADLSPresignedHelper
        elif fsutil.is_local(root):
            expected_type = LocalPresignedHelper

        if expected_type is None:
            assert helper is None
        else:
            assert isinstance(helper, expected_type)

    def test_customer_facing_path(self, root: str):
        customer_path = fsutil.customer_facing_path(root)

        assert customer_path == root

    def test_zip_directory(self, root: str):
        source_dir = osp.join(root, f"zip-source-{random_name()}")
        nested_file = osp.join(source_dir, "nested", "payload.txt")
        top_file = osp.join(source_dir, "payload.txt")

        fsutil.makedirs(osp.dirname(nested_file))
        fsutil.write_text(top_file, data_str(32))
        fsutil.write_text(nested_file, data_str(48))

        if fsutil.is_s3(root) or fsutil.is_gcs(root) or fsutil.is_adls(root):
            zip_path = fsutil.api._zip_directory(source_dir, recursive=True)
            assert zip_path.endswith(".zip")
            assert fsutil.exists(zip_path)
            assert fsutil.api._zip_directory(source_dir, recursive=True) == zip_path

            archive_bytes = fsutil.read_binary(zip_path)
            with ZipFile(BytesIO(archive_bytes)) as zip_file:
                archive_entries = zip_file.namelist()

            assert any(entry.endswith("payload.txt") for entry in archive_entries)
        else:
            with pytest.raises(NotImplementedError):
                fsutil.api._zip_directory(source_dir, recursive=True)

    def test_makedirs(self, root: str):
        if fsutil.is_s3(root) or fsutil.is_gcs(root):
            # Object stores have virtual directories; making one is a no-op.
            d1 = osp.join(root, f"dir-{random_name()}")
            assert not fsutil.exists(d1)
            fsutil.makedirs(d1)
            assert not fsutil.exists(d1)
            return
        d1 = osp.join(root, f"dir-{random_name()}")
        assert not fsutil.isdir(d1)
        fsutil.makedirs(d1, exist_ok=False)
        assert fsutil.isdir(d1)
        fsutil.makedirs(d1, exist_ok=True)
        assert fsutil.isdir(d1)
        with pytest.raises(FileExistsError):
            fsutil.makedirs(d1, exist_ok=False)
        assert fsutil.isdir(d1)

    @pytest.mark.skip("time 214.64s")
    def test_large_filesize(self, root: str):
        f1 = osp.join(root, f"file-{random_name()}")
        for n_bytes in [2**p + o for p in [10, 20, 30] for o in [-1, 0, 1]]:
            fsutil.write_binary(f1, data_bytes(n_bytes))
            assert fsutil.filesize(f1) == _expected_filesize(root, n_bytes)
