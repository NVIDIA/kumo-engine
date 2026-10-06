# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import os.path as osp
import pathlib
import shutil
import threading
import traceback
from typing import Optional
from unittest.mock import patch

import pytest

from io_utils import fsutil
from io_utils.fsutil.credentials import (
    credential_scope,
    credentials_for_provider,
)
from kumo.home import get_home_dir


def teardown_function():
    for f in ["test_fsutil", "test_fsutil_copy"]:
        folder = osp.join(get_home_dir(), f)
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize(('storage_root', 'supported'), [
    ('/tmp/local-root', True),
    ('s3://bucket/root', True),
    ('gs://bucket/root', True),
    ('abfss://filesystem@account.dfs.core.windows.net/root', True),
    ('dbfs:/Volumes/catalog/schema/volume/root', False),
])
def test_storage_upload_capability_is_explicit(storage_root, supported):
    helper = fsutil.get_presigned_helper(storage_root)
    assert bool(helper and helper.supports_multipart_upload) is supported
    assert fsutil.supports_multipart_upload(storage_root) is supported


def test_local_object_identity_binds_regular_file_and_rejects_symlink(
        tmp_path: pathlib.Path):
    target = tmp_path / 'part.parquet'
    target.write_bytes(b'prediction-part')

    first = fsutil.object_identity(str(target))
    second = fsutil.object_identity(str(target))
    assert first == second
    assert first.size == len(b'prediction-part')
    assert first.etag is None
    assert first.version

    link = tmp_path / 'part-link.parquet'
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        fsutil.object_identity(str(link))


def test_is_empty_dir():
    root_dir = osp.dirname(osp.realpath(__file__))
    assert not fsutil.is_empty_dir(root_dir)
    assert fsutil.is_empty_dir(osp.join(root_dir, __file__))
    assert fsutil.is_empty_dir(osp.join(root_dir, 'test_unknown_file'))


def test_copy_exclude():
    temp_dir = osp.join(get_home_dir(), 'test_fsutil')
    src_dir = osp.join(temp_dir, 'source_folder')
    dst_dir = osp.join(temp_dir, 'dest_folder')
    fsutil.makedirs(src_dir)
    fsutil.makedirs(dst_dir)
    fsutil.touch(osp.join(src_dir, 'a'))
    fsutil.touch(osp.join(src_dir, 'b'))
    fsutil.touch(osp.join(src_dir, '__SUCCESS__'))
    fsutil.exists(osp.join(src_dir, 'a'))
    fsutil.exists(osp.join(src_dir, 'b'))
    fsutil.exists(osp.join(src_dir, '__SUCCESS__'))

    fsutil.copy(src_dir, osp.join(dst_dir, 'source_folder'),
                exclude=['__SUCCESS__'])
    assert fsutil.exists(osp.join(dst_dir, 'source_folder'))
    assert fsutil.isdir(osp.join(dst_dir, 'source_folder'))
    assert fsutil.exists(osp.join(dst_dir, 'source_folder', 'a'))
    assert fsutil.exists(osp.join(dst_dir, 'source_folder', 'b'))
    assert not fsutil.exists(osp.join(dst_dir, 'source_folder', '__SUCCESS__'))


def test_copy_nested_tree_creates_local_destination_parents(
        tmp_path: pathlib.Path):
    """A multi-level local->local tree copies completely, and every
    intermediate destination directory -- none of which exist ahead of
    time -- gets created so shutil.copyfile has somewhere to write."""
    src = tmp_path / 'src'
    dst = tmp_path / 'nested' / 'dst'
    (src / 'sub1' / 'sub2').mkdir(parents=True)
    (src / 'top.txt').write_bytes(b'TOP')
    (src / 'sub1' / 'a.txt').write_bytes(b'A')
    (src / 'sub1' / 'sub2' / 'deep.txt').write_bytes(b'DEEP')

    fsutil.copy(str(src), str(dst))

    assert (dst / 'top.txt').read_bytes() == b'TOP'
    assert (dst / 'sub1' / 'a.txt').read_bytes() == b'A'
    assert (dst / 'sub1' / 'sub2' / 'deep.txt').read_bytes() == b'DEEP'


def test_copy_include_and_exclude_filters(tmp_path: pathlib.Path):
    """include restricts by pattern; exclude on a directory name skips the
    directory (and everything under it) entirely rather than just its
    top-level entry."""
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    (src / 'keep.txt').write_bytes(b'KEEP')
    (src / 'keep.log').write_bytes(b'LOG')
    marker_dir = src / '__marker__'
    marker_dir.mkdir()
    (marker_dir / 'inner.txt').write_bytes(b'INNER')

    fsutil.copy(str(src), str(dst), include=['*.txt'], exclude=['__marker__'])

    assert (dst / 'keep.txt').read_bytes() == b'KEEP'
    assert not (dst / 'keep.log').exists()
    assert not (dst / '__marker__').exists()


def test_copy_directory_copy_file_failure_propagates(
        monkeypatch: pytest.MonkeyPatch):
    """A single copy_file failure among many concurrent transfers fails the
    whole copy call."""
    from io_utils.fsutil import generic

    source = '/customer/source2'
    destination = '/kumo/destination2'

    listings = {
        source: [
            generic.api.PathDetail(path=f'{source}/a.txt', is_directory=False),
            generic.api.PathDetail(path=f'{source}/b.txt', is_directory=False),
        ],
    }

    def isdir(path):
        return path.rstrip('/') == source

    def listdir(path, *, detail=False, **_kwargs):
        assert detail
        return listings[path]

    def copy_file(src, dst, overwrite=False, raise_on_exists=True):
        if src.endswith('a.txt'):
            raise RuntimeError('transfer boom')

    monkeypatch.setattr(generic.api, 'isdir', isdir)
    monkeypatch.setattr(generic.api, 'exists', lambda _path: True)
    monkeypatch.setattr(generic.api, 'makedirs', lambda *_a, **_k: None)
    monkeypatch.setattr(generic.api, 'listdir', listdir)
    monkeypatch.setattr(generic, 'copy_file', copy_file)

    with pytest.raises(RuntimeError, match='transfer boom'):
        fsutil.copy(source, destination)


def test_expanduser():
    path = '~'
    expanded_path = fsutil.expanduser(path)
    assert len(expanded_path) > 1

    s3_path = 's3://~'
    expanded_s3_path = fsutil.expanduser(s3_path)
    assert expanded_s3_path == s3_path


def test_abspath():
    path = '.'
    expanded_path = fsutil.abspath(path)
    assert expanded_path != '.'

    s3_path = 's3://~'
    expanded_s3_path = fsutil.expanduser(s3_path)
    assert expanded_s3_path == s3_path


def test_load_not_exist(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError):
            fsutil.torch_load("does_not_exist")

        assert "does_not_exist" in caplog.records[0].message


def test_copy_retry_basic(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError):
            fsutil.copy_file("/random/src", "random/dest")

        assert caplog.records[0].message == (
            'Source file copy failed (error_type=FileNotFoundError)')
        assert '/random/src' not in caplog.text


def test_copy_provider_exception_is_sanitized(caplog, tmp_path):
    copy_poison = 'signed-url-and-temporary-token-must-not-escape'
    cleanup_poison = 'cleanup-provider-secret-must-not-escape'
    destination = str(tmp_path / 'downloaded')

    with patch(
            'io_utils.fsutil.s3_filesystem.download_file',
            side_effect=RuntimeError(copy_poison),
    ), patch(
            'io_utils.fsutil.generic.api.rm',
            side_effect=RuntimeError(cleanup_poison),
    ), caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match='^File copy failed$') as error:
            fsutil.copy_file('s3://customer-bucket/events.parquet', destination)

    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    rendered = ''.join(
        traceback.format_exception(type(error.value), error.value,
                                   error.value.__traceback__))
    for poison in (copy_poison, cleanup_poison):
        assert poison not in rendered
        assert poison not in repr(error.value)
        assert poison not in caplog.text


@pytest.mark.parametrize('failure_stage',
                         ['initialize', 'subdirectory', 'worker'])
def test_two_level_directory_copy_provider_error_is_sanitized(
        caplog, failure_stage):
    poison = f'{failure_stage}-directory-provider-secret-must-not-escape'
    source = '/customer/source'
    destination = '/kumo/destination'
    source_entry = f'{source}/nested'

    class FakeResult:
        def get(self):
            if failure_stage == 'worker':
                raise RuntimeError(poison)

    class FakePool:
        def __init__(self, **_kwargs):
            if failure_stage == 'initialize':
                raise RuntimeError(poison)

        def apply_async(self, *_args, **_kwargs):
            return FakeResult()

        def close(self):
            pass

        def join(self):
            pass

        def terminate(self):
            pass

    def isdir(path):
        if path.rstrip('/') == source:
            return True
        return failure_stage == 'subdirectory' and path == source_entry

    def listdir(path):
        if path.rstrip('/') == source:
            return [source_entry]
        if failure_stage == 'subdirectory':
            raise RuntimeError(poison)
        return []

    with patch('io_utils.fsutil.generic.Pool', FakePool), patch(
            'io_utils.fsutil.generic.api.makedirs'), patch(
                'io_utils.fsutil.generic.api.isdir', side_effect=isdir), patch(
                    'io_utils.fsutil.generic.api.listdir', side_effect=listdir), \
            caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError,
                           match='^Directory copy failed$') as error:
            fsutil.copy_two_level_directory(source, destination)

    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    rendered = ''.join(
        traceback.format_exception(type(error.value), error.value,
                                   error.value.__traceback__))
    assert poison not in rendered
    assert poison not in repr(error.value)
    assert poison not in caplog.text


def test_write_binary_retry_rewinds_source(tmp_path: pathlib.Path):
    # Regression for https://github.com/kumo-ai/kumo/issues/28588
    # A transient BlockingIOError partway through write_binary previously
    # left the source BinaryIO advanced; the retry would then truncate the
    # destination and write only the unread tail of the source, silently
    # producing a corrupt file. The fix captures data.tell() before retry
    # and seeks back at each attempt.
    class FlakyReader(io.BytesIO):
        """BytesIO that raises BlockingIOError once partway through, then
        succeeds. Mimics a transient EAGAIN from the destination FS."""
        def __init__(self, data: bytes, fail_after: int):
            super().__init__(data)
            self._fail_after = fail_after
            self._failed_once = False

        def read(self, size: Optional[int] = -1) -> bytes:
            if not self._failed_once and self.tell() >= self._fail_after:
                self._failed_once = True
                raise BlockingIOError(11, 'Resource temporarily unavailable')
            return super().read(size)

    src_bytes = bytes(range(256)) * 4096  # 1 MiB binary payload
    flaky = FlakyReader(src_bytes, fail_after=200_000)

    dst = tmp_path / 'val_output.pt'
    fsutil.write_binary(str(dst), flaky)

    with open(dst, 'rb') as f:
        written = f.read()
    assert len(written) == len(src_bytes)
    assert written == src_bytes


def test_get_many_if_present(tmp_path):
    # Create temporary files
    file1 = tmp_path / "file1"
    file2 = tmp_path / "file2"
    file1.write_text("content1")
    file2.write_text("content2")
    file_1_path = str(file1)
    file_2_path = str(file2)

    # Test with existing files
    result1 = fsutil.get_many_if_present([file_1_path, file_2_path])
    assert result1 == [b"content1", b"content2"]

    # Test with a mix of existing and non-existing files
    non_existent = tmp_path / "non_existent"
    non_existent_path = str(non_existent)
    result2 = fsutil.get_many_if_present(
        [file_1_path, non_existent_path, file_2_path])
    assert result2 == [b"content1", None, b"content2"]

    # Test with None values
    result3 = fsutil.get_many_if_present([None, file_1_path, None, file_2_path])
    assert result3 == [None, b"content1", None, b"content2"]


def test_copy_one_level_directory(tmp_path):
    temp_dir = osp.join(tmp_path, 'test_fsutil_copy')
    src_dir = osp.join(temp_dir, 'one_level_src')
    dst_dir = osp.join(temp_dir, 'one_level_dst')

    fsutil.makedirs(src_dir)
    # create only top-level files (no subdirectories for this API)
    fsutil.touch(osp.join(src_dir, 'a.txt'))
    fsutil.touch(osp.join(src_dir, 'b.log'))
    fsutil.touch(osp.join(src_dir, '__SUCCESS__'))

    # Basic copy with exclude to skip markers
    fsutil.copy_one_level_directory(src_dir, dst_dir, exclude=['__SUCCESS__'])
    assert fsutil.exists(osp.join(dst_dir, 'a.txt'))
    assert fsutil.exists(osp.join(dst_dir, 'b.log'))
    assert not fsutil.exists(osp.join(dst_dir, '__SUCCESS__'))

    # Running again with raise_on_exists=True should error because dst exists
    with pytest.raises(FileExistsError):
        fsutil.copy_one_level_directory(src_dir, dst_dir,
                                        exclude=['__SUCCESS__'])

    # include filter should allow copying only specific pattern to a new dst
    dst_dir_2 = osp.join(temp_dir, 'one_level_dst_2')
    fsutil.copy_one_level_directory(src_dir, dst_dir_2, include=['*.txt'])
    assert fsutil.exists(osp.join(dst_dir_2, 'a.txt'))
    assert not fsutil.exists(osp.join(dst_dir_2, 'b.log'))
    assert not fsutil.exists(osp.join(dst_dir_2, '__SUCCESS__'))

    # Add a subdirectory to the source dir; this should fail.
    fsutil.makedirs(osp.join(src_dir, 'subdir'))
    dst_dir_3 = osp.join(temp_dir, 'one_level_dst_3')
    with pytest.raises(RuntimeError):
        fsutil.copy_one_level_directory(src_dir, dst_dir_3)


def test_copy_two_level_directory():
    temp_dir = osp.join(get_home_dir(), 'test_fsutil_copy')
    src_dir = osp.join(temp_dir, 'two_level_src')
    dst_dir = osp.join(temp_dir, 'two_level_dst')

    # root files
    fsutil.makedirs(src_dir)
    fsutil.touch(osp.join(src_dir, 'root_a.txt'))
    fsutil.touch(osp.join(src_dir, '__SUCCESS__'))

    # first-level subdir with files
    sub1 = osp.join(src_dir, 'sub1')
    fsutil.makedirs(sub1)
    fsutil.touch(osp.join(sub1, 's1_a.txt'))
    fsutil.touch(osp.join(sub1, '__SUCCESS__'))

    # second first-level subdir with deeper nested folder (third level)
    sub2 = osp.join(src_dir, 'sub2')
    fsutil.makedirs(sub2)
    fsutil.touch(osp.join(sub2, 's2_a.txt'))

    # Copy, excluding marker files; third-level content should not be traversed
    fsutil.copy_two_level_directory(src_dir, dst_dir, exclude=['__SUCCESS__'])

    # root files copied
    assert fsutil.exists(osp.join(dst_dir, 'root_a.txt'))
    assert not fsutil.exists(osp.join(dst_dir, '__SUCCESS__'))

    # first-level subdir files copied, exclude respected
    assert fsutil.exists(osp.join(dst_dir, 'sub1', 's1_a.txt'))
    assert not fsutil.exists(osp.join(dst_dir, 'sub1', '__SUCCESS__'))

    # second subdir first-level files copied
    assert fsutil.exists(osp.join(dst_dir, 'sub2', 's2_a.txt'))

    # copying again with raise_on_exists=False should succeed (skip existing)
    fsutil.copy_two_level_directory(src_dir, dst_dir, raise_on_exists=False)

    deep = osp.join(sub2, 'deep3')
    fsutil.makedirs(deep)
    fsutil.touch(osp.join(deep, 'deep_file.txt'))
    # copy fails with third-level directory
    with pytest.raises(RuntimeError):
        fsutil.copy_two_level_directory(src_dir, dst_dir, raise_on_exists=False)


@pytest.mark.parametrize('provider', ['s3', 'gcs', 'adls'])
@pytest.mark.parametrize('levels', [1, 2])
def test_scoped_directory_copy_uses_exact_credentials_in_threads(
    monkeypatch,
    provider: str,
    levels: int,
) -> None:
    from io_utils.fsutil import generic

    source = '/local/source'
    destination = f'{provider}://customer/destination'
    nested = f'{source}/model'
    files = ([f'{source}/config.pbtxt']
             if levels == 1 else [f'{nested}/weights.bin'])
    credentials = {
        'provider': provider,
        'marker': f'{provider}-customer-credentials',
    }
    copied = []

    def isdir(path):
        return path.rstrip('/') in {source, nested}

    def listdir(path):
        if path.rstrip('/') == source:
            return files if levels == 1 else [nested]
        if path == nested:
            return files
        raise AssertionError(f'unexpected list path: {path}')

    def copy_file(src, dst, overwrite, raise_on_exists):
        scope = credentials_for_provider(provider)
        assert scope is not None
        assert scope.credentials == credentials
        copied.append((src, dst, overwrite, raise_on_exists))

    monkeypatch.setattr(generic.api, 'isdir', isdir)
    monkeypatch.setattr(generic.api, 'listdir', listdir)
    monkeypatch.setattr(generic.api, 'makedirs', lambda *_a, **_k: None)
    monkeypatch.setattr(generic, 'copy_file', copy_file)
    monkeypatch.setattr(
        generic,
        'Pool',
        lambda *_a, **_k: pytest.fail(
            'credential-scoped copy must not create a process pool'),
    )

    with credential_scope(credentials, require_explicit=True):
        if levels == 1:
            fsutil.copy_one_level_directory(source, destination)
        else:
            fsutil.copy_two_level_directory(source, destination)

    assert len(copied) == 1
    assert credentials_for_provider(provider) is None


def test_copy_files_threaded_is_concurrent_and_bounded(monkeypatch) -> None:
    from io_utils.fsutil import generic

    active = 0
    peak = 0
    lock = threading.Lock()
    two_started = threading.Event()
    release = threading.Event()
    copied = []

    def copy_file(src, dst, overwrite, raise_on_exists):
        nonlocal active, peak
        scope = credentials_for_provider('s3')
        assert scope is not None
        assert scope.credentials == {'provider': 's3', 'marker': 'customer'}
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
        try:
            assert release.wait(5)
            copied.append((src, dst, overwrite, raise_on_exists))
        finally:
            with lock:
                active -= 1

    def release_when_concurrent() -> None:
        assert two_started.wait(5)
        with lock:
            assert active == 2
        release.set()

    monkeypatch.setattr(generic, 'copy_file', copy_file)
    controller = threading.Thread(target=release_when_concurrent)
    controller.start()
    credentials = {'provider': 's3', 'marker': 'customer'}
    with credential_scope(credentials, require_explicit=True):
        fsutil.copy_files_threaded(
            [(f's3://internal/part-{index}',
              f's3://customer/part-{index}') for index in range(5)],
            overwrite=True,
            max_workers=2,
        )
    controller.join(timeout=5)

    assert not controller.is_alive()
    assert peak == 2
    assert len(copied) == 5
    assert all(overwrite and raise_on_exists
               for _src, _dst, overwrite, raise_on_exists in copied)


def test_copy_tree_threaded_exclude_predicate(tmp_path: pathlib.Path):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    (src / 'manifest.json').write_bytes(b'top-manifest')
    (src / 'a.parquet').write_bytes(b'a')
    (src / 'nested').mkdir()
    (src / 'nested' / 'manifest.json').write_bytes(b'nested-manifest')
    (src / 'nested' / 'b.parquet').write_bytes(b'b')

    # Skip every file named 'manifest.json' at any depth.
    fsutil.copy_tree_threaded(
        str(src),
        str(dst),
        exclude=lambda p: osp.basename(p) == 'manifest.json',
    )

    assert not (dst / 'manifest.json').exists()
    assert not (dst / 'nested' / 'manifest.json').exists()
    assert (dst / 'a.parquet').read_bytes() == b'a'
    assert (dst / 'nested' / 'b.parquet').read_bytes() == b'b'


def test_copy_tree_threaded_exclude_top_level_only(tmp_path: pathlib.Path):
    # Demonstrates the predicate's flexibility — caller can scope by depth
    # using the absolute path.
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    (src / 'skip_me.txt').write_bytes(b'top-skip')
    (src / 'sub').mkdir()
    (src / 'sub' / 'skip_me.txt').write_bytes(b'nested-keep')

    src_root = str(src).rstrip('/')

    def _exclude(p: str) -> bool:
        # Only skip the top-level 'skip_me.txt'; keep the nested one.
        return osp.dirname(p) == src_root and osp.basename(p) == 'skip_me.txt'

    fsutil.copy_tree_threaded(str(src), str(dst), exclude=_exclude)

    assert not (dst / 'skip_me.txt').exists()
    assert (dst / 'sub' / 'skip_me.txt').read_bytes() == b'nested-keep'


def test_copy_tree_threaded_empty_src_is_noop(tmp_path: pathlib.Path):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()

    fsutil.copy_tree_threaded(str(src), str(dst))

    # No transfers issued; dst is never created since copy_tree_threaded
    # returns early when there is nothing to copy.
    assert not dst.exists()


def test_copy_tree_threaded_same_src_dst_is_noop(tmp_path: pathlib.Path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_bytes(b'a')

    # Copying onto itself must not corrupt or wipe contents.
    fsutil.copy_tree_threaded(str(src), str(src))

    assert (src / 'a.txt').read_bytes() == b'a'


@pytest.fixture
def s3_scratch_prefix():
    """Unique S3 prefix per test, deleted after yield. Skips if no
    AWS creds — hand-rolled to keep this fixture independent of REST app
    initialization.
    """
    import uuid
    if not fsutil.exists('s3://kumo-public-datasets/financial/csv/'):
        pytest.skip('No AWS credentials found for S3')
    prefix = ('s3://kumo-test-dataplane-bucket/fsutil-test/'
              f'copy_tree_threaded_{uuid.uuid4().hex[:12]}')
    yield prefix
    if fsutil.exists(prefix):
        fsutil.rm(prefix)


def test_copy_tree_threaded_local_to_s3(tmp_path: pathlib.Path,
                                        s3_scratch_prefix: str):
    """Local→S3 upload: tree preservation + predicate exclusion against
    the real S3 transport. (The local→local parent-dir bug was invisible
    to S3, so cross-backend coverage is load-bearing.)"""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'top.txt').write_bytes(b'TOP')
    (src / 'manifest.json').write_bytes(b'COMMIT')
    (src / 'sub1').mkdir()
    (src / 'sub1' / 'a.parquet').write_bytes(b'A')
    deep = src / 'sub2' / 'deep'
    deep.mkdir(parents=True)
    (deep / 'leaf.txt').write_bytes(b'LEAF')

    fsutil.copy_tree_threaded(
        str(src),
        s3_scratch_prefix,
        exclude=lambda p: osp.basename(p) == 'manifest.json',
    )

    assert fsutil.read_binary(osp.join(s3_scratch_prefix, 'top.txt')) == b'TOP'
    assert fsutil.read_binary(osp.join(s3_scratch_prefix, 'sub1',
                                       'a.parquet')) == b'A'
    assert fsutil.read_binary(
        osp.join(s3_scratch_prefix, 'sub2', 'deep', 'leaf.txt')) == b'LEAF'
    assert not fsutil.exists(osp.join(s3_scratch_prefix, 'manifest.json'))


def test_copy_tree_threaded_surfaces_failure(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_bytes(b'a')
    (src / 'b.txt').write_bytes(b'b')

    from io_utils.fsutil import generic as fsutil_generic

    real_copy_file = fsutil_generic.copy_file
    call_count = {'n': 0}

    def _failing_copy_file(src_path, dst_path, overwrite=False, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise RuntimeError('synthetic transfer failure')
        return real_copy_file(src_path, dst_path, overwrite=overwrite, **kwargs)

    monkeypatch.setattr(fsutil_generic, 'copy_file', _failing_copy_file)

    with pytest.raises(RuntimeError, match='synthetic transfer failure'):
        fsutil.copy_tree_threaded(str(src), str(tmp_path / 'dst'))


def test_copy_tree_threaded_remote_src_delegates_to_copy(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Every remote source -- regardless of destination locality -- now
    delegates the whole tree to copy(), which does its own parallel
    listing and transfer; copy_tree_threaded keeps no enumeration
    machinery of its own for remote sources."""
    from io_utils.fsutil import generic as fsutil_generic

    monkeypatch.setattr(fsutil_generic, 'is_local', lambda p: False)

    copy_calls = []

    def _spy_copy(src_path, dst_path, **kwargs):
        copy_calls.append((src_path, dst_path, kwargs))

    monkeypatch.setattr(fsutil_generic, 'copy', _spy_copy)

    # os.walk must NOT run for a remote source.
    def _boom_walk(*_a, **_k):
        raise AssertionError('os.walk should not be called for a remote source')

    monkeypatch.setattr(fsutil_generic.os, 'walk', _boom_walk)

    fsutil.copy_tree_threaded('s3://bucket/staging', 'gcs://other-bucket/dst')

    assert len(copy_calls) == 1
    src_path, dst_path, kwargs = copy_calls[0]
    assert src_path == 's3://bucket/staging'
    assert dst_path == 'gcs://other-bucket/dst'
    # Mirror copy_tree_threaded's overwrite semantics; no callable exclude.
    assert kwargs == dict(exclude=None, include=None, overwrite=True,
                          raise_on_exists=False)


def test_copy_tree_threaded_remote_src_to_local_dst_also_delegates(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A remote source with a *local* destination used to take a second,
    bespoke enumerate-then-transfer code path inside copy_tree_threaded
    itself; that path is gone, so this now delegates too, the same as a
    remote-to-remote tree."""
    from io_utils.fsutil import generic as fsutil_generic

    source = 's3://bucket/staging'
    destination = str(tmp_path / 'dst')

    monkeypatch.setattr(fsutil_generic, 'is_local',
                        lambda path: path == destination)

    copy_calls = []
    monkeypatch.setattr(
        fsutil_generic, 'copy', lambda *args, **kwargs: copy_calls.append(
            (args, kwargs)))

    fsutil.copy_tree_threaded(source, destination, max_workers=2)

    assert copy_calls == [
        ((source, destination),
         dict(exclude=None, include=None, overwrite=True,
              raise_on_exists=False)),
    ]


def test_copy_enumerates_full_remote_tree_concurrently(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The multi-level, multi-branch concurrent enumeration this used to
    unit-test on copy_tree_threaded now lives entirely in copy() (via
    walk_directory_tree), since copy_tree_threaded just delegates for a
    remote source. Every directory is enumerated and every file copied,
    even though listings run concurrently."""
    from io_utils.fsutil import generic as fsutil_generic

    source = 's3://bucket/root'
    destination = str(tmp_path / 'dst')
    pathlib.Path(destination).mkdir()

    dir_a = f'{source}/a'
    dir_b = f'{source}/b'
    dir_a_1 = f'{dir_a}/a1'

    listings = {
        source: [
            fsutil_generic.api.PathDetail(path=dir_a, is_directory=True),
            fsutil_generic.api.PathDetail(path=dir_b, is_directory=True),
            fsutil_generic.api.PathDetail(path=f'{source}/root.txt',
                                          is_directory=False),
        ],
        dir_a: [
            fsutil_generic.api.PathDetail(path=dir_a_1, is_directory=True),
            fsutil_generic.api.PathDetail(path=f'{dir_a}/a.txt',
                                          is_directory=False),
        ],
        dir_b: [
            fsutil_generic.api.PathDetail(path=f'{dir_b}/b.txt',
                                          is_directory=False),
        ],
        dir_a_1: [
            fsutil_generic.api.PathDetail(path=f'{dir_a_1}/a1.txt',
                                          is_directory=False),
        ],
    }
    list_calls = []
    copy_calls = []
    makedirs_calls = []

    def isdir(path):
        return path.rstrip('/') == source

    def listdir(path, *, detail=False, **_kwargs):
        assert detail
        list_calls.append(path)
        return listings[path]

    def makedirs(path, exist_ok=False):
        assert exist_ok
        makedirs_calls.append(path)

    monkeypatch.setattr(fsutil_generic.api, 'isdir', isdir)
    monkeypatch.setattr(fsutil_generic.api, 'listdir', listdir)
    monkeypatch.setattr(fsutil_generic.api, 'makedirs', makedirs)
    monkeypatch.setattr(fsutil_generic, 'copy_file',
                        lambda *args: copy_calls.append(args))

    fsutil.copy(source, destination, overwrite=True, raise_on_exists=False)

    assert set(list_calls) == {source, dir_a, dir_b, dir_a_1}
    assert set(makedirs_calls) == {
        destination,
        osp.join(destination, 'a'),
        osp.join(destination, 'b'),
        osp.join(destination, 'a', 'a1'),
    }
    assert set(copy_calls) == {
        (f'{source}/root.txt', osp.join(destination, 'root.txt'), True,
         False),
        (f'{dir_a}/a.txt', osp.join(destination, 'a', 'a.txt'), True, False),
        (f'{dir_b}/b.txt', osp.join(destination, 'b', 'b.txt'), True, False),
        (f'{dir_a_1}/a1.txt', osp.join(destination, 'a', 'a1', 'a1.txt'),
         True, False),
    }


def test_copy_remote_listing_failure_propagates(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A failure enumerating one subdirectory fails the whole copy rather
    than silently yielding a truncated transfer list. Was previously
    exercised through copy_tree_threaded's own (now-removed) remote
    enumeration path; the behavior lives in copy() now."""
    from io_utils.fsutil import generic as fsutil_generic

    source = 's3://bucket/root'
    destination = str(tmp_path / 'dst')
    pathlib.Path(destination).mkdir()
    dir_a = f'{source}/a'
    dir_b = f'{source}/b'

    listings = {
        source: [
            fsutil_generic.api.PathDetail(path=dir_a, is_directory=True),
            fsutil_generic.api.PathDetail(path=dir_b, is_directory=True),
        ],
        dir_a: [
            fsutil_generic.api.PathDetail(path=f'{dir_a}/a.txt',
                                          is_directory=False),
        ],
    }
    copy_calls = []

    def isdir(path):
        return path.rstrip('/') == source

    def listdir(path, *, detail=False, **_kwargs):
        assert detail
        if path == dir_b:
            raise RuntimeError('listing boom')
        return listings[path]

    monkeypatch.setattr(fsutil_generic.api, 'isdir', isdir)
    monkeypatch.setattr(fsutil_generic.api, 'listdir', listdir)
    monkeypatch.setattr(fsutil_generic, 'copy_file',
                        lambda *args: copy_calls.append(args))

    with pytest.raises(RuntimeError, match='listing boom'):
        fsutil.copy(source, destination, overwrite=True, raise_on_exists=False)

    # The failure surfaces before enumeration finishes, so no transfer
    # ever gets a chance to run -- a truncated *set* of copies would be
    # just as wrong as a truncated list.
    assert copy_calls == []


def test_copy_tree_threaded_remote_src_falls_back_without_detailed_listing(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Listings without entry types retain compatibility via explicit probes. Goes
    through copy_tree_threaded's delegate to copy(), which now owns the
    isdir/exists checks that decide file-vs-directory before any listing
    happens -- unlike copy_tree_threaded's old bespoke remote path, which
    never probed the root itself."""
    from io_utils.fsutil import generic as fsutil_generic

    source = '@db.schema.stage/bundle'
    source_file = f'{source}/tt.parquet'
    destination = str(tmp_path / 'dst')
    copy_calls = []

    monkeypatch.setattr(fsutil_generic, 'is_local',
                        lambda path: path == destination)

    def listdir(path, *, detail=False, **_kwargs):
        assert path == source
        if detail:
            raise NotImplementedError
        return [source_file]

    def isdir(path):
        return path.rstrip('/') == source

    monkeypatch.setattr(fsutil_generic.api, 'listdir', listdir)
    monkeypatch.setattr(fsutil_generic.api, 'isdir', isdir)
    monkeypatch.setattr(
        fsutil_generic,
        'copy_file',
        lambda *args: copy_calls.append(args),
    )

    fsutil.copy_tree_threaded(source, destination, max_workers=1)

    assert copy_calls == [(source_file, osp.join(destination,
                                                 'tt.parquet'), True, False)]


def test_copy_tree_threaded_remote_src_copies_whole_bundle(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A source treated as remote copies the entire bundle (nested subdir +
    spec sibling) end to end via the delegated copy() path, where the old code
    copied nothing. Forces is_local False; copy() works local->local."""
    from io_utils.fsutil import generic as fsutil_generic

    src = tmp_path / 'staging'
    dst = tmp_path / 'pod_local_staging'
    src.mkdir()
    (src / 'tt.parquet').write_bytes(b'TT')
    (src / 'tt_ingest_spec.json').write_bytes(b'{"spec": true}')
    (src / 'train_seeds').mkdir()
    (src / 'train_seeds' / 'part-0.parquet').write_bytes(b'SEED0')
    (src / 'train_seeds' / 'part-1.parquet').write_bytes(b'SEED1')

    monkeypatch.setattr(fsutil_generic, 'is_local', lambda p: False)

    fsutil.copy_tree_threaded(str(src), str(dst))

    assert (dst / 'tt.parquet').read_bytes() == b'TT'
    assert (dst / 'tt_ingest_spec.json').read_bytes() == b'{"spec": true}'
    assert (dst / 'train_seeds' / 'part-0.parquet').read_bytes() == b'SEED0'
    assert (dst / 'train_seeds' / 'part-1.parquet').read_bytes() == b'SEED1'


def test_copy_accepts_enable_multi_process_from_core(tmp_path):
    """``kumo_core.explain.subgraph_collector`` passes this keyword.

    ``fsutil`` is a cross-repo API, so a signature change here breaks a repo
    this one's greps do not cover; that is how it broke once already.
    """
    src = tmp_path / 'src'
    (src / 'nested').mkdir(parents=True)
    (src / 'nested' / 'f.txt').write_text('payload')
    dst = tmp_path / 'dst'

    fsutil.copy(str(src), str(dst), overwrite=True, enable_multi_process=True)

    assert (dst / 'nested' / 'f.txt').read_text() == 'payload'


def test_copy_accepts_pool_and_enable_multi_process(tmp_path):
    """``pool`` is accepted and ignored for the same cross-repo-API reason as
    ``enable_multi_process`` above: no caller passing it was found by
    grepping this repo or the installed site-packages, but that grep cannot
    see every out-of-repo caller, and accepting-and-ignoring is safe -- the
    copy still happens, just via threads rather than the caller's pool.
    """
    src = tmp_path / 'src'
    (src / 'nested').mkdir(parents=True)
    (src / 'nested' / 'f.txt').write_text('payload')
    dst = tmp_path / 'dst'

    fsutil.copy(str(src), str(dst), overwrite=True, enable_multi_process=True,
                pool=object())

    assert (dst / 'nested' / 'f.txt').read_text() == 'payload'
