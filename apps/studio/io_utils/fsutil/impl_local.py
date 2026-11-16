# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import os.path as osp
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Optional, TextIO, Tuple
from urllib.parse import urlencode

import pandas.io.common
from smart_open import open as smart_open

from io_utils import logging
from io_utils.fsutil import api
from io_utils.fsutil.api import PartMetadata, PresignedHelper
from io_utils.fsutil.common import (
    DEFAULT_DATA_VERSION,
    fsExceptionHandler,
    should_include,
)
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.fsutil.uri import LocalURI
from io_utils.retry import retry
from monitoring.tracing import TraceOptions, trace

logger = logging.getLogger(__name__)

local_fs_logging_policy = logging_policy(logger)


def local_fs_retry_policy():
    ex_types = (RuntimeError, BlockingIOError)

    return retry(
        exceptions=ex_types,
        tries=3,
        delay=1,
        backoff=2,
        logger=logger,
    )


@api.filesize.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@local_fs_retry_policy()
@local_fs_logging_policy()
def filesize(uri: LocalURI) -> Optional[int]:
    if isdir(uri):
        return None
    try:
        return osp.getsize(uri.path)
    except FileNotFoundError:
        return None


@api.object_identity.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@local_fs_retry_policy()
@local_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def object_identity(uri: LocalURI) -> api.ObjectIdentity:
    """Capture a stable identity for one closed local regular file."""
    metadata = os.lstat(uri.path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(uri.path)
    return api.ObjectIdentity(
        size=metadata.st_size,
        etag=None,
        version=(f'{metadata.st_dev}:{metadata.st_ino}:'
                 f'{metadata.st_mtime_ns}:{metadata.st_ctime_ns}'),
    )


@api.read_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@local_fs_retry_policy()
@local_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_binary(uri: LocalURI, compression: bool) -> bytes:
    if compression:
        with gzip.open(uri.path, 'rb') as f:
            return f.read()
    else:
        with open(uri.path, 'rb') as f:
            return f.read()


@api.read_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@local_fs_retry_policy()
@local_fs_logging_policy(allowed_exceptions=FileNotFoundError)
def read_text(uri: LocalURI, compression: bool,
              encoding: Optional[str] = None) -> str:
    if compression:
        with gzip.open(uri.path, 'rt', encoding=encoding) as f:
            return f.read()
    else:
        with open(uri.path, 'rt', encoding=encoding) as f:
            return f.read()


@api._write_binary.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression']))
@local_fs_logging_policy(skip_args=[1])
def write_binary(uri: LocalURI, data: BinaryIO, compression: bool,
                 **kwargs) -> None:
    # Capture source position once, BEFORE any retry, so each attempt
    # restarts copying from the same offset. Otherwise a transient EAGAIN
    # mid-write leaves `data` advanced; the retry's 'wb' open truncates
    # the destination and writes only the unread tail of `data`, silently
    # corrupting the file.
    start_pos = data.tell() if data.seekable() else None
    _do_write_binary(uri, data, compression, start_pos)


@local_fs_retry_policy()
def _do_write_binary(uri: LocalURI, data: BinaryIO, compression: bool,
                     start_pos: Optional[int]) -> None:
    if start_pos is not None:
        data.seek(start_pos)
    if compression:
        with gzip.open(uri.path, 'wb') as f:
            shutil.copyfileobj(data, f)
    else:
        with open(uri.path, 'wb') as f:
            shutil.copyfileobj(data, f)


@api.write_text.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri', 'compression', 'encoding']))
@local_fs_retry_policy()
@local_fs_logging_policy(skip_args=[1])
def write_text(uri: LocalURI, data: str, compression: bool,
               encoding: Optional[str] = None) -> None:
    if compression:
        with gzip.open(uri.path, 'wt', encoding=encoding) as f:
            f.write(data)
    else:
        with open(uri.path, 'wt', encoding=encoding) as f:
            f.write(data)


@api.write_text_if_absent.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=['uri']))
@local_fs_logging_policy(skip_args=[1])
def write_text_if_absent(uri: LocalURI, data: str) -> bool:
    parent = osp.dirname(uri.path) or '.'
    descriptor, temp_path = tempfile.mkstemp(
        prefix=f'.{osp.basename(uri.path)}.', suffix='.tmp', dir=parent)
    published = False
    try:
        with os.fdopen(descriptor, 'w') as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard link publishes the complete temporary file
            # without replacing an existing claimant.
            os.link(temp_path, uri.path)
        except FileExistsError:
            return False
        published = True
        directory_fd = os.open(parent,
                               os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # Cleanup is idempotent if the temporary name is already gone.
            pass
        if published:
            # Persist removal of the temporary hard-link name as well.
            directory_fd = os.open(parent,
                                   os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


@api._get_binary_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_binary_handle(
    uri: LocalURI,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    with smart_open(
            uri.path, mode, encoding=None,
            compression=".gz" if compression else None) as f:  # type: ignore
        yield f  # type: ignore


@api._get_handle.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
def get_handle(
    uri: LocalURI,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    # We currently rely on `pandas` for this:
    with pandas.io.common.get_handle(
            uri.path, mode, encoding=None,
            compression="gzip" if compression else None) as out:
        yield out.handle if hasattr(out, 'handle') else out[0]  # type: ignore


@api.dirsize.register
@fsExceptionHandler
@trace(TraceOptions(trace_params=True))
@local_fs_retry_policy()
@local_fs_logging_policy()
def dirsize(uri: LocalURI) -> int:
    def _recursive_scandir(path):
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                yield from _recursive_scandir(entry.path)
            else:
                yield entry

    return sum(d.stat().st_size for d in _recursive_scandir(uri.path)
               if d.is_file())


@api.exists.register
@fsExceptionHandler
@local_fs_retry_policy()
@local_fs_logging_policy()
def exists(uri: LocalURI) -> bool:
    return osp.exists(uri.path)


@api.isdir.register
@fsExceptionHandler
@local_fs_retry_policy()
@local_fs_logging_policy()
def isdir(uri: LocalURI) -> bool:
    return osp.isdir(uri.path)


@api._get_if_present.register
@local_fs_logging_policy()
@local_fs_retry_policy()
def get_if_present(uri: LocalURI) -> Optional[bytes]:
    if not osp.exists(uri.path):
        return None
    with open(uri.path, 'rb') as f:
        return f.read()


@api._listdir.register
@fsExceptionHandler
@local_fs_retry_policy()
@local_fs_logging_policy()
def listdir(
    uri: LocalURI,
    max_items: Optional[int] = None,
) -> list[str]:
    return list(
        sorted([osp.join(uri.path, entry) for entry in os.listdir(uri.path)]))


@api._listdir_detail.register
def listdir_detail(
    uri: LocalURI,
    max_items: Optional[int] = None,
) -> list[api.PathDetail]:
    raise NotImplementedError("listdir_detail is not implemented for LocalURI")


@api.makedirs.register
@fsExceptionHandler
@local_fs_retry_policy()
@local_fs_logging_policy()
def makedirs(uri: LocalURI, exist_ok: bool = False) -> None:
    os.makedirs(uri.path, exist_ok=exist_ok)


@api.rm.register
@local_fs_retry_policy()
@local_fs_logging_policy()
def rm(uri: LocalURI, ignore_errors: bool = False,
       include: Optional[List[str]] = None) -> None:
    try:
        if isdir(uri):
            if should_include(uri.path, include):
                shutil.rmtree(uri.path, ignore_errors=ignore_errors)
            else:
                for entry in listdir(uri):
                    rm(LocalURI(path=entry), ignore_errors, include)
        elif exists(uri) and should_include(uri.path, include):
            os.remove(uri.path)
    except Exception:
        if not ignore_errors:
            raise


@api.touch.register
@fsExceptionHandler
@local_fs_retry_policy()
@local_fs_logging_policy()
def touch(uri: LocalURI) -> None:
    Path(uri.path).touch()


@api.has_write_permission.register
@fsExceptionHandler
@local_fs_logging_policy()
def has_write_permission(uri: LocalURI) -> bool:
    dirname = osp.dirname(uri.path)
    if osp.exists(dirname):
        return os.access(dirname, os.W_OK | os.X_OK)
    if dirname == '/':
        # Should never happen, but added to prevent infinite recursion.
        raise RuntimeError("Root path does not exist")
    p = Path(dirname)
    return has_write_permission(LocalURI(path=str(p.parent.absolute())))


@api._contains_subdir.register
@local_fs_retry_policy()
@local_fs_retry_policy()
@local_fs_logging_policy()
def contains_subdir(uri: LocalURI) -> bool:
    """Check if a path contains sub directories."""
    for f in listdir(uri):
        if isdir(LocalURI(path=f)):
            return True
    return False


@api.get_last_modified_time.register
@local_fs_retry_policy()
@local_fs_retry_policy()
@local_fs_logging_policy()
def get_last_modified_time(uri: LocalURI,
                           suffix: Optional[str] = None) -> float:
    if not osp.exists(uri.path):
        return DEFAULT_DATA_VERSION
    if osp.isdir(uri.path):
        files = [
            file for file in os.listdir(uri.path)
            if not osp.isdir(file) and (not suffix or file.endswith(suffix))
        ]
        if len(files) == 0:
            return DEFAULT_DATA_VERSION
        return max(osp.getmtime(osp.join(uri.path, file)) for file in files)

    return osp.getmtime(uri.path)


def local_upload_part_path(temp_upload_path: str, upload_id: str,
                           part_number: int) -> str:
    if not upload_id or osp.sep in upload_id:
        raise ValueError("Invalid upload id.")
    if part_number < 1:
        raise ValueError("part_number must be greater than zero.")
    return f"{temp_upload_path}.{upload_id}.part{part_number:05d}"


def validate_local_upload_path(path: str, root_dir: str) -> None:
    root_abs = osp.abspath(root_dir)
    path_abs = osp.abspath(path)
    if osp.commonpath([root_abs, path_abs]) != root_abs:
        raise ValueError(f"Upload path {path!r} is outside storage root.")


class LocalPresignedHelper(PresignedHelper["LocalURI"]):
    @property
    def supports_multipart_upload(self) -> bool:
        return True

    def _generate_download_url(self, uri: "LocalURI") -> str:
        return uri.path

    def _create_multipart_upload(
        self,
        uri: "LocalURI",
        num_parts: int,
    ) -> Tuple[str, Dict[int, str]]:
        upload_id = str(uuid.uuid4())
        urls = {
            # The storage helper is transport-neutral. The v0/v1 router
            # owning the project prefixes its own project-scoped endpoint.
            part_number:
            "?" + urlencode({
                'temp_upload_path': uri.path,
                'upload_id': upload_id,
                'part_number': part_number,
            })
            for part_number in range(1, num_parts + 1)
        }
        return upload_id, urls

    def _complete_multipart_upload(
        self,
        uri: "LocalURI",
        upload_id: str,
        parts: List[PartMetadata],
    ) -> None:
        from service_framework.deploy_env_config import deploy_env
        root_dir = deploy_env().data_plane_storage_root
        temp_upload_path = uri.path
        validate_local_upload_path(temp_upload_path, root_dir)
        os.makedirs(osp.dirname(temp_upload_path), exist_ok=True)
        with open(temp_upload_path, "wb") as output:
            for part in sorted(parts, key=lambda p: p.part_number):
                part_path = local_upload_part_path(temp_upload_path, upload_id,
                                                   part.part_number)
                if not osp.exists(part_path):
                    raise ValueError(f"Upload part {part.part_number} missing.")
                with open(part_path, "rb") as f:
                    output.write(f.read())
        for part in parts:
            part_path = local_upload_part_path(temp_upload_path, upload_id,
                                               part.part_number)
            if osp.exists(part_path):
                os.remove(part_path)


@api.get_presigned_helper.register
def get_presigned_helper(uri: LocalURI) -> LocalPresignedHelper:
    return LocalPresignedHelper(uri)


@api._zip_directory.register
def zip_directory(
    uri: LocalURI,
    recursive: bool = False,
    output_path: Optional[str] = None,
) -> str:
    raise NotImplementedError("zip_directory is not implemented for Local")


@api.customer_facing_path.register
def customer_facing_path(uri: LocalURI) -> str:
    return uri.path
