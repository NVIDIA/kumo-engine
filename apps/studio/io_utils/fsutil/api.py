# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
""" API definition for fsutil
Functions defined in this file accept paths as str
and re-dispatch based on the type returned by ParseURI.
Filesystem-specific implementations live in impl_*.py

Uses singledispatch to call an overloaded function based
on the type of the first argument, defined by type hint.
https://peps.python.org/pep-0443/
"""
import logging
import os.path as osp
import queue
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import singledispatch
from io import BytesIO
from typing import (
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from async_utils.sync import blocking_func, blocking_method
from io_utils.fsutil.common import fsutil_ctx, match_any
from io_utils.fsutil.credentials import ContextPropagatingThreadPoolExecutor
from io_utils.fsutil.uri import AnyURI, ParseURI

logger = logging.getLogger(__name__)


@singledispatch
def filesize(path: str) -> Optional[int]:
    r"""Returns the size of the file at path. Note that it will be None if the
    path is a directory or does not exist.

    Args:
        path (str): A file path.
    """
    return filesize(ParseURI(path))


@dataclass(frozen=True)
class ObjectIdentity:
    """Immutable identity captured for one cloud object."""

    size: int
    etag: Optional[str]
    version: Optional[str]


@singledispatch
def object_identity(path: str) -> ObjectIdentity:
    """Return the size and provider identity for one exact cloud object."""
    return object_identity(ParseURI(path))


@singledispatch
def read_binary(path: str, compression: bool = False) -> bytes:
    return read_binary(ParseURI(path), compression=compression)


@singledispatch
def read_text(path: str, compression: bool = False,
              encoding: Optional[str] = None) -> str:
    return read_text(ParseURI(path), compression=compression, encoding=encoding)


def write_binary(path: str, data: Union[bytes, BinaryIO], *,
                 compression: bool = False, ttl: Optional[timedelta] = None):
    if isinstance(data, bytes):
        return _write_binary(path, BytesIO(data), compression=compression,
                             ttl=ttl)
    else:
        return _write_binary(path, data, compression=compression, ttl=ttl)


@singledispatch
def _write_binary(path: str, data: BinaryIO, compression: bool = False,
                  ttl: Optional[timedelta] = None) -> None:
    return _write_binary(ParseURI(path), data, compression=compression, ttl=ttl)


@singledispatch
def write_text(path: str, data: str, compression: bool = False,
               encoding: Optional[str] = None) -> None:
    return write_text(ParseURI(path), data, compression=compression,
                      encoding=encoding)


@singledispatch
def write_text_if_absent(path: str, data: str) -> bool:
    """Create one text object without replacing an existing object.

    Returns ``True`` only for the caller that created the destination. A
    concurrent caller observing an existing destination receives ``False``.
    Backends must implement this with a provider-side create precondition;
    an ``exists`` check followed by an unconditional write is not sufficient.
    """
    return write_text_if_absent(ParseURI(path), data)


@singledispatch
def _get_binary_handle(
    path: str,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    yield from _get_binary_handle(ParseURI(path), mode, compression=compression)


@contextmanager
def get_binary_handle(
    path: str,
    mode: str,
    compression: bool = False,
) -> Generator[BinaryIO, None, None]:
    """Returns the file object for the file at path.
    Reads and writes as raw binary e.g. `bytes`

    Args:
        path (str): A file path.
        mode (str): Open mode e.g. 'r', 'w', or 'rw'
        compression (bool): Flag for using gzip compression
    """
    yield from _get_binary_handle(path, mode, compression=compression)


@singledispatch
def _get_handle(
    path: str,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    yield from _get_handle(ParseURI(path), mode, compression=compression)


@contextmanager
def get_handle(
    path: str,
    mode: str,
    compression: bool = False,
) -> Generator[TextIO, None, None]:
    """Returns the file object for the file at path.
    Reads and writes as utf-8 text e.g. `str`

    Args:
        path (str): A file path.
        mode (str): Open mode e.g. 'r', 'w', or 'rw'
        compression (bool): Flag for using gzip compression
    """
    yield from _get_handle(path, mode, compression=compression)


@singledispatch
def dirsize(path: str) -> int:
    r""" returns the sum of file sizes under path.

    Args:
        path (str): A file path."""
    return dirsize(ParseURI(path))


@singledispatch
def exists(path: str) -> bool:
    r"""Return :obj:`True` if path refers to an existing path or a file
    descriptor locally or remotely.

    Args:
        path (str): File path.
    """
    return exists(ParseURI(path))


@singledispatch
def isdir(path: str) -> bool:
    r"""Returns :obj:`True` if path is an existing directory.

    Args:
        path (str): File path.
    """
    return isdir(ParseURI(path))


@blocking_func()
def get_many_if_present(paths: List[Optional[str]] | List[str],
                        parallel: bool = True) -> List[Optional[bytes]]:
    if not parallel:
        return [_get_if_present(p) if p is not None else None for p in paths]

    return list(fsutil_ctx.thread_pool.map(_get_if_present, paths))


@singledispatch
def _get_if_present(path: Optional[str]) -> Optional[bytes]:
    return _get_if_present(ParseURI(path)) if path is not None else None


@dataclass
class PathDetail:
    path: str
    is_directory: bool


@overload
def listdir(
    path: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    suffix: Optional[str] = None,
    max_items: Optional[int] = None,
    detail: Literal[False] = False,
) -> list[str]:
    ...


@overload
def listdir(
    path: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    suffix: Optional[str] = None,
    max_items: Optional[int] = None,
    detail: Literal[True],
) -> list[PathDetail]:
    ...


def listdir(
    path: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    suffix: Optional[str] = None,
    max_items: Optional[int] = None,
    detail: bool = False,
) -> Union[list[str], list[PathDetail]]:
    r"""Return a list containing the full path of the entries in the directory
    given by path locally or remotely.

    All entries will be sorted to ensure deterministic output.

    Args:
        path (str): File path.
        include (List[str], optional): A list of file names or patterns to
            include. (default: :obj:`None`)
        exclude (List[str], optional): A list of file names or patterns to
            exclude. (default: :obj:`None`)
        detail (bool, optional): Return detailed information about the entries
            specifically include information about if the entry is file or dir.
    """
    if max_items is not None and max_items <= 0:
        raise ValueError(f"max_items must be greater than 0, got {max_items}")

    # Without these conditions the underlying implementation can eagerly
    # filter using max_items. Same for the if statement in the else block.
    if any([suffix, exclude, include]):
        eager_max_items = None
    else:
        eager_max_items = max_items

    if detail:
        entries = _listdir_detail(path, max_items=eager_max_items)

        def _entry_filter(entry: PathDetail) -> bool:
            if suffix is not None:
                if not (entry.is_directory or entry.path.endswith(suffix)):
                    return False
            if exclude is not None and match_any(entry.path, exclude):
                return False
            if include is not None and not match_any(entry.path, include):
                return False
            return True

        return list(filter(_entry_filter, entries))[:max_items]

    else:
        paths = _listdir(path, max_items=eager_max_items)

        def _path_filter(path: str) -> bool:
            if suffix is not None:
                # This filter allows directories to pass through.
                if not (path.endswith(suffix) or isdir(path)):
                    return False
            if exclude is not None and match_any(path, exclude):
                return False
            if include is not None and not match_any(path, include):
                return False
            return True

        return list(filter(_path_filter, paths))[:max_items]


@singledispatch
def _listdir(
    path: str,
    max_items: Optional[int] = None,
) -> list[str]:
    return _listdir(ParseURI(path), max_items=max_items)


@singledispatch
def _listdir_detail(
    path: str,
    max_items: Optional[int] = None,
) -> list[PathDetail]:
    return _listdir_detail(ParseURI(path), max_items=max_items)


@singledispatch
def makedirs(path: str, exist_ok: bool = False) -> None:
    r"""Recursively create directory `path`.

    Args:
        path (str): Directory path.
        exist_ok (bool, optional): Raise :obj:`FileExistsError` if
        :obj:`exist_ok` is :obj:`False` and the target directory already
        exists. (default: :obj:`False`)
    """
    return makedirs(ParseURI(path), exist_ok=exist_ok)


def list_remote_directory(source_dir: str,
                          destination_dir: str) -> List[PathDetail]:
    makedirs(destination_dir, exist_ok=True)
    try:
        return listdir(source_dir, detail=True)
    except NotImplementedError:
        # A backend whose listings do not expose entry types falls back to
        # per-entry directory probes; object-store backends retain the
        # bounded detailed-listing path.
        return [
            PathDetail(path=path, is_directory=isdir(path))
            for path in listdir(source_dir)
        ]


def walk_directory_tree(
    src_root: str,
    dst_root: str,
    *,
    max_workers: int,
    entry_filter: Optional[Callable[[str], bool]] = None,
) -> List[tuple[str, str]]:
    r"""Breadth-first, concurrently-enumerated walk of ``src_root``.

    Returns ``(src, dst)`` pairs for every file found, with the directory
    tree mirrored under ``dst_root``. Every directory that is walked is
    created at its destination (via ``list_remote_directory``) before it
    is listed, which local destinations need before ``shutil.copyfile`` can
    write into them.

    ``entry_filter``, if given, is called with each entry's basename (file
    or directory); a directory that it rejects is never recursed into.

    Enumeration is round-trip bound, not just the transfer: a 410-directory
    GCS tree spent 12.5s of a 19s copy listing directories one at a time
    before a byte moved.

    Completions arrive on a queue rather than through
    ``concurrent.futures.wait()``, which re-registers a waiter on every
    pending future per call -- O(n^2) main-thread CPU on a wide tree
    (thousands of sibling shard dirs, the DiskGraph shape), contending
    under the GIL with the listings it waits on. Only the main thread
    submits; the callback just enqueues, so a bounded pool cannot deadlock
    waiting for a slot to discover more work.
    """
    transfers: List[tuple[str, str]] = []
    with ContextPropagatingThreadPoolExecutor(
            max_workers=max_workers) as executor:
        completed: queue.SimpleQueue = queue.SimpleQueue()
        pending_destinations: dict = {}
        outstanding = 0

        def _submit(source_dir: str, destination_dir: str) -> None:
            nonlocal outstanding
            future = executor.submit(list_remote_directory, source_dir,
                                     destination_dir)
            pending_destinations[future] = destination_dir
            outstanding += 1
            future.add_done_callback(completed.put)

        _submit(src_root, dst_root)
        try:
            while outstanding:
                future = completed.get()
                outstanding -= 1
                destination_dir = pending_destinations.pop(future)
                # .result() re-raises a listing failure here rather than
                # letting it be swallowed, so a directory this process
                # can't enumerate fails the whole walk instead of silently
                # truncating the transfer list.
                for entry in future.result():
                    name = osp.basename(entry.path.rstrip('/'))
                    if entry_filter is not None and not entry_filter(name):
                        continue
                    destination = osp.join(destination_dir, name)
                    if entry.is_directory:
                        _submit(entry.path, destination)
                    else:
                        transfers.append((entry.path, destination))
        except BaseException:
            for future in pending_destinations:
                future.cancel()
            raise
    return transfers


@singledispatch
def rm(path: str, ignore_errors: bool = False,
       include: Optional[List[str]] = None) -> None:
    r"""Recursively delete the file or the directory `path`.

    Args:
        path (str): Directory path.
        ignore_errors (bool, optional): Ignore errors if :obj:`True`.
        include (List[str], optional): A list of file names or patterns to
            include. (default: :obj:`None`)
    """
    return rm(ParseURI(path), ignore_errors=ignore_errors, include=include)


@singledispatch
def touch(path: str) -> None:
    r"""Create empty file.
    Args:
        path (str): The file path.
    """
    touch(ParseURI(path))


@singledispatch
def has_write_permission(path: str) -> bool:
    r"""Return true if the caller has permission to write to the given file.
    """
    return has_write_permission(ParseURI(path))


@blocking_func()
def contains_subdir(path: str) -> bool:
    """Check if a path contains sub directories."""
    return _contains_subdir(path)


@singledispatch
def _contains_subdir(path: str) -> bool:
    return _contains_subdir(ParseURI(path))


@singledispatch
def get_last_modified_time(path: str, suffix: Optional[str] = None) -> float:
    """
    Returns last modified time for a file or directory pointed to by `path`

    Args:
        path: File or directory for which last modified time is needed. If it
        is a directory, it will return the last modified time of the latest
        modified file within it.
        suffix: File extension to check for last modified.
    """
    return get_last_modified_time(ParseURI(path), suffix=suffix)


@singledispatch
def generate_directory_read_sas(
    path: str,
    validity: Optional[timedelta] = None,
) -> str:
    """Return a read/list SAS for one ADLS directory."""
    if not isinstance(path, str):
        raise NotImplementedError(
            'Directory read SAS generation is not supported for '
            f'{type(path).__name__}.')
    return generate_directory_read_sas(
        ParseURI(path),
        validity=validity,
    )


# ---------------------------------------------------------------------------
# Presigned URL / multipart upload helpers
# ---------------------------------------------------------------------------


@runtime_checkable
class PartMetadata(Protocol):
    """Protocol matching kumoapi.data_source.PartUploadMetadata and any
    object with part_number + etag attributes."""
    part_number: int
    etag: str


UriT = TypeVar('UriT', bound=AnyURI)


class PresignedHelper(ABC, Generic[UriT]):
    """Abstract base for generating presigned/signed URLs and managing
    multipart uploads on a specific cloud backend.

    Each instance is bound to a URI (and thus a specific backend +
    bucket/container).  Public methods accept an optional *path*
    override (``None`` → stored URI, ``str`` → parsed via
    :func:`ParseURI`, URI object → used directly).

    Subclasses implement the private ``_*`` methods which receive a
    resolved URI object.  The public methods are
    :func:`blocking_method`-decorated so that:

    * calling them from an async context raises immediately (no silent
      event-loop blocking),
    * callers can ``await helper.create_multipart_upload.coroutine(...)``
      to run on the thread-pool.
    """
    def __init__(self, uri: UriT):
        self._uri = uri

    @property
    def supports_multipart_upload(self) -> bool:
        """Whether this helper implements browser multipart upload."""
        return False

    def _resolve(self, path: None | str | UriT) -> UriT:
        """Resolve *path* to a URI object.

        * ``None`` → stored URI from construction time.
        * ``str``  → parsed via :func:`ParseURI`.
        * URI object → returned as-is.
        """
        if path is None:
            return self._uri
        uri = ParseURI(path) if isinstance(path, str) else path
        if type(uri) is not type(self._uri):
            raise ValueError(
                f'{self.__class__} does not support path URI type: {type(uri)}')
        return uri  # type: ignore[return-value]

    @abstractmethod
    def _generate_download_url(self, uri: UriT) -> str:
        ...

    @abstractmethod
    def _create_multipart_upload(
        self,
        uri: UriT,
        num_parts: int,
    ) -> Tuple[str, Dict[int, str]]:
        ...

    @abstractmethod
    def _complete_multipart_upload(
        self,
        uri: UriT,
        upload_id: str,
        parts: List[PartMetadata],
    ) -> None:
        ...

    @blocking_method()
    def generate_download_url(self, path: None | str | UriT = None) -> str:
        return self._generate_download_url(self._resolve(path))

    @blocking_method()
    def create_multipart_upload(
        self,
        num_parts: int,
        *,
        path: None | str | UriT = None,
    ) -> Tuple[str, Dict[int, str]]:
        return self._create_multipart_upload(self._resolve(path), num_parts)

    @blocking_method()
    def complete_multipart_upload(
        self,
        upload_id: str,
        parts: List[PartMetadata],
        *,
        path: None | str | UriT = None,
    ) -> None:
        self._complete_multipart_upload(self._resolve(path), upload_id, parts)


@singledispatch
def get_presigned_helper(path) -> Optional[PresignedHelper]:
    """Return a :class:`PresignedHelper` for *path*, or ``None`` if the
    storage backend does not support presigned URLs (e.g. local files)."""
    return get_presigned_helper(ParseURI(path))


def supports_multipart_upload(path: str) -> bool:
    """Return whether *path* has a complete multipart upload contract."""
    helper = get_presigned_helper(path)
    return helper is not None and helper.supports_multipart_upload


@singledispatch
def atomic_write_text(path: str, data: str, compression: bool = False) -> None:
    """Atomically publish text without exposing partial destination data."""
    return atomic_write_text(ParseURI(path), data, compression=compression)


@blocking_func()
def zip_directory(path: str, recursive: bool = False,
                  output_path: Optional[str] = None) -> str:
    """Zip all files under *path* into a single archive and return the path
    to the resulting zip file."""
    return _zip_directory(path, recursive=recursive, output_path=output_path)


@singledispatch
def _zip_directory(path, recursive: bool = False,
                   output_path: Optional[str] = None) -> str:
    return _zip_directory(ParseURI(path), recursive=recursive,
                          output_path=output_path)


@blocking_func()
def zip_files(paths: Sequence[str], output_path: str) -> str:
    """Zip exactly *paths* without discovering additional prefix objects."""
    if not paths:
        raise ValueError('Cannot create an archive from an empty file list')
    parsed = tuple(ParseURI(path) for path in paths)
    return _zip_files(parsed[0], parsed, output_path)


@singledispatch
def _zip_files(path, paths: Sequence[AnyURI], output_path: str) -> str:
    raise NotImplementedError(
        f'zip_files is not implemented for {type(path).__name__}')


@singledispatch
def customer_facing_path(uri) -> str:
    """Return the canonical, client-accessible URL/URI for *path*.

    For most backends the path string is already a URL (``s3://``,
    ``gs://``, ``dbfs:``, ``abfss://``, …).
    """
    return customer_facing_path(ParseURI(uri))
