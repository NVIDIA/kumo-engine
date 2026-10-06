# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This file is deprecated. Add new functionality to api.py
and filesystem-specific implementations to impl_*.py
"""
import logging
import multiprocessing as mp
import os
import os.path as osp
import shutil
from concurrent.futures import as_completed
from multiprocessing.pool import Pool
from typing import Any, Callable, List, Optional

from asgiref.sync import SyncToAsync

from async_utils.sync import blocking_func
from io_utils.fsutil import api, uri
from io_utils.fsutil.common import (
    TemporaryDirectory,
    fsExceptionHandler,
    is_local,
    should_exclude,
    should_include,
)
from io_utils.fsutil.credentials import (
    ContextPropagatingThreadPoolExecutor,
    has_filesystem_credential_scope,
)
from io_utils.fsutil.logging_policy import logging_policy
from io_utils.retry import retry

logger = logging.getLogger(__name__)

generic_logging_policy = logging_policy(logger)

# Since each row in the predictions or embeddings output may contain a lot of
# data, we limit the number of rows per group to 100k, in order to minimize the
# amount of memory required by each row group. It is recommended to set
# `row_group_size` to this value, whenever exporting an artifact to parquet
# using `DataFrame.to_parquet()`
PARQUET_ROW_GROUP_SIZE = 100000


def is_s3(path: str) -> bool:
    r"""Return :obj:`True` if path refers to a s3 path.

    Args:
        path (str): A s3 file path.
    """
    return path.startswith('s3://') or path.startswith('s3a://')


def is_dbfs(path: str) -> bool:
    r"""Return :obj:`True` if path refers to a dbfs path.

    Args:
        path (str): A dbfs file path.
    """
    return path.startswith('dbfs:/')


def is_gcs(path: str) -> bool:
    r"""Return :obj:`True` if path refers to a GCS path.

    Args:
        path (str): A GCS file path.
    """
    return path.startswith('gs://')


def is_adls(path: str) -> bool:
    """Check if path is an ADLS URI (abfss:// or abfs://)"""
    return isinstance(path, str) and (path.startswith('abfss://')
                                      or path.startswith('abfs://'))


# TODO (min) Move the s3 specific utility functions into a separate
# s3_util.py
def s3_as_s3a(path: str) -> str:
    r"""Change a s3 file path into a s3a file path for consumption by PySpark.
    This is only needed if using open source Spark to read from S3. If using
    EMR Spark, it can read a s3 file path.

    Args:
        path (str): A s3 file path.
    """
    return path.replace('s3:', 's3a:', 1)


def s3a_as_s3(path: str) -> str:
    r"""Change a s3a file path to be consumed by PySpark into a s3 path for
    consumption by the rest of Kumo.
    This is only needed if using open source Spark to read from S3. If using
    EMR Spark, it can read a s3 file path.

    Args:
        path (str): A s3a file path.
    """
    return path.replace('s3a:', 's3:', 1)


@fsExceptionHandler
def file_lines(path: str) -> int:
    r"""Returns the size of the file at path. Note that it will be None if the
    path is a directory.

    Args:
        path (str): A file path.
    """
    try:
        count = 0
        with api.get_handle(path, 'r') as f:
            next(f)  # skip header
            for count, _ in enumerate(f):
                pass
        # count is 0 here if there is 1 or 0 non-header lines so test which
        if count == 0:
            with api.get_handle(path, 'r') as f:
                next(f)
                test = next(f, None)
                if not test:
                    return 0

        return count + 1
    except FileNotFoundError:
        return 0


# TODO(siyang): Decorate exists() as blocking_func to enforce async code to
# use the .coroutine() interface.
async def exists_async(path: str) -> bool:
    r"""Async version of `exists`.

    Args:
        path (str): File path.

    Returns:
        bool: if path refers to an existing file path locally or remotely
    """
    return await SyncToAsync(api.exists, thread_sensitive=False)(path)


@fsExceptionHandler
def is_empty_dir(path: str) -> bool:
    r"""Returns :obj:`True` if the path does not exists, is a file or is a
    directory contains empty files.
    """
    if not api.exists(path) or not api.isdir(path):
        return True
    files = api.listdir(path)
    return len(files) == 0


def _rewrite_uri(path: str) -> tuple[uri.AnyURI, str]:
    return uri.ParseURI(path), path


@generic_logging_policy(allowed_exceptions=FileNotFoundError)
@retry(exceptions=(RuntimeError, BlockingIOError), tries=2, delay=1, backoff=2,
       logger=logger)
def copy_file(
    src: str,
    dst: str,
    overwrite: bool = False,
    raise_on_exists: bool = True,
):
    r"""Copy file from `src` to `dst`.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
        overwrite (bool, optional): Overwrite `dst` if it exists. (default:
            :obj:`False`)
    """
    if api.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if not overwrite and api.exists(dst):
        if raise_on_exists:
            raise FileExistsError(f"File {dst} already exists")
        return

    src_uri, src = _rewrite_uri(src)
    dst_uri, dst = _rewrite_uri(dst)

    source_missing = False
    copy_failed = False
    try:
        if isinstance(src_uri, uri.UCURI) or isinstance(dst_uri, uri.UCURI):
            # Runtime import UCV
            from io_utils.fsutil import impl_ucv  # pants: no-infer-dep
            impl_ucv.copy_file(src, dst)
        elif isinstance(src_uri, uri.S3URI) and isinstance(dst_uri, uri.S3URI):
            from io_utils.fsutil import \
                s3_filesystem as s3_util  # pants: no-infer-dep
            s3_util.copy_file(src, dst)
        elif isinstance(src_uri, uri.S3URI) and isinstance(
                dst_uri, uri.LocalURI):
            from io_utils.fsutil import \
                s3_filesystem as s3_util  # pants: no-infer-dep
            s3_util.download_file(src, dst)
        elif isinstance(src_uri, uri.LocalURI) and isinstance(
                dst_uri, uri.S3URI):
            from io_utils.fsutil import \
                s3_filesystem as s3_util  # pants: no-infer-dep
            s3_util.put_file(src, dst)
        elif isinstance(src_uri, uri.GCSURI) or isinstance(dst_uri, uri.GCSURI):
            from io_utils.fsutil.impl_gcs import \
                copy_file as _gcs_copy_file  # pants: no-infer-dep
            _gcs_copy_file(src, dst)
        elif isinstance(src_uri, uri.ADLSURI) or isinstance(
                dst_uri, uri.ADLSURI):
            # ADLS handling
            from io_utils.fsutil.impl_adls import \
                copy_file  # pants: no-infer-dep
            copy_file(src, dst)
        else:
            shutil.copyfile(src, dst)
    except FileNotFoundError:
        logger.error('Source file copy failed '
                     '(error_type=FileNotFoundError)')
        source_missing = True
    except Exception as error:
        logger.error('File copy failed (error_type=%s)', type(error).__name__)
        try:
            api.rm(dst)  # cleanup before retry
        except Exception as cleanup_error:
            logger.error('File copy cleanup failed (error_type=%s)',
                         type(cleanup_error).__name__)
        copy_failed = True

    # Raise outside the provider exception handler so the stable exception has
    # neither a cause nor a context containing signed URLs, request bodies, or
    # temporary credentials. Retry and Temporal may safely serialize it.
    if source_missing:
        raise FileNotFoundError('Source file not found') from None
    if copy_failed:
        raise RuntimeError('File copy failed') from None


# TODO: fsutil.copy() is not working properly on large data exceeding 1GB.
# Related issue: https://github.com/kumo-ai/kumo/issues/9203 and
# https://github.com/kumo-ai/kumo/issues/9362. We will get an error
# saying connection pool is full. The workaround is to copy each partition
# of the file sequentially in a for loop.
@blocking_func()
@fsExceptionHandler
def copy(
    src_path: str,
    dst_path: str,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    overwrite: bool = False,
    raise_on_exists: bool = True,
    enable_multi_process: bool = False,
    pool: Optional[Pool] = None,
):
    r"""Recursively copies all files from the source path to the destination
        path. Supports both single files and folders.

    Args:
        src_path (str): The source path.
        dst_path (str): The destination path.
        exclude (List[str], optional): A list of file names or patterns to
            exclude. (default: :obj:`None`)
        overwrite (bool, optional): Overwrite `dst_path` if it exists.
            (default: :obj:`False`)
        enable_multi_process, pool: Both ignored. Directory copies are
            always concurrent via threads now, never a caller-supplied
            process pool -- a ``multiprocessing.Pool`` cannot carry the
            credential ContextVar across a process boundary, so threads
            are required regardless of what is passed here. These
            parameters exist only because ``fsutil`` is a cross-repo API:
            the shipped ``kumo_core.explain.subgraph_collector``
            still calls ``copy(..., enable_multi_process=True)``, and a
            grep of this repo cannot prove no out-of-repo caller passes
            ``pool=`` too. Accepting and ignoring both keeps every caller
            working; drop them once verified unused everywhere.
    """
    del enable_multi_process, pool
    if src_path == dst_path:
        return
    logger.info("Copying %s to %s", src_path, dst_path)
    if (is_s3(src_path) and is_s3(dst_path) and api.isdir(src_path)
            and overwrite is True and raise_on_exists is False):
        # Use highly optimized S3 copy, if overwrite is allowed
        if include or exclude:
            raise NotImplementedError("include and exclude are currently not "
                                      "supported for s3 to s3 copy")
        from io_utils.fsutil import \
            s3_filesystem as s3_util  # pants: no-infer-dep
        s3_util.copydir(src_path, dst_path)
        return
    if (is_adls(src_path) and is_adls(dst_path) and api.isdir(src_path)
            and overwrite is True and raise_on_exists is False):
        if include or exclude:
            raise NotImplementedError("include and exclude are currently not "
                                      "supported for adls to adls copy")
        from io_utils.fsutil.impl_adls import copydir  # pants: no-infer-dep
        copydir(src_path, dst_path)
        return
    if (is_gcs(src_path) and is_gcs(dst_path) and api.isdir(src_path)
            and overwrite is True and raise_on_exists is False):
        if include or exclude:
            raise NotImplementedError("include and exclude are currently not "
                                      "supported for gcs to gcs copy")
        from io_utils.fsutil.impl_gcs import \
            copydir as gcs_copydir  # pants: no-infer-dep
        gcs_copydir(src_path, dst_path)
        return

    # TODO(dong): use shutil.copytree, s3 copy, put, get when exclude is empty.
    # For files, copy the file.
    if not api.isdir(src_path) and not should_exclude(
            src_path, exclude=exclude) and should_include(
                src_path, include=include):
        copy_file(src_path, dst_path, overwrite=overwrite,
                  raise_on_exists=raise_on_exists)
        return

    if not api.exists(dst_path):
        api.makedirs(dst_path)

    src_path = src_path + os.sep if not src_path.endswith(os.sep) else src_path

    def _entry_filter(name: str) -> bool:
        return not should_exclude(name, exclude=exclude) and should_include(
            name, include=include)

    # 32 matches _copy_files_with_scoped_threads' default concurrency: on a
    # 2,426-file/10.45GB GCS tree, serial listing + transfer took 414.7s
    # (25.2 MB/s); walking and transferring concurrently at this width took
    # 25.7s (406.4 MB/s).
    max_workers = 32
    transfers = api.walk_directory_tree(src_path.rstrip('/'), dst_path,
                                        max_workers=max_workers,
                                        entry_filter=_entry_filter)
    if not transfers:
        return

    logger.info("copy: %d files %s -> %s (workers=%d)", len(transfers),
                src_path, dst_path, max_workers)
    _copy_files_with_scoped_threads(
        [(src, dst, overwrite, raise_on_exists) for src, dst in transfers],
        max_workers=max_workers,
    )


@blocking_func()
@fsExceptionHandler
def copy_tree_threaded(
    src_dir: str,
    dst_dir: str,
    *,
    exclude: Optional[Callable[[str], bool]] = None,
    max_workers: int = 8,
) -> None:
    r"""Recursively copy every file under ``src_dir`` to ``dst_dir``,
    preserving the directory tree, with parallelism via threads.

    Threads (not processes) because each transfer is network I/O which
    releases the GIL — threads share the underlying client's connection
    pool (no fragmentation across workers), have lower startup overhead
    than processes, and avoid the per-task pickling cost. ``aws s3 sync``
    uses the same model internally; boto3 exposes no bulk-directory
    primitive for local→S3 transfers.

    Use this over ``copy()`` for a *local* source when the workload is many
    small files (thousands of partition parquets, artifact bundles, etc.)
    and you need the callable ``exclude`` predicate — a per-path check that
    list-pattern matching can't express (e.g. "skip only the top-level
    marker, keep nested files with the same name"). A remote source has no
    such predicate to offer, so it is handed to ``copy()`` whole, which
    already parallelizes both the listing and the transfer. For S3↔S3
    directory copies prefer ``s3_filesystem.copydir``.

    Args:
        src_dir (str): Source directory. Local sources are inventoried with
            :func:`os.walk`; remote sources delegate entirely to ``copy()``.
        dst_dir (str): Destination directory; any backend supported by
            :func:`copy_file`. Created if missing.
        exclude: Optional predicate called with the absolute source
            path of each file; when it returns True the file is
            skipped. ``None`` (default) means copy everything. Use this
            for "upload everything except the commit marker, then
            upload the commit marker last" patterns, or for filtering
            by extension / depth / nested location. Only meaningful for a
            local source — asserted ``None`` otherwise, since ``copy()``
            only understands list-pattern ``include``/``exclude``.
        max_workers (int): Cap on concurrent in-flight transfers for a
            *local* source tree. Default 8 — empirically optimal against
            the standard boto3 S3 client whose ``max_pool_connections`` is
            10; past that, threads contend for connections and throughput
            collapses (benchmarked at 3-4x worse at w=32 on a small
            bundle). See scripts/bench_tt_upload.py. Not consulted for a
            remote source: ``copy()`` walks and transfers at its own fixed
            width (32 — measured optimal against real object-store
            listing/transfer latency, a different bottleneck than the
            boto3 client pool this default is tuned for).

    Raises on the first failure. For a local source, in-flight transfers
    are cancelled by the executor's context manager exit, and
    ``overwrite=True`` is always submitted to ``copy_file`` because
    partial-failure retries should not race on a "destination already
    exists" check.
    """
    if src_dir == dst_dir:
        return

    if not is_local(src_dir):
        # copy() takes list-pattern excludes only, not a callable predicate.
        assert exclude is None, (
            'copy_tree_threaded: a callable `exclude` predicate is not '
            'supported for remote sources.')
        copy(
            src_dir,
            dst_dir,
            exclude=None,
            include=None,
            overwrite=True,
            raise_on_exists=False,
        )
        return

    # os.walk sees the LOCAL filesystem only.
    src_root = src_dir.rstrip('/')

    transfers: List[tuple] = []
    for root, _dirs, files in os.walk(src_root):
        rel = osp.relpath(root, src_root)
        is_top = (rel == '.')
        for fname in files:
            src = osp.join(root, fname)
            if exclude is not None and exclude(src):
                continue
            dst = (osp.join(dst_dir, fname) if is_top else osp.join(
                dst_dir, rel, fname))
            transfers.append((src, dst))

    if not transfers:
        return

    # Pre-create every destination parent directory once, on the main
    # thread, before submitting any transfer. local→local needs the
    # parent dir to exist (``shutil.copyfile`` does not autocreate);
    # S3/GCS/ADLS treat ``makedirs`` as a no-op, so this is harmless
    # there. Doing it upfront (not per-task inside ``_copy_one``)
    # avoids redundant work + serializes the makedirs side-effect.
    parent_dirs = {osp.dirname(dst) for _src, dst in transfers}
    for parent in sorted(parent_dirs):
        api.makedirs(parent, exist_ok=True)

    logger.info("copy_tree_threaded: %d files %s -> %s (workers=%d)",
                len(transfers), src_dir, dst_dir, max_workers)

    with ContextPropagatingThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(copy_file, src, dst, True) for src, dst in transfers
        ]
        for fut in as_completed(futures):
            fut.result()  # surfaces the first failure


def _terminate_pool_best_effort(pool: Optional[Pool]) -> None:
    if pool is None:
        return
    try:
        pool.terminate()
    except Exception as error:
        logger.error('Directory copy pool termination failed (error_type=%s)',
                     type(error).__name__)
    try:
        pool.join()
    except Exception as error:
        logger.error('Directory copy pool join failed (error_type=%s)',
                     type(error).__name__)


def _copy_files_with_scoped_threads(
    transfers: List[tuple[str, str, bool, bool]],
    *,
    max_workers: int = 32,
) -> None:
    """Copy under the exact ContextVar scope captured for every task.

    A credential scope must never cross a multiprocessing boundary: Python
    3.14 defaults to forkserver/spawn, and neither transports ContextVars.
    Waiting for running tasks before returning also ensures no destination
    write can outlive the credential scope after cancellation or failure.
    """
    if not transfers:
        return
    futures = []
    with ContextPropagatingThreadPoolExecutor(
            max_workers=min(max_workers, len(transfers))) as executor:
        try:
            futures = [
                executor.submit(copy_file, *transfer) for transfer in transfers
            ]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def copy_files_threaded(
    transfers: List[tuple[str, str]],
    *,
    overwrite: bool = False,
    raise_on_exists: bool = True,
    max_workers: int = 8,
) -> None:
    """Copy an exact file list with bounded, credential-scoped threads."""
    _copy_files_with_scoped_threads(
        [(src, dst, overwrite, raise_on_exists) for src, dst in transfers],
        max_workers=max_workers,
    )


@blocking_func()
@fsExceptionHandler
def copy_one_level_directory(
    src_path: str,
    dst_path: str,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    overwrite: bool = False,
    raise_on_exists: bool = True,
):
    r"""Copies all files from the source path to the destination
        path. This method is optimized for stage mounts and uses
        multi-processing to speed up the copy.

    Args:
        src_path (str): The source path.
        dst_path (str): The destination path.
        exclude (List[str], optional): A list of file names or patterns to
            exclude. (default: :obj:`None`)
        include (List[str], optional): A list of file names or patterns to
            include. (default: :obj:`None`)
        overwrite (bool, optional): Overwrite `dst_path` if it exists.
            (default: :obj:`False`)
        raise_on_exists (bool, optional): Raise an exception if the destination
            exists if set to :obj:`True`. (default: :obj:`True`)
    """
    if src_path == dst_path:
        return
    logger.info("Copying %s to %s", src_path, dst_path)

    if not api.isdir(src_path) and not should_exclude(
            src_path, exclude=exclude) and should_include(
                src_path, include=include):
        copy_file(src_path, dst_path, overwrite=overwrite,
                  raise_on_exists=raise_on_exists)
        return

    pool = None
    directory_copy_failed = False
    destination_exists = False
    try:
        api.makedirs(dst_path, exist_ok=not raise_on_exists)
        src_path = (src_path +
                    os.sep if not src_path.endswith(os.sep) else src_path)

        # For directories, copy the files in the directory recursively.
        transfers: List[tuple[str, str, bool, bool]] = []
        for src_entry in api.listdir(src_path):  # type: ignore
            entry = os.path.basename(src_entry)
            if should_exclude(entry, exclude) or \
                    not should_include(entry, include=include):
                continue

            dst_entry = os.path.join(dst_path, entry)
            transfers.append((src_entry, dst_entry, overwrite, raise_on_exists))

        if has_filesystem_credential_scope():
            logger.info('Copy %s to %s with scoped threads', src_path, dst_path)
            _copy_files_with_scoped_threads(transfers)
        else:
            processes = min(32, mp.cpu_count())
            logger.info(f"Copy {src_path} to {dst_path} with {processes=}")
            pool = Pool(processes=processes)
            futures = [
                pool.apply_async(copy_file, transfer) for transfer in transfers
            ]
            pool.close()
            pool.join()
            # Handle any errors from the worker processes
            for future in futures:
                future.get()
    except Exception as error:
        _terminate_pool_best_effort(pool)
        logger.error('Directory copy failed (error_type=%s)',
                     type(error).__name__)
        if isinstance(error, FileExistsError):
            destination_exists = True
        else:
            directory_copy_failed = True

    if destination_exists:
        raise FileExistsError('Destination already exists') from None
    if directory_copy_failed:
        raise RuntimeError('Directory copy failed') from None


@blocking_func()
@fsExceptionHandler
def copy_two_level_directory(
    src_path: str,
    dst_path: str,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    overwrite: bool = False,
    raise_on_exists: bool = True,
):
    r"""Recursively copies all files from the source path to the destination
        path for directories that are at most 2 levels deep. This method
        uses multiple processes to speed up the copy.

    Args:
        src_path (str): The source path.
        dst_path (str): The destination path.
        exclude (List[str], optional): A list of file names or patterns to
            exclude. (default: :obj:`None`)
        include (List[str], optional): A list of file names or patterns to
            include. (default: :obj:`None`)
        overwrite (bool, optional): Overwrite `dst_path` if it exists.
            (default: :obj:`False`)
    """
    if src_path == dst_path:
        return
    logger.info("Copying %s to %s", src_path, dst_path)

    if not api.isdir(src_path) and not should_exclude(
            src_path, exclude=exclude) and should_include(
                src_path, include=include):
        copy_file(src_path, dst_path, overwrite=overwrite,
                  raise_on_exists=raise_on_exists)
        return

    pool = None
    directory_copy_failed = False
    destination_exists = False
    try:
        api.makedirs(dst_path, exist_ok=not raise_on_exists)

        src_path = (src_path +
                    os.sep if not src_path.endswith(os.sep) else src_path)
        # For directories, copy the files in the directory recursively.
        transfers: List[tuple[str, str, bool, bool]] = []
        for src_entry in api.listdir(src_path):  # type: ignore
            entry = os.path.basename(src_entry)
            if (should_exclude(entry, exclude)
                    or not should_include(entry, include=include)):
                continue

            dst_entry = os.path.join(dst_path, entry)
            if api.isdir(src_entry):
                # 2nd level directory
                api.makedirs(dst_entry, exist_ok=not raise_on_exists)
                for src_entry_2 in api.listdir(src_entry):  # type: ignore
                    filename = os.path.basename(src_entry_2)
                    if (should_exclude(filename, exclude)
                            or not should_include(filename, include=include)):
                        continue
                    dst_entry_2 = os.path.join(dst_entry, filename)
                    transfers.append(
                        (src_entry_2, dst_entry_2, overwrite, raise_on_exists))
            else:
                transfers.append(
                    (src_entry, dst_entry, overwrite, raise_on_exists))

        if has_filesystem_credential_scope():
            logger.info('Copy %s to %s with scoped threads', src_path, dst_path)
            _copy_files_with_scoped_threads(transfers)
        else:
            processes = min(32, mp.cpu_count())
            logger.info(f"Copy {src_path} to {dst_path} with {processes=}")
            pool = Pool(processes=processes)
            results = [
                pool.apply_async(copy_file, transfer) for transfer in transfers
            ]
            pool.close()
            pool.join()
            for result in results:
                result.get()
    except Exception as error:
        _terminate_pool_best_effort(pool)
        logger.error('Two-level directory copy failed (error_type=%s)',
                     type(error).__name__)
        if isinstance(error, FileExistsError):
            destination_exists = True
        else:
            directory_copy_failed = True

    if destination_exists:
        raise FileExistsError('Destination already exists') from None
    if directory_copy_failed:
        raise RuntimeError('Directory copy failed') from None


def expanduser(path: str) -> str:
    r"""Replaces an initial `~` with the user's home directory.

    Args:
        path (str): The file path.
    """
    if not is_local(path):
        return path

    return os.path.expanduser(path)


def abspath(path: str) -> str:
    r"""Convert a relative path to an absolute path on the local file system.

    Args:
        path (str): The file path.
    """
    if not is_local(path):
        return path

    return os.path.abspath(path)


# Torch save and load utilities
@fsExceptionHandler
def torch_save(obj: Any, path: str, **kwargs):
    r"""Saves the object using `torch.save` to the provided :obj:`path`

    Args:
        path (str): Can be S3 or a local filesystem path.
    """
    import torch
    if is_local(path):
        torch.save(obj, path, **kwargs)
    else:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "tmp_file")
            torch.save(obj, tmp_path, **kwargs)
            copy_file(tmp_path, path, overwrite=True)


@fsExceptionHandler
def torch_load(path: str, **kwargs):
    r"""Loads the object from the provided :obj:`path`

    Args:
        path (str): Can be S3 or a local filesystem path.
    """
    import torch
    logger.info(f"torch_load {path}")
    if is_local(path):
        return torch.load(path, **kwargs)

    if "mmap" in kwargs:
        raise ValueError("Cannot mmap against non-local path")

    with TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "tmp_file")
        copy_file(path, tmp_path)
        return torch.load(tmp_path, **kwargs)


def file_tree(path: str, *, eol='\n', level=0) -> str:
    tree = f"{'--' * level}> {osp.basename(path)}"
    if api.isdir(path):
        for sub in api.listdir(path):
            tree += f"{eol}{file_tree(sub, eol=eol, level=level+1)}"
    return tree
