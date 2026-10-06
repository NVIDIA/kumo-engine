# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import multiprocessing
import os.path as osp
import warnings
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Union

import pandas as pd
import pyarrow
import pyarrow.dataset
from pyarrow import parquet as pq

try:
    from botocore.exceptions import (
        ClientError,
        ConnectTimeoutError,
        NoCredentialsError,
        ReadTimeoutError,
    )
except ImportError:
    ClientError = Exception
    NoCredentialsError = Exception
    ConnectTimeoutError = Exception
    ReadTimeoutError = Exception

from async_utils.sync import blocking_func
from io_utils import fsutil, logging
from io_utils.databricks import (
    get_num_rows_databricks,
    get_parquet_head_databricks,
)
from io_utils.fsutil.common import is_ucv
from io_utils.fsutil.credentials import has_filesystem_credential_scope
from io_utils.fsutil.generic import _rewrite_uri
from io_utils.fsutil.uri import ParseURI
from io_utils.pyarrow import arrow_cloud_filesystem
from monitoring.tracing import TraceOptions
from service_framework.env_var import EnvVar

logger = logging.getLogger(__name__)

_MAX_READAHEAD_MEM_SIZE = 100 << 20  # 100MB max memory size for file readahead


def _is_access_or_io_error(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` is an access / auth / IO failure.

    Such errors mean we *could not read* the table and must propagate, rather
    than be swallowed as a row count of 0 (which would report a healthy source
    as empty and silently drop data). A genuinely missing path is not an access
    failure. pyarrow surfaces S3 access/IO failures (including a 403) as
    ``OSError``, and boto3 raises ``ClientError`` and friends, so classifying by
    exception type is sufficient -- no error-message matching.
    """
    if isinstance(exc, FileNotFoundError):
        return False
    module = getattr(type(exc), '__module__', '') or ''
    if any(sdk in module
           for sdk in ('botocore', 'boto3', 's3fs', 'aiobotocore')):
        return True
    return isinstance(exc, (
        ClientError,
        NoCredentialsError,
        ConnectTimeoutError,
        ReadTimeoutError,
        ConnectionError,
        OSError,
        TimeoutError,
    ))


# Hard timeout (seconds) for the isolation subprocess. Overridable per deploy.
_SUBPROCESS_TIMEOUT_SECS = EnvVar(
    'PARQUET_ISOLATED_SUBPROCESS_TIMEOUT_SECS',
    type=int,
    required=False,
    default_value=300,
)

# Fan-out for the per-fragment footer reads in _count_rows_per_file. The reads
# are I/O-bound S3/GCS round-trips, so a large table (thousands of parts) counts
# serially in ~part_count * footer_latency -- enough to blow the subprocess
# timeout. Threads overlap the round-trips; fsutil's cloud clients are
# thread-safe (see JSON_IO_THREAD_POOL).
_NUM_ROWS_MAX_WORKERS = EnvVar(
    'PARQUET_NUM_ROWS_MAX_WORKERS',
    type=int,
    required=False,
    default_value=32,
)


def _run_in_isolated_subprocess(func, args):
    """Run ``func(*args)`` in a single-use child process and return its result.

    Isolates the per-process aws-sdk-cpp ``S3FileSystem`` state: once a read
    hits "curlCode: 28, Timeout was reached" (#12275) the cached client stays
    poisoned and every later read in the process fails the same way. Running
    each read in a throwaway process discards that state.

    py3.14-safety: use an explicit ``fork`` context (the 3.14 default moved to
    forkserver/spawn, whose full ``kumo`` re-import deadlocks under the
    asyncio/logging locks held inside a Temporal activity) and a hard timeout
    that *raises* (a bare ``pool.apply()`` blocks forever on a wedged child).
    On timeout we tear down the pool and raise rather than fall back in-process
    -- the in-process client is the poisoned one this isolation exists to avoid.
    """
    # fork is unavailable on macOS/Windows; run in-process there.
    try:
        ctx = multiprocessing.get_context('fork')
    except ValueError:
        return func(*args)

    timeout_secs = _SUBPROCESS_TIMEOUT_SECS.parse_value()
    # Scope the py3.14 fork()-after-threads DeprecationWarning to this
    # deliberate fork (the timeout is the backstop) so it does not spam reads.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='.*fork.*may lead to deadlocks.*',
            category=DeprecationWarning,
        )
        pool = ctx.Pool(processes=1)
    try:
        async_result = pool.apply_async(func, args)
        result = async_result.get(timeout=timeout_secs)
        pool.close()
        pool.join()
        return result
    except multiprocessing.TimeoutError:
        logger.error(
            "Isolated subprocess for %s timed out after %ss; tearing down the "
            "pool and surfacing the timeout.", getattr(func, '__name__', func),
            timeout_secs)
        pool.terminate()
        pool.join()
        raise TimeoutError(f"Isolated subprocess for "
                           f"{getattr(func, '__name__', func)} timed out after "
                           f"{timeout_secs}s")
    except Exception:
        pool.terminate()
        pool.join()
        raise


# pyarrow's stubs don't re-export Dataset from pyarrow.dataset, though it is
# available at runtime.
@contextmanager
def _get_dataset(
        path: str,
        **kwargs) -> Generator[pyarrow.dataset.Dataset]:  # type: ignore
    # PyArrow doesn't recognize s3a:// (Spark/Hadoop scheme); convert to s3://.
    # Mirrors kumo/data/generic/edge_materialize.py:159,169,563. Without this,
    # `pyarrow.dataset.dataset` raises "Expected an S3 object path... got a URI"
    # which gets swallowed by `_is_empty_helper`'s broad except, mis-reporting
    # tables as empty.
    path = fsutil.s3a_as_s3(path)
    is_adls_path = fsutil.is_adls(path)
    is_cloud_path = (fsutil.is_s3(path) or fsutil.is_gcs(path) or is_adls_path)
    filesystem = None
    if is_cloud_path:
        filesystem = (fsutil.pyarrow_filesystem(path)
                      if is_adls_path or has_filesystem_credential_scope() else
                      arrow_cloud_filesystem(path))

    if fsutil.is_local(path) or is_cloud_path:
        path_without_scheme = (ParseURI(path).path if is_adls_path else
                               path.removeprefix("s3://").removeprefix("gs://"))
        if path.endswith("*"):
            # no filesystem or wildcard character for string matching:
            trimmed_path = path_without_scheme.removesuffix("*")
            dataset = pyarrow.dataset.dataset(osp.dirname(path_without_scheme),
                                              format="parquet",
                                              filesystem=filesystem)
            filtered_paths = [
                f.path for f in dataset.get_fragments()
                if f.path.startswith(trimmed_path)
            ]
            # Rebuild a dataset from those paths using the same filesystem
            yield pyarrow.dataset.dataset(filtered_paths, format="parquet",
                                          filesystem=dataset.filesystem)
        else:
            yield pyarrow.dataset.dataset(path_without_scheme,
                                          filesystem=filesystem, **kwargs)

        return

    assert not path.endswith(
        "*"), "Wildcard is not supported for non-local/non-cloud paths"
    with fsutil.TemporaryDirectory() as temp_dir:
        logger.info(f"Staging parquet data from {path} at {temp_dir}")
        fsutil.copy(path, temp_dir)
        yield pyarrow.dataset.dataset(temp_dir, **kwargs)


def read_parquet(
    path: str,
    exclude_invalid_files: bool = False,
    return_arrow: bool = False,
    offset: int = 0,
    size: int | None = None,
) -> Union[pd.DataFrame, pyarrow.Table]:
    """Read a parquet file or parquet dataset from path.

    Args:
        path (str): The parquet file or dataset to read.
        exclude_invalid_files (bool): Whether to exclude invalid files.
        return_arrow (bool): Whether to return PyArrow table instead of
            DataFrame.
        offset (int): Starting row index (0-based).
        size (Optional[int]): Number of rows to read. If None, read all
            rows from offset.
    """
    path = fsutil.s3a_as_s3(path)
    logger.info(f"read_parquet: {path}, offset: {offset}, size: {size}")

    # Case 1: offset=0 and size=None - use current dataset approach
    if offset == 0 and size is None:
        return _read_full_dataset(path, exclude_invalid_files, return_arrow)

    # Case 2: size is not None - use file-by-file approach
    else:
        return _read_with_file_iteration(path, exclude_invalid_files,
                                         return_arrow, offset, size)


def _read_full_dataset(
        path: str, exclude_invalid_files: bool,
        return_arrow: bool) -> Union[pd.DataFrame, pyarrow.Table]:
    """Read the full dataset using the original approach."""
    with _get_dataset(path, exclude_invalid_files=exclude_invalid_files) as ds:
        table = ds.to_table()

        if return_arrow:
            return table

        # Directly calling pd.read_parquet results in an OOM error, so we
        # manually construct the pandas dataframe by iterating through the
        # pyarrow table incrementally, which minimizes the memory requirement.
        parts = []
        for batch in table.to_batches():
            df = batch.to_pandas()
            parts.append(df)

        # Handle empty result
        if len(parts) == 0:
            return table.to_pandas()

        return pd.concat(parts, ignore_index=True)


def _read_with_file_iteration(
        path: str, exclude_invalid_files: bool, return_arrow: bool, offset: int,
        size: int | None) -> Union[pd.DataFrame, pyarrow.Table]:
    """Read data file by file, respecting offset and size constraints."""

    # List all parquet files in the directory
    files = [path] if not fsutil.isdir(path) else fsutil.listdir(path)

    # Get metadata for each file to determine row counts
    total_rows = 0

    # Determine which files we need to read
    tables_to_concat = []

    for file_path in files:
        try:
            with _get_dataset(file_path) as ds:
                num_rows = ds.count_rows()

                if total_rows + num_rows <= offset:
                    total_rows += num_rows
                    continue

                file_table = ds.to_table()
                file_offset = max(0, offset - total_rows)
                if size is None:
                    size = 0
                    file_end_local = num_rows
                else:
                    file_end_local = min(num_rows, offset + size - total_rows)

                if file_offset > 0 or file_end_local < num_rows:
                    file_table = file_table.slice(file_offset,
                                                  file_end_local - file_offset)

                if len(file_table) > 0:
                    tables_to_concat.append(file_table)

                total_rows += num_rows
                if total_rows > offset + size:
                    break
        except Exception as e:
            error_msg = str(e).lower()
            error_module = getattr(type(e), '__module__', 'unknown')

            # If exclude_invalid_files is True, we need to determine if this is
            # a "real" invalid file error or a more serious issue (S3 permission, timeout, etc.)
            if exclude_invalid_files:
                # Check if this is an S3 permission or timeout error - these should NOT be silenced
                s3_critical_patterns = [
                    # Permission errors
                    'access denied',
                    'accessdenied',
                    'permission denied',
                    'forbidden',
                    '403',
                    'unauthorized',
                    '401',
                    # Timeout and connection errors
                    'timeout',
                    'timed out',
                    'curlcode: 28',
                    'connection reset',
                    'broken pipe',
                    'connection aborted',
                    'connection refused',
                    # Credential errors
                    'no credentials',
                    'credential',
                    'authentication',
                    # AWS specific errors
                    'aws error',
                    'service error',
                    'throttl'
                ]

                # Check for critical S3 error patterns (case-insensitive)
                is_s3_critical = any(pattern in error_msg
                                     for pattern in s3_critical_patterns)

                # Check for specific exception types that indicate serious issues
                is_critical_exception = isinstance(
                    e,
                    (
                        ClientError,  # boto3 S3 errors
                        NoCredentialsError,
                        ConnectTimeoutError,
                        ReadTimeoutError,
                        ConnectionError,
                        OSError,
                        IOError,  # Network and system errors
                        TimeoutError  # General timeout errors
                    ))

                # Check if the error is from botocore/boto3 (AWS SDK)
                is_aws_sdk_error = any(
                    module in error_module
                    for module in ['botocore', 'boto3', 's3fs', 'aiobotocore'])

                # Check for PyArrow errors that might indicate S3 issues
                is_pyarrow_error = 'pyarrow' in error_module
                is_s3_error = any(
                    pattern in error_msg
                    for pattern in ['aws error', 's3', 'connection', 'timeout'])
                is_pyarrow_s3_error = is_pyarrow_error and is_s3_error

                if (is_s3_critical or is_critical_exception or is_aws_sdk_error
                        or is_pyarrow_s3_error):
                    logger.error("[PARQUET_ERROR] CRITICAL ERROR in file "
                                 f"{file_path} - {e}")
                    raise e
                else:
                    logger.warning("[PARQUET_ERROR] Skipping invalid "
                                   f"parquet file {file_path} - {e}")
                    continue
            else:
                # exclude_invalid_files is False - always raise
                logger.error(f"[PARQUET_ERROR] Error in file {file_path} - {e}")
                raise e

    # Concatenate all relevant tables
    if not tables_to_concat:
        empty_schema = pyarrow.schema([])
        final_table = pyarrow.Table.from_arrays([], schema=empty_schema)

        if return_arrow:
            return final_table

        return final_table.to_pandas()

    if return_arrow:
        return pyarrow.concat_tables(tables_to_concat)

    parts = []
    for batch in tables_to_concat:
        df = batch.to_pandas()
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def head_parquet(path: str, max_rows: int = 1000) -> pyarrow.Table:
    """Read the first n rows of a parquet file or dataset from path.

    Args:
        path (str): The parquet file or dataset to read.
        max_rows (int): The maximum number of rows to read.

    Returns:
        pyarrow.Table: The first N rows of the parquet table.
    """
    if is_ucv(path):
        return get_parquet_head_databricks(path, max_rows)

    # If path is not an s3 path, s3a_as_s3 has no effect.
    path = fsutil.s3a_as_s3(path)
    with _get_dataset(path) as ds:
        num_rows = min(max_rows, ds.count_rows())
        return ds.head(num_rows)


@dataclass
class ParquetMetadata:
    cols: List[str]
    num_rows: int
    num_cols: int
    avg_row_size: int

    @property
    def decompressed_size_bytes(self) -> int:
        return self.avg_row_size * self.num_rows


def _cloud_parquet_file_num_rows(path: str) -> int:
    with fsutil.get_binary_handle(path, 'rb') as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def _is_empty_helper(path: str) -> bool:
    """Helper function to check if a parquet table is empty.

    Args:
        path (str): Path where the table to be checked is.

    Returns:
        bool: True if the table is empty, False otherwise.
    """
    # Normalize s3a:// → s3:// before the isdir branch: s3a has no isdir
    # handler, so an S3 directory URI would otherwise be opened as a single
    # file and mis-reported as empty. The isdir call keeps the pre-existing
    # "return True on lookup failure" semantic for missing paths.
    path = fsutil.s3a_as_s3(path)
    try:
        is_dir = fsutil.isdir(path)
    except Exception as e:
        # An access/IO failure here means the table is unreadable, not empty;
        # surface it. Missing/unparseable paths stay "empty".
        if _is_access_or_io_error(e):
            raise
        return True
    if not is_dir:
        try:
            if is_ucv(path):
                return get_num_rows_databricks(path) == 0
            elif fsutil.is_s3(path) or fsutil.is_gcs(path):
                return _cloud_parquet_file_num_rows(path) == 0
            else:
                with _get_dataset(path) as ds:
                    return ds.count_rows() == 0
        except Exception as e:
            # A missing file is empty; an access/IO error means we could not
            # read it and must surface rather than report empty.
            if _is_access_or_io_error(e):
                raise
    else:
        if is_ucv(path) and get_num_rows_databricks(path) > 0:
            return False
        files = fsutil.listdir(path)
        if fsutil.is_s3(path) or fsutil.is_gcs(path):
            for file in files:
                if (file.endswith('.parquet')
                        or fsutil.isdir(file)) and not (_is_empty_helper(file)):
                    return False
            return True
        for file in files:
            try:
                with _get_dataset(file) as ds:
                    if ds.count_rows() > 0:
                        return False
            except Exception as e:
                # Ignore transit / __SUCCESS__ noise, but surface an access/IO
                # failure instead of mis-reporting the table as empty.
                if _is_access_or_io_error(e):
                    raise
    return True


@blocking_func()
def is_empty(path: str) -> bool:
    """Return true if the given parquet dataset is empty.
    """
    # For customers like roco running frequent BP jobs, occasionally
    # pyarrow.dataset.dataset results in the following AWS Error:
    # "AWS Error [code 99]: curlCode: 28, Timeout was reached".
    # Once this error occurs, the subsequent BP jobs fail with the same
    # error - possibly due to S3FS caching underneath. To bypass this
    # issue, we run is_empty in a separate process.
    # Refer - https://github.com/kumo-ai/kumo/issues/12275#issuecomment-2127757914 # noqa: E501
    if fsutil.is_s3(path):
        return _run_in_isolated_subprocess(_is_empty_helper, (path, ))
    else:
        # GCS and UCV stay in-process. Forking a threaded process after a GCS
        # filesystem has been initialized can wedge PyArrow, while the
        # poisoned aws-sdk-cpp client this isolation addresses is S3-only.
        # UCV isolation would also cause frequent Databricks authentication.
        return _is_empty_helper(path)


@blocking_func(tracing_options=TraceOptions(trace_params=True))
def get_parquet_metadata(path: str) -> ParquetMetadata:
    """Get the shape of a parquet file or dataset.

    Args:
        path (str): The parquet file or dataset to read.

    Returns:
        ParquetMetadata: The metadata of the parquet table.
    """
    path = fsutil.s3a_as_s3(path)
    if fsutil.is_ucv(path):
        return _get_parquet_metadata_using_warehouse(path)
    with _get_dataset(path) as ds:
        return _get_parquet_metadata(ds)


def _get_parquet_metadata_using_warehouse(path: str) -> ParquetMetadata:
    """Get the metadata of a parquet table using warehouse compute.
    Exclusively used for UCV paths.

    Args:
        path (str): The parquet file(s) to read.

    Returns:
        ParquetMetadata: The metadata of the parquet table.
    """
    if is_ucv(path):
        rows = get_num_rows_databricks(path)
        pyarrow_table = get_parquet_head_databricks(path)
    else:
        raise ValueError(f"Invalid path: {path}. Expected UCV path.")

    cols = pyarrow_table.schema.names
    columns = pyarrow_table.to_pandas().columns.size
    # avoid divide by zero if no rows
    if rows > 0:
        avg_row_size = pyarrow_table.nbytes / min(1000, rows)  # sample_size
    else:
        avg_row_size = 0

    return ParquetMetadata(
        cols=cols,
        num_rows=rows,
        num_cols=columns,
        avg_row_size=avg_row_size,
    )


def _get_parquet_metadata(
        ds: pyarrow.dataset.Dataset,  # type: ignore
) -> ParquetMetadata:
    """Get the metadata of a parquet table using pyarrow.dataset.Dataset.
    Exclusively used for paths that are not from databricks or snowflake.

    Args:
        ds (pyarrow.dataset.Dataset): The dataset to read.

    Returns:
        ParquetMetadata: The metadata of the parquet table.
    """
    cols = ds.schema.names
    rows = ds.count_rows()
    # Sampling decodes data pages, and PyArrow may decode or prefetch complete
    # column chunks. Wide rows can therefore require far more memory than the
    # requested row count suggests; metadata-only callers should use footers.
    columns = ds.head(10).to_pandas().columns.size
    # avoid divide by zero if no rows
    if rows > 0:
        sample_size = min(1000, rows)
        avg_row_size = ds.head(sample_size).nbytes / sample_size
    else:
        avg_row_size = 0

    return ParquetMetadata(
        cols=cols,
        num_rows=rows,
        num_cols=columns,
        avg_row_size=avg_row_size,
    )


@dataclass
class ParquetBatch:
    df: pd.DataFrame
    batch_number: int
    batch_size_bytes: int


# We benchmarked this parameter on read & write of 10 parquet files each with
# an uncompressed size of 1GB (with avg row size ~28.5 bytes), and found out
# that in general smaller batch size (a few MB) performs much better in terms
# of peak memory footprint:
#
# 100MB -> 6GB (peak)
# 20MB  -> 3.4GB (peak)
# 10MB  -> 2.8GB (peak)
# 3.6MB -> 2.4GB (peak)
# 2MB   -> 1.7GB (peak)
# 1MB   -> 1.4GB (peak)
_DEFAULT_BATCH_BYTESIZE = 2 << 20  # 2MB


def to_batches(
    path: str, target_bytes_per_batch: int = _DEFAULT_BATCH_BYTESIZE
) -> Iterator[ParquetBatch]:
    """Function optimized for iterating through pandas dataframes of a parquet
    dataset on a memory-constrained host.

    "target_bytes_per_batch" can be used to specify the target size per batch
    in order to restrain memory footprint. Note that each batch returned will
    NOT have the exact target bytesize, but with maximum row count being target
    size divided by avg row size (sampled on 1000 head rows for large dataset).
    We DO NOT recommend setting this parameter larger than
    _DEFAULT_BATCH_BYTESIZE due to the concern of high memory footprint as well
    as potential memory leak.
    """
    # PyArrow does not recognize s3a:// (Spark/Hadoop scheme); normalize once
    # up front so the single-file, directory, and wildcard branches below all
    # see an s3:// scheme. The wildcard branch reattaches the scheme onto this
    # outer path, so it must be normalized here and not only inside
    # _get_dataset.
    path = fsutil.s3a_as_s3(path)

    if target_bytes_per_batch > 5 * _DEFAULT_BATCH_BYTESIZE:
        # TODO(siyang): Consider changing this to assert or raise.
        logger.warning(
            'Reading parquet to pandas shards with too large size: %d,'
            'which could lead to potential memory leaks!',
            target_bytes_per_batch)

    # Compute batch sizing and resolve the concrete files to scan up front from
    # a short-lived metadata dataset, then stream each file through its own
    # _get_dataset below. A single long scan holding one filesystem can outlive
    # the frozen-credential TTL and 403 mid-scan; re-entering _get_dataset per
    # file re-acquires a filesystem with refreshed credentials.
    with _get_dataset(path, exclude_invalid_files=True) as ds:
        info = _get_parquet_metadata(ds)
        logger.info("Parquet metdata for %s: %s", path, info)

        if info.avg_row_size == 0:
            logger.error('Error reading empty size parquet dataset: %s', path)
            return

        # Max row count per the batch.
        from kumo_core.artifact_export import artifact_rows_per_batch
        rows_per_batch = artifact_rows_per_batch(info.avg_row_size,
                                                 target_bytes_per_batch)

        # Resolve the files to scan from the metadata dataset's own fragments
        # (recursive, name-agnostic - matches the previous single-dataset scan)
        # so a nested/partitioned layout is never silently truncated. Re-attach
        # the scheme so each path re-enters _get_dataset (and
        # arrow_cloud_filesystem). Sorted for deterministic output order.
        if path.endswith("*") or fsutil.isdir(path):
            scheme = ("s3://" if path.startswith("s3://") else
                      "gs://" if path.startswith("gs://") else "")
            files = sorted(scheme + f.path for f in ds.get_fragments())
        else:
            files = [path]

    if not files:
        raise RuntimeError(f'No parquet file found in {path}')

    logger.info(
        'Scanning parquet dataset %s: max %d rows per batch across '
        '%d files.', path, rows_per_batch, len(files))

    batch_number = 0
    for file_path in files:
        with _get_dataset(file_path, exclude_invalid_files=True) as file_ds:
            for batch in file_ds.to_batches(
                    batch_size=rows_per_batch,
                    use_threads=False,
            ):
                yield ParquetBatch(df=batch.to_pandas(),
                                   batch_number=batch_number,
                                   batch_size_bytes=batch.nbytes)
                batch_number += 1


@contextmanager
def _parquet_write_target(path: str):
    if fsutil.is_local(path):
        yield path
    elif fsutil.is_s3(path) or fsutil.is_gcs(path):
        with fsutil.get_binary_handle(path, 'wb') as handle:
            yield handle
    else:
        with fsutil.TemporaryDirectory() as temp_dir:
            temp_file = osp.join(temp_dir, "temp_file")
            yield temp_file
            fsutil.copy_file(temp_file, path, overwrite=True)


class StreamingDataFrameParquetWriter:
    """Incrementally write dataframes to bounded Parquet files.

    Each input dataframe is sliced before Arrow conversion, so neither the
    Arrow table nor an output file grows beyond the configured in-memory/file
    target under normal row-size estimates. The schema from the first slice is
    reused for every later slice and rotated file.
    """
    def __init__(
        self,
        path_prefix: str,
        *,
        max_file_size: int,
        max_buffer_size: int,
        save_index: bool,
    ) -> None:
        if max_file_size <= 0 or max_buffer_size <= 0:
            raise ValueError('Parquet stream size limits must be positive')
        self.path_prefix = fsutil.s3a_as_s3(path_prefix)
        self.max_file_size = max_file_size
        self.max_buffer_size = min(max_buffer_size, max_file_size)
        self.save_index = save_index
        self._schema: pyarrow.Schema | None = None
        self._writer: pq.ParquetWriter | None = None
        self._target_stack: ExitStack | None = None
        self._current_file_size = 0
        self._next_file_index = 0
        self._index_offset = 0
        self._closed = False

    @property
    def files_written(self) -> int:
        return self._next_file_index

    def _close_file(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            self._writer = None
            if self._target_stack is not None:
                self._target_stack.close()
                self._target_stack = None
            self._current_file_size = 0

    def _open_file(self) -> None:
        assert self._schema is not None
        output_path = (
            f'{self.path_prefix}-{self._next_file_index:05d}.parquet')
        self._next_file_index += 1
        self._target_stack = ExitStack()
        target = self._target_stack.enter_context(
            _parquet_write_target(output_path))
        self._writer = pq.ParquetWriter(
            target,
            self._schema,
            flavor='spark',
            use_deprecated_int96_timestamps=True,
        )

    def _to_table(self, df: pd.DataFrame) -> pyarrow.Table:
        if self.save_index:
            df = df.copy(deep=False)
            df.index = pd.RangeIndex(self._index_offset,
                                     self._index_offset + len(df))
            self._index_offset += len(df)
        table = pyarrow.Table.from_pandas(df, preserve_index=self.save_index)
        if self._schema is None:
            self._schema = table.schema
            return table
        if table.schema.names != self._schema.names:
            raise ValueError('Streaming Parquet schema columns changed from '
                             f'{self._schema.names} to {table.schema.names}')
        if not table.schema.equals(self._schema, check_metadata=False):
            table = table.cast(self._schema.remove_metadata())
        return table.replace_schema_metadata(self._schema.metadata)

    def _write_table(self, table: pyarrow.Table) -> None:
        table_size = max(1, table.nbytes)
        if (self._writer is not None and self._current_file_size > 0
                and self._current_file_size + table_size > self.max_file_size):
            self._close_file()
        if self._writer is None:
            self._open_file()
        assert self._writer is not None
        self._writer.write_table(
            table,
            row_group_size=fsutil.PARQUET_ROW_GROUP_SIZE,
        )
        self._current_file_size += table_size

    def write(self, df: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError('Cannot write to a closed Parquet stream')
        if df.empty:
            return

        memory_usage = max(1, int(df.memory_usage(deep=True).sum()))
        rows_per_buffer = max(
            1,
            int(len(df) * self.max_buffer_size / memory_usage),
        )
        for start in range(0, len(df), rows_per_buffer):
            chunk = df.iloc[start:start + rows_per_buffer]
            self._write_table(self._to_table(chunk))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_file()

    def __enter__(self) -> 'StreamingDataFrameParquetWriter':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_in_chunks(
    df: pd.DataFrame,
    path_prefix: str,
    max_chunk_size: int,
    save_index: bool = True,
):
    """
    Write pandas dataframe to Parquet files in chunks of size `max_chunk_size`

    Args:
        df (pandas.DataFrame): A DataFrame containing the data to be written.
        path_prefix (str): The path where the parquet files will be written.
            E.g. If set to '/tmp/output' the output files will be written as
            '/tmp/output-00000.parquet', '/tmp/output-00001.parquet', ...
        max_chunk_size (int): The maximum size of each chunk in bytes.
        save_index (bool): Whether to save the index of the DataFrame.
            Default :obj:`True`.
    """
    # Ensure the directory exists in case of non-s3/gcs paths:
    _, mapped_prefix = _rewrite_uri(path_prefix)
    if (not fsutil.is_s3(path_prefix) and not fsutil.is_s3(mapped_prefix)
            and not fsutil.is_gcs(path_prefix)
            and not fsutil.is_gcs(mapped_prefix)):
        assert fsutil.exists(osp.dirname(path_prefix))
    # Compute the number of chunks to split the dataframe into:
    # NOTE: We use dataframe.memory_usage function to get the size of the
    # dataframe and this might be smaller than the actual size of the output
    # parquet file. This is okay because it's alright to produce smaller files
    # but not larger files than expected.
    memory_usage = df.memory_usage(deep=True).sum()
    num_rows = len(df)
    num_chunks = max(1, math.ceil(memory_usage / max_chunk_size))
    chunk_size = int(math.ceil(num_rows / num_chunks))
    logger.info(f"Splitting parquet file into {num_chunks} chunks of size: "
                f"{chunk_size} rows")

    # Write the dataframe:
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_rows)
        chunk_df = df.iloc[start:end]
        output_path = path_prefix + f"-{i:05d}.parquet"
        logger.debug(
            f"Writing {i+1} of {num_chunks} chunks to path: {output_path}")
        with _parquet_write_target(output_path) as target:
            chunk_df.to_parquet(
                target,
                index=save_index,
                row_group_size=fsutil.PARQUET_ROW_GROUP_SIZE,
                flavor="spark",
            )


def write_df(df: pd.DataFrame, path: str):
    """
    Write pandas dataframe to a Parquet file.

    Args:
        df (pandas.DataFrame): A DataFrame containing the data to be written.
        path (str): The path where the parquet files will be written.
    """
    # PyArrow doesn't recognize s3a:// (Spark/Hadoop scheme); convert to s3://.
    # Mirrors the read-path fix at _get_dataset.
    path = fsutil.s3a_as_s3(path)
    if path.endswith('/') or fsutil.isdir(path):
        raise ValueError(f"expected filename, got directory {path}")
    table = pyarrow.Table.from_pandas(df)
    write_options = {
        'use_deprecated_int96_timestamps': True,
        'flavor': 'spark',
    }
    with _parquet_write_target(path) as target:
        pq.write_table(table, target, **write_options)


# For customers like roco running frequent BP jobs, occasionally
# pyarrow.dataset.dataset results in the following AWS Error:
# "AWS Error [code 99]: curlCode: 28, Timeout was reached".
# Once this error occurs, the subsequent BP jobs fail with the same
# error - possibly due to S3FS caching underneath. To bypass this
# issue, we run parquet_num_rows in a separate process.
# Refer - https://github.com/kumo-ai/kumo/issues/12275#issuecomment-2127757914 # noqa: E501
def _parquet_num_rows_helper(table_path, skip_check) -> int:
    """Helper function to get the number of rows in a parquet table.

    Args:
        table_path (str): Path where the table to be checked is; may be a
            parquet directory or single file.
        skip_check (bool): If `True`, the caller assume the table path is
            a valid parquet path and we don't need to check exists, is dir,
            and files in the directory. Exception will return 0. Otherwise,
            check the table path.

    Returns:
        int: The number of rows in the parquet table.
    """
    # Note that this is called even if table_path is not an s3 path. Nothing is
    # changed if this is not an s3 path.
    # get_parquet_metadata doesn't work on s3a
    table_path = fsutil.s3a_as_s3(table_path)

    if skip_check:
        # Fast path (see _count_rows_per_file). An access/IO error or a corrupt
        # fragment surfaces; only a genuinely empty/absent path stays "0 rows".
        try:
            if is_ucv(table_path):
                return get_num_rows_databricks(table_path)
            return _count_rows_per_file(table_path)
        except Exception as e:
            if _is_access_or_io_error(e) or isinstance(e, pyarrow.ArrowInvalid):
                logger.error(
                    "Failed to read num rows for %s (access/IO error or "
                    "corrupt fragment); surfacing instead of reporting 0 "
                    "rows: %s", table_path, e)
                raise
            logger.warning("Treating num rows for %s as 0 (empty/missing): %s",
                           table_path, e)
            return 0

    if not fsutil.exists(table_path):
        return 0

    if is_ucv(table_path):
        return get_num_rows_databricks(table_path)

    return _count_rows_per_file(table_path)


def _count_rows_per_file(table_path: str) -> int:
    """Count rows in a single parquet file or a directory of ``.parquet``
    fragments at ``table_path``.

    The directory case counts fragments discovered recursively (partitioned
    layouts included), filtered to ``.parquet`` by extension so non-parquet
    sidecars are skipped.
    """
    is_dir = fsutil.isdir(table_path)
    if fsutil.is_s3(table_path) or fsutil.is_gcs(table_path):
        if not is_dir:
            return _cloud_parquet_file_num_rows(table_path)
        # Flatten the (possibly partitioned) tree to leaf .parquet fragments via
        # cheap one-level LISTs, then read their footers concurrently. The reads
        # are independent, I/O-bound round-trips; a serial sum over thousands of
        # parts blows the subprocess timeout. One pool over all leaves bounds
        # concurrency regardless of partition depth.
        fragments: List[str] = []
        pending = [table_path]
        while pending:
            for path in fsutil.listdir(pending.pop()):
                # Directory-first: an object-store "directory" is only a key
                # prefix, so a partition dir can itself end in .parquet.
                # Classify by isdir before extension so such a dir recurses
                # rather than being mis-read as a single footer.
                if fsutil.isdir(path):
                    pending.append(path)
                elif path.endswith('.parquet'):
                    fragments.append(path)
        if not fragments:
            return 0
        max_workers = min(_NUM_ROWS_MAX_WORKERS.parse_value(), len(fragments))
        if max_workers <= 1:
            return sum(_cloud_parquet_file_num_rows(p) for p in fragments)
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix='parquet_num_rows_') as ex:
            return sum(ex.map(_cloud_parquet_file_num_rows, fragments))

    if not is_dir:
        with _get_dataset(table_path) as ds:
            return ds.count_rows()

    # Rebuild the dataset from the explicit .parquet paths WITHOUT
    # exclude_invalid_files on purpose: a corrupt fragment must raise, not be
    # silently dropped and undercount the table.
    with _get_dataset(table_path) as ds:
        parquet_paths = [
            fragment.path for fragment in ds.get_fragments()
            if fragment.path.endswith('.parquet')
        ]
        if not parquet_paths:
            return 0
        rebuilt = pyarrow.dataset.dataset(parquet_paths, format='parquet',
                                          filesystem=ds.filesystem)
        return rebuilt.count_rows()


@blocking_func(tracing_options=TraceOptions(trace_params=True))
def parquet_num_rows(table_path: str, skip_check: bool = False,
                     use_new_process: bool = True) -> int:
    """
    Returns row count, or zero if the table file doesn't exist at all.

    Args:
        table_path (str): Path where the table to be checked is.
        skip_check (bool): If `True`, the caller assume the table path is
            a valid parquet path and we don't need to check exists, is dir,
            and files in the directory. Exception will return 0. Otherwise,
            check the table path.
        use_new_process (bool): If `True`, run the function in a separate
            process. This is useful to avoid AWS Error [code 99]: curlCode: 28,
            Timeout was reached error for some customers like roco.
            Set this to `False` when using this on a local filesystem.

    Returns:
        int: The number of rows in the parquet table.
    """
    if not use_new_process or not fsutil.is_s3(table_path):
        return _parquet_num_rows_helper(table_path, skip_check)
    return _run_in_isolated_subprocess(_parquet_num_rows_helper,
                                       (table_path, skip_check))
