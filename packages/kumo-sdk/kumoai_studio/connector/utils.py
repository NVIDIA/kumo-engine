# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import io
import math
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from logging import getLogger
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from urllib.parse import urlparse, urlsplit, urlunsplit

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kumoapi.data_source import (
    CompleteFileUploadRequest,
    DeleteUploadedFileRequest,
    PartUploadMetadata,
    StartFileUploadRequest,
    StartFileUploadResponse,
    UploadFileType,
)
from tqdm import tqdm

from kumoai_studio import global_state
# still used for server-side completion retries
from kumoai_studio.exceptions import HTTPException
from kumoai_studio.futures import _KUMO_EVENT_LOOP

# -------------------
# Constants & Globals
# -------------------
logger = getLogger(__name__)

CHUNK_SIZE = 100 * 10**6  # 100 MB (legacy local single-file chunk)
READ_CHUNK_BYTES = 8 * 1024**2  # 8 MiB remote read buffer
UPLOAD_CHUNK_BYTES = 8 * 1024**2  # 8 MiB streamed PUT sub-chunks
MAX_UPLOAD_PARTS = 1000
MAX_LOCAL_FILE_SIZE = 100 * 1024**3
MAX_REMOTE_FILE_SIZE = 1000 * 1024**2
# Retained for deprecated partition helpers; the upload path no longer calls
# them. Remove the helpers and alias in a separate cleanup.
MAX_PARTITION_SIZE = MAX_REMOTE_FILE_SIZE

_TQDM_LOCK = threading.Lock()
_HTTP_SCHEMES = frozenset({'http', 'https'})


# ---------------
# Small utilities
# ---------------
def _fmt_bytes(n: int) -> str:
    value = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} EiB"


def _local_upload_part_size(file_size_bytes: int) -> int:
    minimum = math.ceil(file_size_bytes / MAX_UPLOAD_PARTS)
    if minimum <= CHUNK_SIZE:
        return CHUNK_SIZE
    return math.ceil(minimum / UPLOAD_CHUNK_BYTES) * UPLOAD_CHUNK_BYTES


def _fmt_secs(s: float) -> str:
    if s < 1:
        return f"{s*1000:.0f} ms"
    return f"{s:.2f} s"


def _fmt_rate(nbytes: int, secs: float) -> str:
    if secs <= 0:
        return "-"
    return f"{(nbytes / secs) / 1024**2:.1f} MB/s"


def _short_path(p: str, maxlen: int = 60) -> str:
    if len(p) <= maxlen:
        return p
    try:
        parsed = urlparse(p)
        head = f"{parsed.scheme}://"
        tail = p[-40:]
        return f"{head}…{tail}"
    except Exception:
        return f"…{p[-maxlen:]}"


def _safe_bar_update(bar: tqdm, inc: int) -> None:
    with _TQDM_LOCK:
        try:
            bar.update(inc)
        except Exception:
            pass


def _log_file_timing(label: str, path: str, size: int, tread: float,
                     tval: float, tupl: float) -> None:
    logger.debug("[%s] %s (%s) | read=%s @ %s | validate=%s | upload=%s @ %s",
                 label, path, _fmt_bytes(size), _fmt_secs(tread),
                 _fmt_rate(size, max(tread, 1e-6)), _fmt_secs(tval),
                 _fmt_secs(tupl), _fmt_rate(size, max(tupl, 1e-6)))


# -----------------------
# Async upload primitives
# -----------------------
def _http_origin(url: str) -> Tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in _HTTP_SCHEMES or not parsed.netloc:
        raise ValueError('Upload URLs must use absolute HTTP(S) origins')
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError('Upload URL has an invalid network location') from exc
    if not hostname:
        raise ValueError('Upload URL has an invalid network location')
    return scheme, hostname.casefold(), port or (443
                                                 if scheme == 'https' else 80)


def _api_base_url() -> str:
    api_url = global_state.client._url
    scheme, hostname, port = _http_origin(api_url)
    parsed = urlsplit(api_url)
    if parsed.query or parsed.fragment:
        raise ValueError('Kumo API URL cannot contain a query or fragment')
    host = f'[{hostname}]' if ':' in hostname else hostname
    default_port = 443 if scheme == 'https' else 80
    netloc = host if port == default_port else f'{host}:{port}'
    base_path = parsed.path.rstrip('/')
    return urlunsplit((scheme, netloc, f'{base_path}/', '', ''))


def _resolve_upload_url(url: str) -> str:
    """Resolve relative upload URLs below the configured Kumo API path."""
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError(
            'Upload URL must be a non-empty URL without whitespace')
    parsed = urlsplit(url)
    if parsed.scheme:
        _http_origin(url)
        return url
    if parsed.netloc or url.startswith('//'):
        raise ValueError('Scheme-relative upload URLs are not allowed')
    base = urlsplit(_api_base_url())
    base_path = base.path.rstrip('/')
    upload_path = parsed.path
    already_prefixed = (upload_path == base_path
                        or upload_path.startswith(f'{base_path}/'))
    if base_path and already_prefixed:
        resolved_path = upload_path
    else:
        resolved_path = '/'.join(part for part in (base_path,
                                                   upload_path.lstrip('/'))
                                 if part)
        if not resolved_path.startswith('/'):
            resolved_path = f'/{resolved_path}'
    resolved = urlunsplit((base.scheme, base.netloc, resolved_path,
                           parsed.query, parsed.fragment))
    _http_origin(resolved)
    return resolved


def _same_origin_upload_auth_headers(url: str) -> Dict[str, str]:
    """Return SDK auth only for uploads routed to the configured API origin."""
    if _http_origin(url) != _http_origin(global_state.client._url):
        return {}
    api_key = global_state.client._api_key
    if api_key:
        return {'X-API-Key': api_key}
    return {}


def _iter_memview_stream(
    mv: memoryview,
    subchunk_bytes: int,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Iterator[memoryview]:
    """Yield memoryview slices (zero-copy) for streaming PUT."""
    pos = 0
    n = mv.nbytes
    while pos < n:
        nxt = min(n, pos + subchunk_bytes)
        chunk = mv[pos:nxt]  # zero-copy slice
        pos = nxt
        if progress_cb:
            try:
                progress_cb(len(chunk))
            except Exception:
                pass
        yield chunk


async def _put_with_retry_streamed(
    session: aiohttp.ClientSession,
    url: str,
    mv: memoryview,
    part_no: int,
    subchunk_bytes: int = UPLOAD_CHUNK_BYTES,
    progress_cb: Optional[Callable[[int], None]] = None,
    retries: int = 3,
) -> Tuple[int, str]:
    """Stream a memoryview to a presigned URL using an *async* generator so
    aiohttp does not try to wrap it as multipart/form-data. We also set
    Content-Length explicitly so S3/GCS expects a fixed-size payload (avoids
    chunked TE).
    """

    # Build a fresh async generator per attempt (can't reuse after failure).
    def _make_async_gen() -> Callable[[], Any]:
        async def _agen() -> AsyncIterator[memoryview]:
            # Yield zero-copy memoryview slices; aiohttp can send memoryview
            # directly.
            for chunk in _iter_memview_stream(mv, subchunk_bytes, progress_cb):
                yield chunk
                # cooperative yield; keeps event loop snappy without extra
                # copies
                await asyncio.sleep(0)

        return _agen

    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(mv.nbytes),
    }
    url = _resolve_upload_url(url)
    headers.update(_same_origin_upload_auth_headers(url))

    attempt = 0
    while True:
        try:
            async with session.put(url, data=_make_async_gen()(),
                                   headers=headers) as res:
                # Read/consume response to free the connection
                _ = await res.read()
                if res.status != 200:
                    raise RuntimeError(
                        f"PUT failed {res.status}: {res.reason}")
                etag = res.headers.get("ETag") or res.headers.get("Etag") or ""
                return (part_no + 1, etag)
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            # backoff before retrying; generator will be recreated next loop
            await asyncio.sleep(0.5 * attempt)


async def multi_put_bounded(
    urls: List[str],
    data_iter: Generator[Union[bytes, memoryview], None, None],
    tqdm_bar_position: int = 0,  # kept for compatibility (unused)
    concurrency: int = 4,
    upload_progress_cb: Optional[Callable[[int], None]] = None,
    upload_subchunk_bytes: int = UPLOAD_CHUNK_BYTES,
) -> List[PartUploadMetadata]:
    """Multipart uploader with bounded concurrency and byte-accurate progress.
    No extra progress bar here; caller drives a single byte counter via
    upload_progress_cb.
    """
    results = cast(List[Union[Tuple[int, str], None]], [None] * len(urls))

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(
            ssl=False)) as session:

        async def worker(idx: int, url: str, chunk: Union[bytes,
                                                          memoryview]) -> None:
            mv = chunk if isinstance(chunk, memoryview) else memoryview(chunk)
            results[idx] = await _put_with_retry_streamed(
                session=session,
                url=url,
                mv=mv,
                part_no=idx,
                subchunk_bytes=upload_subchunk_bytes,
                progress_cb=upload_progress_cb,
            )

        pending: set[asyncio.Task] = set()

        try:
            for idx, url in enumerate(urls):
                if len(pending) >= concurrency:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED)
                    await asyncio.gather(*done)
                try:
                    chunk = next(data_iter)
                except StopIteration:
                    break
                pending.add(asyncio.create_task(worker(idx, url, chunk)))
            if pending:
                await asyncio.gather(*pending)
        except Exception:
            for t in pending:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

    out: List[PartUploadMetadata] = []
    for r in results:
        if r is None:
            continue
        out.append(PartUploadMetadata(r[0], r[1]))
    return out


def stream_read(
    f: io.BufferedReader,
    chunk_size: int,
) -> Generator[bytes, None, None]:
    r"""Streams ``chunk_size`` contiguous bytes from buffered reader ``f`` each
    time the generator is yielded from.
    """
    while True:
        byte_buf = f.read(chunk_size)
        if len(byte_buf) == 0:
            break
        yield byte_buf


def _validate_url_ext(url: str, file_type: Union[str, None]) -> str:
    """Validate that `url` ends with .csv or .parquet.  If `file_type` is
    given ("csv" or "parquet"), ensure it matches.  Returns the detected type
    ("csv" or "parquet"), else raises ValueError.
    """
    u = url.lower()
    detected = "csv" if u.endswith(".csv") else "parquet" if u.endswith(
        ".parquet") else None
    if detected is None:
        raise ValueError(f"File path '{url}' must end with .csv or .parquet")

    if file_type is None:
        return detected

    ft = file_type.lower()
    if ft not in ("csv", "parquet"):
        raise ValueError("file_type must be 'csv', 'parquet', or None")

    if ft != detected:
        raise ValueError(f"File path '{url}' must end with .{ft}")
    return detected


def upload_table(
    name: str,
    path: str,
    auto_partition: bool = True,
    partition_size_mb: int = 250,
    parallelism: Optional[int] = None,
    file_type: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    """Upload a CSV/Parquet table to Kumo from a local file or a remote path
    (s3://, gs://, abfs://, abfss://, az://).

    - Local file: uploaded as one logical table using multipart transport;
      files larger than 100 GiB are rejected.
    - Remote file: uploaded via multipart. Files >1000 MiB are rejected
      (re-shard to ~200 MiB and upload as a directory).
    - Remote directory: auto-detects format (or use `file_type`), validates
      each shard, and uploads in parallel with a memory-safe budget.

    Args:
        name: Destination table name in Kumo.
        path: Local path or remote URL to a .csv/.parquet file or directory.
        auto_partition: Deprecated compatibility option; ignored.
        partition_size_mb: Deprecated compatibility option; ignored.
        parallelism: Directory uploads concurrency override.
        file_type: Force "csv" or "parquet" for directories; None = auto-detect
        project_id: Exact Kumo project receiving the upload. ``None`` uses the
            active project.

    Raises:
        ValueError: Bad/mixed types, zero rows, an oversized local or remote
            file, schema/header mismatch, or invalid column names.
        ImportError: Missing filesystem dependency (s3fs/gcsfs/adlfs).
        RuntimeError: Remote stat/list/read or multipart completion failures.

    Notes:
        CSV headers are sanitized (chars → underscore, de-duped). Parquet
        columns must already be valid.
    """
    # Decide local vs remote by scheme
    scheme = urlparse(path).scheme
    if scheme in ("s3", "gs", "abfs", "abfss", "az"):
        return _upload_table_remote(
            name=name,
            path=path,
            parallelism=parallelism,
            file_type=file_type,
            project_id=project_id,
        )
    # Local path
    _validate_url_ext(path, file_type)
    file_size = os.path.getsize(path)
    if file_size > MAX_LOCAL_FILE_SIZE:
        raise ValueError(
            f"Local file {path} is {_fmt_bytes(file_size)}, which exceeds "
            f"the {_fmt_bytes(MAX_LOCAL_FILE_SIZE)} upload limit.")
    return _upload_single_file(name, path, project_id=project_id)


def _handle_duplicate_names(names: List[str]) -> List[str]:
    unique_names: List[str] = []
    unique_counts: dict[str, int] = {}
    for name in names:
        if name not in unique_names:
            unique_counts[name] = 0
            unique_names.append(name)
        else:
            unique_counts[name] += 1
            new_name = f"{name}_{unique_counts[name]}"
            while new_name in names or new_name in unique_names:
                unique_counts[name] += 1
                new_name = f"{name}_{unique_counts[name]}"
            unique_names.append(new_name)
    return unique_names


def _sanitize_columns(names: List[str]) -> Tuple[List[str], bool]:
    """Normalize column names in a CSV or Parquet file.

    Rules:
      - Replace any non-alphanumeric character with "_"
      - Strip leading/trailing underscores
      - Ensure uniqueness by appending suffixes: _1, _2, ...
      - Auto-name empty columns as auto_named_<n>

    Returns:
        (new_column_names, changed)
    """
    _SAN_RE = re.compile(r"[^0-9A-Za-z,\t]")
    # 1) Replace non-alphanumeric sequences with underscore
    new = [_SAN_RE.sub("_", n).strip("_") for n in names]

    # 2) Auto-name any empty column names to match UI behavior
    unnamed_counter = 0
    for i, n in enumerate(new):
        if not n:
            new[i] = f"auto_named_{unnamed_counter}"
            unnamed_counter += 1

    # 3) Ensure uniqueness (append suffixes where needed)
    new = _handle_duplicate_names(new)
    return new, new != names


def sanitize_file(src_path: str) -> Tuple[str, bool]:
    """Normalize column names in a CSV or Parquet file.

    Rules:
      - Replace any non-alphanumeric character with "_"
      - Strip leading/trailing underscores
      - Ensure uniqueness by appending suffixes: _1, _2, ...

    Returns (path, changed):
      - (src_path, False) if no changes were needed
      - (temp_path, True) if a sanitized temp file was written (caller must
        delete)
    """
    if src_path.endswith('.parquet'):
        pf = pq.ParquetFile(src_path)
        new_names, changed = _sanitize_columns(pf.schema.names)
        if not changed:
            return src_path, False
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet',
                                                delete=False)
        original_schema = pf.schema.to_arrow_schema()
        fields = [
            field.with_name(new_name)
            for field, new_name in zip(original_schema, new_names)
        ]
        sanitized_schema = pa.schema(fields)
        writer = pq.ParquetWriter(temp_file.name, sanitized_schema)
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i).rename_columns(new_names)
            writer.write_table(tbl)
        writer.close()
        return temp_file.name, True
    elif src_path.endswith('.csv'):
        cols = pd.read_csv(src_path, nrows=0).columns.tolist()
        new_cols, changed = _sanitize_columns(cols)
        if not changed:
            return src_path, False
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        tmp_path = tmp.name
        tmp.close()
        reader = pd.read_csv(src_path, chunksize=1_000_000)
        with open(tmp_path, 'w', encoding='utf-8', newline='') as out:
            out.write(','.join(new_cols) + '\n')
            for chunk in reader:
                chunk.columns = new_cols
                chunk.to_csv(out, header=False, index=False)
        return tmp_path, True
    else:
        raise ValueError(
            f"File {src_path} must be either a CSV or Parquet file.")


def _upload_single_file(
    name: str,
    path: str,
    tqdm_bar_position: int = 0,
    project_id: Optional[str] = None,
) -> None:
    if not (path.endswith(".parquet") or path.endswith(".csv")):
        raise ValueError(f"Path {path} must be either a CSV or Parquet file. "
                         "Partitioned data is not currently supported.")

    file_type = 'parquet' if path.endswith('parquet') else 'csv'
    source_path = path
    path, temp_file_created = sanitize_file(path)
    try:
        sz = os.path.getsize(path)
        if sz > MAX_LOCAL_FILE_SIZE:
            raise ValueError(
                f"Local file {source_path} is {_fmt_bytes(sz)}, which exceeds "
                f"the {_fmt_bytes(MAX_LOCAL_FILE_SIZE)} upload limit.")
        part_size = _local_upload_part_size(sz)
        if tqdm_bar_position == 0:
            logger.info("Uploading table %s (path: %s), size=%s bytes", name,
                        path, sz)

        upload_res = _start_table_upload(table_name=name, file_type=file_type,
                                         file_size_bytes=sz,
                                         part_size_bytes=part_size,
                                         project_id=project_id)

        urls = list(upload_res.presigned_part_urls.values())
        loop = _KUMO_EVENT_LOOP
        with open(path, 'rb') as stream:
            part_metadata_list_fut = asyncio.run_coroutine_threadsafe(
                multi_put_bounded(
                    urls=urls,
                    data_iter=stream_read(stream, part_size),
                    tqdm_bar_position=tqdm_bar_position,
                    concurrency=min(4, len(urls)),
                    upload_progress_cb=None,
                    upload_subchunk_bytes=UPLOAD_CHUNK_BYTES,
                ),
                loop,
            )
            part_metadata_list = part_metadata_list_fut.result()

        if tqdm_bar_position == 0:
            logger.info("Upload complete. Validating table %s.", name)
        for i in range(5):
            try:
                _complete_table_upload(
                    table_name=name,
                    file_type=file_type,
                    upload_path=upload_res.temp_upload_path,
                    upload_id=upload_res.upload_id,
                    parts_metadata=part_metadata_list,
                    project_id=project_id,
                )
            except HTTPException as e:
                # TODO(manan): this can happen when DELETE above has
                # not propagated. So we retry with delay here. We
                # assume DELETE is processed reasonably quickly:
                if e.status_code == 500 and i < 4:
                    time.sleep(2**(i - 1))
                    continue
                else:
                    raise e
            else:
                break

        if tqdm_bar_position == 0:
            logger.info("Completed uploading table %s to Kumo.", name)
    finally:
        if temp_file_created:
            os.unlink(path)


def _upload_partitioned_parquet(
    name: str,
    path: str,
    partition_size: int,
    project_id: Optional[str] = None,
) -> None:
    r"""Upload a large parquet file by partitioning it into smaller chunks."""
    logger.info("File %s is large, partitioning for upload...", path)
    pf = pq.ParquetFile(path)
    new_columns, _ = _sanitize_columns(pf.schema.names)

    partitions: List[Tuple[int, List[int]]] = []
    part_idx = 0
    current_size = 0
    current_row_groups: list[int] = []

    for rg_idx in range(pf.num_row_groups):
        rg_size = pf.metadata.row_group(rg_idx).total_byte_size
        if rg_size > MAX_PARTITION_SIZE:
            raise ValueError(
                f"Row group {rg_idx} is larger than the maximum partition size"
                f"{MAX_PARTITION_SIZE} bytes")
        if current_size + rg_size > partition_size and current_row_groups:
            partitions.append((part_idx, current_row_groups.copy()))
            part_idx += 1
            current_row_groups = []
            current_size = 0
        current_row_groups.append(rg_idx)
        current_size += rg_size
    if current_row_groups:
        partitions.append((part_idx, current_row_groups))

    logger.info("Splitting %s into %d partitions", path, len(partitions))

    def writer(path: str, row_groups: List[int]) -> None:
        original_schema = pf.schema.to_arrow_schema()
        fields = [
            field.with_name(new_name)
            for field, new_name in zip(original_schema, new_columns)
        ]
        sanitized_schema = pa.schema(fields)
        pq_writer = pq.ParquetWriter(path, sanitized_schema)
        for rg_idx in row_groups:
            tbl = pf.read_row_group(rg_idx).rename_columns(new_columns)
            pq_writer.write_table(tbl)
        pq_writer.close()

    _upload_all_partitions(partitions, name, ".parquet", writer,
                           project_id=project_id)
    logger.info("Upload complete. Validated table %s.", name)


def _upload_partitioned_csv(
    name: str,
    path: str,
    partition_size: int,
    project_id: Optional[str] = None,
) -> None:
    r"""Upload a large CSV file by partitioning it into smaller chunks."""
    partitions: List[Tuple[int, List[str]]] = []
    part_idx = 0
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    new_columns, _ = _sanitize_columns(columns)
    with open(path, 'r', encoding='utf-8') as f:
        _ = f.readline()
        header = ','.join(new_columns) + '\n'
        header_size = len(header.encode('utf-8'))
        current_lines = [header]
        current_size = header_size
        for line in f:
            line_size = len(line.encode('utf-8'))
            if (current_size + line_size > partition_size
                    and len(current_lines) > 1):
                partitions.append((part_idx, current_lines.copy()))
                part_idx += 1
                current_lines = [header]
                current_size = header_size
            current_lines.append(line)
            current_size += line_size
        if len(current_lines) > 1:
            partitions.append((part_idx, current_lines))

    logger.info("Splitting %s into %d partitions", path, len(partitions))

    def writer(path: str, lines: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    _upload_all_partitions(partitions, name, ".csv", writer,
                           project_id=project_id)
    logger.info("Upload complete. Validated table %s.", name)


def _upload_all_partitions(
    partitions: List[Tuple[int, Any]],
    name: str,
    file_suffix: str,
    writer: Callable[[str, Any], None],
    project_id: Optional[str] = None,
) -> None:
    with tqdm(partitions, desc=f"Uploading {name}", position=0) as pbar:
        for part_idx, partition_data in pbar:
            partition_desc = f"Part {part_idx+1}/{len(partitions)}"
            pbar.set_postfix_str(partition_desc)
            _create_and_upload_partition(
                name=name,
                part_idx=part_idx,
                file_suffix=file_suffix,
                partition_writer=writer,
                partition_data=partition_data,
                tqdm_bar_position=1,
                project_id=project_id,
            )


def _create_and_upload_partition(
    name: str,
    part_idx: int,
    file_suffix: str,
    partition_writer: Callable[[str, Any], None],
    partition_data: Any,
    tqdm_bar_position: int = 0,
    project_id: Optional[str] = None,
) -> None:
    r"""Create a partition file, upload it, and delete the local copy."""
    partition_name = (f"{name}{file_suffix}/"
                      f"part_{part_idx+1:04d}{file_suffix}")
    with tempfile.NamedTemporaryFile(suffix=file_suffix,
                                     delete=False) as temp_file:
        partition_path = temp_file.name

    try:
        partition_writer(partition_path, partition_data)
        _upload_single_file(partition_name, partition_path,
                            tqdm_bar_position=tqdm_bar_position,
                            project_id=project_id)
    finally:
        try:
            os.unlink(partition_path)
        except OSError:
            pass


def delete_uploaded_table(name: str, file_type: str,
                          project_id: Optional[str] = None) -> None:
    r"""Synchronously deletes a previously uploaded table from the Kumo data
    plane.

    .. code-block:: python

        import kumoai_studio
        from kumoai_studio.connector import delete_uploaded_table

        # Assume we have uploaded a `.parquet` table named `users`,
        # and we want to delete this table from Kumo:
        delete_uploaded_table(name="users", file_type="parquet")

        # Assume we have uploaded a `.csv` table named `orders`,
        # and we want to delete this table from Kumo:
        delete_uploaded_table(name="orders", file_type="csv")

    Args:
        name: The name of the table to be deleted. This table must have
            previously been uploaded with a call to
            :meth:`~kumoai_studio.connector.upload_table`.
        file_type: The file type of the table to be deleted; this can either
            be :obj:`"parquet"` or :obj:`"csv"`
        project_id: Exact project containing the upload. ``None`` uses the
            active project.
    """
    assert file_type in {'parquet', 'csv'}
    req = DeleteUploadedFileRequest(
        source_table_name=name,
        file_type=UploadFileType(file_type.upper()),
    )
    global_state.client.connector_api.delete_file_upload(
        req, project_id=project_id)
    logger.info("Successfully deleted table %s from Kumo.", name)


def replace_table(name: str, path: str, file_type: str,
                  project_id: Optional[str] = None) -> None:
    r"""Replaces an existing uploaded table on the Kumo data plane with a new
    table.

    .. code-block:: python

        import kumoai_studio
        from kumoai_studio.connector import replace_table

        # Replace an existing `.csv` table named `users`
        # with a new version located at `/data/new_users.csv`:
        replace_table(
            name="users",
            path="/data/new_users.csv",
            file_type="csv",
        )

    Args:
        name: The name of the table to be replaced. This table must have
            previously been uploaded with a call to
            :meth:`~kumoai_studio.connector.upload_table`.
        path: The full path of the new table to be uploaded, on the
            local machine.
        file_type: The file type of the table to be replaced; this
            can either be :obj:`"parquet"` or :obj:`"csv"`.
        project_id: Exact project containing the upload. ``None`` uses the
            active project.

    Raises:
        ValueError: If the specified path does not point to a valid
            `.csv` or `.parquet` file.
    """
    if not (path.endswith(".parquet") or path.endswith(".csv")):
        raise ValueError(f"Path {path} must be either a CSV or Parquet file. "
                         "Partitioned data is not currently supported.")
    try:
        logger.info("Deleting previously uploaded table %s of type %s.", name,
                    file_type)
        delete_uploaded_table(name=name, file_type=file_type,
                              project_id=project_id)
    except Exception:
        pass
    logger.info("Uploading table %s.", name)
    upload_table(name=name, path=path, project_id=project_id)
    logger.info("Successfully replaced table %s with the new table.", name)


def _start_table_upload(
    table_name: str,
    file_type: str,
    file_size_bytes: float,
    project_id: Optional[str] = None,
    part_size_bytes: int = CHUNK_SIZE,
) -> StartFileUploadResponse:
    if file_type not in {'csv', 'parquet'}:
        raise ValueError("file_type must be 'csv' or 'parquet'")
    req = StartFileUploadRequest(
        source_table_name=table_name,
        file_type=UploadFileType(file_type.upper()),
        num_parts=max(1, math.ceil(file_size_bytes / part_size_bytes)),
    )
    return global_state.client.connector_api.start_file_upload(
        req, project_id=project_id)


def _start_table_upload_with_parts(
    table_name: str,
    file_type: str,
    file_size_bytes: int,
    num_parts: int,
    project_id: Optional[str] = None,
) -> StartFileUploadResponse:
    if file_type not in {'csv', 'parquet'}:
        raise ValueError("file_type must be 'csv' or 'parquet'")
    req = StartFileUploadRequest(
        source_table_name=table_name,
        file_type=UploadFileType(file_type.upper()),
        num_parts=max(1, int(num_parts)),
    )
    return global_state.client.connector_api.start_file_upload(
        req, project_id=project_id)


def _complete_table_upload(
    table_name: str,
    file_type: str,
    upload_path: str,
    upload_id: str,
    parts_metadata: List[PartUploadMetadata],
    project_id: Optional[str] = None,
) -> None:
    if file_type not in {'csv', 'parquet'}:
        raise ValueError("file_type must be 'csv' or 'parquet'")
    req = CompleteFileUploadRequest(
        source_table_name=table_name,
        file_type=UploadFileType(file_type.upper()),
        temp_upload_path=str(upload_path),
        upload_id=str(upload_id),
        parts_metadata=parts_metadata,
        validate_data=True,
    )
    return global_state.client.connector_api.complete_file_upload(
        req, project_id=project_id)


# -----------------------
# Remote I/O (fsspec)
# -----------------------

# Define data type for filesystem that does not depend on fsspec
Filesystem = Any


def _make_filesystem(scheme: str) -> Filesystem:
    import fsspec

    if scheme == "s3":
        if find_spec("s3fs") is None:
            raise ImportError("S3 paths require 's3fs'. "
                              "Install: pip install s3fs")
        fs = fsspec.filesystem("s3")
    elif scheme == "gs":
        if find_spec("gcsfs") is None:
            raise ImportError("GCS paths require 'gcsfs'. "
                              "Install: pip install gcsfs")
        fs = fsspec.filesystem("gcs")
    elif scheme in ("abfs", "abfss", "az"):
        if find_spec("adlfs") is None:
            raise ImportError("Azure paths require 'adlfs'. "
                              "Install: pip install adlfs")
        fs = fsspec.filesystem(scheme)
    else:
        raise ValueError(f"Unsupported remote scheme: {scheme}")
    return fs


def _get_fs_and_path(url: str) -> Tuple[Filesystem, str]:
    parsed = urlparse(url)
    scheme = parsed.scheme
    fs = _make_filesystem(scheme)
    return fs, url


def _remote_info(fs: Filesystem, path: str) -> dict:
    try:
        info = fs.info(path)
        if info.get("type") in ("file", "directory"):
            return info
        # s3fs for directories can return {'Key':..., 'Size':...}; normalize
        if info.get("Size") is not None and info.get("Key"):
            return {
                "type": "file",
                "size": info.get("Size"),
                "name": info.get("Key")
            }
        return info
    except Exception as e:
        raise RuntimeError(f"Failed to stat remote path {path}: {e}")


def _remote_dir_manifest(fs: Filesystem, path: str) -> dict:
    # Return lists of parquet and csv entries with size
    try:
        listing = fs.ls(path, detail=True)
    except Exception as e:
        raise RuntimeError(f"Failed to list remote directory {path}: {e}")

    parquet_files: List[dict] = []
    csv_files: List[dict] = []
    for ent in listing:
        if isinstance(ent, dict):
            p = ent.get("name") or ent.get("Key") or ent.get("path")
            s = ent.get("size") or ent.get("Size") or 0
            t = ent.get("type") or ent.get("StorageClass") or ""
            if t == "directory":
                continue
        else:
            p = ent
            try:
                s = fs.info(p).get("size", 0)
            except Exception:
                s = 0
        if not isinstance(p, str):
            continue
        ext = os.path.splitext(p.lower())[1]
        if ext == ".parquet":
            parquet_files.append({"path": p, "size": int(s or 0)})
        elif ext == ".csv":
            csv_files.append({"path": p, "size": int(s or 0)})

    return {"parquet": parquet_files, "csv": csv_files}


def _read_remote_file_with_progress(
    fs: Filesystem,
    path: str,
    expected_size: Optional[int],
    update_bytes: Optional[Callable[[int], Optional[bool]]] = None,
    capture_first_line: bool = False,
) -> Tuple[io.BytesIO, memoryview, Optional[bytes]]:
    """Stream into a single BytesIO (one allocation) and return a zero-copy
    memoryview.
    """
    buf = io.BytesIO()

    header_line: Optional[bytes] = None
    if capture_first_line:
        header_acc = bytearray()
        seen_nl = False
    else:
        header_acc = bytearray()
        seen_nl = True

    with fs.open(path, "rb") as fobj:
        while True:
            chunk = fobj.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            if capture_first_line and not seen_nl:
                nl_idx = chunk.find(b"\n")
                if nl_idx != -1:
                    header_acc += chunk[:nl_idx]
                    # small copy only for header
                    header_line = bytes(header_acc)
                    seen_nl = True
                else:
                    header_acc += chunk
            buf.write(chunk)
            if update_bytes:
                try:
                    update_bytes(len(chunk))
                except Exception:
                    pass

    if capture_first_line and not seen_nl:
        header_line = bytes(header_acc)

    if sys.version_info >= (3, 13):
        mv = memoryview(buf.getvalue())
    else:
        mv = buf.getbuffer()  # zero-copy view of BytesIO internal buffer
    return buf, mv, header_line


# -----------------------
# Memory budget & helpers
# -----------------------
def _compute_mem_budget_bytes(files: List[dict]) -> int:
    # 50% of system RAM
    try:
        psutil = import_module('psutil')
        total = psutil.virtual_memory().total
    except Exception:
        total = 8 * 1024**3  # assume 8 GiB
    budget = int(total * 0.50)
    return max(budget, 512 * 1024**2)  # at least 512 MiB


class MemoryBudget:
    """A byte-level semaphore to prevent OOM when reading many shards."""
    def __init__(self, budget_bytes: int) -> None:
        self.budget = budget_bytes
        self.avail = budget_bytes
        self.cv = threading.Condition()

    def acquire(self, need: int) -> None:
        with self.cv:
            while self.avail < need:
                self.cv.wait(timeout=0.25)
            self.avail -= need

    def release(self, freed: int) -> None:
        with self.cv:
            self.avail += freed
            if self.avail > self.budget:
                self.avail = self.budget
            self.cv.notify_all()


def _determine_parallelism(files: List[dict], requested: Optional[int]) -> int:
    if requested is not None and requested > 0:
        return min(requested, len(files))
    env_par = os.getenv("KUMO_UPLOAD_PARALLELISM")
    if env_par:
        try:
            val = int(env_par)
            if val > 0:
                return min(val, len(files))
        except Exception:
            pass

    budget_bytes = _compute_mem_budget_bytes(files)
    # 128 MiB overhead by default
    try:
        overhead_bytes = max(0, int(os.getenv("KUMO_UPLOAD_OVERHEAD_MB",
                                              "128"))) * 1024**2
    except Exception:
        overhead_bytes = 128 * 1024**2

    needs = []
    for f in files:
        size = int(f.get("size") or 0)
        if size <= 0:
            continue
        needs.append(size + overhead_bytes)
    if not needs:
        return 1
    needs.sort()
    median_need = needs[len(needs) // 2]
    par = max(1, budget_bytes // max(1, median_need))
    return min(int(par), len(files))


def _iter_mv_chunks(mv: memoryview,
                    part_size: int) -> Generator[memoryview, None, None]:
    pos = 0
    n = mv.nbytes
    while pos < n:
        nxt = min(n, pos + part_size)
        if sys.version_info >= (3, 13):
            yield mv[pos:nxt].tobytes()
        else:
            yield mv[pos:nxt]  # zero-copy slice
        pos = nxt


# -----------------------
# Parquet helpers
# -----------------------
def _parquet_schema_from_bytes(data_mv: memoryview) -> pa.Schema:
    reader = pa.BufferReader(pa.py_buffer(data_mv))
    pf = pq.ParquetFile(reader)

    # zero-row guard via metadata (no data scan)
    if getattr(pf.metadata, "num_rows", None) == 0:
        raise ValueError("Parquet file contains zero rows.")

    return pf.schema_arrow


def _parquet_num_rows_from_bytes(data_mv: memoryview) -> int:
    buf = pa.py_buffer(data_mv)
    reader = pa.BufferReader(buf)
    pf = pq.ParquetFile(reader)
    md = pf.metadata
    if md is None:
        total = 0
        for rg in range(pf.num_row_groups):
            total += pf.metadata.row_group(rg).num_rows
        return total
    return md.num_rows


def validate_parquet_schema(schema: pa.Schema, source_name: str) -> None:
    """Validate a PyArrow schema for Kumo compatibility (source_name
    required).

    Disallowed:
      - All large_* types: large_string, large_binary, large_list<*>
      - Any time-of-day types (time32/64<*>); ONLY epoch-based timestamps are
        allowed
      - Any duration types (e.g., pa.duration('ns'))
      - list<string> and list<bool>
      - Unsigned integers (uint8/16/32/64)
      - Null-typed columns

    Allowed:
      - boolean, signed integer, floating, (regular) string, date, timestamp
        (epoch-based), (regular) binary
      - decimal up to configured precision (env KUMO_DECIMAL_MAX_PRECISION,
        default 18)
      - list of {signed integer, float}
      - dictionary<int, string>

    Raises:
      ValueError listing offending columns (including source_name).
    """
    try:
        max_dec_prec = int(os.getenv("KUMO_DECIMAL_MAX_PRECISION", "18"))
    except Exception:
        max_dec_prec = 18

    where = f" in {source_name}"
    errors: list[str] = []

    for col, dt in zip(schema.names, schema.types):
        # 1) Hard-disallow all large_* types
        if pa.types.is_large_string(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported type large_string")
            continue
        if pa.types.is_large_binary(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported type large_binary")
            continue
        if pa.types.is_large_list(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported type {dt} "
                f"(large_list not supported)")
            continue

        # 2) Disallow time-of-day and duration
        if pa.types.is_time(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported time-of-day type "
                f"'{dt}' (only epoch-based timestamps are supported)")
            continue
        if pa.types.is_duration(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported duration "
                f"type '{dt}'")
            continue

        # 3) Disallow unsigned integers and null columns
        if pa.types.is_unsigned_integer(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported unsigned integer "
                "type '{dt}'")
            continue
        if pa.types.is_null(dt):
            errors.append(
                f"  - column '{col}'{where} has unsupported null type '{dt}'")
            continue

        supported = (
            pa.types.is_boolean(dt)
            # signed ints only
            or (pa.types.is_integer(dt)
                and not pa.types.is_unsigned_integer(dt)) or
            pa.types.is_floating(dt) or
            pa.types.is_string(dt)  # regular string only
            or pa.types.is_date(dt) or
            pa.types.is_timestamp(dt)  # epoch-based timestamps
            or pa.types.is_binary(dt)  # regular binary only
        )

        # 4) Decimals with precision limit
        if not supported and pa.types.is_decimal(dt):
            try:
                prec = int(getattr(dt, "precision", 0) or 0)
            except Exception:
                prec = 0
            if 0 < prec <= max_dec_prec:
                supported = True
            else:
                errors.append(
                    f"  - column '{col}'{where} has unsupported decimal "
                    f"precision {prec} (max {max_dec_prec}): type '{dt}'")
                continue

        # 5) Lists: only list of {signed int, float}; explicitly deny
        # list<string> and list<bool>
        if not supported and pa.types.is_list(dt):
            elem = dt.value_type
            if pa.types.is_string(elem):
                errors.append(
                    f"  - column '{col}'{where} is {dt} (list<string> not "
                    f"supported)")
                continue
            if pa.types.is_boolean(elem):
                errors.append(f"  - column '{col}'{where} is {dt} (list<bool> "
                              f"not supported)")
                continue
            if pa.types.is_integer(
                    elem) and not pa.types.is_unsigned_integer(elem):
                supported = True
            elif pa.types.is_floating(elem):
                supported = True
            else:
                errors.append(
                    f"  - column '{col}'{where} is {dt} (only list of signed "
                    f"int/float supported)")
                continue

        # 6) Dictionary<int, string> only
        if not supported and pa.types.is_dictionary(dt):
            if (pa.types.is_integer(dt.index_type)
                    and not pa.types.is_unsigned_integer(dt.index_type)
                    and pa.types.is_string(dt.value_type)):
                supported = True

        if not supported:
            errors.append(
                f"  - column '{col}'{where} has unsupported type '{dt}'")

    if errors:
        raise ValueError(
            "Unsupported Parquet Data Types detected:\n\n" +
            "\n".join(errors) + "\n\nAllowed types: boolean, signed integer, "
            "float, (regular) string, date, "
            "timestamp (epoch-based), (regular) binary, "
            "decimal (<= configured precision), "
            "list of {signed int, float}, dictionary<int,string>.\n"
            "Disallowed examples: large_string, large_binary, "
            "large_list<*>, time32/64<*>, "
            "duration('unit'), list<string>, list<bool>, "
            "unsigned integers, null columns, "
            "structs, maps, and other nested types.")


# -----------------------
# CSV helpers
# -----------------------
def _detect_and_validate_csv(head_bytes: bytes) -> str:
    r"""Detect a CSV delimiter from a small head sample and verify it.

    - Uses csv.Sniffer (preferred delimiters: | , ; \t) with fallback to ','.
    - Reads a handful of complete, quote-aware records (handles newlines inside
      quotes).
    - Re-serializes those rows and validates with pandas (small nrows) to catch
      malformed inputs.
    - Raises ValueError on empty input or if parsing fails with the chosen
      delimiter.
    """
    if not head_bytes:
        raise ValueError("Could not auto-detect a delimiter: file is empty.")

    text = head_bytes.decode("utf-8", errors="ignore").replace("\r\n",
                                                               "\n").replace(
                                                                   "\r", "\n")

    # 1) Detect delimiter (simple preference list; no denylist)
    try:
        delimiter = csv.Sniffer().sniff(text, delimiters="|,;\t").delimiter
    except Exception:
        logger.warning("No separator found in sample; defaulting to ','.")
        delimiter = ','

    # 2) Pull a few complete records with csv.reader (quote-aware,
    # handles embedded newlines)
    rows = []
    try:
        rdr = csv.reader(io.StringIO(text), delimiter=delimiter, quotechar='"',
                         doublequote=True)
        for _ in range(50):  # small, bounded sample
            try:
                rows.append(next(rdr))
            except StopIteration:
                break
    except Exception as e:
        raise ValueError(
            f"Could not auto-detect a valid delimiter. Tried '{delimiter}', "
            f"csv parse failed: {repr(e)}")

    if not rows:
        raise ValueError(
            "Could not auto-detect a valid delimiter: no complete records "
            "found.")

    # 3) Re-serialize snippet and validate minimally with pandas
    out = io.StringIO()
    w = csv.writer(out, delimiter=delimiter, lineterminator="\n",
                   quotechar='"', doublequote=True)
    for r in rows:
        w.writerow(r)

    try:
        pd.read_csv(
            io.StringIO(out.getvalue()),
            sep=delimiter,
            index_col=False,
            on_bad_lines='error',
            nrows=50,
            engine="python",  # more tolerant for quoted/newline combos
            skip_blank_lines=False,
        )
    except Exception as e:
        raise ValueError(
            f"Could not auto-detect a valid delimiter. Tried '{delimiter}', "
            f"pandas parse failed: {repr(e)}")

    return delimiter


def _csv_has_data_rows(data_mv: memoryview) -> bool:
    """Return True if any non-newline, non-carriage-return byte exists after
    the first newline.  Uses zero-copy iteration over the memoryview to avoid
    duplicating buffers.
    """
    mv = data_mv
    if mv.format != 'B':
        try:
            mv = mv.cast('B')  # zero-copy view of bytes
        except TypeError:
            # fallback: create a contiguous view via slicing (still zero-copy)
            mv = mv[:]

    saw_newline = False
    # Iterate in a single pass; break as soon as we see a data-ish byte
    for b in mv:
        if not saw_newline:
            if b == 10:  # '\n'
                saw_newline = True
            continue
        # after header newline: any byte that isn't CR or LF counts as data
        if b not in (10, 13):
            return True
    return False


def _maybe_rewrite_csv_header_buffer(
    data_mv: memoryview,
    header_line: bytes,
    delimiter: str,
) -> tuple[Optional[io.BytesIO], memoryview, bytes, list[str], dict[str, str],
           bool]:
    """Rewrite ONLY the header if needed. Uses a new BytesIO but frees the old
    buffer immediately after swap.
    """
    try:
        header_str = header_line.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError:
        raise ValueError("CSV header is not valid UTF-8.")

    orig_cols = [c.strip() for c in header_str.split(delimiter)]
    new_cols, changed = _sanitize_columns(orig_cols)
    if not changed:
        return None, data_mv, header_line, orig_cols, {}, False

    rename_map = {o: n for o, n in zip(orig_cols, new_cols) if o != n}

    nl_idx = len(header_line)
    if nl_idx >= data_mv.nbytes:
        raise ValueError("Malformed CSV: newline not found in header.")

    new_header_bytes = delimiter.join(new_cols).encode("utf-8")
    new_buf = io.BytesIO()
    new_buf.write(new_header_bytes)
    new_buf.write(b"\n")
    # Write the remainder via a zero-copy memoryview slice; BytesIO will copy
    # into its own buffer, but we free the original immediately after returning
    # to avoid double residency.
    new_buf.write(data_mv[nl_idx + 1:])
    new_mv = new_buf.getbuffer()
    return new_buf, new_mv, new_header_bytes, new_cols, rename_map, True


# -----------------------
# Remote upload (refactor)
# -----------------------
@dataclass
class _RemoteSettings:
    part_size: int
    part_conc: int
    overhead_bytes: int
    parallelism_override: Optional[int]


def _make_remote_settings(parallelism: Optional[int]) -> _RemoteSettings:
    part_mb = int(os.getenv("KUMO_REMOTE_PART_MB", "64"))
    part_size = max(8, part_mb) * 1024**2
    part_conc = int(os.getenv("KUMO_REMOTE_PART_CONCURRENCY", "4"))
    try:
        overhead_bytes = max(0, int(os.getenv("KUMO_UPLOAD_OVERHEAD_MB",
                                              "128"))) * 1024**2
    except Exception:
        overhead_bytes = 128 * 1024**2
    return _RemoteSettings(
        part_size=part_size,
        part_conc=part_conc,
        overhead_bytes=overhead_bytes,
        parallelism_override=parallelism,
    )


def _remote_upload_file(
    name: str,
    fs: Filesystem,
    url: str,
    info: dict,
    st: _RemoteSettings,
    file_type: Optional[str],
    project_id: Optional[str] = None,
) -> None:
    detected_ftype = _validate_url_ext(url, file_type)

    size = int(info.get("size") or 0)
    if size == 0:
        raise ValueError(f"Remote file {url} is empty (0 bytes).")
    if size > MAX_REMOTE_FILE_SIZE:
        raise ValueError(
            "Remote single-file uploads larger than 1GB are not supported. "
            "Please re-partition the source into ~200MB chunks and upload the "
            "whole directory instead.")

    # Read with progress
    with tqdm(total=size, desc=f"Reading {_short_path(url)}", unit="B",
              unit_scale=True, unit_divisor=1024, position=0, leave=False,
              smoothing=0.1) as read_bar:
        tr0 = time.perf_counter()
        buf, data_mv, header_line = _read_remote_file_with_progress(
            fs, url, expected_size=size, update_bytes=read_bar.update,
            capture_first_line=(detected_ftype == "csv"))
        tread = time.perf_counter() - tr0

    # Validate/sanitize
    tv0 = time.perf_counter()
    renamed_cols_msg = None
    if detected_ftype == "parquet":
        schema = _parquet_schema_from_bytes(data_mv)
        _validate_columns_or_raise(list(schema.names))
        validate_parquet_schema(schema, url)
        nrows = _parquet_num_rows_from_bytes(data_mv)
        if nrows <= 0:
            raise ValueError("Parquet file has zero rows.")
        file_type = "parquet"
    else:
        head_len = min(50000, data_mv.nbytes)
        # small bounded copy only for sniffing
        head = bytes(data_mv[:head_len])
        delimiter = _detect_and_validate_csv(head)
        if header_line is None:
            # Shouldn't happen (we captured it during read), but keep a bounded
            # fallback (64 KiB)
            prefix_len = min(64 * 1024, data_mv.nbytes)
            prefix = data_mv[:prefix_len]
            # build header_line from prefix without large copies
            acc = bytearray()
            for b in (prefix.cast('B') if prefix.format != 'B' else prefix):
                if b == 10:  # '\n'
                    break
                acc.append(b)
            header_line = bytes(acc)
        new_buf, new_mv, new_header, cols, rename_map, changed = (
            _maybe_rewrite_csv_header_buffer(data_mv, header_line, delimiter))
        if changed:
            try:
                buf.close()
            except Exception:
                pass
        if changed:
            buf = new_buf
            data_mv = new_mv
            header_line = new_header
            if rename_map:
                pairs = ", ".join(f"{k}->{v}" for k, v in rename_map.items())
                renamed_cols_msg = f"CSV header sanitized (renamed): {pairs}"
        if not _csv_has_data_rows(data_mv):
            raise ValueError(
                "CSV file has zero data rows (only header present).")
        file_type = "csv"
    tval = time.perf_counter() - tv0

    # Multipart upload
    size_bytes = data_mv.nbytes
    num_parts = max(1, math.ceil(size_bytes / st.part_size))
    upload_res = _start_table_upload_with_parts(table_name=name,
                                                file_type=file_type,
                                                file_size_bytes=size_bytes,
                                                num_parts=num_parts,
                                                project_id=project_id)
    try:
        urls = [
            u for k, u in sorted(upload_res.presigned_part_urls.items(),
                                 key=lambda kv: int(kv[0]))
        ]
    except Exception:
        urls = list(upload_res.presigned_part_urls.values())

    loop = _KUMO_EVENT_LOOP
    with tqdm(total=size_bytes, desc="Uploading", unit="B", unit_scale=True,
              unit_divisor=1024, position=2, leave=False,
              smoothing=0.1) as upload_bar:
        part_metadata_list_fut = asyncio.run_coroutine_threadsafe(
            multi_put_bounded(
                urls=urls,
                data_iter=_iter_mv_chunks(data_mv, st.part_size),
                tqdm_bar_position=3,
                concurrency=max(1, min(st.part_conc, len(urls))),
                upload_progress_cb=lambda n: _safe_bar_update(upload_bar, n),
                upload_subchunk_bytes=UPLOAD_CHUNK_BYTES,
            ),
            loop,
        )
        part_metadata_list = part_metadata_list_fut.result()
        upload_bar.set_postfix_str(f"Done — {_short_path(url)}")
        upload_bar.refresh()

    # Complete
    tu0 = time.perf_counter()
    for i in range(5):
        try:
            _complete_table_upload(
                table_name=name,
                file_type=file_type,
                upload_path=upload_res.temp_upload_path,
                upload_id=upload_res.upload_id,
                parts_metadata=part_metadata_list,
                project_id=project_id,
            )
        except HTTPException as e:
            if e.status_code == 500 and i < 4:
                time.sleep(2**(i - 1))
                continue
            else:
                raise
        else:
            break
    tupl = time.perf_counter() - tu0

    _log_file_timing("single-file(multipart)", url, size_bytes, tread, tval,
                     tupl)
    if renamed_cols_msg:
        logger.info(renamed_cols_msg)

    try:
        if isinstance(data_mv, memoryview):
            data_mv.release()
    except Exception:
        pass

    try:
        if buf:
            buf.close()
    except Exception:
        pass

    logger.info("Upload complete. Validated table %s.", name)


def _remote_upload_directory(
    name: str,
    fs: Filesystem,
    url: str,
    info: dict,
    st: _RemoteSettings,
    file_type: Optional[str] = None,  # "csv", "parquet", or None
    project_id: Optional[str] = None,
) -> None:
    manifest = _remote_dir_manifest(fs, url)
    parquet_files = sorted(manifest["parquet"], key=lambda x: x["path"])
    csv_files = sorted(manifest["csv"], key=lambda x: x["path"])

    # Normalize expected type
    if file_type not in (None, "csv", "parquet"):
        raise ValueError("file_type must be 'csv', 'parquet', or None.")

    # Resolve files + detected type
    if file_type is None:
        if not parquet_files and not csv_files:
            raise ValueError("Directory contains no .parquet or .csv files.")
        if parquet_files and csv_files:
            raise ValueError(
                "Mixed CSV and Parquet files detected; keep only one format.")
        files = parquet_files if parquet_files else csv_files
        detected_type = "parquet" if parquet_files else "csv"
    elif file_type == "parquet":
        if not parquet_files:
            raise ValueError(
                "Directory contains no .parquet files (file_type='parquet').")
        if csv_files:
            raise ValueError(
                "Directory also contains CSV files; remove them or set"
                "file_type=None.")
        files, detected_type = parquet_files, "parquet"
    else:  # file_type == "csv"
        if not csv_files:
            raise ValueError(
                "Directory contains no .csv files (file_type='csv').")
        if parquet_files:
            raise ValueError(
                "Directory also contains Parquet files; remove them or "
                "set file_type=None.")
        files, detected_type = csv_files, "csv"

    total_bytes = sum(int(f.get("size") or 0) for f in files)

    too_large = [
        f["path"] for f in files if (f.get("size") or 0) > MAX_REMOTE_FILE_SIZE
    ]
    zero_bytes = [f["path"] for f in files if (f.get("size") or 0) == 0]
    if zero_bytes:
        raise ValueError(
            f"Found zero-byte {detected_type.upper()} files: {zero_bytes[:3]}"
            f"{'...' if len(zero_bytes)>3 else ''}")
    if too_large:
        raise ValueError(
            f"The following files exceed 1GB and must be re-partitioned "
            f"(~200MB each): "
            f"{too_large[:3]}{'...' if len(too_large)>3 else ''}")

    par = _determine_parallelism(files, requested=st.parallelism_override)
    par = max(1, min(par, len(files)))
    budget_bytes = _compute_mem_budget_bytes(files)
    mem_budget = MemoryBudget(budget_bytes)

    from collections import deque
    with (tqdm(total=len(files),
               desc=f"Files ({len(files)}) [{detected_type}] | par={par}",
               position=0) as file_bar,
          tqdm(total=total_bytes, desc="Total bytes (read)", unit="B",
               unit_scale=True, unit_divisor=1024, position=1, smoothing=0.1)
          as bytes_bar,
          tqdm(total=total_bytes, desc="Total bytes (uploaded)", unit="B",
               unit_scale=True, unit_divisor=1024, position=2, smoothing=0.1)
          as uploaded_bar):

        status_lock = threading.Lock()
        recent_paths: Deque[str] = deque(maxlen=5)
        completed_files = {"n": 0}
        file_bar.set_postfix_str(f"Uploaded 0/{len(files)}")
        file_bar.refresh()

        rename_aggregate_lock = threading.Lock()
        rename_aggregate: dict[str, str] = {}

        def _merge_status_update(path: str) -> None:
            with status_lock:
                completed_files["n"] += 1
                recent_paths.append(path)
                tail = ' | '.join(_short_path(p) for p in list(recent_paths))
                msg = f"Uploaded {completed_files['n']}/{len(files)}"
                if tail:
                    msg += f" — {tail}"
                with _TQDM_LOCK:
                    file_bar.set_postfix_str(msg)
                    file_bar.refresh()

        ref_schema_fields: Dict[str, Any] = {"value": None}
        ref_cols: Dict[str, Any] = {"value": None}

        def _worker(idx: int, fmeta: dict) -> None:
            fpath = fmeta["path"]
            fsize = int(fmeta.get("size") or 0)
            need_bytes = (2 * fsize +
                          st.overhead_bytes) if detected_type == "csv" else (
                              fsize + st.overhead_bytes)
            mem_budget.acquire(need_bytes)
            try:
                tr0 = time.perf_counter()
                buf, data_mv, header_line = _read_remote_file_with_progress(
                    fs,
                    fpath,
                    expected_size=fsize if fsize > 0 else None,
                    update_bytes=lambda n: _safe_bar_update(bytes_bar, n),
                    capture_first_line=(detected_type == "csv"),
                )
                tread = time.perf_counter() - tr0

                tv0 = time.perf_counter()
                if detected_type == "parquet":
                    schema = _parquet_schema_from_bytes(data_mv)
                    names = list(schema.names)
                    _validate_columns_or_raise(names)
                    validate_parquet_schema(schema, fpath)
                    nrows = _parquet_num_rows_from_bytes(data_mv)
                    if nrows <= 0:
                        raise ValueError(
                            f"Parquet file has zero rows: {fpath}")
                    fields = [(fld.name, fld.type) for fld in schema]
                    if ref_schema_fields["value"] is None:
                        ref_schema_fields["value"] = fields
                    elif fields != ref_schema_fields["value"]:
                        ref_names = [n for n, _ in ref_schema_fields["value"]]
                        raise ValueError(
                            "Parquet schema mismatch across files. "
                            f"First file columns: {ref_names}; mismatched "
                            f"file: {fpath}")
                    part_name = f"{name}.parquet/part_{idx:04d}.parquet"

                else:
                    head_len = min(50000, data_mv.nbytes)
                    # bounded small copy for sniffing
                    head = bytes(data_mv[:head_len])
                    delimiter = _detect_and_validate_csv(head)
                    if header_line is None:
                        # Bounded fallback (64 KiB) to extract header without
                        # copying whole file
                        prefix_len = min(64 * 1024, data_mv.nbytes)
                        prefix = data_mv[:prefix_len]
                        acc = bytearray()
                        for b in (prefix.cast('B')
                                  if prefix.format != 'B' else prefix):
                            if b == 10:  # '\n'
                                break
                            acc.append(b)
                        header_line = bytes(acc)

                    new_buf, new_mv, new_header, cols, rename_map, changed = (
                        _maybe_rewrite_csv_header_buffer(
                            data_mv, header_line, delimiter))
                    if changed:
                        try:
                            buf.close()
                        except Exception:
                            pass
                        buf = new_buf
                        data_mv = new_mv
                        header_line = new_header
                        if rename_map:
                            with rename_aggregate_lock:
                                rename_aggregate.update(rename_map)

                    if ref_cols["value"] is None:
                        ref_cols["value"] = cols
                    elif cols != ref_cols["value"]:
                        raise ValueError(
                            "CSV header mismatch across files. "
                            f"Expected: {ref_cols['value']}; mismatched file: "
                            f"{fpath} has: {cols}")
                    if not _csv_has_data_rows(data_mv):
                        raise ValueError(
                            f"CSV file has zero data rows: {fpath}")
                    part_name = f"{name}.csv/part_{idx:04d}.csv"
                tval = time.perf_counter() - tv0

                size_bytes = data_mv.nbytes
                num_parts = max(1, math.ceil(size_bytes / st.part_size))
                upload_res = _start_table_upload_with_parts(
                    table_name=part_name,
                    file_type=detected_type,
                    file_size_bytes=size_bytes,
                    num_parts=num_parts,
                    project_id=project_id,
                )
                try:
                    urls = [
                        u for k, u in sorted(
                            upload_res.presigned_part_urls.items(),
                            key=lambda kv: int(kv[0]))
                    ]
                except Exception:
                    urls = list(upload_res.presigned_part_urls.values())

                loop_inner = _KUMO_EVENT_LOOP
                part_metadata_list_fut = asyncio.run_coroutine_threadsafe(
                    multi_put_bounded(
                        urls=urls,
                        data_iter=_iter_mv_chunks(data_mv, st.part_size),
                        tqdm_bar_position=3,
                        concurrency=max(1, min(st.part_conc, len(urls))),
                        upload_progress_cb=lambda n: _safe_bar_update(
                            uploaded_bar, n),
                        upload_subchunk_bytes=UPLOAD_CHUNK_BYTES,
                    ),
                    loop_inner,
                )
                part_metadata_list = part_metadata_list_fut.result()

                for i in range(5):
                    try:
                        _complete_table_upload(
                            table_name=part_name,
                            file_type=detected_type,
                            upload_path=upload_res.temp_upload_path,
                            upload_id=upload_res.upload_id,
                            parts_metadata=part_metadata_list,
                            project_id=project_id,
                        )
                    except HTTPException as e:
                        if e.status_code == 500 and i < 4:
                            time.sleep(2**(i - 1))
                            continue
                        else:
                            raise
                    else:
                        break

                try:
                    if isinstance(data_mv, memoryview):
                        data_mv.release()
                except Exception:
                    pass

                try:
                    if buf:
                        buf.close()
                except Exception:
                    pass

                _safe_bar_update(file_bar, 1)
                _merge_status_update(fpath)
                _log_file_timing("dir-file(multipart)", fpath, fsize, tread,
                                 tval, 0.0)

            finally:
                mem_budget.release(need_bytes)

        indexed = list(enumerate(files, start=1))
        first_ex = None
        with ThreadPoolExecutor(max_workers=par) as ex:
            futures = {
                ex.submit(_worker, idx, fmeta): (idx, fmeta["path"])
                for idx, fmeta in indexed
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    first_ex = e
                    for f2 in futures:
                        f2.cancel()
                    break
        if first_ex:
            raise first_ex

    # after bars close, log any header renames once
    if detected_type == "csv" and rename_aggregate:
        pairs = ", ".join(f"{k}->{v}" for k, v in rename_aggregate.items())
        logger.info("CSV header sanitized (renamed): %s", pairs)

    logger.info("Upload complete. Validated table %s.", name)


def _upload_table_remote(
    name: str,
    path: str,
    parallelism: Optional[int] = None,
    file_type: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    """Dispatch remote upload to file or directory paths."""
    fs, url = _get_fs_and_path(path)
    info = _remote_info(fs, url)
    st = _make_remote_settings(parallelism)

    if info.get("type") == "file":
        return _remote_upload_file(name, fs, url, info, st, file_type,
                                   project_id=project_id)
    if info.get("type") == "directory":
        return _remote_upload_directory(name, fs, url, info, st, file_type,
                                        project_id=project_id)
    raise ValueError(f"Unsupported remote object type for {path}: {info}")


# -----------------------
# Column name validator
# -----------------------
def _validate_columns_or_raise(names: List[str]) -> None:
    # Ensure sanitized form equals original to enforce our header rules (for
    # parquet), but don't modify parquet; for CSV we already sanitize header
    # proactively.
    new, changed = _sanitize_columns(names)
    if changed:
        diffs = [f"{o}->{n}" for o, n in zip(names, new) if o != n]
        raise ValueError(
            "Column names contain invalid characters or duplicates. "
            "Please rename the following columns:\n  " + ", ".join(diffs))
