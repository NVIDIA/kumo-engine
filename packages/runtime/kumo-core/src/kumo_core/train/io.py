# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-neutral output I/O implemented directly with fsspec."""

from __future__ import annotations

import contextvars
import errno
import logging
import os
import os.path as osp
import queue
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.io import filesystem
from kumo_core.io.filesystem import makedirs as _makedirs
from kumo_core.io.filesystem import open_file
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_SAMPLE_ROWS,
    PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_BYTES,
    PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_ROWS,
    FrameWriterStats,
    parquet_file_prefix_logical_bytes,
    schema_fingerprint,
)

ArrowFrame: TypeAlias = pa.RecordBatch | pa.Table
Frame: TypeAlias = pd.DataFrame | ArrowFrame
_QUEUE_STOP = object()
_DEFAULT_ROW_GROUP_ROWS = 1024 * 1024

logger = logging.getLogger(__name__)


class _ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    """Capture request-scoped filesystem credentials for every task."""
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


def makedirs(path: str, exist_ok: bool = True) -> None:
    _makedirs(path, exist_ok=exist_ok)


def read_binary(path: str) -> bytes:
    with open_file(path, "rb") as handle:
        return handle.read()


def write_binary(path: str, value: bytes) -> None:
    parent = osp.dirname(path)
    if parent:
        makedirs(parent)
    with open_file(path, "wb") as handle:
        handle.write(value)


def write_frame(frame: pd.DataFrame, path: str, *, index: bool = True) -> None:
    parent = osp.dirname(path)
    if parent:
        makedirs(parent)
    with open_file(path, "wb") as handle:
        frame.to_parquet(handle, index=index)


def write_frame_chunks(
    frame: pd.DataFrame,
    path_prefix: str,
    *,
    max_chunk_size: int,
    index: bool,
) -> None:
    row_size = max(
        1, int(frame.memory_usage(deep=True).sum() / max(1, len(frame))))
    rows_per_chunk = max(1, max_chunk_size // row_size)
    for part, start in enumerate(range(0, len(frame), rows_per_chunk)):
        path = f"{path_prefix}-{part:05d}.parquet"
        write_frame(frame.iloc[start:start + rows_per_chunk], path, index=index)


class _ByteBoundedQueue:
    """A queue bounded by Arrow's logical byte size, not batch count."""
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._bytes = 0
        self._items: queue.SimpleQueue[tuple[object, int]] = queue.SimpleQueue()
        self._condition = threading.Condition()

    def put(self, item: object, size: int, check_error: Any) -> None:
        with self._condition:
            # A single oversized input is allowed only when the queue is empty.
            while self._bytes and self._bytes + size > self._capacity:
                check_error()
                self._condition.wait(timeout=0.1)
            check_error()
            self._bytes += size
            self._items.put((item, size))
            self._condition.notify_all()

    def get(self) -> object:
        item, size = self._items.get()
        with self._condition:
            self._bytes -= size
            self._condition.notify_all()
        return item

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def discard(self) -> None:
        while True:
            try:
                _, size = self._items.get_nowait()
            except queue.Empty:
                break
            with self._condition:
                self._bytes -= size
        self.wake()

    def abort(self) -> None:
        self.discard()
        self._items.put((_QUEUE_STOP, 0))
        self.wake()


class StreamingFrameWriter:
    """Bounded Arrow-to-Parquet stream with asynchronous publication.

    Pandas inputs remain supported as a compatibility adapter. Arrow inputs
    avoid pandas entirely. A single serializer thread writes deterministic
    local Parquet parts while a bounded executor publishes completed parts to
    fsspec in parallel. Both queued Arrow memory and local staging space are
    bounded, so a slow object store applies backpressure to the producer. Set
    ``KUMO_BP_OUTPUT_SPOOL_ROOT`` (or ``spool_root``) to a capacity-managed
    pod volume. Direct package users that already expose
    ``DISKGRAPH_DATA_ROOT`` default to its ``bp-output-spool`` subdirectory;
    otherwise Python's platform temporary directory is used. The default two
    upload workers bound each stream to at most three staged
    ``max_file_size`` parts. Serializer-side row-group buffering is capped by
    ``max_buffer_size`` and ``max_row_group_rows``.
    """
    def __init__(
        self,
        path_prefix: str,
        *,
        max_file_size: int,
        max_buffer_size: int,
        index: bool,
        upload_workers: int = 2,
        spool_root: str | os.PathLike[str] | None = None,
        max_row_group_rows: int = _DEFAULT_ROW_GROUP_ROWS,
        stats_sample_rows: int | None = None,
    ) -> None:
        if max_file_size <= 0 or max_buffer_size <= 0:
            raise ValueError("Parquet stream size limits must be positive")
        if upload_workers <= 0:
            raise ValueError("upload_workers must be positive")
        if max_row_group_rows <= 0:
            raise ValueError("Parquet row-group size must be positive")
        if (stats_sample_rows is not None and
                not 0 <= stats_sample_rows <= PREDICTION_ARTIFACT_SAMPLE_ROWS):
            raise ValueError("Parquet stats sample row count is invalid")
        self.path_prefix = path_prefix
        self.max_file_size = max_file_size
        self.max_buffer_size = min(max_buffer_size, max_file_size)
        self.index = index
        self.upload_workers = upload_workers
        self.max_row_group_rows = max_row_group_rows
        self.collect_stats = stats_sample_rows is not None
        self.stats_sample_rows = stats_sample_rows or 0
        if self.collect_stats:
            self.max_buffer_size = min(
                self.max_buffer_size,
                PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_BYTES,
            )
            self.max_row_group_rows = min(
                self.max_row_group_rows,
                PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_ROWS,
            )
        self._peak_row_group_bytes = 0
        self._schema: pa.Schema | None = None
        self._index_offset = 0
        self._closed = False
        self._aborted = False
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._queue = _ByteBoundedQueue(self.max_buffer_size)
        configured_spool = (spool_root
                            or os.environ.get("KUMO_BP_OUTPUT_SPOOL_ROOT"))
        if configured_spool is None:
            diskgraph_data_root = os.environ.get("DISKGRAPH_DATA_ROOT")
            if diskgraph_data_root:
                configured_spool = osp.join(diskgraph_data_root,
                                            "bp-output-spool")
        if configured_spool is not None and not filesystem.is_local(
                os.fspath(configured_spool)):
            raise ValueError("Parquet spool root must be a local filesystem")
        spool_dir = Path(configured_spool) if configured_spool else None
        if spool_dir is not None:
            spool_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(
            tempfile.mkdtemp(
                prefix="kumo-parquet-output-",
                dir=spool_dir,
            ))
        # One current file plus at most ``upload_workers`` completed files.
        self._disk_slots = threading.Semaphore(upload_workers + 1)
        self._upload_pool = _ContextPropagatingThreadPoolExecutor(
            max_workers=upload_workers,
            thread_name_prefix="kumo-parquet-upload",
        )
        self._uploads: list[Future[None]] = []
        self._output_paths: list[str] = []
        self._files_written = 0
        self._num_rows = 0
        self._sample_prefix_logical_bytes = [0]
        self._physical_size_bytes = 0
        self._final_stats: FrameWriterStats | None = None
        serializer_context = contextvars.copy_context()
        self._serializer = threading.Thread(
            target=serializer_context.run,
            args=(self._serialize, ),
            name="kumo-parquet-serializer",
            daemon=True,
        )
        self._serializer.start()

    @property
    def files_written(self) -> int:
        return self._files_written

    @property
    def stats(self) -> FrameWriterStats:
        if self._final_stats is None:
            raise RuntimeError(
                "Parquet output stats require a successful close")
        return self._final_stats

    def _set_error(self, error: Exception) -> None:
        first_error = False
        with self._error_lock:
            if self._error is None:
                self._error = error
                first_error = True
        if first_error:
            self._queue.abort()

    def _check_error(self) -> None:
        if self._aborted:
            raise RuntimeError("Parquet output was aborted")
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Asynchronous Parquet output failed") from error

    def _to_table(self, frame: Frame) -> pa.Table:
        if isinstance(frame, pd.DataFrame):
            if self.index:
                frame = frame.copy(deep=False)
                frame.index = pd.RangeIndex(self._index_offset,
                                            self._index_offset + len(frame))
                self._index_offset += len(frame)
            table = pa.Table.from_pandas(frame, preserve_index=self.index)
        elif isinstance(frame, pa.RecordBatch):
            if self.index:
                raise ValueError(
                    "Arrow streams cannot synthesize a pandas index")
            table = pa.Table.from_batches([frame])
        elif isinstance(frame, pa.Table):
            if self.index:
                raise ValueError(
                    "Arrow streams cannot synthesize a pandas index")
            table = frame
        else:
            raise TypeError(
                f"Unsupported streaming frame type: {type(frame)!r}")

        if self._schema is None:
            self._schema = table.schema
            return table
        if table.schema.names != self._schema.names:
            raise ValueError("Streaming Parquet schema columns changed from "
                             f"{self._schema.names} to {table.schema.names}")
        if not table.schema.equals(self._schema, check_metadata=False):
            table = table.cast(self._schema.remove_metadata(), safe=False)
        return table.replace_schema_metadata(self._schema.metadata)

    def _slices(self, table: pa.Table, max_size: int) -> list[pa.Table]:
        if table.num_rows == 0:
            return []
        if not self.collect_stats:
            size = max(1, table.nbytes)
            rows = max(1, int(table.num_rows * max_size / size))
            return [
                table.slice(start, min(rows, table.num_rows - start))
                for start in range(0, table.num_rows, rows)
            ]
        result = []
        start = 0
        while start < table.num_rows:
            remaining = table.slice(start)
            if remaining.nbytes <= max_size:
                result.append(remaining)
                break
            rows = max(
                1,
                int(remaining.num_rows * max_size / max(1, remaining.nbytes)),
            )
            rows = min(rows, remaining.num_rows)
            chunk = remaining.slice(0, rows)
            if chunk.nbytes > max_size and rows > 1:
                chunk = remaining.slice(0, 1)
                low = 2
                high = rows - 1
                while low <= high:
                    middle = (low + high) // 2
                    candidate = remaining.slice(0, middle)
                    if candidate.nbytes <= max_size:
                        chunk = candidate
                        low = middle + 1
                    else:
                        high = middle - 1
            result.append(chunk)
            start += chunk.num_rows
        return result

    def write(self, frame: Frame) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed Parquet stream")
        self._check_error()
        table = self._to_table(frame)
        if self.collect_stats:
            self._num_rows += table.num_rows
        for chunk in self._slices(table, self.max_buffer_size):
            size = max(1, chunk.nbytes)
            if self.collect_stats and size > self.max_buffer_size:
                assert chunk.num_rows == 1
                raise ValueError(
                    "A single normalized prediction artifact row exceeds "
                    f"the {self.max_buffer_size}-byte stats row-group limit")
            self._queue.put(chunk, size, self._check_error)
        self._check_error()

    def _publish(self, local_path: Path, output_path: str) -> None:
        try:
            parent = osp.dirname(output_path)
            if parent:
                makedirs(parent, exist_ok=True)
            if filesystem.is_local(output_path):
                target = filesystem.local_path(output_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._publish_local(local_path, target)
            else:
                with local_path.open("rb") as source, open_file(
                        output_path, "wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                local_path.unlink(missing_ok=True)
        except Exception as error:
            try:
                filesystem.rm(output_path, ignore_errors=True)
            except Exception:
                logger.warning("Failed to remove incomplete Parquet output %s",
                               output_path, exc_info=True)
            local_path.unlink(missing_ok=True)
            self._set_error(error)
            raise
        finally:
            self._disk_slots.release()

    @staticmethod
    def _publish_local(local_path: Path, target: Path) -> None:
        try:
            os.replace(local_path, target)
            return
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise

        # The spool and output are on different filesystems. Copy to a
        # same-filesystem temporary path, then atomically replace the target.
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with local_path.open("rb") as source, os.fdopen(fd, "wb") as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
                dest.flush()
                os.fsync(dest.fileno())
            os.replace(temporary, target)
            local_path.unlink()
        except Exception:
            try:
                os.close(fd)
            except OSError:
                # os.fdopen() owns and may already have closed the descriptor.
                pass
            temporary.unlink(missing_ok=True)
            raise

    def _submit_part(
        self,
        writer: pq.ParquetWriter,
        local_path: Path,
        part_index: int,
        num_rows: int,
    ) -> None:
        writer.close()
        output_path = f"{self.path_prefix}-{part_index:05d}.parquet"
        if self.collect_stats:
            remaining_sample_rows = (
                self.stats_sample_rows -
                (len(self._sample_prefix_logical_bytes) - 1))
            if remaining_sample_rows > 0:
                prefix = parquet_file_prefix_logical_bytes(
                    str(local_path), min(remaining_sample_rows, num_rows))
                base = self._sample_prefix_logical_bytes[-1]
                self._sample_prefix_logical_bytes.extend(
                    base + value for value in prefix[1:])
            self._physical_size_bytes += local_path.stat().st_size
        pending: list[Future[None]] = []
        for upload in self._uploads:
            if upload.done():
                upload.result()
            else:
                pending.append(upload)
        self._uploads = pending
        future = self._upload_pool.submit(self._publish, local_path,
                                          output_path)
        self._uploads.append(future)
        self._output_paths.append(output_path)
        self._files_written = part_index + 1

    def _serialize(self) -> None:
        writer: pq.ParquetWriter | None = None
        local_path: Path | None = None
        current_size = 0
        current_rows = 0
        part_index = 0
        slot_held = False
        group: list[pa.Table] = []
        group_rows = 0
        group_size = 0

        def finish_part() -> None:
            nonlocal writer, local_path, current_size, current_rows
            nonlocal part_index, slot_held
            if writer is None:
                return
            assert local_path is not None
            self._submit_part(writer, local_path, part_index, current_rows)
            writer = None
            local_path = None
            slot_held = False
            current_size = 0
            current_rows = 0
            part_index += 1

        def flush_group() -> None:
            nonlocal writer, local_path, current_size, current_rows, slot_held
            nonlocal group, group_rows, group_size
            if not group:
                return
            # Keep a row group intact at file boundaries. This may leave the
            # prior part below its approximate logical-size target, but avoids
            # tiny tail row groups and never starts an oversized next group in
            # an already populated part.
            if (writer is not None and current_size
                    and current_size + group_size > self.max_file_size):
                finish_part()
            if writer is None:
                self._disk_slots.acquire()
                slot_held = True
                self._check_error()
                local_path = self._temp_dir / f"part-{part_index:05d}.parquet"
                assert self._schema is not None
                writer = pq.ParquetWriter(local_path, self._schema)
            table = pa.concat_tables(group)
            writer.write_table(table, row_group_size=table.num_rows)
            current_size += group_size
            current_rows += group_rows
            group = []
            group_rows = 0
            group_size = 0
            if current_size >= self.max_file_size:
                finish_part()

        try:
            while True:
                self._check_error()
                item = self._queue.get()
                if item is _QUEUE_STOP:
                    break
                assert isinstance(item, pa.Table)
                pending = item
                while pending.num_rows:
                    self._check_error()
                    remaining = self.max_buffer_size - group_size
                    remaining_rows = self.max_row_group_rows - group_rows
                    if remaining <= 0 or remaining_rows <= 0:
                        flush_group()
                        continue
                    pending_size = max(1, pending.nbytes)
                    rows = min(pending.num_rows, remaining_rows)
                    if pending_size > remaining:
                        rows = min(
                            rows,
                            max(
                                1,
                                int(pending.num_rows * remaining /
                                    pending_size),
                            ),
                        )
                    chunk = pending.slice(0, rows)
                    chunk_size = max(1, chunk.nbytes)
                    while chunk_size > remaining and rows > 1:
                        rows = max(1, int(rows * remaining / chunk_size))
                        chunk = pending.slice(0, rows)
                        chunk_size = max(1, chunk.nbytes)
                    if chunk_size > remaining and group:
                        # One wide row cannot be sliced further. Flush the
                        # partial group before admitting it so a stats-enabled
                        # writer retains its hard byte ceiling.
                        flush_group()
                        continue
                    if chunk_size > remaining and self.collect_stats:
                        raise ValueError(
                            "A single normalized prediction artifact row "
                            f"exceeds the {self.max_buffer_size}-byte stats "
                            "row-group limit")
                    pending = pending.slice(rows)
                    group.append(chunk)
                    group_rows += rows
                    group_size += chunk_size
                    self._peak_row_group_bytes = max(self._peak_row_group_bytes,
                                                     group_size)
                    if (group_rows >= self.max_row_group_rows
                            or group_size >= self.max_buffer_size
                            or pending.num_rows):
                        flush_group()
            flush_group()
            finish_part()
        except Exception as error:
            self._set_error(error)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    logger.warning(
                        "Failed to close Parquet writer during cleanup",
                        exc_info=True)
            if local_path is not None:
                local_path.unlink(missing_ok=True)
            if slot_held:
                self._disk_slots.release()
            self._queue.discard()

    def close(self) -> None:
        if self._aborted:
            return
        if self._closed:
            self._check_error()
            return
        self._closed = True
        close_error: Exception | None = None
        try:
            try:
                self._queue.put(_QUEUE_STOP, 0, self._check_error)
            except Exception as error:
                close_error = error
                self._queue.abort()
            self._serializer.join()
            self._upload_pool.shutdown(wait=True)
            for upload in self._uploads:
                try:
                    upload.result()
                except Exception as error:
                    close_error = close_error or error
            try:
                self._check_error()
            except Exception as error:
                close_error = close_error or error
            if close_error is None and self.collect_stats:
                schema = self._schema or pa.schema([])
                sample_rows = len(self._sample_prefix_logical_bytes) - 1
                self._final_stats = FrameWriterStats(
                    path_prefix=self.path_prefix,
                    schema_fingerprint=schema_fingerprint(schema),
                    num_rows=self._num_rows,
                    num_cols=len(schema.names),
                    sample_rows=sample_rows,
                    sample_prefix_logical_bytes=tuple(
                        self._sample_prefix_logical_bytes),
                    physical_size_bytes=self._physical_size_bytes,
                    file_count=self._files_written,
                )
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        if close_error is not None:
            for output_path in self._output_paths:
                try:
                    filesystem.rm(output_path, ignore_errors=True)
                except Exception:
                    logger.warning(
                        "Failed to remove incomplete Parquet output %s",
                        output_path, exc_info=True)
            raise RuntimeError(
                "Failed to close Parquet output") from close_error

    def abort(self) -> None:
        """Stop background work and remove every part from this attempt."""
        if self._aborted:
            return
        self._aborted = True
        self._closed = True
        self._queue.abort()
        self._serializer.join()
        self._upload_pool.shutdown(wait=True, cancel_futures=True)
        for upload in self._uploads:
            if upload.cancelled():
                continue
            try:
                upload.result()
            except Exception:
                logger.warning("Parquet upload failed while aborting output",
                               exc_info=True)
        for output_path in self._output_paths:
            try:
                filesystem.rm(output_path, ignore_errors=True)
            except Exception:
                logger.warning("Failed to remove aborted Parquet output %s",
                               output_path, exc_info=True)
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self) -> "StreamingFrameWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
