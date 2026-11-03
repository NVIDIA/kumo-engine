# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, replace
from functools import partial
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Literal, Mapping, NamedTuple, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kumoapi.typing import Dtype

from ..types.enums import ConnectorType
from . import layout
from .arrow_parquet import (
    UNSUPPORTED_ADVICE,
    count_rows,
    materialize_arrow_streams,
    unsupported_columns,
)
from .cpu import (
    COPY_WORKER_BYTES,
    copy_workers,
    download_workers,
    encode_workers,
    memory_capacity,
    memory_share,
)
from .csv_format import ARROW_NA_TOKENS, CsvOptions
from .errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceError,
    SourceNotFoundError,
    TransientSourceError,
    UnsupportedReadModeError,
)
from .identity import epoch_second
from .object_store_access import ObjectFetch, ObjectIdentity
from .parallel import run_in_parallel
from .prepare import ReadOutput, run_prepare
from .protocol import require_mode, staged_first
from .sql import validated_aliases
from .transfer import (
    DOWNLOAD_CHUNK,
    StagedFile,
    discard_destination,
    fetch_all,
    verify_parquet,
)
from .types import ReadMode, StagedTable

STREAMED_ONLY = staged_first(False)

logger = logging.getLogger(__name__)

_PANDAS_INDEX_COLUMN = re.compile(r"__index_level_\d+__")

#: Whatever a vendor quoted, which is the object it was talking about.
_QUOTED = re.compile(r"""'[^']*'|"[^"]*"|s3://\S+""")

#: An HTTP status as a status, not as a path segment: ``reports/404/part`` is a
#: prefix a customer is free to name.
_STATUS = r"(?:^|[^\w/.]){}(?![\w/])"

#: The hosted connector reads partitioned tables through
#: ``pq.ParquetDataset(partitioning="hive")``, which dictionary-encodes the
#: partition keys where the plain ``"hive"`` shorthand infers int32 instead.
_HIVE = ds.HivePartitioning.discover(infer_dictionary=True)
_PARQUET_BATCH_BYTES = 16 << 20
_PARQUET_PROBE_ROWS = 1_024
_PARQUET_FIRST_BATCH_ROWS = 65_536
_PARQUET_BUFFER_BYTES = 1 << 20
#: About what one re-encode stream holds, ~0.2GiB measured: a batch, the
#: shard pending its write, the writer, and the read buffers.
_STREAM_BYTES = 256 << 20
_IO_THREADS = threading.Lock()
#: The pool size before the first active read raised it, and what each
#: active read asked for.
_io_base: list[int] = []
_io_wanted: list[int] = []


def _source_version(path: str, entries: Sequence[Any]) -> str:
    if len(entries) == 1 and entries[0].path.rstrip("/") == path.rstrip("/"):
        latest = entries[0].mtime
        return epoch_second(latest) if latest is not None else "-1"
    # A maximum mtime misses changes to older files and added/deleted parts.
    metadata = [
        (entry.path, entry.size, getattr(entry, "mtime_ns", None),
         entry.mtime.isoformat() if entry.mtime is not None else None)
        for entry in sorted(entries, key=lambda entry: entry.path)
    ]
    return hashlib.sha256(json.dumps(metadata).encode()).hexdigest()


class _CsvSource(NamedTuple):
    dataset: ds.Dataset
    dtypes: dict


@dataclass
class ObjectStoreSourceReader:
    """The filesystem is injected the way the warehouse readers take an
    authenticated client: resolving credentials belongs to the caller.
    """

    filesystem: Any
    _: KW_ONLY
    root: str = ""
    csv_options: CsvOptions | None = None
    file_format: Literal["csv", "parquet"] | None = None
    file_suffix: str | None = None
    fetch: ObjectFetch | None = None
    object_identity: ObjectIdentity | None = None
    #: DiskGraph converts the staged columns to their declared dtypes itself.
    ingest_converts: bool = False

    source_type: ConnectorType = ConnectorType.S3

    def read_modes(self, table: str) -> Sequence[ReadMode]:
        """An object store stages nothing on the caller's behalf; the
        verbatim copy below is a fast path within the streamed read, not a
        mode."""
        return STREAMED_ONLY

    def tables(self) -> list[str]:
        """The table names under this reader's root."""
        with _arrow_errors(self.root):
            found = layout.scan(self.filesystem, self.root,
                                max_entries=layout.MAX_SCANNED_ENTRIES)
        return layout.discover_tables(found)

    def data_version(self, table: str, *, sql: str | None = None,
                     columns: Mapping[str, Dtype] | None = None) -> str:
        require_no_sql(sql)
        from pyarrow.fs import FileType

        with _arrow_errors(table):
            info = layout.locate(
                self.filesystem, self.root, table,
                exact=self.file_format is not None or not table)
            entries = ([info] if info.type is FileType.File else
                       self._objects(table, info.path, include_hidden=True))
        return self._data_version(info.path, entries)

    def _data_version(self, path: str, entries: Sequence[Any]) -> str:
        return _source_version(path, entries)

    def resolve(self, table: str) -> tuple[str, str]:
        """An explicit format addresses an exact path; discovery infers both."""
        with _arrow_errors(table):
            if self.file_format is not None:
                from pyarrow.fs import FileType

                name = layout.confined(table)
                path = posixpath.join(self.root, name) if name else self.root
                if self.filesystem.get_file_info(
                        path).type is FileType.NotFound:
                    raise SourceNotFoundError(f"Nothing exists at {path}.",
                                              table=table)
                return path, self.file_format
            return layout.resolve(self.filesystem, self.root, table)

    def _objects(self, table: str, path: str, max_entries: int | None = None,
                 include_hidden: bool = False) -> list[Any]:
        """The one listing every read is derived from, behind the one error
        boundary: Arrow reports an unreachable bucket as a bare ``OSError``."""
        with _arrow_errors(table):
            return layout.objects(self.filesystem, path,
                                  suffix=self.file_suffix,
                                  max_entries=max_entries,
                                  include_hidden=include_hidden)

    def object_paths(self, table: str) -> list[str]:
        path, _ = self.resolve(table)
        return [entry.path for entry in self._objects(table, path)]

    def dtypes(self, table: str) -> dict[str, Dtype]:
        path, kind = self.resolve(table)
        if kind == layout.CSV:
            return {
                name: _dtype_of(arrow_type)
                for name, arrow_type in self._csv_catalog(table, path).items()
            }
        return column_dtypes(self.dataset(table).schema)

    def source_types(self, table: str) -> dict[str, pa.DataType]:
        path, kind = self.resolve(table)
        if kind == layout.CSV:
            return self._csv_catalog(table, path)
        schema = self.dataset(table).schema
        return dict(zip(schema.names, schema.types))

    def _csv_catalog(self, table: str, path: str) -> dict[str, pa.DataType]:
        from . import csv_format
        parts = self._csv_parts(table, self._objects(table, path))
        with _arrow_errors(parts[0], "CSV"):
            sep = ((self.csv_options.delimiter if self.csv_options else None)
                   or csv_format.delimiter(self.filesystem, parts[0]))
            try:
                return csv_format.sampled(self.filesystem, parts, sep,
                                          self.csv_options)
            except OverflowError as error:
                raise PermanentSourceError(
                    f"{table} holds a whole number too wide for a 64-bit "
                    f"integer column ({error}). Kumo cannot read it either "
                    f"here or on the platform; give the column a narrower "
                    f"type at the source, or export it as text.",
                    table=table) from error

    def validate_ingest_columns(self, table: str,
                                columns: Mapping[str, Dtype]) -> None:
        """Require graph columns before dataset null-filling.

        General reads retain column-drop evolution. A graph's declared
        fields, including partition keys, must exist in every source shard.
        """
        path, kind = self.resolve(table)
        if kind != layout.PARQUET:
            return  # CSV conversion checks its requested columns.
        found = self._objects(table, path)
        partition_keys = {}
        if layout.hive_partitioned(path, [entry.path for entry in found]):
            dataset = self._dataset(table, path, kind, found)
            partition_keys = {
                fragment.path:
                set(ds.get_partition_keys(fragment.partition_expression))
                for fragment in dataset.get_fragments()
            }
        for entry, schema in zip(found, self._footers(found, None)):
            missing = (set(columns) - set(schema.names) -
                       partition_keys.get(entry.path, set()))
            if missing:
                raise PermanentSourceError(
                    f"Graph source shard {posixpath.basename(entry.path)!r} "
                    f"is missing declared columns: {', '.join(sorted(missing))}",
                    table=table)

    def head(self, table: str, rows: int) -> pa.Table:
        """The first rows, under the types and the columns this reader
        reports rather than the ones the dataset carries: a CSV dataset is
        text until the projection converts it, and a Parquet dataset still
        holds the pandas index column that :meth:`dtypes` drops.
        """
        path, kind = self.resolve(table)
        found = self._objects(table, path)
        if kind != layout.CSV:
            dataset = self._dataset(table, path, kind, found)
            _refuse_unsupported(table, dataset.schema)
            with _arrow_errors(table):
                from .arrow_parquet import read_bounded_dataset
                return read_bounded_dataset(dataset,
                                            list(column_dtypes(dataset.schema)),
                                            rows)
        from . import csv_format, csv_preview
        parts = self._csv_parts(table, found)
        with _arrow_errors(table, "CSV"):
            sep = ((self.csv_options.delimiter if self.csv_options else None)
                   or csv_format.delimiter(self.filesystem, parts[0]))
            frame = csv_preview.read_files(
                self.filesystem, parts, sep=sep, rows=rows,
                null_values=self.csv_options.null_values if self.csv_options
                else None, escape_char=self.csv_options.escape_char
                if self.csv_options else False)
            return pa.Table.from_pandas(frame, preserve_index=False).cast(
                pa.schema(self._csv_catalog(table, path)))

    def dataset(self, table: str, *, validate: bool = True,
                pre_buffer: bool | None = None) -> ds.Dataset:
        path, kind = self.resolve(table)
        dataset = self._dataset(table, path, kind, self._objects(table, path),
                                pre_buffer=pre_buffer)
        if validate:
            _refuse_unsupported(table, dataset.schema)
        return dataset

    def prepare(
        self,
        table: str,
        dest: str,
        *,
        columns: Mapping[str, Dtype] | None,
        mode: ReadMode,
        aliases: Mapping[str, str] | None = None,
        expected_rows: int | None = None,
        worker_ceiling: int | None = None,
        sql: str | None = None,
        version: str | None = None,
        transform: bool = True,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> StagedTable:
        """With no projection, preserve Parquet bytes and physicalize Hive keys.

        Full-file copies need no remote footer scans or row count.
        """
        if not transform and (columns is not None or aliases):
            raise ValueError(
                "Untransformed reads cannot select or rename columns")
        require_no_sql(sql)
        require_mode(table, mode, STREAMED_ONLY)
        if columns is None and aliases:
            raise ValueError("Aliases require an explicit column selection")
        resolved = validated_aliases(columns,
                                     aliases) if columns is not None else {}
        path, kind = self.resolve(table)
        if not transform and kind == layout.CSV and expected_rows is not None:
            raise ValueError("Untransformed CSV reads cannot attest row counts")
        found = self._objects(table, path, max_entries,
                              include_hidden=not transform)
        # Stamp the same metadata listing the read consumes, before fetching
        # any files; later additions must not be stamped onto this output.
        stamped = version or self._data_version(path, found)
        source_bytes = (sum(entry.size for entry in found) if all(
            entry.size is not None and entry.size >= 0
            for entry in found) else None)

        if max_bytes is not None and (source_bytes is None
                                      or source_bytes > max_bytes):
            raise PermanentSourceError(
                "Dataset exceeds the download byte limit")

        def read() -> ReadOutput:
            if not transform:
                with _arrow_errors(table):
                    paths = self._download_objects(
                        path, found, dest, worker_ceiling,
                        parquet=kind == layout.PARQUET)
                return ReadOutput(
                    paths=paths, scanned_bytes=source_bytes,
                    num_rows=(count_rows(paths) if expected_rows is not None
                              else None if paths else 0))
            output = self._read(table, path, kind, found, dest, columns,
                                resolved, worker_ceiling,
                                count_copied_rows=expected_rows is not None)
            return replace(output, scanned_bytes=source_bytes)

        def checked_read() -> ReadOutput:
            # Capture provider identity before schema/footer discovery as well
            # as data reads. Keep this per call, not on the shared reader.
            with _arrow_errors(table):
                identities = self._identities(found, worker_ceiling)
            try:
                output = read()
            except Exception as error:
                try:
                    self._unchanged(table, found, dest, worker_ceiling,
                                    identities, error)
                finally:
                    discard_destination(dest)
                raise
            self._unchanged(table, found, dest, worker_ceiling, identities)
            return output

        return run_prepare(table, dest, mode, stamp=lambda: stamped,
                           read=checked_read, expected_rows=expected_rows)

    def _read(
        self,
        table: str,
        path: str,
        kind: str,
        found: Sequence[Any],
        dest: str,
        columns: Mapping[str, Dtype] | None,
        aliases: Mapping[str, str],
        worker_ceiling: int | None,
        *,
        count_copied_rows: bool = False,
    ) -> ReadOutput:
        """Parquet is read from the bucket once: copied whole where nothing
        would change, otherwise only the selected columns are fetched and
        re-encoded. CSV, which cannot be projected, is downloaded first."""
        if kind == layout.CSV:
            if self.fetch is not None:
                with _arrow_errors(table, kind):
                    return self._read_downloaded(table, path, found, dest,
                                                 columns, aliases,
                                                 worker_ceiling)
            return self._read_csv(table, found, dest, columns, aliases,
                                  worker_ceiling)
        with _arrow_errors(table):
            if columns is None and not layout.nested(
                [entry.path[len(path.rstrip('/')) + 1:] for entry in found]):
                copied = self._copy(path, found, dest, worker_ceiling)
                return ReadOutput(
                    paths=copied,
                    num_rows=count_rows(copied) if count_copied_rows else None)
            if columns is not None and self._verbatim(
                    path, found, columns, aliases, worker_ceiling):
                copied = self._copy(path, found, dest, worker_ceiling)
                return ReadOutput(paths=copied, num_rows=count_rows(copied))
        dataset = self._dataset(table, path, kind, found)
        if columns is not None:
            _refuse_unsupported(table, dataset.schema)
        with _arrow_errors(table):
            written = self._stream(table, dataset, dest, columns, aliases,
                                   worker_ceiling)
        return ReadOutput(paths=written.paths, num_rows=written.rows)

    def _identities(self, found: Sequence[Any],
                    worker_ceiling: int | None) -> list[tuple[str, ...]] | None:
        if self.object_identity is None:
            return None
        if not found:
            return []
        context = contextvars.copy_context()
        return run_in_parallel(
            [partial(context.copy().run, self.object_identity, entry.path)
             for entry in found],
            workers=min(download_workers(worker_ceiling), len(found)),
            stop=threading.Event())

    def _unchanged(self, table: str, found: Sequence[Any], dest: str,
                   worker_ceiling: int | None,
                   identities: list[tuple[str, ...]] | None,
                   error: Exception | None = None) -> None:
        """Reject changed objects before publishing staging. Native reads
        retain projection and concurrency; provider tokens also catch rewrites
        with identical size/mtime. This is not a transaction across objects."""
        try:
            with _arrow_errors(table):
                if identities is not None:
                    unchanged = identities == self._identities(
                        found, worker_ceiling)
                elif found:
                    context = contextvars.copy_context()
                    now = run_in_parallel(
                        [partial(context.copy().run,
                                 self.filesystem.get_file_info, entry.path)
                         for entry in found],
                        workers=min(download_workers(worker_ceiling),
                                    len(found)), stop=threading.Event())
                    unchanged = all(
                        (before.size, before.mtime_ns)
                        == (after.size, after.mtime_ns)
                        for before, after in zip(found, now))
                else:
                    unchanged = True
        except (SourceNotFoundError, FileNotFoundError):
            unchanged = False
        except Exception:
            discard_destination(dest)
            raise
        if unchanged:
            return
        discard_destination(dest)
        raise TransientSourceError(
            f"{table} changed while it was read; retry the read.",
            table=table) from error

    def _read_downloaded(
        self,
        table: str,
        path: str,
        found: Sequence[Any],
        dest: str,
        columns: Mapping[str, Dtype] | None,
        aliases: Mapping[str, str],
        worker_ceiling: int | None,
    ) -> ReadOutput:
        from pyarrow.fs import LocalFileSystem

        self._csv_parts(table, found)
        with TemporaryDirectory(prefix='kumo-raw-source-', dir=os.path.dirname(
                os.path.abspath(dest))) as raw:
            paths = self._download_objects(path, found, raw, worker_ceiling)
            local = replace(self, filesystem=LocalFileSystem(), fetch=None)
            return local._read_csv(table, local.filesystem.get_file_info(paths),
                                   dest, columns, aliases, worker_ceiling)

    def _copy(self, path: str, found: Sequence[Any], dest: str,
              worker_ceiling: int | None) -> tuple[str, ...]:
        if self.fetch is not None:
            return self._download_objects(path, found, dest, worker_ceiling,
                                          parquet=True)
        with _copies(found, worker_ceiling) as workers:
            return fetch_all([
                StagedFile(
                    entry.path, entry.size
                    if entry.size is not None and entry.size >= 0 else None)
                for entry in found
            ], dest, fetch=self._fetch, workers=workers)

    def _download_objects(self, path: str, found: Sequence[Any], dest: str,
                          worker_ceiling: int | None, *,
                          parquet: bool = False) -> tuple[str, ...]:
        def download(entry: Any) -> str:
            relative = (posixpath.basename(entry.path)
                        if entry.path == path.rstrip('/') else
                        entry.path[len(path.rstrip('/')) + 1:])
            target = os.path.join(dest, layout.confined(relative))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            decoded_size = None
            if self.fetch is None:
                self._fetch(StagedFile(entry.path, entry.size), target)
            else:
                decoded_size = self.fetch(entry.path, target)
            if parquet:
                expected_size = decoded_size if decoded_size is not None else entry.size
                verify_parquet(
                    target, entry.path, expected_size=expected_size if
                    expected_size is not None and expected_size >= 0 else None)
            return target

        try:
            with _copies(found, worker_ceiling) as workers, ThreadPoolExecutor(
                    max_workers=workers) as pool:
                futures = [pool.submit(download, entry) for entry in found]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
                return tuple(future.result() for future in futures)
        except BaseException:
            discard_destination(dest)
            raise

    def _read_csv(
        self,
        table: str,
        found: Sequence[Any],
        dest: str,
        columns: Mapping[str, Dtype] | None,
        aliases: Mapping[str, str],
        worker_ceiling: int | None,
    ) -> ReadOutput:
        source = self._csv_source(table, found, columns)
        if columns is None:
            columns = source.dtypes
        with _arrow_errors(table, "CSV"):
            written = self._stream(table, source.dataset, dest, columns,
                                   aliases, worker_ceiling, normalize=True)
        return ReadOutput(paths=written.paths, num_rows=written.rows)

    def _stream(
        self,
        table: str,
        dataset: ds.Dataset,
        dest: str,
        columns: Mapping[str, Dtype] | None,
        aliases: Mapping[str, str],
        worker_ceiling: int | None,
        normalize: bool = False,
    ) -> Any:
        null_values = (self.csv_options.null_values if self.csv_options
                       and self.csv_options.null_values is not None else
                       ARROW_NA_TOKENS)
        partitions = (dataset.partitioning.schema.names
                      if dataset.partitioning is not None else [])
        projection = (_projection(table, dataset.schema, columns, aliases,
                                  normalize=normalize, null_values=null_values,
                                  converts=self.ingest_converts,
                                  partitions=partitions)
                      if columns is not None else {
                          name: pc.field(name)
                          for name in dataset.schema.names
                      })
        if columns is None and dataset.partitioning is not None:
            for field in dataset.partitioning.schema:
                if pa.types.is_dictionary(field.type):
                    # DiskGraph treats dictionary-typed string partitions as
                    # nulls; materialize the partition value type instead.
                    projection[field.name] = pc.field(field.name).cast(
                        field.type.value_type)
        sources = dataset.schema.names if columns is None else list(columns)
        fragments = list(dataset.get_fragments())
        streams = [
            _fragment_batches(fragment, dataset.schema, projection, sources,
                              own_types=self.ingest_converts)
            for fragment in fragments
        ]
        prebuffered, largest = _prebuffered_bytes(fragments, sources,
                                                  worker_ceiling)
        per_stream = _STREAM_BYTES + prebuffered
        capacity = memory_capacity()
        if capacity is not None and per_stream > capacity:
            raise PermanentSourceError(
                f"Re-encoding {largest} needs {per_stream >> 20} MiB, but "
                f"only {capacity >> 20} MiB is free. Rewrite it with smaller "
                f"row groups, or give the reader more memory.", table=table)
        remote = any(_remote(fragment) for fragment in fragments)
        with memory_share(
                per_stream,
                lambda ceiling: min(encode_workers(ceiling),
                                    max(len(streams), 1)),
                worker_ceiling) as workers, _io_threads(
                    workers if remote else 0):
            logger.info("Staging %s: %d parts, %d at a time, %d MiB each",
                        table, len(streams), workers, per_stream >> 20)
            return materialize_arrow_streams(
                streams, dest, worker_ceiling=workers,
                uniform=not self.ingest_converts)

    def _verbatim(
        self,
        path: str,
        found: Sequence[Any],
        columns: Mapping[str, Dtype],
        aliases: Mapping[str, str],
        worker_ceiling: int | None,
    ) -> bool:
        """Copied only where the re-encode path would have cast nothing, so
        the choice between them is a performance one alone.

        Every object's footer is read, not just the first: a dataset takes its
        schema from one fragment, while the re-encode path scans them all.
        """
        if aliases or layout.hive_partitioned(path,
                                              [entry.path for entry in found]):
            return False
        declared = list(columns)
        for schema in self._footers(found, worker_ceiling):
            _refuse_unsupported(path, schema)
            if list(schema.names) != declared:
                return False
            for name, dtype in columns.items():
                if not (self.ingest_converts
                        or _uncast(schema.field(name).type, dtype)):
                    return False
        return True

    def _footers(self, found: Sequence[Any],
                 worker_ceiling: int | None) -> list[pa.Schema]:
        with _arrow_errors(found[0].path):
            return run_in_parallel(
                [
                    partial(pq.read_schema, entry.path,
                            filesystem=self.filesystem) for entry in found
                ],
                workers=min(download_workers(worker_ceiling), len(found)),
                stop=threading.Event(),
            )

    def _fetch(self, staged: StagedFile, target: str) -> None:
        if self.fetch is not None:
            self.fetch(staged.path, target)
            return
        with self.filesystem.open_input_stream(staged.path) as source:
            with open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, DOWNLOAD_CHUNK)

    def _dataset(self, table: str, path: str, kind: str, found: Sequence[Any],
                 *, pre_buffer: bool | None = None) -> ds.Dataset:
        if kind == layout.CSV:
            return self._csv_source(table, found).dataset
        paths = [entry.path for entry in found]
        parquet_format = ds.ParquetFileFormat(
            default_fragment_scan_options=(ds.ParquetFragmentScanOptions(
                pre_buffer=pre_buffer, buffer_size=64 * 1024 *
                1024) if pre_buffer is not None else None))
        with _arrow_errors(paths[0]):
            if layout.hive_partitioned(path, paths):
                return ds.dataset(paths, filesystem=self.filesystem,
                                  format=parquet_format, partitioning=_HIVE,
                                  partition_base_dir=path.rstrip('/'))
            # Pin the physical schema once; partition columns require the
            # dataset discovery above instead of a single-file schema.
            schema = pq.read_metadata(
                paths[0], filesystem=self.filesystem).schema.to_arrow_schema()
            return ds.dataset(paths, filesystem=self.filesystem,
                              format=parquet_format, schema=schema)

    def _csv_parts(self, table: str, found: Sequence[Any]) -> list[str]:
        from . import csv_format
        for refusal in (layout.csv_partitions_refusal(len(found)),
                        layout.csv_size_refusal(
                            sum(max(entry.size or 0, 0) for entry in found))):
            if refusal is not None:
                raise PermanentSourceError(refusal, table=table)
        parts = [entry.path for entry in csv_format.readable_parts(found)]
        if not parts:
            raise SourceNotFoundError(f"Every object backing {table} is empty.",
                                      table=table)
        return parts

    def _csv_source(
        self,
        table: str,
        found: Sequence[Any],
        columns: Mapping[str, Dtype] | None = None,
    ) -> _CsvSource:
        from . import csv_format
        parts = self._csv_parts(table, found)
        with _arrow_errors(parts[0], "CSV"):
            sep = ((self.csv_options.delimiter if self.csv_options else None)
                   or csv_format.delimiter(self.filesystem, parts[0]))
            if columns is None:
                sampled = csv_format.sampled(self.filesystem, parts, sep,
                                             self.csv_options)
                columns = {
                    name: _dtype_of(dtype)
                    for name, dtype in sampled.items()
                }
            return _CsvSource(
                ds.dataset(
                    parts, filesystem=self.filesystem,
                    format=csv_format.text_format(sep, list(columns),
                                                  _unpinned(columns),
                                                  self.csv_options)),
                dict(columns))


def require_no_sql(sql: str | None) -> None:
    if sql is not None:
        raise UnsupportedReadModeError(
            "An object store cannot run SQL; Custom SQL needs a warehouse "
            "connector (Snowflake, Databricks or BigQuery).")


def _fragment_batches(fragment: Any, schema: pa.Schema,
                      projection: Mapping[str, Any], sources: Sequence[str],
                      *, own_types: bool = False) -> Iterator[Any]:
    """A generator, so the scan starts when a worker reaches this fragment.
    Building the scanners up front starts every one: 580MB of Arrow buffers at
    60 fragments, before a single row is read.

    A Parquet fragment is not scanned: a scan decodes a whole row group at a
    time, 3.4GiB per worker for a Spark file's 5.7M-row groups (kumo#31439).
    """
    if not isinstance(fragment, ds.ParquetFileFragment):
        yield from ds.Scanner.from_fragment(
            fragment, schema=schema, columns=dict(projection)).to_batches()
        return
    fields = [schema.field(name) for name in sources]
    if own_types:
        stored = fragment.physical_schema
        keys = ds.get_partition_keys(fragment.partition_expression)
        fields = [
            stored.field(field.name)
            if field.name in stored.names and field.name not in keys else field
            for field in fields
        ]
        _refuse_unsupported(fragment.path, pa.schema(fields))
    # Projected batch by batch: ``Scanner.from_batches`` reads its source on a
    # background thread that queues ahead of a slow writer, ~1GiB per worker.
    for batch in _parquet_batches(fragment, pa.schema(fields)):
        yield ds.InMemoryDataset(batch).to_table(columns=dict(projection),
                                                 use_threads=False)


def _parquet_batches(fragment: Any,
                     schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Conformed to the dataset schema as a scan would: a partition key wins,
    a column the file lacks is null, and any other type is cast.

    One row group at a time, in batches sized from the widest row decoded so
    far, starting from a small probe: a footer counts a repeated dictionary
    value once. A remote part pre-buffers the row group's chunks, coalesced
    and fetched concurrently.
    """
    keys = ds.get_partition_keys(fragment.partition_expression)
    remote = _remote(fragment)
    with fragment.open() as source:
        parquet = pq.ParquetFile(
            source, metadata=fragment.metadata, pre_buffer=remote,
            buffer_size=0 if remote else _PARQUET_BUFFER_BYTES)
        stored = set(parquet.schema_arrow.names)
        columns = [
            name for name in schema.names
            if name in stored and name not in keys
        ]
        probe = next(
            pq.ParquetFile(source, metadata=parquet.metadata,
                           buffer_size=_PARQUET_BUFFER_BYTES).iter_batches(
                               batch_size=_PARQUET_PROBE_ROWS,
                               columns=columns), None)
        width = probe.nbytes / probe.num_rows if probe else 0.0
        del probe
        chunks = _chunks(parquet.metadata, columns)
        for group in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(group)
            encoded = sum(
                metadata.column(index).total_uncompressed_size
                for index in chunks)
            rows = _PARQUET_BATCH_BYTES / max(
                width, encoded / max(metadata.num_rows, 1), 1)
            if not width:
                rows = min(rows, _PARQUET_FIRST_BATCH_ROWS)
            for batch in parquet.iter_batches(batch_size=max(1, int(rows)),
                                              row_groups=[group],
                                              columns=columns):
                if batch.num_rows:
                    width = max(width, batch.nbytes / batch.num_rows)
                yield pa.RecordBatch.from_arrays([
                    _repeated(keys[field.name], field.type, batch.num_rows)
                    if field.name in keys else
                    batch.column(field.name).cast(field.type)
                    if field.name in batch.schema.names else pa.nulls(
                        batch.num_rows, field.type) for field in schema
                ], schema=schema)
        if not parquet.metadata.num_rows:
            # A scan yields one empty batch for an empty part, which is what
            # stages an empty table's schema.
            yield pa.RecordBatch.from_pylist([], schema=schema)


@contextmanager
def _copies(found: Sequence[Any],
            worker_ceiling: int | None) -> Iterator[int]:
    """Arrow-backed downloaders copy on Arrow's IO pool."""
    with memory_share(
            COPY_WORKER_BYTES, lambda ceiling: min(
                copy_workers(ceiling), max(len(found), 1)),
            worker_ceiling) as workers, _io_threads(workers):
        yield workers


@contextmanager
def _io_threads(workers: int) -> Iterator[None]:
    """Arrow has one IO pool per process, with no per-call executor for
    copies or pre-buffered reads. Raised while any read needs it; restored
    once the last one ends, so a long-lived process does not keep it."""
    with _IO_THREADS:
        if not _io_wanted:
            _io_base[:] = [pa.io_thread_count()]
        _io_wanted.append(workers)
        if pa.io_thread_count() < workers:
            pa.set_io_thread_count(workers)
    try:
        yield
    finally:
        with _IO_THREADS:
            _io_wanted.remove(workers)
            pa.set_io_thread_count(max([*_io_base, *_io_wanted]))


def _remote(fragment: Any) -> bool:
    from pyarrow.fs import LocalFileSystem

    return (isinstance(fragment, ds.ParquetFileFragment)
            and not isinstance(fragment.filesystem, LocalFileSystem))


def _prebuffered_bytes(fragments: Sequence[Any], columns: Sequence[str],
                       worker_ceiling: int | None) -> tuple[int, str]:
    """The largest row group's selected chunks, which a remote part holds
    pre-buffered while it is decoded, and where it is."""
    remote = [fragment for fragment in fragments if _remote(fragment)]
    if not remote:
        return 0, ""
    footers = run_in_parallel(
        [partial(getattr, fragment, "metadata") for fragment in remote],
        workers=min(download_workers(worker_ceiling), len(remote)),
        stop=threading.Event())
    largest = (0, "")
    for fragment, footer in zip(remote, footers):
        chunks = _chunks(footer, columns)
        for group in range(footer.num_row_groups):
            metadata = footer.row_group(group)
            largest = max(largest, (sum(
                metadata.column(index).total_compressed_size
                for index in chunks), f"row group {group} of {fragment.path}"))
    return largest


def _chunks(metadata: Any, columns: Sequence[str]) -> list[int]:
    """Column chunk indices of the named top-level fields, by position: a
    field name may itself contain a period."""
    chunks, start = [], 0
    for field in metadata.schema.to_arrow_schema():
        leaves = _leaves(field.type)
        if field.name in columns:
            chunks.extend(range(start, start + leaves))
        start += leaves
    return chunks


def _leaves(dtype: pa.DataType) -> int:
    if not dtype.num_fields:
        return 1
    return sum(_leaves(dtype.field(index).type)
               for index in range(dtype.num_fields))


def _repeated(value: Any, dtype: pa.DataType, rows: int) -> pa.Array:
    if pa.types.is_dictionary(dtype):
        return _repeated(value, dtype.value_type,
                         rows).dictionary_encode().cast(dtype)
    return pa.repeat(pa.scalar(value, dtype), rows)


def _refuse_unsupported(table: str, schema: pa.Schema) -> None:
    decoded = pa.schema([
        field.with_type(field.type.value_type)
        if pa.types.is_dictionary(field.type) else field for field in schema
    ])
    offending = unsupported_columns(decoded)
    if not offending:
        return
    named = ", ".join(f"{name} ({dtype})" for name, dtype in offending)
    raise PermanentSourceError(
        f"{table} has columns Kumo cannot read: {named}. "
        f"{UNSUPPORTED_ADVICE}", table=table)


def column_dtypes(schema: pa.Schema) -> dict[str, Dtype]:
    return dict(iter_column_dtypes(schema))


def iter_column_dtypes(schema: pa.Schema) -> Iterator[tuple[str, Dtype]]:
    """Parquet only: a CSV column genuinely named ``__index_level_0__`` is one
    the platform reports like any other."""
    for field in schema:
        # Matched as a prefix, not in full, because that is how the hosted
        # Parquet reader matches it: `__index_level_0__extra` is dropped there.
        if _PANDAS_INDEX_COLUMN.match(field.name):
            continue
        yield field.name, _dtype_of(field.type)


#: Matched on the message because Arrow reports every failure as one exception
#: type. Ordered credential, denial, retryable, absent, because the vaguest
#: phrasing is the last one's: AWS answers a bad key with "The AWS Access Key
#: Id you provided does not exist in our records". Word boundaries are written
#: ``[_ ]?`` because one vendor error arrives spelled three ways: ``SlowDown``,
#: ``SLOW_DOWN`` and ``Slow Down``.
_TRANSIENT = re.compile(
    r"(?:Slow[_ ]?Down|Throttl|Request[_ ]?Timeout|timed out|"
    r"Connection[_ ]?reset|Connection[_ ]?aborted|temporarily unavailable|"
    r"Service[_ ]?Unavailable|Internal[_ ]?Error|Network[_ ]?Connection|" +
    _STATUS.format("50[0234]") + r")", re.I)
_RECONNECT = re.compile(
    r"(?:Expired[_ ]?Token|Token[_ ]?Refresh[_ ]?Required|"
    r"Invalid[_ ]?Access[_ ]?Key[_ ]?Id|Signature[_ ]?Does[_ ]?Not[_ ]?Match|"
    r"Invalid[_ ]?Token|Unauthorized|"
    r"Access Key Id you provided does not exist|" + _STATUS.format("401") +
    r")", re.I)
_DENIED = re.compile(
    r"(?:Access[_ ]?Denied|Forbidden|Unauthorized[_ ]?Access|" +
    _STATUS.format("403") + r")", re.I)
_MISSING = re.compile(
    r"(?:No[_ ]?Such[_ ]?Key|No[_ ]?Such[_ ]?Bucket|Resource[_ ]?Not[_ ]?"
    r"Found|Not[_ ]?Found|does not exist|" + _STATUS.format("404") + r")", re.I)


def classify_arrow_error(name: str, fmt: str,
                         error: BaseException) -> SourceError:
    """The object's own name is removed before anything is matched: an S3 key
    is free to spell ``reports/404/`` or ``access-denied/``.
    """
    text = str(error)
    signal = _QUOTED.sub("''", text)
    if _RECONNECT.search(signal):
        return ReconnectRequiredError(
            f"The credential this reader signed {name} with is no longer "
            f"accepted: {error}", table=name)
    if isinstance(error, PermissionError) or _DENIED.search(signal):
        return SourceAccessDeniedError(
            f"Kumo is not permitted to read {name}: {error}", table=name)
    if _TRANSIENT.search(signal):
        return TransientSourceError(
            f"Reading {name} failed and may succeed on a retry: {error}")
    if isinstance(error, FileNotFoundError) or _MISSING.search(signal):
        return SourceNotFoundError(f"{name} could not be found: {error}",
                                   table=name)
    return PermanentSourceError(f"{name} is not readable {fmt}: {error}",
                                table=name)


@contextmanager
def _arrow_errors(name: str, fmt: str = "Parquet") -> Iterator[None]:
    """Arrow reports an unreachable object as an ``ArrowIOError``, which is an
    ``OSError`` and not an ``ArrowException`` at all.
    """
    from ..csv_rules import NoValidSeparatorFoundError

    try:
        yield
    except SourceError:
        raise
    except (pa.ArrowException, OSError, NoValidSeparatorFoundError) as error:
        raise classify_arrow_error(name, fmt, error) from error


def _dtype_of(arrow_type: pa.DataType) -> Dtype:
    from ..types.conversions import pyarrow_to_dtype

    if pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    if pa.types.is_decimal(arrow_type):
        # An unsafe cast, matching what the hosted connector chose for a
        # DECIMAL(38,0) ID column.
        return Dtype.int64 if arrow_type.scale == 0 else Dtype.float32
    try:
        return pyarrow_to_dtype(arrow_type)
    except KeyError:
        return Dtype.unsupported


def _uncast(stored: pa.DataType, dtype: Any) -> bool:
    """Whether :func:`_projection` would leave ``stored`` alone. A dtype with
    no Arrow spelling -- the temporal family -- is written out as it was read,
    so agreeing on the dtype is agreement enough.
    """
    wanted = _pyarrow_type(dtype)
    if wanted is None:
        return _dtype_of(stored) is Dtype(dtype)
    return stored.equals(wanted)


def _pyarrow_type(dtype: Any) -> pa.DataType | None:
    from ..types.conversions import TO_PYARROW_DTYPE

    return TO_PYARROW_DTYPE.get(Dtype(dtype))


def _unpinned(columns: Mapping[str, Dtype]) -> list[str]:
    """Columns a CSV read leaves to Arrow's own inference, because the hosted
    export leaves them there too."""
    from .csv_format import pinned_types
    pinned = pinned_types(columns)
    return [name for name in columns if name not in pinned]


def _projection(
    table: str,
    schema: pa.Schema,
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str],
    *,
    normalize: bool = False,
    null_values: Sequence[str] = ARROW_NA_TOKENS,
    converts: bool = False,
    partitions: Sequence[str] = (),
) -> dict[str, Any]:
    missing = [name for name in columns if name not in schema.names]
    if missing:
        raise SourceNotFoundError(f"{table} has no column {', '.join(missing)}",
                                  table=table)
    unpinned = set(_unpinned(columns)) if normalize else set()
    projection = {}
    for name, dtype in columns.items():
        wanted = None if name in unpinned else _pyarrow_type(dtype)
        expression = pc.field(name)
        stored = schema.field(name).type
        if wanted is not None and not stored.equals(wanted):
            if normalize:
                from . import csv_format
                expression = csv_format.convert(expression, wanted,
                                                null_values=null_values)
            elif not converts or (name in partitions
                                  and pa.types.is_dictionary(stored)):
                expression = expression.cast(wanted)
        projection[aliases.get(name, name)] = expression
    return projection
