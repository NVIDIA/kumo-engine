# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules an exported artifact obeys, wherever the export runs.

A prediction artifact leaves Kumo either through a hosted workflow activity
(``rest/backends/ml_backend/artifact_export.py``) or through a local sink
(``kumo_core.object_store_sink.ObjectStoreSinkBackend``). The rules differ in
what they write with and how they classify a refusal; they must not differ in
*what* they write. Both rules below are that shared part, and every side calls them.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

#: Recorded beside a directory destination's parts, so that a retry of *this*
#: export is told from a foreign occupant of the same prefix.
EXPORT_MARKER_NAME = "_KUMO_EXPORT"

#: What a CSV destination cannot represent, worded once. Carried by
#: :class:`MultiDimensionalCsvError`; a caller that needs a retry
#: classification attaches one at its own boundary.
MULTI_DIMENSIONAL_CSV = (
    "Cannot export multi-dimensional array columns to CSV. Please use Parquet "
    "format for data with embedding or array columns, or ensure arrays are "
    "1-dimensional.")


class MultiDimensionalCsvError(ValueError):
    """The one refusal :func:`flatten_arrays_for_csv` makes on purpose.

    A distinct type rather than a bare :class:`ValueError` because a caller
    translates this refusal into its own vocabulary -- the hosted activity
    restates it as a non-retryable ``ArtifactExportError`` -- and a bare
    ``ValueError`` gives that caller no way to tell *this* refusal from any
    other ``ValueError`` raised beneath it by pandas, numpy or PyArrow. Caught
    by type, an unrelated failure keeps its own identity and its own
    retryability instead of reaching a customer worded as this one.

    Still a :class:`ValueError` so that a caller which only wants "the frame
    was rejected" needs no import.
    """


def export_marker(identity: str) -> str:
    """Return the ``_KUMO_EXPORT`` marker recording :obj:`identity`.

    A compatibility contract rather than an implementation detail: a directory
    written by one backend is recognised as *the same export* by the other
    only while both digest the identity the same way. Keyed on the identity of
    the artifact being written, never on its contents -- contents differ
    between a write and its retry, identities do not.
    """
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def flatten_arrays_for_csv(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Return :obj:`frame` with array-valued columns rendered for CSV.

    An embedding column holds a one-dimensional array per row and CSV has no
    column for one, so it becomes a space-separated string. A row holding
    anything deeper is refused with :data:`MULTI_DIMENSIONAL_CSV` rather than
    flattened, because there is no reading of a nested array that round-trips.

    A list or tuple is rendered as well as the ``numpy`` array a frame read
    back from Parquet carries, so that a frame handed straight to an export
    lands the same way as one that went through a file.

    Shallow-copied rather than mutated: the frame belongs to the caller, and
    only the columns actually rewritten are materialised. A deep copy would
    double the peak of an export whose reason to be streamed is that the frame
    is large.

    Raises:
        MultiDimensionalCsvError: If a column holds an array of more than one
            dimension.
    """
    columns = [name for name in frame.columns if _renderable(frame[name])]
    if not columns:
        return frame
    flattened = frame.copy(deep=False)
    for name in columns:
        # Iterated rather than passed to `Series.apply`, which cannot read an
        # Arrow-backed list column on pandas 1.5 -- the low end of this
        # package's declared range. `apply` there begins with
        # `obj.astype(object)._values`, and a list column's object form is a
        # 2-D array of rows, which `check_ndim` rejects with `ValueError:
        # Wrong number of dimensions. values.ndim > ndim [2 > 1]` before the
        # rendering function is called even once. Iteration reads the column
        # as it stands; assignment lets the rendered strings pick their own
        # dtype.
        flattened[name] = [_rendered(value) for value in flattened[name]]
    return flattened


def _rendered(value: Any) -> Any:
    """Render one cell, or refuse it."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise MultiDimensionalCsvError(MULTI_DIMENSIONAL_CSV)
        return " ".join(str(item) for item in value)
    if isinstance(value, (list, tuple)):
        # Read directly rather than through `np.asarray`, whose refusal for a
        # ragged row is numpy's "inhomogeneous shape" and says nothing about
        # which format to export in.
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            raise MultiDimensionalCsvError(MULTI_DIMENSIONAL_CSV)
        return " ".join(str(item) for item in value)
    return value


def _renderable(column: "pd.Series") -> bool:
    """Whether :obj:`column` may hold a value CSV has no column for.

    Object dtype is one way to hold an array and the only way a frame read
    back with the default backend ever holds one; a frame read with
    ``dtype_backend='pyarrow'`` holds the same embeddings as an Arrow list
    column instead, which is not object dtype and would otherwise walk past
    this into PyArrow's ``Unsupported Type: list<item: double>``.
    """
    import pandas as pd

    if pd.api.types.is_object_dtype(column):
        return True
    arrow = getattr(column.dtype, "pyarrow_dtype", None)
    if arrow is None:
        return False
    import pyarrow as pa

    return (pa.types.is_list(arrow) or pa.types.is_large_list(arrow)
            or pa.types.is_fixed_size_list(arrow))


class ArtifactDestinationConflict(ValueError):
    """A directory belongs to another export."""


def copy_selected_parquet_files(files, output_path, *, fs):
    """Hosted selected-file copying, with bounded transfer concurrency."""
    import os.path as osp
    if len(files) == 1 and not output_path.endswith('/'):
        fs.copy_file(files[0], output_path, overwrite=True)
        return
    fs.makedirs(output_path, exist_ok=True)
    fs.copy_files_threaded([(path, osp.join(output_path, osp.basename(path)))
                            for path in files], overwrite=True, max_workers=8)


def prepare_export_destination(output_path, *, output_is_dir, file_type, fs,
                               identity):
    """Apply the hosted filename and destination ownership rules before copying."""
    import os.path as osp
    if output_is_dir:
        output_dir = output_path.rstrip('/') + '/'
    else:
        suffix = '.' + file_type
        if not output_path.casefold().endswith(suffix):
            output_path += suffix
        output_dir = osp.dirname(output_path) + '/'
    output_path_exists = False
    if output_is_dir:
        marker_path = osp.join(output_dir, EXPORT_MARKER_NAME)
        marker = export_marker(identity)
        if fs.exists(output_dir):
            entries = fs.listdir(output_dir, max_items=1)
            if entries and (not fs.exists(marker_path)
                            or fs.read_text(marker_path) != marker):
                raise ArtifactDestinationConflict(
                    f"Output directory '{output_dir}' already contains "
                    'data from another export. Choose an empty destination path.'
                )
            output_path_exists = True
    fs.makedirs(output_dir, exist_ok=True)
    if output_is_dir:
        fs.write_text(marker_path, marker)
        success_path = osp.join(output_dir, '_SUCCESS')
        if fs.exists(success_path):
            fs.rm(success_path)

    return output_path, output_path_exists


def publish_export_destination(output_path, *, fs):
    """Publish success only after the caller's copy/conversion has completed."""
    import os.path as osp
    if fs.isdir(output_path):
        fs.touch(osp.join(output_path, '_SUCCESS'))
        return fs.dirsize(output_path), output_path.rstrip('/') + '/'
    return fs.filesize(output_path) or 0, output_path


def export_selected_files(files, output_path, *, fs, identity,
                          file_type='parquet', convert_csv=None,
                          copy_parquet=None, temporary_directory=None,
                          sidecars={}):
    """Publish the hosted selected-file layout and success marker last.

    Filesystem/authentication and retry wrappers stay with the caller. The
    directory ownership, output names and CSV retry decisions are shared.
    ``sidecars`` (name to bytes) are written into a directory destination
    after the data and before ``_SUCCESS``; ``fs`` then needs ``write_bytes``.
    """
    import os.path as osp
    import tempfile

    output_is_dir = output_path.endswith('/') or len(files) > 1
    output_path, _ = prepare_export_destination(
        output_path, output_is_dir=output_is_dir, file_type=file_type, fs=fs,
        identity=identity)

    if file_type == 'parquet':
        if copy_parquet is None:
            copy_selected_parquet_files(files, output_path, fs=fs)
        else:
            copy_parquet(files, output_path)
    elif file_type == 'csv':
        single = len(files) == 1 and not output_path.endswith('/')
        for index, path in enumerate(files):
            destination = (output_path.rstrip('/') if single else osp.join(
                output_path, f'part-{index:05}.csv'))
            with (temporary_directory
                  or tempfile.TemporaryDirectory)() as scratch:
                staged = osp.join(scratch, osp.basename(destination))
                (convert_csv or parquet_to_csv)(path, staged)
                fs.copy_file(staged, destination, overwrite=True)
    else:
        raise ValueError(f'Unsupported artifact file type: {file_type}')

    if output_is_dir:
        for name, payload in sidecars.items():
            fs.write_bytes(osp.join(output_path, name), payload)
    return publish_export_destination(output_path, fs=fs)


def artifact_rows_per_batch(avg_row_size, target_bytes=2 << 20):
    """The existing hosted 2-MiB target and 128-Krow ceiling."""
    import math
    return min(math.ceil(target_bytes / avg_row_size), 128 << 10)


def local_parquet_batches(path):
    """Read a local artifact in batches under the shared export size bound."""
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    if not metadata.num_rows:
        return
    size = sum(
        metadata.row_group(i).total_byte_size
        for i in range(metadata.num_row_groups))
    rows = artifact_rows_per_batch(max(1, size / metadata.num_rows))
    yield from parquet.iter_batches(batch_size=rows, use_threads=False)


def parquet_to_csv(source, destination, *, batches=None, csv_ready=None):
    """The hosted incremental CSV conversion; batches retain its 2-MiB bound."""
    import pyarrow as pa
    from pyarrow.csv import CSVWriter
    frames = (batch.to_pandas() for batch in local_parquet_batches(source)
              ) if batches is None else batches(source)
    with open(destination, 'wb') as handle:
        writer = None
        try:
            for frame in frames:
                table = pa.Table.from_pandas((csv_ready
                                              or flatten_arrays_for_csv)(frame))
                if writer is None:
                    writer = CSVWriter(handle, table.schema)
                writer.write_table(table)
        finally:
            if writer:
                writer.close()


@contextmanager
def prepared_artifact_file(path, extra_columns=None, *, parquet_options=None,
                           scratch_root=None):
    """Stamp only this destination's columns, staging at most one source file."""
    if not extra_columns and not parquet_options:
        yield path
        return
    import os.path as osp
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix='kumo-artifact-columns-',
                                     dir=scratch_root) as scratch:
        staged = osp.join(scratch, osp.basename(path))
        writer = None
        try:
            for batch in local_parquet_batches(path):
                frame = batch.to_pandas()
                for name, value in (extra_columns or {}).items():
                    frame[name] = value
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(staged, table.schema,
                                              **(parquet_options or {}))
                writer.write_table(table)
        finally:
            if writer:
                writer.close()
        if writer is None:
            # Keep a readable zero-row artifact for legacy completed records.
            table = pq.ParquetFile(path).schema_arrow.empty_table()
            frame = table.to_pandas()
            for name, value in (extra_columns or {}).items():
                frame[name] = value
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                           staged, **(parquet_options or {}))
        yield staged
