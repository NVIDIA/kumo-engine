# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import os.path as osp
import shutil
import tempfile
from typing import Mapping

from kumo_core.sources.cpu import cpu_budget
from kumo_core.sources.prepare import read_table
from kumo_core.sources.protocol import SourceReader
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_core.types.schema import (
    IngestedTableFingerprint,
    SqlTableFingerprint,
)
from kumoapi.typing import Dtype

from io_utils import fsutil
from kumo.connector.connector import ExportMetrics
from kumo.data import MaterializationStage
from kumo.data.utils import get_materialize_path

logger = logging.getLogger(__name__)


def _scratch_dir(data_version: str) -> str:
    try:
        from statedb.config.diskgraph import get_diskgraph_local_ingest_root
        root = osp.join(get_diskgraph_local_ingest_root(), 'source-reader')
        os.makedirs(root, exist_ok=True)
    except Exception:
        # Not every deploy env that reads a table has a DiskgraphConfig.
        return tempfile.mkdtemp(prefix=f'{data_version}-')
    return tempfile.mkdtemp(prefix=f'{data_version}-', dir=root)


def _metrics(staged: StagedTable) -> ExportMetrics:
    return ExportMetrics(num_rows=staged.num_rows,
                         source_size=staged.source_size)


def worker_ceiling(graph_table_count: int | None) -> int | None:
    """A read sized from the whole CPU budget would claim it once per table,
    but only as many reads as the worker has threads can ever run at once."""
    if graph_table_count is None:
        return None
    from rest.constants import DATA_BACKEND_WORKER_NAME
    from temporalio_commons.worker_main import GLOBAL_WORKER_OPTIONS
    pool_width = GLOBAL_WORKER_OPTIONS[
        DATA_BACKEND_WORKER_NAME].max_worker_threads
    assert pool_width is not None
    return max(1, cpu_budget() // min(max(graph_table_count, 1), pool_width))


def read_source(
    reader: SourceReader,
    source_name: str,
    table_fp: IngestedTableFingerprint | SqlTableFingerprint,
    data_version: str,
    *,
    graph_table_count: int | None = None,
    sql: str | None = None,
) -> ExportMetrics:
    r"""``data_version`` is the caller's, because a reader that re-stamped
    here could disagree with the path its own output is written to.

    ``sql`` reads a Custom SQL query instead of the physical table;
    ``source_name`` is then only the name staging prefixes and errors key on.
    """
    return stage_source(
        reader, source_name,
        get_materialize_path(table_fp, data_version,
                             MaterializationStage.INGEST),
        columns=dict(table_fp.cols), sql=sql, version=data_version,
        graph_table_count=graph_table_count, scratch_key=data_version)


def stage_source(
    reader: SourceReader,
    source_name: str,
    destination: str,
    *,
    columns: Mapping[str, Dtype],
    sql: str | None = None,
    mode: ReadMode | None = None,
    version: str | None = None,
    graph_table_count: int | None = None,
    scratch_key: str = 'custom-source',
) -> ExportMetrics:
    r"""``read_source`` addressed by destination rather than by fingerprint: a
    custom training or prediction table is named by the job that asked for it,
    so there is no ingest path to derive.

    ``mode`` pins the read instead of taking the reader's cheapest offer. A
    caller that must not write to the source account pins
    :attr:`ReadMode.STREAMED`, because every staged mode unloads through
    provider storage.

    ``version`` supplies the stamp instead of having the reader derive one.
    Only :attr:`StagedTable.version` carries it and this function returns
    :class:`ExportMetrics`, so a derived stamp is discarded -- and deriving
    it costs a round trip per read (``DESCRIBE HISTORY`` on Databricks). A
    caller whose output is addressed by destination rather than keyed by
    version passes any stable string to skip it.
    """
    scratch = _scratch_dir(scratch_key)
    try:
        staged = read_table(reader, source_name, scratch, columns=columns,
                            sql=sql, mode=mode, version=version,
                            worker_ceiling=worker_ceiling(graph_table_count))
        publish_source(staged, destination)
        logger.info(
            'Read %s with a %s source reader (mode=%s rows=%s scanned=%s)',
            source_name, reader.source_type.value, staged.mode_used.value,
            staged.num_rows, staged.scanned_bytes)
        return _metrics(staged)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def publish_source(staged: StagedTable, destination: str) -> ExportMetrics:
    # A partial delete would mix shards from different attempts.
    try:
        fsutil.rm(destination, ignore_errors=False)
    except Exception as error:
        raise RuntimeError(
            f'could not clear {destination} before publishing; a partial delete '
            'would mix the previous attempt\'s shards into this one') from error
    fsutil.copy(staged.directory, destination, overwrite=True)
    return _metrics(staged)
