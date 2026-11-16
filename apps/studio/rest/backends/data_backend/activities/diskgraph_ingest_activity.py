# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connector-side cleanup activity for the diskgraph pool.

This module retains:

  * ``cleanup_source_files`` — the per-snapshot ingest tmp-dir cleanup
    activity dispatched by ``DiskgraphIngestWorkflow``.
The legacy download/ingest/success-marker shims (``download_table_files``,
``diskgraph_ingest_activity_v2``, ``check_diskgraph_output_v2``,
``write_diskgraph_success_marker_v2``) were removed: ingest and the
success-marker probe/write now run inside the pool daemon via
``DiskgraphService`` directly, so none of those Temporal wrappers were
dispatched by any workflow.
"""
import shutil

from temporalio import activity

from io_utils import fsutil, logging
from kumo.data.utils import get_diskgraph_snapshot_ingest_dir

log = logging.getLogger(__name__)

__all__ = [
    'cleanup_source_files',
]


@activity.defn
def cleanup_source_files(snapshot_id: str) -> None:
    """Rmtree the per-snapshot ingest tmp dir (no-op if absent)."""
    path = get_diskgraph_snapshot_ingest_dir(snapshot_id)
    if fsutil.exists(path):
        assert fsutil.isdir(path), (
            f'cleanup_source_files: expected directory at {path}')
        log.info('cleanup_source_files: removing snapshot=%s path=%s',
                 snapshot_id, path)
        shutil.rmtree(path)
    else:
        log.info('cleanup_source_files: noop snapshot=%s (nothing staged)',
                 snapshot_id)
