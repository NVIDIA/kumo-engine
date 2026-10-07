# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os.path as osp
from typing import List

from kumo_core.types.schema import TableFingerprint

from io_utils import fsutil
from kumo.data import MaterializationStage, get_persist_dir

TEMP_CONNECTOR_NAME = 'temp_table_dataplane_connector__'
ENTITY_INCLUDE_BIT = '__kumo_internal_entity_inclusive_bit_column__'


def get_table_path(pquery_name: str, base_dir: str,
                   training_table: bool = True):
    """
    Generate the train/pred table output path (``<base_dir>/<suffix>/<name>``).

    ``base_dir`` is required: the calling ``generate_*`` method resolves it
    (the per-job home dir, or ``get_home_dir()`` for SDK/test callers) before
    calling, so this helper never touches the process-global home dir.
    """
    from kumo.pquery.predictive_query import PRED_TABLE_NAME, TRAIN_TABLE_NAME
    table_suffix = TRAIN_TABLE_NAME if training_table else PRED_TABLE_NAME
    return osp.join(base_dir, table_suffix, pquery_name)


def get_materialize_path(table_fingerprint: TableFingerprint, data_version: str,
                         stage: MaterializationStage):
    r"""Get the persist-dir path for a table at a materialization stage.

    If the stage is INGEST, we use the ingest path fingerprint to identify the
    path. Otherwise, we use the full fingerprint.

    Always returns a path under the persistent storage directory
    (``get_persist_dir()``). Callers that need the local ingest root
    should use :func:`get_diskgraph_local_ingest_path` instead.

    Args:
        table_fingerprint (TableFingerprint)
        data_version (str): table version
        stage (MaterializationStage): Materialization stage.
    """
    root_path = get_persist_dir()
    if root_path is None:
        raise ValueError("Cannot persist table data if persist dir is not "
                         "explicitly set. Set the persist dir with kumo.init")

    if stage == MaterializationStage.INGEST:
        materialize_path = osp.join(root_path, stage.value,
                                    table_fingerprint.ingest_path_fpid,
                                    data_version)
    else:
        materialize_path = osp.join(root_path, stage.value,
                                    table_fingerprint.fpid(), data_version)

    if not fsutil.exists(materialize_path) or fsutil.isdir(materialize_path):
        return materialize_path
    else:
        raise ValueError(f"Materialization path '{materialize_path}' "
                         f"is not a directory.")


def get_diskgraph_local_ingest_path(table_fingerprint,
                                    data_version: str) -> str:
    """Return the local NVMe path for INGEST data.

    This path remains available to standalone table-stat and preview readers
    for snapshots produced by older releases. It is always under the local
    DiskGraph ingest root, never the remote persist dir.
    """
    from statedb.config.diskgraph import get_diskgraph_local_ingest_root
    local_root = get_diskgraph_local_ingest_root()
    return osp.join(local_root, MaterializationStage.INGEST.value,
                    table_fingerprint.ingest_path_fpid, data_version)


def get_diskgraph_snapshot_ingest_dir(snapshot_id: str) -> str:
    """Per-snapshot tmp dir for source parquets staged during pool ingest.

    The pool worker writes per-snapshot staging here so cleanup can rmtree
    the whole subtree without touching other snapshots' downloads.
    """
    from statedb.config.diskgraph import get_diskgraph_local_ingest_root
    local_root = get_diskgraph_local_ingest_root()
    return osp.join(local_root, MaterializationStage.INGEST.value, snapshot_id)


def get_diskgraph_snapshot_ingest_path(
    snapshot_id: str,
    table_fingerprint: TableFingerprint,
    data_version: str,
) -> str:
    """Per-table source-parquet path under a snapshot's tmp dir."""
    return osp.join(get_diskgraph_snapshot_ingest_dir(snapshot_id),
                    table_fingerprint.ingest_path_fpid, data_version)


def get_columns_from_query(sql_query: str) -> List[str]:
    """
    Returns all the column names that appear in `sql_query`.
    """
    from sql_metadata import Parser
    columns = Parser(sql_query).columns
    return columns
