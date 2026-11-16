# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os.path as osp
from typing import Optional, TypeAlias

from kumo_core.types.schema import TableFingerprint

from kumo.data.common import TableRefreshStage
from service_framework.env_var import EnvVar

MAX_CACHE_SIZE = EnvVar(
    'MAX_CACHE_SIZE',
    type=int,
    required=False,
    default_value=48,
).parse_value()
CACHE_TTL = EnvVar(
    'CACHE_TTL',
    type=int,
    required=False,
    default_value=2 * 3600,  # 2 hours.
).parse_value()

# Unique key representing a table and version for indexing stats caches
TableSourceKey: TypeAlias = str


def table_source_key(source_name: str, connector_name: str,
                     data_version: str) -> TableSourceKey:
    payload = '\0'.join([source_name, connector_name, data_version])
    return hashlib.sha256(payload.encode()).hexdigest()


def table_source_path_component(source_name: str) -> str:
    return hashlib.sha256(source_name.encode()).hexdigest()


class TableCacheDir:
    """This class defines the directory layout of the data cache dir for a
    single table.

    Usage:
        TableCacheDir(table_dir).colstat_task_result

    """
    def __init__(self, table_dir: str,
                 table_fingerprint: Optional[TableFingerprint] = None) -> None:
        self._table_dir = table_dir
        self._table_fingerprint = table_fingerprint

    @property
    def col_stats(self) -> str:
        return osp.join(self._table_dir, "col_stats")

    @property
    def min_col_stats(self) -> str:
        return osp.join(self._table_dir, "min_col_stats")

    @property
    def table_stats(self) -> str:
        return osp.join(self._table_dir, "table_stats")

    def colstat_task_result(self, stage: TableRefreshStage) -> str:
        if stage == TableRefreshStage.COMPUTE_MIN_COL_STATS:
            return osp.join(self._table_dir, 'min_colstat', 'task_result.json')
        if stage == TableRefreshStage.COMPUTE_FULL_COL_STATS:
            return osp.join(self._table_dir, 'full_colstat', 'task_result.json')
        raise AssertionError(f'{stage} must be COMPUTE_[MIN|FULL]_COL_STATS')
