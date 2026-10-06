# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from typing import Optional

from io_utils import logging
from kumo.data.backend_cache import BackendCacheManager
from kumo.data.backend_utils import normalize_persist_dir
from kumo.data.common import DATA_BACKEND_THREAD_POOL, MaterializationStage
from kumo_core.types.conversions import DataFrame

logger = logging.getLogger(__name__)

__all__ = [
    'DATA_BACKEND_THREAD_POOL',
    'DataFrame',
    'MaterializationStage',
    'get_active_cache',
    'reset_active_cache_for_test',
]


# The external role setting, available in IaC2.0.
def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


REMATERIALIZE_ALL_EDGES = _env_bool('REMATERIALIZE_ALL_EDGES')

# Split cols for stats calculation.
MAX_COLS_STAT_BATCH_COLS = _env_int('MAX_COLS_STAT_BATCH_COLS', 25)

# The number of classes we support for a multi-class classification task or a
# multi-label task.
MAX_NUM_CLASSES = _env_int('MAX_NUM_CLASSES', 6000)

ADD_DF_TRACING = _env_bool('ADD_DF_TRACING')


# Exception to raise when creating session is interrupted/aborted.
class AbortCreateSessionError(Exception):
    pass


# Directory name, under the data-plane storage root, that holds materialized
# table data (ingested tables, sliced tables, materialized feature/graph).
# During ingestion Parquet is the canonical format; an ingested table lands at
# ``<persist_dir>/ingest/{table_name}/{data_version}`` (see
# ``data.utils.get_materialize_path``).
DATA_PERSIST_DIR_NAME = 'data-persist-dir'


def get_persist_dir() -> str:
    r"""Resolve the data persist directory from the deploy environment.

    Returns ``<data_plane_storage_root>/data-persist-dir``. This used to be
    read from a process-global ``GlobalDataBackendState`` populated by
    ``set_data_backend`` at worker/app startup; that global was removed in
    favor of deriving the value directly from :func:`deploy_env`, which
    resolves to the identical path in every REST / worker / pool / subprocess
    context (all prior callers set it to exactly this value).
    """
    from service_framework.deploy_env_config import deploy_env
    return normalize_persist_dir(
        osp.join(deploy_env().data_plane_storage_root, DATA_PERSIST_DIR_NAME))


_active_cache: Optional[BackendCacheManager] = None


def get_active_cache() -> BackendCacheManager:
    """Return the process-wide :class:`BackendCacheManager`.

    Lazy-init on first call; rooted at ``<persist_dir>/data_backend_cache``.
    Replaces the old ``get_data_backend().cache()`` accessor.
    """
    global _active_cache
    if _active_cache is None:
        persist_dir = get_persist_dir()
        # ``CACHE_DIR`` import is inlined to avoid a module-load cycle
        # with ``kumo.data.backend_cache``.
        from kumo.data.backend_cache import CACHE_DIR, BackendCacheManager
        _active_cache = BackendCacheManager(osp.join(persist_dir, CACHE_DIR))
    return _active_cache


def reset_active_cache_for_test(
    cache: Optional[BackendCacheManager] = None, ) -> None:
    """Test seam to replace or clear the active :class:`BackendCacheManager`."""
    global _active_cache
    _active_cache = cache
