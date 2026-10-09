# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-directory autouse fixture to reset the in-memory + on-disk
``BackendCacheManager`` state between sibling tests.

The data persist dir is derived from ``deploy_env()``
(``kumo.data.get_persist_dir``) and is shared across sibling tests, so it
can leak ``MATERIALIZED_SIZE`` / ``SOURCE_SIZE`` entries (read-through merge
semantics in ``ReadThroughStatsCache.update``) into tests whose synthetic
parquet fixtures collide on ``data_version`` (parquet mtime, same second).
Resetting both the in-memory active cache and wiping the on-disk cache under
``get_persist_dir()`` between tests breaks the leak.
"""
import os.path as osp

import pytest

from io_utils import fsutil
from kumo.data import get_persist_dir, reset_active_cache_for_test
from kumo.data.backend_cache import CACHE_DIR


def _wipe_backend_cache_state() -> None:
    """Drop the lazily-built ``BackendCacheManager`` and remove its
    on-disk persisted entries under the shared ``data_persist_dir``
    that ``test/conftest.py::run_before_and_after_tests`` points at."""
    # get_persist_dir() derives from deploy_env() and is always set.
    persist_dir = get_persist_dir()
    fsutil.rm(osp.join(persist_dir, CACHE_DIR), ignore_errors=True)
    reset_active_cache_for_test(None)


@pytest.fixture(autouse=True)
def _reset_backend_cache_between_tests():
    _wipe_backend_cache_state()
    yield
    _wipe_backend_cache_state()
