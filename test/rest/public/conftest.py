# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-API test conftest.

Force ``BackendCacheManager.cache_dir`` to a local tmpdir for the
public-API test session. Otherwise ``cache.cache_dir`` resolves to a
remote ``data_plane_storage_root`` under CI matrix configurations that
export ``CUSTOM_S3_DATA_PLANE_BUCKET``, and
``test_public_api_snapshot.py::setup_teardown_test_module`` does
``fsutil.copy(cache.cache_dir, <tmp>)`` + inverse on teardown — many
minutes of S3 round-trips per test module.

Additive: does NOT touch ``setup_teardown_test_module`` (still required
for module-scope backup/restore isolation) or ``CUSTOM_ARTIFACT_ROOT``
(some integ tests need the S3 artifact root for GloVe resolution — see
``test/conftest.py:88-93``). Session-scoped + autouse so it runs before
the module-scope fixture during pytest setup ordering.
"""
from typing import Generator

import pytest

from kumo.data import get_active_cache
from kumo.data.backend_cache import CACHE_DIR


@pytest.fixture(scope='session', autouse=True)
def force_local_cache_dir_for_public_tests(
    tmp_path_factory: pytest.TempPathFactory,
    setup_rest_test_session,
) -> Generator[None, None, None]:
    """Force ``cache.cache_dir`` to a session-scoped local tmpdir.

    Runs after ``setup_rest_test_session`` (which calls ``kumo.init`` and
    materializes the ``BackendCacheManager`` singleton) but before any
    module/function-scope fixtures, so the override is visible to the
    ``setup_teardown_test_module`` fixture in
    ``test_public_api_snapshot.py``.
    """
    cache = get_active_cache()
    local_root = tmp_path_factory.mktemp('public_api_cache_dir')
    local_cache_dir = str(local_root / CACHE_DIR)
    cache.update_cache_dir(local_cache_dir)
    yield
