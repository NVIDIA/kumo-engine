# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache lifecycle shared by local and hosted tests."""

import os

import pytest

import kumo.home
from io_utils import fsutil
from kumo import data


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return True


@pytest.fixture(autouse=True)
def backend_cache(auto_setup_kumo_backend: bool):
    if not auto_setup_kumo_backend:
        yield
        return
    os.makedirs(data.get_persist_dir(), exist_ok=True)
    try:
        yield
    finally:
        data.reset_active_cache_for_test()
        fsutil.rm(data.get_persist_dir(), ignore_errors=True)
        fsutil.rm(
            kumo.home.HomeDirManager(kumo.home.get_home_dir()).pseudo_time_dir,
            ignore_errors=True)
