# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins legacy DiskGraph imports to the portable native implementation."""

from __future__ import annotations

import kumo_core.diskgraph.runner as core_runner
import pytest

from temporalio_commons.error_utils import is_error_non_retryable


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


def test_platform_translates_portable_retry_classification() -> None:
    assert is_error_non_retryable(core_runner.NonRetryableDiskgraphError("bad"))
    assert not is_error_non_retryable(core_runner.DiskgraphNoSpaceError("full"))
