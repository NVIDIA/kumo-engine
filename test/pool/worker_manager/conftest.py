# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def _clear_pod_service_account(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    # Manager construction falls back to the legacy SA name unless a test
    # sets POD_SERVICE_ACCOUNT itself; guarantee a clean slate even when
    # the test runner executes inside a pod with the downward API.
    monkeypatch.delenv('POD_SERVICE_ACCOUNT', raising=False)
