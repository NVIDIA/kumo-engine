# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import test.conftest as root_conftest

import pytest


def test_temporal_loopback_preflight_reports_socket_restriction(monkeypatch):
    def raise_permission_error(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(root_conftest.socket, "socket", raise_permission_error)

    with pytest.raises(pytest.fail.Exception,
                       match="does not allow loopback TCP sockets"):
        root_conftest._assert_loopback_sockets_available_for_temporal()
