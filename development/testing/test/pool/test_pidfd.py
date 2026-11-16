# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Linux pidfd wrappers used by process-session ownership."""
import os
import sys

import pytest

import pool.pidfd as pidfd_module


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='pidfds require Linux')
def test_pidfd_syscall_fallback_opens_polls_and_signals_current_process(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(pidfd_module.os, 'pidfd_open', raising=False)
    monkeypatch.delattr(pidfd_module.signal, 'pidfd_send_signal', raising=False)

    pidfd = pidfd_module.pidfd_open(os.getpid())
    try:
        assert not pidfd_module.pidfd_exited(pidfd)
        pidfd_module.pidfd_send_signal(pidfd, 0)
    finally:
        os.close(pidfd)
