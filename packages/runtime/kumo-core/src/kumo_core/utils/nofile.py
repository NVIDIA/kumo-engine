# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-file limit for DataLoader workers that pass batches as fds."""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

RECOMMENDED_NOFILE = 65535


@functools.cache
def ensure_nofile() -> None:
    """Raise the soft RLIMIT_NOFILE to the hard limit, once per process.

    Never touches the hard limit. Set ``KUMO_RAISE_NOFILE=0`` to skip.
    """
    if os.environ.get("KUMO_RAISE_NOFILE") == "0":
        return
    try:
        import resource
    except ImportError:
        return
    infinite = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = RECOMMENDED_NOFILE if hard == infinite else hard
    if soft != infinite and soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (OSError, ValueError):
            pass
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != infinite and soft < RECOMMENDED_NOFILE:
        logger.warning(
            "The open-file limit is soft=%s hard=%s. DataLoader workers pass "
            "every batch tensor as an open fd and may fail with 'received 0 "
            "items of ancdata'. Raise it to at least %d via "
            "/etc/security/limits.conf (nofile), systemd LimitNOFILE, or "
            "docker --ulimit nofile=%d:%d. Set KUMO_RAISE_NOFILE=0 to stop "
            "Kumo from changing the limit.", soft, hard, RECOMMENDED_NOFILE,
            RECOMMENDED_NOFILE, RECOMMENDED_NOFILE)
