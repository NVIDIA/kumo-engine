# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingest progress that is visible when the application set up no logging."""

import logging
import sys


def _unconfigured(logger: logging.Logger) -> bool:
    if logger.disabled or logging.root.manager.disable >= logging.INFO:
        return False
    root = logging.getLogger()
    node: logging.Logger | None = logger
    while node is not None:
        if node.handlers or (node is not root and node.level):
            return False
        if node is root:
            return root.level == logging.WARNING
        if not node.propagate:
            return False
        node = node.parent
    return False


def log_progress(logger: logging.Logger, message: str, *args: object) -> None:
    """Log at INFO; print to stderr only if nothing in the chain is set up."""
    if _unconfigured(logger):
        try:
            print(message % args, file=sys.stderr, flush=True)
        except (OSError, ValueError):  # Progress must never fail an ingest.
            pass
    else:
        logger.info(message, *args)
