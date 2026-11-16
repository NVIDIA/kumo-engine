# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.version import get_kumo_version

from io_utils import logging

logger = logging.getLogger(__name__)


def log_app_startup() -> None:
    """Emit a log entry recording the app version at startup.

    Intended to be called once at startup from long-running service entry points
    (REST API, Temporal workers, monitoring).
    """
    logger.info("App startup: image_version=%s", get_kumo_version())
