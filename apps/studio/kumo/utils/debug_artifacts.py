# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

logger = logging.getLogger(__name__)


def write_debug_artifact(
    component: str,
    stage: str,
    event: str,
    **kwargs,
) -> None:
    """Log a debug artifact for diskgraph diagnostics."""
    logger.debug(
        "debug_artifact: component=%s stage=%s event=%s data=%s",
        component,
        stage,
        event,
        json.dumps({
            k: str(v)
            for k, v in kwargs.items()
        }, default=str),
    )
