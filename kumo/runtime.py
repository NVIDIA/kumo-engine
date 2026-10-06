# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process initialization for monolith services and command-line workers."""

from __future__ import annotations

import os
import os.path as osp
from typing import Any


def init(
    name: str | None = None,
    home_dir: str | None = None,
    initialize_logging: bool = True,
    initialize_tracing: bool = False,
    logging_mode: Any = None,
) -> None:  # pragma: no cover
    """Initialize process-local logging, tracing, metrics, and storage paths."""
    from io_utils import fsutil, logging
    from kumo.home import get_local_home_dir, set_home_dir

    if home_dir is not None:
        set_home_dir(home_dir)
    if logging_mode is None:
        logging_mode = logging.LoggingMode.DEBUG
    if initialize_logging:
        logging_dir = osp.join(get_local_home_dir(), "logging")
        fsutil.makedirs(logging_dir, exist_ok=True)
        logging.initialize_logging(
            mode=logging_mode,
            logging_dir=logging_dir,
        )

    name = name or os.path.basename(__file__)
    if initialize_tracing:
        from monitoring import tracing

        tracing.initialize_tracing(
            name=name,
            span_exporters=[
                tracing.SpanExporterType.FILE,
                tracing.SpanExporterType.JAEGER,
            ],
        )
        tracing.enable_tracing()

    from kumo.metrics import MetricsClient

    MetricsClient().job_name = name


__all__ = ["init"]
