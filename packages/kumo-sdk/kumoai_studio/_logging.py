# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

_ENV_KUMO_LOG = "KUMO_LOG"


def initialize_logging() -> None:
    r"""Initializes Kumo logging."""
    logger: logging.Logger = logging.getLogger('kumoai_studio')

    # From openai-python/blob/main/src/openai/_utils/_logs.py#L4
    logging.basicConfig(
        format=(
            "[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    default_level = os.getenv(_ENV_KUMO_LOG, "INFO")
    try:
        logger.setLevel(default_level)
    except (TypeError, ValueError):
        logger.setLevel(logging.INFO)
        logger.warning(
            "Logging level %s could not be properly parsed. "
            "Defaulting to INFO log level.", default_level)

    for name in ["matplotlib", "urllib3", "snowflake"]:
        # TODO(dm) required for spcs
        logging.getLogger(name).setLevel(logging.ERROR)
