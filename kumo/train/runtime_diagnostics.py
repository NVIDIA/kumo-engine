# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import subprocess
from collections.abc import Sequence

from service_framework.cloud_metadata import CloudMetadataFactory

logger = logging.getLogger(__name__)


def get_instance_id() -> str:
    """Return the current host identity from the configured metadata source."""
    return CloudMetadataFactory.create_provider().get_instance_id()


def run_command_with_logging(command: Sequence[str]) -> None:
    """Run a diagnostic command and log its combined output."""
    try:
        logger.info("Running command: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            logger.warning("Diagnostic command produced no output stream")
            return
        for line in process.stdout:
            logger.info("%s", line.rstrip())
    except FileNotFoundError:
        logger.error("Diagnostic command %s was not found", command[0])


def run_all_check_commands() -> None:
    """Log host memory, disk, and available GPU diagnostics."""
    run_command_with_logging(["free", "-h"])
    run_command_with_logging(["df", "-h"])
    if shutil.which("nvidia-smi") is not None:
        run_command_with_logging(["nvidia-smi"])
