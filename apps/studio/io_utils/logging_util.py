# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import threading

from io_utils import logging


class LoggingThread(threading.Thread):
    """A logging thread that relays a subprocess's stdout to the logger."""
    def __init__(self, name: str, proc: subprocess.Popen):
        super().__init__(name=name, daemon=True)
        self.proc = proc

    # NOTE: Subprocess logs to stdout which use subprocess.PIPE
    # the default buffer size of pipe is 8192 bytes (io.DEFAULT_BUFFER_SIZE)
    # In the rare scenario when the subprocess emits too many logs and
    # fills up the buffer faster than reading, the subprocess will just
    # hang i.e. it will block at logging.
    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting reading logs for %s", self.name)
        while self.proc.poll() is None:
            if self.proc.stdout is not None:
                line = self.proc.stdout.readline()
                # In the small time window where the process is terminating,
                # the stdout.readline returns empty lines. Avoid logging them.
                if line:
                    logger.info(line.strip())

        logger.info("Stopped reading subprocess logs for %s", self.name)
