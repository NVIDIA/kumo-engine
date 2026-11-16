# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as py_logging
import subprocess

from io_utils import logging, logging_util

logger = logging.getLogger(__name__)


def test_logging_from_subprocess(caplog):
    process = subprocess.Popen(['sleep 1; echo hello_world'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True)

    with caplog.at_level(py_logging.INFO):
        logging_thread = logging_util.LoggingThread('test', process)
        logging_thread.start()
        process.wait()
        logging_thread.join()

        assert 'hello_world' in caplog.text
