# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import logging.config
import multiprocessing
import sys
import time
from pathlib import Path

_BURST_RECORDS = 2000


def _child_log(logger_name):
    logger = logging.getLogger(logger_name)
    logger.info('from_child')
    logger.handlers[0].close()


def _wait_for_lines(path, count):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text().splitlines()) >= count:
            return
        time.sleep(0.01)
    raise RuntimeError(f'{path} did not receive {count} records')


def main():
    path = Path(sys.argv[1])
    logger_name = 'multiprocessing_log_helper'
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'plain': {
                'format': '%(message)s'
            }
        },
        'handlers': {
            'multiprocessing': {
                'class': 'io_utils.logging.MultiProcessingLog',
                'formatter': 'plain',
                'path': path,
                'mode': 'a',
                'maxsize': 1048576,
                'rotate': 1,
            }
        },
        'loggers': {
            logger_name: {
                'handlers': ['multiprocessing'],
                'level': 'INFO',
                'propagate': False,
            }
        },
    })
    logger = logging.getLogger(logger_name)
    logger.info('from_parent')
    _wait_for_lines(path, 1)

    process = multiprocessing.get_context('fork').Process(
        target=_child_log,
        args=(logger_name, ),
        daemon=True,
    )
    try:
        process.start()
        process.join(5.0)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(
                f'logging child did not exit cleanly: {process.exitcode}')
        _wait_for_lines(path, 2)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)

    for index in range(_BURST_RECORDS):
        logger.info('burst_%d', index)


if __name__ == '__main__':
    main()
