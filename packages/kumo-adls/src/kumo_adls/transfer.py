# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from contextlib import contextmanager

from kumo_core.sources.cpu import copy_workers

# The shared copy worker reserves 64 MiB. Serial downloads retain the initial
# response plus a current chunk and transport buffers; this is a payload
# estimate, not a cap on SDK allocations or process RSS.
INITIAL_GET_BYTES = 16 << 20
CHUNK_BYTES = 4 << 20
BULK_DOWNLOAD_CHUNK_BYTES = 16 << 20
MAX_BULK_DOWNLOAD_CONCURRENCY = 1
NETWORK_BLOCK_BYTES = 16 << 20
MAX_FILE_CONCURRENCY = 4
TRANSFER_WORKERS_ENV = 'KUMO_ADLS_TRANSFER_WORKERS'


def _transfer_workers():
    ceiling = copy_workers()
    value = os.environ.get(TRANSFER_WORKERS_ENV)
    if value is None:
        return ceiling
    try:
        limit = int(value)
    except ValueError:
        raise ValueError(
            f'{TRANSFER_WORKERS_ENV} must be a positive integer') from None
    if limit < 1:
        raise ValueError(f'{TRANSFER_WORKERS_ENV} must be a positive integer')
    return min(ceiling, limit)


CONNECTIONS = _transfer_workers()
_SLOTS = threading.BoundedSemaphore(CONNECTIONS)


@contextmanager
def concurrency(size, *, upload=False, ranged=False):
    """Share transfer slots across files/readers/accounts, excluding metadata."""
    initial = 0 if upload or ranged else INITIAL_GET_BYTES
    chunk_bytes = CHUNK_BYTES if upload else BULK_DOWNLOAD_CHUNK_BYTES
    maximum = MAX_BULK_DOWNLOAD_CONCURRENCY if ranged else MAX_FILE_CONCURRENCY
    wanted = min(
        maximum,
        max(1, (max(0, size - initial) + chunk_bytes - 1) // chunk_bytes))
    _SLOTS.acquire()
    acquired = 1
    try:
        while acquired < wanted and _SLOTS.acquire(blocking=False):
            acquired += 1
        yield acquired
    finally:
        for _ in range(acquired):
            _SLOTS.release()
