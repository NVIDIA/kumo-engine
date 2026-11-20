# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import threading
import weakref
from typing import Callable

from google.cloud import storage

logger = logging.getLogger(__name__)


class _LiveClients:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[int, storage.Client] = {}
        self._keys = itertools.count()

    def add(self, client: storage.Client) -> int:
        key = next(self._keys)
        with self._lock:
            self._clients[key] = client
        return key

    def release(self, key: int) -> None:
        with self._lock:
            client = self._clients.pop(key, None)
        if client is not None:
            _close(client)

    def release_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            _close(client)

    def __len__(self) -> int:
        with self._lock:
            return len(self._clients)


class _Lease:
    __slots__ = ('client', '__weakref__')

    def __init__(self, client: storage.Client) -> None:
        self.client = client


def _close(client: storage.Client) -> None:
    try:
        client.close()
    except Exception as error:
        logger.debug('GCS client close failed (error_type=%s)',
                     type(error).__name__)


class ThreadClients:
    """One ``storage.Client`` per thread, closed when that thread exits.

    ``close()`` releases every client still held by a live thread.
    """
    def __init__(self, factory: Callable[[], storage.Client]) -> None:
        self._factory = factory
        self._live = _LiveClients()
        self._local = threading.local()

    def get(self) -> storage.Client:
        lease = getattr(self._local, 'lease', None)
        if lease is None:
            client = self._factory()
            lease = _Lease(client)
            weakref.finalize(lease, self._live.release, self._live.add(client))
            self._local.lease = lease
        return lease.client

    def close(self) -> None:
        self._local = threading.local()
        self._live.release_all()

    def live_count(self) -> int:
        return len(self._live)
