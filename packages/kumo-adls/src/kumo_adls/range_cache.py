# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import OrderedDict

MAX_BYTES = 64 << 20
MAX_ENTRIES = 1024
MAX_ENTRY_BYTES = 128 << 10


class RangeBytes:
    """Process-bounded payload LRU; keys never retain a filesystem or client.

    Budgets cover payload bytes and entry count. Python key/container overhead
    is additional; filesystem namespaces prevent sharing across principals.
    """
    def __init__(self, max_bytes=MAX_BYTES, max_entries=MAX_ENTRIES,
                 max_entry_bytes=MAX_ENTRY_BYTES):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.max_entry_bytes = max_entry_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key, value):
        # Admit immutable, complete successful ranges only. Empty/error/short
        # responses must never turn into a later successful cache hit.
        if (not isinstance(value, bytes) or not value
                or len(value) > self.max_entry_bytes
                or len(value) > self.max_bytes
                or len(value) != key[-1] - key[-2]):
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous)
            self._entries[key] = value
            self._bytes += len(value)
            while (self._bytes > self.max_bytes
                   or len(self._entries) > self.max_entries):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= len(removed)

    def clear_namespace(self, namespace):
        with self._lock:
            for key in list(self._entries):
                if key[0] is namespace:
                    self._bytes -= len(self._entries.pop(key))


# One global allowance, not a new 64 MiB allocation per connector/filesystem.
RANGE_BYTES = RangeBytes()
