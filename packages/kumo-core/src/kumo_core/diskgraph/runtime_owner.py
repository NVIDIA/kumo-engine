# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local ownership shared by local and remote runtime consumers."""

from __future__ import annotations

import os

from diskgraph import RuntimeIndex


class RuntimeOwner:
    """Lazily own one runtime index per process and share it across datasets.

    Args:
        manifest_path: Portable runtime manifest path.
        rnd_seed: Engine and default shuffle seed.
        num_threads: Optional native sampling thread count.
        sampling_endpoints: Remote replica URI(s), or None for local sampling.

    Native handles are omitted when pickled and replaced after a PID change.
    Closing releases the current process's handle; a later get opens a new one.
    """

    def __init__(
        self,
        manifest_path: str,
        rnd_seed: int,
        *,
        num_threads: int | None = None,
        sampling_endpoints: str | list[str] | None = None,
    ) -> None:
        self._manifest_path = manifest_path
        self._rnd_seed = rnd_seed
        self._num_threads = num_threads
        self._sampling_endpoints = (
            list(sampling_endpoints)
            if isinstance(sampling_endpoints, list)
            else sampling_endpoints
        )
        self._index: RuntimeIndex | None = None
        self._pid: int | None = None

    def get(self) -> RuntimeIndex:
        """Return this process's index, constructing it lazily if needed.

        Returns:
            Shared local or remote runtime index.

        Raises:
            ValueError: If replica identities are invalid.
        """
        pid = os.getpid()
        if self._pid != pid:
            self._index = None
            self._pid = pid
        index = self._index
        if index is None:
            if self._sampling_endpoints is None:
                index = RuntimeIndex(
                    self._manifest_path,
                    rnd_seed=self._rnd_seed,
                    num_threads=self._num_threads,
                )
            else:
                index = RuntimeIndex(
                    self._manifest_path,
                    rnd_seed=self._rnd_seed,
                    num_threads=self._num_threads,
                    sampling_endpoints=self._sampling_endpoints,
                )
            self._index = index
        return index

    def close(self) -> None:
        """Release this process's index; safe before opening or when repeated.

        Raises:
            ValueError: If a native reader failed during shutdown.
        """
        index, pid = self._index, self._pid
        self._index = None
        self._pid = None
        if index is not None and pid == os.getpid():
            index.close()

    def __getstate__(self) -> dict[str, object]:
        """Return picklable configuration without a native handle or owning PID."""
        state = vars(self).copy()
        state["_index"] = None
        state["_pid"] = None
        return state
