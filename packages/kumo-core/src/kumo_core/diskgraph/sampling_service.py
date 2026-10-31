# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own an embedded Rust Flight listener through DiskGraph's PyO3 handle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplingServiceConfig:
    """Configure native serving over a prepared graph replica.

    Args:
        index_path: Complete worker-local graph index.
        host: Bind IP address; use 0.0.0.0 for cross-node connections.
        port: TCP port; zero binds an available port atomically.
        max_streams: Positive maximum number of simultaneous native producers.
        max_threads: Positive native thread budget for each opened index.
    """

    index_path: Path
    host: str = "127.0.0.1"
    port: int = 0
    max_streams: int = 1
    max_threads: int = 2


class SamplingService:
    """Start the native Flight server inside the caller's process.

    Args:
        config: Graph location, bind IP address and native resource budgets.

    Raises:
        ValueError: If graph metadata or resource budgets are invalid.
        OSError: If the graph directory cannot be resolved.
        diskgraph.DiskgraphSystemError: If native serving cannot start.
    """

    def __init__(self, config: SamplingServiceConfig) -> None:
        """Bind and start serving on Rust-owned background threads."""
        import diskgraph

        root = config.index_path.resolve(strict=True)
        if not (root / "index_metadata.json").is_file():
            raise ValueError("Sampling requires a completed graph index")
        if min(config.max_streams, config.max_threads) < 1:
            raise ValueError("Sampling stream and thread limits must be positive")
        self._server = diskgraph.SamplingServer(
            root,
            host=config.host,
            port=config.port,
            max_streams=config.max_streams,
            max_threads=config.max_threads,
        )
        self.port: int = self._server.port

    def endpoint(self, advertised_host: str | None = None) -> str:
        """Return the bound Flight URI.

        Args:
            advertised_host: Optional routable worker IP for remote clients.

        Returns:
            The Flight URI using the listener's actual bound port.

        Raises:
            ValueError: If the advertised host is not an IP address.
        """
        return self._server.endpoint(advertised_host)

    def health(self) -> dict[str, str | int | bool]:
        """Return native readiness and counters, including after shutdown.

        Returns:
            Readiness, process ID, port, active streams and batch/request counts.
        """
        return self._server.health()

    def close(self) -> None:
        """Cancel streams and join the native server; safe to repeat.

        The binding releases the GIL while joining native threads.

        Raises:
            diskgraph.DiskgraphSystemError: If the native server failed.
        """
        self._server.close()
