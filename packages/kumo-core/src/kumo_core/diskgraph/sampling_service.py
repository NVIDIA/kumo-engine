# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the native Flight gateway process; sampling and serving run in Rust."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import orjson
from pyarrow import flight


@dataclass(frozen=True)
class SamplingServiceConfig:
    """Configure a native gateway over a prepared graph replica.

    Args:
        index_path: Complete worker-local graph index.
        host: Bind address; use 0.0.0.0 for cross-node connections.
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
    """Launch the wheel's native gateway and retain its logs beside the index.

    Args:
        config: Graph location, bind address and native resource budgets.
    Raises:
        ValueError: If the graph or resource budgets are invalid.
        FileNotFoundError: If the bundled gateway or an explicit override is absent.
        RuntimeError: If the native process exits or fails readiness within 30 seconds.
        OSError: If process creation or log creation fails.
    """

    def __init__(self, config: SamplingServiceConfig) -> None:
        """Launch the gateway and wait for its bound-port readiness record."""
        import diskgraph

        self._config = config
        self._root = config.index_path.resolve(strict=True)
        if not (self._root / "index_metadata.json").is_file():
            raise ValueError("Sampling requires a completed graph index")
        if min(config.max_streams, config.max_threads) < 1:
            raise ValueError("Sampling stream and thread limits must be positive")
        binary = Path(
            os.environ.get("DISKGRAPH_TASK_GATEWAY")
            or (
                Path(diskgraph.__file__).parent
                / "bin"
                / ("diskgraph-gateway.exe" if os.name == "nt" else "diskgraph-gateway")
            )
        )
        if not binary.is_file():
            raise FileNotFoundError(f"Native sampling gateway not found: {binary}")
        identity = uuid.uuid4().hex
        self.log_path = self._root.parent / f"sampling-gateway-{identity}.log"
        ready = self._root.parent / f"sampling-gateway-{identity}.ready.json"
        self._closed = False
        self._last_health: dict[str, str | int | bool] = {}
        self._client: flight.FlightClient | None = None
        with self.log_path.open("ab") as log:
            self._process = subprocess.Popen(
                [
                    str(binary),
                    "--data",
                    str(self._root),
                    "--host",
                    config.host,
                    "--port",
                    str(config.port),
                    "--max-streams",
                    str(config.max_streams),
                    "--max-threads",
                    str(config.max_threads),
                    "--ready-file",
                    str(ready),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            deadline = time.monotonic() + 30
            while not ready.exists():
                if self._process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Native gateway failed readiness; inspect {self.log_path}"
                    )
                time.sleep(0.01)
            self.port = int(orjson.loads(ready.read_bytes())["port"])
            self._client = flight.FlightClient(self.endpoint())
            self.health()
        except BaseException:
            self.close()
            raise

    def endpoint(self, advertised_host: str | None = None) -> str:
        """Return the bound Flight URI, optionally advertising a worker address."""
        host = advertised_host or self._config.host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"grpc://{host}:{self.port}"

    def health(self) -> dict[str, str | int | bool]:
        """Return native process counters, or the last counters after shutdown.

        Raises:
            flight.FlightError: If a running gateway cannot answer its health RPC.
        """
        if self._closed:
            return {**self._last_health, "ready": False}
        assert self._client is not None
        results = list(self._client.do_action(flight.Action("health", b"")))
        if len(results) != 1:
            raise RuntimeError("Native gateway returned invalid health metadata")
        self._last_health = orjson.loads(results[0].body.to_pybytes())
        return self._last_health

    def close(self) -> None:
        """Stop the owned process and close its client; safe to repeat.

        A gateway that cannot shut down within ten seconds is killed and reaped.
        Logs and readiness records are retained for diagnosis.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            if self._client is not None:
                self._client.close()
