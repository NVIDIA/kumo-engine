# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Cluster`: where feeders run. It launches or attaches them and reports a `Topology`.

A cluster never sees a contract, a task or a batch; `DistributedEngine` does that work
over the topology it gets back.
"""

from __future__ import annotations

import os
import subprocess
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from kumo.pytorch.engines.distributed.client import FeederClient
from kumo.pytorch.engines.distributed.server import Producer, launch_feeder, stop_processes

_START_TIMEOUT = 60.0


@dataclass(frozen=True)
class Cluster:
    """Where feeders run: launched here as subprocesses, or already running at endpoints."""

    kind: Literal["subprocess", "endpoints"]
    feeders: int = 1
    urls: tuple[str, ...] = ()

    @staticmethod
    def subprocess(feeders: int = 1) -> Cluster:
        """Launch ``feeders`` feeder processes on this machine, stopped by `close`."""
        return Cluster("subprocess", feeders=feeders)

    @staticmethod
    def endpoints(urls: Sequence[str]) -> Cluster:
        """Attach to feeders started separately (``python -m kumo.pytorch.feeder``)."""
        return Cluster("endpoints", feeders=len(urls), urls=tuple(urls))

    def __post_init__(self) -> None:
        if self.feeders < 1:
            raise ValueError("a cluster needs at least one feeder")

    def launcher(self) -> Launcher:
        """The backend that obtains this cluster's feeders."""
        if self.kind == "subprocess":
            return SubprocessLauncher(self.feeders)
        return EndpointsLauncher(self.urls)


@dataclass(frozen=True)
class FeederRef:
    """One feeder in a topology, and the producer processes it serves batches from."""

    id: str
    endpoint: str
    group: int
    partition: int
    owned: bool
    producers: tuple[str, ...]
    producer_pids: tuple[int, ...]


@dataclass(frozen=True)
class Topology:
    """The feeders a cluster provides: N replica groups of M partitions (M = 1 for now)."""

    feeders: tuple[FeederRef, ...]
    replicas: int
    partitions: int = 1

    def leader(self) -> FeederRef:
        """The feeder that fits; every other feeder installs its state."""
        return self.feeders[0]

    def group(self, group: int) -> tuple[FeederRef, ...]:
        return tuple(feeder for feeder in self.feeders if feeder.group == group)


class Launcher(Protocol):
    """Obtains feeder endpoints and stops the ones it started."""

    owned: bool

    def launch(self) -> Sequence[str]: ...

    def stop(self) -> None: ...

    @property
    def pids(self) -> tuple[int, ...]: ...


class SubprocessLauncher:
    """Feeders as child processes of this one, on loopback; they exit with it."""

    owned = True

    def __init__(self, count: int) -> None:
        self._count = count
        self._processes: list[subprocess.Popen[str]] = []
        self._finalizer = weakref.finalize(self, stop_processes, self._processes)

    def launch(self) -> list[str]:
        urls = []
        for _ in range(self._count):
            process, url = launch_feeder("127.0.0.1", os.getpid())
            self._processes.append(process)
            urls.append(url)
        return urls

    def stop(self) -> None:
        self._finalizer()

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(process.pid for process in self._processes)


class EndpointsLauncher:
    """Feeders someone else runs; attaching starts nothing and detaching stops nothing."""

    owned = False

    def __init__(self, urls: Sequence[str]) -> None:
        self._urls = tuple(urls)

    def launch(self) -> tuple[str, ...]:
        return self._urls

    def stop(self) -> None:
        pass

    @property
    def pids(self) -> tuple[int, ...]:
        return ()


class ClusterRuntime:
    """A started cluster: its launcher and the control-face connection to each feeder."""

    def __init__(self, cluster: Cluster) -> None:
        self._launcher = cluster.launcher()
        self._control: list[FeederClient] = []

    def start(self, producers: int) -> Topology:
        """Launch or attach, wait until each feeder answers, and size its producers."""

        feeders = []
        for group, url in enumerate(self._launcher.launch()):
            control = FeederClient(url, session="control")
            control.wait_ready(_START_TIMEOUT)
            self._control.append(control)
            started = control.call("hello", producers=producers)
            assert isinstance(started, tuple)
            host = url.rsplit(":", 1)[0]
            ports = [p.port for p in started if isinstance(p, Producer)]
            feeders.append(
                FeederRef(
                    id=f"g{group}.p0",
                    endpoint=url,
                    group=group,
                    partition=0,
                    owned=self._launcher.owned,
                    producers=(url, *(f"{host}:{port}" for port in ports[1:])),
                    producer_pids=tuple(p.pid for p in started if isinstance(p, Producer)),
                )
            )
        return Topology(tuple(feeders), replicas=len(feeders))

    @property
    def launched_pids(self) -> tuple[int, ...]:
        return self._launcher.pids

    def release(self) -> None:
        """Stop owned feeders (and their producers); detach from attached ones."""

        if self._launcher.owned:
            for control in self._control:
                try:
                    control.call("shutdown")
                except Exception:
                    pass
        for control in self._control:
            control.close()
        self._control.clear()
        self._launcher.stop()
