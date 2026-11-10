# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`DistributedEngine`: run an engine, and the tasks built on it, on feeder processes."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.dataset import Dataset, InMemoryRows, TableSource
from kumo.pytorch.engine import AnchorTask, Engine, engine_from_name, register_engine
from kumo.pytorch.engines.distributed import wire
from kumo.pytorch.engines.distributed.client import DistributedSource, FeederClient
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue, _known_types

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
        if self.feeders != 1:
            raise ValueError("this prototype runs exactly one feeder")


@dataclass
class _Feeder:
    client: FeederClient
    process: subprocess.Popen[str] | None


def _stop(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class DistributedEngine:
    """Run ``inner`` on feeders; the same script runs with ``inner`` alone.

    `prepare` runs the data-free checks here, then binds the dataset on the
    feeder; `Plan.materialize` fits there and returns an ordinary `Task` whose
    batches stream from it. Datasets and anchors must be paths the feeder can
    read, and the contract must be data: in-memory rows and encoders that are
    not built in are refused before anything starts.

    :param inner: the engine each feeder runs, configured as on the feeder.
    :param cluster: `Cluster.subprocess` or `Cluster.endpoints`.
    :param prefetch: batches buffered per stream on this side.
    """

    name = "distributed"

    def __init__(
        self,
        inner: Engine | Mapping[str, object],
        *,
        cluster: Cluster | Mapping[str, object],
        prefetch: int = 4,
    ) -> None:
        if isinstance(inner, Mapping):
            config = dict(inner)
            inner = engine_from_name(str(config.pop("engine")), **config)
        if isinstance(cluster, Mapping):
            config = dict(cluster)
            kind = config.pop("kind")
            cluster = (
                Cluster.endpoints(config["urls"])  # type: ignore[arg-type]
                if kind == "endpoints"
                else Cluster.subprocess(**config)  # type: ignore[arg-type]
            )
        if prefetch <= 0:
            raise ValueError("prefetch must be positive")
        self.inner = inner
        self.cluster = cluster
        self.prefetch = prefetch
        self._spec = _engine_spec(inner)
        self._session = uuid.uuid4().hex
        self._feeders: list[_Feeder] = []
        self._launched: list[subprocess.Popen[str]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, _stop, self._launched)

    def start(self) -> None:
        """Launch or attach the feeders now rather than at the first `prepare`."""

        with self._lock:
            if self._closed:
                raise ValueError("this engine is closed")
            if self._feeders:
                return
            if self.cluster.kind == "subprocess":
                urls = [self._launch() for _ in range(self.cluster.feeders)]
            else:
                urls = list(self.cluster.urls)
            for url in urls:
                client = FeederClient(url, self._session)
                client.wait_ready(_START_TIMEOUT)
                client.call("hello")
                process = self._launched[len(self._feeders)] if self._launched else None
                self._feeders.append(_Feeder(client, process))

    def _launch(self) -> str:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kumo.pytorch.feeder",
                "--listen",
                "127.0.0.1:0",
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self._launched.append(process)
        assert process.stdout is not None
        line = process.stdout.readline().strip()
        prefix = wire.LISTENING + " "
        if not line.startswith(prefix):
            raise RuntimeError(f"the feeder did not start (exit code {process.poll()}): {line!r}")
        return line[len(prefix) :]

    @property
    def feeder_pids(self) -> tuple[int, ...]:
        """Process IDs of the feeders this engine launched."""
        return tuple(process.pid for process in self._launched)

    def open(self, dataset: Dataset, schema: DataSchema) -> DistributedSource:
        self.start()
        return DistributedSource(self._feeders[0].client, self._spec, dataset, self.prefetch)

    def materialize_anchors(self, table: AnchorTable) -> AnchorTask:
        raise ValueError("a distributed engine materializes anchors on its feeders")

    def check_contract(self, contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
        issues = _foreign_values(contract)
        inner = getattr(self.inner, "check_contract", None)
        if inner is not None:
            issues.extend(inner(contract))
        return tuple(issues)

    def check_sources(
        self, sources: Mapping[str, TableSource], path: str
    ) -> tuple[ValidationIssue, ...]:
        issues = [
            ValidationIssue(
                f"{path}.{name}" if name else path,
                "in-memory rows cannot be shipped to a feeder: they are the data, not a reference",
                "write the table as Parquet at a path the feeder can read",
                code="DISTRIBUTED-IN-MEMORY",
            )
            for name, source in sources.items()
            if isinstance(source, InMemoryRows)
        ]
        inner = getattr(self.inner, "check_sources", None)
        if inner is not None:
            issues.extend(inner(sources, path))
        return tuple(issues)

    def check_anchors(self, table: AnchorTable) -> tuple[ValidationIssue, ...]:
        if isinstance(table.source, InMemoryRows):
            return tuple(self.check_sources({"": table.source}, "anchors.source"))
        inner = getattr(self.inner, "check_anchors", None)
        return () if inner is None else tuple(inner(table))

    def close(self) -> None:
        """Stop the feeders this engine launched; detach from those it attached to."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            feeders, self._feeders = self._feeders, []
        for feeder in feeders:
            try:
                if feeder.process is not None:
                    feeder.client.call("shutdown")
                else:
                    feeder.client.call("release")
            except Exception:
                pass
            feeder.client.close()
        self._finalizer()

    def __enter__(self) -> DistributedEngine:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def _engine_spec(engine: Engine) -> wire.EngineSpec:
    """The engine as its registry name and constructor keywords, read back from the instance."""

    config: dict[str, wire.JSONValue] = {}
    for name, parameter in inspect.signature(type(engine).__init__).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if not hasattr(engine, name):
            raise TypeError(
                f"engine {engine.name!r} does not expose {name!r}, so it cannot be "
                "configured on a feeder"
            )
        value = getattr(engine, name)
        if isinstance(value, Path):
            value = str(value)
        if not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError(f"engine option {name!r} is not plain data: {value!r}")
        config[name] = value
    return wire.EngineSpec(engine.name, config)


def _foreign_values(contract: DataLoaderSpec) -> list[ValidationIssue]:
    """Contract values that are code, not built-in declarations; a feeder cannot receive them."""

    known = set(_known_types().values())
    found: list[ValidationIssue] = []

    def visit(item: object, where: str) -> None:
        if isinstance(item, str | int | float | bool | type(None)):
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{where}.{key}")
        elif isinstance(item, tuple | list | frozenset | set):
            for index, child in enumerate(item):
                visit(child, f"{where}[{index}]")
        elif is_dataclass(item) and type(item) in known:
            for item_field in fields(item):
                visit(getattr(item, item_field.name), f"{where}.{item_field.name}")
        elif is_dataclass(item) or callable(item):
            found.append(
                ValidationIssue(
                    where.lstrip("."),
                    f"{type(item).__module__}.{type(item).__qualname__} is not a built-in "
                    "declaration, so it cannot be shipped to a feeder as data",
                    "use a built-in encoding, or run the inner engine locally",
                    code="DISTRIBUTED-CUSTOM-CODE",
                )
            )

    visit(contract, "")
    return found


register_engine(DistributedEngine.name, DistributedEngine)
