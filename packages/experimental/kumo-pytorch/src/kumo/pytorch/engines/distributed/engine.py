# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`DistributedEngine`: run an engine, and the tasks built on it, on feeder processes."""

from __future__ import annotations

import inspect
import threading
import uuid
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.dataset import Dataset, InMemoryRows, TableSource
from kumo.pytorch.engine import AnchorTask, Engine, engine_from_name, register_engine
from kumo.pytorch.engines.distributed import wire
from kumo.pytorch.engines.distributed.client import DistributedSource, FeederClient
from kumo.pytorch.engines.distributed.cluster import Cluster, ClusterRuntime, Topology
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue, _known_types

__all__ = ["Cluster", "DistributedEngine"]


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
    :param stripes: producer processes per feeder. A rank's pass is split into
        ``feeders * stripes`` stripes by batch index and merged back in order here.
    """

    name = "distributed"

    def __init__(
        self,
        inner: Engine | Mapping[str, object],
        *,
        cluster: Cluster | Mapping[str, object],
        prefetch: int = 4,
        stripes: int = 1,
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
        if stripes <= 0:
            raise ValueError("stripes must be positive")
        self.inner = inner
        self.cluster = cluster
        self.prefetch = prefetch
        self.stripes = stripes
        self._spec = _engine_spec(inner)
        self._session = uuid.uuid4().hex
        self._runtime = ClusterRuntime(cluster)
        self._topology: Topology | None = None
        self._grid: tuple[tuple[FeederClient, ...], ...] = ()
        self._lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        """Launch or attach the feeders now rather than at the first `prepare`."""

        with self._lock:
            if self._closed:
                raise ValueError("this engine is closed")
            if self._topology is not None:
                return
            self._topology = self._runtime.start(self.stripes)
            self._grid = tuple(
                tuple(FeederClient(url, self._session) for url in feeder.producers)
                for feeder in self._topology.feeders
            )

    @property
    def topology(self) -> Topology | None:
        """The feeders this engine runs on, once started."""
        return self._topology

    @property
    def feeder_pids(self) -> tuple[int, ...]:
        """Process IDs of the feeders this engine launched."""
        return self._runtime.launched_pids

    @property
    def producer_pids(self) -> tuple[tuple[int, ...], ...]:
        """Process IDs of every feeder's producers, the feeder itself first."""
        feeders = () if self._topology is None else self._topology.feeders
        return tuple(feeder.producer_pids for feeder in feeders)

    def open(self, dataset: Dataset, schema: DataSchema) -> DistributedSource:
        self.start()
        return DistributedSource(
            self._grid,
            self._spec,
            dataset,
            self.prefetch,
        )

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
            grid, self._grid = self._grid, ()
        for producer in (client for row in grid for client in row):
            try:
                producer.call("close_session")
            except Exception:
                pass
            producer.close()
        self._runtime.release()

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
