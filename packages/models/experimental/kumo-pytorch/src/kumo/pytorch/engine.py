# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine interface: what an engine must provide, and how a caller picks one.

A :class:`~kumo.pytorch.dataset.Dataset` says where the data is;
an :class:`Engine` says what reads it. Both are supplied by the caller, and
neither names the other.

An engine is a factory, not a holder of data: :meth:`Engine.open` binds a
dataset and returns a :class:`SampleSource`. Construction-time configuration
(artifact directories, index paths) belongs to the engine; data binding belongs
to the dataset.

Distributed execution is expressed by ``rank``/``world_size`` on
``Task.batches``.

The inspectable source boundary is
:class:`~kumo.pytorch.contract.sample.BuiltSample` — raw rows for one anchor,
before encoding. An optimized source may additionally implement
``ModelSampleSource`` and keep the same values in typed batched buffers. kumo-pytorch
still owns fitting, encoding and collation in both cases; only the physical
handoff changes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kumo.pytorch.anchors import AnchorTable, SplitSelection
from kumo.pytorch.contract.sample import BuiltSample, SampleRequest, SampleSpec
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.data.model import ModelSourceBatch
from kumo.pytorch.dataset import Dataset

__all__ = [
    "Engine",
    "GraphRoot",
    "AnchorTask",
    "ModelSampleSource",
    "SampleSource",
    "available_engines",
    "engine_from_name",
    "register_engine",
]


@dataclass(frozen=True)
class GraphRoot:
    """One anchor a graph sample is grown from.

    Lives with the seam rather than with any sampler: every engine's graph
    sampling starts from the same three facts, and the batch assembler hands
    them over without knowing which engine will read them.
    """

    sample_id: str | int
    entity_id: int
    cutoff: int


@runtime_checkable
class AnchorTask(Protocol):
    """An anchor table materialized once, read by split and position.

    Returned by `Engine.materialize_anchors` when a plan is materialized, and
    held for the life of the task. The engine fixes the rows' canonical order
    once, so a position means the same anchor for every rank and every epoch.

    ``order`` may be any sequence an engine can serve efficiently, provided every epoch
    visits each anchor of the split exactly once, ranks are disjoint and
    together cover the split, and the sequence is a pure function of its
    arguments. `kumo.pytorch.anchors.shard_positions` satisfies all three and is
    what an engine should use unless it has a reason not to. Equal per-rank
    counts are kumo-pytorch's job: it pads, with a marker, after ``order``.
    """

    def splits(self) -> tuple[str, ...]:
        """The split names present, sorted; empty when anchors declare none."""
        ...

    def count(self, selection: SplitSelection) -> int:
        """Anchors in the selection: `ALL_ANCHORS`, or a set of split names
        where ``None`` selects anchors that carry no split."""
        ...

    def order(
        self,
        selection: SplitSelection,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
        rank: int,
        world_size: int,
    ) -> Iterator[int]:
        """Positions within the selection that one rank consumes this epoch."""
        ...

    def fetch(self, selection: SplitSelection, positions: Sequence[int]) -> Sequence[SampleRequest]:
        """The anchors at ``positions`` within the selection, in that order."""
        ...

    def close(self) -> None:
        """Release whatever the task holds. Must tolerate repeat calls."""
        ...


# --------------------------------------------------------------------------
# The source boundary
# --------------------------------------------------------------------------


@runtime_checkable
class SampleSource(Protocol):
    """A dataset bound to an engine, able to produce raw per-anchor rows.

    Implementations own storage, joins, time windows and graph sampling. They
    do not own encoding: returning ``BuiltSample`` is where the required engine
    contract ends. ``ModelSampleSource`` is an optional batched extension.
    """

    def build(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> list[BuiltSample]:
        """Build the requested samples, in request order."""
        ...

    def fit_graph_fields(
        self,
        sample: SampleSpec,
        spec: Any,
        roots: Sequence[Any],
    ) -> Mapping[tuple[str, str], tuple[Any, ...]]:
        """Collect node-field values for deterministic encoder fitting.

        ``sample`` carries the entity binding and the named neighborhoods, so an
        implementation needs nothing else to resolve the traversal.
        """
        ...

    def sample_graph(
        self,
        sample: SampleSpec,
        spec: Any,
        state: Any,
        roots: Sequence[Any],
        seed: int,
        epoch: int,
    ) -> Any:
        """Sample one batch of neighborhoods."""
        ...

    def close(self) -> None:
        """Release whatever the binding holds. Must tolerate repeat calls."""
        ...


@runtime_checkable
class ModelSampleSource(Protocol):
    """Optional hot-path extension that keeps source values in typed buffers."""

    def build_model(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> ModelSourceBatch:
        """Build one model-facing source batch without per-value Python records."""
        ...


@runtime_checkable
class Engine(Protocol):
    """A factory that binds a dataset and runs a contract over it.

    Before it reads any data, `check` and `prepare` let the engine refuse
    anything it can't run, through optional methods that return `ValidationIssue`
    objects: ``check_sources(sources, path)`` for the sources bound,
    ``check_contract(contract)`` for the contract, and ``check_anchors(table)``
    for an anchor table.
    The package ships two engines: `DiskGraphEngine`, for real data, and
    `InMemoryEngine`, for trying things out on tiny data. The DiskGraph engine
    lists what it doesn't support yet in `DiskGraphEngine.unsupported`.

    :param name: registry key, the string `engine_from_name` resolves.
    """

    name: str

    def open(self, dataset: Dataset, schema: DataSchema) -> SampleSource:
        """Bind a dataset for reading. Called after validation has passed."""
        ...

    def materialize_anchors(self, table: AnchorTable) -> AnchorTask:
        """Materialize an anchor table once, before any batch is read.

        Called by `Plan.materialize` after the binding has been validated. This
        is where an engine does anchor-dependent batch work — forward labels,
        anchor-to-graph edges — and fixes the canonical row order.
        """
        ...


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register_engine(name: str, factory: type) -> None:
    """Register an engine class under a short name.

    Explicit construction is the contract; the registry exists so a configured
    name resolves to the same class, and so an out-of-tree package can add an
    engine without editing this module.
    """

    if not name:
        raise ValueError("an engine name must be non-empty")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"engine {name!r} is already registered to {existing!r}")
    _REGISTRY[name] = factory


def available_engines() -> tuple[str, ...]:
    """Registered engine names, sorted."""
    return tuple(sorted(_REGISTRY))


def engine_from_name(name: str, **config: Any) -> Engine:
    """Resolve a registered name to a class and construct it.

    ``engine_from_name("diskgraph", index=...)`` is exactly
    ``DiskGraphEngine(index=...)``. Nothing is opened, connected or
    initialized — construction is inert, and resources are acquired by
    :meth:`Engine.open`. The two are named differently on purpose.

    This exists so a name from a configuration file or environment variable
    resolves to the same class explicit construction would give.
    """

    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"unknown engine {name!r}; available engines are {list(available_engines())}"
        )
    return factory(**config)
