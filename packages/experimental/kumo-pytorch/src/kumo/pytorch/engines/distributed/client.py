# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver side: a feeder connection, the hosted source, and a hosted task's batch stream."""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import pyarrow.flight as flight

from kumo.pytorch.anchors import AnchorTable, SplitSelection
from kumo.pytorch.artifact import FittedState
from kumo.pytorch.assembly import _fingerprint, fit_plan
from kumo.pytorch.contract.batch import Batch
from kumo.pytorch.contract.sample import BuiltSample, SampleRequest, SampleSpec
from kumo.pytorch.data.model import ModelSourceBatch
from kumo.pytorch.dataset import Dataset
from kumo.pytorch.engines.distributed import wire
from kumo.pytorch.engines.distributed.server import HostedTaskInfo, restore_state, ship_state
from kumo.pytorch.spec import DataLoaderSpec
from kumo.pytorch.task import Task

_NOT_HOSTED = "is not available on a hosted task: its data lives on the feeder"

Result = TypeVar("Result")


def _each(clients: Sequence[FeederClient], call: Callable[[FeederClient], Result]) -> list[Result]:
    """``call`` on every client at once, results in client order."""

    if len(clients) == 1:
        return [call(clients[0])]
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        return list(pool.map(call, clients))


class FeederClient:
    """One feeder connection: control actions, and batch streams with bounded prefetch."""

    def __init__(self, url: str, session: str) -> None:
        self.url = url
        self.session = session
        self._client = flight.FlightClient(url)

    def call(self, action: str, **body: object) -> object:
        results = self._client.do_action(
            flight.Action(action, wire.dumps({"session": self.session, **body}))
        )
        reply = wire.loads(next(iter(results)).body.to_pybytes())
        assert isinstance(reply, wire.Reply)
        if reply.status == "error":
            wire.raise_remote(reply)
        return reply.value

    def wait_ready(self, timeout: float) -> None:
        self._client.wait_for_available(timeout=timeout)

    def stream(self, request: Mapping[str, object], prefetch: int) -> Iterator[Batch]:
        """Start the stream and its prefetch now; batches come from the returned iterator."""

        reader = self._client.do_get(
            flight.Ticket(wire.dumps({"session": self.session, **request}))
        )
        pending: queue.Queue[Batch | BaseException | None] = queue.Queue(maxsize=prefetch)
        stop = threading.Event()

        def put(item: Batch | BaseException | None) -> bool:
            while not stop.is_set():
                try:
                    pending.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def pump() -> None:
            try:
                while True:
                    try:
                        chunk = reader.read_chunk().data
                    except StopIteration:
                        break
                    kind = chunk.column(0)[0].as_py()
                    payload = chunk.column(1)[0].as_buffer().to_pybytes()
                    if kind == wire.ERROR:
                        reply = wire.loads(payload)
                        assert isinstance(reply, wire.Reply)
                        try:
                            wire.raise_remote(reply)
                        except BaseException as error:
                            put(error)
                        return
                    if not put(wire.decode_batch(payload)):
                        return
                put(None)
            except BaseException as error:
                put(error)

        thread = threading.Thread(target=pump, name="feeder-prefetch", daemon=True)
        thread.start()

        def drain() -> Iterator[Batch]:
            try:
                while True:
                    item = pending.get()
                    if item is None:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                stop.set()
                reader.cancel()
                thread.join(timeout=5)

        return drain()

    def close(self) -> None:
        self._client.close()


class Router:
    """Every feeder's producers, and which one builds a stripe of a rank's pass.

    A rank's pass is split into ``feeders * stripes`` stripes by batch index; stripe
    ``k`` runs on feeder ``(rank + k) mod feeders``, producer ``k // feeders``.
    """

    def __init__(self, grid: Sequence[Sequence[FeederClient]]) -> None:
        if not grid or len({len(row) for row in grid}) != 1:
            raise ValueError("every feeder needs the same number of producers")
        self.grid = tuple(tuple(row) for row in grid)
        self.all = tuple(client for row in self.grid for client in row)
        self.primary = self.all[0]

    @property
    def stripes(self) -> int:
        return len(self.all)

    def route(self, rank: int, stripe: int) -> int:
        """Position in ``all`` of the producer that builds ``stripe`` of ``rank``'s pass."""
        feeders = len(self.grid)
        return ((rank + stripe) % feeders) * len(self.grid[0]) + stripe // feeders


@dataclass
class _HostedTask:
    """The private backend of a hosted `Task`: its id on each producer and stream settings."""

    producers: Router
    task_ids: tuple[str, ...]
    prefetch: int

    def host_batches(self, arguments: Mapping[str, object], start: int) -> Iterator[Batch]:
        rank = arguments["rank"]
        assert isinstance(rank, int)
        count = self.producers.stripes
        streams: list[Iterator[Batch]] = []
        for offset in range(count):
            first = start + offset
            where = self.producers.route(rank, first % count)
            streams.append(
                self.producers.all[where].stream(
                    {
                        "task": self.task_ids[where],
                        "arguments": dict(arguments),
                        "first": first,
                        "step": count,
                    },
                    self.prefetch,
                )
            )
        return _merge(streams)


def _merge(streams: list[Iterator[Batch]]) -> Iterator[Batch]:
    """Round-robin over stripes in batch order; the pass ends at the first exhausted stripe."""

    try:
        for stream in itertools.cycle(streams):
            batch = next(stream, None)
            if batch is None:
                return
            yield batch
    finally:
        for stream in streams:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


class _HostedAnchors:
    """The task's anchors as the feeder holds them; only counts are answered remotely."""

    def __init__(
        self, producers: Router, task_ids: tuple[str, ...], splits: tuple[str, ...]
    ) -> None:
        self._producers = producers
        self._task_ids = task_ids
        self._splits = splits
        self._counts: dict[SplitSelection, int] = {}
        self._closed = False

    def splits(self) -> tuple[str, ...]:
        return self._splits

    def count(self, selection: SplitSelection) -> int:
        if selection not in self._counts:
            value = self._producers.primary.call("count", task=self._task_ids[0], split=selection)
            assert isinstance(value, int)
            self._counts[selection] = value
        return self._counts[selection]

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
        raise ValueError(f"anchor order {_NOT_HOSTED}")

    def fetch(self, selection: SplitSelection, positions: Sequence[int]) -> Sequence[SampleRequest]:
        raise ValueError(f"anchor rows {_NOT_HOSTED}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def close(pair: tuple[FeederClient, str]) -> None:
            try:
                pair[0].call("close_task", task=pair[1])
            except flight.FlightError:
                pass

        with ThreadPoolExecutor(max_workers=len(self._task_ids)) as pool:
            list(pool.map(close, zip(self._producers.all, self._task_ids, strict=True)))


class _NoSource:
    """Stands in for a sample source on the driver, where the hosted task never reads data."""

    def build(self, spec: SampleSpec, requests: Sequence[SampleRequest]) -> list[BuiltSample]:
        raise ValueError(f"building samples {_NOT_HOSTED}")

    def build_model(self, spec: SampleSpec, requests: Sequence[SampleRequest]) -> ModelSourceBatch:
        raise ValueError(f"building samples {_NOT_HOSTED}")


class DistributedSource:
    """A dataset bound through feeders: the `SampleSource` `DistributedEngine.open` returns.

    It has the `TaskHost` capability, so `Plan.materialize` materializes on the
    feeder; the per-anchor methods are refused.
    """

    def __init__(
        self,
        grid: Sequence[Sequence[FeederClient]],
        engine: wire.EngineSpec,
        dataset: Dataset,
        prefetch: int,
    ) -> None:
        self._producers = Router(grid)
        self._engine = engine
        self._dataset = dataset
        self._prefetch = prefetch
        self._plans: dict[tuple[str, bool], tuple[str, ...]] = {}
        self._closed = False

    def build(self, spec: SampleSpec, requests: Sequence[SampleRequest]) -> list[BuiltSample]:
        raise ValueError(f"building samples {_NOT_HOSTED}")

    def fit_graph_fields(
        self, sample: SampleSpec, spec: object, roots: Sequence[object]
    ) -> Mapping[tuple[str, str], tuple[object, ...]]:
        raise ValueError(f"graph fitting {_NOT_HOSTED}")

    def sample_graph(
        self,
        sample: SampleSpec,
        spec: object,
        state: object,
        roots: Sequence[object],
        seed: int,
        epoch: int,
    ) -> object:
        raise ValueError(f"graph sampling {_NOT_HOSTED}")

    def host_materialize(
        self,
        contract: DataLoaderSpec,
        anchors: AnchorTable | Sequence[SampleRequest],
        *,
        fit_split: object,
        fitted: FittedState | None,
        inference: bool,
        plan_inference: bool,
    ) -> Task:
        if self._closed:
            raise ValueError("this plan is closed")
        plan_ids = self._plan(contract, plan_inference)
        shipped_anchors = anchors if isinstance(anchors, AnchorTable) else tuple(anchors)
        producers = self._producers

        def materialize(where: int, split: object, state: object) -> HostedTaskInfo:
            info = producers.all[where].call(
                "materialize" if where == 0 else "install_state",
                plan=plan_ids[where],
                anchors=shipped_anchors,
                fit_split=split,
                fitted=state,
                inference=inference,
            )
            assert isinstance(info, HostedTaskInfo)
            return info

        info = materialize(0, fit_split, None if fitted is None else ship_state(fitted))
        others: list[HostedTaskInfo] = []
        if producers.stripes > 1:
            with ThreadPoolExecutor(max_workers=producers.stripes - 1) as pool:
                others = list(
                    pool.map(
                        lambda where: materialize(where, None, info.state),
                        range(1, producers.stripes),
                    )
                )
        task_ids = (info.task_id, *(other.task_id for other in others))
        hosted_anchors = _HostedAnchors(producers, task_ids, info.splits)
        try:
            for other in others:
                if dict(other.fingerprints) != dict(info.fingerprints):
                    raise RuntimeError(
                        "a producer's task does not match the one fitted on the first: "
                        f"{dict(other.fingerprints)} vs {dict(info.fingerprints)}"
                    )
            task = _rebuild_task(
                contract,
                info,
                hosted_anchors,
                dataset_fingerprint=_fingerprint(self._dataset),
                anchors_fingerprint=_fingerprint(shipped_anchors),
                hosted=_HostedTask(producers, task_ids, self._prefetch),
            )
            if dict(task.fingerprints) != dict(info.fingerprints):
                raise RuntimeError(
                    "the feeder's task does not match the one rebuilt on the driver: "
                    f"feeder {dict(info.fingerprints)}, driver {dict(task.fingerprints)}"
                )
        except BaseException:
            hosted_anchors.close()
            raise
        return task

    def _plan(self, contract: DataLoaderSpec, inference: bool) -> tuple[str, ...]:
        key = (contract.fingerprint, inference)
        if key not in self._plans:

            def open_plan(client: FeederClient) -> str:
                plan_id = client.call(
                    "open_plan",
                    contract=contract,
                    dataset=self._dataset.to_dict(),
                    engine=self._engine,
                    inference=inference,
                )
                assert isinstance(plan_id, str)
                return plan_id

            self._plans[key] = tuple(_each(self._producers.all, open_plan))
        return self._plans[key]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for plan_ids in self._plans.values():
            for client, plan_id in zip(self._producers.all, plan_ids, strict=True):
                try:
                    client.call("close_plan", plan=plan_id)
                except flight.FlightError:
                    pass


def _rebuild_task(
    contract: DataLoaderSpec,
    info: HostedTaskInfo,
    anchors: _HostedAnchors,
    *,
    dataset_fingerprint: str,
    anchors_fingerprint: str,
    hosted: _HostedTask,
) -> Task:
    """The feeder's task on the driver: the same fitted state, with batches from the feeder."""

    state = restore_state(contract, info.state)
    assembler = fit_plan(
        contract.sample,
        contract.batch,
        (),
        _NoSource(),
        schema_fingerprint=_fingerprint(contract.schema),
        contract_fingerprint=contract.fingerprint,
        contract=contract,
        input_state=state._input_state,
        target_encoders=state._target_encoders if state.has_targets else {},
        length_summary=state._lengths,
        inference=info.inference,
    )
    return Task(
        contract=contract,
        assembler=assembler,
        anchors=anchors,
        dataset_fingerprint=dataset_fingerprint,
        anchors_fingerprint=anchors_fingerprint,
        fit=state._fit,
        has_targets=state.has_targets,
        hosted=hosted,
    )
