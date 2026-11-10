# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver side: a feeder connection, the hosted source, and a hosted task's batch stream."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

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

    def close(self) -> None:
        self._client.close()


@dataclass
class _HostedTask:
    """The private backend of a hosted `Task`: its id on the feeder and the stream settings."""

    client: FeederClient
    task_id: str
    prefetch: int

    def host_batches(self, arguments: Mapping[str, object], start: int) -> Iterator[Batch]:
        return self.client.stream(
            {"task": self.task_id, "arguments": dict(arguments), "start": start}, self.prefetch
        )


class _HostedAnchors:
    """The task's anchors as the feeder holds them; only counts are answered remotely."""

    def __init__(self, client: FeederClient, task_id: str, splits: tuple[str, ...]) -> None:
        self._client = client
        self._task_id = task_id
        self._splits = splits
        self._counts: dict[SplitSelection, int] = {}
        self._closed = False

    def splits(self) -> tuple[str, ...]:
        return self._splits

    def count(self, selection: SplitSelection) -> int:
        if selection not in self._counts:
            value = self._client.call("count", task=self._task_id, split=selection)
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
        try:
            self._client.call("close_task", task=self._task_id)
        except flight.FlightError:
            pass


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
        client: FeederClient,
        engine: wire.EngineSpec,
        dataset: Dataset,
        prefetch: int,
    ) -> None:
        self._client = client
        self._engine = engine
        self._dataset = dataset
        self._prefetch = prefetch
        self._plans: dict[tuple[str, bool], str] = {}
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
        plan_id = self._plan(contract, plan_inference)
        shipped_anchors = anchors if isinstance(anchors, AnchorTable) else tuple(anchors)
        info = self._client.call(
            "materialize",
            plan=plan_id,
            anchors=shipped_anchors,
            fit_split=fit_split,
            fitted=None if fitted is None else ship_state(fitted),
            inference=inference,
        )
        assert isinstance(info, HostedTaskInfo)
        hosted_anchors = _HostedAnchors(self._client, info.task_id, info.splits)
        try:
            task = _rebuild_task(
                contract,
                info,
                hosted_anchors,
                dataset_fingerprint=_fingerprint(self._dataset),
                anchors_fingerprint=_fingerprint(shipped_anchors),
                hosted=_HostedTask(self._client, info.task_id, self._prefetch),
            )
        except BaseException:
            hosted_anchors.close()
            raise
        if dict(task.fingerprints) != dict(info.fingerprints):
            hosted_anchors.close()
            raise RuntimeError(
                "the feeder's task does not match the one rebuilt on the driver: "
                f"feeder {dict(info.fingerprints)}, driver {dict(task.fingerprints)}"
            )
        return task

    def _plan(self, contract: DataLoaderSpec, inference: bool) -> str:
        key = (contract.fingerprint, inference)
        if key not in self._plans:
            plan_id = self._client.call(
                "open_plan",
                contract=contract,
                dataset=self._dataset.to_dict(),
                engine=self._engine,
                inference=inference,
            )
            assert isinstance(plan_id, str)
            self._plans[key] = plan_id
        return self._plans[key]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for plan_id in self._plans.values():
            try:
                self._client.call("close_plan", plan=plan_id)
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
