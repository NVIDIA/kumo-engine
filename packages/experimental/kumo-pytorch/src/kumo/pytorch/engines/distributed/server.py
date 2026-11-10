# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The feeder: runs the ordinary local prepare, materialize and batches, and streams them."""

from __future__ import annotations

import itertools
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.flight as flight

import kumo.pytorch.engines  # noqa: F401  (registers the built-in engines)
from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.artifact import FittedState
from kumo.pytorch.assembly import restore_fitted_state
from kumo.pytorch.dataset import Dataset, Parquet
from kumo.pytorch.engine import engine_from_name
from kumo.pytorch.engines.distributed import wire
from kumo.pytorch.preparation import Plan, prepare
from kumo.pytorch.spec import DataLoaderSpec
from kumo.pytorch.task import Task


@dataclass(frozen=True)
class ShippedState:
    """Fitted state as data: the written values, length summary and fit record."""

    values: Mapping[str, object]
    lengths: Mapping[str, object] | None
    fit: Mapping[str, object]
    has_targets: bool
    input_fingerprint: str
    contract_fingerprint: str


@dataclass(frozen=True)
class HostedTaskInfo:
    """What the driver needs to rebuild a hosted task: its state and identity."""

    task_id: str
    state: ShippedState
    fingerprints: Mapping[str, str]
    inference: bool
    splits: tuple[str, ...]


@dataclass
class _Session:
    plans: dict[str, Plan] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)


def ship_state(state: FittedState) -> ShippedState:
    return ShippedState(
        values=state.to_dict(),
        lengths=state._lengths,
        fit=dict(state._fit),
        has_targets=state.has_targets,
        input_fingerprint=state.input_fingerprint,
        contract_fingerprint=state.contract_fingerprint,
    )


def restore_state(contract: DataLoaderSpec, shipped: ShippedState) -> FittedState:
    input_state, target_encoders, _, _ = restore_fitted_state(
        contract, shipped.values, shipped.lengths
    )
    return FittedState(
        input_fingerprint=shipped.input_fingerprint,
        contract_fingerprint=shipped.contract_fingerprint,
        _input_state=input_state,
        _target_encoders=target_encoders if shipped.has_targets else None,
        _serialized=shipped.values,
        _lengths=shipped.lengths,
        _fit=shipped.fit,  # type: ignore[arg-type]
    )


def _dataset(tables: Mapping[str, Mapping[str, object]]) -> Dataset:
    return Dataset(
        {
            name: Parquet(path=str(source["path"]), columns=dict(source["columns"]))  # type: ignore[arg-type]
            for name, source in tables.items()
        }
    )


class FeederServer(flight.FlightServerBase):
    """Serves control actions and one batch stream per ``do_get``."""

    def __init__(self, location: str) -> None:
        super().__init__(location)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._actions: dict[str, Callable[[Mapping[str, object]], object]] = {
            "hello": self._hello,
            "open_plan": self._open_plan,
            "materialize": self._materialize,
            "count": self._count,
            "close_task": self._close_task,
            "close_plan": self._close_plan,
            "release": self._release,
            "shutdown": self._shutdown,
        }

    def do_action(
        self, context: flight.ServerCallContext, action: flight.Action
    ) -> Iterator[flight.Result]:
        try:
            body = wire.loads(action.body.to_pybytes())
            assert isinstance(body, Mapping)
            payload = wire.dumps(wire.Reply("ok", self._actions[action.type](body)))
        except BaseException as error:
            payload = wire.error_reply(error)
        yield flight.Result(pa.py_buffer(payload))

    def do_get(
        self, context: flight.ServerCallContext, ticket: flight.Ticket
    ) -> flight.GeneratorStream:
        request = wire.loads(ticket.ticket)
        assert isinstance(request, Mapping)
        return flight.GeneratorStream(wire.STREAM_SCHEMA, self._stream(request))

    def _stream(self, request: Mapping[str, object]) -> Iterator[pa.RecordBatch]:
        try:
            task = self._task(request)
            arguments = request["arguments"]
            assert isinstance(arguments, Mapping)
            batches = task.batches(**arguments)
            start = int(request["start"])  # type: ignore[arg-type]
            for batch in itertools.islice(batches, start, None):
                yield wire.record(wire.BATCH, wire.encode_batch(batch))
        except BaseException as error:
            yield wire.record(wire.ERROR, wire.error_reply(error))

    def _session(self, body: Mapping[str, object]) -> _Session:
        with self._lock:
            return self._sessions.setdefault(str(body["session"]), _Session())

    def _task(self, body: Mapping[str, object]) -> Task:
        task = self._session(body).tasks.get(str(body["task"]))
        if task is None:
            raise ValueError(f"the feeder holds no task {body['task']!r}; it was closed")
        return task

    def _hello(self, body: Mapping[str, object]) -> object:
        return {"pid": os.getpid()}

    def _open_plan(self, body: Mapping[str, object]) -> str:
        contract = body["contract"]
        assert isinstance(contract, DataLoaderSpec)
        spec = body["engine"]
        assert isinstance(spec, wire.EngineSpec)
        engine = engine_from_name(spec.name, **spec.config)
        plan = prepare(
            contract,
            _dataset(body["dataset"]),  # type: ignore[arg-type]
            engine,
            inference=bool(body["inference"]),
        )
        if plan.contract.fingerprint != contract.fingerprint:
            plan.close()
            raise ValueError("the feeder resolved the contract differently from the driver")
        plan_id = uuid.uuid4().hex
        self._session(body).plans[plan_id] = plan
        return plan_id

    def _materialize(self, body: Mapping[str, object]) -> HostedTaskInfo:
        session = self._session(body)
        plan = session.plans[str(body["plan"])]
        anchors = body["anchors"]
        assert isinstance(anchors, AnchorTable | tuple)
        shipped = body["fitted"]
        fitted = (
            None if shipped is None else restore_state(plan.contract, shipped)  # type: ignore[arg-type]
        )
        task = plan.materialize(
            anchors,  # type: ignore[arg-type]
            fit_split=body["fit_split"],  # type: ignore[arg-type]
            fitted=fitted,
            inference=bool(body["inference"]),
        )
        task_id = uuid.uuid4().hex
        session.tasks[task_id] = task
        return HostedTaskInfo(
            task_id=task_id,
            state=ship_state(task.fitted_state),
            fingerprints=dict(task.fingerprints),
            inference=task.inference,
            splits=tuple(task._anchors.splits()),
        )

    def _count(self, body: Mapping[str, object]) -> int:
        return self._task(body).count(body["split"])  # type: ignore[arg-type]

    def _close_task(self, body: Mapping[str, object]) -> None:
        task = self._session(body).tasks.pop(str(body["task"]), None)
        if task is not None:
            task.close()

    def _close_plan(self, body: Mapping[str, object]) -> None:
        plan = self._session(body).plans.pop(str(body["plan"]), None)
        if plan is not None:
            plan.close()

    def _release(self, body: Mapping[str, object]) -> None:
        with self._lock:
            session = self._sessions.pop(str(body["session"]), None)
        if session is None:
            return
        for task in session.tasks.values():
            task.close()
        for plan in session.plans.values():
            plan.close()

    def _shutdown(self, body: Mapping[str, object]) -> None:
        threading.Thread(target=self.shutdown, daemon=True).start()


def watch_parent(parent_pid: int, interval: float = 0.5) -> None:
    """Exit the process once ``parent_pid`` is gone, so a killed driver leaves no feeder."""

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)
            time.sleep(interval)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()
