# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion-order, credit ownership, and cancellation at the Ray boundary."""

from collections.abc import Generator
from dataclasses import dataclass
from threading import Condition, Event, Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from diskgraph.sampling_options import SamplingOptions
from diskgraph.sampling_plan import SamplingPlan
from kumo_core.diskgraph import ray_sampling, remote_loader
from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchRequest,
    HeteroBatchPayload,
)
from kumo_core.diskgraph.hetero_wire import pack_python_hetero
from kumo_core.diskgraph.lightning_data import (
    DiskgraphTaskBatchIterable,
    RemoteDataLoader,
    _ShuffleStreamState,
)
from kumo_core.diskgraph.ray_sampling import RayBackendConfig, SamplerActor
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData

ray = pytest.importorskip("ray")


@dataclass(eq=False)
class _Ref:
    value: HeteroBatchPayload | None
    ordinal: int = -1


class _Source(DiskgraphTaskBatchIterable):
    def __init__(self, config: RayBackendConfig, count: int = 8) -> None:
        super().__init__(
            manifest_path="unused",
            stage="train",
            sampling_plan=SamplingPlan(),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            base_seed=42,
            shuffle_state=_ShuffleStreamState(),
            num_seeds=count,
            sampling_backend=config,
        )
        self.count = count
        self.finalized = Event()

    def __len__(self) -> int:
        return self.count

    def iter_requests(self) -> Generator[BatchRequest, None, None]:
        try:
            for ordinal in range(self.count):
                yield BatchRequest(
                    0,
                    0,
                    1,
                    ordinal,
                    (ordinal,),
                    SamplingOptions(stage="train", batch_size=1),
                    42,
                    1,
                )
        finally:
            self.finalized.set()


class _Transport:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.changed = Condition()
        self.submitted: list[BatchRequest] = []
        self.received = 0
        self.pause = False
        self.failure: Exception | None = None
        self.cancelled: list[_Ref] = []
        self.drain = Mock(side_effect=lambda: _Ref(None))
        self.actors = tuple(
            SimpleNamespace(
                sample=SimpleNamespace(remote=self.sample),
                drain=SimpleNamespace(remote=self.drain),
            )
            for _ in range(3)
        )
        transport = SimpleNamespace(
            is_initialized=lambda: True,
            get=self.get,
            wait=self.wait,
            cancel=lambda ref, **_options: self.cancelled.append(ref),
            exceptions=ray.exceptions,
        )
        monkeypatch.setattr(ray_sampling, "_ray", lambda: transport)
        self.config = RayBackendConfig(
            cast(tuple[SamplerActor, ...], self.actors),
            ArtifactIdentity("snapshot", "task", "mapping"),
            42,
            1,
            request_timeout=1,
        )

    def sample(self, request: BatchRequest) -> _Ref:
        graph = HeteroData()
        graph["nodes"].n_id = torch.tensor([request.batch_ordinal])
        graph["nodes"].batch_size = 1
        graph["nodes"].num_sampled_nodes = [1]
        graph["nodes"].tf = TensorFrame({}, {}, num_rows=1)
        with self.changed:
            self.submitted.append(request)
            self.changed.notify_all()
        return _Ref(pack_python_hetero(graph), request.batch_ordinal)

    def get(self, ref: _Ref, **_options: object) -> HeteroBatchPayload | None:
        if ref.value is not None:
            if self.failure is not None:
                raise self.failure
            with self.changed:
                self.received += 1
                self.changed.notify_all()
        return ref.value

    def wait(
        self, refs: list[_Ref], **_options: object
    ) -> tuple[list[_Ref], list[_Ref]]:
        if self.pause:
            Event().wait(0.005)
            return [], refs
        ready = max(refs, key=lambda ref: ref.ordinal)
        return [ready], [ref for ref in refs if ref is not ready]

    def await_submitted(self, count: int) -> None:
        with self.changed:
            assert self.changed.wait_for(
                lambda: len(self.submitted) >= count, timeout=3
            )


def test_completion_order_and_refill_without_another_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound all stages by actor count and overlap sampling with model work."""
    transport = _Transport(monkeypatch)
    source = _Source(transport.config)
    encoder = Mock(side_effect=AssertionError("trainer must not encode"))
    monkeypatch.setattr(
        "kumo_core.diskgraph.runtime_adapter.runtime_batch_to_hetero", encoder
    )
    with RemoteDataLoader(source) as loader:
        iterator = iter(loader)
        with transport.changed:
            assert transport.changed.wait_for(
                lambda: transport.received == 3, timeout=3
            )
        assert len(transport.submitted) == 3
        assert loader.num_workers == 0
        assert not loader.in_order
        first = next(iterator)
        assert first["nodes"].n_id.item() == 2
        # The producer refills while the trainer is between calls to next().
        transport.await_submitted(4)
        assert len(transport.submitted) == 4
        actual = [first, *iterator]
        assert sorted(int(batch["nodes"].n_id.item()) for batch in actual) == list(
            range(8)
        )
        assert len(loader) == len(source)
    encoder.assert_not_called()
    assert source.finalized.is_set()


def test_reconstruction_retains_credit_until_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not overrun the memory budget while the trainer reconstructs a graph."""
    transport = _Transport(monkeypatch)
    entered, release = Event(), Event()
    unpack = remote_loader.unpack_python_hetero

    def blocked_unpack(*payload: object) -> HeteroData:
        entered.set()
        assert release.wait(3)
        return unpack(*cast(HeteroBatchPayload, payload))

    monkeypatch.setattr(remote_loader, "unpack_python_hetero", blocked_unpack)
    with RemoteDataLoader(_Source(transport.config)) as loader:
        iterator = iter(loader)
        results: list[HeteroData] = []
        consumer = Thread(target=lambda: results.append(next(iterator)))
        consumer.start()
        try:
            assert entered.wait(3)
            assert len(transport.submitted) == 3
        finally:
            release.set()
            consumer.join(3)
        assert not consumer.is_alive()
        assert len(results) == 1
        transport.await_submitted(4)


def test_close_wakes_consumer_and_drains_before_epoch_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain outstanding serial calls before a new epoch can reuse its actors."""
    transport = _Transport(monkeypatch)
    transport.pause = True
    source = _Source(transport.config)
    with RemoteDataLoader(source) as loader:
        iterator = iter(loader)
        transport.await_submitted(3)
        stopped = Event()

        def consume() -> None:
            assert next(iterator, None) is None
            stopped.set()

        consumer = Thread(target=consume)
        consumer.start()
        loader.close()
        consumer.join(3)
        assert stopped.is_set()
        assert len(transport.cancelled) >= 3
        assert transport.drain.call_count == 3
        assert source.finalized.is_set()
        active = loader._batch_dataset._remote_iterator
        assert active is not None
        assert not active._dispatcher.is_alive()
        # Starting another iterator must also close an abandoned previous epoch.
        transport.pause = False
        assert len(list(loader)) == len(source)


@pytest.mark.parametrize("where", ["receive", "decode", "schedule"])
def test_failures_propagate_and_stop_receiver(
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    """Forward errors to the trainer and release all owned actor requests."""
    transport = _Transport(monkeypatch)
    source = _Source(transport.config)
    error = ValueError(f"{where} failed")
    if where == "receive":
        transport.failure = error
    elif where == "decode":
        monkeypatch.setattr(
            remote_loader, "unpack_python_hetero", Mock(side_effect=error)
        )
    else:

        def fail_schedule() -> Generator[BatchRequest, None, None]:
            yield from ()
            raise error

        monkeypatch.setattr(source, "iter_requests", fail_schedule)
    with RemoteDataLoader(source) as loader:
        with pytest.raises(ValueError, match=f"{where} failed"):
            next(iter(loader))
        active = loader._batch_dataset._remote_iterator
        assert active is not None
        assert not active._dispatcher.is_alive()


def test_empty_epoch_and_abandoned_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Support empty ranks and close an unconsumed iterator on epoch reset."""
    transport = _Transport(monkeypatch)
    with RemoteDataLoader(_Source(transport.config, 0)) as loader:
        assert list(loader) == []
    transport.pause = True
    with RemoteDataLoader(_Source(transport.config)) as loader:
        first = iter(loader)
        transport.await_submitted(3)
        second = iter(loader)
        assert next(first, None) is None
        transport.pause = False
        assert len(list(second)) == 8


def test_drain_timeout_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed drain retains ownership until a subsequent close succeeds."""
    transport = _Transport(monkeypatch)
    transport.pause = True
    loader = RemoteDataLoader(_Source(transport.config))
    iter(loader)
    transport.await_submitted(3)
    transport.drain.side_effect = TimeoutError("drain timed out")
    with pytest.raises(TimeoutError, match="drain timed out"):
        loader.close()
    transport.drain.side_effect = lambda: _Ref(None)
    loader.close()
    loader.close()


@pytest.mark.parametrize("failure", ["object", "actor"])
def test_confirmed_transport_loss_retries_same_request(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Retain the original credit while retrying on an available healthy actor."""
    original_get = _Transport.get
    lost: list[int] = []

    def lose_once(
        transport: _Transport,
        ref: _Ref,
        **options: object,
    ) -> HeteroBatchPayload | None:
        if ref.value is not None and not lost:
            lost.append(ref.ordinal)
            if failure == "actor":
                raise ray.exceptions.ActorDiedError()
            raise ray.exceptions.ObjectLostError("0" * 56, None, "test")
        return original_get(transport, ref, **options)

    monkeypatch.setattr(_Transport, "get", lose_once)
    transport = _Transport(monkeypatch)
    with RemoteDataLoader(_Source(transport.config)) as loader:
        received = list(loader)
    assert sorted(int(batch["nodes"].n_id.item()) for batch in received) == list(
        range(8)
    )
    assert len(transport.submitted) == 9
    assert sum(request.batch_ordinal == lost[0] for request in transport.submitted) == 2


def test_task_dataset_owns_direct_remote_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct dataset users receive graphs and cannot pickle a live receiver."""
    transport = _Transport(monkeypatch)
    source = _Source(transport.config)
    try:
        first = iter(source)
        assert isinstance(next(first), HeteroData)
        previous = source._remote_iterator
        assert previous is not None
        source.__dict__["_runtime_index_handle"] = object()
        state = source.__getstate__()
        assert state["_remote_iterator"] is None
        assert "_runtime_index_handle" not in state
        # Replacing the dataset iterator drains its previous receiver directly.
        second = iter(source)
        assert previous.closed
        assert not previous._dispatcher.is_alive()
        assert next(first, None) is None
        assert len(list(second)) == len(source)
    finally:
        source.close()


def test_remote_dataset_rejects_subprocess_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a forked remote dataset before it can connect or submit requests."""
    transport = _Transport(monkeypatch)
    source = _Source(transport.config)
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info", lambda: object()
    )
    with pytest.raises(ValueError, match="in-process DataLoader"):
        iter(source)
    assert not transport.submitted


def test_remote_loader_requires_a_remote_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent an unencoded local schedule from entering the remote-only loader."""
    transport = _Transport(monkeypatch)
    source = _Source(transport.config)
    source.sampling_backend = None
    with pytest.raises(ValueError, match="remote sampling backend"):
        RemoteDataLoader(source)
