# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded completion-order delivery of actor-encoded graphs in one process."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Generator, Iterator
from queue import Empty, Queue
from threading import Event, Lock, Semaphore, Thread
from typing import Protocol, Self

from diskgraph import DiskgraphConfigError
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import HeteroData

from kumo_core.diskgraph.batch_contract import BatchRequest, HeteroBatchPayload
from kumo_core.diskgraph.hetero_wire import unpack_python_hetero
from kumo_core.diskgraph.ray_sampling import (
    RayBackendConfig,
    SamplingUnavailableError,
    _SamplerPool,
)

logger = logging.getLogger(__name__)


class RequestSource(Protocol):
    """Trainer-owned epoch schedule without sampling or column encoding."""

    def iter_requests(self) -> Generator[BatchRequest, None, None]:
        """Return this epoch's logical requests; closing releases its schedule."""
        ...

    def __len__(self) -> int:
        """Return the number of batches in this rank's epoch."""
        ...


class _RemoteIterator(Iterator[HeteroData]):
    def __init__(self, source: RequestSource, config: RayBackendConfig) -> None:
        self.samplers = _SamplerPool(config)
        self.requests = source.iter_requests()
        self.depth = len(self.samplers.replicas)
        self._credits = Semaphore(self.depth)
        self._credit_available = Event()
        self._stop = Event()
        self._close_lock = Lock()
        self._delivery_lock = Lock()
        self._iteration_lock = Lock()
        self._close_complete = False
        self._drain_error: Exception | None = None
        self._ready: Queue[HeteroBatchPayload | Exception | None] = Queue()
        self._dispatcher = Thread(
            target=self._dispatch, name="remote-sampling-receiver", daemon=True
        )
        self._dispatcher.start()

    @property
    def closed(self) -> bool:
        """Whether cancellation has fenced further trainer delivery."""
        return self._stop.is_set()

    def __next__(self) -> HeteroData:
        try:
            with self._iteration_lock:
                if self.closed:
                    raise StopIteration
                try:
                    item = self._ready.get(timeout=self.samplers.config.request_timeout)
                except Empty as error:
                    raise TimeoutError("Remote batch delivery timed out") from error
                if self.closed or item is None:
                    raise StopIteration
                if isinstance(item, Exception):
                    raise item
                batch = unpack_python_hetero(*item)
                with self._delivery_lock:
                    if self.closed:
                        raise StopIteration
                    self._credits.release()
                    self._credit_available.set()
                    return batch
        except BaseException:
            # Includes consumer interruption: no iterator may abandon owned RPCs.
            self.close()
            raise

    def _dispatch(self) -> None:
        from ray import ObjectRef

        pending: dict[ObjectRef[HeteroBatchPayload], tuple[BatchRequest, int]] = {}
        retries: deque[tuple[BatchRequest, int]] = deque()
        exhausted = False
        try:
            while not self.closed:
                while self.samplers.available and not self.closed:
                    if retries:
                        request, attempts = retries.popleft()
                    else:
                        if exhausted or not self._credits.acquire(False):
                            break
                        try:
                            request = next(self.requests)
                        except StopIteration:
                            exhausted = True
                            self._credits.release()
                            break
                        attempts = 0
                    pending[self.samplers.submit(request)] = (request, attempts)
                if not pending:
                    if retries:
                        raise SamplingUnavailableError(
                            "No healthy sampling actor remains"
                        )
                    if exhausted or self.closed:
                        return
                    self._credit_available.wait(0.05)
                    self._credit_available.clear()
                    continue
                ready, _ = self.samplers.ray.wait(
                    list(pending), num_returns=1, timeout=0.05
                )
                if ready:
                    result = ready[0]
                    request, attempts = pending.pop(result)
                    try:
                        payload = self.samplers.get(result)
                    except (
                        self.samplers.ray.exceptions.ActorDiedError,
                        self.samplers.ray.exceptions.ObjectLostError,
                    ) as error:
                        if attempts >= self.samplers.config.max_retries:
                            raise SamplingUnavailableError(
                                "Remote sampling retry budget exhausted"
                            ) from error
                        retries.append((request, attempts + 1))
                        logger.warning("Retrying remote sampling after %s", error)
                        continue
                    if not self.closed:
                        self._ready.put(payload)
                    del payload, result
                    ready.clear()
        except (
            DiskgraphConfigError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            self.samplers.ray.exceptions.RayError,
        ) as error:
            if self.closed:
                logger.debug("Remote receiver stopped during cancellation: %s", error)
            else:
                self._ready.put(error)
        finally:
            try:
                self.samplers.close()
            except (
                RuntimeError,
                OSError,
                self.samplers.ray.exceptions.RayError,
            ) as error:
                self._drain_error = error
                self._ready.put(error)
            finally:
                try:
                    self.requests.close()
                except (RuntimeError, ValueError, OSError) as error:
                    self._ready.put(error)
                finally:
                    self._ready.put(None)

    def close(self) -> None:
        """Fence delivery, cancel requests, and confirm serial actors have drained.

        Raises:
            RuntimeError: If the receiver cannot stop within the transport budget.
            ray.exceptions.RayError: If outstanding actor work cannot be drained.
        """
        with self._close_lock:
            if self._close_complete:
                return
            with self._delivery_lock:
                self._stop.set()
            self._credit_available.set()
            self._ready.put(None)
            try:
                self.samplers.cancel()
            finally:
                self._dispatcher.join(2 * self.samplers.config.request_timeout + 1)
            if self._dispatcher.is_alive():
                raise RuntimeError("Remote sampling receiver did not drain")
            with self._iteration_lock:
                if self._drain_error is not None:
                    self.samplers.close()
                    logger.warning(
                        "Retried remote request drain: %s", self._drain_error
                    )
                    self._drain_error = None
                while True:
                    try:
                        self._ready.get_nowait()
                    except Empty:
                        break
            self._close_complete = True


class _RemoteDataset(IterableDataset[HeteroData]):
    def __init__(self, source: RequestSource, config: RayBackendConfig) -> None:
        self.source = source
        self.config = config
        self.active: _RemoteIterator | None = None

    def __iter__(self) -> _RemoteIterator:
        if self.active is not None:
            self.active.close()
        self.active = _RemoteIterator(self.source, self.config)
        return self.active

    def __len__(self) -> int:
        return len(self.source)


class RemoteDataLoader(DataLoader[HeteroData]):
    """Receive actor-encoded graphs without trainer worker subprocesses.

    One slot per actor covers pending RPCs, received buffers, and reconstruction
    until the graph is handed to DataLoader. A receiver thread immediately refills
    released slots independently of model computation. Results use completion
    order, so actor timing can change optimizer batch order.

    Args:
        dataset: Epoch schedule shared with local DiskGraph training.
        config: Exclusively owned sampler pool and its connection settings.
        phase: Lifecycle name used by the DataModule shutdown barrier.
        pin_memory: Pin reconstructed CPU tensors using ordinary DataLoader logic.
    """

    def __init__(
        self,
        dataset: RequestSource,
        config: RayBackendConfig,
        *,
        phase: str = "train",
        pin_memory: bool = False,
    ) -> None:
        self.phase = phase
        self._remote_dataset = _RemoteDataset(dataset, config)
        super().__init__(
            self._remote_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=pin_memory,
        )
        self.in_order = False

    def close(self) -> None:
        """Cancel and drain active requests, including after early termination.

        Raises:
            RuntimeError: If outstanding remote work cannot drain before its deadline.
        """
        if self._remote_dataset.active is not None:
            self._remote_dataset.active.close()

    def __enter__(self) -> Self:
        """Return a loader whose requests will be drained on context exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the loader on normal exit or an exception.

        Args:
            *_exc: Context manager exception information.
        """
        self.close()
