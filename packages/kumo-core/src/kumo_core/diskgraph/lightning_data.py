# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightning datamodule over native DiskGraph task batch iterators."""
from __future__ import annotations

import ctypes
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import select
import signal
import sys
import threading
import time
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping

import pytorch_lightning as pl
from kumo_core.diskgraph.runtime_adapter import (
    MapperBank,
    engine_int64_to_ns,
    runtime_batch_to_hetero,
)
from kumo_core.task_dataset import DiskgraphTaskManifest
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


@contextmanager
def suppress_iterable_len_warning() -> Iterator[None]:
    r"""Silences Lightning's "IterableDataset has __len__ defined" warning
    for the duration of the wrapped call.

    Scoped with ``catch_warnings()`` (rather than a permanent, module-level
    ``filterwarnings()`` call) so it only silences this exact message for
    the duration of the ``pl.Trainer.{fit,validate,test,predict}`` call
    that actually queries this dataset's length -- not for unrelated
    ``IterableDataset``s that may run elsewhere in the same process and
    could have a genuinely broken ``__len__``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Your `IterableDataset` has `__len__` defined\.',
            category=UserWarning,
        )
        yield


@contextmanager
def suppress_batch_indices_warning() -> Iterator[None]:
    r"""Silences Lightning's "Couldn't infer the batch indices" warning
    for the duration of the wrapped ``predict()`` call.

    Lightning's PredictionLoop._get_batch_indices() (loops/prediction_loop.py)
    only knows how to recover per-batch sample indices when the dataloader's
    ``batch_sampler`` is wrapped by its own ``_IndexBatchSamplerWrapper``. Our
    ``_WorkerPoolDataLoader`` uses DiskGraph's own worker-pool batching
    instead, so Lightning can't infer batch indices and warns once, on every
    ``predict()`` call that uses a ``BasePredictionWriter`` with an
    on-batch write interval. Kumo's chunked prediction writers never read
    the ``batch_indices`` argument passed to ``write_on_batch_end`` -- they
    derive output partitioning from their own batch counter -- so the
    warning is cosmetic noise, not a signal of a real problem.

    Scoped with ``catch_warnings()`` (rather than a permanent, module-level
    ``filterwarnings()`` call) so it only silences this exact message for
    the duration of the ``pl.Trainer.predict()`` call that triggers it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r"Couldn't infer the batch indices fetched from your "
            r'dataloader',
            category=UserWarning,
        )
        yield


@contextmanager
def suppress_explain_workers_warning() -> Iterator[None]:
    r"""Silences Lightning's "does not have many workers" hint for the
    duration of an explanation pass.

    Explanation passes load data without worker processes: while capturing a
    stage's explanation inputs, the data module records each batch's input
    nodes on itself (see ``_make_dataloader``), and ``explain_dataloader``
    replays only the selected seed nodes. The hint suggests raising
    ``num_workers``, which is neither possible nor useful here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r"The '\w+' does not have many workers",
            category=UserWarning,
        )
        yield


@contextmanager
def suppress_checkpoint_callback_state_warnings() -> Iterator[None]:
    r"""Silences Lightning's warnings about training-callback state that a
    ``predict()`` call restoring ``ckpt_path`` cannot reload.

    The checkpoint carries the state of training callbacks such as early
    stopping and model checkpointing, and a published bundle no longer lives
    in the directory it was trained in. Prediction neither stops early nor
    saves checkpoints, so this state does not affect the result.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Be aware that when using `ckpt_path`, callbacks used '
            r'to create the checkpoint',
            category=UserWarning,
        )
        warnings.filterwarnings(
            'ignore',
            message=r'The dirpath has changed from',
            category=UserWarning,
        )
        yield


@contextmanager
def suppress_lightning_leafspec_warning() -> Iterator[None]:
    r"""Silences torch's ``LeafSpec`` deprecation warning for the duration of
    a Lightning ``fit``/``validate``/``test``/``predict`` call.

    From torch 2.10, constructing ``torch.utils._pytree.LeafSpec`` raises a
    ``FutureWarning``. Lightning's ``CombinedLoader`` flattens dataloaders with
    its own ``tree_flatten`` copy (``pytorch_lightning/utilities/_pytree.py``),
    which still constructs one in every release, including 2.6.6; upstream
    treats the warning as harmless. Kumo cannot act on it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'`isinstance\(treespec, LeafSpec\)` is deprecated',
            category=FutureWarning,
        )
        yield


_WORKER_COOPERATIVE_GRACE_SECONDS = 5.0
_WORKER_TERM_GRACE_SECONDS = 5.0
_WORKER_SHUTDOWN_DEADLINE_SECONDS = 90.0
_WORKER_SHUTDOWN_POLL_SECONDS = 0.05
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_PIDFD_OPEN_SYSCALL = 434


def _load_pidfd_libc() -> ctypes.CDLL | None:
    if not sys.platform.startswith('linux'):
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


_LIBC = _load_pidfd_libc()


def _raise_pidfd_error() -> None:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))


def _pidfd_open(pid: int) -> int | None:
    """Open a stable process handle on Linux, including minimal CPython."""
    if not sys.platform.startswith('linux'):
        return None
    native = getattr(os, 'pidfd_open', None)
    if native is not None:
        return native(pid)
    assert _LIBC is not None
    result = _LIBC.syscall(
        ctypes.c_long(_PIDFD_OPEN_SYSCALL),
        ctypes.c_int(pid),
        ctypes.c_uint(0),
    )
    if result < 0:
        _raise_pidfd_error()
    return int(result)


def _pidfd_send_signal(pidfd: int, sig: int) -> None:
    """Signal a pidfd even when CPython omitted the optional wrapper."""
    native = getattr(signal, 'pidfd_send_signal', None)
    if native is not None:
        native(pidfd, sig)
        return
    assert _LIBC is not None
    result = _LIBC.syscall(
        ctypes.c_long(_PIDFD_SEND_SIGNAL_SYSCALL),
        ctypes.c_int(pidfd),
        ctypes.c_int(sig),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )
    if result < 0:
        _raise_pidfd_error()


class DataLoaderWorkerShutdownError(RuntimeError):
    """A DiskGraph DataLoader pool did not reach a reusable shutdown state."""


class _WorkerPoolState(Enum):
    INITIALIZING = 'initializing'
    ACTIVE = 'active'
    CLOSING = 'closing'
    QUIESCENT = 'quiescent'
    FAILED = 'failed'


@dataclass
class _WorkerHandle:
    """Stable kernel identity used while PyTorch owns the Process object."""
    process: Any
    pid: int
    sentinel_fd: int
    pidfd: int | None
    _closed: bool = False

    @classmethod
    def open(cls, process: Any) -> _WorkerHandle:
        """Capture a worker identity before its private shutdown can reap it."""
        pid = process.pid
        if pid is None:
            raise RuntimeError('DataLoader worker has no pid')
        sentinel_fd = os.dup(process.sentinel)
        pidfd = None
        try:
            pidfd = _pidfd_open(pid)
        except ProcessLookupError:
            # The duplicated sentinel still gives an authoritative exit
            # observation for a worker that won the startup/exit race.
            pass
        except BaseException:
            try:
                os.close(sentinel_fd)
            except OSError:
                pass
            raise
        return cls(process, int(pid), sentinel_fd, pidfd)

    @staticmethod
    def _fd_ready(fd: int) -> bool:
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        return bool(poller.poll(0))

    def exited(self) -> bool:
        """Return whether the whole worker process has exited."""
        return self._fd_ready(
            self.pidfd if self.pidfd is not None else self.sentinel_fd)

    def send_signal(self, sig: int) -> None:
        """Signal the captured process identity without touching Process."""
        if self.exited():
            return
        try:
            if self.pidfd is not None:
                _pidfd_send_signal(self.pidfd, sig)
            elif not sys.platform.startswith('linux'):
                # Non-Linux development fallback. Production Linux/Python 3.14
                # uses pidfds, avoiding numeric-PID reuse entirely.
                os.kill(self.pid, sig)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        """Release duplicated observation descriptors."""
        if self._closed:
            return
        self._closed = True
        first_error: OSError | None = None
        try:
            os.close(self.sentinel_fd)
        except OSError as error:
            first_error = error
        if self.pidfd is not None:
            try:
                os.close(self.pidfd)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _worker_task_states(pid: int) -> dict[int, str]:
    """Best-effort per-task states for a worker that missed its deadline."""
    states: dict[int, str] = {}
    try:
        task_names = os.listdir(f'/proc/{pid}/task')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return states
    for task_name in task_names:
        if not task_name.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/task/{task_name}/stat', 'rb') as stat_file:
                stat = stat_file.read()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        fields = stat[stat.rfind(b')') + 2:].split()
        if fields:
            states[int(task_name)] = fields[0].decode(errors='replace')
    return states


class _WorkerPoolRegistry:
    """Own active DG DataLoader iterators until their workers are reaped."""
    def __init__(self) -> None:
        self._active: dict[int, _WorkerPoolIterator] = {}
        self._lock = threading.RLock()

    def register(self, iterator: _WorkerPoolIterator) -> None:
        with self._lock:
            self._active[id(iterator)] = iterator

    def discard(self, iterator: _WorkerPoolIterator) -> None:
        with self._lock:
            self._active.pop(id(iterator), None)

    def complete(self, iterator: _WorkerPoolIterator) -> None:
        """Atomically publish quiescence and remove one active iterator."""
        with self._lock:
            if iterator._publish_quiescent():
                self._active.pop(id(iterator), None)

    def _matching(self, phase: str | None) -> tuple[_WorkerPoolIterator, ...]:
        with self._lock:
            return tuple(iterator for iterator in self._active.values()
                         if phase is None or iterator.phase == phase)

    def ensure_can_start(self, phase: str) -> None:
        active_phases = tuple(iterator.phase
                              for iterator in self._matching(None))
        if active_phases:
            raise RuntimeError(
                f"Cannot start {phase!r} DataLoader workers while "
                f"{active_phases!r} workers are active. DiskGraph phase-local "
                "worker pools must not overlap; use epoch-boundary validation.")

    def shutdown(
        self,
        phase: str | None = None,
        *,
        reason: str = 'phase-barrier',
    ) -> None:
        """Strictly close every matching pool before another phase can fork."""
        first_error: BaseException | None = None
        for iterator in self._matching(phase):
            try:
                iterator.close(reason)
            except BaseException as error:  # attempt every owned pool
                if first_error is None:
                    first_error = error
                try:
                    logger.exception("Failed to close %s DataLoader workers",
                                     iterator.phase)
                except BaseException:
                    pass
        if first_error is not None:
            raise first_error

    @property
    def active_phases(self) -> tuple[str, ...]:
        """Return phases that still own registered iterator state."""
        return tuple(iterator.phase for iterator in self._matching(None))

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(pid for iterator in self._matching(None)
                     for pid in iterator.worker_pids)


class _WorkerPoolIterator(Iterator[Any]):
    """Own one worker pool and close it synchronously at a phase barrier.

    PyTorch exposes no public iterator-close API. This is the same private
    ``_shutdown_workers`` hook Lightning 2.6 uses in ``CombinedLoader.reset``,
    isolated in one helper while the phase thread owns deadline enforcement.
    """
    def __init__(
        self,
        iterator: Iterator[Any],
        *,
        loader: DataLoader,
        registry: _WorkerPoolRegistry,
        phase: str,
    ) -> None:
        self._iterator: Iterator[Any] | None = iterator
        self._loader_ref = weakref.ref(loader)
        self._registry_ref = weakref.ref(registry)
        self.phase = phase
        self.fetched = 0
        self._workers = tuple(getattr(iterator, '_workers', ()))
        self._worker_handles: tuple[_WorkerHandle, ...] = ()
        self._state = _WorkerPoolState.INITIALIZING
        self._state_condition = threading.Condition()
        self._close_started = False
        self._started_at: float | None = None
        self._deadline_at: float | None = None
        self._error: DataLoaderWorkerShutdownError | None = None
        self._remaining_worker_pids: tuple[int, ...] = ()
        registry.register(self)
        self._initialize_worker_handles()

    def __iter__(self) -> _WorkerPoolIterator:
        return self

    def __next__(self) -> Any:
        iterator = self._iterator
        if iterator is None:
            raise StopIteration
        try:
            batch = next(iterator)
        except StopIteration:
            self.close('stop-iteration')
            raise
        except BaseException:
            try:
                self.close('fetch-error')
            except BaseException:
                try:
                    logger.exception(
                        "Failed to close DataLoader after fetch error")
                except BaseException:
                    pass
            raise

        self.fetched += 1
        # A successful fetch is never coupled to cleanup. Lightning may stop at
        # len(loader) without requesting another item, so its phase callback
        # owns the strict barrier after this batch has been fully processed.
        return batch

    @property
    def worker_pids(self) -> tuple[int, ...]:
        with self._state_condition:
            if self._state is _WorkerPoolState.FAILED:
                return self._remaining_worker_pids
            return tuple(handle.pid for handle in self._worker_handles
                         if not handle.exited())

    @property
    def state(self) -> _WorkerPoolState:
        """Return the current lifecycle state for tests and diagnostics."""
        with self._state_condition:
            return self._state

    def _initialize_worker_handles(self) -> None:
        """Capture every stable identity before publishing an active pool."""
        handles: list[_WorkerHandle] = []
        try:
            for worker in self._workers:
                handles.append(_WorkerHandle.open(worker))
        except BaseException as error:
            self._close_handles(handles)
            remaining = tuple(
                int(worker.pid) for worker in self._workers
                if getattr(worker, 'pid', None) is not None)
            self._fail_cleanup(
                f'Could not capture stable worker handles: {error!r}',
                remaining_pids=remaining,
                close_worker_handles=False,
            )
            self._emergency_cleanup_after_initialization_failure()
            raise

        close_raced = False
        rejected_error: DataLoaderWorkerShutdownError | None = None
        with self._state_condition:
            if self._state is _WorkerPoolState.FAILED:
                rejected_error = self._error
            else:
                self._worker_handles = tuple(handles)
                handles.clear()
                close_raced = self._close_started
                self._state = (_WorkerPoolState.CLOSING
                               if close_raced else _WorkerPoolState.ACTIVE)
                self._state_condition.notify_all()
        if handles:
            self._close_handles(handles)
        if rejected_error is not None:
            # A close-side initialization deadline can publish FAILED before
            # this thread finishes capturing handles. The close owner had no
            # handles to act on, so the resumed initializer must still reap
            # the already-forked workers before it propagates the poison.
            self._emergency_cleanup_after_initialization_failure()
            raise rejected_error
        if close_raced:
            self.close('initialization-race')
            raise DataLoaderWorkerShutdownError(
                f'{self.phase} DataLoader was closed during initialization')

    @staticmethod
    def _close_handles(
        handles: list[_WorkerHandle] | tuple[_WorkerHandle, ...]
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                errors.append(error)
                try:
                    logger.exception('Failed to close worker handle for pid=%s',
                                     handle.pid)
                except BaseException:
                    pass
        return errors

    def _emergency_cleanup_after_initialization_failure(self) -> None:
        """Bound cleanup when stable-handle capture failed after worker start."""
        iterator = self._iterator
        shutdown = getattr(iterator, '_shutdown_workers', None)
        private_done = threading.Event()

        def _private_shutdown() -> None:
            try:
                if callable(shutdown):
                    shutdown()
            except BaseException:
                try:
                    logger.exception(
                        'PyTorch shutdown failed after handle-capture failure')
                except BaseException:
                    pass
            finally:
                private_done.set()

        try:
            private_thread = threading.Thread(
                target=_private_shutdown,
                name=f'diskgraph-{self.phase}-construction-shutdown',
                daemon=True,
            )
            private_thread.start()
        except BaseException:
            try:
                logger.exception(
                    'Could not start emergency DataLoader shutdown helper')
            except BaseException:
                pass

        started_at = time.monotonic()
        deadline = started_at + _WORKER_SHUTDOWN_DEADLINE_SECONDS
        term_at = started_at + _WORKER_COOPERATIVE_GRACE_SECONDS
        kill_at = term_at + _WORKER_TERM_GRACE_SECONDS
        term_sent = False
        kill_sent = False
        while True:
            live = []
            for worker in self._workers:
                try:
                    if _WorkerHandle._fd_ready(worker.sentinel):
                        continue
                except BaseException:
                    pass
                live.append(worker)
            if not live:
                break
            now = time.monotonic()
            if now >= deadline:
                break
            if not term_sent and now >= term_at:
                for worker in live:
                    try:
                        worker.terminate()
                    except BaseException:
                        pass
                term_sent = True
            if not kill_sent and now >= kill_at:
                for worker in live:
                    try:
                        worker.kill()
                    except BaseException:
                        pass
                kill_sent = True
            next_event = min(
                deadline, kill_at if term_sent and not kill_sent else
                term_at if not term_sent else deadline)
            private_done.wait(
                min(_WORKER_SHUTDOWN_POLL_SECONDS,
                    max(0.0, next_event - time.monotonic())))

        for worker in self._workers:
            try:
                worker.join(timeout=0)
            except BaseException:
                pass
        if live:
            try:
                logger.error(
                    'DataLoader workers remained alive after emergency '
                    'construction cleanup: phase=%s pids=%s', self.phase,
                    tuple(worker.pid for worker in live))
            except BaseException:
                pass

    def _live_handles(self) -> tuple[_WorkerHandle, ...]:
        # Hold the same lock used while detaching/closing the handles. Registry
        # diagnostics may otherwise poll a descriptor after cleanup closed it.
        with self._state_condition:
            return tuple(handle for handle in self._worker_handles
                         if not handle.exited())

    def _signal_live_workers(self, sig: int) -> None:
        live = self._live_handles()
        if not live:
            return
        try:
            logger.warning(
                'Escalating %s DataLoader worker shutdown: signal=%s workers=%s',
                self.phase,
                signal.Signals(sig).name, tuple(handle.pid for handle in live))
        except BaseException:
            pass
        errors: list[BaseException] = []
        for handle in live:
            try:
                handle.send_signal(sig)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def _is_expected_worker_signal_error(self, error: RuntimeError) -> bool:
        """Recognize PyTorch's SIGCHLD report for a worker we signaled."""
        message = str(error)
        with self._state_condition:
            worker_pids = tuple(handle.pid for handle in self._worker_handles)
        return any(
            message.startswith(
                f'DataLoader worker (pid {pid}) is killed by signal:')
            for pid in worker_pids)

    def _run_close(self, deadline: float) -> None:
        started_at = self._started_at
        assert started_at is not None
        term_at = started_at + _WORKER_COOPERATIVE_GRACE_SECONDS
        kill_at = term_at + _WORKER_TERM_GRACE_SECONDS
        private_done = threading.Event()
        private_errors: list[BaseException] = []
        private_start_error: BaseException | None = None
        iterator = self._iterator
        shutdown = getattr(iterator, '_shutdown_workers', None)

        def _private_shutdown() -> None:
            try:
                assert callable(shutdown)
                shutdown()
            except BaseException as error:
                private_errors.append(error)
            finally:
                private_done.set()

        if not callable(shutdown):
            private_start_error = RuntimeError(
                'PyTorch DataLoader has no shutdown hook')
        else:
            try:
                private_thread = threading.Thread(
                    target=_private_shutdown,
                    name=f'diskgraph-{self.phase}-pytorch-shutdown',
                    daemon=True,
                )
                private_thread.start()
            except BaseException as error:
                private_start_error = error

        term_sent = False
        kill_sent = False
        while True:
            try:
                live = self._live_handles()
                private_finished = (private_start_error is not None
                                    or private_done.is_set())
                if not live and private_finished:
                    break
                now = time.monotonic()
                if now >= deadline:
                    details = {
                        handle.pid: _worker_task_states(handle.pid)
                        for handle in live
                    }
                    self._fail_cleanup(
                        'DataLoader workers did not terminate before the '
                        f'absolute deadline: pids={[h.pid for h in live]} '
                        f'task_states={details} '
                        f'private_done={private_done.is_set()}')
                    return
                if not term_sent and now >= term_at:
                    term_sent = True
                    self._signal_live_workers(signal.SIGTERM)
                if not kill_sent and now >= kill_at:
                    kill_sent = True
                    self._signal_live_workers(signal.SIGKILL)
                next_event = min(
                    deadline, kill_at if term_sent and not kill_sent else
                    term_at if not term_sent else deadline)
                private_done.wait(
                    min(_WORKER_SHUTDOWN_POLL_SECONDS,
                        max(0.0, next_event - time.monotonic())))
            except RuntimeError as error:
                if (not (term_sent or kill_sent)
                        or not self._is_expected_worker_signal_error(error)):
                    raise

        if private_start_error is not None:
            self._fail_cleanup(
                'Could not start PyTorch DataLoader shutdown helper: '
                f'{private_start_error!r}')
            return
        if private_errors:
            self._detach_iterator()
            self._fail_cleanup(
                f'PyTorch DataLoader shutdown hook failed: {private_errors!r}')
            return

        normalization_errors: list[BaseException] = []
        with self._state_condition:
            handles = self._worker_handles
        for handle in handles:
            try:
                handle.process.join(timeout=0)
            except BaseException as error:
                normalization_errors.append(error)
        self._detach_iterator()
        if normalization_errors:
            self._fail_cleanup('Failed to normalize reaped DataLoader workers: '
                               f'{normalization_errors!r}')
            return
        handle_errors = self._close_worker_handles()
        if handle_errors:
            self._fail_cleanup(
                f'Failed to close stable worker handles: {handle_errors!r}',
                close_worker_handles=False,
            )
            return
        self._complete_cleanup()

    def _detach_iterator(self) -> None:
        with self._state_condition:
            iterator = self._iterator
            self._iterator = None
            self._workers = ()
        loader = self._loader_ref()
        if loader is not None and getattr(loader, '_iterator',
                                          None) is iterator:
            # Required for persistent_workers=True and to make Lightning's
            # later CombinedLoader.reset() a no-op for this pool.
            loader._iterator = None

    def _close_worker_handles(self) -> list[BaseException]:
        with self._state_condition:
            handles = self._worker_handles
            self._worker_handles = ()
        return self._close_handles(handles)

    def _publish_quiescent(self) -> bool:
        """Publish success only if no concurrent deadline failure won."""
        with self._state_condition:
            if self._state is _WorkerPoolState.FAILED:
                return False
            self._state = _WorkerPoolState.QUIESCENT
            self._state_condition.notify_all()
            return True

    def _complete_cleanup(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            registry.complete(self)
        else:
            self._publish_quiescent()
        with self._state_condition:
            quiescent = self._state is _WorkerPoolState.QUIESCENT
            started_at = self._started_at
        if not quiescent:
            return
        elapsed = (time.monotonic() -
                   started_at if started_at is not None else 0.0)
        try:
            logger.info('%s DataLoader worker shutdown completed in %.3fs',
                        self.phase, elapsed)
        except BaseException:
            pass

    def _remaining_pids_locked(self) -> tuple[int, ...]:
        remaining = []
        for handle in self._worker_handles:
            try:
                if not handle.exited():
                    remaining.append(handle.pid)
            except BaseException:
                remaining.append(handle.pid)
        return tuple(remaining)

    def _set_failure_locked(
        self,
        detail: str,
        *,
        remaining_pids: tuple[int, ...] | None = None,
    ) -> bool:
        if self._state is _WorkerPoolState.QUIESCENT:
            return False
        if self._state is _WorkerPoolState.FAILED:
            return False
        self._remaining_worker_pids = (self._remaining_pids_locked()
                                       if remaining_pids is None else
                                       remaining_pids)
        self._error = DataLoaderWorkerShutdownError(
            f'{self.phase} DataLoader worker shutdown failed: {detail}')
        self._state = _WorkerPoolState.FAILED
        self._state_condition.notify_all()
        return True

    def _fail_cleanup(
        self,
        detail: str,
        *,
        remaining_pids: tuple[int, ...] | None = None,
        close_worker_handles: bool = True,
    ) -> None:
        """Idempotently publish failure without depending on worker I/O."""
        with self._state_condition:
            published = self._set_failure_locked(detail,
                                                 remaining_pids=remaining_pids)
            error = self._error
        if close_worker_handles:
            self._close_worker_handles()
        if published and error is not None:
            try:
                logger.error('%s', error)
            except BaseException:
                pass

    def _raise_terminal_error(self) -> None:
        with self._state_condition:
            error = self._error
        if error is not None:
            raise error

    def close(self, reason: str = 'explicit-close') -> None:
        """Synchronously enforce one absolute phase-level shutdown deadline."""
        owner = False
        close_handles = False
        with self._state_condition:
            if self._state is _WorkerPoolState.QUIESCENT:
                return
            if self._state is _WorkerPoolState.FAILED:
                assert self._error is not None
                raise self._error
            if self._close_started:
                deadline = self._deadline_at
                assert deadline is not None
                while self._state not in (_WorkerPoolState.QUIESCENT,
                                          _WorkerPoolState.FAILED):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        close_handles = self._set_failure_locked(
                            'Synchronous close owner did not publish a terminal '
                            'state before the absolute deadline')
                        break
                    self._state_condition.wait(remaining)
            else:
                owner = True
                self._close_started = True
                self._started_at = time.monotonic()
                self._deadline_at = (self._started_at +
                                     _WORKER_SHUTDOWN_DEADLINE_SECONDS)
                while self._state is _WorkerPoolState.INITIALIZING:
                    remaining = self._deadline_at - time.monotonic()
                    if remaining <= 0:
                        close_handles = self._set_failure_locked(
                            'Stable worker-handle initialization did not finish '
                            'before the absolute deadline')
                        break
                    self._state_condition.wait(remaining)
                if self._state is _WorkerPoolState.ACTIVE:
                    self._state = _WorkerPoolState.CLOSING

        if close_handles:
            self._close_worker_handles()
        if not owner:
            self._raise_terminal_error()
            return
        self._raise_terminal_error()
        with self._state_condition:
            deadline = self._deadline_at
        assert deadline is not None
        try:
            logger.info('Closing %s DataLoader workers: reason=%s workers=%s',
                        self.phase, reason,
                        tuple(handle.pid for handle in self._worker_handles))
        except BaseException:
            pass
        try:
            self._run_close(deadline)
        except BaseException as error:
            self._fail_cleanup(
                f'Synchronous cleanup failed unexpectedly: {error!r}')
            if not isinstance(error, Exception):
                raise
        self._raise_terminal_error()


class _WorkerPoolDataLoader(DataLoader):
    """DataLoader whose iterator has explicit worker-pool ownership."""
    def __init__(self, *args: Any, phase: str, registry: _WorkerPoolRegistry,
                 **kwargs: Any) -> None:
        self._worker_pool_phase = phase
        self._worker_pool_registry = registry
        super().__init__(*args, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        # Check before super().__iter__(), which is where PyTorch forks the
        # processes. In particular, reject unsupported mid-epoch validation
        # without first doubling the process/memory footprint.
        self._worker_pool_registry.ensure_can_start(self._worker_pool_phase)
        try:
            iterator = super().__iter__()
        except BaseException:
            # Persistent-worker setup may assign a partially constructed
            # iterator before raising. Take ownership if it is discoverable so
            # already-started workers cannot escape the registry.
            partial = getattr(self, '_iterator', None)
            if partial is not None:
                try:
                    managed = _WorkerPoolIterator(
                        partial,
                        loader=self,
                        registry=self._worker_pool_registry,
                        phase=self._worker_pool_phase,
                    )
                    managed.close('pytorch-iterator-construction-error')
                except BaseException:
                    try:
                        logger.exception(
                            'Failed to clean a partially constructed %s '
                            'DataLoader iterator', self._worker_pool_phase)
                    except BaseException:
                        pass
            raise
        return _WorkerPoolIterator(
            iterator,
            loader=self,
            registry=self._worker_pool_registry,
            phase=self._worker_pool_phase,
        )


def _load_json(path: str) -> Mapping[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _stage_bucket_counts(
    runtime_manifest_path: str,
    stage: str,
) -> list[tuple[int, int]]:
    """Per-timestamp bucket ``(ts, count)`` list for ``stage``, read from the
    DG ``diskgraph_manifest.json`` (the same file the wheel's
    ``read_bucket_index`` consumes).

    The flat ``runtime_manifest.json`` names each stage's ``index_path`` (the TT
    extension root) and ``table_name``; ``diskgraph_manifest.json`` lives at
    ``<index_path>/diskgraph_manifest.json`` and keys stages by stage-key with a
    per-stage ``table_name`` + ``bucket_index.buckets``. We match the stage by
    ``table_name`` (mirroring ``read_bucket_index``). FAILS LOUD when grouping
    was requested but the artifact / bucket list is missing — never a silent
    flat fallback."""
    runtime = _load_json(runtime_manifest_path)
    stage_ref = (runtime.get("stages") or {}).get(stage)
    if not isinstance(stage_ref, Mapping):
        raise ValueError(f"runtime_manifest.json has no stage {stage!r}; "
                         f"available: {list((runtime.get('stages') or {}))}")
    ext_root = stage_ref.get("index_path")
    table_name = stage_ref.get("table_name")
    if not ext_root or not table_name:
        raise ValueError(
            f"runtime_manifest stage {stage!r} missing index_path/table_name")
    dg_manifest_path = os.path.join(str(ext_root), "diskgraph_manifest.json")
    dg_manifest = _load_json(dg_manifest_path)
    stages = (dg_manifest.get("task_dataset") or {}).get("stages") or {}
    matches = [
        v for v in stages.values()
        if isinstance(v, Mapping) and v.get("table_name") == table_name
    ]
    if not matches:
        raise ValueError(
            f"group_by_timestamp requested but no stage with table_name "
            f"{table_name!r} in {dg_manifest_path}")
    if len(matches) > 1:
        raise ValueError(
            f"{dg_manifest_path} has multiple stages with table_name "
            f"{table_name!r} (stage table_names must be distinct)")
    bucket_index = matches[0].get("bucket_index")
    if not isinstance(bucket_index, Mapping):
        raise ValueError(
            f"group_by_timestamp requested but stage {table_name!r} has no "
            f"bucket_index in {dg_manifest_path} (ingest did not emit timestamp "
            f"grouping for this stage)")
    buckets = bucket_index.get("buckets")
    if not isinstance(buckets, list):
        raise ValueError(
            f"bucket_index for stage {table_name!r} has no buckets array")
    return [(int(b["ts"]), int(b["count"])) for b in buckets]


def _binary_label_index_for_stage(
    runtime_manifest_path: str,
    stage: str,
) -> Mapping[str, Any]:
    """Binary target sidecar metadata for weighted replacement seed sampling."""
    runtime = _load_json(runtime_manifest_path)
    stage_ref = (runtime.get("stages") or {}).get(stage)
    if not isinstance(stage_ref, Mapping):
        raise ValueError(f"runtime_manifest.json has no stage {stage!r}; "
                         f"available: {list((runtime.get('stages') or {}))}")
    index = stage_ref.get("binary_label_index")
    if not isinstance(index, Mapping):
        raise ValueError(
            f"WeightMode.SAMPLE requires binary_label_index for stage {stage!r}; "
            "re-materialize the TT-NG task dataset with a wheel that emits "
            "binary label sidecars.")
    for key in ("false_count", "true_count"):
        if key not in index:
            raise ValueError(
                f"binary_label_index for stage {stage!r} missing {key}")
    return index


def _sample_weight_index_for_stage(
    runtime_manifest_path: str,
    stage: str,
) -> Mapping[str, Any]:
    """Row-weight sidecar metadata for weighted replacement seed sampling."""
    runtime = _load_json(runtime_manifest_path)
    stage_ref = (runtime.get("stages") or {}).get(stage)
    if not isinstance(stage_ref, Mapping):
        raise ValueError(f"runtime_manifest.json has no stage {stage!r}; "
                         f"available: {list((runtime.get('stages') or {}))}")
    index = stage_ref.get("sample_weight_index")
    if not isinstance(index, Mapping):
        raise ValueError(
            f"WeightMode.SAMPLE requires sample_weight_index for stage {stage!r}; "
            "re-materialize the TT-NG task dataset with a wheel that emits "
            "row-weight sidecars.")
    return index


def _bucket_total_weight(index: Mapping[str, Any], bucket: str) -> float:
    value = index.get(bucket)
    if not isinstance(value, Mapping):
        raise ValueError(f"sample_weight_index missing {bucket!r} bucket")
    return float(value.get("total_weight", 0.0))


@dataclass
class _ShuffleStreamState:
    stream_id: int = 0


def _num_nodes_for_type(
    graph_runtime_metadata: Mapping[str, Any] | None,
    node_type: str,
) -> int:
    """Node capacity (``max_id + 1``) for ``node_type`` from
    ``index_metadata.json``; ``0`` if missing."""
    index_path = (graph_runtime_metadata or {}).get('graph_index_path')
    if not index_path:
        return 0
    meta_path = os.path.join(index_path, 'index_metadata.json')
    if not os.path.exists(meta_path):
        return 0
    with open(meta_path) as f:
        idx_meta = json.load(f)
    for entry in idx_meta.get('node_types') or []:
        if entry.get('name') == node_type:
            max_id = entry.get('max_id', -1)
            return max_id + 1 if max_id >= 0 else 0
    return 0


# DG TimeUnit name -> numpy datetime64 unit. The wheel emits seed times in the
# index's stored unit (``index_metadata.json::timestamp_unit``); default secs.
_DG_UNIT_TO_NP = {
    'seconds': 's',
    'second': 's',
    'secs': 's',
    's': 's',
    'milliseconds': 'ms',
    'millis': 'ms',
    'ms': 'ms',
    'microseconds': 'us',
    'micros': 'us',
    'us': 'us',
    'nanoseconds': 'ns',
    'nanos': 'ns',
    'ns': 'ns',
}

# numpy datetime64 unit -> nanoseconds-per-one-tick. Used to convert Kumo's
# ns-canonical timestamps to the engine's configured unit at engine boundaries
# (e.g. the NodeRange ``rhs_seed_time``), and vice-versa for normalizing engine
# outputs to ns at the adapter.
_NP_UNIT_TO_NS_DIVISOR = {
    's': 1_000_000_000,
    'ms': 1_000_000,
    'us': 1_000,
    'ns': 1,
}


def _index_np_time_unit(
    graph_runtime_metadata: Mapping[str, Any] | None, ) -> str:
    """numpy datetime64 unit for the wheel's emitted seed times, read from
    ``index_metadata.json::timestamp_unit``. Falls back to ``'s'`` (the wheel
    default) when the index/field is absent or unrecognized."""
    index_path = (graph_runtime_metadata or {}).get('graph_index_path')
    if not index_path:
        return 's'
    meta_path = os.path.join(index_path, 'index_metadata.json')
    if not os.path.exists(meta_path):
        return 's'
    with open(meta_path) as f:
        unit = (json.load(f).get('timestamp_unit') or '').lower()
    return _DG_UNIT_TO_NP.get(unit, 's')


class DiskgraphValidRHS:
    """Temporally-valid RHS candidate provider for TT-NG LP.

    Implements just the ``valid_rhs_idx_dict`` / ``get_valid_rhs_idx`` contract
    that ShallowRHS / Hybrid consume via ``datamodule.lp_train_table``.
    Validity (per ``RHSTime.get_valid_nodes``)::

        create_time <= seed_time + unix_end_offset
        AND (end_time == -1 OR end_time > seed_time + unix_start_offset)

    Computed on the fly per ``seed_time``.
    """
    def __init__(
        self,
        *,
        num_rhs_nodes: int,
        unix_start_offset: int,
        unix_end_offset: int,
        create_time: "Any",  # torch.Tensor
        end_time: "Any | None" = None,  # torch.Tensor | None
        stage_time_dict: "Mapping[Any, tuple[int, int]] | None" = None,
    ) -> None:
        from kumo_core.training.supervision.link_prediction import RHSTime

        self.num_rhs_nodes = int(num_rhs_nodes)
        self._stage_time_dict = dict(stage_time_dict or {})
        # Reuse RHSTime for the validity arithmetic.
        self._rhs_time = RHSTime(
            unix_start_offset=int(unix_start_offset),
            unix_end_offset=int(unix_end_offset),
            create_time=create_time,
            end_time=end_time,
        )

    @property
    def valid_rhs_idx_dict(self) -> "dict | None":
        """Non-``None`` sentinel iff temporal validity info is present (callers
        short-circuit to "no filter" on ``None``). Empty dict, not the legacy
        per-timestamp mapping, since validity is computed on demand.
        """
        if self._rhs_time.create_time is None and (self._rhs_time.end_time
                                                   is None):
            return None
        return {}

    def get_valid_rhs_idx(
        self,
        time_or_stage: "Any",
        device: "Any | None" = None,
    ) -> "Any":
        """Return the global RHS node ids valid at ``time_or_stage``.

        ``time_or_stage`` is either an integer seed time (train + eval
        single-time path used by ShallowRHS / Hybrid) or a
        ``TrainingStage`` (range over ``stage_time_dict``).
        """
        import torch
        from kumoapi.train import TrainingStage

        if self.valid_rhs_idx_dict is None:
            return torch.arange(self.num_rhs_nodes, device=device)

        if isinstance(time_or_stage, TrainingStage):
            min_time, max_time = self._stage_time_dict[time_or_stage]
            times = torch.tensor(sorted({int(min_time),
                                         int(max_time)}), dtype=torch.long)
            per_time = self._rhs_time.get_valid_nodes(times)
            assert per_time is not None
            mask = torch.zeros(self.num_rhs_nodes, dtype=torch.bool)
            for idx in per_time.values():
                mask[idx] = True
            return mask.nonzero().view(-1).to(device)

        t = torch.tensor([int(time_or_stage)], dtype=torch.long)
        per_time = self._rhs_time.get_valid_nodes(t)
        assert per_time is not None
        return per_time[int(time_or_stage)].to(device)


def _interleave_stage_batches(targets: Mapping[str, int]) -> list[str]:
    """Spread each stage's allocated batches across the epoch."""
    positive = {stage: count for stage, count in targets.items() if count > 0}
    total = sum(positive.values())
    current = {stage: 0 for stage in positive}
    remaining = dict(positive)
    schedule: list[str] = []
    for _ in range(total):
        for stage, weight in positive.items():
            if remaining[stage] > 0:
                current[stage] += weight
        stage = max(
            (name for name in positive if remaining[name] > 0),
            key=lambda name: current[name],
        )
        schedule.append(stage)
        current[stage] -= total
        remaining[stage] -= 1
    return schedule


def _allocate_stage_batches(
    weights: Mapping[str, float],
    total_batches: int,
    *,
    require_each_stage: bool,
) -> dict[str, int]:
    nonfinite = [
        stage for stage, weight in weights.items()
        if not math.isfinite(float(weight))
    ]
    if nonfinite:
        raise ValueError(
            f"cannot allocate batches for non-finite stages: {nonfinite}")
    nonpositive = [
        stage for stage, weight in weights.items() if float(weight) <= 0
    ]
    if require_each_stage and nonpositive:
        raise ValueError(
            "cannot allocate a required batch to zero-mass stages: "
            f"{nonpositive}")
    positive = {
        stage: float(weight)
        for stage, weight in weights.items() if float(weight) > 0
    }
    if not positive:
        raise ValueError("cannot allocate batches across zero-mass stages")
    if require_each_stage and total_batches < len(positive):
        raise ValueError(
            f"{total_batches} batches cannot cover {len(positive)} stages")
    total_weight = sum(positive.values())
    exact = {
        stage: total_batches * weight / total_weight
        for stage, weight in positive.items()
    }
    allocations = {stage: math.floor(value) for stage, value in exact.items()}
    unallocated = total_batches - sum(allocations.values())
    order = sorted(
        positive,
        key=lambda stage: exact[stage] - math.floor(exact[stage]),
        reverse=True,
    )
    for stage in order[:unallocated]:
        allocations[stage] += 1
    if require_each_stage:
        for stage in positive:
            if allocations[stage] > 0:
                continue
            donor = max(
                (name for name, count in allocations.items() if count > 1),
                key=lambda name: allocations[name] - exact[name],
            )
            allocations[donor] -= 1
            allocations[stage] = 1
    return allocations


class DiskgraphTaskBatchIterable(IterableDataset):
    """Torch iterable over ``diskgraph.RuntimeIndex.iter``."""
    def __init__(
        self,
        *,
        manifest_path: str,
        stage: str | tuple[str, ...],
        sampling_plan: Any,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        base_seed: int,
        shuffle_state: _ShuffleStreamState,
        num_seeds: int | None = None,
        max_batches: int | None = None,
        replay: bool = True,
        runtime_options: Mapping[str, Any] | None = None,
        runtime_num_workers: int = 1,
        runtime_worker_id: int = 0,
        dataloader_num_workers: int = 0,
        group_by_timestamp: bool = False,
        min_steps_per_epoch: int = 1,
        max_steps_per_epoch: int | None = None,
        stage_batch_targets: Mapping[str, int] | None = None,
        stage_num_seeds: Mapping[str, int] | None = None,
        stage_runtime_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.stages = (stage, ) if isinstance(stage, str) else tuple(stage)
        self.sampling_plan = sampling_plan
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.base_seed = int(base_seed)
        self.shuffle_state = shuffle_state
        self.num_seeds = num_seeds
        self.max_batches = max_batches
        self.replay = bool(replay)
        self.runtime_options = dict(runtime_options or {})
        self.runtime_num_workers = max(1, int(runtime_num_workers))
        self.runtime_worker_id = int(runtime_worker_id)
        self.dataloader_num_workers = max(1, int(dataloader_num_workers))
        if self.runtime_num_workers > 1 and (
                max_batches is not None or group_by_timestamp
                or stage_batch_targets is not None):
            raise ValueError('runtime sharding requires uncapped iteration')
        # Single-timestamp grouped batches (temporal-LP train loader only): each
        # timestamp bucket is chunked into batch_size with its own partial tail,
        # so the true batch count is Σ_buckets ceil(count_b/batch_size) over the
        # per-stage bucket lists in diskgraph_manifest.json, NOT
        # ceil(num_seeds/batch_size). See __len__ / _grouped_global_batches.
        self.group_by_timestamp = group_by_timestamp
        self.min_steps_per_epoch = min_steps_per_epoch
        self.max_steps_per_epoch = (int(max_steps_per_epoch) if
                                    max_steps_per_epoch is not None else None)
        self.stage_batch_targets = ({
            str(stage): int(count)
            for stage, count in stage_batch_targets.items()
        } if stage_batch_targets is not None else None)
        self.stage_num_seeds = ({
            str(stage): int(count)
            for stage, count in stage_num_seeds.items()
        } if stage_num_seeds is not None else None)
        self.stage_runtime_options = {
            str(stage): dict(options)
            for stage, options in (stage_runtime_options or {}).items()
        }
        # Memoized GLOBAL grouped batch count (derived once from the manifest
        # bucket lists; grouping is deterministic ingest-side so it does not
        # vary by epoch/shuffle).
        self._grouped_global_count: int | None = None
        # Per-stage memo backing both _grouped_global_batches and __iter__'s
        # per-replay-pass stage count; avoids re-parsing the manifest JSON on
        # every pass (and in every DataLoader worker).
        self._grouped_stage_batch_counts: dict[str, int] = {}

    def _effective_max_batches(self) -> int | None:
        """The per-epoch GLOBAL batch target (shared by ``__len__`` and
        ``__iter__``). Flat: ``self.max_batches`` unchanged. Grouped: the GLOBAL
        grouped batch count (sum over ALL stages, unsharded) clamped by the
        per-epoch step caps. Distributing this single global value across workers
        with the flat formula keeps ``__len__`` == Σ-yielded and spans every
        stage (so refit val/test are never dropped)."""
        if self.stage_batch_targets is not None:
            return sum(self.stage_batch_targets.values())
        if self.group_by_timestamp:
            return self._apply_step_caps(self._grouped_global_batches())
        return self.max_batches

    def __len__(self) -> int:
        if self.stage_batch_targets is not None:
            return sum(
                sum(
                    self._stage_targets_for_worker(
                        local_num_workers=self.dataloader_num_workers,
                        local_worker_id=worker_id,
                    )[0].values())
                for worker_id in range(self.dataloader_num_workers))
        if self.group_by_timestamp:
            return self._effective_max_batches()
        if self.max_batches is not None:
            return self.max_batches
        # PL calls len() before iteration; manifest stage_counts is the source.
        if self.num_seeds is None:
            raise TypeError(
                f"DiskgraphTaskBatchIterable(stages={self.stages!r}) has "
                "no num_seeds; cannot compute __len__")
        total_workers = self.runtime_num_workers * self.dataloader_num_workers
        return sum(
            self._natural_batches(
                self._worker_num_seeds(
                    self.num_seeds,
                    total_workers,
                    worker_id * self.runtime_num_workers +
                    self.runtime_worker_id,
                ),
                self.batch_size,
                drop_last=self.drop_last,
            ) for worker_id in range(self.dataloader_num_workers))

    def _apply_step_caps(self, natural: int) -> int:
        """Clamp a natural per-epoch batch count by min/max steps per epoch
        (mirrors ``DiskgraphLightningDataModule._max_train_batches``)."""
        target = max(natural, self.min_steps_per_epoch)
        if self.max_steps_per_epoch is not None:
            target = min(target, self.max_steps_per_epoch)
        return target

    def _grouped_global_batches(self) -> int:
        """The GLOBAL (unsharded) grouped batch count: sum over ALL stages of
        ``Σ_buckets ceil(count_b / batch_size)``, read straight from each stage's
        per-timestamp bucket list in ``diskgraph_manifest.json`` (the same
        artifact the wheel's ``read_bucket_index`` consumes). Each timestamp
        bucket is chunked into ``batch_size`` with its own partial tail, so this
        is exactly the batch count the grouped iterator yields for the stage; the
        flat per-worker formula then distributes it. No sampler/reader is opened.
        Grouping is deterministic ingest-side, so the count is epoch-independent
        and memoized once."""
        if self._grouped_global_count is not None:
            return self._grouped_global_count
        total = sum(
            self._grouped_stage_batches(stage) for stage in self.stages)
        self._grouped_global_count = total
        return total

    def _grouped_stage_batches(self, stage: str) -> int:
        """Ceil-batch count for one stage: ``Σ_buckets ceil(count_b /
        batch_size)``, memoized per stage (see ``_grouped_global_batches``)."""
        cached = self._grouped_stage_batch_counts.get(stage)
        if cached is not None:
            return cached
        count = sum(
            -(-rows // self.batch_size)  # ceil(rows/batch_size)
            for _, rows in _stage_bucket_counts(self.manifest_path, stage))
        self._grouped_stage_batch_counts[stage] = count
        return count

    @staticmethod
    def _worker_num_seeds(
        num_seeds: int,
        num_workers: int,
        worker_id: int,
    ) -> int:
        if num_seeds <= 0 or worker_id >= num_seeds:
            return 0
        return (num_seeds - 1 - worker_id) // num_workers + 1

    @staticmethod
    def _natural_batches(
        num_seeds: int,
        batch_size: int,
        *,
        drop_last: bool,
    ) -> int:
        if drop_last:
            return num_seeds // batch_size
        return (num_seeds + batch_size - 1) // batch_size

    def _stage_targets_for_worker(
        self,
        *,
        local_num_workers: int,
        local_worker_id: int,
    ) -> tuple[dict[str, int], dict[str, int]]:
        assert self.stage_batch_targets is not None
        local_targets: dict[str, int] = {}
        stage_worker_counts: dict[str, int] = {}
        for stage, target in self.stage_batch_targets.items():
            num_stage_seeds = ((self.stage_num_seeds or {}).get(stage)
                               if self.stage_num_seeds is not None else None)
            natural_batches = (self._natural_batches(
                int(num_stage_seeds),
                self.batch_size,
                drop_last=self.drop_last,
            ) if num_stage_seeds is not None else local_num_workers)
            active_workers = min(
                local_num_workers,
                max(1, target),
                max(1, natural_batches),
            )
            stage_worker_counts[stage] = active_workers
            if local_worker_id >= active_workers:
                local_targets[stage] = 0
                continue
            quotient, remainder = divmod(target, active_workers)
            local_targets[stage] = quotient + int(local_worker_id < remainder)
        return local_targets, stage_worker_counts

    def _runtime_index(self) -> Any:
        """One opened index per worker process, reused for every epoch."""
        cached = self.__dict__.get('_runtime_index_handle')
        if cached is None:
            from diskgraph import RuntimeIndex
            cached = RuntimeIndex(
                self.manifest_path,
                rnd_seed=self.base_seed,
                num_threads=self.runtime_options.get('num_threads'),
            )
            self.__dict__['_runtime_index_handle'] = cached
        return cached

    def __getstate__(self) -> dict:
        # Never pickle the handle; workers rebuild their own.
        state = dict(self.__dict__)
        state.pop('_runtime_index_handle', None)
        return state

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        worker = get_worker_info()
        local_num_workers = worker.num_workers if worker is not None else 1
        local_worker_id = worker.id if worker is not None else 0
        # Process-major: W-independent per-process seeds.
        num_workers = self.runtime_num_workers * local_num_workers
        worker_id = (local_worker_id * self.runtime_num_workers +
                     self.runtime_worker_id)
        base_epoch = int(self.shuffle_state.stream_id)

        index = self._runtime_index()
        # num_threads belongs to the opened engine, not to iter().
        iter_options = {
            k: v
            for k, v in self.runtime_options.items() if k != 'num_threads'
        }

        def _open(
            stage: str,
            epoch_offset: int,
            stage_num_workers: int | None = None,
            stage_worker_id: int | None = None,
        ):
            stage_options = self.stage_runtime_options.get(stage, {})
            iterator, num_samples = index.iter(
                stage=stage,
                sampling_plan=self.sampling_plan,
                num_workers=(stage_num_workers or num_workers),
                worker_id=(worker_id
                           if stage_worker_id is None else stage_worker_id),
                shuffle=self.shuffle,
                epoch=base_epoch + epoch_offset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                group_by_timestamp=self.group_by_timestamp,
                **{**iter_options, **stage_options},
            )
            return iterator, int(num_samples)

        if self.stage_batch_targets is not None:
            local_targets, stage_worker_counts = (
                self._stage_targets_for_worker(
                    local_num_workers=local_num_workers,
                    local_worker_id=local_worker_id,
                ))

            schedule = _interleave_stage_batches(local_targets)
            iterators: dict[str, Iterator[Mapping[str, Any]]] = {}
            replays = {stage: 0 for stage in self.stages}
            yielded = {stage: 0 for stage in self.stages}
            for stage in schedule:
                while True:
                    iterator = iterators.get(stage)
                    if iterator is None:
                        iterator, _ = _open(
                            stage,
                            replays[stage],
                            stage_worker_counts[stage],
                        )
                        iterator = iter(iterator)
                        iterators[stage] = iterator
                    try:
                        batch = next(iterator)
                        yielded[stage] += 1
                        yield batch
                        break
                    except StopIteration:
                        if not self.replay:
                            raise RuntimeError(
                                f"stage {stage!r} exhausted before its "
                                "allocated batch target")
                        replays[stage] += 1
                        replacement, _ = _open(
                            stage,
                            replays[stage],
                            stage_worker_counts[stage],
                        )
                        replacement = iter(replacement)
                        try:
                            batch = next(replacement)
                        except StopIteration as exc:
                            reason = ("for its initial worker shard"
                                      if yielded[stage] == 0 else
                                      "while replaying its allocated target")
                            raise RuntimeError(
                                f"stage {stage!r} produced no batches {reason}"
                            ) from exc
                        iterators[stage] = replacement
                        yielded[stage] += 1
                        yield batch
                        break
            return

        if self.group_by_timestamp:
            remaining = self._effective_max_batches()
            offset = 0
            replay = 0
            while remaining > 0:
                available = 0
                for stage in self.stages:
                    count = self._grouped_stage_batches(stage)
                    take = min(count, remaining)
                    stage_worker_id = (worker_id - offset) % num_workers
                    quotient, remainder = divmod(take, num_workers)
                    quota = quotient + int(stage_worker_id < remainder)
                    if quota:
                        iterator, _ = _open(
                            stage, replay, stage_worker_id=stage_worker_id)
                        yield from itertools.islice(iterator, quota)
                    offset += count
                    remaining -= take
                    available += count
                    if remaining == 0:
                        return
                if not available or not self.replay:
                    return
                replay += 1
            return

        if self.max_batches is None:
            # val/test/predict: yield natural until exhaustion.
            for stage in self.stages:
                iterator, _ = _open(stage, 0)
                yield from iterator
            return

        effective_max_batches = self._effective_max_batches()
        per_worker_target = (effective_max_batches + num_workers -
                             1) // num_workers
        emitted = 0
        replay = 0
        while emitted < per_worker_target:
            had_any = False
            for stage in self.stages:
                iterator, _ = _open(stage, replay)
                for batch in iterator:
                    yield batch
                    emitted += 1
                    had_any = True
                    if emitted >= per_worker_target:
                        return
            if not had_any or not self.replay:
                # Defensive: empty shard (e.g. fewer seeds than workers).
                return
            replay += 1


def _exit_when_parent_exits() -> None:
    """Bypass queue finalizers once the inherited parent sentinel closes."""
    parent = mp.parent_process()
    if parent is None:
        return
    parent.join()
    os._exit(0)


def _init_diskgraph_worker(_worker_id: int) -> None:
    """Start logical-parent death detection in a DataLoader worker."""
    threading.Thread(
        target=_exit_when_parent_exits,
        daemon=True,
    ).start()


class _SetShuffleEpochCallback(pl.Callback):
    """PL callback propagating ``trainer.current_epoch`` to the wheel shuffle
    stream each epoch. Needed because an ``IterableDataset`` never gets PL's
    ``sampler.set_epoch``, so the row order would otherwise repeat.

    Defined at module scope (not as a ``<locals>`` class) so it is picklable:
    Python 3.14's default ``forkserver`` start method spawns dataloader workers
    by pickling trainer/callback state, which fails for local classes.
    """
    def on_train_epoch_start(self, trainer: pl.Trainer,
                             pl_module: pl.LightningModule) -> None:
        dm = trainer.datamodule
        if isinstance(dm, DiskgraphLightningDataModule):
            dm.set_shuffle_stream_id(int(trainer.current_epoch))


class _ShutdownWorkerPoolsCallback(pl.Callback):
    """Close DG DataLoader pools at epoch/failure boundaries."""
    @staticmethod
    def _shutdown(trainer: pl.Trainer, phase: str | None = None) -> None:
        dm = trainer.datamodule
        if isinstance(dm, DiskgraphLightningDataModule):
            if (phase is not None
                    and dm.is_dataloader_worker_shutdown_deferred(phase)):
                return
            dm.shutdown_dataloader_workers(phase)

    def on_train_batch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        # Covers Trainer(limit_train_batches=...) where the Lightning limit is
        # shorter than len(loader). This hook runs before the validation loop
        # constructs its iterator, unlike on_validation_start.
        if trainer.is_last_batch:
            self._shutdown(trainer, 'train')

    def on_validation_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:
        self._shutdown(trainer, 'val')

    def on_test_end(self, trainer: pl.Trainer,
                    pl_module: pl.LightningModule) -> None:
        self._shutdown(trainer, 'test')

    def on_predict_end(self, trainer: pl.Trainer,
                       pl_module: pl.LightningModule) -> None:
        self._shutdown(trainer, 'predict')

    def on_train_end(self, trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        self._shutdown(trainer)

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                     exception: BaseException) -> None:
        try:
            self._shutdown(trainer)
        except BaseException:
            # Do not replace the job's original exception with a cleanup error.
            try:
                logger.exception(
                    "Failed to close DataLoader workers after error")
            except BaseException:
                pass


class DiskgraphLightningDataModule(pl.LightningDataModule):
    """Lightning datamodule over native DiskGraph task batch iterators."""
    def __init__(
        self,
        manifest: DiskgraphTaskManifest | str,
        mapper_bank: MapperBank,
        *,
        sampling_plan: Any,
        train_batch_size: int,
        eval_batch_size: int,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        base_seed: int = 0,
        max_steps_per_epoch: int | None = None,
        min_steps_per_epoch: int = 1,
        max_val_steps: int | None = None,
        max_test_steps: int | None = None,
        train_stages: tuple[str, ...] = ("train", ),
        runtime_options: Mapping[str, Any] | None = None,
        runtime_num_workers: int = 1,
        runtime_worker_id: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        task_type: str | None = None,
        neg_rhs_node_type: str | None = None,
        neg_amount_m: int | None = None,
        rhs_node_type: str | None = None,
        unix_start_offset: int | None = None,
        unix_end_offset: int | None = None,
        stage_time_dict: Mapping[Any, "tuple[int, int]"] | None = None,
        resolve_entity_node_type: str | None = None,
        resolve_target_node_type: str | None = None,
        target_encoder: Any | None = None,
        bidirectional_output: bool = False,
        use_lhs_node_loader: bool = False,
        weighted_seed_sampling: Mapping[str, float | bool] | None = None,
    ) -> None:
        super().__init__()
        self.task_manifest = (DiskgraphTaskManifest.load(manifest)
                              if isinstance(manifest, str) else manifest)
        self.mapper_bank = mapper_bank
        self.sampling_plan = sampling_plan
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.shuffle_train = bool(shuffle_train)
        self.drop_last_train = bool(drop_last_train)
        self.base_seed = int(base_seed)
        self.max_steps_per_epoch = (int(max_steps_per_epoch) if
                                    max_steps_per_epoch is not None else None)
        self.min_steps_per_epoch = int(min_steps_per_epoch)
        if self.max_steps_per_epoch is not None and (
                self.min_steps_per_epoch > self.max_steps_per_epoch):
            self.min_steps_per_epoch = self.max_steps_per_epoch
        self.max_val_steps = int(
            max_val_steps) if max_val_steps is not None else None
        self.max_test_steps = int(
            max_test_steps) if max_test_steps is not None else None
        self.train_stages = tuple(train_stages or ("train", ))
        self.runtime_options = dict(runtime_options or {})
        self.runtime_num_workers = max(1, int(runtime_num_workers))
        self.runtime_worker_id = int(runtime_worker_id)
        if not 0 <= self.runtime_worker_id < self.runtime_num_workers:
            raise ValueError(
                'runtime_worker_id must be in [0, runtime_num_workers)')
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = (int(prefetch_factor)
                                if prefetch_factor is not None else None)
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            # Fail here rather than later, when the DataLoader is built.
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}")
        # Two-tower LP engine negatives (train loader only): M negatives per
        # anchor over the neg_rhs_node_type id-space.
        self.neg_rhs_node_type = neg_rhs_node_type
        self.neg_amount_m = (int(neg_amount_m)
                             if neg_amount_m is not None else None)
        # FK-free LP resolution: iterator seeds off entity_node_id and emits
        # ragged lp_pos_* from target_node_ids. None keeps legacy behavior.
        self.resolve_entity_node_type = resolve_entity_node_type
        self.resolve_target_node_type = resolve_target_node_type
        self.bidirectional_output = bool(bidirectional_output)
        self._train_table_name = self.task_manifest.train_table_name
        self._seed_node_type = self.task_manifest.seed_node_type
        self._feature_schema = self.task_manifest.feature_schema
        self._task_type = task_type or self.task_manifest.task_type
        # Whether the link-pred module seeds off LHS nodes (shallow_rhs / id /
        # hybrid). Mirrors ``module_kwargs.module.use_lhs_node_loader`` in legacy
        # ``util.py`` and is one of the three conditions that gate single-
        # timestamp grouping. two_tower (no LHS node loader) -> never grouped.
        self._use_lhs_node_loader = bool(use_lhs_node_loader)
        self._target_encoder = target_encoder
        self._weighted_seed_sampling = (dict(weighted_seed_sampling)
                                        if weighted_seed_sampling else None)
        if self._weighted_seed_sampling is not None:
            use_sample_weights = bool(
                self._weighted_seed_sampling.get("use_sample_weights", False))
            has_class_weights = ("false_weight" in self._weighted_seed_sampling
                                 or "true_weight"
                                 in self._weighted_seed_sampling)
            for stage in self.train_stages:
                if use_sample_weights:
                    _sample_weight_index_for_stage(
                        self.task_manifest.runtime_manifest_path, stage)
                elif has_class_weights:
                    _binary_label_index_for_stage(
                        self.task_manifest.runtime_manifest_path, stage)
        self._shuffle_state = _ShuffleStreamState()

        # Canonical batch_size source: Trainer.explain mutates this for XAI
        # batches; eval/test/predict loaders read it so the mutation flows through.
        self.eval_loader_kwargs: dict[str, Any] = {
            'batch_size': self.eval_batch_size,
        }

        # (feature_store, graph_store) for the explainer; populated only during
        # DiskgraphNestedTrainer.explain.
        self._data: tuple[Any, Any] | None = None

        # Stage-stable cache for ``stage_inputs``: stage -> (input_type,
        # input_nodes, input_time, input_id).
        self._stage_inputs_cache: dict[str, tuple[Any, Any, Any, Any]] = {}
        self._explain_capture_stage: str | None = None
        self._explain_capture_input_nodes: list[Any] = []
        self._explain_capture_stage_node_ids: list[Any] = []

        # Per-namespace PK + time column names, derived from
        # ``index_metadata.json``. Threaded into the runtime adapter so
        # it can decorate the HeteroData with ``.pkey`` / ``.time``.
        self._pk_col_dict: dict[str, str] | None = None
        self._time_col_dict: dict[str, str] | None = None

        # Temporally-valid RHS provider for ShallowRHS / Hybrid temporal LP
        # (via lp_train_table); built lazily. See _build_lp_train_table.
        self._rhs_node_type = rhs_node_type
        self._unix_start_offset = unix_start_offset
        self._unix_end_offset = unix_end_offset
        # stage_time_dict: TrainingStage -> (min, max) seed time; built from the
        # bundle's stage_seed_times unless an explicit kwarg overrides.
        if stage_time_dict:
            self._stage_time_dict = dict(stage_time_dict)
        else:
            self._stage_time_dict = self._build_stage_time_dict()
        self._lp_train_table: DiskgraphValidRHS | None = None
        self._lp_train_table_built = False
        self._worker_pool_registry = _WorkerPoolRegistry()
        self._worker_pool_shutdown_deferrals: dict[str, int] = {}

    def _build_stage_time_dict(self) -> dict[Any, "tuple[int, int]"]:
        """Map the bundle's ``stage_seed_times`` (stage-name -> ``(min, max)``)
        onto ``TrainingStage`` keys. ``{}`` when the bundle has none."""
        raw = self.task_manifest.stage_seed_times
        if not raw:
            return {}
        from kumoapi.train import TrainingStage
        name_to_stage = {
            "train": TrainingStage.TRAIN,
            "val": TrainingStage.VAL,
            "test": TrainingStage.TEST,
            # Predict-only bundles emit a single predict seed-time range.
            "predict": TrainingStage.PRED,
        }
        out: dict[Any, tuple[int, int]] = {}
        for name, rng in raw.items():
            stage = name_to_stage.get(str(name))
            if stage is None:
                continue
            out[stage] = (int(rng[0]), int(rng[1]))
        return out

    def set_shuffle_stream_id(self, shuffle_stream_id: int) -> None:
        self._shuffle_state.stream_id = int(shuffle_stream_id)

    @staticmethod
    def shuffle_stream_callback() -> pl.Callback:
        """Return a picklable PL callback that advances the wheel shuffle stream
        each epoch (see :class:`_SetShuffleEpochCallback`)."""
        return _SetShuffleEpochCallback()

    @staticmethod
    def worker_pool_lifecycle_callback() -> pl.Callback:
        return _ShutdownWorkerPoolsCallback()

    def setup(self, stage: str | None = None) -> None:
        """Install phase-boundary cleanup for managed and direct PL trainers."""
        trainer = self.trainer
        if trainer is not None and not any(
                isinstance(callback, _ShutdownWorkerPoolsCallback)
                for callback in trainer.callbacks):
            trainer.callbacks.append(self.worker_pool_lifecycle_callback())
        super().setup(stage)

    @contextmanager
    def defer_dataloader_worker_shutdown(
        self,
        phase: str,
    ) -> Iterator[None]:
        """Defer a strict phase barrier until an output transaction closes."""
        self._worker_pool_shutdown_deferrals[phase] = (
            self._worker_pool_shutdown_deferrals.get(phase, 0) + 1)
        try:
            yield
        finally:
            remaining = self._worker_pool_shutdown_deferrals[phase] - 1
            if remaining:
                self._worker_pool_shutdown_deferrals[phase] = remaining
            else:
                del self._worker_pool_shutdown_deferrals[phase]

    def is_dataloader_worker_shutdown_deferred(self, phase: str) -> bool:
        """Return whether a writer transaction owns this callback barrier."""
        return bool(self._worker_pool_shutdown_deferrals.get(phase, 0))

    def shutdown_dataloader_workers(self, phase: str | None = None) -> None:
        """Synchronously close every matching worker pool."""
        self._worker_pool_registry.shutdown(phase)

    @property
    def active_dataloader_worker_pids(self) -> tuple[int, ...]:
        """Worker PIDs owned by active phase-local DataLoader iterators."""
        return self._worker_pool_registry.worker_pids

    def teardown(self, stage: str | None = None) -> None:
        self.shutdown_dataloader_workers()
        super().teardown(stage)

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_stages, is_train=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val", max_eval_steps=self.max_val_steps)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test", max_eval_steps=self.max_test_steps)

    def predict_dataloader(self) -> DataLoader:
        return self._make_dataloader("predict")

    def transfer_batch_to_device(self, batch: Any, device: Any,
                                 dataloader_idx: int) -> Any:
        r"""Move a mini-batch to :obj:`device`, overlapping the copy with
        compute when the batch came from a pinned loader.

        Lightning only requests a non-blocking copy for real tensors, so a
        :class:`HeteroData` batch always transferred synchronously. Only safe
        out of pinned memory, hence the guard.
        """
        if not (self.pin_memory and device.type == 'cuda'):
            return super().transfer_batch_to_device(batch, device,
                                                    dataloader_idx)
        to = getattr(batch, 'to', None)
        if to is None:
            return super().transfer_batch_to_device(batch, device,
                                                    dataloader_idx)
        try:
            return to(device, non_blocking=True)
        except TypeError as error:
            if 'non_blocking' not in str(error):
                # Only a signature mismatch justifies falling back.
                raise
            return super().transfer_batch_to_device(batch, device,
                                                    dataloader_idx)

    def begin_explain_selection_capture(self, stage: str) -> None:
        stage = str(stage)
        self._explain_capture_stage = stage
        self._explain_capture_input_nodes = []
        self._explain_capture_stage_node_ids = []
        self._stage_inputs_cache.pop(stage, None)

    def clear_explain_selection_capture(self) -> None:
        self._explain_capture_stage = None
        self._explain_capture_input_nodes = []
        self._explain_capture_stage_node_ids = []

    @staticmethod
    def _array_to_long_tensor(values: Any) -> Any:
        import torch
        if hasattr(values, "to_pylist"):
            values = values.to_pylist()
        return torch.as_tensor(values, dtype=torch.long)

    def _capture_explain_selection_batch(
        self,
        batch: Mapping[str, Any],
    ) -> None:
        if self._explain_capture_stage is None:
            return
        task = batch.get("task") or {}
        seed_node_ids = task.get("seed_node_ids")
        stage_node_ids = task.get("tt_stage_node_ids")
        if seed_node_ids is None or stage_node_ids is None:
            raise RuntimeError(
                "TT-NG explain selection capture requires task.seed_node_ids "
                "and task.tt_stage_node_ids")
        input_nodes = self._array_to_long_tensor(seed_node_ids)
        input_id = self._array_to_long_tensor(stage_node_ids)
        if input_nodes.numel() != input_id.numel():
            raise RuntimeError("TT-NG explain selection capture saw mismatched "
                               "seed_node_ids and tt_stage_node_ids lengths")
        self._explain_capture_input_nodes.append(input_nodes.cpu())
        self._explain_capture_stage_node_ids.append(input_id.cpu())

    def _finish_explain_selection_capture(
        self,
        stage: str,
    ) -> tuple[str, Any, Any, Any]:
        if self._explain_capture_stage != stage:
            raise RuntimeError(
                "DLDM.stage_inputs requires an active explain selection "
                f"capture for stage={stage!r}")
        import torch
        input_nodes = (torch.cat(self._explain_capture_input_nodes, dim=0)
                       if self._explain_capture_input_nodes else torch.zeros(
                           0, dtype=torch.long))
        input_id = (torch.cat(self._explain_capture_stage_node_ids, dim=0)
                    if self._explain_capture_stage_node_ids else torch.zeros(
                        0, dtype=torch.long))
        result = (self._seed_node_type, input_nodes, None, input_id)
        self._stage_inputs_cache[stage] = result
        self.clear_explain_selection_capture()
        return result

    # RHS dataloaders for the two-tower MIPS structure: seed [0, num_rhs) of the
    # target type at the stage's FIXED query time via the engine's NodeRange mode.
    #
    # MIPS-ordering contract: init_mips_structure cats batches in order assuming
    # row i == node-id i, so seeds must be [0, num_rhs) with no gaps/reorder/drop
    # (num_workers=0, no shuffle/drop_last/replay; enforced by
    # _assert_rhs_loader_ordering).

    def rhs_val_dataloader(self) -> DataLoader:
        """RHS MIPS dataloader for VAL — fixed seed time = ``val_time``."""
        return self._make_rhs_dataloader(self.val_time, "val")

    def rhs_test_dataloader(self) -> DataLoader:
        """RHS MIPS dataloader for TEST — fixed seed time = ``test_time``."""
        return self._make_rhs_dataloader(self.test_time, "test")

    def rhs_predict_dataloader(self) -> DataLoader:
        """RHS MIPS dataloader for PREDICT — fixed seed time = ``predict_time``
        (the prediction anchor; NodeRange mode requires a non-None time)."""
        return self._make_rhs_dataloader(self.predict_time, "predict")

    def _pseudo_time_options(self, stage: str) -> dict[str, Any]:
        from kumo_core.training.splits import pseudo_seed_time, uses_pseudo_time
        if uses_pseudo_time(self._task_type) and stage in ('pred', 'predict'):
            return {'pseudo_seed_time': None, 'disable_pseudo_time': True}
        seed_time = pseudo_seed_time(self._task_type, stage)
        return {} if seed_time is None else {
            'pseudo_seed_time': seed_time, 'disable_pseudo_time': False
        }

    @property
    def predict_time(self) -> "Any | None":
        """Per-stage seed time for PREDICT eval (``stage_seed_times['predict']``,
        the prediction anchor); falls back to ``test_time``."""
        from kumoapi.train import TrainingStage
        pred = self._stage_seed_time(TrainingStage.PRED)
        return pred if pred is not None else self.test_time

    def _seed_time_to_index_unit(self, seed_time: "Any | None") -> int | None:
        """Collapse a stage seed-time tensor to the single int the engine's
        NodeRange ``rhs_seed_time`` expects, converting Kumo's ns-canonical
        ``val_time`` / ``test_time`` to the *engine's* timestamp unit.

        This is the ONE ns -> engine-unit conversion on this path, placed exactly
        at the engine boundary (the value is threaded into ``runtime_options``
        and consumed by the wheel's NodeRange query, NOT by the pure-arithmetic
        ``RHSTime`` validity path — that one stays all-ns). The divisor is driven
        by the manifest's configured unit (``index_metadata.json::timestamp_unit``
        via ``index_np_time_unit``), never hardcoded; ``None`` (non-temporal)
        passes through.
        """
        if seed_time is None:
            return None
        ns = int(seed_time.reshape(-1)[0].item())
        return ns // _NP_UNIT_TO_NS_DIVISOR[self.index_np_time_unit]

    @staticmethod
    def _assert_rhs_loader_ordering(loader: DataLoader) -> None:
        """Guard the MIPS-ordering contract (row i == node-id i): single worker,
        no shuffle, no dropped tail."""
        if loader.num_workers != 0:
            raise AssertionError(
                "RHS MIPS dataloader requires num_workers=0 (in-order, single "
                f"shard) for the position==node-id contract; got "
                f"{loader.num_workers}")
        ds = loader.dataset
        if getattr(ds, "shuffle", False):
            raise AssertionError(
                "RHS MIPS dataloader must not shuffle (position==node-id)")
        if getattr(ds, "drop_last", False):
            raise AssertionError(
                "RHS MIPS dataloader must not drop the last partial batch "
                "(it would truncate the RHS index)")
        if getattr(ds, "replay", False):
            raise AssertionError(
                "RHS MIPS dataloader must not use the train replay path")

    def _make_rhs_dataloader(self, seed_time: "Any | None",
                             stage: str) -> DataLoader:
        """Build a NodeRange RHS dataloader seeding ``[0, num_rhs)`` of the
        target node type at a fixed ``seed_time``. See the MIPS-ordering
        contract block above."""
        target_node_type = self.task_manifest.target_node_type
        num_rhs = _num_nodes_for_type(self.task_manifest.graph_runtime_metadata,
                                      target_node_type)
        if num_rhs <= 0:
            raise RuntimeError(
                "rhs_*_dataloader requires a positive RHS node count for "
                f"target_node_type={target_node_type!r}; got {num_rhs} "
                "(check graph_runtime_metadata.graph_index_path / "
                "index_metadata.json max_id).")
        batch_size = int(self.eval_loader_kwargs['batch_size'])
        rhs_seed_time = self._seed_time_to_index_unit(seed_time)
        # Static LP has no temporal seed-time range, but NodeRange still
        # requires a fixed seed-time int — fall back to the 0 sentinel.
        if rhs_seed_time is None:
            rhs_seed_time = 0
        # NodeRange ignores stage seeding but still opens the stage table, so
        # pass any present stage (a hardcoded ``test`` would break predict-only).
        _present = list(self.task_manifest.stage_counts or {})
        rhs_open_stage = ('predict' if 'predict' in _present else
                          'test' if 'test' in _present else
                          (_present[0] if _present else 'test'))
        runtime_options = {
            **self.runtime_options,
            'rhs_seed_node_type': target_node_type,
            'rhs_seed_time': rhs_seed_time,
            **self._pseudo_time_options(stage),
        }
        iterable = DiskgraphTaskBatchIterable(
            manifest_path=self.task_manifest.runtime_manifest_path,
            stage=rhs_open_stage,
            sampling_plan=self.sampling_plan,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            base_seed=self.base_seed,
            shuffle_state=self._shuffle_state,
            num_seeds=num_rhs,
            max_batches=None,
            replay=False,
            runtime_options=runtime_options,
        )
        loader = DataLoader(
            iterable,
            batch_size=None,
            num_workers=0,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            collate_fn=lambda batch: self._rhs_to_hetero(
                batch, target_node_type),
        )
        self._assert_rhs_loader_ordering(loader)
        return loader

    def _rhs_to_hetero(self, batch: Mapping[str, Any],
                       target_node_type: str) -> Any:
        """Collate a NodeRange RHS batch, stamping ``seed_time`` onto the TARGET
        block. The shared adapter mirrors ``seed_time`` only onto the LHS seed
        type, but ``init_mips_structure`` reads it off ``batch[rhs_name]`` (the
        target type), which differs from the LHS seed type in real two-tower
        topologies (e.g. User -> Item)."""
        import pyarrow as pa
        import torch

        hetero = self._to_hetero(batch)
        seed_ts = (batch.get("task") or {}).get("seed_timestamps")
        if seed_ts is not None and target_node_type in hetero.node_types:
            block = hetero[target_node_type]
            if "seed_time" not in block:
                if isinstance(seed_ts, pa.ChunkedArray):
                    seed_ts = seed_ts.combine_chunks()
                if isinstance(seed_ts, pa.Array):
                    # Engine-unit -> ns (Kumo canonical), matching the seed_time
                    # the shared adapter stamps on the LHS seed block.
                    seed_ts = engine_int64_to_ns(seed_ts,
                                                 self.index_np_time_unit)
                    seed_ts = seed_ts.to_numpy(zero_copy_only=False)
                block.seed_time = torch.as_tensor(seed_ts, dtype=torch.long)
        return hetero

    def _make_dataloader(
        self,
        stage: str | tuple[str, ...],
        *,
        is_train: bool = False,
        max_eval_steps: int | None = None,
    ) -> DataLoader:
        stages = (stage, ) if isinstance(stage, str) else tuple(stage)
        loader_num_workers = self.num_workers
        capture_stage = self._explain_capture_stage
        capture_matches_stage = capture_stage in stages or (
            capture_stage == "pred" and "predict" in stages)
        if capture_stage is not None and capture_matches_stage:
            loader_num_workers = 0
        num_seeds = self._num_seeds_for_stages(stages)
        # Single-timestamp grouped batches: TRAIN seed loader only, and ONLY
        # under EXACTLY legacy's ``TimestampSampler`` condition (util.py:404-421):
        #   1. task is link-pred (static OR temporal),
        #   2. the module seeds off LHS nodes (shallow_rhs / id / hybrid; NOT
        #      two_tower), and
        #   3. the RHS (target) table is temporal (has a time/end_time column).
        # Grouping exists to make per-batch RHS temporal-validity correct (all
        # anchors in a batch share one seed time), so it is a no-op benefit on an
        # atemporal RHS and instead floors the metric by fragmenting the epoch
        # into tiny single-timestamp buckets. Legacy picks ``TimestampSampler``
        # iff ``rhs_time is not None or rhs_end_time is not None`` AND
        # ``module.use_lhs_node_loader``, NOT merely on temporal-LP task type.
        # Val/test/predict are single-timestamp by contract (E1) so the flat
        # path already yields single-ts batches; RHS/NodeRange loaders never
        # group (they build their own loaders directly, not via this method).
        group_by_timestamp = (is_train and self._is_link_pred
                              and self._use_lhs_node_loader
                              and self._rhs_is_temporal)
        if is_train:
            batch_size = self.train_batch_size
            max_batches = self._max_train_batches(num_seeds)
        else:
            batch_size = int(self.eval_loader_kwargs['batch_size'])
            max_batches = self._max_eval_batches(num_seeds, batch_size,
                                                 max_eval_steps)
        stage_batch_targets: dict[str, int] | None = None
        stage_num_seeds: dict[str, int] | None = None
        stage_runtime_options: dict[str, dict[str, Any]] = {}
        if is_train and len(stages) > 1:
            counts = self.task_manifest.stage_counts
            stage_num_seeds = {
                stage: int(counts[stage])
                for stage in stages if counts.get(stage) is not None
            }
            if len(stage_num_seeds) != len(stages):
                raise ValueError(
                    "multi-stage training requires a row count for every stage")
            if group_by_timestamp:
                natural_by_stage = {
                    stage:
                    sum(
                        math.ceil(count / batch_size)
                        for _timestamp, count in _stage_bucket_counts(
                            self.task_manifest.runtime_manifest_path, stage))
                    for stage in stages
                }
            else:
                natural_by_stage = {
                    stage:
                    self._natural_batches(
                        count,
                        batch_size,
                        drop_last=self.drop_last_train,
                    )
                    for stage, count in stage_num_seeds.items()
                }
            empty_stages = [
                stage for stage, count in natural_by_stage.items() if count <= 0
            ]
            if empty_stages:
                raise ValueError(
                    "multi-stage training requires every stage to yield at "
                    f"least one batch; empty stages: {empty_stages}")
            max_batches = max(sum(natural_by_stage.values()),
                              self.min_steps_per_epoch)
            if self.max_steps_per_epoch is not None:
                max_batches = min(max_batches, self.max_steps_per_epoch)
            stage_batch_targets = _allocate_stage_batches(
                natural_by_stage,
                max_batches,
                require_each_stage=True,
            )
        # Engine negatives are TRAIN-ONLY; eval/test/predict get no negatives.
        runtime_options = self.runtime_options
        if (is_train and self.neg_rhs_node_type is not None
                and self.neg_amount_m is not None):
            runtime_options = {
                **self.runtime_options,
                'neg_rhs_node_type': self.neg_rhs_node_type,
                'neg_amount_m': self.neg_amount_m,
            }
        # FK-free LP resolution: thread entity/target node types so the engine
        # seeds off entity_node_id and emits lp_pos_* labels. PREDICT is
        # entity-only, so pass target_node_type for labelled stages only.
        if self.resolve_entity_node_type is not None:
            _stage_seq = ([stages] if isinstance(stages, str) else list(stages))
            _is_predict = 'predict' in _stage_seq
            runtime_options = {
                **runtime_options,
                'entity_node_type':
                self.resolve_entity_node_type,
            }
            if not _is_predict:
                runtime_options['target_node_type'] = (
                    self.resolve_target_node_type)
        if is_train and self._weighted_seed_sampling is not None:
            if self.drop_last_train:
                raise ValueError(
                    "WeightMode.SAMPLE on TT-NG is incompatible with "
                    "drop_last_train=True.")
            if group_by_timestamp:
                raise ValueError(
                    "WeightMode.SAMPLE on TT-NG is incompatible with "
                    "timestamp-grouped training.")
            if len(stages) == 1:
                weighted_num_samples = self._weighted_seed_num_samples(
                    stage=stages[0],
                    config=self._weighted_seed_sampling,
                )
                runtime_options = {
                    **runtime_options,
                    "weighted_seed_sampling": {
                        **self._weighted_seed_sampling,
                        "num_samples":
                        weighted_num_samples,
                    },
                }
                num_seeds = weighted_num_samples
                max_batches = self._max_train_batches(num_seeds)
            else:
                stage_masses = {
                    stage:
                    self._weighted_seed_mass(
                        stage=stage,
                        config=self._weighted_seed_sampling,
                    )
                    for stage in stages
                }
                weighted_num_samples = self.train_batch_size * math.ceil(
                    sum(stage_masses.values()) / self.train_batch_size)
                max_batches = self._max_train_batches(weighted_num_samples)
                assert max_batches is not None
                stage_batch_targets = _allocate_stage_batches(
                    stage_masses,
                    max_batches,
                    require_each_stage=True,
                )
                stage_num_seeds = {
                    stage: batches * self.train_batch_size
                    for stage, batches in stage_batch_targets.items()
                }
                stage_runtime_options = {
                    stage: {
                        "weighted_seed_sampling": {
                            **self._weighted_seed_sampling,
                            "num_samples":
                            stage_num_seeds[stage],
                        },
                    }
                    for stage in stages
                }
                num_seeds = sum(stage_num_seeds.values())
        for name in stages:
            if pseudo_time := self._pseudo_time_options(name):
                stage_runtime_options[name] = {
                    **stage_runtime_options.get(name, {}),
                    **pseudo_time,
                }
        iterable = DiskgraphTaskBatchIterable(
            manifest_path=self.task_manifest.runtime_manifest_path,
            stage=stages,
            sampling_plan=self.sampling_plan,
            batch_size=batch_size,
            shuffle=(is_train and self.shuffle_train),
            drop_last=(is_train and self.drop_last_train),
            base_seed=self.base_seed,
            shuffle_state=self._shuffle_state,
            num_seeds=num_seeds,
            max_batches=max_batches,
            replay=is_train,
            runtime_options=runtime_options,
            runtime_num_workers=self.runtime_num_workers,
            runtime_worker_id=self.runtime_worker_id,
            dataloader_num_workers=loader_num_workers,
            group_by_timestamp=group_by_timestamp,
            min_steps_per_epoch=self.min_steps_per_epoch,
            max_steps_per_epoch=(self.max_steps_per_epoch
                                 if self.max_steps_per_epoch is not None else
                                 None),
            stage_batch_targets=stage_batch_targets,
            stage_num_seeds=stage_num_seeds,
            stage_runtime_options=stage_runtime_options,
        )
        loader_kwargs: dict[str, Any] = dict(
            batch_size=None,
            num_workers=loader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers
                                if loader_num_workers > 0 else False),
            collate_fn=(self._to_hetero_train if is_train else self._to_hetero),
        )
        # Pin fork: py3.14's default (forkserver) pickles the dataset and crashes
        # re-initializing CUDA in a worker; fork inherits via COW and never pickles.
        if loader_num_workers > 0:
            # multiprocessing_context is only valid with worker processes
            # (PyTorch errors if set when num_workers == 0).
            loader_kwargs['multiprocessing_context'] = 'fork'
            # Parent death can otherwise strand the worker's queue finalizer.
            loader_kwargs['worker_init_fn'] = _init_diskgraph_worker
            # Also worker-only; PyTorch rejects it at num_workers == 0.
            if self.prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = self.prefetch_factor
            return _WorkerPoolDataLoader(
                iterable,
                phase='train' if is_train else stages[0],
                registry=self._worker_pool_registry,
                **loader_kwargs,
            )
        return DataLoader(iterable, **loader_kwargs)

    @property
    def _is_link_pred(self) -> bool:
        """Whether this run is a link-prediction task (static OR temporal).
        This mirrors legacy ``task_type.is_link_pred`` -- one of the three
        conditions that gate single-timestamp grouping (``TimestampSampler``).
        """
        from kumoapi.task import TaskType
        try:
            return TaskType(self._task_type).is_link_pred
        except ValueError:
            return False

    @property
    def _rhs_is_temporal(self) -> bool:
        """Whether the RHS (target) node table is temporal, i.e. has a
        time (create-time) or end-time column. This is the exact signal legacy
        uses to gate single-timestamp grouping (``TimestampSampler``): grouping
        is only meaningful when RHS validity varies with the anchor's seed time.
        Returns ``False`` (-> flat path) when the RHS type is unknown or the
        index metadata is unavailable (e.g. unit tests with a fake manifest).

        SMELL-2 (RHS-temporal source of truth): this LOAD gate derives the
        predicate from ``index_metadata.json`` (the index's view of the RHS
        table). The INGEST gate (DG ``tt_executor.rs`` -> ``bucket_index``) uses
        the SAME logical predicate computed gen-side in
        ``diskgraph_train_table.py`` (``rhs_table.has_time_col() or
        has_end_time_col()``) and carried on the plan as
        ``lp_rhs_is_temporal``.
        Two derivations of one predicate (load from the index, ingest from the
        graph schema) — consistent by construction. The only divergent direction
        (load wants to group but ingest emitted no ``bucket_index``) FAILS LOUD:
        the wheel's ``read_bucket_index`` raises rather than silently
        mis-batching. See TS_GROUPING_NOTES.md.
        """
        rhs = self._rhs_node_type
        if not rhs:
            return False
        if self.time_col_dict.get(rhs):
            return True
        return self._end_time_col_for(rhs) is not None

    def _num_seeds_for_stages(self, stages: tuple[str, ...]) -> int | None:
        counts = self.task_manifest.stage_counts
        values = [counts.get(stage) for stage in stages]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values if value is not None)

    @staticmethod
    def _natural_batches(
        num_seeds: int,
        batch_size: int,
        *,
        drop_last: bool = False,
    ) -> int:
        return num_seeds // batch_size if drop_last else math.ceil(num_seeds /
                                                                   batch_size)

    def _max_train_batches(self, num_seeds: int | None) -> int | None:
        if num_seeds is None:
            return None
        natural = self._natural_batches(
            num_seeds,
            self.train_batch_size,
            drop_last=self.drop_last_train,
        )
        target = max(natural, self.min_steps_per_epoch)
        if self.max_steps_per_epoch is not None:
            target = min(target, self.max_steps_per_epoch)
        return target

    def _weighted_seed_num_samples(
        self,
        *,
        stage: str,
        config: Mapping[str, Any],
    ) -> int:
        weighted_rows = self._weighted_seed_mass(stage=stage, config=config)
        weighted_rows = self.train_batch_size * math.ceil(
            weighted_rows / self.train_batch_size)
        min_num_samples = self.min_steps_per_epoch * self.train_batch_size
        if self.max_steps_per_epoch is None:
            max_num_samples = weighted_rows
        else:
            max_num_samples = self.max_steps_per_epoch * self.train_batch_size
        return max(min(weighted_rows, max_num_samples), min_num_samples)

    def _weighted_seed_mass(
        self,
        *,
        stage: str,
        config: Mapping[str, Any],
    ) -> float:
        use_sample_weights = bool(config.get("use_sample_weights", False))
        has_class_weights = ("false_weight" in config
                             or "true_weight" in config)
        false_weight = float(config.get("false_weight", 1.0))
        true_weight = float(config.get("true_weight", 1.0))
        if use_sample_weights:
            sample_index = _sample_weight_index_for_stage(
                self.task_manifest.runtime_manifest_path, stage)
            if has_class_weights:
                weighted_rows = (
                    _bucket_total_weight(sample_index, "false_bucket") *
                    false_weight +
                    _bucket_total_weight(sample_index, "true_bucket") *
                    true_weight)
            else:
                weighted_rows = _bucket_total_weight(sample_index, "all")
        else:
            label_index = _binary_label_index_for_stage(
                self.task_manifest.runtime_manifest_path, stage)
            weighted_rows = (int(label_index["false_count"]) * false_weight +
                             int(label_index["true_count"]) * true_weight)
        if not math.isfinite(weighted_rows):
            raise ValueError(
                f"stage {stage!r} has non-finite weighted seed mass")
        if use_sample_weights and weighted_rows > 0:
            weighted_rows = max(1, round(weighted_rows))
        return float(weighted_rows)

    def _max_eval_batches(
        self,
        num_seeds: int | None,
        batch_size: int,
        max_steps: int | None,
    ) -> int | None:
        if max_steps is None:
            return None
        if num_seeds is None:
            return max_steps
        natural = self._natural_batches(num_seeds, batch_size)
        # If the cap doesn't bite, return None so the loader exhausts all worker
        # shards; the bounded path would truncate eval and nan-out AUROC/AP.
        if max_steps >= natural:
            return None
        return max_steps

    def _to_hetero_train(self, batch: Mapping[str, Any]) -> Any:
        r"""Collate for the TRAIN loader, which does not need ``.pkey``.

        Building it is not free: the primary key arrives as an Arrow column
        and becomes a numpy *object* array via ``to_pylist()``, one Python
        object per row per namespace, on every batch. Nothing on the
        training path reads it -- ``pkey`` is consumed only by
        ``evaluation_step``, ``predict_step`` and the XAI subgraph
        collector, all of which run on the val/test/predict loaders that
        keep it. A custom module that did read it during training would
        raise :class:`AttributeError`, which is the right failure: loud,
        immediate, and not a silent change in what the model sees.
        """
        return self._to_hetero(batch, attach_pkeys=False)

    def _to_hetero(self, batch: Mapping[str, Any], *,
                   attach_pkeys: bool = True) -> Any:
        self._capture_explain_selection_batch(batch)
        hetero, _, _ = runtime_batch_to_hetero(
            batch,
            attach_pkeys=attach_pkeys,
            train_table_name=self._train_table_name,
            mapper_bank=self.mapper_bank,
            feature_schema=self._feature_schema,
            task_type=self._task_type,
            seed_node_type=self._seed_node_type,
            pk_col_dict=self.pk_col_dict,
            time_col_dict=self.time_col_dict,
            target_encoder=self._target_encoder,
            index_np_time_unit=self.index_np_time_unit,
            bidirectional_output=self.bidirectional_output,
        )
        return hetero

    @property
    def data(self) -> tuple[Any, Any]:
        """``(feature_store, graph_store)``, populated only during
        RHS feature reads or ``Trainer.explain``; raises otherwise."""
        if self._data is None:
            raise RuntimeError(
                "DiskgraphLightningDataModule.data is only populated "
                "when a runtime feature/graph store is attached.")
        return self._data

    @property
    def feature_store(self) -> Any:
        return self.data[0]

    @property
    def graph_store(self) -> Any:
        return self.data[1]

    @property
    def pk_col_dict(self) -> dict[str, str]:
        if self._pk_col_dict is None:
            self._pk_col_dict, self._time_col_dict = (
                self._load_index_pk_and_time_cols())
        return self._pk_col_dict

    @property
    def time_col_dict(self) -> dict[str, str]:
        if self._time_col_dict is None:
            self._pk_col_dict, self._time_col_dict = (
                self._load_index_pk_and_time_cols())
        return self._time_col_dict

    @property
    def index_np_time_unit(self) -> str:
        """numpy datetime64 unit of the wheel's emitted seed times, read from
        the index (``timestamp_unit``); ``'s'`` by default. Use this to build
        prediction/eval time columns instead of assuming seconds."""
        unit = getattr(self, '_index_np_time_unit', None)
        if unit is None:
            unit = _index_np_time_unit(
                self.task_manifest.graph_runtime_metadata)
            self._index_np_time_unit = unit
        return unit

    def _stage_seed_time(self, stage: "Any") -> "Any | None":
        """Representative seed time for a stage (the stage ``max``) as a
        1-element ``long`` tensor, or ``None`` (-> eval skips the RHS filter).
        Mirrors the legacy ``val_time`` / ``test_time`` contract.
        """
        rng = self._stage_time_dict.get(stage)
        if rng is None:
            return None
        import torch
        _min, _max = rng
        return torch.tensor([int(_max)], dtype=torch.long)

    @property
    def val_time(self) -> "Any | None":
        """Per-stage seed time for VAL eval (see ``_stage_seed_time``)."""
        from kumoapi.train import TrainingStage
        return self._stage_seed_time(TrainingStage.VAL)

    @property
    def test_time(self) -> "Any | None":
        """Per-stage seed time for TEST eval (see ``_stage_seed_time``)."""
        from kumoapi.train import TrainingStage
        return self._stage_seed_time(TrainingStage.TEST)

    @property
    def lp_train_table(self) -> "DiskgraphValidRHS":
        """The temporally-valid RHS provider for ShallowRHS / Hybrid temporal LP.

        Lazily built on first access (a one-time RHS time-column scan). Raises
        for runs that should never need it (non-LP, or LP without an RHS type).
        """
        if not self._lp_train_table_built:
            self._lp_train_table = self._build_lp_train_table()
            self._lp_train_table_built = True
        if self._lp_train_table is None:
            raise RuntimeError(
                "DiskgraphLightningDataModule.lp_train_table is unavailable: "
                "no temporally-valid RHS provider was built. This is expected "
                "for non-LP and STATIC link-prediction runs (which never need "
                "it). It is only built for TEMPORAL link prediction with a "
                f"configured RHS node type (task_type={self._task_type!r}, "
                f"rhs_node_type={self._rhs_node_type!r}).")
        return self._lp_train_table

    def _build_lp_train_table(self) -> "DiskgraphValidRHS | None":
        """Build the temporally-valid RHS provider, or ``None`` when not needed.

        ``None`` (property then raises) only for non-LP. STATIC LP and temporal
        LP missing RHS type / offsets / create-time get a sized null provider
        ("no filter" but a valid ``num_rhs_nodes``).
        """
        from kumoapi.task import TaskType

        is_lp = self._task_type in (
            TaskType.TEMPORAL_LINK_PREDICTION.value,
            TaskType.STATIC_LINK_PREDICTION.value,
        )
        if not is_lp:
            return None

        # STATIC LP: no temporal validity, but the MIPS-readiness check reads
        # num_rhs_nodes off this provider, so return a sized null provider.
        if self._task_type != TaskType.TEMPORAL_LINK_PREDICTION.value:
            num_rhs = 0
            if self._rhs_node_type:
                num_rhs = _num_nodes_for_type(
                    self.task_manifest.graph_runtime_metadata,
                    self._rhs_node_type)
            return self._null_valid_rhs(max(num_rhs, 0))

        # Temporal LP always returns a provider; a null one ("no filter") when
        # validity info is unavailable.
        if (not self._rhs_node_type or self._unix_start_offset is None
                or self._unix_end_offset is None):
            return self._null_valid_rhs()

        graph_runtime_metadata = self.task_manifest.graph_runtime_metadata
        index_path = (graph_runtime_metadata or {}).get('graph_index_path')
        if not index_path:
            return self._null_valid_rhs()

        num_rhs = _num_nodes_for_type(graph_runtime_metadata,
                                      self._rhs_node_type)
        if num_rhs <= 0:
            return self._null_valid_rhs()

        create_col = self.time_col_dict.get(self._rhs_node_type)
        if not create_col:
            # No create-time column -> no temporal validity; degrade to "no filter".
            return self._null_valid_rhs(num_rhs)
        end_col = self._end_time_col_for(self._rhs_node_type)

        create_time = self._fetch_rhs_time_col(index_path, self._rhs_node_type,
                                               num_rhs, create_col)
        end_time = None
        if end_col:
            end_time = self._fetch_rhs_time_col(index_path, self._rhs_node_type,
                                                num_rhs, end_col)

        return DiskgraphValidRHS(
            num_rhs_nodes=num_rhs,
            unix_start_offset=int(self._unix_start_offset),
            unix_end_offset=int(self._unix_end_offset),
            create_time=create_time,
            end_time=end_time,
            stage_time_dict=self._stage_time_dict,
        )

    def _null_valid_rhs(self, num_rhs: int = 0) -> "DiskgraphValidRHS":
        """A provider with no temporal info (``valid_rhs_idx_dict`` is ``None``
        => "no filter")."""
        return DiskgraphValidRHS(
            num_rhs_nodes=int(num_rhs),
            unix_start_offset=int(self._unix_start_offset or 0),
            unix_end_offset=int(self._unix_end_offset or 0),
            create_time=None,
            end_time=None,
            stage_time_dict=self._stage_time_dict,
        )

    def _end_time_col_for(self, node_type: str) -> str | None:
        """End-time column for ``node_type`` from ``index_metadata.json``;
        ``None`` if absent (never-expire)."""
        for entry in self._load_index_node_types():
            if entry.get('name') == node_type:
                col = entry.get('end_time_col')
                return str(col) if col else None
        return None

    @staticmethod
    def _fetch_rhs_time_col(
        index_path: str,
        node_type: str,
        num_nodes: int,
        col_name: str,
    ) -> "Any":
        """Fetch a time/end_time column for node IDs ``[0, num_nodes)`` from the
        DG index as an int64 NANOSECONDS tensor (NaT => -1, the ``RHSTime``
        never-expire sentinel).

        Nanoseconds is the Kumo-Python-internal timestamp unit, so this column is
        directly comparable to the ns ``seed_time`` and ns ``unix_*_offset`` in
        ``RHSTime.get_valid_nodes`` (all-ns RHS validity)."""
        import pandas as pd
        import torch
        from kumo_core.diskgraph.stores import read_node_column

        ser = pd.to_datetime(
            read_node_column(index_path, node_type, col_name,
                             num_nodes).to_pandas(),
            errors="coerce")
        # Return int64 NANOSECONDS regardless of the pandas datetime resolution.
        # Kumo Python is ns-canonical, so the RHS validity arithmetic compares
        # this create/end-time against an ns ``seed_time`` and ns ``unix_*_offset``
        # (all-ns; the engine unit is converted only at engine boundaries). A bare
        # ``.astype('int64')`` is NOT resolution-stable: pandas >=2 preserves the
        # source arrow resolution (e.g. ``datetime64[s]`` => int64 is *seconds*),
        # while pandas <2 always normalized to ``datetime64[ns]`` (=> ns). Forcing
        # ``datetime64[ns]`` via numpy yields nanoseconds on every version; getting
        # this wrong (e.g. mixing seconds create_time with ns seed/offsets) empties
        # the valid-RHS set (~1e9x off => ``create_time <= seed_time`` False for
        # every RHS node => zero training signal).
        vals = ser.to_numpy(dtype="datetime64[ns]").astype("int64").copy()
        # NaT => -1 (RHSTime never-expire sentinel).
        vals[pd.isna(ser).to_numpy()] = -1
        return torch.from_numpy(vals)

    def _load_index_pk_and_time_cols(
        self, ) -> tuple[dict[str, str], dict[str, str]]:
        # index_metadata.json is authoritative for primary_key_map.column and
        # create_time_col (feature_schema.json lacks them).
        index_path = (self.task_manifest.graph_runtime_metadata
                      or {}).get('graph_index_path')
        if not index_path:
            return {}, {}
        meta_path = os.path.join(index_path, 'index_metadata.json')
        if not os.path.exists(meta_path):
            return {}, {}
        with open(meta_path) as f:
            idx_meta = json.load(f)
        pk_dict: dict[str, str] = {}
        time_dict: dict[str, str] = {}
        for entry in idx_meta.get('node_types') or []:
            name = entry.get('name')
            if not name:
                continue
            pk_map = entry.get('primary_key_map') or {}
            pk_col = pk_map.get('column')
            if pk_col:
                pk_dict[name] = pk_col
            time_col = entry.get('create_time_col')
            if time_col:
                time_dict[name] = time_col
        return pk_dict, time_dict

    def stage_inputs(
        self,
        stage: str,
    ) -> tuple[str, Any, Any, Any]:
        """Return TT-NG XAI selection inputs captured from eval batches.

        ``input_nodes`` are root/entity node ids used for ranking and pkey
        mapping. ``input_id`` values are TT-NG stage node ids
        (``task.tt_stage_node_ids``) used for selected replay.
        """
        cached = self._stage_inputs_cache.get(stage)
        if cached is not None:
            return cached
        return self._finish_explain_selection_capture(stage)

    def _load_index_node_types(self) -> list[dict[str, Any]]:
        index_path = (self.task_manifest.graph_runtime_metadata
                      or {}).get('graph_index_path')
        if not index_path:
            return []
        meta_path = os.path.join(index_path, 'index_metadata.json')
        if not os.path.exists(meta_path):
            return []
        with open(meta_path) as f:
            return list(json.load(f).get('node_types') or [])

    def explain_dataloader(
        self,
        stage: str,
        seed_nodes: Any,
    ) -> DataLoader:
        """Eval DataLoader for the XAI second pass over selected stage ids."""
        runtime_stage = "predict" if stage == "pred" else stage
        if runtime_stage == "predict" and self._is_link_pred:
            return self._make_dataloader("predict")

        if hasattr(seed_nodes, 'tolist'):
            selected_stage_node_ids = seed_nodes.tolist()
        else:
            selected_stage_node_ids = list(seed_nodes)

        num_seeds = len(selected_stage_node_ids)
        batch_size = int(self.eval_loader_kwargs['batch_size'])
        # FK-free LP eval/test rows carry target positives. Predict/BP rows do
        # not, so selected replay must use the same seed-only mode as BP predict.
        runtime_options = {
            **self.runtime_options,
            'selected_stage_node_ids':
            selected_stage_node_ids,
            **self._pseudo_time_options(runtime_stage),
        }
        if (runtime_stage != "predict"
                and self.resolve_entity_node_type is not None):
            runtime_options = {
                **runtime_options,
                'entity_node_type': self.resolve_entity_node_type,
                'target_node_type': self.resolve_target_node_type,
            }
        iterable = DiskgraphTaskBatchIterable(
            manifest_path=self.task_manifest.runtime_manifest_path,
            stage=runtime_stage,
            sampling_plan=self.sampling_plan,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            base_seed=self.base_seed,
            shuffle_state=self._shuffle_state,
            num_seeds=num_seeds,
            runtime_options=runtime_options,
            dataloader_num_workers=0,
        )

        return DataLoader(
            iterable,
            batch_size=None,
            num_workers=0,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            collate_fn=self._to_hetero,
        )
