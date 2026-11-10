# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import multiprocessing as mp
import signal
import time

import pytest
from kumo_core.diskgraph import lightning_data
from torch.utils.data import IterableDataset


@pytest.mark.parametrize('code', [errno.ENOSYS, errno.EPERM])
def test_workers_are_stopped_where_pidfds_are_refused(monkeypatch, code):
    def refused(pid):
        raise OSError(code, 'pidfd_open refused')

    monkeypatch.setattr(lightning_data, '_pidfd_open', refused)
    process = mp.get_context('spawn').Process(target=time.sleep, args=(60, ))
    process.start()
    handle = lightning_data._WorkerHandle.open(process)
    try:
        assert not handle.exited()
        handle.send_signal(signal.SIGKILL)
        process.join(10)
        assert handle.exited() and process.exitcode == -signal.SIGKILL
    finally:
        handle.close()
        process.kill()
        process.join()


def test_other_pidfd_errors_still_fail(monkeypatch):
    def broken(pid):
        raise OSError(errno.EBADF, 'bad')

    monkeypatch.setattr(lightning_data, '_pidfd_open', broken)
    process = mp.get_context('spawn').Process(target=time.sleep, args=(60, ))
    process.start()
    try:
        with pytest.raises(OSError, match='bad'):
            lightning_data._WorkerHandle.open(process)
    finally:
        process.kill()
        process.join()


class _Unpicklable:
    def __init__(self, error):
        self.error = error

    def __reduce__(self):
        raise self.error


class _UnpicklableBatches(IterableDataset):
    def __init__(self, error, endless):
        self.error = error
        self.endless = endless

    def __iter__(self):
        yield _Unpicklable(self.error)
        while self.endless:
            yield 0


@pytest.mark.parametrize('error, endless, prefetch, expected', [
    (RuntimeError('could not unlink the shared memory file'), True, 2,
     'could not unlink the shared memory file'),
    (RuntimeError('not picklable'), False, 2, 'not picklable'),
    (RuntimeError('not picklable'), False, 1, 'not picklable'),
    (type('LocalError', (Exception, ), {})('local'), True, 2, 'local'),
])
def test_unsendable_batch_fails_instead_of_being_dropped(
        error, endless, prefetch, expected):
    def hung(signum, frame):
        raise TimeoutError('DataLoader waited forever for a dropped batch')

    loader = lightning_data._WorkerPoolDataLoader(
        _UnpicklableBatches(error, endless),
        batch_size=None,
        num_workers=1,
        prefetch_factor=prefetch,
        multiprocessing_context='fork',
        worker_init_fn=lightning_data._init_diskgraph_worker,
        phase='predict',
        registry=lightning_data._WorkerPoolRegistry(),
    )
    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(30)
    try:
        with pytest.raises(RuntimeError, match=expected):
            next(iter(loader))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
