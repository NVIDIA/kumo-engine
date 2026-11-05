# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import multiprocessing as mp
import os
import signal
import time
from types import SimpleNamespace

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
    (RuntimeError('could not unlink the shared memory file /torch_1_2_3 : '
                  'No such file or directory (2)'), True, 2,
     'loginctl enable-linger'),
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


_SYSTEMD = 'run/systemd/system/.keep'
_LOGIND = 'etc/systemd/logind.conf'
_DROP_IN = 'etc/systemd/logind.conf.d/50-ipc.conf'
_VENDOR_LOGIND = 'usr/lib/systemd/logind.conf'
_VENDOR_DROP_IN = 'usr/lib/systemd/logind.conf.d/50-ipc.conf'


@pytest.mark.parametrize('files, uid, removes', [
    ({}, 1500, True),
    ({_LOGIND: 'RemoveIPC=no\n'}, 1500, False),
    ({_LOGIND: 'RemoveIPC=no\n', _DROP_IN: 'RemoveIPC=yes\n'}, 1500, True),
    ({_DROP_IN: '[Login]\nRemoveIPC = off\n'}, 1500, False),
    ({'var/lib/systemd/linger/job': ''}, 1500, False),
    ({'etc/login.defs': 'UID_MIN\t\t2000\n'}, 1500, False),
    ({}, 0, False),
    ({_LOGIND: None}, 1500, True),
    ({_LOGIND: None, _VENDOR_LOGIND: 'RemoveIPC=no\n'}, 1500, False),
    ({_VENDOR_DROP_IN: 'RemoveIPC=no\n'}, 1500, False),
    ({_VENDOR_DROP_IN: 'RemoveIPC=no\n', _DROP_IN: ''}, 1500, True),
    ({_SYSTEMD: None}, 1500, False),
])
def test_logind_removes_ipc(tmp_path, monkeypatch, files, uid, removes):
    base = {_SYSTEMD: '', _LOGIND: '[Login]\n#RemoveIPC=yes\n'}
    for name, text in {**base, **files}.items():
        if text is not None:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
    monkeypatch.setattr(os, 'geteuid', lambda: uid)
    monkeypatch.setattr(lightning_data.pwd, 'getpwuid',
                        lambda uid: SimpleNamespace(pw_name='job'))
    assert lightning_data._logind_removes_ipc(str(tmp_path)) is removes
