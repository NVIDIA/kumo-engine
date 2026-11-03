# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import multiprocessing as mp
import signal
import time

import pytest
from kumo_core.diskgraph import lightning_data


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
