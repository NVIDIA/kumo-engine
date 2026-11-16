# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linux pidfd operations with fallbacks for minimal CPython builds."""
import ctypes
import errno
import os
import select
import signal
import sys

_PIDFD_SEND_SIGNAL_SYSCALL = 424
_PIDFD_OPEN_SYSCALL = 434
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


def _raise_syscall_error() -> None:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))


def pidfd_open(pid: int) -> int:
    """Open a stable Linux process handle without requiring CPython support."""
    if not sys.platform.startswith('linux'):
        raise NotImplementedError('pidfds require Linux')
    native = getattr(os, 'pidfd_open', None)
    if native is not None:
        return native(pid)
    result = _LIBC.syscall(
        ctypes.c_long(_PIDFD_OPEN_SYSCALL),
        ctypes.c_int(pid),
        ctypes.c_uint(0),
    )
    if result < 0:
        _raise_syscall_error()
    return int(result)


def pidfd_send_signal(pidfd: int, sig: int) -> None:
    """Signal a pidfd without requiring CPython's optional signal wrapper."""
    native = getattr(signal, 'pidfd_send_signal', None)
    if native is not None:
        native(pidfd, sig)
        return
    result = _LIBC.syscall(
        ctypes.c_long(_PIDFD_SEND_SIGNAL_SYSCALL),
        ctypes.c_int(pidfd),
        ctypes.c_int(sig),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )
    if result < 0:
        _raise_syscall_error()


def pidfd_exited(pidfd: int) -> bool:
    """Return whether a process pidfd observed its whole thread group exit."""
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    events = poller.poll(0)
    if any(mask & select.POLLNVAL for _, mask in events):
        raise OSError(errno.EBADF, os.strerror(errno.EBADF))
    return bool(events)


def pidfd_process_exited(pid: int) -> bool:
    """Open, poll, and close a stable process pidfd."""
    pidfd = pidfd_open(pid)
    try:
        return pidfd_exited(pidfd)
    finally:
        os.close(pidfd)
