# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import multiprocessing as mp
import os
import queue as python_queue
import resource
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

import psutil

if sys.platform.startswith("linux"):
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
else:
    libgcc_s = None


def withMemory(
    limit: int = resource.RLIM_INFINITY,
    return_result: bool = True,
) -> Callable:
    """
    This decorator provides the ability for functions to be executed in a
    subprocess with limited address space size (RLIMIT_AS). Notice that this
    limits the entire virtual memory used by the process, including swap
    memory, shared libraries, code, and, .data. A rough aproximation of the
    current virtual memory size is made so that users can treat the limit
    parameter as a relatively tight representation of available memory for
    this process.

    Args:
        limit (int): the number of bytes to allocate for the address space
            size of the subprocess (default: no limit).
        return_result (bool): whether to return the result of the function. If
            disabled, will return True if the function succeeds and raise an
            error if it fails. Note that returning a result is implemented as
            writing and reading from a shared multiprocess queue, which can
            impose overhead for large objects (default: True).
    """
    def _with_memory(function: Callable) -> Callable:
        if libgcc_s is None:
            raise OSError(f"Memory testing only supported on Linux "
                          f"(got '{sys.platform}')")

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            queue = kwargs.pop('__queue', None)
            result = None
            if not queue:  # pylint: disable=no-else-return
                queue = mp.SimpleQueue()
                proc = mp.Process(target=wrapper, args=args, kwargs={
                    '__queue': queue,
                    **kwargs
                })
                proc.start()
                try:
                    result = queue.get()
                    proc.join()
                except python_queue.Empty:
                    proc.terminate()

                if isinstance(result, BaseException):
                    raise result
                return result
            else:
                try:
                    assert limit, "limit cannot be None."

                    baseline_vms = psutil.Process(
                        os.getpid()).memory_full_info().vms

                    limit_ = limit
                    if limit_ != resource.RLIM_INFINITY:
                        limit_ += baseline_vms

                    resource.setrlimit(resource.RLIMIT_AS, (limit_, limit_))
                    result = function(*args, **kwargs)

                    if not return_result:
                        result = True
                except BaseException as exc:
                    result = exc

                queue.put(result)
                return result

        return wrapper

    return _with_memory
