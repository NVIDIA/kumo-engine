# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from collections.abc import Callable


def skip_if(condition: bool, reason: str, func: Callable) -> Callable:
    """Wrap func to call pytest.skip(reason) when condition is True.

    Handles regular, async, and generator test functions.
    """
    import pytest

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async_wrapper(*args, **kwargs):
            if condition:
                pytest.skip(reason)
            return await func(*args, **kwargs)

        return _async_wrapper

    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def _generator_wrapper(*args, **kwargs):
            if condition:
                pytest.skip(reason)
                return
            yield from func(*args, **kwargs)

        return _generator_wrapper

    @functools.wraps(func)
    def _sync_wrapper(*args, **kwargs):
        if condition:
            pytest.skip(reason)
        return func(*args, **kwargs)

    return _sync_wrapper
