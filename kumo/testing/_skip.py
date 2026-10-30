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


def requires_resource(provider: str, func: Callable | None = None, *,
                      profile: str | None = None) -> Callable:
    """Describe live test/fixture dependencies without accessing credentials."""
    if func is None:
        return functools.partial(requires_resource, provider, profile=profile)
    resources = set(getattr(func, '__kumo_test_resources__', ()))
    resources.add(provider)
    setattr(func, '__kumo_test_resources__', tuple(sorted(resources)))
    if profile:
        profiles = set(getattr(func, '__kumo_ci_profiles__', ()))
        profiles.add(profile)
        func.__kumo_ci_profiles__ = tuple(sorted(profiles))
    return func
