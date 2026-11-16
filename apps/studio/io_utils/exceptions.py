# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Tuple, Type

from async_utils.caching import EventLoopBlockedError
from io_utils import logging

logger = logging.getLogger(__name__)

_EXCEPTIONS_MUST_RAISE = (EventLoopBlockedError, )


@contextmanager
def catch_and_log(ex_to_catch: Tuple[Type[Exception], ...] = (Exception, ),
                  ex_to_raise: Tuple[Type[Exception], ...] = ()):
    r"""
    Catches and logs exceptions. If no `ex_to_catch` are provided, it catches
    all exceptions. The caught exception will not be raised, unless it's
    one of _EXCEPTIONS_MUST_RAISE or `ex_to_raise`.

    Example Usage:
    with catch_and_log():
        raise ValueError("This will be caught and logged")

    with catch_and_log((ValueError, TypeError)):
        raise ValueError("This will be caught and logged")

    with catch_and_log((ValueError, TypeError)):
        raise AssertionError("This exception will still be raised")

    with catch_and_log(None, (TypeError)):
        raise RuntimeError('This will be caught and logged')
        raise TypeError('This exception will raise')
    """
    ex_to_raise = ex_to_raise + _EXCEPTIONS_MUST_RAISE
    try:
        yield
    except ex_to_catch as e:  # type: ignore
        logger.exception(f"Encountered exception: {e}")
        if ex_to_raise and isinstance(e, ex_to_raise):
            raise
