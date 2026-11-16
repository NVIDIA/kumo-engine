# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from logging import Logger
from typing import Any, Optional, TypeAlias

from io_utils import logging

_logger = logging.getLogger(__name__)

_retry_disabled: ContextVar[bool] = ContextVar('_retry_disabled',
                                               default=False)

_ExceptionMatcher: TypeAlias = Callable[[Exception], bool]
_ExceptionCallback: TypeAlias = Callable[[Exception], None]


def __retry_internal(
    f,
    exceptions: Any = Exception,
    callbacks: dict[_ExceptionMatcher, _ExceptionCallback] = {},
    tries: int = -1,
    delay: float | int = 0,
    max_delay: Optional[float | int] = None,
    backoff: float | int = 1,
    jitter: float | int | tuple[float | int, float | int] = 0,
    logger: Optional[Logger] = _logger,
):
    """
    Executes a function and retries it if it failed.

    :param f: the function to execute.
    :param exceptions: an exception or a tuple of exceptions to catch.
                       default: Exception.
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts.
                    default: 1 (no backoff).
    :param jitter: extra seconds added to delay between attempts. Fixed if a
                   number, random uniform over the range if a ``(min, max)``
                   tuple. default: 0.
    :param logger: logger.warning(...) will be called on failed attempts.
                   default: _logger. if None, logging is disabled.
    :returns: the result of the f function.
    """
    if _retry_disabled.get():
        return f()

    _tries, _delay = tries, delay
    while _tries:
        try:
            return f()
        except exceptions as e:
            _tries -= 1
            if not _tries:
                raise

            for matcher, callback in callbacks.items():
                if matcher(e):
                    callback(e)

            if logger:
                # Exceptions from cloud SDKs and database drivers can contain
                # request bodies, SQL, or credentials.  Retry logging is an
                # infrastructure boundary, so never interpolate provider
                # exception text here.
                logger.warning(
                    'Operation failed (error_type=%s), retrying in %s '
                    'seconds...',
                    type(e).__name__,
                    _delay,
                )

            time.sleep(_delay)
            _delay *= backoff
            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            if max_delay is not None:
                _delay = min(_delay, max_delay)


def retry(
    exceptions: Any = Exception,
    callbacks: dict[_ExceptionMatcher, _ExceptionCallback] = {},
    tries: int = -1,
    delay: float | int = 0,
    max_delay: Optional[float | int] = None,
    backoff: float | int = 1,
    jitter: float | int | tuple[float | int, float | int] = 0,
    logger: Optional[Logger] = _logger,
):
    """Returns a retry decorator.

    :param exceptions: an exception or a tuple of exceptions to catch.
                       default: Exception.
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts.
                    default: 1 (no backoff).
    :param jitter: extra seconds added to delay between attempts.
                   default: 0.
    :param logger: logger.warning(...) will be called on failed attempts.
                   default: _logger. if None, logging is disabled.
    :returns: a retry decorator.
    """
    def retry_decorator(f):
        # NOTE: ``functools.wraps`` (not the ``decorator`` library) so the
        # wrapper propagates ``__annotate__`` (PEP 649). The fsutil impls stack
        # this under ``@api.<op>.register``; on Python 3.14 ``singledispatch``
        # reads the first-arg annotation off the outermost wrapper, and a
        # ``decorator``-built wrapper sets ``__annotate__`` to ``None``.
        @wraps(f)
        def wrapper(*fargs, **fkwargs):
            return __retry_internal(
                partial(f, *fargs, **fkwargs),
                exceptions,
                callbacks,
                tries,
                delay,
                max_delay,
                backoff,
                jitter,
                logger,
            )

        return wrapper

    return retry_decorator


@contextmanager
def no_retry() -> Generator[None, None, None]:
    """Disable ``io_utils.retry`` for the current context (including nested calls).

    Uses a context variable (not monkeypatching) so retries are skipped while
    the context is active. Nesting restores the prior flag via ContextVar tokens.
    """
    var = _retry_disabled
    token = var.set(True)
    try:
        yield
    finally:
        var.reset(token)
