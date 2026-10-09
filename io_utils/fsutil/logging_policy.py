# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
import time
from functools import wraps
from inspect import currentframe, getframeinfo
from types import FrameType
from typing import Callable, Optional, Tuple, TypeAlias

from io_utils.errors import mark_exception_restricted

ExceptionSet: TypeAlias = Tuple[type[Exception], ...] | type[Exception] | None


def _frame_up(frame: FrameType | None, steps: int) -> FrameType | None:
    if steps == 0 or not frame:
        return frame
    return _frame_up(frame.f_back, steps - 1)


def logging_policy(
    logger: logging.Logger,
    *,
    level_success: int = logging.DEBUG,
    level_slow: int = logging.INFO,
    level_failure: int = logging.ERROR,
    slow_threshold_ms: Optional[float] = 1000,
    restricted_failures: bool = False,
):
    """
    Define a logging policy for a file that can be customized per-function
    ```
    logger = logging.getLogger(__name__)
    policy = logging_policy(logger, level_success=logging.INFO)

    @policy()
    def func1(): ...

    @policy(skip_args=[0])
    def func2(arg_to_avoid_logging): ...

    @policy(transform_args={0: len})
    def func3(arg_to_log_length_of): ...

    @policy(log_return=True)
    def func4() -> str: ...

    @policy(log_return=len)
    def func5() -> bytes: ...
    ```
    """
    def _logging_policy(
        *,
        skip_args: list[int] = [],
        transform_args: dict[int, Callable] = {},
        log_return: bool | Callable = False,
        allowed_exceptions: ExceptionSet = None,
    ):
        skipped_value = "<not shown>"

        def _transform_args(args: tuple, transforms: dict[int, Callable]):
            return tuple([
                transforms[i](x) if i in transforms else x
                for i, x in enumerate(args)
            ])

        def _filter_args(args: tuple, to_skip: set[int]):
            return tuple([
                x if i not in to_skip else skipped_value
                for i, x in enumerate(args)
            ])

        def _result_log(result) -> str:
            if callable(log_return):
                return f" -> {log_return(result)}"
            elif log_return:
                return f" -> {result}"
            else:
                return ""

        def _wrapper(func):
            # N.B. _wrapper is called from impl_*.py, so stack has context
            # when _wrap is called, it's higher in the stack
            # record context here, inject it when recording logs
            frame = _frame_up(currentframe(), 1)
            caller = getframeinfo(frame) if frame else None
            filename = osp.basename(
                caller.filename) if caller else "(unknown file)"
            lineno = caller.lineno if caller else 0

            def _record_log(level: int, msg: str, *args, exc_info=None):
                # Honor the logger's effective level. logger.handle() bypasses
                # the level check that logger.debug()/info()/etc. perform, so
                # without this guard suppressing fsutil DEBUG via setLevel
                # (e.g. in pytest) has no effect.
                if not logger.isEnabledFor(level):
                    return
                record = logger.makeRecord(logger.name, level, filename, lineno,
                                           msg, args, exc_info, func.__name__)
                logger.handle(record)

            def _get_log_level(duration_ms: float) -> int:
                if slow_threshold_ms is not None and duration_ms > slow_threshold_ms:
                    return level_slow
                return level_success

            def _mark_restricted_failure(error: Exception) -> None:
                marked_error = mark_exception_restricted(
                    error,
                    safe_message=(f'Remote filesystem operation '
                                  f'{func.__name__} failed'),
                )
                if marked_error is not error:
                    raise marked_error from None

            @wraps(func)
            def _wrap(*raw_args, **kwargs):
                args_t = _transform_args(raw_args, transform_args)
                args = _filter_args(args_t, set(skip_args))
                log_line = f"{func.__name__} {args=} {kwargs=}"
                start_time = time.perf_counter()
                if allowed_exceptions is not None:
                    try:
                        result = func(*raw_args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_log(_get_log_level(duration_ms), "success: %s%s (%.1fms)",
                                    log_line, _result_log(result), duration_ms)
                        return result
                    except allowed_exceptions as e:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_log(_get_log_level(duration_ms), "success[%s]: %s (%.1fms)",
                                    type(e).__name__, log_line, duration_ms)
                        if restricted_failures:
                            # Allowed exceptions are expected to preserve their
                            # type so callers can continue matching them.
                            _mark_restricted_failure(e)
                        raise e
                    except Exception as e:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        # Provider exceptions can contain signed URLs,
                        # temporary credentials, request bodies, and account
                        # identifiers. Record only the exception class; the
                        # activity boundary converts it to a stable public
                        # error without serializing provider text.
                        _record_log(level_failure,
                                    "failure[%s]: %s (%.1fms)",
                                    type(e).__name__, log_line, duration_ms)
                        if restricted_failures:
                            _mark_restricted_failure(e)
                        raise
                else:
                    try:
                        result = func(*raw_args, **kwargs)
                    except Exception as e:
                        if not restricted_failures:
                            raise
                        _mark_restricted_failure(e)
                        raise
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    _record_log(_get_log_level(duration_ms), "success: %s%s (%.1fms)",
                                log_line, _result_log(result), duration_ms)
                    return result

            return _wrap

        return _wrapper

    return _logging_policy
