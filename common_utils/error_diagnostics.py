# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared policy for rendering exception messages in operational logs."""

from __future__ import annotations

from typing import Iterator, Optional

from common_utils.error import PublicException
from io_utils.errors import (
    RESTRICTED_LOG_MESSAGE_ATTRIBUTE,
    SAFE_LOG_MESSAGE_ATTRIBUTE,
)

_MAX_EXCEPTION_CHAIN_DEPTH = 20


def _exception_has_restricted_marker(error: BaseException) -> bool:
    # Dedicated Kumo boundary-wrapper types may declare this invariant on the
    # class. This is explicit marker metadata, not a registry keyed by an
    # arbitrary provider exception class or module name.
    return (getattr(error, RESTRICTED_LOG_MESSAGE_ATTRIBUTE, False) is True
            or getattr(type(error), RESTRICTED_LOG_MESSAGE_ATTRIBUTE,
                       False) is True)


def _iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = error
    depth = 0
    while (current is not None and id(current) not in seen
           and depth < _MAX_EXCEPTION_CHAIN_DEPTH):
        seen.add(id(current))
        yield current
        cause = current.__cause__
        if cause is not None:
            current = cause
        else:
            # ``raise ... from None`` suppresses traceback rendering but keeps
            # the caught exception in ``__context__``. Diagnostic safety must
            # inspect that retained context even when the traceback printer
            # omits it.
            current = current.__context__
        depth += 1


def exception_chain_contains_restricted_detail(error: BaseException) -> bool:
    """Whether a retained cause/context chain reaches an explicit boundary."""
    for current in _iter_exception_chain(error):
        if _exception_has_restricted_marker(current):
            return True
    return False


def exception_message_should_be_logged_at_error(error: BaseException) -> bool:
    """Return whether ``str(error)`` belongs in ordinary ``ERROR`` logs."""
    if _exception_has_restricted_marker(error):
        return False
    if isinstance(error, PublicException):
        return True

    # Preserve a Kumo-authored outer diagnostic while preventing the common
    # accidental leak ``raise RuntimeError(f'failed: {provider_error}')``.
    # A marked cause does not make unrelated outer text sensitive merely by
    # being chained to it; the boundary must mark a transformed/copying wrapper
    # explicitly when the restricted text is not retained verbatim.
    try:
        message = str(error)
    except BaseException:  # The bounded renderer contains this separately.
        return True

    for current in _iter_exception_chain(error):
        if current is error:
            continue
        if _exception_has_restricted_marker(current):
            try:
                restricted_detail = str(current)
            except BaseException:
                return False
            restricted_renderings = [restricted_detail]
            try:
                restricted_renderings.append(repr(current))
            except BaseException:
                pass
            if any(detail and detail in message
                   for detail in restricted_renderings):
                return False
    return True


def safe_exception_message_for_log(error: BaseException) -> Optional[str]:
    """Return a boundary-selected safe message for a restricted exception."""
    if not _exception_has_restricted_marker(error):
        return None
    message = getattr(error, SAFE_LOG_MESSAGE_ATTRIBUTE, None)
    return message if isinstance(message, str) and message else None
