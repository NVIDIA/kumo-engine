# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connector-local blocking decorators with provider-safe tracing."""

from dataclasses import replace
from typing import Optional

from async_utils.sync import blocking_func as _blocking_func
from async_utils.sync import blocking_method as _blocking_method
from monitoring.tracing import TraceOptions
from monitoring.tracing import trace as _trace


def _safe_options(options: Optional[TraceOptions]) -> TraceOptions:
    if options is None:
        return TraceOptions(record_exceptions=False)
    return replace(options, record_exceptions=False)


def blocking_method(executor=None, *, tracing_options=None, **kwargs):
    """Wrap connector methods without exporting provider exception bodies."""
    return _blocking_method(
        executor,
        tracing_options=_safe_options(tracing_options),
        **kwargs,
    )


def blocking_func(executor=None, *, tracing_options=None):
    """Wrap connector functions without exporting provider exception bodies."""
    return _blocking_func(
        executor,
        tracing_options=_safe_options(tracing_options),
    )


def trace(options=None):
    """Trace connector calls without exporting provider exception bodies."""
    return _trace(_safe_options(options))
