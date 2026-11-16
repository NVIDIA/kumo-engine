# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Callers act on the class, so the hierarchy is the contract."""

import pytest
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    RowCountMismatchError,
    SourceAccessDeniedError,
    SourceNotFoundError,
    TransientSourceError,
    UnsupportedReadModeError,
)

TERMINAL = [
    PermanentSourceError, ReconnectRequiredError, UnsupportedReadModeError,
    RowCountMismatchError, SourceAccessDeniedError, SourceNotFoundError
]


@pytest.mark.parametrize("error", TERMINAL)
def test_a_terminal_failure_is_never_mistaken_for_a_retryable_one(error):
    assert issubclass(error, PermanentSourceError)
    assert not issubclass(error, TransientSourceError)
    assert not issubclass(TransientSourceError, PermanentSourceError)


def test_a_failure_carries_the_table_and_a_short_read_carries_both_counts():
    assert TransientSourceError("timed out", table="ORDERS").table == "ORDERS"
    error = RowCountMismatchError(expected=1000, actual=999, table="ORDERS")
    assert (error.expected, error.actual) == (1000, 999)
    assert "999" in str(error) and "1000" in str(error)


def test_only_a_missing_grant_selects_the_streamed_fallback():
    assert issubclass(SourceAccessDeniedError, PermanentSourceError)
    assert not any(
        issubclass(error, SourceAccessDeniedError)
        for error in TERMINAL if error is not SourceAccessDeniedError)
