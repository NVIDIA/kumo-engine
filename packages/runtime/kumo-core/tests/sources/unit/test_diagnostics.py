# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failure must be actionable without republishing what the vendor said."""

import pytest
from kumo_core.sources.diagnostics import safe_summary, translated
from kumo_core.sources.errors import (
    PermanentSourceError,
    SourceError,
    TransientSourceError,
)

SECRET = "https://bucket/f.parquet?X-Amz-Signature=abcdef"


class Refused(Exception):
    def __init__(self):
        super().__init__(f"403 Forbidden for url: {SECRET}")
        self.code = 403
        self.reason = "forbidden"


class Lost(Refused):
    pass


def summarize(error):
    return safe_summary(error, code=getattr(error, "code", None),
                        reason=getattr(error, "reason", None))


def raising(error):
    rules = ((Lost, TransientSourceError), (Refused, PermanentSourceError))
    with translated("read of orders", table="orders", rules=rules,
                    summarize=summarize):
        raise error


def test_a_failure_is_named_by_its_type_and_only_the_codes_it_gave():
    assert safe_summary(Refused(), code=403) == "Refused code=403"
    assert safe_summary(Refused(), code=None, reason="") == "Refused"


def test_a_translated_failure_carries_its_codes_its_table_and_no_vendor_text():
    with pytest.raises(PermanentSourceError) as failure:
        raising(Refused())
    reported = str(failure.value)
    assert SECRET not in reported
    assert "Refused code=403 reason=forbidden" in reported
    assert failure.value.table == "orders"
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__


def test_the_first_matching_rule_wins():
    """A subclass listed first is not swallowed by its own base."""
    with pytest.raises(TransientSourceError):
        raising(Lost())


def test_a_failure_matching_no_rule_is_left_alone():
    """Misclassifying an unknown failure is worse than not classifying it."""
    with pytest.raises(KeyError):
        raising(KeyError("no rule for this"))


def test_a_failure_already_in_the_taxonomy_passes_through():
    original = TransientSourceError("already classified")
    with pytest.raises(SourceError) as failure:
        raising(original)
    assert failure.value is original
