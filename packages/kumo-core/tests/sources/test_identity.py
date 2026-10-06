# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One spelling of the version stamp, because it is a path segment."""

from datetime import datetime, timezone

import pytest
from kumo_core.sources.identity import epoch_second, hour_version

UTC = timezone.utc


@pytest.mark.parametrize("moment,expected", [
    (datetime(2026, 8, 27, tzinfo=UTC), "1787788800"),
    (datetime(2026, 8, 27, 0, 0, 0, 500_000, tzinfo=UTC), "1787788800"),
    (datetime(1969, 12, 31, 23, 59, 59, 500_000, tzinfo=UTC), "-0"),
])
def test_the_second_is_truncated_the_way_every_written_stamp_already_was(
        moment, expected):
    """``int`` would spell the sub-epoch case ``0``, renaming a live path."""
    assert epoch_second(moment) == expected


def test_the_fallback_names_the_hour_it_was_taken_in_and_no_finer():
    now = datetime.now(UTC)
    assert hour_version() == epoch_second(
        now.replace(minute=0, second=0, microsecond=0))
    assert int(hour_version()) % 3600 == 0
