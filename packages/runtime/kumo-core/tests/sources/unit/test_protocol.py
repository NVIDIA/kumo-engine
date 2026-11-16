# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.sources.errors import UnsupportedReadModeError
from kumo_core.sources.protocol import require_mode, staged_first
from kumo_core.sources.types import ReadMode


def test_staging_leads_where_a_warehouse_offers_it():
    assert staged_first(True) == (ReadMode.STAGED, ReadMode.STREAMED)
    assert staged_first(False) == (ReadMode.STREAMED, )


def test_a_mode_outside_the_offer_is_refused_naming_asked_and_offered():
    require_mode("orders", ReadMode.STAGED, staged_first(True))
    with pytest.raises(UnsupportedReadModeError) as failure:
        require_mode("orders", ReadMode.STAGED, staged_first(False))
    assert "staged" in str(failure.value)
    assert "streamed" in str(failure.value)
    assert failure.value.table == "orders"
