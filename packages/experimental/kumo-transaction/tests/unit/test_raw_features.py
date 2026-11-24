# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from tests.conftest import make_event

from tfm.contracts.schema import TransactionSequence
from tfm.data.collate import raw_feature_vector


@pytest.mark.parametrize("amount_key", ["amount", "gross_amount"])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([4.0, math.nan, math.inf, -math.inf, None, 2.0], [1.0, 4.0, 6.0]),
        ([-4.0, -2.0, math.nan, None], [-1.5, 0.0, 4.0]),
        ([None, math.nan, math.inf, -math.inf], [0.0, 0.0, 4.0]),
        ([4.0, 2.0], [3.0, 4.0, 2.0]),
    ],
)
def test_nonfinite_raw_amounts_follow_the_missing_field_convention(amount_key, values, expected):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    events = tuple(
        replace(
            make_event("owner", f"event-{i}", start + timedelta(hours=i)),
            numeric={} if value is None else {amount_key: value},
        )
        for i, value in enumerate(values)
    )
    sequence = TransactionSequence("owner", events, start + timedelta(days=1), None)
    features = raw_feature_vector(sequence, amount_key=amount_key)
    omitted = replace(
        sequence,
        events=tuple(
            replace(e, numeric={}) if value is None or not math.isfinite(value) else e
            for e, value in zip(events, values, strict=True)
        ),
    )
    assert features == raw_feature_vector(omitted, amount_key=amount_key)
    assert features[:3] == expected
    assert all(math.isfinite(value) for value in features)


def test_empty_raw_history_still_has_zero_features():
    sequence = TransactionSequence("owner", (), datetime(2024, 1, 1, tzinfo=UTC), None)
    assert raw_feature_vector(sequence) == [0.0] * 5
