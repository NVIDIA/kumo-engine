# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real-data history checker on a hand-auditable raw AML fixture."""

from datetime import UTC, datetime
from test.kumo_pytorch.aml_source import (
    account_identity_map,
    prepare_aml_sources,
    prepared_dataset,
)
from test.kumo_pytorch.history_checks import check_histories, expected_history
from test.kumo_pytorch.involvement import source_reference

import pytest

from .test_aml_source import account, source_files, transaction


@pytest.mark.parametrize(
    "filtered", [False, True], ids=["windows", "filters-and-windows"]
)
def test_raw_aml_multisource_history(tmp_path, filtered):
    # Two identical physical source rows, timestamp ties across roles, a self
    # transfer appearing in both components, and distinct cutoff/filter traps.
    rows = [
        transaction(timestamp="2022/09/01 12:00", payment_format="ACH"),
        transaction(timestamp="2022/09/01 12:00", payment_format="ACH"),
        transaction(
            timestamp="2022/09/01 12:00",
            from_bank="2",
            from_account="0008",
            to_bank="1",
            to_account="0007",
            payment_format="ACH",
        ),
        transaction(
            timestamp="2022/09/01 12:00",
            to_bank="1",
            to_account="0007",
            payment_format="ACH",
        ),
        transaction(timestamp="2022/09/01 13:00", amount_paid="-1"),
        transaction(
            timestamp="2022/09/01 23:30",
            from_bank="2",
            from_account="0008",
            to_bank="1",
            to_account="0007",
            payment_format="ACH",
        ),
        transaction(timestamp="2022/09/02 00:00", payment_format="ACH"),
        transaction(
            timestamp="2022/09/01 14:00",
            from_bank="2",
            from_account="0008",
            to_bank="1",
            to_account="0007",
            payment_format="Wire",
        ),
    ]
    source, binding, checksums = source_files(
        tmp_path, rows=[dict(row, is_laundering="0") for row in rows]
    )
    prepared = tmp_path / "prepared"
    prepare_aml_sources(
        source, binding, prepared, "hi-small", expected_checksums=checksums
    )
    root = account(1, "0007")
    cutoff = int(datetime(2022, 9, 2, tzinfo=UTC).timestamp()) * 10**9
    groups = {
        split: [{"sample_id": split, "entity_id": root, "as_of_time": cutoff}]
        for split in ("train", "valid", "test")
    }
    reference = source_reference(source, [root], include_values=True)
    # Check the oracle against literals before asking the package to execute.
    expected = [(0, 0), (0, 1), (0, 3), (1, 2), (1, 3)]
    if not filtered:
        expected += [(0, 4), (1, 7)]
    assert (
        expected_history(groups["train"][0], reference["events"], filtered=filtered)
        == expected
    )
    result = check_histories(
        prepared_dataset(prepared),
        account_identity_map(prepared),
        groups,
        reference,
        directory=tmp_path / "run",
        filtered=filtered,
    )
    assert result["status"] == "passed" and result["samples"] == 3
    assert result["events_by_input"]["all_ascending"] == 3 * len(expected)


@pytest.mark.parametrize("keep", ["all", "first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_tied_histories_preserve_membership_and_time_boundaries(keep, order):
    from test.kumo_pytorch.history_checks import verify_history_selection

    eligible = [(1, 0, 10), (2, 0, 20), (2, 1, 21), (2, 1, 22), (3, 0, 30)]
    valid = {
        "all": [(0, 10), (1, 22), (0, 20), (1, 21), (0, 30)],
        "first": [(0, 10), (1, 22), (0, 20)],
        "last": [(1, 22), (0, 20), (0, 30)],
    }[keep]
    if order == "descending":
        valid = valid[::-1]
    verify_history_selection(valid, eligible, keep=keep, order=order)
    with pytest.raises(AssertionError):
        verify_history_selection(valid + valid[:1], eligible, keep=keep, order=order)
    with pytest.raises(AssertionError):
        verify_history_selection(valid[:-1], eligible, keep=keep, order=order)
    with pytest.raises(AssertionError):
        verify_history_selection(valid[::-1], eligible, keep=keep, order=order)
    with pytest.raises(AssertionError):
        verify_history_selection(
            [(99, 99)] + valid[1:], eligible, keep=keep, order=order
        )
