"""Hand-computed witnesses for the independent task reference itself."""

from copy import deepcopy
from datetime import datetime, timezone
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test.kumo_pytorch.involvement import (
    binary_evaluation,
    expected_anchor,
    source_reference,
    verify_anchors,
)

T = int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp())
DAY = 86400


def row(account, offset=0, label=0, split="train"):
    return {
        "sample_id": f"{account}@{offset}",
        "entity_id": account,
        "as_of_time": (T + offset) * 10**9,
        "involvement": label,
        "forecast_split": split,
    }


@pytest.fixture
def reference():
    accounts = ("sender", "receiver", "self", "exclusive_end", "past_only", "clean")
    events = [
        {"id": index, "time": T - 1, "accounts": {account}, "laundering": True}
        for index, account in enumerate(accounts)
    ]
    events += [
        {"id": 10, "time": T, "accounts": {"sender", "receiver"}, "laundering": True},
        {"id": 11, "time": T + DAY - 1, "accounts": {"self"}, "laundering": True},
        {"id": 12, "time": T + DAY, "accounts": {"exclusive_end"}, "laundering": True},
        {"id": 13, "time": T + 60, "accounts": {"clean"}, "laundering": False},
    ]
    return {
        "events": events,
        "cuts": [T + 2 * DAY, T + 4 * DAY],
        "start": T - 60 * DAY,
        "end": T + 6 * DAY,
    }


@pytest.mark.parametrize(
    "account,label",
    [
        ("sender", 1),
        ("receiver", 1),
        ("self", 1),
        ("exclusive_end", 0),
        ("past_only", 0),
        ("clean", 0),
    ],
)
def test_inclusive_start_exclusive_end_and_endpoint_or(reference, account, label):
    assert expected_anchor(row(account), reference) == ("train", label)


def test_maturity_ties_and_observation_end(reference):
    assert expected_anchor(row("clean", DAY), reference) == (
        "train",
        0,
    )  # Window ends at split boundary.
    assert expected_anchor(row("clean", 2 * DAY), reference) == (
        "valid",
        0,
    )  # Equal-time anchors go later.
    assert expected_anchor(row("clean", 5 * DAY), reference) == (
        "test",
        0,
    )  # Ends exactly at observation end.
    assert expected_anchor(row("clean", 6 * DAY), reference) == (None, None)
    reference["cuts"][0] = T + DAY - 60
    assert expected_anchor(row("clean"), reference) == (
        None,
        None,
    )  # Split-crossing horizon is not a negative.
    with pytest.raises(AssertionError, match="no prior activity"):
        expected_anchor(row("unseen"), reference)


@pytest.mark.parametrize("damage", ["label", "mask", "split", "clock"])
def test_bad_anchor_is_detected(reference, damage):
    anchor = row("sender", label=1)
    anchor["graph_as_of_time"] = (T - 1) * 10**9
    assert verify_anchors([anchor], reference) == {anchor["sample_id"]: 1}
    bad = deepcopy(anchor)
    if damage == "label":
        bad["involvement"] = 0
    if damage == "mask":
        bad["involvement"] = None
    if damage == "split":
        bad["forecast_split"] = "test"
    if damage == "clock":
        bad["graph_as_of_time"] += 10**9
    with pytest.raises(AssertionError):
        verify_anchors([bad], reference)


def test_reference_bce_known_values_and_exclusion_counts():
    result = binary_evaluation(
        [0.0, 0.0, float("nan"), float("nan")],
        [0, 1, None, 1],
        [False, False, False, True],
    )
    assert result == {
        "loss_sum": 2 * math.log(2),
        "eligible": 2,
        "invalid": 1,
        "padding": 1,
        "mean_loss": math.log(2),
    }
    empty = binary_evaluation([100, -100], [None, 1], [False, True])
    assert (
        empty["eligible"] == 0 and empty["mean_loss"] is None and empty["loss_sum"] == 0
    )
    assert binary_evaluation([-1000, 1000], [1, 0], [False, False])["mean_loss"] == 1000


def test_raw_reader_uses_all_rows_for_quantiles_and_preserves_identity(tmp_path):
    timestamps = [
        "2022/09/01 00:00",
        "2022/09/01 00:00",
        "2022/09/02 00:00",
        "2022/09/02 00:00",
        "2022/09/04 00:00",
        "2022/09/06 00:00",
    ]
    rows = [
        {
            "timestamp": stamp,
            "from_bank": "001" if index == 0 else "2",
            "from_account": "0007",
            "to_bank": "3",
            "to_account": "7",
            "is_laundering": str(index % 2),
        }
        for index, stamp in enumerate(timestamps)
    ]
    path = tmp_path / "transactions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    root = '["hi-small","account","1","0007"]'
    actual = source_reference(path, [root])
    assert actual["source_rows"] == 6
    assert actual["cuts"] == [
        int(datetime(2022, 9, day, tzinfo=timezone.utc).timestamp()) for day in (2, 4)
    ]
    assert [event["id"] for event in actual["events"]] == [0]
    assert actual["events"][0]["sender"] == root
