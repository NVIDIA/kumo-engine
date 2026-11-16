"""Source-to-token oracle shared by AML task checks, without engine encoders.

Fitting and ID-space allocation are covered by package tests. Given their saved
state, independently apply the prototype's vocabulary and numeric-bin rules.
"""

from bisect import bisect_right
from datetime import datetime, timezone
import json

FIELDS = (
    "direction",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
)


def history_tokens(anchor, events, state):
    """Reconstruct a selected AML flat history from source values, before shifting."""
    cutoff = anchor["as_of_time"] // 10**9
    sequence = []
    for index, event in enumerate(events):
        if index:
            sequence.append(state["spaces"]["history"]["roles"]["separator"])
        direction = (
            "self"
            if event["sender"] == event["receiver"]
            else "outgoing"
            if event["sender"] == anchor["entity_id"]
            else "incoming"
        )
        stamp = datetime.fromtimestamp(event["time"], timezone.utc)
        values = {
            **event["values"],
            "direction": direction,
            "age_range": cutoff - event["time"],
            "hour": stamp.hour,
            "weekday": stamp.weekday(),
            "day": stamp.day,
        }
        for field in ("age_range", *FIELDS, "hour", "weekday", "day"):
            assert values[field] is not None, (
                "Prototype flat-token fields reject missing values"
            )
            sequence.append(encoded_value(state, field, values[field]))
    return sequence


def encoded_value(state, field, value):
    binding = json.dumps(["inputs", "history", "fields", field], separators=(",", ":"))
    encoder = state["encoders"][binding]
    if encoder["kind"] == "vocabulary":
        local = (
            encoder["roles"]["missing"]
            if value is None
            else dict(encoder["entries"]).get(value, encoder["roles"]["unknown"])
        )
    else:
        assert encoder["kind"] == "buckets" and not encoder["separate_zero"]
        local = (
            0
            if value is None
            else bisect_right(encoder["boundaries"], float(value)) + 1
        )
    return dict(state["spaces"]["history"]["members"][binding])[local]
