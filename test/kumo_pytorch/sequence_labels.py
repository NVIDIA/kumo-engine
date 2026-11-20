# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent composite-history and auxiliary-label answers from original rows."""

import json
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby

import torch


def scalar(encoder, value):
    if encoder["kind"] == "vocabulary":
        return 0 if value is None else dict(encoder["entries"]).get(value, 1)
    assert encoder["kind"] == "buckets" and not encoder["separate_zero"]
    return 0 if value is None else bisect_right(encoder["boundaries"], float(value)) + 1


def values_at(event, cutoff):
    stamp = datetime.fromtimestamp(event["time"], timezone.utc)
    return {
        **event["values"],
        "age_range": cutoff - event["time"],
        "hour": stamp.hour,
        "weekday": stamp.weekday(),
        "day": stamp.day,
    }


def encoder_for(state, section, name, field):
    matches = [
        identity
        for key, identity, space in state[section][name]["bindings"]
        if key == field and space is None
    ]
    assert len(matches) == 1
    return state["encoders"][matches[0]]


def composite_id(state, values):
    key = []
    for field, identity, space in state["inputs"]["history"]["bindings"]:
        assert space is None
        encoder = state["encoders"][identity]
        value = values[field]
        if value is None:
            dtype = encoder["dtype"]
            # Typed missing is distinct from an encoder's missing numeric ID.
            text = (
                "DType("
                + ", ".join(
                    f"{k}={dtype[k]!r}"
                    for k in ("name", "unit", "timezone", "precision", "scale")
                )
                + ")"
            )
            key.append(("missing", text))
        else:
            key.append(
                (
                    {"vocabulary": "Vocabulary", "buckets": "Buckets"}[encoder["kind"]],
                    scalar(encoder, value),
                )
            )
    entries = {
        tuple(tuple(c) for c in key): value
        for key, value in state["inputs"]["history"]["composite"]
    }
    local = entries.get(tuple(key), 1)
    return dict(
        state["spaces"]["history"]["members"][
            json.dumps(["composite", "history"], separators=(",", ":"))
        ]
    )[local]


def packed_layout(stream, lengths, *, tokens=False):
    offsets = [0]
    for size in lengths:
        offsets.append(offsets[-1] + size)
    assert (
        stream["offsets"].dtype == torch.int64 and stream["offsets"].tolist() == offsets
    ), "Packed sample offsets differ"
    assert (
        stream["cu_seqlens"].dtype == torch.int32
        and stream["cu_seqlens"].tolist() == offsets
    )
    assert stream["max_seqlen"] == max(lengths, default=0)
    if tokens:
        for name in ("value_ids", "position_ids", "source_event_indices"):
            assert stream[name].dtype == torch.int64 and stream[name].shape == (
                offsets[-1],
            )
        assert (
            stream["valid"].dtype == torch.bool
            and stream["valid"].shape == (offsets[-1],)
            and stream["valid"].all()
        )
        assert stream["position_ids"].tolist() == [0] * offsets[-1]
        assert stream["source_event_indices"].tolist() == list(range(offsets[-1]))


def history_checker(binding, anchors, events, state, *, auxiliary=False):
    from .bindings import profile

    options = profile(binding, "input")["auxiliary"]["aligned"]
    threshold = options["threshold"]
    fields = [name for name, _, _ in state["inputs"]["events"]["bindings"]]
    labels = [*options["fields"], threshold["name"]]
    assert state["targets"]["aligned"]["thresholds"] == {
        threshold["name"]: threshold["threshold"]
    }
    for field in labels:
        observed = Counter()
        for anchor in anchors.values():
            if anchor["split"] != "train":
                continue
            for event in events[anchor["entity_id"]]:
                if event["time"] >= anchor["as_of_time"]:
                    continue
                value = event["values"][
                    threshold["field"] if field == threshold["name"] else field
                ]
                if value is not None:
                    label = (
                        ("abnormal" if value > threshold["threshold"] else "normal")
                        if field == threshold["name"]
                        else value
                    )
                    observed[label] += 1
        encoder = encoder_for(state, "targets", "aligned", field)
        expected = {
            value: i + 2
            for i, value in enumerate(sorted(observed, key=lambda v: (-observed[v], v)))
        }
        assert (
            encoder["kind"] == "vocabulary" and dict(encoder["entries"]) == expected
        ), "Auxiliary target vocabulary must fit train histories only"
        assert (
            encoder["roles"] == {"missing": 0, "unknown": 1}
            and encoder["vocabulary_size"] == len(expected) + 2
        )

    def expected(event, cutoff):
        values = values_at(event, cutoff)
        answer = [composite_id(state, values)]
        answer.extend(
            (
                scalar(encoder_for(state, "inputs", "events", f), values[f]),
                values[f] is not None,
            )
            for f in fields
        )
        if auxiliary:
            raw = {f: values[f] for f in options["fields"]}
            value = values[threshold["field"]]
            raw[threshold["name"]] = (
                None
                if value is None
                else "abnormal"
                if value > threshold["threshold"]
                else "normal"
            )
            answer.extend(
                (
                    scalar(encoder_for(state, "targets", "aligned", f), raw[f]),
                    raw[f] is not None,
                )
                for f in labels
            )
        return tuple(answer)

    def check(batch):
        histories = [
            sorted(
                [
                    e
                    for e in events[anchors[s]["entity_id"]]
                    if e["time"] < anchors[s]["as_of_time"]
                ],
                key=lambda e: e["time_ns"],
            )
            for s in batch.metadata["sample_ids"]
        ]
        lengths = list(map(len, histories))
        history, columns = batch.inputs["history"], batch.inputs["events"]
        packed_layout(history, lengths, tokens=True)
        packed_layout(columns, lengths)
        total = sum(lengths)
        assert (
            columns["valid"].dtype == torch.bool
            and columns["valid"].shape == (total,)
            and columns["valid"].all()
        )
        groups = [(columns, fields)]
        if auxiliary:
            packed_layout(batch.targets["aligned"], lengths)
            groups.append((batch.targets["aligned"], labels))
        for group, names in groups:
            for field in names:
                assert (
                    group[field]["values"].dtype == torch.int64
                    and group[field]["valid"].dtype == torch.bool
                )
                assert (
                    group[field]["values"].shape
                    == group[field]["valid"].shape
                    == (total,)
                )
        offset = 0
        for sample, events_ in zip(
            batch.metadata["sample_ids"], histories, strict=True
        ):
            for _, same_time in groupby(events_, key=lambda e: e["time_ns"]):
                tied = list(same_time)
                observed = []
                for i in range(offset, offset + len(tied)):
                    observed.append(
                        (
                            int(history["value_ids"][i]),
                            *(
                                (int(group[f]["values"][i]), bool(group[f]["valid"][i]))
                                for group, names in groups
                                for f in names
                            ),
                        )
                    )
                assert Counter(observed) == Counter(
                    expected(e, anchors[sample]["as_of_time"]) for e in tied
                ), (
                    f"{sample}: composite/field/label membership or chronological alignment differs at position {offset}"
                )
                offset += len(tied)

    return check


def matching_checker(anchors, state):
    roots = sorted({r["entity_id"] for r in anchors.values() if r["split"] == "train"})
    groups = {root: i for i, root in enumerate(roots)}
    assert state["targets"]["matching"]["groups"] == [
        [[["int64", root]], i] for root, i in groups.items()
    ], "Matching groups must fit training identities only"

    def check(batch):
        expected = [
            groups.get(anchors[s]["entity_id"], -1)
            for s in batch.metadata["sample_ids"]
        ]
        target = batch.targets["matching"]
        assert (
            target["values"].dtype == torch.int64
            and target["valid"].dtype == torch.bool
        )
        assert target["values"].shape == target["valid"].shape == (len(expected),)
        assert target["values"].tolist() == expected and target["valid"].tolist() == [
            v >= 0 for v in expected
        ]
        identity = batch.inputs["identity"]["root_id"]
        assert (
            identity["values"].dtype == torch.int64
            and identity["valid"].dtype == torch.bool
        )
        assert identity["valid"].shape == (len(expected),) and identity["valid"].all()
        assert identity["values"].tolist() == [
            anchors[s]["entity_id"] for s in batch.metadata["sample_ids"]
        ]

    return check


def check_allowlist(batch, state, fields, withheld):
    assert set(batch.inputs) == {"history", "events", "identity"}
    assert set(batch.inputs["history"]) == {
        "value_ids",
        "position_ids",
        "source_event_indices",
        "valid",
        "offsets",
        "cu_seqlens",
        "max_seqlen",
    }
    assert set(batch.inputs["events"]) == set(fields) | {
        "valid",
        "offsets",
        "cu_seqlens",
        "max_seqlen",
    }
    assert set(batch.inputs["identity"]) == {"root_id"}
    for name in ("history", "events"):
        actual = {f for f, _, _ in state["inputs"][name]["bindings"]}
        assert actual == set(fields) and not actual & set(withheld), (
            "Withheld target source exposed as input"
        )
