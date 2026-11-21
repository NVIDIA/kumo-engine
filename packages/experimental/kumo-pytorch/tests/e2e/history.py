# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native graphs, histories and aggregates with answers from source rows."""

from bisect import bisect_right
from collections import Counter
from datetime import UTC, datetime
from itertools import accumulate, groupby, permutations
from operator import itemgetter
from statistics import mean, pstdev

import pytest
import torch

from kumo.pytorch import api
from tests.test_inference_lifecycle import assert_same


def check_source_graph(task, batch, expected, layouts, *, root_table="roots"):
    """Compare one-hop root graphs with original rows keyed by owner, type and physical ID."""
    roots = list(batch.metadata["entity_ids"])
    graph = batch.inputs["graph"]
    nodes, edges = graph["nodes"], graph["edges"]
    keys = {}
    for family, node in nodes.items():
        ids, owners = node["ids"], node["batch"]
        assert ids.dtype == owners.dtype == torch.int64 and ids.ndim == owners.ndim == 1
        keys[family] = list(zip(owners.tolist(), ids.tolist(), strict=True))
    root_keys = keys.pop(root_table)
    assert not nodes[root_table]["features"] and sorted(root_keys) == list(enumerate(roots))
    assert sorted((o, family, i) for family, rows in keys.items() for o, i in rows) == sorted(
        expected
    )
    assert set(edges) == {layouts[family][0] for family in keys}
    for family, members in keys.items():
        relation, configs = layouts[family]
        edge = edges[relation]
        assert (edge["source_type"], edge["destination_type"]) == (root_table, family)
        index = edge["index"]
        assert (
            index.dtype == torch.int64 and index.shape == (2, len(members)) and (index >= 0).all()
        )
        assert sorted((root_keys[a], members[b]) for a, b in index.T.tolist()) == sorted(
            ((owner, roots[owner]), (owner, i)) for owner, i in members
        )
        features = nodes[family]["features"]
        bindings = task.output("graph").bindings[family]
        assert {name: binding.encoder.config for name, binding in bindings.items()} == configs
        assert set(features) == set(configs)
        for name, binding in bindings.items():
            encoder = binding.encoder
            raw = [expected[owner, family, i][name] for owner, i in members]
            valid = torch.tensor([v is not None for v in raw], dtype=torch.bool)
            assert_same(features[name]["valid"], valid)
            values = [v for v in raw if v is not None]
            dtype = torch.int64
            if isinstance(encoder.config, api.ZScore):
                values = [(v - encoder.mean) / encoder.scale for v in values]
                dtype = torch.float32
            elif isinstance(encoder.config, api.Vocabulary):
                lookup = dict(encoder.entries)
                values = [lookup.get(v, encoder.config.unknown_id) for v in values]
            assert_same(features[name]["values"][valid], torch.tensor(values, dtype=dtype))


def check_normalized_aggregates(task, batch, answers, training, *, raw_only=()):
    samples = batch.metadata["sample_ids"]
    actual = batch.inputs["aggregates"]
    assert actual.keys() == answers[samples[0]].keys() - set(raw_only)
    for name, field in actual.items():
        train = [v for s in training if (v := answers[s][name]) is not None]
        center, scale = (mean(train), pstdev(train) or 1.0) if train else (0.0, 1.0)
        encoder = task.output("aggregates").field(name=name).encoder
        assert encoder.config == api.ZScore()
        assert (encoder.mean, encoder.scale) == pytest.approx((center, scale))
        values = [answers[s][name] for s in samples]
        valid = torch.tensor([v is not None for v in values], dtype=torch.bool)
        assert_same(field["valid"], valid)
        expected = torch.tensor(
            [(v - center) / scale for v in values if v is not None], dtype=torch.float32
        )
        torch.testing.assert_close(field["values"][valid], expected, rtol=2e-6, atol=2e-6)


def _scalar_id(encoder, value):
    if value is None:
        return 0
    if isinstance(encoder.config, api.Buckets):
        return bisect_right(encoder.boundaries, float(value)) + 1
    return dict(encoder.entries).get(value, encoder.config.unknown_id)


def scalar_token(encoder, space, value):
    assert value is not None
    return space.members[encoder.identity][_scalar_id(encoder, value)]


def _history_fields(fields):
    fields = {"age_range": api.Buckets(bins=64), **fields}
    for name, values in (("hour", range(24)), ("weekday", range(7)), ("day", range(1, 32))):
        fields[name] = api.Vocabulary(values=tuple((i, i + 2) for i in values))
    return fields


def _event_values(cutoff, seconds, fields):
    stamp = datetime.fromtimestamp(seconds, UTC)
    return dict(
        fields, age_range=cutoff - seconds, hour=stamp.hour, weekday=stamp.weekday(), day=stamp.day
    )


def check_scalar_history(task, batch, fields, histories):
    fields = _history_fields(fields)
    bindings = task.output("history").bindings
    assert {name: binding.encoder.config for name, binding in bindings.items()} == fields
    space = task.spaces["history"]

    stream = batch.inputs["history"]
    valid = stream["valid"]
    assert valid.dtype == torch.bool and valid.ndim == 2 and valid.shape[0] == len(histories)
    assert set(stream) == {"value_ids", "source_event_indices", "position_ids", "valid"}
    for name in ("value_ids", "source_event_indices", "position_ids"):
        assert stream[name].dtype == torch.int64 and stream[name].shape == valid.shape
    target = batch.targets.get("next_token")
    if target is not None:
        assert_same(target["valid"], valid)
        assert target["values"].dtype == torch.int64 and target["values"].shape == valid.shape
        assert (target["values"][~valid] == -100).all()

    offset = 0
    for owner, (cutoff, history) in enumerate(histories):
        expected = []
        for seconds, fields_from_source in history:
            values = _event_values(cutoff, seconds, fields_from_source)
            tokens = tuple(
                scalar_token(bindings[n].encoder, space, values[n]) for n in fields if n in values
            )
            expected.append((seconds, tokens))
        mask = valid[owner]
        size = max(0, sum(len(record) + 1 for _, record in expected) - 1 - int(target is not None))
        assert_same(mask, torch.arange(mask.numel()) < size)
        assert not stream["position_ids"][owner][mask].any()
        tokens = stream["value_ids"][owner]
        assert (tokens[~mask] == space.special(role="pad")).all()
        ids = tokens[mask].tolist()
        if target is not None:
            labels = target["values"][owner][mask].tolist()
            assert labels[:-1] == ids[1:]
            ids.extend(labels[-1:])

        records = [[]] if ids else []
        for value in ids:
            if value == space.special(role="separator"):
                records.append([])
            else:
                records[-1].append(value)
        # Compare whole events within each timestamp, preserving duplicates and chronology.
        times = sorted(seconds for seconds, _ in expected)
        assert len(times) == len(records)
        assert sorted(zip(times, map(tuple, records), strict=True)) == sorted(expected)
        sources = []
        for index, record in enumerate(records, offset):
            if sources:
                sources.append(-1)
            sources.extend([index] * len(record))
        assert stream["source_event_indices"][owner][mask].tolist() == sources[:size]
        offset += len(history)


def source_vocabulary(values):
    counts = Counter(value for value in values if value is not None)
    return {key: i for i, key in enumerate(sorted(counts, key=lambda k: (-counts[k], k)), 2)}


def check_masked_history(task, batch, fields, histories, *, family=None, text_pieces=None):
    bindings, space = task.output("history").bindings, task.spaces["history"]
    assert {name: b.encoder.config for name, b in bindings.items()} == fields
    field_ids = {
        (source, name): space.members['["fields","fields"]'][i]
        for (_, source, name), i in task.fitted_state["field_names"]["fields"]
    }

    def source_record(row):
        record = []
        for name in fields:
            if (value := row.get(name)) is None:
                continue
            key = field_ids[row.get("source_kind", family), name]
            pieces = (
                text_pieces[name][value]
                if text_pieces and name in text_pieces
                else [scalar_token(bindings[name].encoder, space, value)]
            )
            record.extend((key, position, token) for position, token in enumerate(pieces))
        return record

    packed, target = batch.inputs["history"], batch.targets["masked_values"]
    after, changed = packed["value_ids"], target["corruption"]
    masked, unknown = after == space.special(role="mask"), space.special(role="unknown")
    assert_same(target["valid"], masked)
    assert (target["values"][~masked] == -100).all()
    assert changed[masked].all() and (after[changed & ~masked] == unknown).all()
    assert packed["valid"].all() and packed["field_valid"].all()
    offsets = packed["field_offsets"].tolist()
    assert offsets[0] == 0 and offsets[-1] == len(after)
    assert_same(target["offsets"], packed["field_offsets"])
    visible = torch.where(masked, target["values"], after)
    tokens = list(
        zip(
            packed["field_ids"].tolist(),
            packed["position_ids"].tolist(),
            visible.tolist(),
            (changed & ~masked).tolist(),
            strict=True,
        )
    )

    def matches(record, index):
        actual = tokens[offsets[index] : offsets[index + 1]]
        return len(record) == len(actual) and all(
            (field, pos) == (key, position)
            and (original != unknown if replaced else original == value)
            for (key, position, original), (field, pos, value, replaced) in zip(
                record, actual, strict=True
            )
        )

    seq = packed["sequence"]
    valid = seq["valid"]
    assert valid.shape == (len(histories), max(len(rows) for _, rows in histories))
    owners, positions = valid.nonzero(as_tuple=True)
    assert_same(packed["owner"], owners)
    assert_same(packed["history_position"], positions)
    assert_same(seq["event_indices"][valid], torch.arange(len(owners)))
    assert len(offsets) == len(owners) + 1
    event = 0
    for owner, (cutoff, history) in enumerate(histories):
        ordered = sorted(history, key=itemgetter(0))
        assert_same(valid[owner], torch.arange(valid.shape[1]) < len(ordered))
        if ordered:
            assert masked[offsets[event] : offsets[event + len(ordered)]].any()
        calendar = [_event_values(cutoff, stamp, {}) for stamp, _ in ordered]
        for name in ("hour", "weekday", "day", "age_seconds"):
            age = name == "age_seconds"
            channel = batch.inputs["history_timing" if age else "history_calendar"]
            assert_same(channel["valid"], valid)
            assert_same(channel[name]["valid"], valid)
            assert channel[name]["values"][owner][valid[owner]].tolist() == [
                row["age_range" if age else name] for row in calendar
            ]
        # Both pinned extracts have at most two tied events. Either order is valid after masking.
        for stamp, tied in groupby(ordered, key=itemgetter(0)):
            records = [source_record(row) for _, row in tied]
            assert any(
                all(matches(r, i) for i, r in enumerate(order, event))
                for order in permutations(records)
            ), (owner, stamp)
            event += len(records)
    return field_ids


def check_mbd_packed_history(task, batch, registered, events, fields):
    def labels(row):
        amount = row["amount"]
        positive = None if amount is None else ("abnormal" if amount > 0 else "normal")
        return row["event_type"], row["event_subtype"], positive

    samples = batch.metadata["sample_ids"]
    histories = {
        sample: sorted(event for event in events[row["entity_id"]] if event[0] < row["as_of_time"])
        for sample, row in registered.items()
    }
    heads = ("event_type", "event_subtype", "positive_amount")
    training = [
        labels(row)
        for s, h in histories.items()
        if registered[s]["split"] == "train"
        for _, _, row in h
    ]
    aux = [source_vocabulary(row[i] for row in training) for i in range(len(heads))]
    bindings = task.target("aligned").bindings
    assert set(bindings) == set(heads)
    for name, vocab in zip(heads, aux, strict=True):
        encoder = bindings[name].encoder
        assert encoder.config == api.Vocabulary() and dict(encoder.entries) == vocab
    state = task.fitted_state.to_dict()
    assert state["targets"]["aligned"]["thresholds"] == {"positive_amount": 0.0}

    withheld = {"event_type", "event_subtype", "amount", "event_time_seconds"}
    fields = _history_fields({k: v for k, v in fields.items() if k not in withheld})
    encoders = task.output("history").bindings
    event_encoders = task.output("events").bindings
    for bindings in (encoders, event_encoders):
        assert {k: b.encoder.config for k, b in bindings.items()} == fields
    composite = {tuple(map(tuple, key)): i for key, i in state["inputs"]["history"]["composite"]}
    space = task.spaces["history"].members['["composite","history"]']

    def record(cutoff, row):
        values = _event_values(cutoff, row["event_time_seconds"], row)
        scalars, key = [], []
        for name, binding in encoders.items():
            encoder = binding.encoder
            value = values[name]
            scalars.append((_scalar_id(event_encoders[name].encoder, value), value is not None))
            key.append(
                ("missing", str(encoder.dtype))
                if value is None
                else (type(encoder.config).__name__, _scalar_id(encoder, value))
            )
        aligned = [
            (0 if v is None else vocab.get(v, 1), v is not None)
            for v, vocab in zip(labels(row), aux, strict=True)
        ]
        return (space[composite.get(tuple(key), 1)], *scalars, *aligned)

    lengths = [len(histories[s]) for s in samples]
    offsets = torch.tensor(list(accumulate(lengths, initial=0)))
    size = sum(lengths)
    layout = dict(offsets=offsets, cu_seqlens=offsets.int(), max_seqlen=max(lengths, default=0))
    valid = torch.ones(size, dtype=torch.bool)
    assert set(batch.inputs) == {"history", "events", "identity"}
    history, columns = batch.inputs["history"], batch.inputs["events"]
    aligned = batch.targets["aligned"]
    tokens = history["value_ids"]
    assert tokens.dtype == torch.int64 and tokens.shape == (size,)
    indices = dict(position_ids=torch.zeros_like(tokens), source_event_indices=torch.arange(size))
    assert_same(history, dict(layout, **indices, valid=valid, value_ids=tokens))
    assert_same(columns["valid"], valid)
    packed = [tokens.tolist()]
    for stream, names, extras in ((columns, encoders, {"valid"}), (aligned, heads, set())):
        assert set(stream) == layout.keys() | set(names) | extras
        assert_same({k: stream[k] for k in layout}, layout)
        for name in names:
            values, mask = stream[name]["values"], stream[name]["valid"]
            assert values.dtype == torch.int64 and mask.dtype == torch.bool
            assert values.shape == mask.shape == (size,)
            packed.append(zip(values.tolist(), mask.tolist(), strict=True))
    actual = iter(zip(*packed, strict=True))
    for sample in samples:
        cutoff = registered[sample]["as_of_time"] // 10**9
        for ns, tied in groupby(histories[sample], key=lambda event: event[0]):
            expected = [record(cutoff, row) for _, _, row in tied]
            # Only complete events sharing an original nanosecond can exchange positions.
            assert sorted(next(actual) for _ in expected) == sorted(expected), (sample, ns)
