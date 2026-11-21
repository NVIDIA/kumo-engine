# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native graphs, histories and aggregates with answers from source rows."""

from bisect import bisect_right
from datetime import UTC, datetime
from statistics import mean, pstdev

import pytest
import torch

from kumo.pytorch import api
from tests.test_inference_lifecycle import assert_same


def check_source_graph(task, batch, expected, layouts):
    """Compare one-hop root graphs with original rows keyed by owner, type and physical ID."""
    roots = list(batch.metadata["entity_ids"])
    graph = batch.inputs["graph"]
    nodes, edges = graph["nodes"], graph["edges"]
    keys = {}
    for family, node in nodes.items():
        ids, owners = node["ids"], node["batch"]
        assert ids.dtype == owners.dtype == torch.int64 and ids.ndim == owners.ndim == 1
        keys[family] = list(zip(owners.tolist(), ids.tolist(), strict=True))
    root_keys = keys.pop("roots")
    assert not nodes["roots"]["features"] and sorted(root_keys) == list(enumerate(roots))
    assert sorted((o, family, i) for family, rows in keys.items() for o, i in rows) == sorted(
        expected
    )
    assert set(edges) == {layouts[family][0] for family in keys}
    for family, members in keys.items():
        relation, configs = layouts[family]
        edge = edges[relation]
        assert (edge["source_type"], edge["destination_type"]) == ("roots", family)
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


def scalar_token(encoder, space, value):
    assert value is not None
    if isinstance(encoder.config, api.Buckets):
        local = bisect_right(encoder.boundaries, float(value)) + 1
    else:
        local = dict(encoder.entries).get(value, encoder.config.unknown_id)
    return space.members[encoder.identity][local]


def check_scalar_history(task, batch, fields, histories):
    fields = {"age_range": api.Buckets(bins=64), **fields}
    for name, values in (("hour", range(24)), ("weekday", range(7)), ("day", range(1, 32))):
        fields[name] = api.Vocabulary(values=tuple((i, i + 2) for i in values))
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
            stamp = datetime.fromtimestamp(seconds, UTC)
            values = dict(
                fields_from_source,
                age_range=cutoff - seconds,
                hour=stamp.hour,
                weekday=stamp.weekday(),
                day=stamp.day,
            )
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
