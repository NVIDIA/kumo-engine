# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native correctness campaigns; see ../TESTING.md for resource controls."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from _correctness import (
    assert_oracle,
    assert_wire_roundtrip,
    instant,
    make_case,
    prepare,
)
from _fixture import LARGE, build_index, when
from kumo.pytorch import api

from kumo import fm

pytestmark = pytest.mark.dev


def size(name, default):
    value = int(os.environ.get(name, default))
    assert value > 0, f"{name} must be positive"
    return value


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("seed", [391, 2026, 73013])
@pytest.mark.parametrize("string_keys", [False, True])
def test_randomized_high_fanout_against_python_oracle(tmp_path, seed, string_keys):
    engine, keys, events = make_case(
        tmp_path, seed=seed, string_keys=string_keys,
        entities=max(4, size("FM_DEV_ENTITIES", 24)),
        events_per_entity=size("FM_DEV_EVENTS", 1000))
    before = hashes(engine.index)
    context = [(key, 5000, i - 2**53) for i, key in enumerate(keys + keys[:3])]
    query = [(key, 20000, i + 2**53) for i, key in enumerate(reversed(keys))]
    retained = None
    for batch_size in (1, 7, 4096):
        from _correctness import sample_for

        with prepare(engine, sample=sample_for(fanout=50), batch_size=batch_size,
                     budgets={"max_payload_bytes": 256 * 1024**2,
                              "max_decoded_arrow_bytes": 256 * 1024**2}) as loader:
            directory = Path(loader._temporary.name)
            batch = loader.load(
                context=fm.Context(pairs=[(k, instant(t), i) for k, t, i in context]),
                query=fm.Query(pairs=[(k, instant(t), i) for k, t, i in query]))
        assert not directory.exists()
        assert_oracle(batch, events, {"context": context, "predict": query}, fanout=50)
        assert_wire_roundtrip(batch)
        if retained is not None:
            assert batch.encode() == retained
        retained = batch.encode()
    assert hashes(engine.index) == before


def test_four_hop_many_occurrences_and_reopened_loaders(tmp_path):
    index, _ = build_index(tmp_path)
    engine = api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=index, id="dev"), num_threads=2)
    before = hashes(index)
    names = ("order_user", "payment_order", "audit_payment", "note_audit")
    target = api.Aggregate(api.Rows(path=(names[0],), time=api.TimeWindow("at", start=0, stop=7*86400)), "count")
    sample = api.SampleSpec(entity=api.EntitySpec("users"), targets={"future": target},
                            neighborhoods={"history": api.NeighborhoodSpec(tuple(
                                api.GraphLayer((api.GraphSample(name, fanout=2, direction="reverse",
                                                               strategy="recent"),)) for name in names))})
    count = size("FM_DEV_OCCURRENCES", 512)
    context = fm.Context(pairs=[(LARGE if i % 2 == 0 else -5, when(10), i) for i in range(count)])
    query = fm.Query(pairs=[(LARGE if i % 2 == 0 else -5, when(25), i) for i in range(count)])
    expected = {
        "orders": ("order_id", "order_user", [11, 12], [13], [LARGE, LARGE], [-5]),
        "payments": ("payment_id", "payment_order", [21, 22], [23], [11, 12], [13]),
        "audits": ("audit_id", "audit_payment", [31, 32], [33], [21, 22], [23]),
        "notes": ("note_id", "note_audit", [41, 42], [43], [31, 32], [33]),
    }
    for _ in range(size("FM_DEV_REPEATS", 3)):
        with prepare(engine, sample=sample, features={name: [] for name in
                     ("users", "orders", "payments", "audits", "notes")}, batch_size=127,
                     budgets={"max_instances": max(count, 10000),
                              "max_payload_bytes": 256 * 1024**2,
                              "max_decoded_arrow_bytes": 256 * 1024**2}) as loader:
            batch = loader.load(context=context, query=query)
            # Query-free context would permit unobservable future labels.
            assert batch.tables("context")["instance_table"]["__fm_target"].to_pylist() == [
                1.0 if i % 2 == 0 else 0.0 for i in range(count)]
        for split in batch.splits:
            for table, (key, relation, even, odd, even_parents, odd_parents) in expected.items():
                actual = batch.tables(split)["related_tables"][table].to_pylist()
                selected_even = [12, 14] if split == "predict" and table == "orders" else even
                parents_even = [LARGE, LARGE] if table == "orders" else even_parents
                if split == "predict" and table != "orders":
                    selected_even, parents_even = even[1:], even_parents[1:]
                assert [(r["instance_id"], r[key], r["__fm_edge_" + relation]) for r in actual] == [
                    (i, child, parent) for i in range(count)
                    for child, parent in zip(selected_even if i % 2 == 0 else odd,
                                             parents_even if i % 2 == 0 else odd_parents, strict=True)]
        assert_wire_roundtrip(batch)
    assert hashes(index) == before
