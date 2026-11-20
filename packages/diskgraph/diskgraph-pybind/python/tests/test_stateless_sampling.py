# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native batch requests share local iterator sampling semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pytest
from diskgraph import (
    DiskgraphConfigError,
    DiskgraphDataError,
    RuntimeIndex,
    SamplerEngine,
)
from diskgraph._c import SeedSchedule
from test_runtime_batch_iterator_e2e import (
    _sampling_plan,
    seed_table_fixture,  # noqa: F401 -- shared native ingest fixture
)


def _assert_equal(actual: object, expected: object) -> None:
    if isinstance(actual, (pa.Array, pa.ChunkedArray, pa.RecordBatch, pa.Table)):
        assert actual.equals(expected)
    elif isinstance(actual, Mapping):
        assert isinstance(expected, Mapping)
        assert actual.keys() == expected.keys()
        for name in actual:
            if name == "timings_ns":
                durations, reference_durations = actual[name], expected[name]
                assert durations.keys() == reference_durations.keys()
                assert all(
                    isinstance(value, int) and value >= 0
                    for value in (*durations.values(), *reference_durations.values())
                )
                continue
            _assert_equal(actual[name], expected[name])
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple))
        assert len(actual) == len(expected)
        if actual and isinstance(actual[0], Mapping) and "edge_type" in actual[0]:
            actual = sorted(
                actual, key=lambda entry: tuple(entry["edge_type"].values())
            )
            expected = sorted(
                expected, key=lambda entry: tuple(entry["edge_type"].values())
            )
        for value, reference in zip(actual, expected, strict=True):
            _assert_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("extras", ["none", "neighborhood", "history"])
def test_stateless_requests_replay_without_advancing_local_cursor(
    seed_table_fixture: dict[str, Any],  # noqa: F811 -- imported pytest fixture
    extras: str,
) -> None:
    """Explicit requests preserve graph/history outputs and the iterator cursor."""
    index = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=2
    )
    options = {
        "stage": "train",
        "sampling_plan": _sampling_plan(),
        "entity_node_type": "users",
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "epoch": 0,
        "batch_size": 2,
    }
    iterator, _ = index.iter(**options)
    reference, _ = index.iter(**options)
    for current in (iterator, reference):
        if extras == "neighborhood":
            current.with_neighborhoods(
                json.dumps({"context": {"plan": _sampling_plan(), "projections": {}}})
            )
        elif extras == "history":
            sampler = SamplerEngine(seed_table_fixture["parent_root"])
            relations = json.loads(sampler.get_type_mapping())["edge_relations"]
            relation = next(
                item["name"] for item in relations if item["src_type"] == "users"
            )
            plan = {
                "frames": [
                    {
                        "name": "recent",
                        "root_type": "users",
                        "path": [{"relation": relation, "destination_type": "orders"}],
                        "time_column": "order_time",
                        "start_seconds": -2000,
                        "stop_seconds": 0,
                        "distinct": True,
                        "columns": ["order_id"],
                    }
                ],
                "sequences": [
                    {
                        "name": "orders",
                        "frame": "recent",
                        "columns": ["order_id"],
                        "order": "ascending",
                        "limit": 4,
                    }
                ],
                "aggregates": [],
            }
            current.with_history(sampler, "users", json.dumps(plan), "[]")
    first = next(iterator)
    if extras == "neighborhood":
        assert "context" in first["graphs"]
    elif extras == "history":
        assert "history" in first
    seeds = first["task"]["tt_stage_node_ids"].to_pylist()
    replay = iterator.sample_request(seeds)
    _assert_equal(replay, first)
    _assert_equal(iterator.sample_request(seeds), first)
    with pytest.raises(DiskgraphConfigError, match="seed count"):
        iterator.sample_request([])
    with pytest.raises(DiskgraphConfigError, match="seed count"):
        iterator.sample_request([0, 1, 2])
    with pytest.raises(DiskgraphDataError, match="outside stage"):
        iterator.sample_request([2**63])
    second = next(iterator)
    assert second["task"]["tt_stage_node_ids"].to_pylist() != seeds
    next(reference)
    _assert_equal(second, next(reference))
    list(iterator)
    _assert_equal(iterator.sample_request(seeds), first)


@pytest.mark.parametrize("mode", ["flat", "selected", "rhs"])
def test_seed_only_schedule_matches_native_iterator(
    seed_table_fixture: dict[str, Any],  # noqa: F811 -- imported pytest fixture
    mode: str,
) -> None:
    """Standalone native scheduling preserves the iterator batch membership."""
    index = RuntimeIndex(
        seed_table_fixture["manifest_path"], rnd_seed=42, num_threads=2
    )
    options: dict[str, Any] = {
        "stage": "train",
        "sampling_plan": _sampling_plan(),
        "entity_node_type": "users",
        "num_workers": 2,
        "worker_id": 1,
        "shuffle": mode == "flat",
        "epoch": 3,
        "batch_size": 2,
        "rnd_seed": 42,
    }
    if mode == "selected":
        options["selected_stage_node_ids"] = [4, 1, 4, 2, 0]
    elif mode == "rhs":
        options.update(rhs_seed_node_type="users", rhs_seed_time=1000)
    size = seed_table_fixture["n_users" if mode == "rhs" else "n_seeds"]
    config = {
        "stage_key": "users" if mode == "rhs" else "train_seeds",
        "ranges": [[0, size - 1]],
        **{
            name: options[name]
            for name in (
                "num_workers",
                "worker_id",
                "shuffle",
                "rnd_seed",
                "epoch",
                "batch_size",
            )
        },
        "drop_last": False,
        "rhs": mode == "rhs",
        "selected_stage_node_ids": options.get("selected_stage_node_ids"),
    }
    schedule = SeedSchedule(json.dumps(config))
    count = schedule.num_samples()
    iterator, expected_count = index.iter(**options)
    assert count == expected_count
    for batch in iterator:
        field = "seed_node_ids" if mode == "rhs" else "tt_stage_node_ids"
        seed_ids = batch["task"][field].to_pylist()
        assert schedule.next_seed_ids() == seed_ids
        _assert_equal(iterator.sample_request(seed_ids), batch)
    assert schedule.next_seed_ids() is None


def test_seed_only_schedule_needs_no_graph_index() -> None:
    """Metadata alone is enough to enumerate canonical seed ids."""
    config = {
        "stage_key": "train",
        "ranges": [[10, 12], [20, 21]],
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "rnd_seed": None,
        "epoch": 0,
        "batch_size": 2,
        "drop_last": False,
        "rhs": False,
    }
    schedule = SeedSchedule(json.dumps(config))
    assert schedule.num_samples() == 5
    assert [schedule.next_seed_ids() for _ in range(4)] == [
        [10, 11],
        [12, 20],
        [21],
        None,
    ]
