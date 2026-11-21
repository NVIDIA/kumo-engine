# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""History and event merges accompany native runtime iteration."""

import json
from pathlib import Path
from typing import cast

import pytest
from diskgraph import DiskgraphConfigError, RuntimeIndex, SamplerEngine
from diskgraph.sampling_options import SamplingOptions
from test_runtime_batch_iterator_e2e import (
    _sampling_plan,
)
from test_runtime_batch_iterator_e2e import (
    seed_table_fixture as seed_table_fixture,  # noqa: PLC0414
)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("temporal", [False, True])
def test_runtime_history_uses_batch_anchors(
    seed_table_fixture: dict[str, str | int],
    tmp_path: Path,
    batch_size: int,
    temporal: bool,
) -> None:
    """Match each history row and merge index to its batch seed and cutoff."""
    sampler = SamplerEngine(cast(str, seed_table_fixture["parent_root"]))
    relation = next(
        relation["name"]
        for relation in json.loads(sampler.get_type_mapping())["edge_relations"]
        if relation["src_type"] == "users"
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
    manifest = json.loads(
        Path(cast(str, seed_table_fixture["manifest_path"])).read_text()
    )
    manifest["task"] = {"is_temporal": temporal}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    index = RuntimeIndex(str(manifest_path), num_threads=1)
    iterator, _ = index.iter(
        SamplingOptions(
            stage="train",
            sampling_plan=_sampling_plan(),
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=batch_size,
            entity_node_type="users",
        )
    )
    iterator.with_neighborhoods(
        json.dumps(
            {
                "context": {
                    "plan": {
                        "layers": [
                            {
                                "fanouts": {},
                                "default_fanout": {
                                    "fanout": 4,
                                    "mode": "uniform",
                                    "absent_key_policy": "uniform_fallback",
                                },
                            }
                        ]
                        * 2
                    },
                    "projections": {},
                },
            }
        )
    )
    iterator.with_history(
        sampler,
        "users",
        json.dumps(plan),
        json.dumps(
            [
                {
                    "name": "events",
                    "sequences": ["orders"],
                    "ascending": [True],
                    "maximum": 1,
                    "keep_last": True,
                    "descending": False,
                }
            ]
        ),
    )

    batches = list(iterator)

    assert sum(len(batch.task.seed_node_ids) for batch in batches) == 6
    for batch in batches:
        assert "context" in batch.graphs
        assert isinstance(batch.history, dict)
        sequence = batch.history["sequences"][0]
        seeds = batch.task.seed_node_ids.to_pylist()
        offsets, order = batch.history_merges["events"]
        if temporal:
            assert sequence["offsets"].to_pylist() == list(
                range(0, 2 * len(seeds) + 1, 2)
            )
            assert sequence["batch"].column("order_id").to_pylist() == [
                seed * 4 + event for seed in seeds for event in (0, 1)
            ]
            assert offsets.to_pylist() == list(range(len(seeds) + 1))
            assert order.to_pylist() == list(range(1, 2 * len(seeds), 2))
        else:
            assert sequence["offsets"].to_pylist() == [0] * (len(seeds) + 1)
            assert offsets.to_pylist() == [0] * (len(seeds) + 1)
            assert order.to_pylist() == []


def test_runtime_history_rejects_invalid_merge_configuration(
    seed_table_fixture: dict[str, str | int],
) -> None:
    """Reject unknown sequence names and mismatched sort orders at setup."""
    sampler = SamplerEngine(cast(str, seed_table_fixture["parent_root"]))
    index = RuntimeIndex(cast(str, seed_table_fixture["manifest_path"]))
    iterator, _ = index.iter(
        SamplingOptions(
            stage="train",
            sampling_plan=_sampling_plan(),
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=3,
            entity_node_type="users",
        )
    )
    plan = json.dumps({"frames": [], "sequences": [], "aggregates": []})
    for ascending, message in [([], "counts differ"), ([True], "missing")]:
        with pytest.raises(DiskgraphConfigError, match=message):
            iterator.with_history(
                sampler,
                "users",
                plan,
                json.dumps(
                    [
                        {
                            "name": "events",
                            "sequences": ["missing"],
                            "ascending": ascending,
                            "maximum": 1,
                            "keep_last": True,
                            "descending": False,
                        }
                    ]
                ),
            )
