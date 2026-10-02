# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from diskgraph import DiskGraphTTBuilder, EdgeRef, RuntimeIndex, SplitSpec, execute_tt_plan
from diskgraph._c import DiskgraphConfigError, DiskgraphDataError
from test_ttng import _build_parent_index


def test_runtime_index_infers_and_overrides_generated_fact_cutoffs(tmp_path):
    parent = _build_parent_index(tmp_path)
    builder = DiskGraphTTBuilder()
    builder.link_prediction(include_frequency=False)
    builder.for_each("users", "user_id")
    builder.set_target_stype("ID")
    builder.aggregate("links", EdgeRef("users", "user_id", "orders", "user_id"),
                      "list_distinct", target_col="order_id")
    builder.split_ratio(SplitSpec(kind="RandomSplit", ratio=[0.5, 0.25, 0.25], rnd_seed=17))
    plan = json.loads(builder.build())
    plan["fact_pseudo_time"] = {
        "table": "orders", "entity_col": "user_id", "target_col": "order_id",
    }
    plan["lp_resolution"] = [{
        "src_col": "ENTITY1", "dst_table": "users", "kind": "scalar",
        "node_id_col": "entity_node_id",
    }]
    result = execute_tt_plan(json.dumps(plan), str(parent), "pseudo")
    task = result["task_dataset"]
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_text(json.dumps({
        "kind": "diskgraph_task_dataset", "graph": {"index_path": str(parent)},
        "stages": task["stages"], "task": {"is_temporal": False},
    }))
    extension = parent / "train_tables" / "pseudo"
    manifest_path = extension / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 4
    sidecar = (extension / "fact_pseudo_time.bin").read_bytes()[24:]
    assert set(sidecar) == {1, 2, 3}
    index = RuntimeIndex(str(runtime_manifest), rnd_seed=17, num_threads=1)
    fan = {"fanout": 100, "mode": "uniform", "absent_key_policy": "uniform_fallback"}
    sampling = {"layers": [{"fanouts": {}, "default_fanout": fan}] * 4}

    def sampled(stage, **options):
        batches, _ = index.iter(
            stage=stage, sampling_plan=sampling, num_workers=1, worker_id=0,
            shuffle=False, epoch=0, batch_size=32, entity_node_type="users",
            **options,
        )
        return {
            node for batch in batches
            for node in (batch["graphs"]["main"]["node_ids"]["orders"].to_pylist()
                         if "orders" in batch["graphs"]["main"]["node_ids"] else [])
        }

    for stage, cutoff in [("train", 1), ("val", 1), ("test", 2)]:
        full = sampled(stage, disable_pseudo_time=True)
        assert full
        assert sampled(stage) == {node for node in full if sidecar[node] <= cutoff}
        assert sampled(stage, pseudo_seed_time=3) == full
    with pytest.raises(DiskgraphConfigError, match="cannot be combined"):
        sampled("train", pseudo_seed_time=1, disable_pseudo_time=True)
    for invalid in [-1, 4]:
        with pytest.raises(DiskgraphConfigError, match="between 0 and 3"):
            sampled("train", pseudo_seed_time=invalid)

    manifest.pop("fact_pseudo_time")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DiskgraphDataError, match="version 4 requires fact_pseudo_time"):
        sampled("train")


@pytest.mark.parametrize("metadata", [None, {"version": -1, "future_field": 123}])
def test_runtime_index_accepts_extensions_without_fact_metadata(tmp_path, metadata):
    from test_ttng import _build_tt_plan

    parent = _build_parent_index(tmp_path)
    task = execute_tt_plan(_build_tt_plan(), str(parent), "legacy")["task_dataset"]
    metadata_path = parent / "train_tables" / "legacy" / "manifest.json"
    if metadata is None:
        metadata_path.unlink()
    else:
        metadata_path.write_text(json.dumps(metadata))
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_text(json.dumps({
        "kind": "diskgraph_task_dataset", "graph": {"index_path": str(parent)},
        "stages": task["stages"],
    }))
    index = RuntimeIndex(str(runtime_manifest), num_threads=1)
    batches, size = index.iter(
        stage="train", sampling_plan={"layers": []}, num_workers=1,
        worker_id=0, shuffle=False, epoch=0, batch_size=32, entity_node_type="users",
    )
    assert sum(len(batch["task"]["input_ids"]) for batch in batches) == size > 0
