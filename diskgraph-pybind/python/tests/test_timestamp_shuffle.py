# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from itertools import islice

import pytest

from diskgraph import DiskGraphTTBuilder, EdgeRef, RuntimeIndex, SplitSpec, execute_tt_plan

from diskgraph._c import DiskgraphDataError


def test_timestamp_batches_mix_without_repeating_rows(tmp_path, parent_index):
    builder = DiskGraphTTBuilder()
    builder.temporal([1000, 2000, 3000, 4000], forecast_length=1, unit="seconds")
    builder.link_prediction(include_frequency=False)
    builder.for_each("users", "user_id")
    builder.set_target_stype("ID")
    builder.aggregate("links", EdgeRef("users", "user_id", "orders", "user_id"),
                      "list_distinct", target_col="order_id")
    builder.split_ratio(SplitSpec(kind="TimeRangeSplit", ranges=[(0, 5000), (5001, 6000), (6001, 7000)],
                                 is_temporal=True, timestamp_unit="seconds", unit="seconds"))
    plan = json.loads(builder.build())
    plan["lp_resolution"] = [{"src_col": "ENTITY1", "dst_table": "users",
                              "kind": "scalar", "node_id_col": "entity_node_id"}]
    plan["lp_rhs_is_temporal"] = True
    result = execute_tt_plan(json.dumps(plan), str(parent_index), "timestamp_shuffle")
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"kind": "diskgraph_task_dataset",
                                   "graph": {"index_path": str(parent_index)},
                                   "stages": result["task_dataset"]["stages"],
                                   "task": {"is_temporal": True}}))
    index = RuntimeIndex(str(manifest), rnd_seed=17, num_threads=1)

    def batches(epoch, worker=0, workers=1, cap=None):
        iterator, _ = index.iter(stage="train", sampling_plan={"layers": []},
                                 num_workers=workers, worker_id=worker, shuffle=True,
                                 epoch=epoch, batch_size=5, entity_node_type="users",
                                 group_by_timestamp=True)
        output = []
        for batch in islice(iterator, cap):
            task = batch["task"]
            times = task["seed_timestamps"].to_pylist()
            assert len(set(times)) == 1
            output.append((times[0], tuple(task["tt_stage_node_ids"].to_pylist())))
        return output

    full = batches(0)
    ids = [row for _, rows in full for row in rows]
    assert len(ids) == len(set(ids)) == 48
    assert sorted(Counter(time for time, _ in full).values()) == [3, 3, 3, 3]
    assert sorted(len(rows) for _, rows in full) == [2] * 4 + [5] * 8
    assert batches(0) == full
    other = execute_tt_plan(json.dumps(plan), str(parent_index), "other_run")
    manifest.write_text(json.dumps({"kind": "diskgraph_task_dataset",
                                   "graph": {"index_path": str(parent_index)},
                                   "stages": other["task_dataset"]["stages"],
                                   "task": {"is_temporal": True}}))
    index = RuntimeIndex(str(manifest), rnd_seed=17, num_threads=1)
    other_run = batches(0)
    assert other_run != full
    assert sorted(row for _, rows in other_run for row in rows) == sorted(ids)
    assert batches(0) == other_run
    assert batches(1) != other_run
    full = other_run
    for workers in [2, 5, 16]:
        for worker in range(workers):
            assert batches(0, worker, workers) == full[worker::workers]

    counts = Counter()
    mixed = 0
    for epoch in range(400):
        prefix = batches(epoch, cap=2)
        selected = [row for _, rows in prefix for row in rows]
        assert len(selected) == len(set(selected))
        counts.update(selected)
        mixed += len({time for time, _ in prefix}) > 1
    expected = 400 * 2 / len(full)
    assert set(counts) == set(ids)
    assert all(abs(count - expected) < 30 for count in counts.values())
    assert mixed > 250

    extension = parent_index / "train_tables" / "other_run"
    metadata = json.loads((extension / "diskgraph_manifest.json").read_text())
    bucket_path = extension / metadata["task_dataset"]["stages"]["train"]["bucket_index"]["path"]
    original = bucket_path.read_bytes()
    for contents, message in [(original[:-1], "not a multiple of 8"), (b"", "exceeds")]:
        bucket_path.write_bytes(contents)
        with pytest.raises(DiskgraphDataError, match=message):
            batches(0)
