# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from datetime import timedelta

import pyarrow.parquet as pq

from kumo import fm

from _fixture import CUTOFF, USER_A, USER_B, generate, write_bundle


def fixture(tmp_path):
    original = generate(tmp_path / "original")
    tables = deepcopy(original.tables)
    rows = {name: pq.read_table(original.paths(name)).to_pylist() for name in tables}
    rows["interactions"] = [
        {"interaction_id": event_id, "user_id": user, "item_id": "shared",
         "event_time": CUTOFF + timedelta(days=days), "amount": 5.0}
        for event_id, user, days in ((1, USER_A, -41), (2, USER_A, -1),
                                     (3, USER_B, -41), (4, USER_B, -1))
    ]
    tasks = deepcopy(original.manifest["tasks"])
    for task in tasks.values():
        for split in ("context", "predict", "labels"):
            task[split] = pq.read_table(original.path(task[split])).to_pylist()
    bundle = write_bundle(tmp_path / "supported", tables, deepcopy(original.relations),
                          rows, tasks, recipe={"generator": "kumo_fm_integration"})
    spec = json.loads((original.root / "workloads/binary_classification-2hop.json").read_text())
    spec.update(logical_id=bundle.logical_id, snapshot_id=bundle.snapshot_id,
                selection="native_multihop", layers=spec["layers"][:1],
                projections={name: list(bundle.tables[name]["columns"])
                             for name in ("users", "interactions")})
    spec["layers"][0][0]["fanout"] = 1
    return bundle, spec


def test_installed_wheel_samples_once_and_batch_survives_close(tmp_path):
    bundle, spec = fixture(tmp_path)
    index = tmp_path / "index"
    with fm.prepare(bundle, spec, index=index) as loader:
        assert loader.preparation_mode == "index_build"
        with loader.materialize() as request:
            batch = request.load()
            loader._backend.extract = lambda *_: (_ for _ in ()).throw(
                AssertionError("encoding resampled"))
            arrays = batch.encode("arrays")
            ipc = batch.encode("arrow_ipc")
    assert arrays and ipc
    assert batch.tables("predict")["instance_table"].num_rows == 4
    context_events = batch.tables("context")["related_tables"]["interactions"]
    query_events = batch.tables("predict")["related_tables"]["interactions"]
    assert set(context_events["interaction_id"].to_pylist()) == {1, 3}
    assert set(query_events["interaction_id"].to_pylist()) == {2, 4}
    assert set(zip(query_events["instance_id"].to_pylist(),
                   query_events["interaction_id"].to_pylist())) == {
                       (100, 2), (101, 4), (102, 2)}
    assert all(value is not None for value in query_events["__fm_edge_interaction_user"].to_pylist())
    with fm.prepare(bundle, spec, index=index) as loader:
        assert loader.preparation_mode == "cached_index_open"
        with loader.materialize(context=[], predict=[100]) as request:
            single = request.load()
    assert single.splits == ("predict",)
    assert single.tables("predict")["instance_table"].num_rows == 1
