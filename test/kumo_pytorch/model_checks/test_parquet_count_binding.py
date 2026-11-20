# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct, partitioned sources retain full scale while prediction cohorts stay bounded."""

import hashlib
from datetime import datetime, timedelta
from test.kumo_pytorch.bindings import dataset_definition, load_bindings
from test.kumo_pytorch.contracts import binding_contract
from test.kumo_pytorch.data import prepare_binding, schema_for
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.parquet_features import count_graph_features

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r


def fixture(tmp_path):
    binding = load_bindings()["T01"]
    definition = dataset_definition(binding)
    definition["anchor_generation"]["cohort_size"] = 4
    binding["dataset"] = definition
    root, files = tmp_path / "sources", {}
    start = datetime(2025, 12, 10)

    def write(name, data, part=0):
        key = definition["tables"][name]["prefix"] + f"part-{part}.parquet"
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), path)
        files[key] = {
            "key": key,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    for part, ids in enumerate([[4, 0, 2], [3, 1]]):
        write(
            "viewers",
            {
                "viewer_id": ids,
                "signup_time": pa.array(
                    [start - timedelta(days=10)] * len(ids), pa.timestamp("ms")
                ),
            },
            part,
        )
    event_id = 0
    for part, ids in enumerate([[0, 1, 4], [2, 3]]):
        rows = []
        for entity in ids:
            for week in range(4) if entity else range(1):
                t = start + timedelta(days=7 * week)
                offsets = [-2000, -1000, -1, 0, 1] if entity else [-2000, -1000, -1]
                for offset in offsets:
                    rows.append(
                        {
                            "session_id": event_id,
                            "viewer_id": entity,
                            "title_id": 10,
                            "start_time": t + timedelta(milliseconds=offset),
                            "watch_seconds": 10 + event_id % 7,
                            "completion_frac": 0.1 + 0.1 * (event_id % 5),
                            "device_type": "tv" if entity % 2 else "phone",
                        }
                    )
                    event_id += 1
        if 1 in ids:
            for n in range(40):
                rows.append(
                    {
                        "session_id": event_id,
                        "viewer_id": 1,
                        "title_id": 10,
                        "start_time": start - timedelta(days=1, seconds=n),
                        "watch_seconds": 4 + n,
                        "completion_frac": 0.5,
                        "device_type": "tv",
                    }
                )
                event_id += 1
        table = pa.Table.from_pylist(rows)
        table = table.set_column(
            table.schema.get_field_index("start_time"),
            "start_time",
            table["start_time"].cast(pa.timestamp("ms")),
        )
        write("sessions", table.to_pydict(), part)
    write(
        "titles",
        {
            "title_id": [10],
            "series_id": [20],
            "added_time": pa.array([start - timedelta(days=100)], pa.timestamp("ms")),
        },
    )
    write(
        "series",
        {
            "series_id": [20],
            "genre_id": [30],
            "created_time": pa.array([start - timedelta(days=101)], pa.timestamp("ms")),
        },
    )
    write("genres", {"genre_id": [30]})
    write("title_genres", {"title_id": [10], "genre_id": [30], "relevance": [1.0]})
    write(
        "ratings",
        {
            "viewer_id": [0],
            "title_id": [10],
            "rated_time": pa.array([start], pa.timestamp("ms")),
            "thumbs_up": [True],
        },
    )
    return root, files, binding


def prepared(tmp_path):
    root, files, binding = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    return binding, out, prepare_binding(root, out, binding, files, limit=None)


@pytest.mark.parametrize("mode", ["smoke", "full", "training", "lifecycle"])
def test_direct_parquet_binding_campaigns(tmp_path, mode):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    result = run_case(
        dataset,
        anchors,
        population,
        binding,
        out,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert result["status"] == "passed", result.get("traceback", result)


@pytest.fixture
def native(tmp_path):
    binding, out, (dataset, anchors, population, mapping, capacity) = prepared(tmp_path)
    contract = binding_contract(
        schema_for(dataset, binding), binding, max_events=capacity
    )
    with r.prepare(
        contract, dataset, r.DiskGraphEngine(index=out / "index", num_threads=1)
    ) as plan:
        with plan.materialize(
            anchors,
            fit_split="train",
            encoder_cutoff=population["encoder_cutoff_seconds"],
        ) as task:
            batch = next(
                iter(
                    task.batches(
                        split="train",
                        batch_size=4,
                        shuffle=False,
                        output_format="model",
                    )
                )
            )
            checks = count_graph_features(
                binding,
                out / "anchors.parquet",
                mapping,
                task.fitted_state.to_dict(),
                mode="smoke",
            )
            checks(batch)
            yield batch, checks
