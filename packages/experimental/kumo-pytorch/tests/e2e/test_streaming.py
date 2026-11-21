# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from e2e.datasets.streaming import prepare_streaming
from kumo.pytorch import api
from tests.e2e.history import check_source_graph
from tests.test_inference_lifecycle import assert_same


def test_streaming_weekly_session_forecast(tmp_path):
    first = int(datetime(2025, 12, 10, tzinfo=UTC).timestamp()) * 1000
    week = 7 * 86400 * 1000

    def timestamps(offsets):
        return pa.array([first + ms for ms in offsets], pa.timestamp("ms"))

    # Viewer 1 signs up at the cutoff, and 10 sorts after 2 and 5 numerically.
    viewers = pa.table(
        {"viewer_id": [10, 1, 5, 2], "signup_time": timestamps([-4000, 0, -4000, -4000])}
    )
    pq.write_table(viewers, tmp_path / "viewers.parquet")
    # The two oldest eligible sessions tie for slot 32.
    offsets = [-35_000, -35_000, *range(-31_000, 0, 1000), -1]
    # Two sessions at the second week's start distinguish [start, end) from (start, end].
    offsets += [0, 1, week - 1, week, week, 2 * week - 1, 2 * week, 3 * week - 1, 3 * week]
    owners = [2] * len(offsets) + [5, 10, 1]
    offsets.extend([-2000, 0, 0])
    ids = range(len(owners))
    sessions = pa.table(
        {
            "session_id": ids,
            "viewer_id": owners,
            "start_time": timestamps(offsets),
            "watch_seconds": [i + 1 for i in ids],
            "completion_frac": [0.1 + 0.1 * (i % 7) for i in ids],
            "device_type": ["TV" if i % 2 else "phone" for i in ids],
        }
    )
    pq.write_table(sessions, tmp_path / "sessions.parquet")
    spec, data, table = prepare_streaming(tmp_path, cohort_size=2)
    counts = {"train:2": 3, "train:5": 0, "valid:2": 3, "valid:5": 0, "test:2": 2, "test:5": 0}
    anchors = pq.read_table(table.source.path)
    assert anchors["future_count"].type == pa.int64()
    assert anchors["count"].type == pa.float32()
    assert anchors["sample_id"].to_pylist() == list(counts)
    assert anchors["future_count"].to_pylist() == list(counts.values())
    fields = {
        "watch_seconds": api.ZScore(),
        "completion_frac": api.ZScore(),
        "device_type": api.Vocabulary(),
    }
    layout = {"sessions": ("viewers_sessions", fields)}
    rows = sessions.to_pylist()
    delivered = []
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=first // 1000 - 1) as task,
    ):
        for batch in task.batches(batch_size=2, output_format="model"):
            meta = batch.metadata
            samples = meta["sample_ids"]
            delivered.extend(samples)
            assert not any(meta["is_padding"])
            nodes = batch.inputs["graph"]["nodes"]["sessions"]
            expected, future_counts = {}, []
            for owner, sample in enumerate(samples):
                viewer, split = meta["entity_ids"][owner].item(), meta["splits"][owner]
                assert sample == f"{split}:{viewer}"
                start = {"train": 0, "valid": week, "test": 2 * week}[split]
                assert meta["as_of_time"][owner].item() == (first + start) // 1000 - 1
                request = meta["request_metadata"][owner]
                assert request["registered_entity_id"] == str(viewer)
                assert request["prediction_time_ns"] == (first + start) * 10**6
                times = {i: offsets[i] for i in ids if owners[i] == viewer}
                future_counts.append(sum(start <= ms < start + week for ms in times.values()))
                # Native graph timestamps round up to whole seconds.
                eligible = {i: (ms + 999) // 1000 for i, ms in times.items() if ms <= start - 1000}
                selected = nodes["ids"][nodes["batch"] == owner].tolist()
                assert sorted(eligible[i] for i in selected) == sorted(eligible.values())[-32:]
                expected.update(((owner, "sessions", i), rows[i]) for i in selected)
            check_source_graph(task, batch, expected, layout, root_table="viewers")
            values = torch.tensor(future_counts, dtype=torch.float32)
            target = {"values": values, "valid": torch.ones(len(samples), dtype=torch.bool)}
            assert_same(batch.targets, {"count": target})
    assert sorted(delivered) == sorted(counts)
