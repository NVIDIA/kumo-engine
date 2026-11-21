# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from e2e.datasets.staging import verify
from e2e.datasets.streaming import T01_MANIFEST, load_streaming, prepare_streaming
from kumo.pytorch import api
from tests.e2e.history import check_source_graph
from tests.test_inference_lifecycle import assert_same

FIRST = int(datetime(2025, 12, 10, tzinfo=UTC).timestamp()) * 1000
WEEK = 7 * 86400 * 1000
FULL_PIN = T01_MANIFEST.parents[1] / "streaming-25b-full.json"


def _bounded_tables():
    def timestamps(offsets):
        return pa.array([FIRST + ms for ms in offsets], pa.timestamp("ms"))

    # Viewer 1 signs up at the cutoff, and 10 sorts after 2 and 5 numerically.
    viewers = pa.table(
        {"viewer_id": [10, 1, 5, 2], "signup_time": timestamps([-4000, 0, -4000, -4000])}
    )
    # The two oldest eligible sessions tie for slot 32.
    offsets = [-35_000, -35_000, *range(-31_000, 0, 1000), -1]
    # Two sessions at the second week's start distinguish [start, end) from (start, end].
    offsets += [0, 1, WEEK - 1, WEEK, WEEK, 2 * WEEK - 1, 2 * WEEK, 3 * WEEK - 1, 3 * WEEK]
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
    return viewers, sessions


@pytest.mark.parametrize("source", ["bounded", "25b", "25b-full"])
def test_streaming_weekly_session_forecast(tmp_path, source):
    manifest = None
    weekly = {3: (0, 1, 0), 873: (2, 0, 1)}
    fields = {
        "watch_seconds": api.ZScore(),
        "completion_frac": api.ZScore(),
        "device_type": api.Vocabulary(),
    }
    if source == "25b-full":
        root = os.environ.get("KUMO_PYTORCH_25B_ROOT")
        if not root:
            pytest.skip("set KUMO_PYTORCH_25B_ROOT for the full-source run")
        manifest = verify(Path(root), FULL_PIN, download=True)["manifest"]
        spec, data, table = prepare_streaming(
            tmp_path, cohort_size=10_000, source_manifest=manifest, replay_viewers=tuple(weekly)
        )
        sessions = pq.read_table(
            manifest.parent / "sessions",
            columns=["session_id", "viewer_id", "start_time", *fields],
            filters=[("viewer_id", "<=", max(weekly)), ("viewer_id", "in", list(weekly))],
            use_threads=False,
        )
    else:
        if source == "bounded":
            viewers, sessions = _bounded_tables()
            weekly = {2: (3, 3, 2), 5: (0, 0, 0)}
        else:
            root = os.environ.get("KUMO_PYTORCH_DATA_ROOT")
            if root is None:
                pytest.skip("set KUMO_PYTORCH_DATA_ROOT to the staged package datasets")
            viewers, sessions = load_streaming(Path(root))
            assert sessions.num_rows == 36
        pq.write_table(viewers, tmp_path / "viewers.parquet")
        pq.write_table(sessions, tmp_path / "sessions.parquet")
        spec, data, table = prepare_streaming(tmp_path, cohort_size=2)
    counts = {
        f"{split}:{viewer}": values[i]
        for i, split in enumerate(("train", "valid", "test"))
        for viewer, values in weekly.items()
    }
    anchors = pq.read_table(table.source.path)
    assert anchors["future_count"].type == pa.int64()
    assert anchors["count"].type == pa.float32()
    assert anchors["sample_id"].to_pylist() == list(counts)
    assert anchors["future_count"].to_pylist() == list(counts.values())
    layout = {"sessions": ("viewers_sessions", fields)}
    timed = sessions.append_column("ms", sessions["start_time"].cast(pa.int64()))
    rows = {r["session_id"]: r for r in timed.to_pylist()}
    delivered = []
    index = tmp_path / "index"
    # The full sessions table exceeds the native per-shard 32-bit row ID range.
    engine = (
        api.DiskGraphEngine(index=index, num_shards=8)
        if manifest
        else api.DiskGraphEngine(index=index, num_threads=1)
    )
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=FIRST // 1000 - 1) as task,
    ):
        if manifest:
            tables = json.loads(manifest.read_text())["tables"]
            expected_rows = {name: table["rows"] for name, table in tables.items()}
            stats = json.loads((index / "ingest_stats.json").read_text())["stats"]
            assert stats["source_rows_by_table"] == stats["node_counts"] == expected_rows
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
                start = FIRST + {"train": 0, "valid": WEEK, "test": 2 * WEEK}[split]
                assert meta["as_of_time"][owner].item() == start // 1000 - 1
                request = meta["request_metadata"][owner]
                assert request["registered_entity_id"] == str(viewer)
                assert request["prediction_time_ns"] == start * 10**6
                times = {i: r["ms"] for i, r in rows.items() if r["viewer_id"] == viewer}
                future_counts.append(sum(start <= ms < start + WEEK for ms in times.values()))
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
