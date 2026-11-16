# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from kumo_core.train import artifact_stats as artifact_stats_module
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES,
    PredictionArtifactStatsCollector,
    PredictionArtifactStatsPayload,
    PredictionArtifactStatsRequest,
    ShardArtifactStats,
)
from kumo_core.train.output import ArrowPredOutput, StreamingPredOutputWriter
from kumo_core.types import PredictionArtifactType

_ARTIFACT_TYPES = (
    PredictionArtifactType.EMBEDDINGS,
    PredictionArtifactType.PREDICTIONS,
)


def _request(output_dir: Path, *, stop: int = 2,
             artifact_types=_ARTIFACT_TYPES) -> PredictionArtifactStatsRequest:
    return PredictionArtifactStatsRequest(
        attempt_id="attempt-0",
        job_id="bp-job-1",
        runtime_config_sha256=hashlib.sha256(b"runtime-0").hexdigest(),
        output_dir=str(output_dir),
        runtime_shard_id=0,
        runtime_shard_count=1,
        output_partition_start=0,
        output_partition_stop=stop,
        expected_artifact_types=artifact_types,
    )


def _write_partition(output_dir: Path, partition_id: int, start: int) -> dict:
    rows = 600
    predictions = pa.table({
        "ENTITY": [f"entity-{row}" for row in range(start, start + rows)],
        "CLASS": [f"class-{row % 13}" for row in range(start, start + rows)],
        "SCORE": np.arange(start, start + rows, dtype=np.float32),
    })
    embeddings = pa.table({
        "ID": [f"entity-{row}" for row in range(start, start + rows)],
        "TYPE": ["entity"] * rows,
        "EMBEDDING": [[float(row), None] for row in range(start, start + rows)],
    })
    writer = StreamingPredOutputWriter(
        str(output_dir), partition_id, max_size_per_file=2048,
        max_buffer_size=1024,
        artifact_stats_sample_rows={item: 1000 for item in _ARTIFACT_TYPES},
    )
    writer.write(ArrowPredOutput(predictions, embeddings))
    writer.close()
    return writer.artifact_stats


def test_prediction_artifact_payload_round_trip_is_aggregate_only(tmp_path):
    collector = PredictionArtifactStatsCollector(str(tmp_path), _ARTIFACT_TYPES)
    collector.add_partition(0, _write_partition(tmp_path, 0, 0))
    collector.add_partition(1, _write_partition(tmp_path, 1, 600))
    request = _request(tmp_path)

    payload = collector.build_payload(request)
    parsed = PredictionArtifactStatsPayload.from_json(payload.to_json())
    parsed.validate(request)

    assert parsed == payload
    assert parsed.written_partition_count == 2
    encoded = payload.to_json()
    assert 'part-00000' not in encoded
    assert 'row_group' not in encoded
    for artifact in parsed.artifacts:
        paths = sorted(Path(artifact.output_prefix).glob("*.parquet"))
        dataset = ds.dataset(paths, format="parquet")
        assert artifact.file_count == len(paths)
        assert artifact.num_rows == dataset.count_rows() == 1200
        assert artifact.num_cols == len(dataset.schema.names)
        assert artifact.sample_prefix_logical_bytes[-1] == dataset.head(
            1000).nbytes
        assert artifact.physical_size_bytes == sum(
            path.stat().st_size for path in paths)


def test_prediction_artifact_payload_rejects_identity_and_totals(tmp_path):
    collector = PredictionArtifactStatsCollector(str(tmp_path), _ARTIFACT_TYPES)
    collector.add_partition(0, _write_partition(tmp_path, 0, 0))
    request = _request(tmp_path, stop=1)
    payload = collector.build_payload(request)

    with pytest.raises(ValueError, match="identity mismatch"):
        payload.validate(dataclasses.replace(request, attempt_id="stale"))
    with pytest.raises(ValueError, match="stats set is invalid"):
        dataclasses.replace(
            payload, artifacts=payload.artifacts + (payload.artifacts[0], )
        ).validate(request)
    corrupt = dataclasses.replace(
        payload.artifacts[0], file_count=0)
    with pytest.raises(ValueError, match="file totals"):
        dataclasses.replace(
            payload, artifacts=(corrupt, payload.artifacts[1])).validate(request)


def test_collector_rejects_gaps_duplicates_and_schema_changes(tmp_path):
    collector = PredictionArtifactStatsCollector(str(tmp_path), _ARTIFACT_TYPES)
    stats = _write_partition(tmp_path, 0, 0)
    collector.add_partition(0, stats)
    with pytest.raises(ValueError, match="Duplicate"):
        collector.add_partition(0, stats)

    other_dir = tmp_path / "other"
    other = _write_partition(other_dir, 2, 600)
    adjusted = {
        kind: dataclasses.replace(
            frame,
            path_prefix=str(tmp_path / kind.output_file_name / "part-00002"),
            schema_fingerprint=("0" * 64 if kind is
                                PredictionArtifactType.PREDICTIONS else
                                frame.schema_fingerprint),
        )
        for kind, frame in other.items()
    }
    collector.add_partition(2, adjusted)
    with pytest.raises(ValueError, match="not contiguous"):
        collector.build_payload(_request(tmp_path, stop=3))


def test_empty_runtime_payload_is_bounded_and_has_no_artifacts(tmp_path):
    request = _request(tmp_path, stop=1)
    payload = PredictionArtifactStatsCollector(
        str(tmp_path), _ARTIFACT_TYPES).build_payload(request)
    assert payload.written_partition_count == 0
    assert payload.artifacts == ()
    assert PredictionArtifactStatsPayload.from_json(payload.to_json()) == payload


def test_live_shaped_file_count_does_not_scale_control_payload(tmp_path):
    request = _request(
        tmp_path, stop=1,
        artifact_types=(PredictionArtifactType.PREDICTIONS, ))
    artifact = ShardArtifactStats(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        output_prefix=str(tmp_path / "predictions.parquet"),
        partition_count=1,
        schema_fingerprint="a" * 64,
        num_rows=29_300,
        num_cols=4,
        sample_rows=1000,
        sample_prefix_logical_bytes=tuple(i * 100 for i in range(1001)),
        size_bytes=2_930_000,
        physical_size_bytes=29_300 * 65_000_000,
        file_count=29_300,
    )
    payload = PredictionArtifactStatsPayload(request, 1, (artifact, ))
    payload.validate(request)
    encoded = payload.to_json().encode()

    assert len(encoded) < 16 * 1024
    assert b"part-" not in encoded
    assert PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES == 256 * 1024

    monkeypatch_limit = artifact_stats_module.PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES
    artifact_stats_module.PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES = 1
    try:
        with pytest.raises(ValueError, match="payload is too large"):
            payload.to_json()
    finally:
        artifact_stats_module.PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES = monkeypatch_limit
