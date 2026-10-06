# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded writer metadata for batch-prediction ownership manifests.

This module runs inside ``kumo-pred``.  It emits one transient payload for the
pool activity to merge into the authoritative shard manifest; it does not own
a durable manifest, pointer, or publication lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
from dataclasses import dataclass
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.types import PredictionArtifactType

PREDICTION_ARTIFACT_SAMPLE_ROWS = 1000
PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES = 256 * 1024
PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_BYTES = 128 * 1024 * 1024
PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_ROWS = 1024 * 1024
PREDICTION_ARTIFACT_STATS_REQUEST_ENV = "KUMO_PRED_ARTIFACT_STATS_REQUEST"
PREDICTION_ARTIFACT_STATS_PATH_ENV = "KUMO_PRED_ARTIFACT_STATS_PATH"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _bounded_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES:
        raise ValueError("Prediction artifact stats payload is too large")
    return encoded


def schema_fingerprint(schema: pa.Schema) -> str:
    canonical = schema.remove_metadata().serialize().to_pybytes()
    return hashlib.sha256(canonical).hexdigest()


def parquet_file_prefix_logical_bytes(
    path: str,
    sample_rows: int,
) -> tuple[int, ...]:
    """Return post-Parquet Arrow bytes for bounded row-group prefixes."""
    if not 0 <= sample_rows <= PREDICTION_ARTIFACT_SAMPLE_ROWS:
        raise ValueError("Sample row count is invalid")
    if sample_rows == 0:
        return (0, )

    result = [0]
    parquet_file = pq.ParquetFile(path)
    for row_group_index in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_index)
        take = min(sample_rows - (len(result) - 1), row_group.num_rows)
        base = result[-1]
        result.extend(base + row_group.slice(0, prefix_rows).nbytes
                      for prefix_rows in range(1, take + 1))
        if len(result) - 1 == sample_rows:
            return tuple(result)
    raise ValueError("Parquet artifact contains fewer rows than its sample")


@dataclass(frozen=True)
class FrameWriterStats:
    path_prefix: str
    schema_fingerprint: str
    num_rows: int
    num_cols: int
    sample_rows: int
    sample_prefix_logical_bytes: tuple[int, ...]
    physical_size_bytes: int
    file_count: int


@dataclass(frozen=True)
class ShardArtifactStats:
    artifact_type: PredictionArtifactType
    output_prefix: str
    partition_count: int
    schema_fingerprint: str
    num_rows: int
    num_cols: int
    sample_rows: int
    sample_prefix_logical_bytes: tuple[int, ...]
    size_bytes: int
    physical_size_bytes: int
    file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "output_prefix": self.output_prefix,
            "partition_count": self.partition_count,
            "schema_fingerprint": self.schema_fingerprint,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "sample_rows": self.sample_rows,
            "sample_prefix_logical_bytes": list(
                self.sample_prefix_logical_bytes),
            "size_bytes": self.size_bytes,
            "physical_size_bytes": self.physical_size_bytes,
            "file_count": self.file_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShardArtifactStats":
        return cls(
            artifact_type=PredictionArtifactType(value["artifact_type"]),
            output_prefix=str(value["output_prefix"]),
            partition_count=int(value["partition_count"]),
            schema_fingerprint=str(value["schema_fingerprint"]),
            num_rows=int(value["num_rows"]),
            num_cols=int(value["num_cols"]),
            sample_rows=int(value["sample_rows"]),
            sample_prefix_logical_bytes=tuple(
                int(size) for size in value["sample_prefix_logical_bytes"]),
            size_bytes=int(value["size_bytes"]),
            physical_size_bytes=int(value["physical_size_bytes"]),
            file_count=int(value["file_count"]),
        )

    def validate(self, output_dir: str, assigned: range) -> None:
        expected_prefix = osp.join(output_dir,
                                   self.artifact_type.output_file_name)
        if self.output_prefix != expected_prefix:
            raise ValueError("Prediction artifact output prefix mismatch")
        if self.partition_count <= 0 or self.partition_count > len(assigned):
            raise ValueError("Prediction artifact partition count is invalid")
        if not re.fullmatch(_DIGEST, self.schema_fingerprint):
            raise ValueError("Prediction artifact schema fingerprint is invalid")
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise ValueError("Prediction artifact shape is invalid")
        if self.sample_rows != min(PREDICTION_ARTIFACT_SAMPLE_ROWS,
                                   self.num_rows):
            raise ValueError("Prediction artifact sample row count is invalid")
        if len(self.sample_prefix_logical_bytes) != self.sample_rows + 1:
            raise ValueError("Prediction artifact sample prefix is incomplete")
        if self.sample_prefix_logical_bytes[0] != 0 or any(
                left > right for left, right in zip(
                    self.sample_prefix_logical_bytes,
                    self.sample_prefix_logical_bytes[1:])):
            raise ValueError("Prediction artifact sample prefix is invalid")
        expected_size = int(
            round(self.sample_prefix_logical_bytes[-1] / self.sample_rows *
                  self.num_rows))
        if self.size_bytes != expected_size:
            raise ValueError("Prediction artifact estimated size is invalid")
        if self.file_count <= 0 or self.physical_size_bytes <= 0:
            raise ValueError("Prediction artifact file totals are invalid")


@dataclass(frozen=True)
class PredictionArtifactStatsRequest:
    attempt_id: str
    job_id: str
    runtime_config_sha256: str
    output_dir: str
    runtime_shard_id: int
    runtime_shard_count: int
    output_partition_start: int
    output_partition_stop: int
    expected_artifact_types: tuple[PredictionArtifactType, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "job_id": self.job_id,
            "runtime_config_sha256": self.runtime_config_sha256,
            "output_dir": self.output_dir,
            "runtime_shard_id": self.runtime_shard_id,
            "runtime_shard_count": self.runtime_shard_count,
            "output_partition_start": self.output_partition_start,
            "output_partition_stop": self.output_partition_stop,
            "expected_artifact_types": [
                artifact.value for artifact in self.expected_artifact_types
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str,
                                      Any]) -> "PredictionArtifactStatsRequest":
        return cls(
            attempt_id=str(value["attempt_id"]),
            job_id=str(value["job_id"]),
            runtime_config_sha256=str(value["runtime_config_sha256"]),
            output_dir=str(value["output_dir"]),
            runtime_shard_id=int(value["runtime_shard_id"]),
            runtime_shard_count=int(value["runtime_shard_count"]),
            output_partition_start=int(value["output_partition_start"]),
            output_partition_stop=int(value["output_partition_stop"]),
            expected_artifact_types=tuple(
                PredictionArtifactType(item)
                for item in value["expected_artifact_types"]),
        )

    def validate(self) -> None:
        if not self.attempt_id or not self.job_id or not self.output_dir:
            raise ValueError("Prediction artifact stats identity is incomplete")
        if not re.fullmatch(_DIGEST, self.runtime_config_sha256):
            raise ValueError("Prediction artifact runtime digest is invalid")
        if self.runtime_shard_count <= 0 or not (
                0 <= self.runtime_shard_id < self.runtime_shard_count):
            raise ValueError("Prediction artifact runtime shard is invalid")
        if not 0 <= self.output_partition_start < self.output_partition_stop:
            raise ValueError("Prediction artifact output range is invalid")
        expected = tuple(
            sorted(set(self.expected_artifact_types), key=lambda item: item.value))
        if not expected or expected != self.expected_artifact_types:
            raise ValueError("Prediction artifact expected set is invalid")


@dataclass(frozen=True)
class PredictionArtifactStatsPayload:
    request: PredictionArtifactStatsRequest
    written_partition_count: int
    artifacts: tuple[ShardArtifactStats, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "written_partition_count": self.written_partition_count,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def to_json(self) -> str:
        return _bounded_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "PredictionArtifactStatsPayload":
        if len(value.encode()) > PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES:
            raise ValueError("Prediction artifact stats payload is too large")
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Prediction artifact stats payload must be an object")
        return cls(
            request=PredictionArtifactStatsRequest.from_dict(parsed["request"]),
            written_partition_count=int(parsed["written_partition_count"]),
            artifacts=tuple(
                ShardArtifactStats.from_dict(artifact)
                for artifact in parsed["artifacts"]),
        )

    def validate(self, expected: PredictionArtifactStatsRequest) -> None:
        expected.validate()
        self.request.validate()
        if self.request != expected:
            raise ValueError("Prediction artifact stats identity mismatch")
        assigned = range(expected.output_partition_start,
                         expected.output_partition_stop)
        if not 0 <= self.written_partition_count <= len(assigned):
            raise ValueError("Written output partition count is invalid")
        if not self.written_partition_count:
            if self.artifacts:
                raise ValueError("Empty prediction shard has artifact stats")
            return
        by_type = {
            artifact.artifact_type: artifact for artifact in self.artifacts
        }
        if (len(by_type) != len(self.artifacts)
                or set(by_type) != set(expected.expected_artifact_types)):
            raise ValueError("Prediction artifact stats set is invalid")
        for artifact in self.artifacts:
            if artifact.partition_count != self.written_partition_count:
                raise ValueError("Prediction artifact partitions are inconsistent")
            artifact.validate(expected.output_dir, assigned)


class PredictionArtifactStatsCollector:
    """Aggregate bounded writer metadata for one ``kumo-pred`` runtime shard."""
    def __init__(
        self,
        output_dir: str,
        expected_artifact_types: tuple[PredictionArtifactType, ...],
    ) -> None:
        self.output_dir = output_dir
        self.expected_artifact_types = expected_artifact_types
        self._partitions: dict[int, Mapping[PredictionArtifactType,
                                            FrameWriterStats]] = {}

    def add_partition(
        self,
        partition_id: int,
        artifacts: Mapping[PredictionArtifactType, FrameWriterStats],
    ) -> None:
        if partition_id in self._partitions:
            raise ValueError(
                f"Duplicate prediction output partition {partition_id}")
        if not artifacts:
            raise ValueError("Prediction output partition contained no artifacts")
        self._partitions[partition_id] = dict(artifacts)

    def sample_rows_needed(self, artifact_type: PredictionArtifactType) -> int:
        if artifact_type not in self.expected_artifact_types:
            return 0
        rows = sum(frame.num_rows for artifacts in self._partitions.values()
                   if (frame := artifacts.get(artifact_type)) is not None)
        return max(0, PREDICTION_ARTIFACT_SAMPLE_ROWS - rows)

    def build_payload(
        self,
        request: PredictionArtifactStatsRequest,
    ) -> PredictionArtifactStatsPayload:
        partition_ids = tuple(sorted(self._partitions))
        if not partition_ids:
            payload = PredictionArtifactStatsPayload(request, 0, ())
            payload.validate(request)
            return payload
        expected_partition_ids = tuple(
            range(request.output_partition_start,
                  request.output_partition_start + len(partition_ids)))
        if partition_ids != expected_partition_ids:
            raise ValueError("Written output partitions are not contiguous")
        artifacts = []
        for artifact_type in request.expected_artifact_types:
            frames = []
            for partition_id in partition_ids:
                frame = self._partitions[partition_id].get(artifact_type)
                if frame is None:
                    raise ValueError(
                        f"Partition {partition_id} is missing {artifact_type.value}")
                expected_prefix = osp.join(
                    request.output_dir,
                    artifact_type.output_file_name,
                    f"part-{partition_id:05d}",
                )
                if frame.path_prefix != expected_prefix:
                    raise ValueError("Writer artifact path prefix mismatch")
                frames.append(frame)
            schemas = {frame.schema_fingerprint for frame in frames}
            columns = {frame.num_cols for frame in frames}
            if len(schemas) != 1 or len(columns) != 1:
                raise ValueError("Prediction artifact schemas changed")
            sample_rows = 0
            sample_prefix = [0]
            for frame in frames:
                if sample_rows >= PREDICTION_ARTIFACT_SAMPLE_ROWS:
                    break
                take = min(PREDICTION_ARTIFACT_SAMPLE_ROWS - sample_rows,
                           frame.sample_rows)
                base = sample_prefix[-1]
                sample_prefix.extend(
                    base + value
                    for value in frame.sample_prefix_logical_bytes[1:take + 1])
                sample_rows += take
            num_rows = sum(frame.num_rows for frame in frames)
            if sample_rows != min(PREDICTION_ARTIFACT_SAMPLE_ROWS, num_rows):
                raise ValueError("Prediction artifact sample is incomplete")
            artifacts.append(
                ShardArtifactStats(
                    artifact_type=artifact_type,
                    output_prefix=osp.join(request.output_dir,
                                           artifact_type.output_file_name),
                    partition_count=len(partition_ids),
                    schema_fingerprint=frames[0].schema_fingerprint,
                    num_rows=num_rows,
                    num_cols=frames[0].num_cols,
                    sample_rows=sample_rows,
                    sample_prefix_logical_bytes=tuple(sample_prefix),
                    size_bytes=int(
                        round(sample_prefix[-1] / sample_rows * num_rows)),
                    physical_size_bytes=sum(frame.physical_size_bytes
                                            for frame in frames),
                    file_count=sum(frame.file_count for frame in frames),
                ))
        payload = PredictionArtifactStatsPayload(request, len(partition_ids),
                                                 tuple(artifacts))
        payload.validate(request)
        _bounded_json(payload.to_dict())
        return payload
