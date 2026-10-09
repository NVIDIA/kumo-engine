# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded ownership summaries for batch-prediction output roots."""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from kumo_core.train._artifact_storage import _LOCAL_STORAGE
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_SAMPLE_ROWS,
    PredictionArtifactStatsPayload,
)
from kumo_core.types import PredictionArtifactType

PREDICTION_ARTIFACT_MANIFEST_VERSION = 2
PREDICTION_SHARD_MANIFEST_MAX_BYTES = 256 << 10
PREDICTION_SELECTION_MANIFEST_MAX_BYTES = 1 << 20
PREDICTION_MANIFEST_MAX_ROOTS = 10_000
PREDICTION_CONFIG_MAX_BYTES = 8 << 20
PREDICTION_PRODUCER_ABI = "kumo-pred-output-v2"

_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTITY_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PART = re.compile(r"part-(?P<partition>\d{5,})-(?P<part>\d{5,})\.parquet")


def prediction_shard_producer_abi(runtime_shard_count: int) -> str:
    # v3: process-major seed ids; identical to v2 unsharded.
    return ("kumo-pred-output-v3"
            if runtime_shard_count > 1 else PREDICTION_PRODUCER_ABI)


class PredictionProducerMismatchError(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "This batch prediction job straddled a Kumo version change, so "
            "its shards were produced by incompatible versions. Re-run the "
            "job.")


def _json_bytes(value: Mapping[str, Any], limit: int, *,
                pretty: bool = False) -> bytes:
    # Both storage adapters write indent=2; preflight those actual bytes for
    # manifests. Conditional pointers retain their compact representation.
    encoded = (json.dumps(value, indent=2) if pretty else json.dumps(
        value, sort_keys=True, separators=(",", ":"))).encode()
    if len(encoded) > limit:
        raise ValueError("Prediction artifact manifest exceeds its size limit")
    return encoded


def _read_bounded_json(path: str, limit: int, *,
                       _storage=_LOCAL_STORAGE) -> tuple[dict[str, Any], bytes]:
    size = _storage.fs.filesize(path)
    if size is None or size <= 0 or size > limit:
        raise ValueError("Prediction artifact manifest has an invalid size")
    encoded = _storage.fs.read_binary(path)
    if len(encoded) != size or len(encoded) > limit:
        raise ValueError("Prediction artifact manifest read is inconsistent")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError("Prediction artifact manifest must be an object")
    return value, encoded


def hash_prediction_config(path: str, *, _storage=_LOCAL_STORAGE) -> str:
    """Hash one bounded local or remote prediction configuration."""
    size = _storage.fs.filesize(path)
    if size is None or size <= 0 or size > PREDICTION_CONFIG_MAX_BYTES:
        raise ValueError("Prediction config has an invalid size")
    digest = hashlib.sha256()
    total = 0
    with _storage.fs.get_binary_handle(path, "rb") as handle:
        while total <= PREDICTION_CONFIG_MAX_BYTES:
            chunk = handle.read(
                min(1 << 20, PREDICTION_CONFIG_MAX_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > PREDICTION_CONFIG_MAX_BYTES:
                raise ValueError("Prediction config exceeds its size limit")
            digest.update(chunk)
    if total != size:
        raise ValueError("Prediction config changed while it was read")
    return digest.hexdigest()


def _canonical_coordinate(path: str) -> tuple[str, str, str]:
    split = urlsplit(path)
    if split.scheme:
        normalized = osp.normpath('/' + split.path.lstrip('/'))
        return split.scheme.casefold(), split.netloc.casefold(), normalized
    return "local", "", osp.abspath(path)


def require_owned_path(path: str, root: str) -> None:
    """Require ``path`` to be strictly contained by ``root``."""
    path_scheme, path_authority, path_value = _canonical_coordinate(path)
    root_scheme, root_authority, root_value = _canonical_coordinate(root)
    if (path_scheme, path_authority) != (root_scheme, root_authority):
        raise ValueError("Prediction artifact path has a different provider")
    prefix = root_value.rstrip('/') + '/'
    if path_value == root_value or not path_value.startswith(prefix):
        raise ValueError("Prediction artifact path is outside its owned root")


@dataclass(frozen=True)
class PredictionArtifactInventory:
    """Publish-last producer summary for one artifact root.

    No field grows with the number of Parquet files. Empty runtime shards carry
    an empty summary and are omitted from the selected root set.
    """

    artifact_type: PredictionArtifactType
    root: str
    file_count: int
    schema_fingerprint: str | None = None
    num_rows: int = 0
    num_cols: int = 0
    sample_rows: int = 0
    sample_prefix_logical_bytes: tuple[int, ...] = ()
    size_bytes: int = 0
    physical_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "root": self.root,
            "file_count": self.file_count,
            "schema_fingerprint": self.schema_fingerprint,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "sample_rows": self.sample_rows,
            "sample_prefix_logical_bytes":
            list(self.sample_prefix_logical_bytes),
            "size_bytes": self.size_bytes,
            "physical_size_bytes": self.physical_size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str,
                                      Any]) -> "PredictionArtifactInventory":
        return cls(
            artifact_type=PredictionArtifactType(value["artifact_type"]),
            root=str(value["root"]),
            file_count=int(value["file_count"]),
            schema_fingerprint=(None if value["schema_fingerprint"] is None else
                                str(value["schema_fingerprint"])),
            num_rows=int(value["num_rows"]),
            num_cols=int(value["num_cols"]),
            sample_rows=int(value["sample_rows"]),
            sample_prefix_logical_bytes=tuple(
                int(item) for item in value["sample_prefix_logical_bytes"]),
            size_bytes=int(value["size_bytes"]),
            physical_size_bytes=int(value["physical_size_bytes"]),
        )

    def validate(self, generation_root: str) -> None:
        expected_root = osp.join(generation_root,
                                 self.artifact_type.output_file_name)
        if self.root != expected_root:
            raise ValueError("Prediction artifact root is misbound")
        if self.file_count < 0:
            raise ValueError("Prediction artifact file count is invalid")
        if self.file_count == 0:
            if (self.schema_fingerprint is not None or self.num_rows
                    or self.num_cols or self.sample_rows
                    or self.sample_prefix_logical_bytes or self.size_bytes
                    or self.physical_size_bytes):
                raise ValueError("Empty prediction artifact has statistics")
            return
        if (self.schema_fingerprint is None
                or not _DIGEST.fullmatch(self.schema_fingerprint)
                or self.num_rows <= 0 or self.num_cols <= 0
                or self.physical_size_bytes <= 0):
            raise ValueError("Prediction artifact aggregate is invalid")
        if self.sample_rows != min(PREDICTION_ARTIFACT_SAMPLE_ROWS,
                                   self.num_rows):
            raise ValueError("Prediction artifact sample count is invalid")
        prefix = self.sample_prefix_logical_bytes
        if (len(prefix) != self.sample_rows + 1 or prefix[0] != 0
                or any(left > right
                       for left, right in zip(prefix, prefix[1:]))):
            raise ValueError("Prediction artifact sample is invalid")
        if self.size_bytes != int(
                round(prefix[-1] / self.sample_rows * self.num_rows)):
            raise ValueError("Prediction artifact logical size is invalid")


@dataclass(frozen=True)
class PredictionShardManifest:
    schema_version: int
    job_id: str
    attempt_id: str
    graph_snapshot_id: str
    source_config_sha256: str
    runtime_config_sha256: str
    producer_abi: str
    canonical_output_root: str
    generation_root: str
    runtime_shard_id: int
    runtime_shard_count: int
    output_partition_start: int
    output_partition_stop: int
    requested_artifacts: tuple[PredictionArtifactType, ...]
    artifacts: tuple[PredictionArtifactInventory, ...]
    producer_stats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "graph_snapshot_id": self.graph_snapshot_id,
            "source_config_sha256": self.source_config_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "producer_abi": self.producer_abi,
            "canonical_output_root": self.canonical_output_root,
            "generation_root": self.generation_root,
            "runtime_shard_id": self.runtime_shard_id,
            "runtime_shard_count": self.runtime_shard_count,
            "output_partition_start": self.output_partition_start,
            "output_partition_stop": self.output_partition_stop,
            "requested_artifacts":
            [item.value for item in self.requested_artifacts],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "producer_stats": self.producer_stats,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionShardManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            job_id=str(value["job_id"]),
            attempt_id=str(value["attempt_id"]),
            graph_snapshot_id=str(value["graph_snapshot_id"]),
            source_config_sha256=str(value["source_config_sha256"]),
            runtime_config_sha256=str(value["runtime_config_sha256"]),
            producer_abi=str(value["producer_abi"]),
            canonical_output_root=str(value["canonical_output_root"]),
            generation_root=str(value["generation_root"]),
            runtime_shard_id=int(value["runtime_shard_id"]),
            runtime_shard_count=int(value["runtime_shard_count"]),
            output_partition_start=int(value["output_partition_start"]),
            output_partition_stop=int(value["output_partition_stop"]),
            requested_artifacts=tuple(
                PredictionArtifactType(item)
                for item in value["requested_artifacts"]),
            artifacts=tuple(
                PredictionArtifactInventory.from_dict(item)
                for item in value["artifacts"]),
            producer_stats=bool(value["producer_stats"]),
        )

    def validate(self) -> None:
        if self.schema_version != PREDICTION_ARTIFACT_MANIFEST_VERSION:
            raise ValueError("Unsupported prediction artifact manifest")
        if not self.job_id or not self.attempt_id or not self.graph_snapshot_id:
            raise ValueError("Prediction artifact identity is incomplete")
        if (not _DIGEST.fullmatch(self.source_config_sha256)
                or not _DIGEST.fullmatch(self.runtime_config_sha256)):
            raise ValueError("Prediction config digest is invalid")
        if self.producer_abi != prediction_shard_producer_abi(
                self.runtime_shard_count):
            raise PredictionProducerMismatchError()
        if not self.producer_stats:
            raise ValueError("Prediction producer aggregate is required")
        if self.runtime_shard_count <= 0 or not (0 <= self.runtime_shard_id <
                                                 self.runtime_shard_count):
            raise ValueError("Prediction runtime shard identity is invalid")
        if not 0 <= self.output_partition_start < self.output_partition_stop:
            raise ValueError("Prediction output partition range is invalid")
        expected_generation = osp.join(self.canonical_output_root, "_attempts",
                                       f"runtime-{self.runtime_shard_id:05d}",
                                       self.attempt_id)
        if self.generation_root != expected_generation:
            raise ValueError("Prediction generation root is misbound")
        expected = tuple(
            sorted(set(self.requested_artifacts), key=lambda item: item.value))
        if not expected or expected != self.requested_artifacts:
            raise ValueError("Prediction requested artifact set is invalid")
        by_type = {item.artifact_type: item for item in self.artifacts}
        if len(by_type) != len(self.artifacts) or set(by_type) != set(expected):
            raise ValueError("Prediction artifact summary set is invalid")
        for artifact in self.artifacts:
            artifact.validate(self.generation_root)


@dataclass(frozen=True)
class PredictionManifestRef:
    path: str
    sha256: str
    size_bytes: int
    job_id: str
    attempt_id: str
    runtime_shard_id: int


@dataclass(frozen=True)
class SelectedPredictionArtifact:
    artifact_type: PredictionArtifactType
    roots: tuple[str, ...]
    file_count: int
    schema_fingerprint: str
    num_rows: int
    num_cols: int
    size_bytes: int
    physical_size_bytes: int

    @property
    def root(self) -> str:
        """Primary root retained for APIs that also receive the full file list."""
        return self.roots[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "roots": list(self.roots),
            "file_count": self.file_count,
            "schema_fingerprint": self.schema_fingerprint,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "size_bytes": self.size_bytes,
            "physical_size_bytes": self.physical_size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str,
                                      Any]) -> "SelectedPredictionArtifact":
        return cls(
            artifact_type=PredictionArtifactType(value["artifact_type"]),
            roots=tuple(str(item) for item in value["roots"]),
            file_count=int(value["file_count"]),
            schema_fingerprint=str(value["schema_fingerprint"]),
            num_rows=int(value["num_rows"]),
            num_cols=int(value["num_cols"]),
            size_bytes=int(value["size_bytes"]),
            physical_size_bytes=int(value["physical_size_bytes"]),
        )


@dataclass(frozen=True)
class PredictionSelectionManifest:
    schema_version: int
    job_id: str
    selection_id: str
    selection_attempt_id: str
    graph_snapshot_id: str
    source_config_sha256: str
    producer_abi: str
    canonical_output_root: str
    selection_root: str
    shard_manifest_sha256s: tuple[str, ...]
    artifacts: tuple[SelectedPredictionArtifact, ...]
    producer_stats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "selection_id": self.selection_id,
            "selection_attempt_id": self.selection_attempt_id,
            "graph_snapshot_id": self.graph_snapshot_id,
            "source_config_sha256": self.source_config_sha256,
            "producer_abi": self.producer_abi,
            "canonical_output_root": self.canonical_output_root,
            "selection_root": self.selection_root,
            "shard_manifest_sha256s": list(self.shard_manifest_sha256s),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "producer_stats": self.producer_stats,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str,
                                      Any]) -> "PredictionSelectionManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            job_id=str(value["job_id"]),
            selection_id=str(value["selection_id"]),
            selection_attempt_id=str(value["selection_attempt_id"]),
            graph_snapshot_id=str(value["graph_snapshot_id"]),
            source_config_sha256=str(value["source_config_sha256"]),
            producer_abi=str(value["producer_abi"]),
            canonical_output_root=str(value["canonical_output_root"]),
            selection_root=str(value["selection_root"]),
            shard_manifest_sha256s=tuple(
                str(item) for item in value["shard_manifest_sha256s"]),
            artifacts=tuple(
                SelectedPredictionArtifact.from_dict(item)
                for item in value["artifacts"]),
            producer_stats=bool(value["producer_stats"]),
        )

    def validate(self) -> None:
        if self.schema_version != PREDICTION_ARTIFACT_MANIFEST_VERSION:
            raise ValueError("Unsupported prediction selection manifest")
        if (not self.job_id or not self.graph_snapshot_id
                or not _IDENTITY_TOKEN.fullmatch(self.selection_id)
                or not _IDENTITY_TOKEN.fullmatch(self.selection_attempt_id)):
            raise ValueError("Prediction selection identity is incomplete")
        if (not _DIGEST.fullmatch(self.source_config_sha256)
                or self.producer_abi != PREDICTION_PRODUCER_ABI
                or not self.producer_stats):
            raise ValueError(
                "Prediction selection producer identity is invalid")
        if not self.shard_manifest_sha256s or any(
                not _DIGEST.fullmatch(item)
                for item in self.shard_manifest_sha256s):
            raise ValueError("Prediction shard digest set is invalid")
        expected_control_root = osp.join(self.canonical_output_root,
                                         "_selections", self.selection_id,
                                         "attempts", self.selection_attempt_id)
        if self.selection_root != expected_control_root:
            raise ValueError("Prediction selection control root is misbound")
        types = [item.artifact_type for item in self.artifacts]
        if types != sorted(set(types), key=lambda item: item.value):
            raise ValueError("Prediction selection artifact set is invalid")
        all_roots: set[str] = set()
        root_count = 0
        for artifact in self.artifacts:
            if (not artifact.roots or artifact.file_count <= 0
                    or not _DIGEST.fullmatch(artifact.schema_fingerprint)
                    or artifact.num_rows <= 0 or artifact.num_cols <= 0
                    or artifact.size_bytes <= 0
                    or artifact.physical_size_bytes <= 0):
                raise ValueError("Selected prediction aggregate is invalid")
            for root in artifact.roots:
                require_owned_path(
                    root, osp.join(self.canonical_output_root, "_attempts"))
                if root in all_roots:
                    raise ValueError("Selected prediction root is duplicated")
                all_roots.add(root)
                root_count += 1
        if root_count > PREDICTION_MANIFEST_MAX_ROOTS:
            raise ValueError("Prediction selection has too many roots")


@dataclass(frozen=True)
class PredictionSelectionRef:
    manifest_path: str
    manifest_sha256: str
    manifest_size_bytes: int
    pointer_path: str
    job_id: str
    selection_id: str
    selection_attempt_id: str


def _write_manifest(path: str, value: Mapping[str, Any], limit: int, *,
                    _storage=_LOCAL_STORAGE) -> bytes:
    _json_bytes(value, limit, pretty=True)
    try:
        _storage.save_json_atomic(value, path)
    except Exception:
        current_value, _ = _read_bounded_json(path, limit, _storage=_storage)
        if current_value != value:
            raise
    current_value, current = _read_bounded_json(path, limit, _storage=_storage)
    if current_value != value:
        raise RuntimeError("Published prediction manifest differs from source")
    return current


def publish_shard_manifest(manifest: PredictionShardManifest, *,
                           _storage=_LOCAL_STORAGE) -> PredictionManifestRef:
    manifest.validate()
    path = osp.join(manifest.generation_root, "_artifact_manifest.json")
    encoded = _write_manifest(path, manifest.to_dict(),
                              PREDICTION_SHARD_MANIFEST_MAX_BYTES,
                              _storage=_storage)
    return PredictionManifestRef(
        path=path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        job_id=manifest.job_id,
        attempt_id=manifest.attempt_id,
        runtime_shard_id=manifest.runtime_shard_id,
    )


def load_shard_manifest(ref: PredictionManifestRef, *,
                        _storage=_LOCAL_STORAGE) -> PredictionShardManifest:
    value, encoded = _read_bounded_json(ref.path,
                                        PREDICTION_SHARD_MANIFEST_MAX_BYTES,
                                        _storage=_storage)
    if (len(encoded) != ref.size_bytes
            or hashlib.sha256(encoded).hexdigest() != ref.sha256):
        raise ValueError("Prediction shard manifest reference changed")
    manifest = PredictionShardManifest.from_dict(value)
    manifest.validate()
    if (manifest.job_id != ref.job_id or manifest.attempt_id != ref.attempt_id
            or manifest.runtime_shard_id != ref.runtime_shard_id):
        raise ValueError("Prediction shard manifest reference is misbound")
    return manifest


def _selection_pointer_value(
    manifest_path: str,
    manifest_sha256: str,
    manifest_size_bytes: int,
    job_id: str,
    selection_id: str,
    selection_attempt_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_ARTIFACT_MANIFEST_VERSION,
        "job_id": job_id,
        "selection_id": selection_id,
        "selection_attempt_id": selection_attempt_id,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": manifest_size_bytes,
    }


def _read_selection_pointer(path: str, *,
                            _storage=_LOCAL_STORAGE) -> dict[str, Any]:
    value, _ = _read_bounded_json(path, PREDICTION_SHARD_MANIFEST_MAX_BYTES,
                                  _storage=_storage)
    return value


def _selection_ref_from_pointer(
    pointer_path: str,
    pointer: Mapping[str, Any],
    manifest: PredictionSelectionManifest,
) -> PredictionSelectionRef:
    return PredictionSelectionRef(
        manifest_path=str(pointer["manifest_path"]),
        manifest_sha256=str(pointer["manifest_sha256"]),
        manifest_size_bytes=int(pointer["manifest_size_bytes"]),
        pointer_path=pointer_path,
        job_id=manifest.job_id,
        selection_id=manifest.selection_id,
        selection_attempt_id=manifest.selection_attempt_id,
    )


def _selection_matches(
    manifest: PredictionSelectionManifest,
    *,
    job_id: str,
    selection_id: str,
    canonical_output_root: str,
    graph_snapshot_id: str,
    source_config_sha256: str,
    producer_abi: str,
    shard_manifest_sha256s: tuple[str, ...],
) -> bool:
    return (manifest.job_id == job_id and manifest.selection_id == selection_id
            and manifest.graph_snapshot_id == graph_snapshot_id
            and manifest.source_config_sha256 == source_config_sha256
            and manifest.producer_abi == producer_abi
            and manifest.canonical_output_root == canonical_output_root
            and manifest.shard_manifest_sha256s == shard_manifest_sha256s)


def _aggregate_inventory_stats(
    inventories: Sequence[PredictionArtifactInventory], ) -> dict[str, Any]:
    populated = [item for item in inventories if item.file_count]
    if not populated:
        raise ValueError("Prediction artifact output is empty")
    schemas = {item.schema_fingerprint for item in populated}
    columns = {item.num_cols for item in populated}
    if len(schemas) != 1 or len(columns) != 1:
        raise ValueError("Prediction artifact schemas changed across shards")
    total_rows = sum(item.num_rows for item in populated)
    sample_rows = 0
    sample_bytes = 0
    for item in populated:
        if sample_rows >= PREDICTION_ARTIFACT_SAMPLE_ROWS:
            break
        take = min(PREDICTION_ARTIFACT_SAMPLE_ROWS - sample_rows,
                   item.sample_rows)
        sample_bytes += item.sample_prefix_logical_bytes[take]
        sample_rows += take
    if sample_rows != min(PREDICTION_ARTIFACT_SAMPLE_ROWS, total_rows):
        raise ValueError("Prediction artifact sample is incomplete")
    return {
        "roots": tuple(item.root for item in populated),
        "file_count": sum(item.file_count for item in populated),
        "schema_fingerprint": populated[0].schema_fingerprint,
        "num_rows": total_rows,
        "num_cols": populated[0].num_cols,
        "size_bytes": int(round(sample_bytes / sample_rows * total_rows)),
        "physical_size_bytes":
        sum(item.physical_size_bytes for item in populated),
    }


def publish_prediction_selection(
        refs: Sequence[PredictionManifestRef], *, job_id: str,
        selection_id: str, selection_attempt_id: str,
        canonical_output_root: str, graph_snapshot_id: str,
        source_config_sha256: str, expected_shards: int,
        expected_output_partitions: int,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionRef:
    """Select complete shard roots without copying their Parquet objects."""
    if expected_shards <= 0 or len(refs) != expected_shards:
        raise ValueError("Prediction shard manifest set is incomplete")
    if (not _IDENTITY_TOKEN.fullmatch(selection_id)
            or not _IDENTITY_TOKEN.fullmatch(selection_attempt_id)):
        raise ValueError("Prediction selection identity is invalid")
    refs_by_runtime = {item.runtime_shard_id: item for item in refs}
    if len(refs_by_runtime) != expected_shards or set(refs_by_runtime) != set(
            range(expected_shards)):
        raise ValueError("Prediction runtime shard reference set is invalid")
    digests = tuple(refs_by_runtime[index].sha256
                    for index in range(expected_shards))
    selection_root = osp.join(canonical_output_root, "_selections",
                              selection_id, "attempts", selection_attempt_id)
    manifest_path = osp.join(selection_root, "_selection_manifest.json")
    pointer_path = osp.join(canonical_output_root, "_selected_artifacts.json")

    if _storage.fs.exists(pointer_path):
        current = load_current_prediction_selection_metadata(
            canonical_output_root, _storage=_storage)
        if not _selection_matches(current, job_id=job_id,
                                  selection_id=selection_id,
                                  canonical_output_root=canonical_output_root,
                                  graph_snapshot_id=graph_snapshot_id,
                                  source_config_sha256=source_config_sha256,
                                  producer_abi=PREDICTION_PRODUCER_ABI,
                                  shard_manifest_sha256s=digests):
            raise ValueError(
                "Prediction output is already owned by another selection")
        return _selection_ref_from_pointer(
            pointer_path,
            _read_selection_pointer(pointer_path, _storage=_storage), current)

    manifests = [load_shard_manifest(ref, _storage=_storage) for ref in refs]
    by_runtime = {item.runtime_shard_id: item for item in manifests}
    if len(by_runtime) != expected_shards or set(by_runtime) != set(
            range(expected_shards)):
        raise ValueError("Prediction runtime shard manifest set is invalid")
    ordered = [by_runtime[index] for index in range(expected_shards)]
    expected_start = 0
    expected_artifacts = ordered[0].requested_artifacts
    attempts: set[str] = set()
    for manifest in ordered:
        if (manifest.runtime_shard_count != expected_shards
                or manifest.output_partition_start != expected_start):
            raise ValueError("Prediction shard partition ranges overlap or gap")
        expected_start = manifest.output_partition_stop
        if manifest.attempt_id in attempts:
            raise ValueError("Prediction activity attempt is duplicated")
        attempts.add(manifest.attempt_id)
        if (manifest.job_id != job_id
                or manifest.canonical_output_root != canonical_output_root
                or manifest.graph_snapshot_id != graph_snapshot_id
                or manifest.source_config_sha256 != source_config_sha256
                or manifest.requested_artifacts != expected_artifacts):
            raise ValueError("Prediction shard manifests are misbound")
    if expected_output_partitions <= 0 or expected_start != expected_output_partitions:
        raise ValueError("Prediction shard partition union is incomplete")

    selected = []
    for artifact_type in expected_artifacts:
        inventories = [
            next(item for item in manifest.artifacts
                 if item.artifact_type is artifact_type) for manifest in ordered
        ]
        selected.append(
            SelectedPredictionArtifact(
                artifact_type=artifact_type,
                **_aggregate_inventory_stats(inventories),
            ))
    selection = PredictionSelectionManifest(
        schema_version=PREDICTION_ARTIFACT_MANIFEST_VERSION,
        job_id=job_id,
        selection_id=selection_id,
        selection_attempt_id=selection_attempt_id,
        graph_snapshot_id=graph_snapshot_id,
        source_config_sha256=source_config_sha256,
        producer_abi=PREDICTION_PRODUCER_ABI,
        canonical_output_root=canonical_output_root,
        selection_root=selection_root,
        shard_manifest_sha256s=digests,
        artifacts=tuple(selected),
    )
    selection.validate()
    cleanup_control_root = True
    try:
        manifest_bytes = _write_manifest(
            manifest_path, selection.to_dict(),
            PREDICTION_SELECTION_MANIFEST_MAX_BYTES, _storage=_storage)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        pointer = _selection_pointer_value(manifest_path, manifest_sha256,
                                           len(manifest_bytes), job_id,
                                           selection_id, selection_attempt_id)
        pointer_text = _json_bytes(
            pointer, PREDICTION_SHARD_MANIFEST_MAX_BYTES).decode()
        try:
            created = _storage.fs.write_text_if_absent(pointer_path,
                                                       pointer_text)
        except Exception as publish_error:
            try:
                current_pointer = _read_selection_pointer(
                    pointer_path, _storage=_storage)
            except Exception:
                cleanup_control_root = False
                raise publish_error
            created = current_pointer == pointer
        if not created:
            current_pointer = _read_selection_pointer(pointer_path,
                                                      _storage=_storage)
            current = load_current_prediction_selection_metadata(
                canonical_output_root, _storage=_storage)
            if not _selection_matches(
                    current, job_id=job_id, selection_id=selection_id,
                    canonical_output_root=canonical_output_root,
                    graph_snapshot_id=graph_snapshot_id,
                    source_config_sha256=source_config_sha256,
                    producer_abi=PREDICTION_PRODUCER_ABI,
                    shard_manifest_sha256s=digests):
                raise ValueError(
                    "Prediction output is already owned by another selection")
            if current.selection_attempt_id == selection_attempt_id:
                cleanup_control_root = False
            elif cleanup_control_root:
                _storage.fs.rm(selection_root, ignore_errors=False)
                cleanup_control_root = False
            return _selection_ref_from_pointer(pointer_path, current_pointer,
                                               current)
        cleanup_control_root = False
        result = PredictionSelectionRef(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=len(manifest_bytes),
            pointer_path=pointer_path,
            job_id=job_id,
            selection_id=selection_id,
            selection_attempt_id=selection_attempt_id,
        )
        load_prediction_selection(result, validate_objects=False,
                                  _storage=_storage)
        return result
    except BaseException:
        if cleanup_control_root:
            _storage.fs.rm(selection_root, ignore_errors=True)
        raise


def _validate_selected_artifact_roots(artifact: SelectedPredictionArtifact, *,
                                      _storage=_LOCAL_STORAGE) -> None:
    selected_prediction_artifact_files(artifact, _storage=_storage)


def load_prediction_selection(
        ref: PredictionSelectionRef, *, validate_objects: bool = True,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionManifest:
    pointer = _read_selection_pointer(ref.pointer_path, _storage=_storage)
    expected_pointer = _selection_pointer_value(ref.manifest_path,
                                                ref.manifest_sha256,
                                                ref.manifest_size_bytes,
                                                ref.job_id, ref.selection_id,
                                                ref.selection_attempt_id)
    if pointer != expected_pointer:
        raise ValueError("Prediction selection pointer changed")
    value, encoded = _read_bounded_json(
        ref.manifest_path, PREDICTION_SELECTION_MANIFEST_MAX_BYTES,
        _storage=_storage)
    if (len(encoded) != ref.manifest_size_bytes
            or hashlib.sha256(encoded).hexdigest() != ref.manifest_sha256):
        raise ValueError("Prediction selection manifest changed")
    manifest = PredictionSelectionManifest.from_dict(value)
    manifest.validate()
    expected_path = osp.join(manifest.canonical_output_root, "_selections",
                             manifest.selection_id, "attempts",
                             manifest.selection_attempt_id,
                             "_selection_manifest.json")
    if (manifest.job_id != ref.job_id
            or manifest.selection_id != ref.selection_id
            or manifest.selection_attempt_id != ref.selection_attempt_id
            or ref.manifest_path != expected_path):
        raise ValueError("Prediction selection reference is misbound")
    if validate_objects:
        for artifact in manifest.artifacts:
            _validate_selected_artifact_roots(artifact, _storage=_storage)
    return manifest


def _load_current_prediction_selection(
        canonical_output_root: str, *, validate_objects: bool,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionManifest:
    pointer_path = osp.join(canonical_output_root, "_selected_artifacts.json")
    pointer = _read_selection_pointer(pointer_path, _storage=_storage)
    if int(pointer.get("schema_version",
                       -1)) != PREDICTION_ARTIFACT_MANIFEST_VERSION:
        raise ValueError("Unsupported prediction selection pointer")
    manifest_path = str(pointer["manifest_path"])
    require_owned_path(manifest_path, canonical_output_root)
    size = int(pointer["manifest_size_bytes"])
    digest = str(pointer["manifest_sha256"])
    if size <= 0 or not _DIGEST.fullmatch(digest):
        raise ValueError("Prediction selection pointer is invalid")
    value, encoded = _read_bounded_json(
        manifest_path, PREDICTION_SELECTION_MANIFEST_MAX_BYTES,
        _storage=_storage)
    if len(encoded) != size or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("Prediction selection pointer is stale")
    manifest = PredictionSelectionManifest.from_dict(value)
    manifest.validate()
    expected_path = osp.join(canonical_output_root, "_selections",
                             manifest.selection_id, "attempts",
                             manifest.selection_attempt_id,
                             "_selection_manifest.json")
    if (manifest.canonical_output_root != canonical_output_root
            or manifest.job_id != str(pointer["job_id"])
            or manifest.selection_id != str(pointer["selection_id"])
            or manifest.selection_attempt_id != str(
                pointer["selection_attempt_id"])
            or manifest_path != expected_path):
        raise ValueError("Prediction selection pointer is misbound")
    if validate_objects:
        for artifact in manifest.artifacts:
            _validate_selected_artifact_roots(artifact, _storage=_storage)
    return manifest


def load_current_prediction_selection(
        canonical_output_root: str, *,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionManifest:
    return _load_current_prediction_selection(canonical_output_root,
                                              validate_objects=True,
                                              _storage=_storage)


def load_current_prediction_selection_metadata(
        canonical_output_root: str, *,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionManifest:
    return _load_current_prediction_selection(canonical_output_root,
                                              validate_objects=False,
                                              _storage=_storage)


def current_prediction_selection_ref(
        canonical_output_root: str, *,
        _storage=_LOCAL_STORAGE) -> PredictionSelectionRef:
    manifest = load_current_prediction_selection_metadata(
        canonical_output_root, _storage=_storage)
    pointer_path = osp.join(canonical_output_root, "_selected_artifacts.json")
    return _selection_ref_from_pointer(
        pointer_path, _read_selection_pointer(pointer_path, _storage=_storage),
        manifest)


def inventory_generation(
    generation_root: str,
    artifact_types: Sequence[PredictionArtifactType],
    producer_stats: PredictionArtifactStatsPayload | None = None,
) -> tuple[PredictionArtifactInventory, ...]:
    """Build O(artifact-types) summaries without listing output objects."""
    if producer_stats is None:
        raise ValueError("Prediction producer aggregate is required")
    stats_by_type = {
        item.artifact_type: item
        for item in producer_stats.artifacts
    }
    expected = tuple(sorted(set(artifact_types), key=lambda item: item.value))
    if producer_stats.written_partition_count and set(stats_by_type) != set(
            expected):
        raise ValueError("Prediction producer stats artifact set is invalid")
    if not producer_stats.written_partition_count and stats_by_type:
        raise ValueError("Empty prediction shard has artifact stats")
    result = []
    for artifact_type in expected:
        root = osp.join(generation_root, artifact_type.output_file_name)
        stats = stats_by_type.get(artifact_type)
        if stats is None:
            result.append(
                PredictionArtifactInventory(artifact_type=artifact_type,
                                            root=root, file_count=0))
        else:
            result.append(
                PredictionArtifactInventory(
                    artifact_type=artifact_type,
                    root=root,
                    file_count=stats.file_count,
                    schema_fingerprint=stats.schema_fingerprint,
                    num_rows=stats.num_rows,
                    num_cols=stats.num_cols,
                    sample_rows=stats.sample_rows,
                    sample_prefix_logical_bytes=(
                        stats.sample_prefix_logical_bytes),
                    size_bytes=stats.size_bytes,
                    physical_size_bytes=stats.physical_size_bytes,
                ))
    return tuple(result)


@dataclass(frozen=True)
class ResolvedPredictionArtifact:
    root: str
    files: tuple[str, ...] | None
    roots: tuple[str, ...]
    manifest_backed: bool
    selection_ref: PredictionSelectionRef | None


def selected_prediction_artifact_files(
        selected, *, _storage=_LOCAL_STORAGE) -> tuple[str, ...]:
    """List selected immutable roots only when a data consumer needs files."""
    files = []
    names = set()
    for root in selected.roots:
        for path in _storage.fs.listdir(root):
            name = osp.basename(path)
            if _PART.fullmatch(name) is None or name in names:
                raise ValueError(
                    'Selected prediction root contains invalid data')
            names.add(name)
            files.append(path)
    if len(files) != selected.file_count:
        raise ValueError('Selected prediction file count changed')
    return tuple(files)


def resolve_prediction_selection_artifact(
    selection_ref: PredictionSelectionRef,
    artifact_type: PredictionArtifactType,
    *,
    _storage=_LOCAL_STORAGE,
) -> ResolvedPredictionArtifact | None:
    """Resolve and validate one artifact from a bounded selection reference."""
    selection = load_prediction_selection(selection_ref, validate_objects=False,
                                          _storage=_storage)
    selected = next((artifact for artifact in selection.artifacts
                     if artifact.artifact_type is artifact_type), None)
    if selected is None:
        return None
    return ResolvedPredictionArtifact(
        root=selected.root,
        files=selected_prediction_artifact_files(selected, _storage=_storage),
        roots=selected.roots,
        manifest_backed=True,
        selection_ref=selection_ref,
    )
