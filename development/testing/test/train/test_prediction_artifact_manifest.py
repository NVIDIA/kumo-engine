# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from kumo_core.train.artifact_stats import (
    PredictionArtifactStatsPayload,
    PredictionArtifactStatsRequest,
    ShardArtifactStats,
)
from kumo_core.types import PredictionArtifactType

from kumo.train import artifact_manifest as manifest_module
from kumo.train.artifact_manifest import (
    PREDICTION_ARTIFACT_MANIFEST_VERSION,
    PredictionProducerMismatchError,
    PredictionShardManifest,
    hash_prediction_config,
    inventory_generation,
    load_current_prediction_selection,
    load_current_prediction_selection_metadata,
    prediction_shard_producer_abi,
    publish_prediction_selection,
    publish_shard_manifest,
    require_owned_path,
)

PREDICTIONS = (PredictionArtifactType.PREDICTIONS, )


def _stats(generation: Path, start: int, stop: int, *, files: int = 1,
           rows: int = 3) -> PredictionArtifactStatsPayload:
    request = PredictionArtifactStatsRequest(
        attempt_id="producer-attempt",
        job_id="bp-job-1",
        runtime_config_sha256="2" * 64,
        output_dir=str(generation),
        runtime_shard_id=start,
        runtime_shard_count=max(1, stop),
        output_partition_start=start,
        output_partition_stop=stop,
        expected_artifact_types=PREDICTIONS,
    )
    sample_rows = min(1000, rows)
    artifact = ShardArtifactStats(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        output_prefix=str(generation / "predictions.parquet"),
        partition_count=stop - start,
        schema_fingerprint="a" * 64,
        num_rows=rows,
        num_cols=3,
        sample_rows=sample_rows,
        sample_prefix_logical_bytes=tuple(i * 10
                                          for i in range(sample_rows + 1)),
        size_bytes=rows * 10,
        physical_size_bytes=files * 17,
        file_count=files,
    )
    payload = PredictionArtifactStatsPayload(request, stop - start,
                                             (artifact, ))
    payload.validate(request)
    return payload


def _shard(root: Path, *, attempt: str, runtime: int = 0, runtimes: int = 1,
           start: int = 0, stop: int = 1, files: int = 1,
           create_files: bool = True):
    generation = root / "_attempts" / f"runtime-{runtime:05d}" / attempt
    artifact_root = generation / "predictions.parquet"
    if create_files:
        artifact_root.mkdir(parents=True)
        for index in range(files):
            (artifact_root /
             f"part-{start:05d}-{index:05d}.parquet").write_bytes(b"x" * 17)
    payload = _stats(generation, start, stop, files=files,
                     rows=max(files, 1) * 3)
    manifest = PredictionShardManifest(
        schema_version=PREDICTION_ARTIFACT_MANIFEST_VERSION,
        job_id="bp-job-1",
        attempt_id=attempt,
        graph_snapshot_id="graph@version",
        source_config_sha256="1" * 64,
        runtime_config_sha256="2" * 64,
        producer_abi=prediction_shard_producer_abi(runtimes),
        canonical_output_root=str(root),
        generation_root=str(generation),
        runtime_shard_id=runtime,
        runtime_shard_count=runtimes,
        output_partition_start=start,
        output_partition_stop=stop,
        requested_artifacts=PREDICTIONS,
        artifacts=inventory_generation(str(generation), PREDICTIONS, payload),
    )
    return publish_shard_manifest(manifest)


def _select(root: Path, refs, *, attempt="selection-attempt"):
    return publish_prediction_selection(
        refs,
        job_id="bp-job-1",
        selection_id="selection",
        selection_attempt_id=attempt,
        canonical_output_root=str(root),
        graph_snapshot_id="graph@version",
        source_config_sha256="1" * 64,
        expected_shards=len(refs),
        expected_output_partitions=len(refs),
    )


def test_prediction_config_hash_streams_remote_uri_with_hard_cap(monkeypatch):
    payload = b"x" * ((1 << 20) + 13)
    reads = []

    class GuardedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1 << 20
            reads.append(size)
            return super().read(size)

    @contextmanager
    def open_remote(path, mode):
        assert (path, mode) == ("s3://bucket/config.json", "rb")
        yield GuardedReader(payload)

    monkeypatch.setattr(manifest_module.fsutil, "filesize",
                        lambda _: len(payload))
    monkeypatch.setattr(manifest_module.fsutil, "get_binary_handle",
                        open_remote)
    assert hash_prediction_config("s3://bucket/config.json") == hashlib.sha256(
        payload).hexdigest()
    assert len(reads) >= 2


def test_owned_paths_reject_cross_provider_and_siblings():
    require_owned_path("s3://bucket/root/child", "s3://bucket/root")
    for candidate in ("s3://other/root/child", "s3://bucket/root2/child"):
        with pytest.raises(ValueError):
            require_owned_path(candidate, "s3://bucket/root")


def test_shard_summary_is_publish_last_bounded_and_has_no_file_payload(
        tmp_path):
    ref = _shard(tmp_path, attempt="attempt-a", files=29_300,
                 create_files=False)
    encoded = Path(ref.path).read_bytes()
    value = json.loads(encoded)

    assert len(encoded) < 16 * 1024
    assert value["schema_version"] == 2
    assert value["artifacts"][0]["file_count"] == 29_300
    assert b"part-" not in encoded
    assert b"row_group" not in encoded
    assert b"identity" not in encoded


def test_inventory_never_lists_or_reads_remote_data(monkeypatch, tmp_path):
    generation = tmp_path / "_attempts/runtime-00000/attempt-a"
    payload = _stats(generation, 0, 1, files=29_300, rows=29_300)
    monkeypatch.setattr(
        manifest_module.fsutil, "listdir",
        lambda *_args, **_kwargs: pytest.fail("producer must not list data"))
    monkeypatch.setattr(
        manifest_module.fsutil, "object_identity",
        lambda *_args, **_kwargs: pytest.fail("producer must not stat parts"))

    inventory = inventory_generation(str(generation), PREDICTIONS, payload)
    assert inventory[0].file_count == 29_300


def test_multi_shard_selection_is_zero_copy_and_o_shards(monkeypatch, tmp_path):
    refs = [
        _shard(tmp_path, attempt="attempt-a", runtime=0, runtimes=2, start=0,
               stop=1, files=2),
        _shard(tmp_path, attempt="attempt-b", runtime=1, runtimes=2, start=1,
               stop=2, files=3),
    ]
    monkeypatch.setattr(
        manifest_module.fsutil, "copy_file",
        lambda *_args, **_kwargs: pytest.fail("selection must not copy data"))
    ref = _select(tmp_path, refs)
    selected = load_current_prediction_selection(tmp_path.as_posix())

    artifact = selected.artifacts[0]
    assert artifact.roots == (
        str(tmp_path / "_attempts/runtime-00000/attempt-a/predictions.parquet"),
        str(tmp_path / "_attempts/runtime-00001/attempt-b/predictions.parquet"),
    )
    assert artifact.file_count == 5
    assert Path(ref.manifest_path).stat().st_size < 16 * 1024
    assert all(Path(root).exists() for root in artifact.roots)


def test_late_selection_attempt_cannot_replace_create_once_pointer(tmp_path):
    refs = [_shard(tmp_path, attempt="accepted")]
    winner = _select(tmp_path, refs, attempt="finalizer-a")
    late = _select(tmp_path, refs, attempt="finalizer-b")
    selected = load_current_prediction_selection_metadata(str(tmp_path))

    assert late.manifest_path == winner.manifest_path
    assert selected.selection_attempt_id == "finalizer-a"
    assert not (tmp_path /
                "_selections/selection/attempts/finalizer-b").exists()


def test_selected_root_mutation_fails_object_validation(tmp_path):
    refs = [_shard(tmp_path, attempt="accepted")]
    _select(tmp_path, refs)
    root = tmp_path / "_attempts/runtime-00000/accepted/predictions.parquet"
    (root / "unexpected.txt").write_text("late")

    load_current_prediction_selection_metadata(str(tmp_path))
    with pytest.raises(ValueError, match="invalid data"):
        load_current_prediction_selection(str(tmp_path))


def test_v1_manifest_is_intentionally_rejected(tmp_path):
    ref = _shard(tmp_path, attempt="attempt-a")
    value = json.loads(Path(ref.path).read_text())
    value["schema_version"] = 1
    Path(ref.path).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reference changed"):
        _select(tmp_path, [ref])


@pytest.mark.parametrize("producer_abis,accepted", [
    (["kumo-pred-output-v2"], True),
    (["kumo-pred-output-v2", "kumo-pred-output-v3"], False),
    (["kumo-pred-output-v2", "kumo-pred-output-v2"], False),
])
def test_selection_requires_shard_count_producer_abi(tmp_path, producer_abis,
                                                     accepted):
    refs = []
    for index, abi in enumerate(producer_abis):
        ref = _shard(tmp_path, attempt=f"attempt-{index}", runtime=index,
                     runtimes=len(producer_abis), start=index,
                     stop=index + 1)
        # Rewrite as the producing release would have written it.
        value = json.loads(Path(ref.path).read_text())
        value["producer_abi"] = abi
        encoded = json.dumps(value, indent=2).encode()
        Path(ref.path).write_bytes(encoded)
        refs.append(
            dataclasses.replace(ref,
                                sha256=hashlib.sha256(encoded).hexdigest(),
                                size_bytes=len(encoded)))
    if accepted:
        _select(tmp_path, refs)
    else:
        with pytest.raises(PredictionProducerMismatchError,
                           match="version change.*Re-run"):
            _select(tmp_path, refs)


def test_hosted_delegates_to_core_and_preserves_json_bytes(tmp_path):
    from kumo_core.train import artifact_manifest as shared
    from kumo_core.train._artifact_storage import _LOCAL_STORAGE

    from io_utils import fsutil
    from io_utils.json import save_json_atomic
    from kumo.train import artifact_manifest as hosted

    assert hosted.PredictionShardManifest is shared.PredictionShardManifest
    assert hosted.publish_prediction_selection.func is shared.publish_prediction_selection
    assert hosted.publish_shard_manifest.func is shared.publish_shard_manifest
    assert hosted._STORAGE.fs is fsutil
    assert hosted._STORAGE.save_json_atomic is save_json_atomic
    value = {'schema_version': 2, 'nested': ['x', 4]}
    local_path, hosted_path = str(tmp_path / 'local'), str(tmp_path / 'hosted')
    _LOCAL_STORAGE.save_json_atomic(value, local_path)
    save_json_atomic(value, hosted_path)
    assert fsutil.read_binary(local_path) == fsutil.read_binary(hosted_path)


@pytest.mark.parametrize('hosted', [False, True])
def test_manifest_limit_matches_stored_encoding(tmp_path, hosted):
    from kumo_core.train._artifact_storage import _LOCAL_STORAGE
    from kumo_core.train.artifact_manifest import _write_manifest

    storage = manifest_module._STORAGE if hosted else _LOCAL_STORAGE
    value = {'roots': ['root-' + str(i) for i in range(100)]}
    compact = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    stored = json.dumps(value, indent=2).encode()
    assert len(stored) > len(compact)
    path = tmp_path / 'manifest.json'
    with pytest.raises(ValueError, match='size limit'):
        _write_manifest(str(path), value, len(compact), _storage=storage)
    assert not path.exists()
    encoded = _write_manifest(str(path), value, len(stored), _storage=storage)
    assert encoded == stored == path.read_bytes()
