# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import PredictionArtifactType

from io_utils import fsutil, parquet_util
from rest.backends.ml_backend import prediction_artifacts
from rest.backends.ml_backend.activities import \
    parquet_io_activities as activity_module
from rest.backends.ml_backend.activities.parquet_io_activities import (
    _get_selected_parquet_metadata,
    get_artifact_stats,
)


def _write(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'ID': values, 'SCORE': [0.5] * len(values)}), path)


def test_selected_stats_reads_only_files_from_selected_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / 'attempt-a' / 'predictions.parquet'
    second_root = tmp_path / 'attempt-b' / 'predictions.parquet'
    first = first_root / 'part-00000-00000.parquet'
    second = second_root / 'part-00001-00000.parquet'
    _write(first, [1, 2])
    _write(second, [3])

    monkeypatch.setattr(
        parquet_util,
        'get_parquet_metadata',
        lambda *_args, **_kwargs: pytest.fail('must not scan selected root'),
    )
    metadata = _get_selected_parquet_metadata([str(first), str(second)])
    assert metadata.num_rows == 3
    assert metadata.cols == ['ID', 'SCORE']

    selection_ref = object()
    monkeypatch.setattr(
        activity_module,
        'load_prediction_selection',
        lambda value, **_kwargs: SimpleNamespace(artifacts=(SimpleNamespace(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            root=str(first_root),
            roots=(str(first_root), str(second_root)),
            file_count=2,
            num_rows=None,
            num_cols=None,
            size_bytes=None,
        ), )) if value is selection_ref else pytest.fail('wrong selection'),
    )
    stats = get_artifact_stats(
        {},
        selection_ref,
        [PredictionArtifactType.PREDICTIONS],
    )
    assert stats[PredictionArtifactType.PREDICTIONS].num_rows == 3


def test_selected_producer_stats_skip_all_parquet_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_ref = object()
    selected = tmp_path / 'selected'
    part = selected / 'part-00000-00000.parquet'
    selected.mkdir()
    part.write_bytes(b'not-read')
    monkeypatch.setattr(
        activity_module,
        'load_prediction_selection',
        lambda value, **_kwargs: SimpleNamespace(artifacts=(SimpleNamespace(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            root=str(selected),
            roots=(str(selected), ),
            file_count=1,
            num_rows=123,
            num_cols=3,
            size_bytes=4567,
        ), )) if value is selection_ref else pytest.fail('wrong selection'),
    )
    monkeypatch.setattr(
        activity_module,
        '_get_selected_parquet_metadata',
        lambda *_args, **_kwargs: pytest.fail('must not read Parquet'),
    )
    monkeypatch.setattr(
        parquet_util,
        'get_parquet_metadata',
        lambda *_args, **_kwargs: pytest.fail('must not scan selected root'),
    )

    stats = get_artifact_stats(
        {},
        selection_ref,
        [PredictionArtifactType.PREDICTIONS],
    )

    result = stats[PredictionArtifactType.PREDICTIONS]
    assert result.artifact_type is PredictionArtifactType.PREDICTIONS
    assert result.num_rows == 123
    assert result.num_cols == 3
    assert result.size_bytes == 4567
    assert result.output_path == str(selected)


def test_selected_producer_stats_1800_file_path_is_metadata_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_ref = object()
    expected = {
        PredictionArtifactType.PREDICTIONS:
        (1700, 26_628_349_900, 3, 1_740_136_037_615),
        PredictionArtifactType.EMBEDDINGS:
        (100, 266_288_665, 3, 84_933_302_279),
    }
    artifacts = []
    for artifact_type, (count, rows, cols, size_bytes) in expected.items():
        root = tmp_path / artifact_type.output_file_name
        artifacts.append(
            SimpleNamespace(
                artifact_type=artifact_type,
                root=str(root),
                roots=(str(root), ),
                file_count=count,
                num_rows=rows,
                num_cols=cols,
                size_bytes=size_bytes,
            ))
    monkeypatch.setattr(
        activity_module,
        'load_prediction_selection',
        lambda value, **_kwargs: SimpleNamespace(artifacts=tuple(artifacts))
        if value is selection_ref else pytest.fail('wrong selection'),
    )
    monkeypatch.setattr(
        activity_module,
        '_get_selected_parquet_metadata',
        lambda *_args, **_kwargs: pytest.fail('must not read Parquet'),
    )
    monkeypatch.setattr(
        parquet_util,
        'get_parquet_metadata',
        lambda *_args, **_kwargs: pytest.fail('must not read Parquet'),
    )
    monkeypatch.setattr(fsutil, 'listdir', lambda root: [
        f'{root}/part-{index:05d}-00000.parquet'
        for index in range(next(artifact.file_count for artifact in artifacts
                                if artifact.root == root))
    ])
    monkeypatch.setattr(
        fsutil,
        'get_binary_handle',
        lambda *_args, **_kwargs: pytest.fail('must not read artifact objects'),
    )

    started = time.perf_counter()
    stats = get_artifact_stats({}, selection_ref, list(expected))
    elapsed = time.perf_counter() - started

    assert sum(artifact.file_count for artifact in artifacts) == 1800
    assert elapsed < 1.007
    for artifact_type, (_, rows, cols, size_bytes) in expected.items():
        assert stats[artifact_type].num_rows == rows
        assert stats[artifact_type].num_cols == cols
        assert stats[artifact_type].size_bytes == size_bytes


def test_selected_stats_rejects_schema_mismatch(tmp_path: Path) -> None:
    first = tmp_path / 'first.parquet'
    second = tmp_path / 'second.parquet'
    _write(first, [1])
    pq.write_table(pa.table({'OTHER': [1]}), second)

    with pytest.raises(ValueError, match='schemas are inconsistent'):
        _get_selected_parquet_metadata([str(first), str(second)])


def test_finalizer_uses_temporal_attempt_for_selection_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / 'prediction-config.json'
    config_path.write_text('{}')
    captured = {}
    expected = object()
    monkeypatch.setattr(
        activity_module.PredictionConfig,
        'load',
        lambda _path: SimpleNamespace(
            output_dir=str(tmp_path / 'output'), num_partitions=1,
            output_num_partitions=97, effective_output_num_partitions=97,
            diskgraph_task_dataset_bundle_dir=('/tmp/bundle')),
    )
    monkeypatch.setattr(activity_module.activity, 'info',
                        lambda: SimpleNamespace(attempt=2))

    def publish(refs, **kwargs):
        captured.update(kwargs)
        assert refs == []
        return expected

    monkeypatch.setattr(activity_module, 'publish_prediction_selection',
                        publish)
    actual = activity_module.finalize_prediction_job_output(
        SimpleNamespace(id='bp-job-1'),
        str(config_path),
        'graph@version',
        'selection',
        [],
        1,
    )

    assert actual is expected
    assert captured['selection_id'] == 'selection'
    assert captured['selection_attempt_id'] == 'temporal-00002'
    assert captured['expected_output_partitions'] == 97


def test_finalizer_reports_mixed_producers_as_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kumo.train.artifact_manifest import PredictionProducerMismatchError
    from statedb.utils.errors import PredictionOutputProcessingError

    config_path = tmp_path / 'prediction-config.json'
    config_path.write_text('{}')
    monkeypatch.setattr(
        activity_module.PredictionConfig, 'load',
        lambda _path: SimpleNamespace(
            output_dir=str(tmp_path / 'output'), num_partitions=2,
            effective_output_num_partitions=2,
            diskgraph_task_dataset_bundle_dir='/tmp/bundle'))
    monkeypatch.setattr(activity_module.activity, 'info',
                        lambda: SimpleNamespace(attempt=1))

    def publish(_refs, **_kwargs):
        raise PredictionProducerMismatchError()

    monkeypatch.setattr(activity_module, 'publish_prediction_selection',
                        publish)
    with pytest.raises(PredictionOutputProcessingError,
                       match='Re-run') as raised:
        activity_module.finalize_prediction_job_output(
            SimpleNamespace(id='bp-job-1'), str(config_path),
            'graph@version', 'selection', [], 2)
    assert raised.value.non_retryable


@pytest.mark.asyncio
async def test_selected_artifact_validation_never_blocks_event_loop(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kumo.home import HomeDirManager
    from rest.backends.ml_backend import prediction_artifacts as module

    home = HomeDirManager(str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(module.fsutil, 'exists', lambda _path: True)

    def blocking_selection(_root):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError('test did not release resolver')
        return SimpleNamespace(artifacts=())

    monkeypatch.setattr(module, 'load_current_prediction_selection_metadata',
                        blocking_selection)
    task = asyncio.create_task(
        prediction_artifacts.resolve_prediction_artifact.coroutine(
            home, PredictionArtifactType.PREDICTIONS))
    assert await asyncio.to_thread(started.wait, 2)

    # A direct provider scan on this event loop would prevent this yield.
    ticked = False

    async def tick():
        nonlocal ticked
        await asyncio.sleep(0)
        ticked = True

    await tick()
    assert ticked
    release.set()
    assert await task is None
