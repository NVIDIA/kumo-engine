# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_core.job_config import PredictionConfig
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_STATS_PATH_ENV,
    PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
    PredictionArtifactStatsRequest,
)
from kumo_core.types import PredictionArtifactType, TaskType
from kumoapi.jobs import PredictionProgress

from io_utils.json import load_json, save_json
from kumo.config.prediction import PartitionedPredictionProgress


def _load_kumo_pred_module():
    path = Path(__file__).parents[2] / "bin" / "kumo-pred"
    loader = SourceFileLoader("kumo_pred_bin", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_stats_request_binds_config_and_runtime_identity(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    config = PredictionConfig(
        home_dir=str(tmp_path / "job"),
        train_dir=str(tmp_path / "train"),
        task_name="task",
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "output"),
        return_predictions=True,
        return_embeddings=True,
        pred_table_path=str(tmp_path / "pred-table"),
        diskgraph_task_dataset_bundle_dir=str(tmp_path / "bundle"),
        num_partitions=2,
        output_num_partitions=4,
        start_partition_id=1,
        stop_partition_id=2,
    )
    config_path = tmp_path / "prediction-config.json"
    save_json(config, str(config_path))
    request = PredictionArtifactStatsRequest(
        attempt_id="attempt-1",
        job_id="bp-job-1",
        runtime_config_sha256=hashlib.sha256(
            config_path.read_bytes()).hexdigest(),
        output_dir=config.output_dir,
        runtime_shard_id=1,
        runtime_shard_count=2,
        output_partition_start=2,
        output_partition_stop=4,
        expected_artifact_types=(
            PredictionArtifactType.EMBEDDINGS,
            PredictionArtifactType.PREDICTIONS,
        ),
    )
    payload_path = str(tmp_path / "producer-stats.json")
    monkeypatch.setenv(PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
                       json.dumps(request.to_dict()))
    monkeypatch.setenv(PREDICTION_ARTIFACT_STATS_PATH_ENV, payload_path)

    assert mod._artifact_stats_request(str(config_path),
                                       config) == (payload_path, request)

    stale = dataclasses.replace(request, runtime_config_sha256="0" * 64)
    monkeypatch.setenv(PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
                       json.dumps(stale.to_dict()))
    with pytest.raises(ValueError, match="config digest mismatch"):
        mod._artifact_stats_request(str(config_path), config)

    predictions_only = dataclasses.replace(
        request,
        expected_artifact_types=(PredictionArtifactType.PREDICTIONS, ),
    )
    monkeypatch.setenv(PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
                       json.dumps(predictions_only.to_dict()))
    with pytest.raises(ValueError, match="output set mismatch"):
        mod._artifact_stats_request(str(config_path), config)


def test_artifact_stats_payload_is_not_published_when_prediction_fails(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    collector = SimpleNamespace(build_payload=pytest.fail)

    class Trainer:
        log_dir = str(tmp_path / "model" / "version_0")
        encoders = object()

        def predict(self, **_kwargs):
            raise RuntimeError("injected output upload failure")

    config = PredictionConfig(
        home_dir=str(tmp_path),
        train_dir=str(tmp_path / "train"),
        task_name="task",
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "output"),
        return_predictions=True,
        return_embeddings=False,
        pred_table_path=str(tmp_path / "pred-table"),
        diskgraph_task_dataset_bundle_dir=str(tmp_path / "bundle"),
    )
    request = PredictionArtifactStatsRequest(
        attempt_id="attempt-1",
        job_id="bp-job-1",
        runtime_config_sha256="a" * 64,
        output_dir=config.output_dir,
        runtime_shard_id=0,
        runtime_shard_count=1,
        output_partition_start=0,
        output_partition_stop=1,
        expected_artifact_types=(PredictionArtifactType.PREDICTIONS, ),
    )
    task = SimpleNamespace(to_resolved_numerical_task=lambda _: object())
    publish_calls = []

    monkeypatch.setattr(mod.DiskgraphTaskManifest, "load", lambda _: object())
    monkeypatch.setattr(mod, "load_encoders", lambda _: object())
    monkeypatch.setattr(
        mod,
        "load_json",
        lambda *_: PredictionProgress(total_iterations=1),
    )
    monkeypatch.setattr(
        mod,
        "PredictionArtifactStatsCollector",
        lambda _output_dir, _artifact_types: collector,
    )
    monkeypatch.setattr(
        mod,
        "save_json_atomic",
        lambda *_args, **_kwargs: publish_calls.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphNestedTrainer.load",
        lambda *_, **__: Trainer(),
    )

    with pytest.raises(RuntimeError, match="output upload failure"):
        mod.run_batch_prediction(
            config,
            task,
            artifact_stats_request=(str(tmp_path / "stats.json"), request),
        )

    assert publish_calls == []


def test_ttng_run_batch_prediction_shards_runtime_iterator(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    encoders = object()
    resolved_task = object()
    captured = []
    loaded_bundles = []
    loaded_runtime_keys = []

    class Trainer:
        log_dir = "/models/task/version_0"
        encoders = None

        def predict(self, **kwargs):
            captured.append(kwargs)

    def load_trainer(*_, **kwargs):
        loaded_bundles.append(kwargs["diskgraph_task_dataset_bundle_dir"])
        loaded_runtime_keys.append(
            {key
             for key in kwargs if key.endswith("_runtime")})
        return Trainer()

    pred_config = PredictionConfig(
        home_dir=str(tmp_path),
        train_dir="/home/train",
        task_name="task",
        model_dir="/models/task",
        output_dir="/home/bp/output",
        return_embeddings=True,
        return_predictions=False,
        pred_table_path="/home/bp/pred_table",
        diskgraph_task_dataset_bundle_dir="/dg/task_datasets/bp",
        num_partitions=2,
        output_num_partitions=4,
        start_partition_id=1,
        stop_partition_id=2,
        start_time=datetime(2026, 6, 4, 12, 0, 0),
        ignore_time_col=True,
        pred_time_col_name="ANCHOR_TIMESTAMP",
        max_prediction_file_size_bytes=12345,
        top_k=3,
    )
    task = SimpleNamespace(
        to_resolved_numerical_task=lambda _: resolved_task,
        to_task_config=lambda: object(),
    )

    monkeypatch.setattr(mod.DiskgraphTaskManifest, "load", lambda _: object())
    monkeypatch.setattr(mod, "load_encoders", lambda _: encoders)
    monkeypatch.setattr(
        mod,
        "load_json",
        lambda *_: PredictionProgress(total_iterations=7),
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphNestedTrainer.load",
        load_trainer,
    )

    mod.run_batch_prediction(pred_config, task, model_dir="/models/task")

    assert loaded_bundles == [
        "/dg/task_datasets/bp",
    ]
    assert loaded_runtime_keys == [{
        "artifact_runtime",
        "diskgraph_runtime",
        "explain_runtime",
    }]
    assert [call["partition_id"] for call in captured] == [2]
    for call in captured:
        assert call == {
            "output_dir": pred_config.output_dir,
            "task": resolved_task,
            "return_embeddings": True,
            "return_predictions": False,
            "job_start_time": pred_config.start_time,
            "ignore_time_col": True,
            "pred_time_col_name": "ANCHOR_TIMESTAMP",
            "output_extra_fields": {},
            "max_output_file_size": 12345,
            "progress_log_dir": pred_config.home_dir,
            "progress_log_id": 1,
            "partition_id": call["partition_id"],
            "total_partitions": 4,
            "output_partition_ids": [2, 3],
            "runtime_num_workers": 2,
            "runtime_worker_id": 1,
            "top_k": 3,
        }
    partition_progress = load_json(
        str(tmp_path / "partitioned_progress_1.json"),
        PartitionedPredictionProgress,
    )
    assert partition_progress.total_partitions == 1
    assert partition_progress.completed_partitions == 1
    prediction_progress = load_json(
        str(tmp_path / "prediction_progress_1.json"),
        PredictionProgress,
    )
    assert prediction_progress.total_iterations == 7


def test_ttng_run_batch_prediction_explains_first_node_pred_worker(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    encoders = object()
    resolved_task = object()
    calls = []

    class Trainer:
        log_dir = "/models/task/version_0"
        encoders = None

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))

        def set_runtime_worker_shard(self, **kwargs):
            calls.append(("set_runtime_worker_shard", kwargs))

        def explain(self, **kwargs):
            calls.append(("explain", kwargs))

    def load_trainer(*_, **kwargs):
        return Trainer()

    pred_config = PredictionConfig(
        home_dir=str(tmp_path),
        train_dir="/home/train",
        task_name="task",
        model_dir="/models/task",
        output_dir="/home/bp/output",
        return_embeddings=False,
        return_predictions=True,
        pred_table_path="/home/bp/pred_table",
        diskgraph_task_dataset_bundle_dir="/dg/task_datasets/bp",
        disable_explain=False,
        num_partitions=2,
        start_partition_id=0,
        stop_partition_id=1,
        top_k=3,
    )
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        to_resolved_numerical_task=lambda _: resolved_task,
        to_task_config=lambda: object(),
    )

    monkeypatch.setattr(mod.DiskgraphTaskManifest, "load", lambda _: object())
    monkeypatch.setattr(mod, "load_encoders", lambda _: encoders)
    monkeypatch.setattr(
        mod,
        "load_json",
        lambda *_: PredictionProgress(total_iterations=7),
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphNestedTrainer.load",
        load_trainer,
    )

    mod.run_batch_prediction(pred_config, task, model_dir="/models/task")

    assert [name for name, _ in calls] == [
        "predict",
        "set_runtime_worker_shard",
        "explain",
    ]
    assert calls[1][1] == {
        "runtime_num_workers": 1,
        "runtime_worker_id": 0,
    }
    assert calls[2][1] == {
        "explain_methods":
        ['column_analysis', 'gradient', 'subgraph_collector'],
        "stage": 'pred',
        "root_dir": str(tmp_path / "xai"),
        "top_k": 3,
    }


def test_ttng_run_batch_prediction_explains_first_link_pred_worker(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    encoders = object()
    resolved_task = object()
    calls = []

    class Trainer:
        log_dir = "/models/task/version_0"
        encoders = None

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))

        def set_runtime_worker_shard(self, **kwargs):
            calls.append(("set_runtime_worker_shard", kwargs))

        def explain(self, **kwargs):
            calls.append(("explain", kwargs))

    pred_config = PredictionConfig(
        home_dir=str(tmp_path),
        train_dir="/home/train",
        task_name="task",
        model_dir="/models/task",
        output_dir="/home/bp/output",
        return_embeddings=False,
        return_predictions=True,
        pred_table_path="/home/bp/pred_table",
        diskgraph_task_dataset_bundle_dir="/dg/task_datasets/bp",
        disable_explain=False,
        num_partitions=1,
        start_partition_id=0,
        stop_partition_id=1,
        top_k=5,
    )
    task = SimpleNamespace(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        to_resolved_numerical_task=lambda _: resolved_task,
        to_task_config=lambda: object(),
    )

    monkeypatch.setattr(mod.DiskgraphTaskManifest, "load", lambda _: object())
    monkeypatch.setattr(mod, "load_encoders", lambda _: encoders)
    monkeypatch.setattr(
        mod,
        "load_json",
        lambda *_: PredictionProgress(total_iterations=7),
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphNestedTrainer.load",
        lambda *_, **__: Trainer(),
    )

    mod.run_batch_prediction(pred_config, task, model_dir="/models/task")

    assert [name for name, _ in calls] == [
        "predict",
        "set_runtime_worker_shard",
        "explain",
    ]
    assert calls[1][1] == {
        "runtime_num_workers": 1,
        "runtime_worker_id": 0,
    }
    assert calls[2][1] == {
        "explain_methods":
        ['column_analysis', 'gradient', 'subgraph_collector'],
        "stage": 'pred',
        "root_dir": str(tmp_path / "xai"),
        "top_k": 5,
    }


def test_ttng_run_batch_prediction_skips_explain_on_nonfirst_worker(
        monkeypatch, tmp_path):
    mod = _load_kumo_pred_module()
    encoders = object()
    resolved_task = object()
    calls = []

    class Trainer:
        log_dir = "/models/task/version_0"
        encoders = None

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))

        def set_runtime_worker_shard(self, **kwargs):
            calls.append(("set_runtime_worker_shard", kwargs))

        def explain(self, **kwargs):
            calls.append(("explain", kwargs))

    pred_config = PredictionConfig(
        home_dir=str(tmp_path),
        train_dir="/home/train",
        task_name="task",
        model_dir="/models/task",
        output_dir="/home/bp/output",
        return_embeddings=False,
        return_predictions=True,
        pred_table_path="/home/bp/pred_table",
        diskgraph_task_dataset_bundle_dir="/dg/task_datasets/bp",
        disable_explain=False,
        num_partitions=2,
        start_partition_id=1,
        stop_partition_id=2,
    )
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        to_resolved_numerical_task=lambda _: resolved_task,
        to_task_config=lambda: object(),
    )

    monkeypatch.setattr(mod.DiskgraphTaskManifest, "load", lambda _: object())
    monkeypatch.setattr(mod, "load_encoders", lambda _: encoders)
    monkeypatch.setattr(
        mod,
        "load_json",
        lambda *_: PredictionProgress(total_iterations=7),
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphNestedTrainer.load",
        lambda *_, **__: Trainer(),
    )

    mod.run_batch_prediction(pred_config, task, model_dir="/models/task")

    assert [name for name, _ in calls] == ["predict"]


def test_ttng_link_pred_predict_receives_runtime_and_top_k():
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer

    calls = []

    class Trainer:
        task = SimpleNamespace(task_type=TaskType.TEMPORAL_LINK_PREDICTION)
        set_runtime_worker_shard = (
            DiskgraphNestedTrainer.set_runtime_worker_shard)

        def _predict_lp(self, **kwargs):
            calls.append(kwargs)
            return object()

    trainer = Trainer()
    manifest = object()

    DiskgraphNestedTrainer.predict(
        trainer,
        output_dir="/home/bp/output",
        task=object(),
        runtime_num_workers=3,
        runtime_worker_id=2,
        top_k=7,
        task_manifest=manifest,  # type: ignore[arg-type]
    )

    assert trainer._runtime_num_workers == 3
    assert trainer._runtime_worker_id == 2
    assert calls == [{
        "output_dir": "/home/bp/output",
        "task": calls[0]["task"],
        "job_start_time": None,
        "return_embeddings": False,
        "return_predictions": True,
        "ignore_time_col": False,
        "pred_time_col_name": None,
        "output_extra_fields": None,
        "output_constant_columns": None,
        "max_output_file_size": 1024**3,
        "progress_log_dir": None,
        "progress_log_id": 0,
        "partition_id": 0,
        "total_partitions": 1,
        "output_partition_ids": None,
        "top_k": 7,
        "task_manifest": manifest,
    }]


def test_ttng_set_runtime_worker_shard_updates_existing_datamodule():
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer

    trainer = SimpleNamespace(
        _runtime_num_workers=3,
        _runtime_worker_id=2,
        datamodule=SimpleNamespace(
            runtime_num_workers=3,
            runtime_worker_id=2,
        ),
    )

    DiskgraphNestedTrainer.set_runtime_worker_shard(
        trainer,
        runtime_num_workers=1,
        runtime_worker_id=0,
    )

    assert trainer._runtime_num_workers == 1
    assert trainer._runtime_worker_id == 0
    assert trainer.datamodule.runtime_num_workers == 1
    assert trainer.datamodule.runtime_worker_id == 0
