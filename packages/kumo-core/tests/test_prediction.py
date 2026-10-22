# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import kumo_core.prediction as prediction
import pytest
import torch
from kumo_core.pquery import (
    NativeGraph,
    NativeGraphTable,
    ValidatedNativeQuery,
)
from kumo_core.prediction import (
    PredictionTableRequest,
    PreparedCustomPredictionTable,
    forecast_task_options,
    generate_custom_prediction_table,
    generate_prediction_table,
    materialize_prediction_task,
)
from kumo_core.prediction_options import (
    resolve_classification_threshold,
    resolve_num_classes_to_return,
)
from kumo_core.task_dataset import TaskDatasetMaterializeSpec
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
)
from kumo_core.types import TaskType
from kumoapi.typing import TimeUnit
from pandas import Timestamp


@pytest.mark.parametrize("task_type", [TaskType.STATIC_LINK_PREDICTION,
                                       TaskType.TEMPORAL_LINK_PREDICTION])
@pytest.mark.parametrize("count", [1, 500])
def test_link_prediction_count_accepts_boundaries(task_type, count):
    assert resolve_num_classes_to_return(task_type, count, top_k=3) == count


@pytest.mark.parametrize("task_type", [TaskType.STATIC_LINK_PREDICTION,
                                       TaskType.TEMPORAL_LINK_PREDICTION])
@pytest.mark.parametrize("count, message", [
    (None, "specify the number of predictions"),
    (501, "less than or equal to 500"),
    (0, "positive integer"),
    (-1, "positive integer"),
    (True, "positive integer"),
    (1.5, "positive integer"),
])
def test_link_prediction_count_rejects_invalid_values(task_type, count, message):
    with pytest.raises(ValueError, match=message):
        resolve_num_classes_to_return(task_type, count, top_k=3)


@pytest.mark.parametrize("task_type", [TaskType.STATIC_LINK_PREDICTION,
                                       TaskType.TEMPORAL_LINK_PREDICTION])
@pytest.mark.parametrize("count", [None, 501])
def test_link_embeddings_clear_count_without_prediction_limit(task_type, count):
    assert resolve_num_classes_to_return(
        task_type, count, top_k=3, has_predictions=False) is None


@pytest.mark.parametrize("task_type", [TaskType.MULTICLASS_CLASSIFICATION,
                                       TaskType.MULTILABEL_CLASSIFICATION,
                                       TaskType.MULTILABEL_RANKING])
def test_classification_count_defaults_and_overrides(task_type):
    assert resolve_num_classes_to_return(task_type, None, top_k=3) == 3
    assert resolve_num_classes_to_return(task_type, 501, top_k=3) == 501
    assert resolve_num_classes_to_return(task_type, None) is None
    assert resolve_num_classes_to_return(
        task_type, 2, top_k=3, has_predictions=False) is None


@pytest.mark.parametrize("task_type", [TaskType.REGRESSION,
                                       TaskType.BINARY_CLASSIFICATION,
                                       TaskType.FORECASTING])
def test_prediction_count_rejects_tasks_with_single_output(task_type):
    with pytest.raises(ValueError, match="can only be specified"):
        resolve_num_classes_to_return(task_type, 3)


@pytest.mark.parametrize(
    "task_type",
    [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION],
)
def test_classification_threshold_defaults_thresholded_tasks(task_type):
    assert resolve_classification_threshold(task_type, None) == 0.5


def test_binary_classification_threshold_can_be_required():
    with pytest.raises(ValueError, match="must be specified"):
        resolve_classification_threshold(
            TaskType.BINARY_CLASSIFICATION,
            None,
            binary_required=True,
        )


@pytest.mark.parametrize("threshold", [0.0, 0.25, 1.0])
def test_classification_threshold_preserves_valid_boundaries(threshold):
    assert resolve_classification_threshold(
        TaskType.MULTILABEL_CLASSIFICATION,
        threshold,
        binary_required=True,
    ) == threshold


@pytest.mark.parametrize(
    "threshold",
    [-0.01, 1.01, float("nan"),
     float("inf"), float("-inf")],
)
def test_classification_threshold_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_classification_threshold(
            TaskType.MULTILABEL_CLASSIFICATION,
            threshold,
        )


@pytest.mark.parametrize("threshold", [None, 0.0, 0.75])
def test_irrelevant_classification_threshold_is_cleared(threshold):
    assert resolve_classification_threshold(
        TaskType.REGRESSION,
        threshold,
        binary_required=True,
    ) is None


def request(tmp_path, *, family="node_prediction"):
    task_type = ("static_link_prediction"
                 if family == "link_prediction" else "binary_classification")
    query = ValidatedNativeQuery(
        query=SimpleNamespace(entity_column="users.id"),
        task_type=task_type,
        task_family=family,
        entity_table="users",
        entity_column="id",
        target_stype="categorical",
        query_type="static",
        rhs_node_type="items" if family == "link_prediction" else None,
        rhs_is_temporal=False,
        list_distinct_target_is_array=family == "link_prediction",
        forecast_length=1,
    )
    graph = NativeGraph((NativeGraphTable(
        name="users",
        source_table_name="main.s.users",
        columns=(),
        primary_key="id",
        time_column=None,
        end_time_column=None,
        foreign_keys=(),
    ), ))
    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path=str(tmp_path / "extension" / "manifest.json"),
        job_id="pred-1",
        artifact_root=str(tmp_path / "bundle"),
        task_family=family,
        task_type=task_type,
        is_temporal=False,
        seed_node_type="users",
        train_table_name="pred_table",
        entity_table_name="users",
        rhs_node_type=("items" if family == "link_prediction" else None),
        feature_schema={
            "node_types": {
                "users": {
                    "column_specs": [],
                    "feature_columns": [],
                    "mapper_key": "users",
                    "time_column": None,
                }
            },
            "target_encoder_input": None,
        },
        graph_node_types=("users", ),
        graph_edge_types=(),
    )
    return PredictionTableRequest(
        query=query,
        graph=graph,
        timestamps=(),
        index_path=str(tmp_path / "index"),
        tt_id="pred-1",
        task_dataset=spec,
    )


class Runtime:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.compile_options = None

    def DiskGraphPlanCompiler(self):
        runtime = self

        class Compiler:
            def compile(self, **kwargs):
                runtime.compile_options = kwargs
                return "plan"

        return Compiler()

    def execute_tt_gen(self, plan, index_path, tt_id):
        assert (plan, index_path, tt_id) == (
            "resolved-plan",
            str(self.tmp_path / "index"),
            "pred-1",
        )
        return {
            "staging_dir": str(self.tmp_path / "staging"),
            "tt_parquet_path": str(self.tmp_path / "staging" / "tt.parquet"),
            "row_count": 7,
        }


def test_generate_prediction_table_uses_prediction_mode(monkeypatch, tmp_path):
    runtime = Runtime(tmp_path)
    monkeypatch.setattr(
        prediction,
        "apply_reference_resolution",
        lambda plan, **kwargs: "resolved-plan",
    )

    result = generate_prediction_table(request(tmp_path), runtime=runtime)

    assert result.row_count == 7
    assert runtime.compile_options["is_prediction"] is True
    assert runtime.compile_options["task_type"] == "binary_classification"


def test_generate_custom_prediction_table_builds_predict_bundle(tmp_path):
    parquet = tmp_path / "custom.parquet"
    parquet.touch()
    calls = []

    class CustomRuntime:
        @staticmethod
        def build_prediction_table_artifacts_for_existing_table(**kwargs):
            calls.append(kwargs)
            return {
                "staging_dir": str(tmp_path / "staging"),
                "tt_parquet_path": str(parquet),
                "row_count": 3,
            }

    result = generate_custom_prediction_table(
        request(tmp_path),
        PreparedCustomPredictionTable(str(parquet)),
        runtime=CustomRuntime,
    )

    assert result.row_count == 3
    assert calls == [{
        "output_path": str(parquet),
        "input_timestamp_unit": "nanoseconds",
        "tt_id": "pred-1",
        "split_stats_config": {
            "entity_table": "users",
            "entity_pk_column": "id",
        },
        "train_table_col_stats_config": {
            "target_stype": "categorical",
            "time_col": "TIMESTAMP",
        },
        "index_path": str(tmp_path / "index"),
        "target_entity_table": None,
        "task_type": "binary_classification",
    }]


def test_link_prediction_compiles_lhs_only(monkeypatch, tmp_path):
    runtime = Runtime(tmp_path)
    monkeypatch.setattr(
        prediction,
        "apply_reference_resolution",
        lambda plan, **kwargs: "resolved-plan",
    )

    generate_prediction_table(
        request(tmp_path, family="link_prediction"),
        runtime=runtime,
    )

    assert runtime.compile_options["lp_prediction_mode"] == "LHS"
    assert runtime.compile_options["list_distinct_target_is_array"] is True


def test_materialize_prediction_writes_predict_only_bundle(
        monkeypatch, tmp_path):
    req = request(tmp_path)
    generated = SimpleNamespace(
        staging_dir=str(tmp_path / "staging"),
        table_path=str(tmp_path / "staging" / "tt.parquet"),
        row_count=7,
        tt_id="pred-1",
    )

    class IngestRuntime:
        @staticmethod
        def execute_tt_ingest(**kwargs):
            assert kwargs["predict_stage_name"] == "predict"
            return {
                "tt_extension_path": str(tmp_path / "extension"),
                "task_dataset": {
                    "tt_extension_path": str(tmp_path / "extension"),
                    "stages": {
                        "predict": {
                            "table_name": "pred_table",
                            "row_count": 7,
                            "index_path": str(tmp_path / "extension"),
                        }
                    },
                },
            }

    calls = {}

    def thin(spec, **kwargs):
        calls["thin"] = kwargs
        return str(tmp_path / "bundle" / "manifest.json")

    def flat(spec, **kwargs):
        calls["flat"] = kwargs
        return str(tmp_path / "bundle" / "runtime_manifest.json")

    monkeypatch.setattr(prediction, "write_thin_manifest", thin)
    monkeypatch.setattr(prediction, "write_flat_manifest", flat)

    result = materialize_prediction_task(
        req,
        generated,
        graph_index_id="graph/version",
        runtime=IngestRuntime(),
    )

    assert result.bundle_dir == str(tmp_path / "bundle")
    assert calls["thin"]["stage_counts"] == {"predict": 7}
    assert calls["thin"]["column_stats"] is False
    assert set(calls["thin"]["stage_table_refs"]) == {"predict"}


def forecast_request(tmp_path):
    from kumo_core.pquery import _sequence_encoder_input

    original = request(tmp_path)
    query = replace(
        original.query,
        query=SimpleNamespace(
            entity_column="users.id",
            get_all_target_aggregations=lambda:
            [SimpleNamespace(time_col='orders.ts')],
            get_combined_date_offset_range=lambda: SimpleNamespace(
                start=0, end=1, unit=TimeUnit.DAYS),
        ),
        task_type="forecasting",
        task_family="forecasting",
        query_type="temporal",
        forecast_length=3,
    )
    specs = [
        _sequence_encoder_input(FUTURE_LABELS_COL, 3),
        _sequence_encoder_input(AUTOREGRESSION_COL, 1),
    ]
    schema = {
        "node_types": {
            **original.task_dataset.feature_schema["node_types"], "train_table":
            {
                "column_specs": specs,
                "feature_columns": [FUTURE_LABELS_COL, AUTOREGRESSION_COL],
                "mapper_key": "train_table",
                "time_column": None,
            }
        },
        "target_encoder_input": None,
    }
    spec = replace(
        original.task_dataset,
        task_family="forecasting",
        task_type="forecasting",
        is_temporal=True,
        train_table_name="train_table",
        feature_schema=schema,
    )
    return replace(
        original,
        query=query,
        task_dataset=spec,
        timestamps=(1706486400000000000, ),
        compiler_options={
            "lag_offsets_secs": (-86400, ),
            "future_offsets_secs": (0, 86400, 172800),
        },
    )


def test_forecast_prediction_compiles_only_historical_features(
        monkeypatch, tmp_path):
    runtime = Runtime(tmp_path)
    monkeypatch.setattr(
        prediction, '_temporal_target_time_range', lambda *args:
        (Timestamp('2024-01-01'), Timestamp('2024-06-01'), None))
    monkeypatch.setattr(prediction, "apply_reference_resolution",
                        lambda plan, **kwargs: "resolved-plan")
    req = forecast_request(tmp_path)

    generate_prediction_table(req, runtime=runtime)

    assert runtime.compile_options["lag_offsets_secs"] == (-86400, )
    assert "future_offsets_secs" not in runtime.compile_options
    assert req.compiler_options["future_offsets_secs"] == (0, 86400, 172800)
    assert forecast_task_options(req) == {
        "forecast_length": 3,
        "lag_timesteps": 1,
        "unix_start_offset": 0,
        "unix_end_offset": 86400000000000,
        "unix_step_size": 86400000000000,
    }


def test_forecast_task_options_uses_fitted_timeframe_step(tmp_path):
    req = replace(forecast_request(tmp_path), forecast_timeframe_step=2)

    assert forecast_task_options(req)['unix_step_size'] == 172800000000000


@pytest.mark.parametrize("options", [{
    "query": "changed"
}, {
    "lag_offsets_secs": (-1, )
}])
def test_prediction_rejects_incompatible_compiler_options(tmp_path, options):
    req = (replace(forecast_request(tmp_path),
                   compiler_options=options) if "query" in options else replace(
                       request(tmp_path), compiler_options=options))
    with pytest.raises(ValueError, match="forecast offsets|forecasting query"):
        generate_prediction_table(req, runtime=Runtime(tmp_path))


def test_forecast_materialization_keeps_checkpoint_table_identity(
        monkeypatch, tmp_path):
    req = forecast_request(tmp_path)
    generated = SimpleNamespace(staging_dir=str(tmp_path / "staging"))
    runtime = SimpleNamespace(
        execute_tt_ingest=lambda **kwargs: {
            "task_dataset": {
                "tt_extension_path": str(tmp_path / "extension"),
                "stages": {
                    "predict": {
                        "table_name": "generated_predict_table",
                        "row_count": 7,
                        "index_path": str(tmp_path / "extension"),
                    }
                },
            }
        })
    captured = {}

    def thin(spec, **kwargs):
        captured["spec"] = spec
        return str(tmp_path / "bundle" / "manifest.json")

    monkeypatch.setattr(prediction, "write_thin_manifest", thin)
    monkeypatch.setattr(prediction, "write_flat_manifest",
                        lambda *args, **kwargs: str(tmp_path / "flat.json"))

    materialize_prediction_task(req, generated, graph_index_id="graph",
                                runtime=runtime)

    spec = captured["spec"]
    assert spec.train_table_name == "train_table"
    features = spec.feature_schema["node_types"]["train_table"]
    assert features["feature_columns"] == [AUTOREGRESSION_COL]
    assert req.task_dataset.feature_schema["node_types"]["train_table"][
        "feature_columns"] == [FUTURE_LABELS_COL, AUTOREGRESSION_COL]


def _score_shard_on_device(monkeypatch, tmp_path, devices, worker_id,
                          accelerator='auto'):
    r"""Run one real prediction-worker load into ``create_pl_trainer``."""
    from kumo_core.encoders.tensor_frame_encoder import (
        HeteroTensorFrameEncoder,
    )
    from kumo_core.io.json import save_json
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumo_core.modeling.heads.node_prediction import NodePredHead
    from kumo_core.resolved_task import (
        ResolvedNumericalTask,
        ResolvedTaskGraph,
        ResolvedTrainTable,
    )
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.train.trainer import Trainer
    from kumo_core.types.enums import PredictionArtifactType

    task = ResolvedNumericalTask(
        name="test",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics=["auroc"],
        graph=ResolvedTaskGraph(node_types=("users", ), edge_types=(),
                                seed_node_type="users",
                                target_node_type="users"),
        train_table=ResolvedTrainTable(name="train_table",
                                       target_column="TARGET", time_column=None,
                                       weight_column=None),
    )
    model_dir = tmp_path / "model"
    save_json(
        Trainer(task=task, encoder=HeteroTensorFrameEncoder(),
                model=HeteroGNN(channels=8, num_mp_layers=1),
                head=NodePredHead(), root_dir=str(model_dir)).config(),
        str(model_dir / "config.json"))
    monkeypatch.setattr(DiskgraphTaskManifest, "load", lambda *_: None)
    monkeypatch.setattr(ResolvedNumericalTask, "from_manifest",
                        lambda *_, **__: task)
    monkeypatch.setattr(prediction, "_publish_scored_shard",
                        lambda *_, **__: None)
    trainers = []

    def predict(self, **_):
        # A refit re-runs ``init``; placement must survive the second build.
        for _ in range(2):
            self.create_pl_trainer()
            trainers.append(self.pl_trainer)

    monkeypatch.setattr(DiskgraphNestedTrainer, "predict", predict)
    prediction._score_prediction_shard(
        worker_id=worker_id,
        config=prediction._PredictionWorkerConfig(
            worker_count=len(devices), devices=devices,
            manifest_path="manifest", bundle_dir="bundle",
            model_dir=str(model_dir), run_id="run",
            artifacts=frozenset({PredictionArtifactType.PREDICTIONS}),
            top_k=None, extra_fields=None, job_start_time=None,
            pred_time_col_name=None, output_dir=str(tmp_path / "output"),
            attempt_id="attempt", graph_snapshot_id="snapshot",
            source_config_sha256="0" * 64, runtime_config_sha256="1" * 64,
            binary_threshold=None, forecast_options=None,
            accelerator=accelerator),
    )
    return trainers


def test_prediction_worker_pins_lightning_to_its_device(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    monkeypatch.setenv("KUMO_ACCELERATOR", "cpu")
    monkeypatch.setattr(pl, "Trainer", lambda **kwargs: kwargs)

    trainers = _score_shard_on_device(monkeypatch, tmp_path, (None, 3), 1)

    assert [(t["accelerator"], t["devices"])
            for t in trainers] == [("auto", [3])] * 2


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
def test_prediction_worker_accelerator_overrides_environment(
        monkeypatch, tmp_path, accelerator):
    import pytorch_lightning as pl
    monkeypatch.setenv('KUMO_ACCELERATOR', 'cpu')
    monkeypatch.setattr(pl, 'Trainer', lambda **kwargs: kwargs)
    trainers = _score_shard_on_device(monkeypatch, tmp_path, (None,), 0,
                                    accelerator=accelerator)
    assert [trainer['accelerator'] for trainer in trainers] == [accelerator] * 2


def test_explicit_cpu_runs_lightning_on_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv('KUMO_ACCELERATOR', 'mps')
    trainers = _score_shard_on_device(monkeypatch, tmp_path, (None,), 0,
                                    accelerator='cpu')
    assert all(trainer.strategy.root_device.type == 'cpu'
               for trainer in trainers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prediction_worker_runs_lightning_on_cuda_device(monkeypatch, tmp_path):
    monkeypatch.delenv("KUMO_ACCELERATOR", raising=False)

    trainers = _score_shard_on_device(monkeypatch, tmp_path, (0, ), 0)

    assert [(t.strategy.root_device, t.device_ids)
            for t in trainers] == [(torch.device("cuda", 0), [0])] * 2


@pytest.mark.parametrize('task_type',
                         [TaskType.FORECASTING, TaskType.REGRESSION])
@pytest.mark.parametrize('fields', [
    [],
    ['JOB_TIMESTAMP'],
    ['ANCHOR_TIMESTAMP'],
    ['JOB_TIMESTAMP', 'ANCHOR_TIMESTAMP'],
])
def test_score_prediction_metadata_options(monkeypatch, tmp_path, task_type,
                                           fields):
    from datetime import datetime

    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train import artifact_manifest
    from kumo_core.types import PredictionArtifactType

    kind = PredictionArtifactType.PREDICTIONS
    started = datetime(2026, 1, 2)
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda _: SimpleNamespace(
            task_type=task_type.value, stage_counts={'predict': 1}))
    monkeypatch.setattr(artifact_manifest, 'hash_prediction_config',
                        lambda _: 'runtime-hash')
    monkeypatch.setattr(artifact_manifest, 'publish_prediction_selection',
                        lambda references, **kwargs: references[0])
    monkeypatch.setattr(prediction, '_prediction_worker_devices', lambda _, accelerator=None:
                        (None, ))
    monkeypatch.setattr(prediction, '_score_prediction_shard',
                        lambda *, worker_id, config, **_: config)

    config = prediction.score_prediction(
        materialized=SimpleNamespace(manifest_path='manifest',
                                     bundle_dir='bundle'),
        model_dir=str(tmp_path),
        run_id='prediction',
        scratch_dir=str(tmp_path),
        artifacts=frozenset({kind}),
        extra_fields={kind.value: fields},
        started=started,
        graph_snapshot_id='snapshot',
        source_config_sha256='source',
    )

    forecasting = task_type == TaskType.FORECASTING
    assert config.pred_time_col_name == ('ANCHOR_TIMESTAMP' if not forecasting
                                         and 'ANCHOR_TIMESTAMP' in fields else
                                         None)
    assert config.job_start_time == (started
                                     if 'JOB_TIMESTAMP' in fields else None)
    assert config.extra_fields == {kind.value: fields}
    assert config.ignore_time_col == (not forecasting
                                      and 'ANCHOR_TIMESTAMP' not in fields)


@pytest.mark.parametrize('extra_fields', [None, {}, {'PREDICTIONS': []}])
@pytest.mark.parametrize('task_type',
                         [TaskType.FORECASTING, TaskType.REGRESSION])
def test_absent_metadata_selection_requests_nothing(task_type, extra_fields):
    from kumo_core.prediction_options import resolve_prediction_time_options

    options = resolve_prediction_time_options(task_type, extra_fields)
    assert options.ignore_time_col is (task_type != TaskType.FORECASTING)
    assert options.pred_time_col_name is None
    assert options.include_job_timestamp is False


@pytest.mark.parametrize('year', [2000, 2024, 2030])
def test_temporal_prediction_checks_anchor_before_compilation(
        monkeypatch, tmp_path, year):
    runtime = Runtime(tmp_path)
    req = request(tmp_path)
    temporal = replace(
        req.query, query_type='temporal',
        query=SimpleNamespace(get_all_target_aggregations=lambda:
                              [SimpleNamespace(time_col='orders.ts')]))
    req = replace(req, query=temporal,
                  timestamps=(Timestamp(f'{year}-01-01').value, ))
    monkeypatch.setattr(
        prediction, '_temporal_target_time_range', lambda *args:
        (Timestamp('2024-01-01'), Timestamp('2024-06-01'), None))
    monkeypatch.setattr(prediction, 'apply_reference_resolution',
                        lambda plan, **kwargs: 'resolved-plan')
    if year == 2000:
        with pytest.raises(ValueError, match='Invalid Prediction Anchor Time'):
            generate_prediction_table(req, runtime=runtime)
        assert runtime.compile_options is None
    else:
        assert generate_prediction_table(req, runtime=runtime).row_count == 7


def test_in_process_scoring_parses_the_task_manifest_once(
        monkeypatch, tmp_path):
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train import artifact_manifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.types import PredictionArtifactType

    loaded, loads, predicts = [], [], []

    def load_manifest(path):
        loaded.append(SimpleNamespace(task_type='regression',
                                      stage_counts={'predict': 1}))
        return loaded[-1]

    def load_trainer(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(predict=lambda **kw: predicts.append(kw))

    monkeypatch.setattr(DiskgraphTaskManifest, 'load', load_manifest)
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *_, **__: None)
    monkeypatch.setattr(DiskgraphNestedTrainer, 'load', load_trainer)
    monkeypatch.setattr(artifact_manifest, 'hash_prediction_config',
                        lambda _: 'runtime-hash')
    monkeypatch.setattr(artifact_manifest, 'publish_prediction_selection',
                        lambda references, **kwargs: references[0])
    monkeypatch.setattr(prediction, '_publish_scored_shard',
                        lambda *_, **__: None)
    monkeypatch.setattr(prediction, '_prediction_worker_devices', lambda _, accelerator=None:
                        (None, ))
    encoders = object()

    prediction.score_prediction(
        materialized=SimpleNamespace(manifest_path='manifest',
                                     bundle_dir='bundle'),
        model_dir=str(tmp_path),
        run_id='prediction',
        scratch_dir=str(tmp_path),
        artifacts=frozenset({PredictionArtifactType.PREDICTIONS}),
        graph_snapshot_id='snapshot',
        source_config_sha256='source',
        encoders=encoders,
    )

    assert len(loaded) == 1
    assert predicts[0]['task_manifest'] is loaded[0]
    assert loads[0]['encoders'] is encoders
