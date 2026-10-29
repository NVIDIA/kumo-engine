# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import kumo_core.training.orchestration as training_module
import kumo_core.automl.runtime as automl_runtime
from kumo_core.train.trainer import Trainer as CoreTrainer
import pytest
from kumo_core.artifacts import VitalArtifactSources
from kumo_core.io.json import load_json, save_json
from kumo_core.model_plan import ModelPlan
from kumo_core.resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTrainTable,
)
from kumo_core.training.config import FitMode, TrainerConfig
from kumo_core.training.orchestration import (
    CallableTrainingOptions,
    ResolvedTrainingInput,
    TrainingBindings,
    TrainingTaskFamily,
    UnsupportedTrainingTaskError,
    resolve_training_task_family,
    run_callable_training,
)
from kumo_core.types import TaskType


class Evaluation:
    def __init__(self, metrics):
        self.metrics = metrics
        self.saved = []

    def save_all(self, **kwargs):
        self.saved.append(kwargs)


class Trainer:
    def __init__(self, *, refitted_trainval=False, refitted_full=False):
        self.original_log_dir = "/artifacts/best"
        self.log_dir = "/artifacts/best/version_0"
        self.checkpoint_path = None
        self.refit_cfg = SimpleNamespace(
            refitted_trainval=refitted_trainval,
            refitted_full=refitted_full,
        )
        self.refit_calls = []
        self.test_calls = []
        self.evaluations = [Evaluation({"test_accuracy": 0.8})]
        self.best_ckpt = "/warm/checkpoint.ckpt"

    def _set_fit_mode_to_use(self):
        return None

    def refit(self, *, fit_mode):
        self.refit_calls.append(fit_mode)
        if fit_mode == FitMode.refit_trainval:
            self.refit_cfg.refitted_trainval = True
        if fit_mode == FitMode.refit_full:
            self.refit_cfg.refitted_full = True

    def test(self, *, fit_mode_to_use, **options):
        self.test_calls.append(fit_mode_to_use)
        return self.evaluations.pop(0)

    def get_train_results(self):
        return {"train_loss": 0.2, "internal": 123}


class AutoTrainer:
    def __init__(self, trainer, *, refit_trainval=False, refit_full=False):
        self.trainer = trainer
        self.refit_trainval = refit_trainval
        self.refit_full = refit_full
        self.checkpoint_path = "/warm/checkpoint.ckpt"
        self.log_dir = "/artifacts/automl"
        self.run_calls = []
        self.searcher = SimpleNamespace(analyze=self._analyze)
        self.analysis_paths = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        return self.trainer

    def _analyze(self, path):
        self.analysis_paths.append(path)


def vital_sources():
    return VitalArtifactSources(
        **{role: f"/source/{role}"
           for role in VitalArtifactSources.roles()})


class ArtifactPublisher:
    def __init__(self, calls):
        self.calls = calls

    def publish(self, sources):
        self.calls.append(("publish", sources))
        return {
            role: f"/published/{role}"
            for role in VitalArtifactSources.roles()
        }


class Runtime:
    def default_home_dir(self):
        return "/native/home"

    def metrics_job_name(self):
        return "test"

    def collect_encoders(self, task):
        return None

    def load_task(self, task_name, home_dir):
        raise NotImplementedError

    def load_automl_config(self, job_config):
        raise NotImplementedError

    def trainer_runtime_kwargs(self):
        return {}


class Instrumentation:
    def __init__(self, calls):
        self.calls = calls

    def record_metrics_job_name(self, job_name):
        return None

    @contextmanager
    def span(self, name, **fields):
        yield

    @contextmanager
    def timing(self, name, **fields):
        self.calls.append(("timing", name))
        yield


def make_bindings(task, auto_trainer, calls, monkeypatch):
    monkeypatch.setattr(training_module, "AutoTrainer",
                        lambda **kwargs: auto_trainer)
    monkeypatch.setattr(
        training_module,
        "save_json",
        lambda value, path: calls.append(("save_best", value, path)),
    )
    monkeypatch.setattr(
        CoreTrainer,
        "load",
        classmethod(lambda cls, *args, **kwargs: auto_trainer.trainer),
    )
    monkeypatch.setattr(automl_runtime, "load_json",
                        lambda path, cls: ModelPlan())

    return TrainingBindings(
        runtime=Runtime(),
        instrumentation=Instrumentation(calls),
        resolve_vital_artifacts=lambda input_, auto, trainer, metrics:
        (calls.append(
            ("resolve_vital", input_, auto, trainer, metrics)) or vital_sources(
            )),
        artifact_publisher=ArtifactPublisher(calls),
        before_evaluate=lambda: calls.append(("before_evaluate", )),
        load_validation_metrics=lambda trainer: {"val_accuracy": 0.75},
    )


def resolved(task, **kwargs):
    if not isinstance(task, ResolvedNumericalTask):
        task = numerical_task(TaskType(task.task_type))
    return ResolvedTrainingInput(
        task=task,
        model_plan=ModelPlan(),
        home_dir="/native/home",
        task_dataset_bundle_dir="/native/bundle",
        **kwargs,
    )


def numerical_task(task_type: TaskType) -> ResolvedNumericalTask:
    return ResolvedNumericalTask(
        name="test",
        task_type=task_type,
        metrics=["auroc"],
        graph=ResolvedTaskGraph(
            node_types=("users", ),
            edge_types=(),
            seed_node_type="users",
            target_node_type="users",
        ),
        train_table=ResolvedTrainTable(
            name="train_table",
            target_column="TARGET",
            time_column=None,
            weight_column=None,
        ),
    )


@pytest.mark.parametrize(
    ("task_type", "family"),
    [
        ("binary_classification", TrainingTaskFamily.NODE_PREDICTION),
        ("regression", TrainingTaskFamily.NODE_PREDICTION),
        ("static_link_prediction", TrainingTaskFamily.LINK_PREDICTION),
        ("temporal_link_prediction", TrainingTaskFamily.LINK_PREDICTION),
    ],
)
def test_selected_node_and_link_task_families(task_type, family):
    task = SimpleNamespace(task_type=task_type)
    assert resolve_training_task_family(task) is family


def test_full_callable_flow_returns_mlflow_ready_result(monkeypatch):
    calls = []
    task = SimpleNamespace(task_type="binary_classification")
    trainer = Trainer()
    auto_trainer = AutoTrainer(trainer)
    bindings = make_bindings(task, auto_trainer, calls, monkeypatch)
    config = resolved(task)

    result = run_callable_training(config, bindings=bindings)

    assert auto_trainer.run_calls == [{
        "init_best_trainer": True,
        "debug": False,
    }]
    assert trainer.test_calls == [FitMode.fit]
    assert trainer.refit_calls == []
    assert result.metrics == {
        "val_accuracy": 0.75,
        "test_accuracy": 0.8,
        "train_loss": 0.2,
    }
    assert set(result.artifact_paths) == set(VitalArtifactSources.roles())
    assert result.model_uri == "/published/selected_checkpoint"
    assert auto_trainer.analysis_paths == ["/artifacts/automl/analyze"]
    assert calls[-1][0] == "publish"
    json.dumps(result.to_stage_output(), allow_nan=False)


def test_callable_flow_omits_non_finite_metrics(monkeypatch):
    calls = []
    task = SimpleNamespace(task_type="multilabel_classification")
    trainer = Trainer()
    trainer.evaluations = [
        Evaluation({
            "test_ap_micro": 0.5,
            "test_ap_macro": float("nan"),
        })
    ]
    trainer.get_train_results = lambda: {
        "train_loss": 0.2,
        "train_ap_macro": float("nan"),
    }
    auto_trainer = AutoTrainer(trainer)
    bindings = make_bindings(task, auto_trainer, calls, monkeypatch)
    bindings = replace(
        bindings,
        load_validation_metrics=lambda _trainer: {
            "val_ap_micro": 0.4,
            "val_ap_macro": float("nan"),
            "val_loss": float("inf"),
        },
    )

    result = run_callable_training(resolved(task), bindings=bindings)

    assert result.metrics == {
        "val_ap_micro": 0.4,
        "test_ap_micro": 0.5,
        "train_loss": 0.2,
    }
    json.dumps(result.to_stage_output(), allow_nan=False)


def test_failed_training_does_not_publish_artifacts(monkeypatch):
    calls = []
    task = SimpleNamespace(task_type="binary_classification")
    trainer = Trainer()
    trainer.evaluations = [Evaluation({"test_accuracy": "invalid"})]
    auto_trainer = AutoTrainer(trainer)

    with pytest.raises(ValueError, match="could not convert"):
        run_callable_training(
            resolved(task),
            bindings=make_bindings(task, auto_trainer, calls, monkeypatch),
        )

    assert not any(call[0] == "resolve_vital" for call in calls)
    assert not any(call[0] == "publish" for call in calls)


def test_refit_and_warm_start_preserve_existing_behavior(monkeypatch):
    calls = []
    task = SimpleNamespace(task_type="static_link_prediction")
    trainer = Trainer()
    trainer.evaluations.append(Evaluation({"test_mrr": 0.9}))
    auto_trainer = AutoTrainer(
        trainer,
        refit_trainval=True,
        refit_full=True,
    )
    bindings = make_bindings(task, auto_trainer, calls, monkeypatch)
    config = resolved(task, warm_start_trainer_dir="/warm/trainer")

    result = run_callable_training(
        config,
        bindings=bindings,
        options=CallableTrainingOptions(debug=True),
    )

    assert auto_trainer.run_calls == [{
        "init_best_trainer": False,
        "debug": True,
    }]
    assert trainer.refit_calls == [FitMode.refit_trainval, FitMode.refit_full]
    assert trainer.checkpoint_path is None
    assert trainer.test_calls == [FitMode.fit, FitMode.refit_trainval]
    assert result.metrics["test_refit_mrr"] == 0.9
    assert result.refitted_trainval
    assert result.refitted_full


def test_callable_options_override_model_plan_refits(monkeypatch):
    calls = []
    task = SimpleNamespace(task_type="binary_classification")
    trainer = Trainer()
    trainer.evaluations.append(Evaluation({"test_accuracy": 0.9}))
    auto_trainer = AutoTrainer(trainer)

    result = run_callable_training(
        resolved(task),
        bindings=make_bindings(task, auto_trainer, calls, monkeypatch),
        options=CallableTrainingOptions(
            refit_trainval=True,
            refit_full=True,
        ),
    )

    assert trainer.refit_calls == [FitMode.refit_trainval, FitMode.refit_full]
    assert result.refitted_trainval is True
    assert result.refitted_full is True


def test_unsupported_task_fails_before_automl():
    auto_trainer = AutoTrainer(Trainer())
    invalid_task = SimpleNamespace(task_type="custom_training_table")

    with pytest.raises(UnsupportedTrainingTaskError, match="node/link"):
        resolve_training_task_family(invalid_task)
    assert auto_trainer.run_calls == []


def test_non_numeric_evaluation_metric_is_rejected(monkeypatch):
    calls = []
    trainer = Trainer()
    trainer.evaluations = [Evaluation({"test_accuracy": "not-a-number"})]
    auto_trainer = AutoTrainer(trainer)
    bindings = make_bindings(
        SimpleNamespace(task_type="binary_classification"),
        auto_trainer,
        calls,
        monkeypatch,
    )

    with pytest.raises(ValueError, match="could not convert"):
        run_callable_training(
            resolved(SimpleNamespace(task_type="binary_classification")),
            bindings=bindings,
        )


def test_core_rejects_unresolved_job_config_shape(monkeypatch):
    with pytest.raises(TypeError, match="ResolvedTrainingInput"):
        run_callable_training(  # type: ignore[arg-type]
            SimpleNamespace(task_name="task-id", automl_cfg_path="model.json"),
            bindings=make_bindings(SimpleNamespace(), AutoTrainer(Trainer()),
                                   [], monkeypatch),
        )


@pytest.mark.parametrize(
    "missing",
    ["task", "model_plan", "home_dir", "task_dataset_bundle_dir"],
)
def test_resolved_input_requires_concrete_boundary_values(missing):
    values = {
        "task": numerical_task(TaskType.BINARY_CLASSIFICATION),
        "model_plan": ModelPlan(),
        "home_dir": "/home",
        "task_dataset_bundle_dir": "/bundle",
    }
    values[missing] = None
    error_type = TypeError if missing in {"task", "model_plan"} else ValueError
    with pytest.raises(error_type, match=missing):
        ResolvedTrainingInput(**values)


def test_trainer_refit_state_round_trip(tmp_path):
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
    from kumo_core.modeling.heads.node_prediction import NodePredHead

    trainer = CoreTrainer(
        task=numerical_task(TaskType.BINARY_CLASSIFICATION),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        root_dir=str(tmp_path),
    )
    trainer.refit_cfg.best_val_epoch = 7
    trainer.refit_cfg.refitted_trainval = True
    trainer.refit_cfg.refitted_full = True
    path = str(tmp_path / "trainer-config.json")
    save_json(trainer.config(), path)

    config = load_json(path, TrainerConfig)
    loaded = CoreTrainer.from_config(
        config,
        task=numerical_task(TaskType.BINARY_CLASSIFICATION),
    )

    assert loaded.refit_cfg.best_val_epoch == 7
    assert loaded.refit_cfg.refitted_trainval
    assert loaded.refit_cfg.refitted_full
    assert isinstance(loaded.encoder, HeteroTensorFrameEncoder)
    assert isinstance(loaded.model, HeteroGNN)
    assert isinstance(loaded.head, NodePredHead)

    config.model = {'_target_': 'unsupported.Model'}
    with pytest.raises(ValueError, match='unsupported model config target'):
        CoreTrainer.from_config(
            config,
            task=numerical_task(TaskType.BINARY_CLASSIFICATION),
        )


@pytest.mark.parametrize('portable', [False, True])
def test_warm_start_resolves_selected_model_without_copying_training_controls(
        tmp_path, monkeypatch, portable):
    task = numerical_task(TaskType.BINARY_CLASSIFICATION)
    destination = ModelPlan()
    destination.optimization.max_epochs = 7
    destination.training_job.refit_full = False
    source = ModelPlan()
    source.model_architecture.channels = [16]
    source.optimization.max_epochs = 100
    source.training_job.refit_full = True
    plan_path = tmp_path / 'model_plan.json'
    save_json(source, str(plan_path))
    selected = tmp_path / 'checkpoint'
    encoder_dir = tmp_path / 'bundle-model'
    selections = []
    trainer = SimpleNamespace(
        _set_fit_mode_to_use=lambda: selections.append('selected'),
        best_ckpt=str(selected),
    )
    monkeypatch.setattr(CoreTrainer, 'load', lambda *args, **kwargs: trainer)
    schema_path = tmp_path / 'prediction_schema.json'
    save_json({'task_type': task.task_type.value}, str(schema_path))
    artifacts = {
        'selected_checkpoint': str(selected),
        'encoders': str(encoder_dir / 'encoders.json'),
        'resolved_model_plan': str(plan_path),
        'prediction_schema': str(schema_path),
    } if portable else None
    checkpoint, encoders = automl_runtime.resolve_warm_start(
        destination, task, Runtime(), trainer_dir=str(tmp_path),
        artifacts=artifacts)
    assert checkpoint == str(selected)
    assert encoders == (str(encoder_dir) if portable else None)
    assert selections == ([] if portable else ['selected'])
    assert destination.model_architecture.channels == [16]
    assert destination.optimization.max_epochs == 7
    assert destination.training_job.refit_full is False


@pytest.mark.parametrize('current,source,message', [
    (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION, 'different'),
    (TaskType.REGRESSION, TaskType.FORECASTING, 'different'),
    (TaskType.FORECASTING, TaskType.FORECASTING,
     'not supported for forecasting'),
])
def test_bundle_warm_start_rejects_incompatible_task_before_loading_model(
        tmp_path, monkeypatch, current, source, message):
    schema_path = tmp_path / 'prediction_schema.json'
    save_json({'task_type': source.value}, str(schema_path))
    artifacts = {
        'prediction_schema': str(schema_path),
        'selected_checkpoint': '/must-not-load/checkpoint',
        'resolved_model_plan': '/must-not-load/model_plan.json',
        'encoders': '/must-not-load/encoders.json',
    }
    monkeypatch.setattr(
        CoreTrainer, 'load',
        lambda *args, **kwargs: pytest.fail('trainer loaded before validation'))
    with pytest.raises(ValueError, match=message):
        automl_runtime.resolve_warm_start(ModelPlan(), numerical_task(current),
                                          Runtime(), artifacts=artifacts)


def test_hosted_warm_start_rejects_forecasting_before_loading_trainer(
        monkeypatch):
    monkeypatch.setattr(
        CoreTrainer, 'load',
        lambda *args, **kwargs: pytest.fail('trainer loaded before validation'))
    with pytest.raises(ValueError, match='not supported for forecasting'):
        automl_runtime.resolve_warm_start(ModelPlan(),
                                          numerical_task(TaskType.FORECASTING),
                                          Runtime(),
                                          trainer_dir='/must-not-load/trainer')
    assert automl_runtime.resolve_warm_start(
        ModelPlan(), numerical_task(TaskType.FORECASTING),
        Runtime()) == (None, None)
