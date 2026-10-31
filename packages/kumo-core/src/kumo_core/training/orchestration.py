# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable orchestration for the core-owned numerical training pipeline."""

from __future__ import annotations

import math
import posixpath
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

import kumo_core.resolved_task as resolved_task
from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.automl.runtime import AutoTrainingRuntime, resolve_warm_start
from kumo_core.instrumentation import TrainingInstrumentation
from kumo_core.io.json import save_json
from kumo_core.job_config import BestTrainerConfig
from kumo_core.model_plan import ModelPlan
from kumo_core.training.config import FitMode
from kumoapi.train import TrainingStage

from ..artifacts import VitalArtifactPublisher, VitalArtifactSources

MetricsLoader = Callable[[Any], Mapping[str, float]]


class TrainingTaskFamily(str, Enum):
    NODE_PREDICTION = "node_prediction"
    LINK_PREDICTION = "link_prediction"


class UnsupportedTrainingTaskError(ValueError):
    """The materialized task is outside the native training v1 scope."""


@dataclass(frozen=True)
class ResolvedTrainingInput:
    """Fully resolved objects required by the native training stage.

    Resolution from IDs, config paths, REST payloads, or StateDB records is an
    adapter responsibility and is deliberately not represented here.
    """

    task: resolved_task.ResolvedNumericalTask
    model_plan: ModelPlan
    home_dir: str
    task_dataset_bundle_dir: str
    warm_start_trainer_dir: str | None = None
    warm_start_artifacts: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, resolved_task.ResolvedNumericalTask):
            raise TypeError(
                "resolved training input task must be ResolvedNumericalTask")
        if not isinstance(self.model_plan, ModelPlan):
            raise TypeError(
                "resolved training input model_plan must be ModelPlan")
        if not self.home_dir:
            raise ValueError("resolved training input requires home_dir")
        if not self.task_dataset_bundle_dir:
            raise ValueError(
                "resolved training input requires task_dataset_bundle_dir")


@dataclass(frozen=True)
class CallableTrainingOptions:
    random_state_seed: int | None = None
    debug: bool = False
    overwrite_train_table_edge: bool = False
    analyze: bool = True
    refit_trainval: bool | None = None
    refit_full: bool | None = None
    warm_start_trainer_dir: str | None = None
    binary_classification_threshold: float | None = None
    target_edges: tuple[float, ...] = ()


def _no_op() -> None:
    return None


def _empty_metrics(_trainer: Any) -> Mapping[str, float]:
    return {}


@dataclass(frozen=True)
class TrainingBindings:
    """Environment and artifact lifecycle capabilities used by training."""

    runtime: AutoTrainingRuntime
    instrumentation: TrainingInstrumentation
    resolve_vital_artifacts: Callable[
        [ResolvedTrainingInput, Any, Any, Mapping[str, float]],
        VitalArtifactSources,
    ]
    artifact_publisher: VitalArtifactPublisher
    before_evaluate: Callable[[], None] = _no_op
    load_validation_metrics: MetricsLoader = _empty_metrics


@dataclass(frozen=True)
class CallableTrainingResult:
    """Portable result plus live handles needed by the legacy CLI tail."""

    task_family: TrainingTaskFamily
    metrics: Mapping[str, float]
    artifact_paths: Mapping[str, str]
    model_uri: str | None
    refitted_trainval: bool
    refitted_full: bool
    explain_disabled: bool
    task: Any = field(repr=False, compare=False)
    auto_trainer: Any = field(repr=False, compare=False)
    best_trainer: Any = field(repr=False, compare=False)
    published_counts: Any = field(default=None, repr=False, compare=False)

    def to_stage_output(self) -> dict[str, Any]:
        """Return the JSON-safe shape consumed by ``StageRunner``/MLflow."""

        return {
            "task_family": self.task_family.value,
            "metrics": dict(self.metrics),
            "artifact_paths": dict(self.artifact_paths),
            "model_uri": self.model_uri,
            "refitted_trainval": self.refitted_trainval,
            "refitted_full": self.refitted_full,
            "explain_disabled": self.explain_disabled,
        }


class TrainingStageContext(Protocol):
    """Exact context surface consumed by the portable training stages."""

    completed_outputs: Mapping[str, Any]


ResolvedInputFactory = Callable[[TrainingStageContext], ResolvedTrainingInput]


@dataclass(frozen=True)
class CoreTrainingStages:
    """Own the Kumo AutoML/train/evaluate stage boundary.

    Environment adapters resolve their input artifacts and construct the
    original Kumo ``AutoTrainer``/``Trainer`` bindings.  Model construction,
    optimization, refits, evaluation, and prediction therefore remain the
    established Kumo implementation; adapters must not provide another model
    or training loop.
    """

    resolve_input: ResolvedInputFactory
    bindings: TrainingBindings
    options: CallableTrainingOptions = field(
        default_factory=CallableTrainingOptions)

    def run(self, context: TrainingStageContext) -> CallableTrainingResult:
        """Run training while retaining live trainer handles for adapters."""

        training_input = self.resolve_input(context)
        return run_callable_training(
            training_input,
            bindings=self.bindings,
            options=self.options,
        )

    def train(self, context: TrainingStageContext) -> Mapping[str, Any]:
        return self.run(context).to_stage_output()

    @staticmethod
    def evaluate(context: TrainingStageContext) -> Mapping[str, Any]:
        try:
            training = context.completed_outputs["train"]
        except KeyError as error:
            raise RuntimeError(
                "evaluate requires completed train output") from error
        if not isinstance(training, Mapping):
            raise TypeError("train output must be a mapping")
        return {
            "metrics": dict(training.get("metrics", {})),
            "artifact_paths": dict(training.get("artifact_paths", {})),
            "model_uri": training.get("model_uri"),
            "task_family": training.get("task_family"),
        }

    def as_mapping(self) -> Mapping[str, Callable[[Any], Mapping[str, Any]]]:
        return {"train": self.train, "evaluate": self.evaluate}


_NODE_TASK_TYPES = frozenset({
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
    "regression",
    "forecasting",
    "node_prediction",
})
_LINK_TASK_TYPES = frozenset({
    "link_prediction",
    "static_link_prediction",
    "temporal_link_prediction",
})


def resolve_training_task_family(task: Any) -> TrainingTaskFamily:
    """Classify existing and portable task-type representations."""

    task_type = getattr(task, "task_type", None)
    if task_type is None:
        raise UnsupportedTrainingTaskError(
            "materialized task does not expose task_type")

    if bool(getattr(task_type, "is_node_pred", False)):
        return TrainingTaskFamily.NODE_PREDICTION
    if bool(getattr(task_type, "is_link_pred", False)):
        return TrainingTaskFamily.LINK_PREDICTION

    value = getattr(task_type, "value", task_type)
    normalized = str(value).strip().lower()
    if normalized in _NODE_TASK_TYPES:
        return TrainingTaskFamily.NODE_PREDICTION
    if normalized in _LINK_TASK_TYPES:
        return TrainingTaskFamily.LINK_PREDICTION
    raise UnsupportedTrainingTaskError(
        f"task type {normalized!r} is not a selected node/link training task")


def _metrics_from_output(output: Any) -> dict[str, float]:
    values = getattr(output, "metrics", None)
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("evaluation output metrics must be a mapping")
    return _finite_metrics(values)


def _finite_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        metric = float(value)
        if math.isfinite(metric):
            result[str(name)] = metric
    return result


def _training_metrics(best_trainer: Any) -> dict[str, float]:
    get_results = getattr(best_trainer, "get_train_results", None)
    if not callable(get_results):
        return {}
    values = get_results()
    if not isinstance(values, Mapping):
        raise TypeError("Trainer.get_train_results() must return a mapping")
    return _finite_metrics({
        str(name): float(value)
        for name, value in values.items() if str(name).startswith("train")
    })


def _refit_metrics(output: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in _metrics_from_output(output).items():
        key = name.replace("test_", "test_refit_", 1)
        if key == name:
            key = f"refit_{name}"
        result[key] = value
    return result


def _create_auto_trainer(
    training_input: ResolvedTrainingInput,
    bindings: TrainingBindings,
    options: CallableTrainingOptions,
) -> AutoTrainer:
    """Construct the core AutoTrainer from already-resolved numerical input."""
    model_plan = training_input.model_plan
    checkpoint_path, checkpoint_encoder_dir = resolve_warm_start(
        model_plan,
        training_input.task,
        bindings.runtime,
        trainer_dir=(options.warm_start_trainer_dir
                     or training_input.warm_start_trainer_dir),
        artifacts=training_input.warm_start_artifacts,
    )

    return AutoTrainer(
        space=model_plan,
        task=training_input.task,
        runtime=bindings.runtime,
        instrumentation=bindings.instrumentation,
        random_state_seed=options.random_state_seed,
        checkpoint_path=checkpoint_path,
        checkpoint_encoder_dir=checkpoint_encoder_dir,
        home_dir=training_input.home_dir,
        overwrite_train_table_edge=options.overwrite_train_table_edge,
        diskgraph_task_dataset_bundle_dir=(
            training_input.task_dataset_bundle_dir),
    )


def run_callable_training(
    training_input: ResolvedTrainingInput,
    *,
    bindings: TrainingBindings,
    options: CallableTrainingOptions | None = None,
) -> CallableTrainingResult:
    """Run AutoML, refits, evaluation, and analysis in the caller process.

    The input is resolved at the adapter boundary.  This function never loads
    a task, model plan, or bundle from an identifier or config path.
    """

    options = options or CallableTrainingOptions()
    if not isinstance(training_input, ResolvedTrainingInput):
        raise TypeError("training_input must be ResolvedTrainingInput")
    task = training_input.task
    task_family = resolve_training_task_family(task)

    with bindings.instrumentation.timing("kumo_auto.auto_trainer_from_config"):
        auto_trainer = _create_auto_trainer(training_input, bindings, options)

    if options.refit_trainval is not None:
        auto_trainer.refit_trainval = options.refit_trainval
    if options.refit_full is not None:
        auto_trainer.refit_full = options.refit_full

    should_init_best_trainer = (not bool(auto_trainer.refit_trainval)
                                and not bool(auto_trainer.refit_full))
    with bindings.instrumentation.timing("kumo_auto.auto_trainer_run"):
        best_trainer = auto_trainer.run(
            init_best_trainer=should_init_best_trainer,
            debug=options.debug,
        )

    is_warm_start = bool(options.warm_start_trainer_dir
                         or training_input.warm_start_trainer_dir
                         or training_input.warm_start_artifacts)
    if is_warm_start:
        best_trainer.checkpoint_path = auto_trainer.checkpoint_path

    if (bool(auto_trainer.refit_trainval)
            and not bool(best_trainer.refit_cfg.refitted_trainval)):
        with bindings.instrumentation.timing("kumo_auto.refit_trainval"):
            best_trainer.refit(fit_mode=FitMode.refit_trainval)

    if (bool(auto_trainer.refit_full)
            and not bool(best_trainer.refit_cfg.refitted_full)):
        with bindings.instrumentation.timing("kumo_auto.refit_full"):
            best_trainer.refit(fit_mode=FitMode.refit_full)

    if is_warm_start:
        best_trainer.checkpoint_path = None

    best_trainer_config_path = posixpath.join(str(auto_trainer.log_dir),
                                              "best_trainer.json")
    save_json(
        BestTrainerConfig(str(best_trainer.original_log_dir)),
        best_trainer_config_path,
    )

    bindings.before_evaluate()
    with bindings.instrumentation.timing("kumo_auto.best_trainer_test_fit"):
        evaluation = best_trainer.test(
            fit_mode_to_use=FitMode.fit,
            binary_classification_threshold=(
                options.binary_classification_threshold),
            target_edges=options.target_edges)
    with bindings.instrumentation.timing("kumo_auto.save_test_outputs_fit"):
        evaluation.save_all(
            root_dir=best_trainer.log_dir,
            stage=TrainingStage.TEST,
            task=task,
        )

    metrics = _finite_metrics(bindings.load_validation_metrics(best_trainer))
    metrics.update(_metrics_from_output(evaluation))
    metrics.update(_training_metrics(best_trainer))

    refitted_trainval = bool(best_trainer.refit_cfg.refitted_trainval)
    published = evaluation
    if refitted_trainval:
        with bindings.instrumentation.timing(
                "kumo_auto.best_trainer_test_refit_trainval"):
            refit_evaluation = best_trainer.test(
                fit_mode_to_use=FitMode.refit_trainval,
                binary_classification_threshold=(
                    options.binary_classification_threshold),
                target_edges=options.target_edges)
        with bindings.instrumentation.timing(
                "kumo_auto.save_test_outputs_refit_trainval"):
            refit_evaluation.save_all(
                root_dir=best_trainer.log_dir,
                stage=TrainingStage.TEST,
                task=task,
            )
        metrics.update(_refit_metrics(refit_evaluation))
        published = refit_evaluation

    analysis_path = posixpath.join(str(auto_trainer.log_dir), "analyze")
    if options.analyze:
        with bindings.instrumentation.timing("kumo_auto.searcher_analyze"):
            auto_trainer.searcher.analyze(analysis_path)

    vital_sources = bindings.resolve_vital_artifacts(
        training_input,
        auto_trainer,
        best_trainer,
        metrics,
    )
    if not isinstance(vital_sources, VitalArtifactSources):
        raise TypeError(
            "resolve_vital_artifacts must return VitalArtifactSources")
    artifact_paths = dict(bindings.artifact_publisher.publish(vital_sources))
    expected_roles = set(VitalArtifactSources.roles())
    if set(artifact_paths) != expected_roles:
        raise ValueError(
            "artifact publisher must return exactly the vital artifact roles")

    return CallableTrainingResult(
        task_family=task_family,
        metrics=metrics,
        artifact_paths=artifact_paths,
        model_uri=artifact_paths["selected_checkpoint"],
        refitted_trainval=refitted_trainval,
        refitted_full=bool(best_trainer.refit_cfg.refitted_full),
        explain_disabled=bool(
            training_input.model_plan.training_job.disable_explain),
        task=task,
        auto_trainer=auto_trainer,
        best_trainer=best_trainer,
        published_counts=getattr(published, 'counts', None),
    )


__all__ = [
    "CallableTrainingOptions",
    "CallableTrainingResult",
    "CoreTrainingStages",
    "ResolvedInputFactory",
    "ResolvedTrainingInput",
    "TrainingStageContext",
    "TrainingBindings",
    "TrainingTaskFamily",
    "UnsupportedTrainingTaskError",
    "resolve_training_task_family",
    "run_callable_training",
]
