# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from os import path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kumo_core.evaluation.suite import MetricStore
from kumo_core.job_config import JobConfig, TrainStatus
from kumo_core.model_plan import ModelPlan
from kumo_core.train.output import EvalOutput
from kumo_core.train.stats import (
    AllTargetRegressionStats,
    TargetHistogramAndStatsPair,
)
from kumo_core.train.trainer import Trainer
from kumo_core.training.config import FitMode
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus

from io_utils import fsutil, logging
from io_utils.json import load_json, save_json
from kumo.config.explain import ColumnContributionSummary
from kumo.config.task import TimePeriod
from kumo.data import get_persist_dir
from kumo.diskgraph import job_sidecars
from kumo.home import HomeDirManager
from kumo.task.task import Task
from kumo.train.config_io import load_job_automl_config, load_job_task
from kumo.utils.evaluate import BinaryConfusionMatrix, get_confusion_matrix
from kumo.validation import ValidationResponse, ValidationWarning
from rest.backends.ml_backend.reporters import TrainingJobReporter
from rest.backends.ml_backend.timing import kumo_timing
from rest.config.constants import TrainingJobInputJSON
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.ml_backend import JobState
from rest.config.training_job import (
    Metric,
    SplitSizes,
    TestEvalResults,
    TestExplanation,
    TimePeriods,
    TrainingJobOutput,
    TrainingJobResults,
)
from statedb.config.common import (
    ConnectorID,
    JobEventLogEntry,
    JobID,
    JobStatus,
    TrainingJobID,
)
from statedb.utils.errors import (
    BestTrainerCfgFileMissingError,
    TrainingJobOutputError,
)
from temporalio_commons.error_utils import is_error_non_retryable
from temporalio_commons.heartbeat_utils import auto_heartbeater

from ...data_backend import connector_activity_registry
from ..ml_job_output_store import TRAINING_JOB_TEST_EXPLAIN_FILENAME

log = logging.getLogger(__name__)


@dataclass
class PollJobResult:
    """Result of polling a training job's state."""
    state: JobState
    err_msg: Optional[str] = None
    stacktrace: Optional[str] = None
    warning_msg: Optional[str] = None


class TrainingJobActivities:
    @activity.defn(name='log_failed_training_job')
    def log_failed_job(self, job_id: TrainingJobID, job_config_path: str,
                       warning_msg: Optional[str] = None) -> None:
        # Save warnings even for failed jobs (e.g., batch size reduction before failure)
        if warning_msg:
            job_config = JobConfig.load(job_config_path)
            _save_training_warning(job_config.home_dir, job_id, warning_msg)

        log.error(f"Job `{job_id}` failed. Reproduce cmd: "
                  f"./bin/kumo-auto --json {job_config_path}")

    @connector_activity_registry.connector_authorized_activity(
        name='postprocess_training_job_output')
    @auto_heartbeater
    def postprocess_job_output(
        self,
        job_id: TrainingJobID,
        job_config_path: str,
        connector_refs: Dict[ConnectorID, ConnectorExecutionRef],
        snapshot_id: str | None = None,
    ) -> TrainingJobOutput:
        job_config = JobConfig.load(job_config_path)
        home_dir_mgr = HomeDirManager(job_config.home_dir)
        best_trainer_cfg_path = home_dir_mgr.best_trainer_config_path(
            job_config.task_name)
        if not fsutil.exists(best_trainer_cfg_path):
            self.log_failed_job(job_id, job_config_path)
            raise BestTrainerCfgFileMissingError(non_retryable=True)

        # Load the best trainer and evaluate result.
        # TODO(dong) refactor the code so that we don't need to restore the
        # model and just keep the path reference.

        best_train_dir = home_dir_mgr.get_best_trainer_dir(job_config.task_name)
        task = load_job_task(job_config)
        with kumo_timing("training_postprocess.load_best_trainer_first"):
            best_trainer = Trainer.load(
                task.to_task_config(),
                best_train_dir,
                version=None,
            )
        automl_cfg = load_job_automl_config(job_config)
        assert isinstance(automl_cfg, ModelPlan)
        with kumo_timing("training_postprocess.load_metrics"):
            trainer_metrics, confusion_matrix = (
                _load_metrics_and_confusion_matrix(
                    best_trainer,
                    task.train_table.target_col_name,
                    automl_cfg,
                    job_id,
                ))

        explain_dir = osp.join(best_trainer.log_dir,
                               osp.basename(best_trainer.explain_dir("test")))

        with kumo_timing("training_postprocess.column_contrib"):
            column_contrib = ColumnContributionSummary.from_trainer(explain_dir)

        # TT-NG never loads the training table's in-memory ``.data``: the
        # per-stage ``time_periods`` / ``prediction_horizons`` / ``split_sizes``
        # are derived from the wheel manifest below.
        # TODO(wheel-gap): the wheel should expose per-stage
        # ``time_periods`` / ``prediction_horizons`` / ``split_sizes``
        # on the manifest directly so this bridge can go.
        stats, split_info = _load_ttng_published_metadata(
            job_config,
            snapshot_id=snapshot_id,
            job_config_path=job_config_path,
        )
        time_periods = _time_periods_from_ttng_stats(stats)
        # Prediction horizon derivation requires PQL target offsets;
        # leave as None until the wheel emits it. Downstream
        # accepts None.
        prediction_horizons = None
        split_sizes = _split_sizes_from_ttng_split_info(split_info)

        with kumo_timing("training_postprocess.load_best_trainer_second"):
            best_trainer = Trainer.load(
                task.to_task_config(),
                best_train_dir,
                version=None,
            )
        with kumo_timing("training_postprocess.generate_outputs"):
            _generate_outputs_for_best_trainer(best_trainer.log_dir,
                                               job_config.home_dir, job_id)

        log_dir = home_dir_mgr.automl_dir(job_config.task_name)
        automl_cfg = load_job_automl_config(job_config)
        assert isinstance(automl_cfg, ModelPlan)
        tune_metric = automl_cfg.training_job.tune_metric
        assert isinstance(tune_metric, str)
        reporter = TrainingJobReporter(
            log_dir=log_dir,
            metric=tune_metric,
            metric_mode=MetricStore.tune_mode(tune_metric),
        )
        # Load TrainStatus to get warning_msg
        train_status = TrainStatus.load(job_config.home_dir)
        warning_msg = train_status.warning_msg

        # Save ValidationResponse for /warnings endpoint if there are warnings
        if warning_msg:
            _save_training_warning(job_config.home_dir, job_id, warning_msg)

        with kumo_timing("training_postprocess.build_job_results"):
            return self._build_job_output(job_id, best_train_dir,
                                          trainer_metrics, reporter,
                                          column_contrib, time_periods,
                                          prediction_horizons, confusion_matrix,
                                          split_sizes)

    def _build_job_output(
        self,
        job_id: TrainingJobID,
        trainer_dir: str,
        metrics: Dict[str, float],
        reporter: TrainingJobReporter,
        column_contrib: ColumnContributionSummary,
        time_periods: Dict[TrainingStage, TimePeriod],
        prediction_horizons: Optional[Dict[TrainingStage, TimePeriod]],
        confusion_matrix: Optional[BinaryConfusionMatrix],
        split_sizes: SplitSizes,
    ) -> TrainingJobOutput:
        log.debug("Building the results for job: %s", job_id)
        output = _make_done_job_output(
            metrics=metrics,
            reporter=reporter,
            column_contrib=column_contrib,
            time_periods=time_periods,
            prediction_horizons=prediction_horizons,
            confusion_matrix=confusion_matrix,
            split_sizes=split_sizes,
        )

        # Save the (potentially large) test-explain blob to remote storage and
        # keep only the file path on the output, so the returned object stays
        # small enough to flow through the Temporal payload to the
        # save_training_job_output activity (which re-resolves the path).
        if output.results:
            test_explain_path = osp.join(trainer_dir,
                                         TRAINING_JOB_TEST_EXPLAIN_FILENAME)
            output.results.save_test_explain_to_remote(test_explain_path)

        return output


def _make_done_job_output(
    metrics: Dict[str, float],
    reporter: TrainingJobReporter,
    column_contrib: ColumnContributionSummary,
    time_periods: Dict[TrainingStage, TimePeriod],
    prediction_horizons: Optional[Dict[TrainingStage, TimePeriod]],
    confusion_matrix: Optional[BinaryConfusionMatrix],
    split_sizes: SplitSizes,
) -> TrainingJobOutput:
    metric_list: List[Metric] = []
    for metric_name in sorted(metrics):
        metric_list.append(Metric(name=metric_name, val=metrics[metric_name]))

    job_progress = reporter.get_training_job_progress()
    if job_progress is None:
        log.warning("Training progress not available for job at %s",
                    reporter._log_dir)
        training_time = 0
        total_iterations = 0
    else:
        training_time = job_progress.elapsed_time.seconds
        total_iterations = job_progress.iterations_completed
    test_explain = TestExplanation(contribution_summary=column_contrib)

    test_eval = None
    if confusion_matrix is not None:
        test_eval = TestEvalResults(confusion_matrix=confusion_matrix)

    prediction_horizons_time_periods = None
    if prediction_horizons is not None:
        prediction_horizons_time_periods = TimePeriods(
            test=prediction_horizons.get(TrainingStage.TEST),
            val=prediction_horizons.get(TrainingStage.VAL),
            train=prediction_horizons.get(TrainingStage.TRAIN),
        )

    return TrainingJobOutput(
        status=JobStatus.DONE,
        workflow_end_state=WorkflowExecutionStatus.COMPLETED,
        log=JobEventLogEntry.done(),
        progress=job_progress,
        results=TrainingJobResults(
            total_training_time_sec=training_time,
            total_iterations=total_iterations,
            test_explain=test_explain,
            test_eval=test_eval,
            time_periods=TimePeriods(
                test=time_periods.get(TrainingStage.TEST),
                val=time_periods.get(TrainingStage.VAL),
                train=time_periods.get(TrainingStage.TRAIN),
            ),
            prediction_horizons=prediction_horizons_time_periods,
            split_sizes=split_sizes,
        ).with_metrics(metric_list),
        failure_info=None,
    )


def _save_training_warning(home_dir: str, job_id: TrainingJobID,
                           warning_msg: str) -> None:
    """Save a training warning to the ValidationResponse file."""
    validation_response = ValidationResponse.load_or_create(home_dir, job_id)
    validation_response.warnings.append(
        ValidationWarning(
            title="Training Warning",
            message=warning_msg,
        ))
    validation_response.save(home_dir, job_id)


def _load_ttng_published_metadata(
    job_config: JobConfig,
    *,
    snapshot_id: str | None = None,
    job_config_path: str | None = None,
) -> tuple[dict, dict]:
    """Read TT-NG post-ingest metadata from Kumo's task-dataset bundle.

    Generator manifests are private inputs to ``execute_tt_ingest``. The
    training postprocess runs after ingestion and therefore consumes only the
    durable sidecars Kumo wrote from the published extension.
    """
    if snapshot_id is not None and job_config_path is not None:
        bundle_dir = job_sidecars.dataplane_job_sidecar_bundle_dir_from_config_path(
            snapshot_id,
            job_config_path,
            prediction=False,
            data_persist_dir=get_persist_dir(),
        )
    else:
        bundle_dir = job_config.diskgraph_task_dataset_bundle_dir
    if not bundle_dir:
        raise RuntimeError(
            'TT-NG training postprocess requires the materialized task-dataset '
            'bundle metadata, but no bundle path was resolved.')

    def _read_dict(filename: str) -> dict:
        path = osp.join(bundle_dir, filename)
        if not fsutil.exists(path):
            raise RuntimeError(
                f'TT-NG task-dataset bundle is missing {filename!r} at '
                f'{path!r}; rematerialize the published ingest extension.')
        try:
            payload = json.loads(fsutil.read_text(path))
        except (OSError, ValueError) as ex:
            raise RuntimeError(
                f'TT-NG task-dataset bundle has unreadable {filename!r} at '
                f'{path!r}; rematerialize the published ingest extension.'
            ) from ex
        if not isinstance(payload, dict):
            raise RuntimeError(
                f'TT-NG task-dataset bundle has malformed {filename!r} at '
                f'{path!r}; rematerialize the published ingest extension.')
        return payload

    return _read_dict('stats.json'), _read_dict('split_info.json')


def _time_periods_from_ttng_stats(
        stats: dict) -> Dict[TrainingStage, TimePeriod]:
    """Derive per-stage time bounds from published TT-NG ``stats.json``.

    Groups ``stats.payload.split_payload.artifacts_v3.timestamp_stage_assignments``
    by stage and takes min/max timestamp per stage.

    TODO(wheel-gap): the wheel should surface per-stage ``time_periods``
    on the manifest directly so this re-projection can go.
    """
    from datetime import timezone
    assignments = (stats.get('payload', {}).get('split_payload', {}).get(
        'artifacts_v3', {}).get('timestamp_stage_assignments') or [])
    if not assignments:
        return {}
    stage_map = {
        'train': TrainingStage.TRAIN,
        'val': TrainingStage.VAL,
        'test': TrainingStage.TEST,
    }
    by_stage: Dict[TrainingStage, List[int]] = {}
    for entry in assignments:
        stage = stage_map.get(entry.get('stage'))
        ts = entry.get('timestamp')
        if stage is None or ts is None:
            continue
        by_stage.setdefault(stage, []).append(int(ts))
    out: Dict[TrainingStage, TimePeriod] = {}
    for stage, ts_list in by_stage.items():
        if not ts_list:
            continue
        out[stage] = TimePeriod(
            start=datetime.fromtimestamp(min(ts_list), tz=timezone.utc),
            end=datetime.fromtimestamp(max(ts_list), tz=timezone.utc),
            is_pseudo=False,
        )
    return out


def _split_sizes_from_ttng_split_info(split_info: dict) -> SplitSizes:
    """Reshape published ``split_info.stage_counts`` into ``SplitSizes``.

    TODO(wheel-gap): the wheel should emit ``SplitSizes``-shaped
    output natively.
    """
    counts = split_info.get('stage_counts') or {}
    test = counts.get('test')
    out = SplitSizes(
        train=int(counts.get('train', 0)),
        val=int(counts.get('val', 0)),
    )
    if test is not None and int(test) > 0:
        out.test = int(test)
    return out


# TODO: Move this to a classmethod in BinaryConfusionMatrix
def _get_confusion(
    task_type: TaskType,
    target_col_name: str,
    eval_out: EvalOutput,
) -> Optional[BinaryConfusionMatrix]:
    if task_type != TaskType.BINARY_CLASSIFICATION:
        return None
    if eval_out.df is None:
        return None

    df: pd.DataFrame = eval_out.df
    target = df[target_col_name]
    assert isinstance(target, pd.Series)
    prob = df['True_PROB']
    assert isinstance(prob, pd.Series)
    thresholds = np.linspace(0, 1, 101).tolist()
    return get_confusion_matrix(target, prob, thresholds)


def _generate_outputs_for_best_trainer(log_dir: str, home_dir: str,
                                       job_id: JobID) -> None:
    """Generates outputs (eg. regression histogram) from the log_dir of the
    best trainer, to home_dir/outputs, so that the REST API can display
    them on the UI more easily. This also may apply some transformations to
    standardize the data structures, as these data structures should remain
    backwards-compatible across code versions.
    """
    log.debug("Copying outputs for the best trainer: %s", job_id)
    # In theory, it is possible for copying to unexpectedly fail if the k8s
    # trainer is running old (incompatible) version of code, so we catch
    # and log exceptions.
    try:
        generate_regression_histogram(log_dir, home_dir)
    except Exception:
        log.exception(
            "Failed to copy regression histogram for "
            "job_id=%s best_trainer.log_dir=%s home_dir=%s", job_id, log_dir,
            home_dir)

    # TODO(zack): confusion matrix should be moved here, rather than being
    # stored in the db and returned as a temporal RPC response, since it
    # may exceed the size limit of a single temporal message.


def generate_regression_histogram(log_dir: str, home_dir: str) -> None:
    # Regression Histogram Predictions. It is normal for some of these to
    # be missing, because stats are not always generated for all splits.
    stats = AllTargetRegressionStats.empty()

    val_stats = load_json(
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.VAL),
        TargetHistogramAndStatsPair, default=None, raise_on_missing=False)
    if val_stats is not None:
        stats.label_distribution.val_table_stats = val_stats.label
        stats.prediction_distribution.val_table_stats = (val_stats.prediction)

    test_stats = load_json(
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.TEST),
        TargetHistogramAndStatsPair, default=None, raise_on_missing=False)
    if test_stats is not None:
        stats.label_distribution.test_table_stats = test_stats.label
        stats.prediction_distribution.test_table_stats = (test_stats.prediction)

    train_stats = load_json(
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.TRAIN),
        TargetHistogramAndStatsPair, default=None, raise_on_missing=False)
    if train_stats is not None:
        stats.label_distribution.train_table_stats = train_stats.label
        stats.prediction_distribution.train_table_stats = (
            train_stats.prediction)

    mgr = HomeDirManager(home_dir)
    save_json(stats, mgr.all_target_regression_stats)


def _load_metrics_and_confusion_matrix(
    trainer: Trainer,
    target_col_name: str,
    automl_cfg: ModelPlan,
    job_id: JobID,
) -> Tuple[Dict[str, float], Optional[BinaryConfusionMatrix]]:
    r"""Loads the val, test, and train metrics from the given trainer.

    Args:
        trainer: Helps us get the output paths, train result, and confusion
            matrix.
        target_col_name: Name of the target column required for confusion
            matrix.
        automl_cfg: Tells us if we want to get refit results
        job_id: Necessary for logging

    Returns:
        tuple: First element is a dictionary of metrics and second element is
            a :class:`BinaryConfusionMatrix` if the task type is binary
            classification, otherwise :obj:`None`.
    """
    metrics: Dict[str, float] = {}

    log.debug("Getting validation result for the best trainer: %s", job_id)
    val_results_path = EvalOutput.get_output_path(
        trainer.log_dir,
        TrainingStage.VAL,
    )
    try:
        val_results = EvalOutput.load(val_results_path)
    except Exception as ex:
        msg = (
            "Unable to fetch evaluation metrics on the validation dataset. "
            "Training may have failed before evaluation completed; check training logs."
        )
        raise TrainingJobOutputError(
            message=msg,
            error=ex,
            non_retryable=is_error_non_retryable(ex),
        ) from ex
    metrics.update(val_results.metrics)

    log.debug("Getting test result for the best trainer: %s ", job_id)
    test_results_path = EvalOutput.get_output_path(
        trainer.log_dir,
        TrainingStage.TEST,
    )
    test_results: EvalOutput | None = None
    try:
        test_results = EvalOutput.load(test_results_path)
        metrics.update(test_results.metrics)
    except Exception as e:
        log.exception(e)

    if automl_cfg.training_job.refit_trainval:
        log.debug("Getting refit result for the best trainer: %s", job_id)
        orig_fit_mode = trainer.fit_mode
        trainer.fit_mode = FitMode.refit_trainval
        refit_results_path = EvalOutput.get_output_path(
            trainer.get_log_dir(original=False),
            TrainingStage.TEST,
        )
        trainer.fit_mode = orig_fit_mode
        try:
            refit_results = EvalOutput.load(refit_results_path)
            metrics.update({
                key.replace('test_', 'test_refit_'): value
                for key, value in refit_results.metrics.items()
            })
        except Exception as e:
            log.exception(e)

    log.debug("Getting training eval metrics for the best trainer: %s", job_id)
    train_results = _with_train_prefix(trainer.get_train_results())
    metrics.update(train_results)

    confusion_matrix = _get_confusion(
        trainer.task.task_type,
        target_col_name,
        test_results or val_results,
    )

    return (metrics, confusion_matrix)


def _with_train_prefix(results: Dict[str, float]) -> Dict[str, float]:
    """Only copy over the metrics that are used for training, which are
    prefixed by the word "train"
    """
    return {k: v for k, v in results.items() if k.startswith('train')}


# This is an expensive operation that loads data from s3
@lru_cache(maxsize=None)
def _get_metrics(
        job_config_path: str, best_train_dir: str,
        job_id: JobID) -> Tuple[List[Metric], Optional[TestEvalResults]]:
    job_config: JobConfig = JobConfig.load(job_config_path)
    task: Task = load_job_task(job_config)
    best_trainer = Trainer.load(
        task.to_task_config(),
        best_train_dir,
        version=None,
    )

    automl_cfg = load_job_automl_config(job_config)
    metrics, confusion_matrix = _load_metrics_and_confusion_matrix(
        best_trainer,
        task.train_table.target_col_name,
        automl_cfg,
        job_id,
    )

    test_eval = None
    if confusion_matrix is not None:
        test_eval = TestEvalResults(confusion_matrix=confusion_matrix)

    metric_list: List[Metric] = []
    for metric_name in sorted(metrics):
        metric_list.append(Metric(name=metric_name, val=metrics[metric_name]))

    return metric_list, test_eval


def load_job_results(
    job_config_path: str,
    job_id: str,
) -> Tuple[TrainingJobInputJSON, Optional[TrainingJobResults]]:
    # Configuration
    log.debug("Loading configs: %s", job_id)
    job_config: JobConfig = JobConfig.load(job_config_path)
    home_dir_mgr = HomeDirManager(job_config.home_dir)

    # Get input
    # Could not parse to TrainingJobInput due to validator error
    job_input: TrainingJobInputJSON = load_json(
        home_dir_mgr.training_job_input_file, Dict)

    # Load progress
    log.debug("Loading progress: %s", job_id)
    log_dir = home_dir_mgr.automl_dir(job_config.task_name)
    automl_cfg = load_job_automl_config(job_config)
    assert isinstance(automl_cfg, ModelPlan)
    tune_metric = automl_cfg.training_job.tune_metric
    assert isinstance(tune_metric, str)
    reporter = TrainingJobReporter(
        log_dir=log_dir,
        metric=tune_metric,
        metric_mode=MetricStore.tune_mode(tune_metric),
    )

    job_progress = reporter.get_training_job_progress()
    if job_progress is None:
        log.warning("Training progress not available for job %s", job_id)
        return job_input, None

    training_time = job_progress.elapsed_time.seconds
    total_iterations = job_progress.iterations_completed

    # Load metrics
    log.debug("Loading trainer: %s", job_id)
    try:
        best_train_dir = home_dir_mgr.get_best_trainer_dir(job_config.task_name)
    except RuntimeError:
        log.warning("Best trainer not available for job %s", job_id)
        return job_input, None

    metric_list, test_eval = _get_metrics(job_config_path, best_train_dir,
                                          job_id)

    test_explain_file_path = osp.join(best_train_dir,
                                      TRAINING_JOB_TEST_EXPLAIN_FILENAME)

    results = TrainingJobResults(
        total_training_time_sec=training_time,
        total_iterations=total_iterations,
        _test_explain_file_path=test_explain_file_path,
        test_eval=test_eval,
        time_periods=None,
        split_sizes=None,
    ).with_metrics(metric_list)

    return job_input, results
