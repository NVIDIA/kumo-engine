# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta

from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.callbacks import PredictionProgressLogger
from kumoapi.jobs import TrainerProgress

from io_utils import logging
from kumo.train.utils.progress_util import get_prediction_progress
from rest.config.batch_prediction_job import BatchPredictionJobProgress
from rest.config.training_job import Metric, TrainingJobProgress, TrialProgress

log = logging.getLogger(__name__)


class BatchPredictionJobReporter:
    def __init__(
        self,
        log_dir: str,
    ):
        r"""
        Class for reporting batch prediction's job progress periodically to the
        output store.

        Args:
            log_dir (str): log directory of Trainer
        """
        self._log_dir = log_dir
        self._progress: BatchPredictionJobProgress | None = None

        log.info(
            'Initializing Batch Prediction Job reporter reading from %s',
            os.path.join(self._log_dir, PredictionProgressLogger.FILE_NAME))

    def get_batch_prediction_job_progress(
            self) -> BatchPredictionJobProgress | None:
        r"""
        Returns :obj:`None` if there is no progress information available.
        """
        progress = get_prediction_progress(self._log_dir)
        if not progress:
            return None
        iterations_completed = progress.completed_iterations
        iterations_remaining = (progress.total_iterations -
                                iterations_completed)
        start_time = progress.start_time
        elapsed_time = progress.elapsed_prediction_time
        # Progress file is first written before job actually starts
        if iterations_completed > 0 and elapsed_time > timedelta(seconds=0):
            assert start_time is not None
        else:
            return None
        estimated_remaining_time = progress.estimated_prediction_time
        estimated_end_time = start_time + elapsed_time + (
            estimated_remaining_time or timedelta(seconds=0))
        self._progress = BatchPredictionJobProgress(
            iterations_completed,
            iterations_remaining,
            start_time,
            elapsed_time,
            estimated_end_time,
        )
        return self._progress


class TrainingJobReporter:
    def __init__(
        self,
        log_dir: str,
        metric: str,
        metric_mode: str,
    ):
        r"""
        Class for reporting training job progress periodically to the
        output store.

        Args:
            log_dir (str): log directory of AutoTrainer
            metric (str): Metric to report
            metric_mode (str): How to optimize the metric in question
        """
        assert metric_mode in {"min", "max"}

        self._log_dir = log_dir
        self._metric = metric
        self._metric_mode = metric_mode
        self._progress: TrainingJobProgress | None = None

    # Trainer Progress utility methods ########################################
    @staticmethod
    def readable_metric_name(metric: str):
        # A mapping of metric to user-friendly names
        METRIC_NAME_MAPPING = {
            "acc": "Accuracy",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "loss": "Loss",
        }
        if metric in METRIC_NAME_MAPPING:
            return METRIC_NAME_MAPPING[metric]
        else:
            return metric

    def get_best_metric(
        self,
        progress: TrainerProgress,
        stage: str,
    ) -> float | None:
        r"""
        Returns the best metric across all the epochs. If there are no metrics
        with the given name or stage, returns None
        """
        assert stage in {"train", "val"}, "stage must be one of {train, val}"

        key = "_".join([stage, self._metric])
        if stage == "train":
            metric_values = [
                m.train_metrics[key] for m in progress.metrics.values()
                if key in m.train_metrics
            ]
        else:
            if key in (progress.final_validation_metrics or {}):
                return progress.final_validation_metrics[key]

            metric_values = [
                m.validation_metrics[key] for m in progress.metrics.values()
                if key in m.validation_metrics
            ]

        if self._metric_mode == "max":
            return max(metric_values) if metric_values else None
        else:
            return min(metric_values) if metric_values else None

    # TODO(kiriti): Add a cache to store finished trial's progress instead of
    # reading from S3 everytime.
    def get_trial_progress(
        self,
        trial_id: str,
        progress: TrainerProgress,
    ) -> TrialProgress:
        # Get training and validation metrics of the best epoch
        num_epochs = progress.num_epochs_completed
        trial_progress = TrialProgress(trial_id)
        if num_epochs > 0:
            # In some cases even when the epoch has started, we might not be
            # guaranteed to have both train and validation metrics. In that
            # case, get_best_metric would return None.
            best_train_metric_value = self.get_best_metric(
                progress, stage="train")
            if best_train_metric_value is not None:
                trial_progress.best_train_metric = Metric(
                    self.readable_metric_name(self._metric),
                    best_train_metric_value)

            best_validation_metric_value = self.get_best_metric(
                progress, stage="val")
            if best_validation_metric_value is not None:
                trial_progress.best_validation_metric = Metric(
                    self.readable_metric_name(self._metric),
                    best_validation_metric_value)

        return trial_progress

    def get_training_job_progress(self) -> TrainingJobProgress | None:
        r"""
        Returns :obj:`None` if there is no progress information available.
        """
        auto_trainer_progress = AutoTrainer.get_training_progress(self._log_dir)
        if not auto_trainer_progress:
            return None
        iterations_completed = auto_trainer_progress.completed_trials
        iterations_remaining = (auto_trainer_progress.total_trials -
                                iterations_completed)
        start_time = auto_trainer_progress.start_time
        assert start_time is not None
        elapsed_time = timedelta(
            seconds=auto_trainer_progress.elapsed_training_time)
        estimated_end_time = auto_trainer_progress.estimated_training_time
        if estimated_end_time is not None:
            estimated_end_time = start_time + timedelta(
                seconds=estimated_end_time)

        trial_progress: list[TrialProgress] = []
        for trial_id, progress in auto_trainer_progress.trial_progress.items():
            trial_progress.append(self.get_trial_progress(trial_id, progress))
        self._progress = TrainingJobProgress(
            iterations_completed,
            iterations_remaining,
            start_time,
            elapsed_time,
            estimated_end_time,
            trial_progress,
            metric_mode=self._metric_mode,
        )
        return self._progress
