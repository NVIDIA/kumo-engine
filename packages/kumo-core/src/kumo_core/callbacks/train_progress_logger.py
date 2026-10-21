# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from datetime import datetime, timezone
from typing import Any, Callable

import kumoapi.jobs as jobs_api
from kumo_core.io.json import load_json, save_json
from kumoapi.jobs import EpochMetrics, TrainerProgress
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.trainer.states import TrainerFn

logger = logging.getLogger(__name__)

# In Pytorch Lightning, at the end of the fit_loop, the callbacks are called in
# the following order:
# 1. `LightningModule.on_validation_epoch_end`: computes validation metrics
# 2. on_train_epoch_end:
#    2.1. All callbacks `on_train_epoch_end` except the monitoring callbacks
#         `(EarlyStopping, Checkpoint)`
#         Here is where the `TrainingProgressLogger` would log the metrics if
#         it extended `Callback` instead of `Checkpoint`.
#    2.2. LightningModule.on_train_epoch_end:
#         Here is where we compute the training metrics.
#    2.3. Monitoring callbacks on_train_epoch_end: callbacks extending
#         `(EarlyStopping, Checkpoint)`
#
# See https://github.com/Lightning-AI/pytorch-lightning/blob/5d89996472994be708eda253a30fa1b79a31f23c/src/lightning/pytorch/loops/fit_loop.py#L464-L470 # pylint: disable=C0301
# See https://github.com/Lightning-AI/pytorch-lightning/blob/2.2.1/src/lightning/pytorch/trainer/call.py#L198-L202 # pylint: disable=C0301
#
# In order to log the training metrics for the current epoch in the
# `TrainingProgressLogger` we need to call the `on_train_epoch_end` hook after
# the `LightningModule.on_train_epoch_end`hook.
# This is done by extending `Checkpoint` instead of `Callback`.
# This is a hack given that Pytorch Lightning does not provide a way to create
# "MonitoringCallbacks". Since Checkpoint is an empty subclass of Callback,
# at the moment there is no risk in doing this, but it should be revisited in
# future updates of Pytorch Lightning versions.
MonitoringCallback = Checkpoint


class AutoTrainerProgress(jobs_api.AutoTrainerProgress):
    @staticmethod
    def load(automl_dir: str) -> 'AutoTrainerProgress':
        path = osp.join(automl_dir, 'auto_train_progress.json')
        progress = load_json(
            path,
            AutoTrainerProgress,
            default=AutoTrainerProgress(),
            raise_on_missing=False,
        )
        assert progress is not None
        return progress


class TrainingProgressLogger(MonitoringCallback):  # pragma: no cover
    r"""Log all train and validation metrics during model training."""

    FILE_NAME = "train_progress.json"  # Filename to log to.

    def __init__(self, trainer: Any):
        super().__init__()
        self.path = osp.join(trainer.log_dir, self.FILE_NAME)
        self.max_epochs = trainer.max_epochs
        self.tune_metric = trainer.tune_metric

        self.progress = TrainerProgress()
        # Trainer initialization is also used for validation, test, and
        # prediction. Defer the first write until training actually starts so
        # those flows cannot replace completed progress with an empty record.

        self._epoch_start: datetime | None = None

    def on_train_start(self, pl_trainer, pl_module):
        self.progress.start_time = datetime.now(timezone.utc)
        self.progress.accelerator = pl_module.device.type
        save_json(self.progress, self.path)

    def on_train_epoch_start(self, pl_trainer, pl_module):
        self._epoch_start = datetime.now(timezone.utc)

    def on_train_epoch_end(self, pl_trainer, pl_module):
        now = datetime.now(timezone.utc)
        curr_epoch = self.progress.num_epochs_completed

        self.progress.num_epochs_completed += 1
        self.progress.elapsed_time = (now -
                                      self.progress.start_time).total_seconds()
        self.progress.estimated_train_time = (
            (self.progress.elapsed_time / self.progress.num_epochs_completed) *
            self.max_epochs)

        all_metrics = pl_trainer.progress_bar_metrics
        epoch_metrics = EpochMetrics()
        epoch_metrics.time_taken = (now - self._epoch_start).total_seconds()
        epoch_metrics.train_metrics = {
            key: value
            for key, value in all_metrics.items() if key.startswith('train_')
        }
        epoch_metrics.validation_metrics = {
            key: value
            for key, value in all_metrics.items() if key.startswith('val_')
        }
        self.progress.metrics[curr_epoch] = epoch_metrics

        self._epoch_start = None
        save_json(self.progress, self.path)

        train_metric = f'train_{self.tune_metric}'
        if train_metric not in epoch_metrics.train_metrics:
            train_metric = 'train_loss'  # Fallback
        val_metric = f'val_{self.tune_metric}'
        if val_metric not in epoch_metrics.validation_metrics:
            val_metric = 'val_loss'  # Fallback

        train_perf = epoch_metrics.train_metrics[train_metric]
        epoch_info = f'Epoch: {self.progress.num_epochs_completed - 1}, '
        epoch_info += f'{train_metric}: {train_perf:.5f}'
        # check if we have validation performance to report since we do not
        # report it in refit(FitMode.refit_full)
        if val_metric in epoch_metrics.validation_metrics:
            val_perf = epoch_metrics.validation_metrics[val_metric]
            epoch_info += f', {val_metric}: {val_perf:.5f}'
        logger.info(epoch_info)

    def on_validation_end(self, pl_trainer, pl_module):
        if pl_trainer.state.fn == TrainerFn.VALIDATING:
            self.progress.final_validation_metrics = (
                pl_trainer.progress_bar_metrics)
            save_json(self.progress, self.path)


class AutoTrainingProgressLogger(MonitoringCallback):  # pragma: no cover
    r"""Log all train and validation metrics during model training."""

    FILE_NAME = "auto_train_progress.json"  # Filename to log to.
    # Constant to account for overhead not captured in trial estimation:
    OVERHEAD_CONSTANT = 1.30

    def __init__(self, log_dir: str, num_experiments: int):
        super().__init__()
        self.path = osp.join(log_dir, self.FILE_NAME)

        self.progress = load_json(self.path, AutoTrainerProgress,
                                  raise_on_missing=False)
        if self.progress is None:
            self.progress = AutoTrainerProgress(total_trials=num_experiments)

    def set_current_trainer(self, trainer: Any | None):
        self.trainer = trainer

    def on_train_start(self, pl_trainer, pl_module):
        if self.progress.start_time is None:
            self.progress.start_time = datetime.now(timezone.utc)
        self.progress.trial_progress[self.trainer.version] = (
            self.trainer.training_progress_logger.progress)
        save_json(self.progress, self.path)

    def on_train_end(self, pl_trainer, pl_module):
        self.progress.completed_trials += 1

        save_json(self.progress, self.path)

    def on_train_epoch_end(self, pl_trainer, pl_module):
        now = datetime.now(timezone.utc)

        self.progress.elapsed_training_time = (
            now - self.progress.start_time).total_seconds()

        if self.progress.completed_trials == 0:
            training_progress_logger = self.trainer.training_progress_logger
            self.progress.estimated_training_time = (
                self.progress.total_trials *
                training_progress_logger.progress.estimated_train_time)
        else:
            progresses = list(self.progress.trial_progress.values())[:-1]
            self.progress.estimated_training_time = (
                sum([p.elapsed_time
                     for p in progresses]) * self.OVERHEAD_CONSTANT *
                (self.progress.total_trials / self.progress.completed_trials))

        save_json(self.progress, self.path)

    def on_validation_end(self, pl_trainer, pl_module):
        if pl_trainer.state.fn == TrainerFn.VALIDATING:
            save_json(self.progress, self.path)


class EpochEventEmitter(MonitoringCallback):
    def __init__(self, on_event: Callable[[dict[str, Any]], None],
                 trial: int):
        super().__init__()
        self.on_event = on_event
        self.trial = trial

    def on_train_epoch_end(self, pl_trainer, pl_module):
        try:
            self.on_event({
                'name': 'epoch_ended',
                'trial': self.trial,
                'epoch': pl_trainer.current_epoch + 1,
                'metrics': {
                    key: float(value)
                    for key, value in pl_trainer.progress_bar_metrics.items()
                    if key.startswith(('train_', 'val_'))
                },
            })
        except Exception:
            logger.warning("Could not report progress event 'epoch_ended'",
                           exc_info=True)
