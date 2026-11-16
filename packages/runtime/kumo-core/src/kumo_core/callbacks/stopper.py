# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-trial early stopping for portable AutoML."""

import copy
import logging
import math
import operator
import os.path as osp
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

log = logging.getLogger(__name__)


class TopKEarlyStopping(EarlyStopping):
    r"""Monitor a metric and stop training of a trial when it stops improving.
    Scores of top K performing trials will be recorded to determine if the
    current trial should be stopped or not. It is used as a callback across
    trials in :class:`AutoTrainer` to early stop unpromising AutoML trials
    based on the performance of previous trials to speedup the AutoML search
    process.

    Args:
        monitor (str): The quantity to be monitored.
        min_epochs (int, optional): The min training epochs of a trial before
            running early stopping check. (default: :obj:`1`)
        k (int, optional): The number of top performing trials to track.
            (default: :obj:`3`)
        continuing_mode (str, optional): The mode to continue the training of
            a trial. If set to :obj:`"any"`, the trial will continue if the
            current score is better than any historical score of the top k
            trials. If set to :obj:`"all"`, the trial will continue if the
            current score is better than all historical scores of the top k
            trials. (default: :obj:`"any"`)
        min_delta (float, optional): The minimum change in the monitored
            quantity to qualify as an improvement, i.e. an absolute change of
            less than or equal to :obj:`min_delta`, will count as no
            improvement. (default: :obj:`0.0`)
        patience (int, optional): The number of checks with no improvement
            after which training will be stopped. Under the default
            configuration, one check happens after every training epoch.
            However, the frequency of validation can be modified by setting
            various parameters in the :obj:`Trainer`, for example
            :obj:`check_val_every_n_epoch` and :obj:`val_check_interval`.
            (default: :obj:`0`)
        mode (str, optional): One of :obj:`"min"`, :obj:`"max"`. In
            :obj:`"min"` mode, training will stop when the quantity monitored
            has stopped decreasing and in :obj:`"max"` mode it will stop when
            the quantity monitored has stopped increasing.
            (default: :obj:`"min"`)
         **kwargs (optional): Additional arguments of
            :class:`pytorch_lightning.callbacks.EarlyStopping`.
    """
    mode_dict = {"min": operator.lt, "max": operator.gt}
    order_dict = {"min": "<", "max": ">"}
    continuing_mode_list = ["any", "all"]
    isfinite = math.isfinite

    def __init__(
        self,
        monitor: str,
        min_epochs: int = 1,
        k: int = 3,
        continuing_mode: str = "any",
        min_delta: float = 0.0,
        patience: int = 0,
        mode: str = "min",
        **kwargs,
    ):
        super().__init__(
            monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            **kwargs,
        )
        self.min_epochs = min_epochs
        self.reduce_fn_epoch = min if self.mode == "min" else max
        self.reduce_fn_trial = max if self.mode == "min" else min
        self.updated_top_k = False
        self.worst_trial_idx = None

        inf_value = float('inf')
        if mode == 'min':
            self.fill_value = inf_value
        else:
            self.fill_value = -inf_value

        self.top_k_trials_scores = [[] for _ in range(k)]
        self.current_trial_score = []
        self.top_k_trials_dirs = ['' for _ in range(k)]

        if continuing_mode not in self.continuing_mode_list:
            raise ValueError(
                f"`continuing_mode` can be in "
                f"{self.continuing_mode_list}, got {continuing_mode}.")
        if continuing_mode == "any":
            self.reduce_fn_score = max if self.mode == "min" else min
        if continuing_mode == "all":
            self.reduce_fn_score = min if self.mode == "min" else max

    def state_dict(self) -> dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
            "top_k_trials_scores": self.top_k_trials_scores,
            "top_k_trials_dirs": self.top_k_trials_dirs,
            "current_trial_score": self.current_trial_score,
            "updated_top_k": self.updated_top_k,
            "worst_trial_idx": self.worst_trial_idx,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.patience = state_dict["patience"]
        self.top_k_trials_scores = state_dict["top_k_trials_scores"]
        self.top_k_trials_dirs = state_dict["top_k_trials_dirs"]
        self.current_trial_score = state_dict["current_trial_score"]
        self.updated_top_k = state_dict["updated_top_k"]
        self.worst_trial_idx = state_dict["worst_trial_idx"]

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        r""" Checks whether the early stopping condition is met and if so tells
        the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        # Disable `early_stopping` with `fast_dev_run`
        # Short circuit if metric not present
        if trainer.fast_dev_run or not self._validate_condition_metric(logs):
            return

        # Reset the `early_stopping` state at start
        if trainer.current_epoch == 0:
            self._reset_state()

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(
            current,
            trainer.current_epoch,
            osp.dirname(trainer.log_dir),
        )

        # Stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evaluate_stopping_criteria(
        self,
        current: torch.Tensor,
        current_epoch: int,
        current_dir: str,
    ) -> tuple[bool, str | None]:
        current = current.item()
        should_stop = False
        reason = None
        historical_score = self._get_historical_score(current_epoch)
        self._update_current_trial_score(current)

        # Only start `early_stopping` after training for `min_epochs`
        if current_epoch < self.min_epochs - 1:
            return False, reason

        if self.check_finite and not self.isfinite(current):
            should_stop = True
            reason = (
                f"\n{self.__class__.__name__}: "
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Historical Score was {historical_score:.3f}. "
                "Signaling Trainer to stop.")
        elif self.stopping_threshold is not None and self.monitor_op(
                current, self.stopping_threshold):
            should_stop = True
            reason = (
                f"\n{self.__class__.__name__}: "
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} "
                f"{self.stopping_threshold}. Signaling Trainer to stop.")
        elif self.divergence_threshold is not None and self.monitor_op(
                -current, -self.divergence_threshold):
            should_stop = True
            reason = (
                f"\n{self.__class__.__name__}: "
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} "
                f"{self.divergence_threshold}. Signaling Trainer to stop.")
        elif self.monitor_op(current - self.min_delta, historical_score):
            should_stop = False
            reason = self._improvement_message(current, historical_score)
            self.wait_count = 0
            if not self.updated_top_k:
                self._update_top_k_trials(current_dir)
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (f"\n{self.__class__.__name__}: "
                          f"Monitored metric {self.monitor} did not improve in "
                          f"the last {self.wait_count} records. "
                          f"Historical Score: {historical_score:.3f}. "
                          f"Current Score: {current:.3f}. "
                          f"Min delta: {self.min_delta:.3f}. "
                          "Signaling Trainer to stop.")

        return should_stop, reason

    def _update_current_trial_score(
        self,
        current: float,
    ) -> None:
        if not self.updated_top_k:
            self.current_trial_score.append(current)
        else:
            self.current_trial_score.append(current)
            self.top_k_trials_scores[self.worst_trial_idx].append(current)

    def _update_top_k_trials(self, current_dir: str) -> None:
        trial_best_scores = []
        for epoch_scores in self.top_k_trials_scores:
            if epoch_scores == []:
                trial_best_scores.append(self.fill_value)
            else:
                trial_best_scores.append(self.reduce_fn_epoch(epoch_scores))

        worst_trial_best_score = self.reduce_fn_trial(trial_best_scores)
        worst_trial_idx = trial_best_scores.index(worst_trial_best_score)

        current_trial_best_score = self.reduce_fn_epoch(
            self.current_trial_score)

        if self.monitor_op(current_trial_best_score, worst_trial_best_score):
            self.top_k_trials_scores[worst_trial_idx] = copy.deepcopy(
                self.current_trial_score)
            self.top_k_trials_dirs[worst_trial_idx] = current_dir
            self.updated_top_k = True
            self.worst_trial_idx = worst_trial_idx

    def _reset_state(self) -> None:
        self.wait_count = 0
        self.stopped_epoch = 0
        self.updated_top_k = False
        self.worst_trial_idx = None
        self.current_trial_score = []

    def _get_historical_score(self, current_epoch: int):
        # Get the latest score no later than the `current_epoch`
        # The current trial will be included after `updated_top_k`
        historical_scores = []
        for epoch_scores in self.top_k_trials_scores:
            if len(epoch_scores) == 0:
                historical_scores.append(self.fill_value)
            elif len(epoch_scores) < current_epoch + 1:
                historical_scores.append(epoch_scores[-1])
            else:
                historical_scores.append(epoch_scores[current_epoch])
        return self.reduce_fn_score(historical_scores)

    def _improvement_message(
        self,
        current: float,
        historical_score: float,
    ) -> str:
        r"""Formats a log message that informs the user about an improvement
        in the monitored score.
        """
        if self.isfinite(historical_score):
            msg = (f"\n{self.__class__.__name__}: "
                   f"Metric {self.monitor} improved by "
                   f"{abs(historical_score - current):.3f} >= "
                   f"min_delta = {abs(self.min_delta)}. "
                   f"New best score: {current:.3f}.")
        else:
            msg = (f"\n{self.__class__.__name__}: "
                   f"Metric {self.monitor} improved. "
                   f"New best score: {current:.3f}.")
        return msg
