# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from abc import ABC
from typing import Any

import torch
from kumo_core.io import filesystem as fsutil
from kumo_core.train.base_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
    BaseLightningModule,
)
from kumo_core.train.checkpoint import load_kumo_checkpoint
from kumo_core.train.loss import DistributionLoss, loss_resolver
from kumo_core.types import TaskType, TrainModuleOutput
from kumoapi.train import TrainingStage
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.modeling.calibration import PlatScaling, TemperatureScaling
from torch import Tensor
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
MONTHS_PER_YEAR = 12


class BaseNodeLightningModule(BaseLightningModule, ABC):
    def __init__(
        self,
        task_type: TaskType | str,
        loss: str | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        # `BaseLightningModule` arguments:
        target_encoder: TargetEncoder | None = None,
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
    ) -> None:
        super().__init__(
            task_type=task_type,
            target_encoder=target_encoder,
            metrics=metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            max_epochs_for_lr_scheduler=max_epochs_for_lr_scheduler,
            enable_compile=enable_compile,
        )

        self.loss = loss_resolver(self.task_type, loss, **(loss_kwargs or {}),
                                  target_encoder=target_encoder)

        if self.task_type in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            self.calibrator = PlatScaling(task_type)
        else:
            self.calibrator = TemperatureScaling(task_type)

    # Evaluation Callbacks ####################################################

    def evaluation_epoch_end(
        self,
        stage: TrainingStage,
        step_outputs: list[TrainModuleOutput],
    ) -> TrainModuleOutput:

        outputs = super().evaluation_epoch_end(stage, step_outputs)

        assert outputs is not None
        logits = outputs.get('logits')
        if logits is None:
            raise ValueError("Logits are not available for evaluation")
        assert isinstance(logits, Tensor)

        target = outputs.get('target')
        if target is None:
            raise ValueError("Ground-truth labels are not available for "
                             "evaluation")
        assert isinstance(target, Tensor)

        ckpt_path = self.trainer.ckpt_path  # type: ignore
        if (not self.is_fitting and self.calibrator.is_supported
                and ckpt_path is not None):
            # TODO(matthias): Refactor code to avoid GPU OOM for evaluation.
            logits = logits.to(device=target.device)
            if not self.calibrator.is_fitted:
                self.calibrator.fit(logits, target)
                # Override current checkpoint with calibration parameters.
                # Optimizer states are preserved (unlike trainer.save_checkpoint,
                # which would save an empty list during evaluation).
                with fsutil.open_file(ckpt_path, 'rb') as checkpoint_file:
                    ckpt = load_kumo_checkpoint(
                        checkpoint_file,
                        map_location='cpu',
                    )
                ckpt['state_dict'] = self.state_dict()
                with fsutil.open_file(ckpt_path, 'wb') as checkpoint_file:
                    torch.save(ckpt, checkpoint_file)
            logits = self.calibrator(logits)

            # Save reliability diagrams to the `version_*/` folder:
            base_dir = str(osp.dirname(osp.dirname(ckpt_path)))
            base_dir = base_dir.removeprefix('fsutil://')
            base_dir = base_dir.removeprefix('file://')
            if self.task_type in {
                    TaskType.BINARY_CLASSIFICATION,
                    TaskType.MULTILABEL_CLASSIFICATION,
            }:
                folder = osp.join(base_dir, 'calibration')
                fsutil.makedirs(folder, exist_ok=True)
                filename = f'reliability_diagram_{stage}.png'
                path = osp.join(folder, filename)
                self.calibrator.reliability_diagram(logits.sigmoid(), target,
                                                    path)

            if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                # Save a separate reliability diagram for each label:
                folder = osp.join(base_dir, 'calibration', 'multilabel')
                fsutil.makedirs(folder, exist_ok=True)
                for i in range(logits.size(1)):
                    filename = f'reliability_diagram_{stage}_label_{i}.png'
                    path = osp.join(folder, filename)
                    self.calibrator.reliability_diagram(
                        logits[:, i].sigmoid(),
                        target[:, i],
                        path,
                    )

        prob = outputs.get('prob')
        if not isinstance(self.loss, DistributionLoss):
            prob = self.loss.to_probability(logits)
        else:
            assert isinstance(prob, dict)
            prob = self.loss.to_probability(prob)

        self.outputs = dict(
            pkey=outputs.get('pkey'),
            time=outputs.get('time'),
            pred=self.loss.to_prediction(logits),  # type: ignore
            prob=prob,
            target=target,
        )

        return self.outputs

    # Prediction Callbacks ####################################################

    def on_predict_batch_end(
        self,
        outputs: dict[str, Tensor | None],
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

        logits = outputs.pop('logits', None)  # type: ignore
        if logits is None:
            raise ValueError("Logits are not available for prediction")

        if self.calibrator.is_fitted:
            logits = self.calibrator(logits)
        elif self.calibrator.is_supported:
            logger.warning("Found calibrator which is not fitted. For "
                           "calibrated outputs, run `trainer.validate()` "
                           "or `trainer.test()` beforehand.")

        outputs['pred'] = self.loss.to_prediction(logits).cpu()
        prob = self.loss.to_probability(logits)
        outputs['prob'] = prob.cpu() if prob is not None else None
