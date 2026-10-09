# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytorch_lightning as pl
import torch
from kumoapi.train import TrainingStage
from kumo_core.evaluation.suite import MetricStore
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

logger = logging.getLogger(__name__)

# No need to include FITTING because AutoTrainer re-creates a new kumo.Trainer
# when an AutoML trial fails on CUDA OOM. For other functions, we need to reset
# metrics as the accumulated states will be leaked in the subsequent retries.
_TRAINER_FN_TO_STAGE = {
    TrainerFn.VALIDATING: TrainingStage.VAL,
    TrainerFn.TESTING: TrainingStage.TEST,
}


class CudaOOMHandler(Callback):
    r"""Reset metrics on CUDA OOM."""
    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: Exception,
    ) -> None:
        if (isinstance(exception, torch.cuda.OutOfMemoryError)
                and trainer.state.fn in _TRAINER_FN_TO_STAGE):
            logger.info("Resetting metrics due to CUDA OOM")
            metric_store: MetricStore = pl_module.metric_store
            stage = _TRAINER_FN_TO_STAGE[trainer.state.fn]
            metric = metric_store.get_metric_collection(stage)
            metric.reset()
