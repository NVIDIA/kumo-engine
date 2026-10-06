# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from datetime import datetime, timedelta, timezone

import pytorch_lightning as pl
from kumo_core.io.json import save_json
from kumoapi.jobs import PredictionProgress
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch_geometric.data import HeteroData

_PROGRESS_UPDATE_INTERVAL_SEC = 5
_FILE_PREFIX = "prediction_progress_"


class PredictionProgressLogger(Callback):  # pragma: no cover
    r"""Log completion progress for Trainer.predict."""

    FILE_NAME = "prediction_progress.json"  # Filename to log to.
    FILE_NAME_WITH_ID = "prediction_progress_{progress_log_id}.json"

    def __init__(self, log_dir: str, progress_log_id: int,
                 total_iterations: int):
        super().__init__()
        self.path = os.path.join(
            log_dir,
            f'{_FILE_PREFIX}{progress_log_id}.json',
        )
        self.total_iterations = total_iterations

        self.progress = PredictionProgress(total_iterations=total_iterations)
        self.last_update_time: float = 0

        # Save to file
        save_json(self.progress, self.path)

    def on_predict_start(
        self,
        pl_trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.progress.start_time is None:
            self.progress.start_time = datetime.now(timezone.utc)

        # Save to file
        save_json(self.progress, self.path)

    def on_predict_batch_end(
        self,
        pl_trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Tensor],
        batch: HeteroData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Update the number of iterations elapsed
        self.progress.completed_iterations += 1

        # Update the elapsed time
        now = datetime.now(timezone.utc)
        assert self.progress.start_time is not None
        self.progress.elapsed_prediction_time = (now - self.progress.start_time)

        # Calculate the time remaining
        seconds_per_iteration = (self.progress.elapsed_prediction_time.seconds /
                                 self.progress.completed_iterations)
        iterations_remaining = (self.progress.total_iterations -
                                self.progress.completed_iterations)

        self.progress.estimated_prediction_time = timedelta(
            seconds=seconds_per_iteration * iterations_remaining)

        # Save to file
        try:
            if (time.time() - self.last_update_time
                    > _PROGRESS_UPDATE_INTERVAL_SEC):
                save_json(self.progress, self.path)
                self.last_update_time = time.time()
        except RuntimeError:
            # Ignore errors in saving the file
            pass

    def on_predict_end(self, pl_trainer, pl_module):
        # Always save the progress to the file after predict end.
        save_json(self.progress, self.path)
