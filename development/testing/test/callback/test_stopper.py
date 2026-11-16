# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from kumo_core.callbacks.stopper import TopKEarlyStopping
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(LightningModule):
    """A model with overridden training return."""
    def __init__(self, return_values) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.return_values = return_values

    def on_train_epoch_end(self):
        loss = self.return_values[self.current_epoch]
        self.log("train_loss", loss)

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(
            prediction,
            torch.ones_like(prediction),
        )

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@pytest.mark.parametrize('min_epochs', [1, 2, 3])
def test_top_k_early_stopping(min_epochs):
    loss_values = torch.tensor([[10, 6, 5, 5, 5, 5], [9, 5, 4, 4, 4, 3],
                                [8, 5, 3, 3, 3, 2], [11, 10, 10, 10, 10, 10]])
    num_trials, max_epochs = loss_values.size()
    expected_stop_epochs = [6, 6, 6, min_epochs]
    stop_epochs = []

    early_stop_callback = TopKEarlyStopping(
        monitor="train_loss",
        min_epochs=min_epochs,
        k=2,
        continuing_mode='any',
        mode="min",
        check_on_train_epoch_end=True,
    )

    for i in range(num_trials):
        model = Model(loss_values[i])
        trainer = Trainer(
            num_sanity_val_steps=0,
            max_epochs=max_epochs,
            enable_progress_bar=False,
        )
        trainer.callbacks.append(early_stop_callback)
        trainer.fit(model)
        stop_epochs.append(trainer.current_epoch)

        if i == 0:
            assert early_stop_callback.top_k_trials_scores == [
                loss_values[0].tolist(), []
            ]
        if i == 1:
            assert early_stop_callback.top_k_trials_scores == [
                loss_values[0].tolist(), loss_values[1].tolist()
            ]
        if i == 2 or i == 3:
            assert early_stop_callback.top_k_trials_scores == [
                loss_values[2].tolist(), loss_values[1].tolist()
            ]

    assert stop_epochs == expected_stop_epochs
