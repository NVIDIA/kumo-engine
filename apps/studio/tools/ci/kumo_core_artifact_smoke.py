#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed Kumo-core artifact outside its source checkout."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import kumo_core
import kumoapi
import torch
from kumo_core.evaluation import BinaryAUROC
from kumo_core.explain import Explainer
from kumo_core.modeling import Module
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage

CAPABILITY_MODULES = (
    'kumo_core.encoders',
    'kumo_core.automl.auto_trainer',
    'kumo_core.evaluation',
    'kumo_core.explain',
    'kumo_core.features',
    'kumo_core.modeling',
    'kumo_core.sampling',
    'kumo_core.training',
    'kumo_core.training.runtime',
)


class SmokeModule(Module):
    def init_modules(self) -> None:
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class SmokeExplainer(Explainer):
    supported_task_types = {TaskType.BINARY_CLASSIFICATION}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--forbid-prefix', type=Path, required=True)
    args = parser.parse_args()

    installed_path = Path(kumo_core.__file__).resolve()
    installed_api_path = Path(kumoapi.__file__).resolve()
    checkout = args.forbid_prefix.resolve()
    if installed_path.is_relative_to(checkout):
        raise AssertionError(
            f'kumo_core leaked from checkout: {installed_path}')
    if installed_api_path.is_relative_to(checkout):
        raise AssertionError(
            f'kumoapi leaked from checkout: {installed_api_path}')
    for module in CAPABILITY_MODULES:
        importlib.import_module(module)

    model = SmokeModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0], [0.0]])
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if not torch.isfinite(loss):
        raise AssertionError(f'non-finite training loss: {loss}')

    metric = BinaryAUROC()
    metric.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    if not torch.equal(metric.compute(), torch.tensor(1.0)):
        raise AssertionError('unexpected BinaryAUROC result')

    explainer = SmokeExplainer(TaskType.BINARY_CLASSIFICATION, '.')
    probability, transformed_target = explainer.transform_prediction(
        TrainingStage.VAL,
        {
            'logits': torch.tensor([-1.0, 1.0]),
            'target': torch.tensor([0, 1]),
        },
    )
    if transformed_target is None:
        raise AssertionError('validation target was dropped')
    if not torch.equal(transformed_target, torch.tensor([0.0, 1.0])):
        raise AssertionError('unexpected transformed target')
    if not torch.allclose(probability, torch.tensor([0.2689, 0.7311]),
                          atol=1e-4):
        raise AssertionError('unexpected transformed probability')

    print(f'Kumo-core artifact smoke passed from {installed_path}')


if __name__ == '__main__':
    main()
