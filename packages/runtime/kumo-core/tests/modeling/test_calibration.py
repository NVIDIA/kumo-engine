# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os.path as osp
import pathlib

import pytest
import torch
import torch.nn.functional as F
from kumoapi.task import TaskType
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.modeling.calibration import PlatScaling, TemperatureScaling


def get_logits_and_target(
    task_type: TaskType,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        logits = torch.rand(100, 1, device=device) - 0.5
        logits[(logits >= -0.1) & (logits <= 0.0)] = -0.1
        logits[(logits <= +0.1) & (logits >= 0.0)] = 0.1
        y = (logits > 0).to(torch.long)
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        logits = torch.rand(4, 6, device=device) - 0.5
        y = logits.argmax(dim=-1, keepdim=True)
        logits.scatter_add_(1, y, torch.ones_like(y, dtype=logits.dtype))
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        logits = torch.rand(100, 6, device=device) - 0.5
        logits[(logits >= -0.1) & (logits <= 0.0)] = -0.1
        logits[(logits <= +0.1) & (logits >= 0.0)] = 0.1
        y = (logits > 0).to(torch.long)
    else:
        raise NotImplementedError

    return logits, y


@withCUDA
@pytest.mark.parametrize('task_type', TaskType.get_classification_tasks())
def test_temperature_scaling(
    task_type: TaskType,
    device: torch.device,
) -> None:
    model = TemperatureScaling(task_type).to(device)
    assert str(model) == 'TemperatureScaling()'
    assert model.temperature.tolist() == [1.0]

    logits, y = get_logits_and_target(task_type, device=device)

    with pytest.raises(RuntimeError, match="is not yet fitted"):
        model(logits)

    assert not model.is_fitted
    model.fit(logits, y)
    assert model.is_fitted

    with pytest.raises(RuntimeError, match="is already fitted"):
        model.fit(logits, y)

    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        assert model.temperature.size() == (6, )
    else:
        assert model.temperature.size() == (1, )

    assert torch.all(model.temperature < 0.2)

    out = model(logits)
    assert out.dtype == logits.dtype
    assert out.device == logits.device
    assert out.size() == logits.size()

    if task_type == TaskType.BINARY_CLASSIFICATION:
        orig_loss = F.binary_cross_entropy_with_logits(logits, y.float())
        new_loss = F.binary_cross_entropy_with_logits(out, y.float())
        prob = out.sigmoid()
        mask = y == 1
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        orig_loss = F.cross_entropy(logits, y.view(-1))
        new_loss = F.cross_entropy(out, y.view(-1))
        prob = out.softmax(dim=-1)
        mask = torch.zeros_like(prob, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), True)
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        orig_loss = F.binary_cross_entropy_with_logits(logits, y.float())
        new_loss = F.binary_cross_entropy_with_logits(out, y.float())
        prob = out.sigmoid()
        mask = y == 1
    else:
        raise NotImplementedError

    assert new_loss < orig_loss
    assert prob[~mask].allclose(torch.tensor([0.0], device=device), atol=1e-2)
    assert prob[mask].allclose(torch.tensor([1.0], device=device), atol=1e-2)

    state_dict = model.state_dict()
    assert state_dict['_extra_state']['is_fitted']
    model = TemperatureScaling(task_type)
    model.load_state_dict(state_dict)
    assert model.is_fitted


@withCUDA
@pytest.mark.parametrize('task_type', [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
])
def test_plat_scaling(task_type: TaskType, device: torch.device) -> None:
    model = PlatScaling(task_type).to(device)
    assert str(model) == 'PlatScaling()'
    assert model.weight_and_bias.tolist() == [[1.0], [0.0]]

    logits, y = get_logits_and_target(task_type, device=device)

    with pytest.raises(RuntimeError, match="is not yet fitted"):
        model(logits)

    assert not model.is_fitted
    model.fit(logits, y)
    assert model.is_fitted

    with pytest.raises(RuntimeError, match="is already fitted"):
        model.fit(logits, y)

    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        assert model.weight_and_bias.size() == (2, 6)
    else:
        assert model.weight_and_bias.size() == (2, 1)

    out = model(logits)
    assert out.dtype == logits.dtype
    assert out.device == logits.device
    assert out.size() == logits.size()

    if task_type == TaskType.BINARY_CLASSIFICATION:
        orig_loss = F.binary_cross_entropy_with_logits(logits, y.float())
        new_loss = F.binary_cross_entropy_with_logits(out, y.float())
        prob = out.sigmoid()
        mask = y == 1
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        orig_loss = F.binary_cross_entropy_with_logits(logits, y.float())
        new_loss = F.binary_cross_entropy_with_logits(out, y.float())
        prob = out.sigmoid()
        mask = y == 1
    else:
        raise NotImplementedError

    assert new_loss < orig_loss
    assert prob[~mask].allclose(torch.tensor([0.0], device=device), atol=1e-2)
    assert prob[mask].allclose(torch.tensor([1.0], device=device), atol=1e-2)

    state_dict = model.state_dict()
    assert state_dict['_extra_state']['is_fitted']
    model = PlatScaling(task_type)
    model.load_state_dict(state_dict)
    assert model.is_fitted


@withCUDA
def test_reliability_diagram(
    tmp_path: pathlib.Path,
    device: torch.device,
) -> None:
    model = TemperatureScaling(TaskType.BINARY_CLASSIFICATION).to(device)

    prob = torch.rand(10, device=device)
    prob[0] = 0.0
    prob[-1] = 1.0
    y = torch.randint(0, 2, (10, ), device=device)
    path = osp.join(tmp_path, 'reliability_plot.png')

    model.reliability_diagram(prob, y, path, num_bins=5)
    assert osp.exists(path)


def test_unsupported_task_type() -> None:
    model = PlatScaling(TaskType.REGRESSION)
    logits, y = get_logits_and_target(TaskType.BINARY_CLASSIFICATION)

    with pytest.raises(RuntimeError, match="does not support the given task"):
        model.fit(logits, y)

    with pytest.raises(RuntimeError, match="does not support the given task"):
        model(logits)
