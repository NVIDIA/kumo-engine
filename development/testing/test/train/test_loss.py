# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper
from kumo_core.evaluation.distribution import STANDARD_QUANTILE_LEVELS
from kumo_core.train.loss import (
    HuberLoss,
    MAELoss,
    MSELoss,
    MultiQuantileLoss,
    OrdinalLogLoss,
    WAPELoss,
    loss_resolver,
)
from kumo_core.types import TaskType


@pytest.mark.parametrize('weighted', [False, True])
def test_mae_loss(weighted: bool) -> None:
    y_pred = torch.randn(4)
    y_true = torch.randn(4)
    weight = torch.tensor([1.0, 2.0, 1.0, 2.0]) if weighted else None

    loss_fn = MAELoss(task_type=TaskType.REGRESSION)
    loss = loss_fn(y_pred, y_true, weight=weight)
    expected = (y_pred - y_true).abs()
    if weight is None:
        assert loss.allclose(expected.mean())
    else:
        assert loss.allclose((expected * weight).mean())


@pytest.mark.parametrize('weighted', [False, True])
def test_mse_loss(weighted: bool) -> None:
    y_pred = torch.randn(4)
    y_true = torch.randn(4)
    weight = torch.randn(4) if weighted else None

    loss_fn = MSELoss(task_type=TaskType.REGRESSION)
    loss = loss_fn(y_pred, y_true, weight=weight)
    expected = (y_pred - y_true).pow(2)
    if weight is None:
        assert loss.allclose(expected.mean())
    else:
        assert loss.allclose((expected * weight).mean())


@pytest.mark.parametrize('weighted', [False, True])
def test_wape_loss(weighted: bool) -> None:
    y_pred = torch.randn(4)
    y_true = torch.tensor([1.0, -2.0, 3.0, -4.0])
    weight = torch.tensor([1.0, 2.0, 1.0, 2.0]) if weighted else None

    loss_fn = WAPELoss(task_type=TaskType.REGRESSION)
    loss = loss_fn(y_pred, y_true, weight=weight)

    abs_error = (y_pred - y_true).abs()
    if weight is None:
        scale = y_true.abs().sum()
        assert loss.allclose(abs_error.sum() / scale)
    else:
        scale = (y_true.abs() * weight).sum()
        assert loss.allclose((abs_error * weight).sum() / scale)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('delta', [1.0, 2.0])
def test_huber_loss(weighted: bool, delta: float) -> None:
    y_pred = torch.randn(4)
    y_true = torch.randn(4)
    weight = torch.randn(4) if weighted else None

    loss_fn = HuberLoss(task_type=TaskType.REGRESSION, delta=delta)
    loss = loss_fn(y_pred, y_true, weight=weight)
    mask = (y_pred - y_true).abs() < delta
    expected = torch.where(
        mask,
        0.5 * (y_pred - y_true).pow(2),
        delta * ((y_pred - y_true).abs() - 0.5 * delta),
    )
    if weight is None:
        assert loss.allclose(expected.mean())
    else:
        assert loss.allclose((expected * weight).mean())


@pytest.mark.parametrize('weighted', [False, True])
def test_multi_quantile_loss(weighted: bool) -> None:
    quantiles = STANDARD_QUANTILE_LEVELS
    y_pred = torch.stack([
        torch.linspace(1.0, 2.0, len(quantiles)),
        torch.linspace(2.0, 3.0, len(quantiles)),
    ])
    y_true = torch.tensor([2.0, 1.0])
    weight = torch.tensor([1.0, 2.0]) if weighted else None

    loss_fn = MultiQuantileLoss(task_type=TaskType.REGRESSION)
    loss = loss_fn(y_pred, y_true, weight=weight)

    quantiles_tensor = torch.tensor(quantiles)
    error = torch.where(
        y_pred <= y_true[:, None],
        quantiles_tensor * (y_true[:, None] - y_pred),
        (1.0 - quantiles_tensor) * (y_pred - y_true[:, None]),
    )
    if weight is not None:
        error = error * weight[:, None]
    assert loss.allclose(error.mean())


def test_multi_quantile_loss_outputs() -> None:
    loss_fn = MultiQuantileLoss(task_type=TaskType.FORECASTING)
    regression_loss_fn = MultiQuantileLoss(task_type=TaskType.REGRESSION)
    y_pred = torch.stack([
        torch.linspace(2.0, 1.0, len(loss_fn.quantiles)),
        torch.linspace(5.0, 4.0, len(loss_fn.quantiles)),
    ]).unsqueeze(0)

    expected = y_pred.sort(dim=-1).values
    assert torch.equal(loss_fn.to_probability(y_pred), expected)
    assert (expected[..., 1:] >= expected[..., :-1]).all()

    expected = expected[..., loss_fn.prediction_index]
    assert torch.equal(loss_fn.to_prediction(y_pred), expected)

    regression_pred = y_pred[:, 0, :]
    assert regression_loss_fn.to_prediction(regression_pred).shape == (1, 1)
    assert regression_loss_fn.to_probability(regression_pred).shape == (
        1, 1, len(regression_loss_fn.quantiles))


def test_multi_quantile_loss_resolver_default_quantiles() -> None:
    loss = loss_resolver(
        TaskType.REGRESSION,
        'multi_quantile_loss',
    )

    assert isinstance(loss, MultiQuantileLoss)
    assert loss.quantiles == list(STANDARD_QUANTILE_LEVELS)


@pytest.mark.parametrize('dtype', [int, float, str])
def test_ordinal_log_loss(dtype: type) -> None:
    alpha = 1.7
    WHATEVER = 1
    loss_fn = OrdinalLogLoss(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        target_encoder=CategoricalTargetEncoder(
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            tensor_mapper=CategoricalTensorMapper(category_counts=[
                [dtype(1.1), WHATEVER],
                [dtype(2.2), WHATEVER],
                [dtype(3.3), WHATEVER],
            ]),
        ),
        alpha=alpha,
    )

    # mimic logits by scaling with `.log()` while keeping test readability:
    y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]).log()
    y_true = torch.tensor([[0], [2]])
    out = loss_fn(y_pred, y_true)

    if dtype == str:
        # OrdinalLogLoss should fall back to cross entropy loss if the target
        # values cannot be interpreted as ordinal numbers.
        assert loss_fn.target_values is None
        assert torch.allclose(out, F.cross_entropy(y_pred, y_true.view(-1)))
    else:
        expected = torch.tensor([
            # 1st sample:
            [
                -math.log(1 - 0.1) * abs(dtype(1.1) - dtype(1.1))**alpha,
                -math.log(1 - 0.2) * abs(dtype(2.2) - dtype(1.1))**alpha,
                -math.log(1 - 0.7) * abs(dtype(3.3) - dtype(1.1))**alpha,
            ],
            # 2nd sample:
            [
                -math.log(1 - 0.4) * abs(dtype(1.1) - dtype(3.3))**alpha,
                -math.log(1 - 0.4) * abs(dtype(2.2) - dtype(3.3))**alpha,
                -math.log(1 - 0.2) * abs(dtype(3.3) - dtype(3.3))**alpha,
            ],
        ]).mean()
        assert torch.allclose(out, expected)
