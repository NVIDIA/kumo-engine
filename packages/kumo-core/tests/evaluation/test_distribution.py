# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.distributions as dist

from kumo_core.evaluation.distribution import STANDARD_QUANTILE_LEVELS, LogNormalDistribution, NegativeBinomialDistribution, NormalDistribution


def test_normal_distribution_loss() -> None:
    std = 0.1
    n = 10000
    target = torch.ones([n, 1])
    scale = torch.ones_like(target) * std
    parameters = {'loc': target, 'scale': scale.log() * 2}
    metric = NormalDistribution()
    assert metric.quantiles == list(STANDARD_QUANTILE_LEVELS)
    assert metric.quantiles is not STANDARD_QUANTILE_LEVELS

    # Test sample
    samples = metric.sample(parameters, 1)
    assert torch.isclose(target.mean(), samples.mean(), atol=0.1, rtol=0.5)

    # Test loss
    loss_value = metric.loss(parameters, target)
    # This is the expected neg log likelihood loss for the mean prediction
    # with std=0.1
    assert torch.isclose(loss_value[0], torch.tensor(-1.38364660))

    # Test quantiles — default levels produce 27 sorted outputs
    quantiles = metric.to_quantiles(parameters)
    assert quantiles.shape[-1] == 27
    assert (quantiles[0, 0, 1:] >= quantiles[0, 0, :-1]).all()
    # Spot-check a fixed set of levels for precise values
    quantiles_spot = metric.to_quantiles(
        parameters, quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    quantiles_gt = torch.tensor(
        [0.7946, 0.8718, 0.9326, 1.0000, 1.0674, 1.1282, 1.2054])
    assert torch.allclose(quantiles_spot[0, 0, :], quantiles_gt, atol=1e-3)

    # Test metric functions
    metric.update(parameters, target)
    loss_value = metric.compute()  # type: ignore
    assert torch.isclose(loss_value, torch.tensor(-1.38364660))
    target = torch.ones([n, 1]) * 2
    metric.update(parameters, target)
    full = metric.compute()  # type: ignore
    assert torch.isclose(full, torch.tensor(23.6163), atol=1e-3)


def test_negative_binomial_distribution_loss() -> None:
    mean = 100.0
    shape = 1.0
    n = 10000
    target = dist.NegativeBinomial(total_count=1.0 / shape,
                                   probs=mean / (mean + (1.0 / shape))).sample(
                                       torch.Size([n, 1]))
    parameters = {
        'mean': target.log(),
        'shape': (1.0 * torch.ones_like(target)).log()
    }
    loss = NegativeBinomialDistribution()

    samples = loss.sample(parameters, 1)
    assert torch.isclose(target.mean(), samples.mean(), atol=0.1, rtol=0.5)
    assert torch.isclose(target.std(), samples.std(), atol=0.5, rtol=1.0)


def test_log_normal_distribution_loss() -> None:
    mean = 2.0
    std = 0.2
    n = 10000
    target = dist.LogNormal(loc=mean, scale=std).sample(torch.Size([n, 1]))
    target = torch.log(target)
    scale = torch.ones_like(target) * target.std()
    parameters = {'loc': target, 'scale': scale.log()}
    loss = LogNormalDistribution()
    samples = loss.sample(parameters, 1)
    assert torch.isclose(torch.as_tensor(mean),
                         samples.log().mean(), atol=0.1, rtol=0.2)
    assert torch.isclose(torch.as_tensor(std),
                         samples.log().std(), atol=0.1, rtol=0.7)
