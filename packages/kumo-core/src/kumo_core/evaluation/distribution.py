# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-FileCopyrightText: Copyright 2020 Jan Beitner
# SPDX-License-Identifier: Apache-2.0 AND MIT
import inspect
import warnings
from abc import abstractmethod
from typing import Any

import torch
import torchmetrics
from torch import Tensor
from torch import distributions as dist

STANDARD_QUANTILE_LEVELS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.025, 0.05,
                                               0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                                               0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                                               0.975, 0.98, 0.99, 0.995)


# These metric classes are adapted from Pytorch forecasting
# Available under MIT license: https://github.com/jdb78/pytorch-forecasting
class DistributionLossMetric(torchmetrics.Metric):
    r"""Base metric class that has basic functions that can handle predicting
    quantiles and operate in log space. See the `torchmetrics documentation
    <https://lightning.ai/docs/torchmetrics/stable/>`_
    for details of how to implement a new metric.

    Class should be inherited for all distribution losses, i.e. if a network
    predicts the parameters of a probability distribution,
    DistributionLossMetric can be used to score those parameters and
    calculate loss for given true values.

    Define two class attributes in a child class:

    Attributes:
        distribution_arguments: list of parameter names for the distribution

    Further, implement the method :meth:`param_to_distribution`.

    Args:
        quantiles: quantiles for probability range. Defaults to
            ``STANDARD_QUANTILE_LEVELS``.
        reduction: Reduction, ``"none"``, ``"mean"`` or ``"sqrt-mean"``.
    """
    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(
        self,
        quantiles: list[float] | None = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.quantiles = quantiles
        if self.quantiles is None:
            self.quantiles = list(STANDARD_QUANTILE_LEVELS)
        self.reduction = reduction
        self.add_state("losses", default=torch.tensor(0.0),
                       dist_reduce_fx="sum" if reduction != "none" else "cat")
        self.add_state("lengths", default=torch.tensor(0),
                       dist_reduce_fx="sum" if reduction != "none" else "mean")

    @property
    @abstractmethod
    def distribution_parameters(self) -> list[str]:
        r"""Defines the distribution parameter names, has to be defined for
        each inheriting distribution class.

        Returns:
            list[str]: returns a list of all distribution parameter keys.
        """

    @abstractmethod
    def param_to_distribution(
        self,
        params: dict[str, Tensor],
    ) -> dist.Distribution:
        r"""Map the a tensor of parameters to a probability distribution.

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``.

        Returns:
            distributions.Distribution: torch probability distribution
        """

    def loss(
        self,
        params: dict[str, Tensor],
        y_actual: Tensor,
    ) -> Tensor:
        r"""Calculate negative likelihood.

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``.
            y_actual: target values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.param_to_distribution(params)
        loss = -distribution.log_prob(y_actual)
        return loss

    def to_prediction(
        self,
        params: dict[str, Tensor],
        n_samples: int = 100,
    ) -> Tensor:
        r"""Convert network prediction into a point prediction.

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``.
            n_samples: number of samples

        Returns:
            torch.Tensor: mean prediction
        """
        distribution = self.param_to_distribution(params)
        try:
            return distribution.mean
        except NotImplementedError:
            return self.sample(params, n_samples=n_samples).mean(-1)

    def sample(
        self,
        params: dict[str, Tensor],
        n_samples: int,
    ) -> Tensor:
        r"""Sample from distribution.

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``.
            n_samples: number of samples to draw

        Returns:
            torch.Tensor: tensor with samples of size
                ``[batch_size, n_samples]``.
        """
        dist = self.param_to_distribution(params)
        samples = dist.sample(torch.Size((n_samples, )))
        samples = samples.transpose(0, 1)
        return samples

    def to_quantiles(
        self,
        params: dict[str, Tensor],
        quantiles: list[float] | None = None,
        n_samples: int = 100,
    ) -> Tensor:
        r"""Convert network prediction into a quantile prediction.

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``.
            quantiles: quantiles for probability range. Defaults to quantiles
                as defined in the class initialization.
            n_samples: number of samples to draw for quantiles.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """
        if quantiles is None:
            quantiles = self.quantiles
        try:
            distribution = self.param_to_distribution(params)
            device = [*params.values()][0].device
            out = distribution.icdf(
                torch.tensor(quantiles, device=device)[:, None,
                                                       None]).permute(1, 2, 0)
        except NotImplementedError:  # resort to derive quantiles empirically
            samples = torch.sort(self.sample(params, n_samples), -1).values
            out = torch.quantile(
                samples, torch.tensor(quantiles, device=samples.device),
                dim=1).permute(1, 2, 0)
        return out

    def update(
        self,
        params: dict[str, Tensor],
        target: Tensor,
    ) -> None:
        r"""Update method of metric that handles masking of values.

        Do not override this method but :meth:`loss` instead

        Args:
            params: prediction output of network where each tensor is of size
                ``[batch_size, num_params]``
            target: actual values. Can be a tuple of tensors with
                ``(target, weight)``. If used, the losses are weighted before
                update. Weights can be of shape ``[target.size(0)]`` or of
                shape ``[target.size(0), target.size(1)]``.
        """
        # unpack weight
        if isinstance(target, list | tuple):
            target, weight = target
        else:
            weight = None

        # unpack target
        lengths = torch.full((target.size(0), ), fill_value=target.size(1),
                             dtype=torch.long, device=target.device)

        losses = self.loss(params, target)
        # weight samples
        if weight is not None:
            if weight.dim == 1:
                weight = weight.unsqueeze(-1)
            losses = losses * weight
        self._update_losses_and_lengths(losses, lengths)

    def _update_losses_and_lengths(
        self,
        losses: Tensor,
        lengths: Tensor,
    ) -> None:
        self.losses: Tensor
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9",
                              stacklevel=2)
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def compute(self) -> Tensor:
        r"""Compute the metric.

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        loss = self.reduce_loss(self.losses, lengths=self.lengths)
        return loss

    def reduce_loss(
        self,
        losses: Tensor,
        lengths: Tensor,
        reduction: str | None = None,
    ) -> Tensor:
        r"""Reduce loss.

        Args:
            losses: total loss. first dimenion are samples, second timesteps
            lengths: total length
            reduction: type of reduction. Defaults to :attr:`reduction`.

        Returns:
            torch.Tensor: reduced loss
        """
        if reduction is None:
            reduction = self.reduction
        if reduction == "none":
            return losses  # return immediately, no checks
        elif reduction == "mean":
            loss = losses.sum() / lengths.sum()
        elif reduction == "sqrt-mean":
            loss = losses.sum() / lengths.sum()
            loss = loss.sqrt()
        else:
            raise ValueError(f"reduction {reduction} unknown")
        assert not torch.isnan(loss), (
            "Loss should not be nan - i.e. something went wrong "
            "in calculating the loss (e.g. log of a negative number)")
        assert torch.isfinite(
            loss
        ), "Loss should not be infinite - i.e. something went wrong (e.g.\
            input is not in log space)"

        return loss

    def extra_repr(self) -> str:
        forbidden_attributes = ["name", "reduction"]
        attributes = list(inspect.signature(self.__class__).parameters.keys())
        return ", ".join([
            f"{name}={repr(getattr(self, name))}" for name in attributes
            if hasattr(self, name) and name not in forbidden_attributes
        ])


class NormalDistribution(DistributionLossMetric):
    r"""Normal distribution loss.

    Distribution Parameters: ``'loc'``, ``'scale'``
    """
    @property
    def distribution_parameters(self) -> list[str]:
        return ['loc', 'scale']

    def param_to_distribution(
        self,
        params: dict[str, Tensor],
    ) -> dist.Normal:
        for param in self.distribution_parameters:
            assert param in params
        distr = dist.Normal(
            loc=params['loc'],
            # Predicting log variance - see VAE
            scale=(0.5 * params['scale']).exp())
        return distr

    def to_prediction(
        self,
        params: dict[str, Tensor],
        n_samples: int = 100,
    ) -> Tensor:
        return params['loc']


class NegativeBinomialDistribution(DistributionLossMetric):
    r"""Negative binomial loss, e.g. for count data.

    Distribution Parameters: ``'mean'``, ``'shape'``

    Requirements for original target normalizer:
        * not centered normalization (only rescaled)
    """
    @property
    def distribution_parameters(self) -> list[str]:
        return ['mean', 'shape']

    def param_to_distribution(
        self,
        params: dict[str, Tensor],
    ) -> dist.NegativeBinomial:
        for param in self.distribution_parameters:
            assert param in params
        mean = params['mean'].exp()
        shape = params['shape'].exp()
        r = 1.0 / shape
        p = mean / (mean + r)
        return dist.NegativeBinomial(total_count=r, probs=p)

    def loss(
        self,
        params: dict[str, Tensor],
        y_actual: Tensor,
    ) -> Tensor:
        r"""Calculate negative likelihood.

        Args:
            params: prediction output of network. Value tensors in the dict are
                of size ``[batch_size x num_params]``.
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.param_to_distribution(params)
        # clip y_actual to avoid infinite losses
        loss = -distribution.log_prob(torch.round(y_actual.relu()).int())
        return loss

    def to_prediction(
        self,
        params: dict[str, Tensor],
        n_samples: int = 100,
    ) -> Tensor:
        return params['mean'].exp()


class LogNormalDistribution(DistributionLossMetric):
    r"""Log-normal loss.

    Distribution Parameters: ``'loc'``, ``'scale'``

    Requirements for original target normalizer:
        * normalized target in log space
    """

    eps = 1e-6

    @property
    def distribution_parameters(self) -> list[str]:
        return ['loc', 'scale']

    def param_to_distribution(
        self,
        params: dict[str, Tensor],
    ) -> dist.LogNormal:
        for param in self.distribution_parameters:
            assert param in params
        return dist.LogNormal(loc=params['loc'], scale=params['scale'].exp())

    def loss(
        self,
        params: dict[str, Tensor],
        y_actual: Tensor,
    ) -> Tensor:
        r"""Calculate negative likelihood.

        Args:
            params: prediction output of network. Value tensors in the dict are
                of size ``[batch_size x num_params]``.
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.param_to_distribution(params)
        # clip y_actual to avoid infinite losses
        loss = -distribution.log_prob(y_actual.clip(self.eps, None))
        return loss

    def to_prediction(
        self,
        params: dict[str, Tensor],
        n_samples: int = 100,
    ) -> Tensor:
        return torch.exp(params['loc'] + ((params['scale'].exp())**2 / 2))
