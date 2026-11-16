# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from kumoapi.task import TaskType
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.utils import scatter

from kumo_core.io.filesystem import open_file


class Calibration(torch.nn.Module, ABC):
    r"""Base class for performing calibration of model predictions.

    Args:
        task_type: The type of the task.
    """
    MAX_EXAMPLES = 100_000

    def __init__(self, task_type: TaskType | str):
        super().__init__()
        self.task_type = TaskType(task_type)
        self._is_fitted: bool = False

    def get_extra_state(self) -> dict[str, Any]:
        return dict(is_fitted=self._is_fitted)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._is_fitted = state['is_fitted']

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        r"""Whether the calibrator is already fitted."""
        return self._is_fitted

    def fit(self, logits: Tensor, y: Tensor) -> None:
        r"""Trains the calibrator.

        Args:
            logits: The logits to calibrate.
            y: The ground-truth labels.
        """
        if not self.is_supported:
            raise RuntimeError(f"'{self.__class__.__name__}' does not support "
                               f"the given task (got '{self.task_type}')")
        if self._is_fitted:
            raise RuntimeError(f"'{self.__class__.__name__}' is already "
                               f"fitted. Call `reset_parameters()` first.")

        perm = torch.randperm(y.size(0), device=y.device)[:self.MAX_EXAMPLES]
        self._fit(logits.detach()[perm], y[perm])

        self._is_fitted = True

    @torch.no_grad()
    def forward(self, logits: Tensor) -> Tensor:
        r"""Calibrate the logits.

        Args:
            logits: The logits to calibrate.
        """
        if not self.is_supported:
            raise RuntimeError(f"'{self.__class__.__name__}' does not support "
                               f"the given task (got '{self.task_type}')")
        if not self._is_fitted:
            raise RuntimeError(f"'{self.__class__.__name__}' is not yet "
                               f"fitted. Please run `fit()` first.")

        return self._forward(logits)

    @property
    @abstractmethod
    def is_supported(self) -> bool:
        r"""Whether the calibrator supports the given task type."""

    @torch.no_grad()
    def reliability_diagram(
        self,
        prob: Tensor,
        y: Tensor,
        path: str,
        num_bins: int = 10,
    ) -> None:
        r"""Plots the reliability diagram, *i.e.* the fraction of examples
        whose predicted probability falls into a given bin. It gives a measure
        of how well calibrated a given model is.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if self.task_type not in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            raise NotImplementedError("Reliability plots are only supported "
                                      "for binary classification tasks")

        prob = prob.view(-1)
        y = y.to(torch.float).view(-1)

        boundaries = torch.linspace(start=0.0, end=1.0, steps=num_bins + 1,
                                    device=prob.device)[1:-1]
        bucket = torch.bucketize(prob, boundaries)

        avg_prob = scatter(prob, bucket, dim_size=num_bins, reduce='mean')
        rel_freq = scatter(y, bucket, dim_size=num_bins, reduce='mean')
        mask = avg_prob != 0  # Filter out buckets without any prediction.

        avg_prob = avg_prob[mask].cpu().numpy()
        rel_freq = rel_freq[mask].cpu().numpy()

        p1 = plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        p2 = plt.plot(avg_prob, rel_freq, marker='s', color='darkorange')

        plt.title('Reliability Diagram')
        plt.xlabel('Average Predicted Probability')
        plt.ylabel('Relative Frequency of Positive Examples')
        plt.legend((p1[0], p2[0]), ('Perfectly calibrated', 'Model'))
        plt.tight_layout()

        with open_file(path, 'wb') as f:
            plt.savefig(f, format='png')

        plt.close()

    @abstractmethod
    def _fit(self, logits: Tensor, y: Tensor) -> None:
        pass

    @abstractmethod
    def _forward(self, logits: Tensor) -> Tensor:
        pass

    def _loss(self, logits: Tensor, y: Tensor) -> Tensor:
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, y)

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return F.cross_entropy(logits, y.view(-1))

        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            y = y.to(logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, y)

        raise NotImplementedError

    def _get_optimizer(self) -> Optimizer:
        return torch.optim.LBFGS(
            self.parameters(),
            lr=0.001,
            max_iter=10_000,
            line_search_fn='strong_wolfe',
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class TemperatureScaling(Calibration):
    r"""Temperature scaling scales logits via a single parameter :math:`T > 0`.

    Output logits are given by:

    .. math::

        \mathbf{z}^{\prime} = \mathbf{z} / T.
    """
    def __init__(self, task_type: TaskType | str):
        super().__init__(task_type)

        self.temperature = torch.nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.ones_(self.temperature)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Loads this module's state from a state dictionary. This requires
        customization as the shape of ``temperature`` changes during
        evaluation, i.e., we initialize ``self.temperature`` with shape ``[1]``
        but load a ``state_dict`` with shape ``[num_classes]``.
        """
        key = prefix + 'temperature'
        if key in state_dict:
            self.temperature = torch.nn.Parameter(state_dict[key])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def is_supported(self) -> bool:
        return self.task_type in TaskType.get_classification_tasks()

    def _fit(self, logits: Tensor, y: Tensor) -> None:
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            self.temperature = torch.nn.Parameter(logits.new_empty(y.size(1)))
            self.reset_parameters()

        optimizer = self._get_optimizer()

        def _step_fn() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss(self._forward(logits), y)
            loss.backward()
            return loss

        optimizer.step(_step_fn)  # type: ignore

    def _forward(self, logits: Tensor) -> Tensor:
        # TODO(dong) optimize the device transfer if it becomes a bottleneck.
        temperature = self.temperature.to(logits.device)
        return logits / temperature.clamp(min=1e-7)


class PlatScaling(Calibration):
    r"""A logistic regression model to transform probabilities."""
    def __init__(self, task_type: TaskType | str):
        super().__init__(task_type)

        self.weight_and_bias = torch.nn.Parameter(torch.empty(2, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.ones_(self.weight_and_bias[0])
        torch.nn.init.zeros_(self.weight_and_bias[1])

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Loads this module's state from a state dictionary. This requires
        customization as the shape of ``weight_and_bias`` changes during
        evaluation, i.e., we initialize ``self.weight_and_bias`` with shape
        ``[2]`` but load a ``state_dict`` with shape ``[2 * num_classes]``.
        """
        key = prefix + 'weight_and_bias'
        if key in state_dict:
            self.weight_and_bias = torch.nn.Parameter(state_dict[key])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def is_supported(self) -> bool:
        return self.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }

    def _fit(self, logits: Tensor, y: Tensor) -> None:
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            self.weight_and_bias = torch.nn.Parameter(
                logits.new_empty(2, y.size(1)))
            self.reset_parameters()

        optimizer = self._get_optimizer()

        def _step_fn() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss(self._forward(logits), y)
            loss.backward()
            return loss

        optimizer.step(_step_fn)  # type: ignore

    def _forward(self, logits: Tensor) -> Tensor:
        # TODO(dong) optimize the device transfer if it becomes a bottleneck.
        weight, bias = self.weight_and_bias.to(logits.device)
        return weight.clamp(min=1e-3) * logits + bias
