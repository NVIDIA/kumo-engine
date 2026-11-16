# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from kumo_core.types import TaskType
from kumoapi.model_plan import (
    FocalLossConfig,
    HuberLossConfig,
    OrdinalLogLossConfig,
    QuantileLossConfig,
)
from kumo_core.evaluation.distribution import STANDARD_QUANTILE_LEVELS, DistributionLossMetric, LogNormalDistribution, NegativeBinomialDistribution, NormalDistribution
from kumo_core.encoders.target.base import TargetEncoder
from torch import Tensor

logger = logging.getLogger(__name__)


class Loss(torch.nn.Module, ABC):
    r"""Base class for loss computation.

    Args:
        task_type (TaskType or str): The type of the task.
    """
    supported_tasks: list[TaskType]

    def __init__(self, task_type: TaskType, **kwargs):
        super().__init__()

        if task_type not in self.supported_tasks:
            raise ValueError(f"The task type '{task_type}' is not supported "
                             f"by '{self.__class__.__name__}'")

        self.task_type = task_type

    @abstractmethod
    def to_prediction(self, y_pred: Tensor) -> Tensor:
        pass

    @abstractmethod
    def to_probability(
        self,
        y_pred: Tensor | dict[str, Tensor],
    ) -> Tensor | None:
        pass

    def forward(self, y_pred: Tensor, y_true: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class CrossEntropyLoss(Loss):
    supported_tasks: list[TaskType] = [
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_RANKING,
    ]

    def to_prediction(self, y_pred: Tensor) -> Tensor:
        return y_pred.argmax(dim=-1)

    def to_probability(self, y_pred: Tensor) -> Tensor:
        return y_pred.softmax(dim=-1)

    def forward(self, y_pred: Tensor, y_true: Tensor, **kwargs) -> Tensor:
        if y_true.size(0) == y_true.numel():
            y_true = y_true.view(-1)
        return F.cross_entropy(y_pred, y_true)


class OrdinalLogLoss(CrossEntropyLoss):
    def __init__(
        self,
        task_type: TaskType,
        target_encoder: TargetEncoder,
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(task_type)
        self.alpha = alpha
        self.target_encoder = target_encoder
        np_target_values = self.target_encoder.tensor_mapper.mapping.to_numpy()

        self.target_values: Tensor | None
        try:
            target_values = torch.from_numpy(np_target_values).to(torch.float32)
        except TypeError:
            self.target_values = None
            logger.warning(
                f"OrdinalLogLoss is being used, but since the user-provided "
                f"categorical values, e.g., {np_target_values[:5].tolist()}, "
                f"cannot be interpreted as ordinal numbers, falling back to "
                f"cross entropy loss.", exc_info=True)
        else:
            self.register_buffer('target_values', target_values)

    def forward(self, y_pred: Tensor, y_true: Tensor, **kwargs) -> Tensor:
        if self.target_values is None:
            return super().forward(y_pred, y_true)

        distance = (self.target_values - self.target_values[y_true]).abs()

        # NOTE: This is equivalent to the following but with better numerical
        # stability:
        # >>> loss = torch.log(1-y_pred.softmax(dim=-1))
        lse = torch.logsumexp(y_pred, dim=-1, keepdim=True)
        loss = torch.log1p(-torch.exp(y_pred - lse))
        return -(loss * (distance**self.alpha)).mean()


class BinaryClassificationLoss(Loss):
    supported_tasks: list[TaskType] = [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
    ]

    def __init__(self, task_type: TaskType, threshold: float = 0.5, **kwargs):
        super().__init__(task_type, **kwargs)
        self.threshold = threshold

    def to_prediction(self, y_pred: Tensor) -> Tensor:
        pred = (y_pred.sigmoid() > self.threshold)
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            # TODO Multi-label classification currently requires integer output
            # due to weird downstream assumptions in `format_prediction` :(
            pred = pred.to(torch.long)
        else:  # Reduce to vector to align with other non-multilabel outputs:
            pred = pred.view(-1)

        return pred

    def to_probability(self, y_pred: Tensor) -> Tensor:
        prob = y_pred.sigmoid()
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            prob = torch.cat([1 - prob, prob], dim=-1)
        return prob


class BinaryCrossEntropyLoss(BinaryClassificationLoss):
    def __init__(self, task_type: TaskType, threshold: float = 0.5, **kwargs):
        super().__init__(task_type, threshold, **kwargs)

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            y_pred,
            y_true.to(y_pred.dtype),
            weight=weight,
        )


class FocalLoss(BinaryClassificationLoss):
    def __init__(self, task_type: TaskType, alpha: float = 0.25,
                 gamma: float = 2.0, threshold: float = 0.5, **kwargs):
        super().__init__(task_type, threshold, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        **kwargs,
    ) -> Tensor:

        y_true = y_true.to(y_pred.dtype)

        prob = y_pred.sigmoid()
        prob_t = prob * y_true + (1 - prob) * (1 - y_true)

        ce_loss = F.binary_cross_entropy_with_logits(
            y_pred,
            y_true,
            reduction='none',
        )

        loss = ce_loss * ((1 - prob_t)**self.gamma)

        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = alpha_t * loss

        return loss.mean()


class RegressionLoss(Loss, ABC):
    supported_tasks: list[TaskType] = [
        TaskType.REGRESSION, TaskType.FORECASTING
    ]

    def to_prediction(self, y_pred: Tensor) -> Tensor:
        return y_pred

    def to_probability(self, y_pred: Tensor) -> None:
        return None


class MSELoss(RegressionLoss):
    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if weight is None:
            return F.mse_loss(y_pred, y_true)

        error = F.mse_loss(y_pred, y_true, reduction='none')
        return (error * weight).mean()


class MAELoss(RegressionLoss):
    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if weight is None:
            return F.l1_loss(y_pred, y_true)

        error = F.l1_loss(y_pred, y_true, reduction='none')
        return (error * weight).mean()


class WAPELoss(RegressionLoss):
    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        abs_error = (y_pred - y_true).abs()
        abs_true = y_true.abs()
        if weight is not None:
            abs_error = abs_error * weight
            abs_true = abs_true * weight
        scale = abs_true.sum().clamp(min=1e-8)
        return abs_error.sum() / scale


class HuberLoss(RegressionLoss):
    def __init__(self, task_type: TaskType, delta: float = 1.0, **kwargs):
        super().__init__(task_type, **kwargs)
        self.delta = delta

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if weight is None:
            return F.huber_loss(y_pred, y_true, delta=self.delta)

        error = F.huber_loss(y_pred, y_true, reduction='none', delta=self.delta)
        return (error * weight).mean()


def _pinball_loss(
    y_pred: Tensor,
    y_true: Tensor,
    q: float | list[float],
    weight: Tensor | None = None,
) -> Tensor:
    qt = torch.as_tensor(q, device=y_pred.device, dtype=y_pred.dtype)
    if qt.dim() > 0:
        y_true = y_true.unsqueeze(-1)
    error = torch.where(
        y_pred <= y_true,
        qt * (y_true - y_pred),
        (1.0 - qt) * (y_pred - y_true),
    )
    if weight is not None:
        if qt.dim() > 0:
            weight = weight.unsqueeze(-1)
        error = error * weight
    return error.mean()


class QuantileLoss(RegressionLoss):
    def __init__(self, task_type: TaskType, q: float, **kwargs):
        super().__init__(task_type, **kwargs)
        self.q = q

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        return _pinball_loss(y_pred, y_true, self.q, weight)


class MultiQuantileLoss(RegressionLoss):
    def __init__(self, task_type: TaskType, **kwargs):
        super().__init__(task_type, **kwargs)
        self.quantiles = list(STANDARD_QUANTILE_LEVELS)
        self.prediction_index = self.quantiles.index(0.5)

    @staticmethod
    def _sort_quantiles(y_pred: Tensor) -> Tensor:
        return y_pred.sort(dim=-1).values

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        return _pinball_loss(y_pred, y_true, self.quantiles, weight)

    def to_prediction(self, y_pred: Tensor) -> Tensor:
        y_pred = self._sort_quantiles(y_pred)
        pred = y_pred[..., self.prediction_index]
        if self.task_type == TaskType.REGRESSION:
            return pred.unsqueeze(-1)
        assert self.task_type == TaskType.FORECASTING
        return pred

    def to_probability(self, y_pred: Tensor) -> Tensor:
        y_pred = self._sort_quantiles(y_pred)
        if self.task_type == TaskType.REGRESSION:
            return y_pred.unsqueeze(1)
        assert self.task_type == TaskType.FORECASTING
        return y_pred


class DistributionLoss(RegressionLoss):
    def __init__(self, task_type: TaskType,
                 distribution: DistributionLossMetric, **kwargs):
        super().__init__(task_type, **kwargs)
        self.distribution = distribution

    def forward(self, params: dict[str, Tensor], y_true: Tensor,
                **kwargs) -> Tensor:
        return self.distribution.loss(params=params, y_actual=y_true).mean()

    def to_prediction(self, params: dict[str, Tensor]) -> Tensor:
        if isinstance(params, dict):
            return self.distribution.to_prediction(params=params)
        else:
            return super().to_prediction(params)

    def sample(self, params: dict[str, Tensor]) -> Tensor:
        return self.distribution.sample(params=params, n_samples=1)

    def to_probability(self, params: dict[str, Tensor]) -> Tensor:
        if isinstance(params, dict):
            return self.distribution.to_quantiles(params)
        else:
            return super().to_probability(params)  # type: ignore


# Loss resolver ###############################################################


def loss_resolver(
    task_type: TaskType,
    loss: str | FocalLossConfig | HuberLossConfig | QuantileLossConfig
    | OrdinalLogLossConfig | None = None,
    target_encoder: TargetEncoder | None = None,
    **kwargs,
) -> Loss:

    if loss is not None:
        if loss == 'binary_cross_entropy_loss':
            return BinaryCrossEntropyLoss(task_type, **kwargs)
        if loss == 'focal_loss':
            return FocalLoss(task_type, **kwargs)
        if isinstance(loss, FocalLossConfig):
            return FocalLoss(
                task_type,
                alpha=loss.alpha,  # type: ignore
                gamma=loss.gamma,  # type: ignore
                **kwargs,
            )
        if loss == 'cross_entropy_loss':
            return CrossEntropyLoss(task_type, **kwargs)
        if loss == 'ordinal_log_loss':
            return OrdinalLogLoss(task_type, target_encoder, **kwargs)
        if isinstance(loss, OrdinalLogLossConfig):
            return OrdinalLogLoss(
                task_type,
                target_encoder,
                alpha=loss.alpha,
                **kwargs,
            )
        if loss == 'huber_loss':
            return HuberLoss(task_type, **kwargs)
        if isinstance(loss, HuberLossConfig):
            return HuberLoss(
                task_type,
                delta=loss.delta,  # type: ignore
                **kwargs,
            )
        if loss == 'mse_loss':
            return MSELoss(task_type, **kwargs)
        if loss == 'mae_loss':
            return MAELoss(task_type, **kwargs)
        if loss == 'wape_loss':
            return WAPELoss(task_type, **kwargs)
        if isinstance(loss, QuantileLossConfig):
            return QuantileLoss(
                task_type,
                q=loss.q,
                **kwargs,
            )
        if loss == 'multi_quantile_loss':
            return MultiQuantileLoss(
                task_type,
                **kwargs,
            )
        if loss == 'normal_distribution_loss':
            return DistributionLoss(
                task_type,
                distribution=NormalDistribution(),
                **kwargs,
            )
        if loss == 'negative_binomial_distribution_loss':
            return DistributionLoss(
                task_type,
                distribution=NegativeBinomialDistribution(),
                **kwargs,
            )
        if loss == 'log_normal_distribution_loss':
            return DistributionLoss(
                task_type,
                distribution=LogNormalDistribution(),
                **kwargs,
            )

        raise ValueError(f"Invalid custom loss function '{loss}'")

    if task_type in [TaskType.MULTICLASS_CLASSIFICATION]:
        return CrossEntropyLoss(task_type, **kwargs)

    elif task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
    ]:
        return BinaryCrossEntropyLoss(task_type, **kwargs)

    elif task_type in [TaskType.MULTILABEL_RANKING]:
        return CrossEntropyLoss(task_type, **kwargs)

    elif task_type in [TaskType.REGRESSION, TaskType.FORECASTING]:
        return HuberLoss(task_type, **kwargs)

    raise ValueError(f"Could not assign loss function for task '{task_type}'")
