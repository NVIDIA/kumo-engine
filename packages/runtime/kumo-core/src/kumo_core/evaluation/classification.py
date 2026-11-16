# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
import torchmetrics
from torch import Tensor
from torchmetrics.functional.classification.auroc import \
    _multilabel_auroc_compute  # isort: skip
from torchmetrics.functional.classification.average_precision import \
    _multilabel_average_precision_compute  # isort: skip
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_compute,
)
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat


class BinaryAUROC(torchmetrics.classification.BinaryAUROC):
    def compute(self) -> Tensor:
        target = dim_zero_cat(self.target)
        if (target[0] == target).all():
            # If all labels are the same, return NaN
            return torch.tensor([torch.nan], device=self.device)
        return super().compute()


class BinaryAveragePrecision(torchmetrics.classification.BinaryAveragePrecision
                             ):
    def compute(self) -> Tensor:
        target = dim_zero_cat(self.target)
        if (target[0] == target).all():
            # If all labels are the same, return NaN
            return torch.tensor([torch.nan], device=self.device)
        return super().compute()


class BinaryF1Score(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('target', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        precision, recall, _ = _binary_precision_recall_curve_compute(
            state=(dim_zero_cat(self.preds), dim_zero_cat(self.target)),
            thresholds=None,
        )

        f1_scores = (2 * (precision * recall) /
                     (precision + recall).clamp(min=1e-10))

        return f1_scores.max()


class MultilabelAveragePrecision(
        torchmetrics.classification.MultilabelAveragePrecision):
    def compute(self) -> Tensor:
        state = dim_zero_cat(self.preds), dim_zero_cat(self.target)
        if (average := self.average) != "micro":
            average = "none"
        res = _multilabel_average_precision_compute(
            state,
            self.num_labels,
            average,
            self.thresholds,
            self.ignore_index,
        )
        assert isinstance(res, Tensor)
        if average == "none":
            target = state[1]
            # For labels whose values are the same across data points, we
            # fill-in NaN.
            allsame = torch.all(target[0] == target, dim=0)
            res[allsame] = torch.nan
            if self.average == "macro":
                return torch.nanmean(res)
            else:
                return res
        else:
            return res


class MultilabelAUROC(torchmetrics.classification.MultilabelAUROC):
    def compute(self) -> Tensor:
        state = dim_zero_cat(self.preds), dim_zero_cat(self.target)
        if (average := self.average) != "micro":
            average = "none"
        res = _multilabel_auroc_compute(
            state,
            self.num_labels,
            average,
            self.thresholds,
            self.ignore_index,
        )
        assert isinstance(res, Tensor)
        if average == "none":
            target = state[1]
            # For labels whose values are the same across data points, we
            # fill-in NaN.
            allsame = torch.all(target[0] == target, dim=0)
            res[allsame] = torch.nan
            if self.average == "macro":
                return torch.nanmean(res)
            else:
                return res
        else:
            return res


def _binary_auprc_from_precision_recall(
    precision: Tensor,
    recall: Tensor,
) -> Tensor:
    order = torch.argsort(recall, descending=True, stable=True)
    precision, recall = precision[order], recall[order]
    return auc(recall, precision)


class BinaryAUPRC(torchmetrics.classification.BinaryPrecisionRecallCurve):
    r"""A custom metric for Area Under the Precision Recall Curve (AUPRC) for
    binary classification. Follows the same interface as TorchMetrics AUROC.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def compute(self) -> Tensor:  # type: ignore
        target = dim_zero_cat(self.target)
        if (target[0] == target).all():
            # If all labels are the same, return NaN
            return torch.tensor([torch.nan], device=self.device)
        precision, recall, _ = super().compute()
        assert isinstance(recall, Tensor)
        return _binary_auprc_from_precision_recall(precision, recall)


class MultilabelAUPRC(
        torchmetrics.classification.MultilabelPrecisionRecallCurve):
    r"""A custom metric for Area Under the Precision Recall Curve (AUPRC) for
    multi-label classification. Follows the same interface as TorchMetrics
    AUROC.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_labels: int,
        average: Literal["micro", "macro", "none"] | None = "macro",
    ) -> None:
        super().__init__(num_labels=num_labels)
        self.average = average

    def compute(self) -> Tensor:  # type: ignore
        state = dim_zero_cat(self.preds), dim_zero_cat(self.target)
        precision: Tensor | list[Tensor]
        recall: Tensor | list[Tensor]
        if self.average == "micro":
            state = state[0].flatten(), state[1].flatten()
            precision, recall, _ = _binary_precision_recall_curve_compute(
                state, thresholds=None)
            return _binary_auprc_from_precision_recall(precision, recall)
        else:
            precision, recall, _ = _multilabel_precision_recall_curve_compute(
                state, self.num_labels, thresholds=None)
            if isinstance(precision, Tensor) and isinstance(recall, Tensor):
                result = _binary_auprc_from_precision_recall(precision, recall)
            else:
                res = []
                for p, r in zip(precision, recall):
                    res.append(_binary_auprc_from_precision_recall(p, r))
                result = torch.stack(res)
            assert isinstance(result, Tensor)
            target = state[1]
            # For labels whose values are the same across data points, we
            # fill-in NaN.
            allsame = torch.all(target[0] == target, dim=0)
            result[allsame] = torch.nan
        if self.average is None or self.average == "none":
            return result
        elif self.average == "macro":
            return torch.nanmean(result)
        else:
            raise ValueError(f"Unsupported average called {self.average}.")
