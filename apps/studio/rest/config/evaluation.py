# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Dict, List, Optional, Union

import numpy as np
from kumo_core.train.output import TargetPredictionHeatMap
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from pydantic import model_validator

from io_utils import kumo_dataclass
from kumo.config.cohort import DataCube
from kumo.config.explain import ColumnContributionSummary
from kumo.utils.evaluate import BinaryConfusionMatrix

TRUE_CLASS = 'True'
FALSE_CLASS = 'False'
EPS = 0.000001

# Maximum number of cells allowed for regression heatmap
MAX_CELLS = 10_000

ColumnContributionResponse = ColumnContributionSummary


@kumo_dataclass
class ClassificationInputColumnStatsResponse:
    classes: Dict[str, List[Dict[str, Union[str, float]]]]


def _default_thresholds() -> List[float]:
    """Before variable-precision classification metrics were implemented, we
    only produced classification metrics for 101 thresholds from 0 to 1. This
    function generates a list of thresholds that is compatible with the legacy
    data (on s3) that was generated before this time.
    """
    result: List[float] = []
    for i in range(0, 101):
        result.append(i / 100)
    return result


def _safe_divide(numerator: Union[List[int], np.ndarray],
                 denominator: Union[int, List[int], np.ndarray],
                 fill_na: float = 1.0) -> List[float]:
    r"""Perform division using `np.divide` and fill any `nan` values
    with `fill_na`"""
    denominator = (denominator
                   if isinstance(denominator, int) else np.array(denominator))
    out = np.divide(np.array(numerator), denominator)
    return np.nan_to_num(out, nan=fill_na).tolist()


@kumo_dataclass
class BinaryClassificationMetrics:
    # Used to generate ROC curve, PR curve and cumulative gain chart
    precision: List[float]  # precision across different thresholds
    recall: List[float]
    accuracy: List[float]
    predicted_pos_rate: List[float]
    false_positive_rate: List[float]
    ppr_ideal: float
    f1: List[float]

    # Used for confusion matrix
    num_samples: int
    true_positive: List[int]
    true_negative: List[int]
    false_positive: List[int]
    false_negative: List[int]
    total_positive_actual: int  # Doesn't change
    total_negative_actual: int  # Doesn't change
    total_positive_pred: List[int]
    total_negative_pred: List[int]

    # The classification threshold that was used to generate
    # each of the above metrics. See #4224 for details
    threshold: List[float] = field(default_factory=_default_thresholds)

    @classmethod
    def from_binary_confusion(
            cls,
            confusion: BinaryConfusionMatrix) -> 'BinaryClassificationMetrics':

        total_negative_actual = confusion.true_negatives[
            0] + confusion.false_positives[0]
        total_positive_actual = confusion.true_positives[
            0] + confusion.false_negatives[0]
        num_samples = total_negative_actual + total_positive_actual
        # In case of zero division we default to 0.0 for recall
        # this corresponds to the case with TP=0 and FN=0
        recall = _safe_divide(confusion.true_positives, total_positive_actual,
                              fill_na=0.0)
        # In case of zero division we defaulto to 1.0 precision
        # this corresponds to the case with TP=0 and FP=0
        precision = _safe_divide(
            confusion.true_positives,
            np.array(confusion.true_positives) +
            np.array(confusion.false_positives), fill_na=1.0)
        accuracy = _safe_divide(
            np.array(confusion.true_positives) +
            np.array(confusion.true_negatives), num_samples)
        predicted_pos_rate = _safe_divide(
            np.array(confusion.true_positives) +
            np.array(confusion.false_positives), num_samples)
        false_positive_rate = _safe_divide(confusion.false_positives,
                                           total_negative_actual)
        total_positive_pred = (np.array(confusion.true_positives) +
                               np.array(confusion.false_positives)).tolist()
        total_negative_pred = (np.array(confusion.true_negatives) +
                               np.array(confusion.false_negatives)).tolist()
        ppr_ideal = total_positive_actual / num_samples

        f1 = _safe_divide(
            2 * np.array(confusion.true_positives),
            2 * np.array(confusion.true_positives) +
            np.array(confusion.false_positives) +
            np.array(confusion.false_negatives), fill_na=0.0)

        return cls(
            num_samples=num_samples,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            predicted_pos_rate=predicted_pos_rate,
            ppr_ideal=ppr_ideal,
            f1=f1,
            true_positive=confusion.true_positives,
            true_negative=confusion.true_negatives,
            false_positive=confusion.false_positives,
            false_negative=confusion.false_negatives,
            false_positive_rate=false_positive_rate,
            total_positive_actual=total_positive_actual,
            total_negative_actual=total_negative_actual,
            total_positive_pred=total_positive_pred,
            total_negative_pred=total_negative_pred,
        )


@kumo_dataclass
class ClassificationEvaluationResponse:
    r"""Retruns metrics assuming a class is positive"""
    classes: Dict[str, BinaryClassificationMetrics]

    @classmethod
    def from_binary_confusion_matrix(
            cls, confusion: BinaryConfusionMatrix
    ) -> 'ClassificationEvaluationResponse':
        # Get confusion matrix saved to db during training job.
        true_class = BinaryClassificationMetrics.from_binary_confusion(
            confusion)

        # Compute metrics with positive and
        # negative classes switched.
        confusion_switched = BinaryConfusionMatrix(
            true_positives=list(reversed(confusion.true_negatives)),
            true_negatives=list(reversed(confusion.true_positives)),
            false_negatives=list(reversed(confusion.false_positives)),
            false_positives=list(reversed(confusion.false_negatives)),
            thresholds=confusion.thresholds,
        )
        false_class = BinaryClassificationMetrics.from_binary_confusion(
            confusion_switched)

        response = cls({})
        response.classes[TRUE_CLASS] = true_class
        response.classes[FALSE_CLASS] = false_class
        return response


@kumo_dataclass
class TableColumn:
    table_id: str
    column_name: str


@kumo_dataclass
class ConfusionMatrixCell:
    count: int
    percentage: float


@kumo_dataclass
class ConfusionMatrixResponse:
    classes: List[str]
    sums_bottom: List[int]
    sums_right: List[int]
    values: List[List[ConfusionMatrixCell]]

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def check_dims(self) -> 'ConfusionMatrixResponse':
        classes, values = self.classes, self.values
        sums_bottom, sums_right = self.sums_bottom, self.sums_right

        lengths = [len(classes), len(sums_bottom), len(sums_right), len(values)]
        for value in values:
            lengths.append(len(value))

        if not len(set(lengths)) == 1:
            raise ValueError('Dimensions are mismatched, all elements '
                             'must have the same size')

        return self


@kumo_dataclass
class RegressionResiduals:
    buckets: List[List[int]]
    y_values: List[float]
    x_values: List[float]

    @staticmethod
    def heatmap_cell_width_resize(heatmap: TargetPredictionHeatMap) -> float:
        r"""Adjust the cell width of heatmap such that the number of cells
        in the `buckets` is mostly smaller than `MAX_CELLS` for more stable UI
        rendering.
        """
        x_len = heatmap.label_bin_end - heatmap.label_bin_start
        y_len = heatmap.prediction_bin_end - heatmap.prediction_bin_start

        # Get cell width that satisfies x_len / cell_width * y_len / cell_width
        # < MAX_CELLS
        # NOTE: because of the way we generate cells, the actual number of
        # cell in x-axis and y-axis might be 1 larger than our estimation
        # here, resulting in the total number of cells exceed MAX_CELLS
        # by a small margin, but this should not cause any problems
        cell_width_adjust = ((x_len * y_len) / MAX_CELLS)**0.5

        # if `heatmap.cell_width` already satisify the restriction, return
        # `heatmap.cell_width` instead
        cell_width_adjust = max(cell_width_adjust, heatmap.cell_width)
        return cell_width_adjust

    @classmethod
    def from_target_prediction_heatmap(
            cls, heatmap: TargetPredictionHeatMap) -> 'RegressionResiduals':
        """
        Converts a TargetPredictionHeatMap object to a
        RegressionResiduals object"""
        cell_width_adjust = RegressionResiduals.heatmap_cell_width_resize(
            heatmap)
        x_values = np.arange(heatmap.label_bin_start,
                             heatmap.label_bin_end + 1e-6,
                             cell_width_adjust).tolist()
        y_values = np.arange(heatmap.prediction_bin_start,
                             heatmap.prediction_bin_end + 1e-6,
                             cell_width_adjust).tolist()

        buckets = np.zeros((len(y_values), len(x_values)))
        for index, (cell_x, cell_y) in enumerate(heatmap.cells):
            # Find the indices of this bucket
            i = round((cell_x - x_values[0]) / cell_width_adjust)
            j = round((cell_y - y_values[0]) / cell_width_adjust)
            # Set the value to be the proper count
            buckets[j, i] = heatmap.counts[index]

        return cls(buckets=buckets.tolist(), x_values=x_values,
                   y_values=y_values)


@kumo_dataclass
class RegressionHeatmapResponse:
    """Contains information that can be used to render a predicted vs actuals
    heatmap in the UI
    """
    residuals_chart: Dict[TrainingStage, RegressionResiduals]


@kumo_dataclass
class ColumnExplanation:
    col_id: str
    stype: Optional[Stype] = None
    variance_score: Optional[float] = None
    gradient_score: Optional[float] = None
    cohort: Optional[DataCube] = None


@kumo_dataclass
class NavigationInfo:
    root_table_name: str
    navigation_tree: Dict[str, List[str]]


ColumnName = str


@kumo_dataclass
class XAIResponse:
    tables: Dict[str, Dict[ColumnName,
                           ColumnExplanation]] = field(default_factory=dict)
    navigation: Optional[NavigationInfo] = None
