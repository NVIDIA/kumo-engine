# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from io_utils import kumo_dataclass


@kumo_dataclass
class BinaryConfusionMatrix:
    """Confusion matrix at various `thresholds."""
    true_positives: List[int]
    false_positives: List[int]
    true_negatives: List[int]
    false_negatives: List[int]
    thresholds: List[float]  # decision threshold


def get_confusion_matrix(
        y_true: pd.Series, y_prob: pd.Series,
        thresholds: List[int] = None) -> BinaryConfusionMatrix:
    r"""Get confusion matrix across various `thresholds` for a binary
    classification task.

    Args:
        y_true(pd.Series): Binary target value.
        y_prob(pd.Series): Probability of positve class.
        thresholds (optional, List): The list of threholds to compute
            conufsion matrix. (default: :obj:`None`)
    """
    if thresholds is None:
        thresholds = [0.5]
    thresholds = np.array(thresholds)

    confusions = []
    y_true = np.where(y_true, 1, 0)
    for threshold in thresholds:
        y_pred = np.where(y_prob > threshold, 1, 0)
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
        confusions.append(confusion)

    confusions = np.stack(confusions)
    return BinaryConfusionMatrix(true_positives=confusions[:, 1, 1].tolist(),
                                 false_positives=confusions[:, 0, 1].tolist(),
                                 true_negatives=confusions[:, 0, 0].tolist(),
                                 false_negatives=confusions[:, 1, 0].tolist(),
                                 thresholds=thresholds.tolist())
