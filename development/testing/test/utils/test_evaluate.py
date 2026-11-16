# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from kumo.utils.evaluate import get_confusion_matrix


def test_get_confusion():
    target = pd.Series([1, 0, 0, 1])
    prob = pd.Series([0.9, 0.1, 0.1, 0.9])
    matrix = get_confusion_matrix(target, prob, thresholds=[0, 0.5, 0.6, 1])
    tp = matrix.true_positives
    fp = matrix.false_positives
    tn = matrix.true_negatives
    fn = matrix.false_negatives

    assert all([i + j + k + l == 4 for i, j, k, l in zip(tp, fp, tn, fn)])
    assert tp[0] == tp[1] == tn[1] == tp[2] == tn[2] == tn[3] == 2
    assert fp[0] == fn[3] == 2
