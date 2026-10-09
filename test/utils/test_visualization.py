# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp

import numpy as np
import pandas as pd

from kumo.utils.visualization import (
    bar_plot,
    binary_confusion_matrix_plot,
    binary_pr_curve_plot,
    binary_roc_curve_plot,
    box_plot,
    scatter_plot,
    visualize_scalar_distribution,
)


def test_visualization_plot():
    data = pd.DataFrame(data={
        'x': [1., 2, 3, 4, 5, 6, 7, 8],
        'y': [1., 2, 3, 4, 5, 6, 7, 8]})
    filepath = osp.join(
        '.', 'visualize_scalar_distribution.pdf')
    assert not osp.exists(filepath)
    visualize_scalar_distribution(data, filepath)
    assert osp.exists(filepath)
    os.remove(filepath)

    filepath = osp.join('.', 'bar_plot.pdf')
    assert not osp.exists(filepath)
    bar_plot(data, filepath, xlabel_name='x', ylabel_name='y')
    assert osp.exists(filepath)
    os.remove(filepath)

    filepath = osp.join('.', 'scatter_plot.pdf')
    assert not osp.exists(filepath)
    scatter_plot(data, filepath, xlabel_name='x', ylabel_name='y')
    assert osp.exists(filepath)
    os.remove(filepath)

    filepath = osp.join('.', 'box_plot.pdf')
    assert not osp.exists(filepath)
    box_plot(data, filepath, xlabel_name='x', ylabel_name='y')
    assert osp.exists(filepath)
    os.remove(filepath)

    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.4, 0.4])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    filepath = osp.join('.', 'binary_roc_curve_plot.pdf')
    assert not osp.exists(filepath)
    binary_roc_curve_plot(y_true, y_score, filepath)
    assert osp.exists(filepath)
    os.remove(filepath)

    filepath = osp.join('.', 'binary_pr_curve_plot.pdf')
    assert not osp.exists(filepath)
    binary_pr_curve_plot(y_true, y_score, filepath,
                         positive_rate=(y_true == 1).sum() / len(y_true))
    assert osp.exists(filepath)
    os.remove(filepath)

    filepath = osp.join('.', 'binary_confusion_matrix_plot.pdf')
    assert not osp.exists(filepath)
    binary_confusion_matrix_plot(y_true, y_pred, filepath)
    assert osp.exists(filepath)
    os.remove(filepath)
