# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# from rest.config.training_job import Metric
# from rest.config.utils import filter_metrics, remove_test_prefix
import numpy as np
import pytest
from kumo_core.train.output import TargetPredictionHeatMap

from rest.config.evaluation import RegressionResiduals


def test_regression_residuals_basic():
    # test generate a regression residual class from a heatmap without the
    # need of cell width resize
    small_heatmap = TargetPredictionHeatMap(
        cell_width=2.0,
        label_bin_start=-6.0,
        label_bin_end=32.0,
        prediction_bin_start=-20.0,
        prediction_bin_end=52.0,
        cells=[(-6.0, -20.0), (-4.0, 0.0), (0.0, 50.0), (10.0, 4.0),
               (30.0, -10.0)],
        counts=[1, 1, 1, 1, 1],
        home_dir='./test_small_heatmap',
    )
    adjust_cell_width = RegressionResiduals.heatmap_cell_width_resize(
        small_heatmap)
    assert adjust_cell_width == 2.0
    small_regression_residuals = (
        RegressionResiduals.from_target_prediction_heatmap(small_heatmap))
    assert small_regression_residuals.x_values == np.arange(
        small_heatmap.label_bin_start, small_heatmap.label_bin_end + 1e-6,
        small_heatmap.cell_width).tolist()
    assert small_regression_residuals.y_values == np.arange(
        small_heatmap.prediction_bin_start,
        small_heatmap.prediction_bin_end + 1e-6,
        small_heatmap.cell_width).tolist()

    # test generate a regression residual class from a heatmap width the
    # need of cell width resize
    large_heatmap = TargetPredictionHeatMap(
        cell_width=0.2,
        label_bin_start=-6.0,
        label_bin_end=32.0,
        prediction_bin_start=-20.0,
        prediction_bin_end=52.0,
        cells=[(-6.0, -20.0), (-4.0, 0.0), (0.0, 50.0), (10.0, 4.0),
               (30.0, -10.0)],
        counts=[1, 1, 1, 1, 1],
        home_dir='./test_large_heatmap',
    )
    adjust_cell_width = RegressionResiduals.heatmap_cell_width_resize(
        large_heatmap)
    assert adjust_cell_width == pytest.approx(0.523, 0.001)
    large_regression_residuals = (
        RegressionResiduals.from_target_prediction_heatmap(large_heatmap))
    assert len(large_regression_residuals.x_values) == 73
    assert len(large_regression_residuals.y_values) == 138
