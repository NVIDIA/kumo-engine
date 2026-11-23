# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check rendered explanation labels; provider qualification still needs real notebooks."""

import importlib.util
from pathlib import Path

import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')


@pytest.fixture
def layout():
    path = (
        Path(__file__).resolve().parents[2]
        / 'packages/kumo-engine/test/snowflake_notebook/explanation.py'
    )
    spec = importlib.util.spec_from_file_location('notebook_explanation_layout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('count', [10, 11, 12])
def test_default_size_rotated_cohort_labels_do_not_falsely_overlap(layout, count):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_xticks(
        range(count),
        [f'({i * 1000} - {(i + 1) * 1000}]' for i in range(count)],
        rotation=35,
        ha='right',
    )
    figure.tight_layout()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer) for text in axis.get_xticklabels()]
        assert any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
        layout.check_layout(figure)
        assert tuple(figure.get_size_inches()) == (8, 6)
    finally:
        plt.close(figure)


def test_rotated_outlines_still_reject_actual_text_overlap(layout):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_xlim(0, 10)
    axis.set_xticks(
        [4, 4.01],
        ['Long cohort label one', 'Long cohort label two'],
        rotation=35,
        ha='right',
    )
    figure.tight_layout()
    try:
        with pytest.raises(AssertionError):
            layout.check_layout(figure)
    finally:
        plt.close(figure)


def test_long_cohort_boundary_keeps_default_plot_size(layout):
    import matplotlib.pyplot as plt

    figures = layout.layout_examples(dict(plots={}))
    try:
        for kind, figure, _ in figures:
            layout.check_layout(figure)
            if kind == 'cohort-layout':
                assert tuple(figure.get_size_inches()) == (8, 6)
    finally:
        for _, figure, _ in figures:
            plt.close(figure)
