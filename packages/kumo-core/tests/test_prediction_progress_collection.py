# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional prediction observation must not change unobserved loaders."""
from types import SimpleNamespace

from kumo_core import runtime_progress
from kumo_core.train import diskgraph_nested_trainer as nested_trainer


def test_unobserved_prediction_does_not_build_a_loader():
    def unexpected():
        raise AssertionError('progress constructed an unused dataloader')

    assert nested_trainer._prediction_progress_callbacks(
        SimpleNamespace(predict_dataloader=unexpected), None, 0) == []


def test_observed_streaming_prediction_has_unknown_total(monkeypatch):
    monkeypatch.setattr(runtime_progress, 'has_observer', lambda: True)
    callbacks = nested_trainer._prediction_progress_callbacks(
        SimpleNamespace(predict_dataloader=lambda: iter([1, 2])), None, 0)
    assert len(callbacks) == 1
    assert callbacks[0].progress.total_iterations == 0


def test_file_progress_is_collected_without_runtime_observer(tmp_path):
    callbacks = nested_trainer._prediction_progress_callbacks(
        SimpleNamespace(predict_dataloader=lambda: [1, 2]), str(tmp_path), 7)
    assert callbacks[0].progress.total_iterations == 2
    assert (tmp_path / 'prediction_progress_7.json').is_file()
