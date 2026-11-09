# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from types import SimpleNamespace
from typing import Any

import optuna
import pytest
from kumo_core.automl.optuna_searcher import TPESearcher

SPACE = SimpleNamespace(training_job=SimpleNamespace(tune_metric='auroc'))


def test_tpe_searcher_construction_emits_no_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        TPESearcher(SPACE)
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize(('version', 'passed'), [
    ('4.8.0', True),
    ('4.9.0', False),
    ('4.10.1', False),
])
def test_independent_sampling_warnings_off_on_every_optuna_version(
        monkeypatch: pytest.MonkeyPatch, version: str, passed: bool) -> None:
    sampler_kwargs: dict[str, Any] = {}

    def sampler(**kwargs: Any) -> optuna.samplers.BaseSampler:
        sampler_kwargs.update(kwargs)
        return optuna.samplers.RandomSampler()

    monkeypatch.setattr(optuna, '__version__', version)
    monkeypatch.setattr(optuna.samplers, 'TPESampler', sampler)
    TPESearcher(SPACE)
    # Before 4.9 the sampler warns by default; from 4.9 the argument is
    # deprecated and False is the default.
    assert ('warn_independent_sampling' in sampler_kwargs) == passed
    assert sampler_kwargs.get('warn_independent_sampling', False) is False


def test_analyze_emits_no_experimental_warnings(tmp_path) -> None:
    searcher = TPESearcher(SPACE)
    searcher.study.optimize(lambda trial: trial.suggest_float('lr', 0, 1),
                            n_trials=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        searcher.analyze(str(tmp_path))
    assert [
        str(warning.message) for warning in caught
        if issubclass(warning.category, optuna.exceptions.ExperimentalWarning)
    ] == []
