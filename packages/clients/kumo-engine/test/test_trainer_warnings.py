# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
from kumoapi.model_plan import ModelPlan

from kumo.engine import Trainer
from kumo.engine.trainer.trainer import _without_unactionable_warnings

HINT = ("The usage of `scatter(reduce='min')` can be accelerated via the "
        "'torch-scatter' package, but it was not found")


def test_unactionable_warnings_are_hidden_and_other_warnings_are_not() -> None:
    @_without_unactionable_warnings
    def run() -> str:
        warnings.warn(HINT, UserWarning)
        warnings.warn('This process (pid=1) is multi-threaded, use of fork() '
                      'may lead to deadlocks in the child.',
                      DeprecationWarning)
        warnings.warn('Your `IterableDataset` has `__len__` defined.',
                      UserWarning)
        warnings.warn('an unrelated warning', UserWarning)
        return 'done'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert run() == 'done'
        warnings.warn(HINT, UserWarning)
    messages = [str(warning.message) for warning in caught]
    assert messages == ['an unrelated warning', HINT]


@pytest.mark.parametrize(('method', 'boundary'), [
    ('fit', 'kumo.engine.trainer.trainer._to_core_model_plan'),
    ('predict', 'kumo_core.prediction_options.validate_prediction_workers'),
])
def test_fit_and_predict_hide_the_torch_scatter_hint(
        monkeypatch: pytest.MonkeyPatch, method: str, boundary: str) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        warnings.warn(HINT, UserWarning)
        warnings.warn('an unrelated warning', UserWarning)
        raise RuntimeError('stop at test boundary')

    monkeypatch.setattr(boundary, stop)
    trainer = Trainer(ModelPlan())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(RuntimeError, match='stop at test boundary'):
            getattr(trainer, method)(None, snapshot=None)
        warnings.warn(HINT, UserWarning)
    assert [str(warning.message) for warning in caught] == [
        'an unrelated warning', HINT
    ]
