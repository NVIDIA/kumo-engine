# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from kumo.engine import Trainer
from kumo.engine.trainer.trainer import _without_torch_scatter_hint

HINT = ("The usage of `scatter(reduce='min')` can be accelerated via the "
        "'torch-scatter' package, but it was not found")


def test_torch_scatter_hint_is_hidden_and_other_warnings_are_not() -> None:
    @_without_torch_scatter_hint
    def run() -> str:
        warnings.warn(HINT, UserWarning)
        warnings.warn('an unrelated warning', UserWarning)
        return 'done'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert run() == 'done'
    messages = [str(warning.message) for warning in caught]
    assert messages == ['an unrelated warning']


def test_fit_and_predict_hide_the_torch_scatter_hint() -> None:
    for method in (Trainer.fit, Trainer.predict):
        assert method.__wrapped__.__name__ == method.__name__
