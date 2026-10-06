# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared training-budget clamps for slow e2e tests."""

from __future__ import annotations

from typing import Any

DEFAULT_E2E_NUM_EXPERIMENTS = 1
DEFAULT_E2E_MAX_EPOCHS = 2
DEFAULT_E2E_MAX_STEPS_PER_EPOCH = 100
DEFAULT_E2E_MAX_VAL_STEPS = 100
DEFAULT_E2E_MAX_TEST_STEPS = 200


def _clamp_int_attr(target: Any, attr_name: str, max_value: int) -> None:
    current = getattr(target, attr_name, None)
    if isinstance(current, int):
        setattr(target, attr_name, min(current, max_value))
    elif current is None:
        setattr(target, attr_name, max_value)


def clamp_e2e_training_budget(
    model_plan: Any,
    *,
    keep_refit_trainval: bool = False,
    keep_refit_full: bool = False,
) -> Any:
    """Clamp an e2e model plan to a smaller training budget.

    This only reduces existing budgets. It does not increase already-small
    plans such as those created from DEBUG-mode defaults.
    """
    training_job = getattr(model_plan, 'training_job', None)
    optimization = getattr(model_plan, 'optimization', None)

    if training_job is not None:
        _clamp_int_attr(training_job, 'num_experiments',
                        DEFAULT_E2E_NUM_EXPERIMENTS)
        if hasattr(training_job, 'refit_trainval') and not keep_refit_trainval:
            training_job.refit_trainval = False
        if hasattr(training_job, 'refit_full') and not keep_refit_full:
            training_job.refit_full = False

    if optimization is not None:
        _clamp_int_attr(optimization, 'max_epochs', DEFAULT_E2E_MAX_EPOCHS)
        _clamp_int_attr(optimization, 'max_steps_per_epoch',
                        DEFAULT_E2E_MAX_STEPS_PER_EPOCH)
        _clamp_int_attr(optimization, 'max_val_steps',
                        DEFAULT_E2E_MAX_VAL_STEPS)
        _clamp_int_attr(optimization, 'max_test_steps',
                        DEFAULT_E2E_MAX_TEST_STEPS)

    return model_plan
