# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from kumo_core.diskgraph.lightning_data import (
    suppress_checkpoint_callback_state_warnings,
    suppress_explain_workers_warning,
)

# Messages as Lightning 2.6 formats them.
WORKERS = ("The 'test_dataloader' does not have many workers which may be a "
           "bottleneck. Consider increasing the value of the `num_workers` "
           "argument` to `num_workers=15` in the `DataLoader` to improve "
           "performance.")
CALLBACKS = ("Be aware that when using `ckpt_path`, callbacks used to create "
             "the checkpoint need to be provided during `Trainer` "
             "instantiation. Please add the following callbacks: "
             "['EarlyStopping{'monitor': 'val_auroc', 'mode': 'max'}'].")
DIRPATH = ("The dirpath has changed from '/runs/fit/checkpoints' to "
           "'/models/bundle/checkpoints', therefore `best_model_score`, "
           "`kth_best_model_path`, `kth_value`, `last_model_path` and "
           "`best_k_models` won't be reloaded. Only `best_model_path` will be "
           "reloaded.")


@pytest.mark.parametrize(('suppress', 'message', 'category'), [
    (suppress_explain_workers_warning, WORKERS, PossibleUserWarning),
    (suppress_checkpoint_callback_state_warnings, CALLBACKS, UserWarning),
    (suppress_checkpoint_callback_state_warnings, DIRPATH, UserWarning),
])
def test_suppresses_only_its_lightning_messages(
    suppress: Callable[[], AbstractContextManager[None]],
    message: str,
    category: type[Warning],
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with suppress():
            warnings.warn(message, category)
            warnings.warn('an unrelated warning', UserWarning)
        warnings.warn(message, category)
    assert [str(warning.message) for warning in caught] == [
        'an unrelated warning', message
    ]
