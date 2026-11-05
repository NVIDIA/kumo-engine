# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.automl.auto_trainer import AutoTrainer


def test_all_nan_tune_metric_names_the_cause(tmp_path):
    for trial in ('trial_0', 'trial_1'):
        (tmp_path / trial / 'checkpoint').mkdir(parents=True)
        (tmp_path / trial / 'checkpoint' / 'best.ckpt').touch()
        (tmp_path / trial / 'val_result.csv').write_text(
            'val_auroc,val_loss\nnan,0.0345\n')
    trainer = SimpleNamespace(log_dir=str(tmp_path), tune_metric='auroc',
                              val_tune_metric='val_auroc')

    with pytest.raises(RuntimeError, match="'val_auroc' is NaN in all 2"):
        AutoTrainer.get_best_trials(trainer, k=1)
