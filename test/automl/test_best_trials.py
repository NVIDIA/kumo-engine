# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.automl.auto_trainer import AutoTrainer


@pytest.mark.parametrize('metric, single_class_hint', [('auroc', True),
                                                       ('mae', False)])
def test_all_nan_tune_metric(tmp_path, monkeypatch, metric, single_class_hint):
    for trial in ('trial_0', 'trial_1'):
        (tmp_path / trial / 'checkpoint').mkdir(parents=True)
        (tmp_path / trial / 'checkpoint' / 'best.ckpt').touch()
        (tmp_path / trial / 'val_result.csv').write_text(f'val_{metric}\nnan\n')
    monkeypatch.setattr(AutoTrainer, 'log_dir', str(tmp_path))
    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.tune_metric = metric

    with pytest.raises(RuntimeError) as error:
        trainer.get_best_trials(k=1)
    message = str(error.value)
    assert f"'val_{metric}' is NaN in all 2 completed trials" in message
    assert ('single class' in message) == single_class_hint
