# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core import runtime_progress
from kumoapi.model_plan import ModelPlan

from kumo.engine import Trainer


@pytest.mark.parametrize('method,kind,error_type', [
    ('fit', 'training', AttributeError),
    ('predict', 'prediction', RuntimeError),
])
def test_public_operation_reports_original_failure(tmp_path, monkeypatch,
                                                   method, kind, error_type):
    events = []
    observer = SimpleNamespace(
        operation_started=lambda *args: events.append(('start', args)),
        operation_finished=lambda *args: events.append(('finish', args)),
    )
    monkeypatch.setattr(Trainer, 'scratch_dir', property(lambda _: tmp_path))
    trainer = Trainer(ModelPlan())
    token = runtime_progress.install(observer)
    try:
        with pytest.raises(error_type) as caught:
            getattr(trainer, method)(None, snapshot=None)
    finally:
        runtime_progress.reset(token)
    assert len(events) == 2
    assert events[0][0] == 'start'
    assert events[0][1][0] == kind
    assert events[1][0] == 'finish'
    assert events[1][1][0] == kind
    assert events[1][1][2] is caught.value
