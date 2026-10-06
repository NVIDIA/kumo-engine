# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from kumo_core.io import filesystem as fsutil
from kumo_core.train.base_module import BaseLightningModule
from kumo_core.train.base_node_module import BaseNodeLightningModule
from kumo_core.train.loss import loss_resolver
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage


@pytest.mark.parametrize('task_type', [
    TaskType.BINARY_CLASSIFICATION,
])
def test_binary_logits_to_predictions(task_type):
    loss = loss_resolver(task_type)
    logits = torch.tensor([[-1.0], [0.0], [1.0]])

    assert loss.to_prediction(logits).tolist() == [False, False, True]
    prob = loss.to_probability(logits)
    assert prob is not None
    assert torch.allclose(
        prob,
        torch.tensor([[0.7311, 0.2689], [0.5, 0.5], [0.2689, 0.7311]]),
        atol=1e-4,
    )


def test_multiclass_classification_logits_to_predictions():
    loss = loss_resolver(TaskType.MULTICLASS_CLASSIFICATION)
    logits = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])

    assert loss.to_prediction(logits).tolist() == [1, 0]
    prob = loss.to_probability(logits)
    assert prob is not None
    assert torch.allclose(
        prob,
        torch.tensor([[0.1192, 0.8808], [0.8808, 0.1192]]),
        atol=1e-4,
    )


def test_multilabel_classification_logits_to_predictions():
    loss = loss_resolver(TaskType.MULTILABEL_CLASSIFICATION)
    logits = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

    assert loss.to_prediction(logits).tolist() == [[1, 0, 1], [1, 0, 0]]
    prob = loss.to_probability(logits)
    assert prob is not None
    assert torch.allclose(
        prob,
        torch.tensor([[0.7311, 0.5, 0.7311], [0.7311, 0.5, 0.2689]]),
        atol=1e-4,
    )


@pytest.mark.parametrize('task_type', [TaskType.REGRESSION])
def test_logits_to_predictions_passthrough(task_type):
    loss = loss_resolver(TaskType.REGRESSION)
    inputs = torch.randn(5, 1)

    assert torch.allclose(loss.to_prediction(inputs), inputs)
    assert loss.to_probability(inputs) is None


def test_calibration_uses_registered_filesystem_for_adls_checkpoint(
        monkeypatch):
    monkeypatch.setattr(BaseNodeLightningModule, '__abstractmethods__', set())
    module = BaseNodeLightningModule(TaskType.BINARY_CLASSIFICATION)
    calibrator = torch.nn.Identity()
    calibrator.is_supported = True
    calibrator.is_fitted = False
    calibrator.fit = Mock()
    calibrator.reliability_diagram = Mock()
    module.calibrator = calibrator
    checkpoint_path = ('fsutil://abfss://data@account.dfs.core.windows.net/'
                       'jobs/version_0/checkpoints/best.ckpt')
    module._trainer = SimpleNamespace(
        ckpt_path=checkpoint_path,
        state=SimpleNamespace(fn=None),
    )
    source = BytesIO()
    torch.save({'state_dict': {}, 'optimizer_states': ['preserved']}, source)
    target = BytesIO()

    def open_file(path, mode):
        assert path == checkpoint_path
        return nullcontext(
            BytesIO(source.getvalue()) if mode == 'rb' else target)

    outputs = {
        'logits': torch.tensor([[-1.0], [1.0]]),
        'target': torch.tensor([[0.0], [1.0]]),
    }
    with (
            patch.object(BaseLightningModule, 'evaluation_epoch_end',
                         return_value=outputs),
            patch.object(fsutil, 'open_file', side_effect=open_file) as
            open_mock,
            patch.object(fsutil, 'makedirs') as makedirs_mock,
    ):
        module.evaluation_epoch_end(TrainingStage.TEST, [])

    assert [call.args[1] for call in open_mock.call_args_list] == ['rb', 'wb']
    target.seek(0)
    assert torch.load(target,
                      weights_only=False)['optimizer_states'] == ['preserved']
    makedirs_mock.assert_called_once_with(
        'abfss://data@account.dfs.core.windows.net/jobs/version_0/calibration',
        exist_ok=True,
    )
    assert calibrator.reliability_diagram.call_args.args[-1] == (
        'abfss://data@account.dfs.core.windows.net/jobs/version_0/'
        'calibration/reliability_diagram_test.png')
