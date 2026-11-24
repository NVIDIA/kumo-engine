# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional real customer-model lifecycle; needs kumo-pytorch and MLflow."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test' /
          'wheel_mlops_pytorch.py')


def test_regression_model_recovers_after_producer_scratch_is_removed(tmp_path):
    pytest.importorskip('kumo.pytorch.api')
    pytest.importorskip('mlflow.pytorch')
    # Run both stages with -I: customer code and dependencies must be available
    # without pytest's sys.path changes or the training process's objects.
    program = (
        'import importlib.util, sys; from pathlib import Path; '
        f's=importlib.util.spec_from_file_location("customer", {str(SCRIPT)!r}); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        f'root=Path({str(tmp_path)!r}); '
        'result=m.run_phase(PHASE, root, None); '
        'assert "kumo.engine.trainer" not in sys.modules; '
        'print(result)')
    environment = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT='1')
    for name in ('train', 'reload'):
        completed = subprocess.run(
            [sys.executable, '-I', '-c',
             program.replace('PHASE', repr(name))], cwd=tmp_path,
            env=environment, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        if name == 'train':
            shutil.rmtree(tmp_path / 'scratch-train')
    state = json.loads((tmp_path / 'customer-state.json').read_text())
    assert len({version['version'] for version in state['versions']}) == 2
    assert len({version['run'] for version in state['versions']}) == 2


@pytest.mark.parametrize(('platform', 'device', 'tensor_input'), [
    ('sagemaker', 'cuda', True),
    ('sagemaker', 'cpu', False),
    ('databricks', 'cuda', False),
    ('databricks', 'cpu', False),
    (None, 'cpu', False),
])
def test_mlflow_receives_an_example_its_platform_can_serialize(
        monkeypatch, platform, device, tensor_input):
    mlflow_pytorch = pytest.importorskip('mlflow.pytorch')
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('customer', SCRIPT)
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    # Simulate device transfer while exercising the exporter call on CPU CI.
    input_tensor = Mock()
    device_tensor = SimpleNamespace(device=SimpleNamespace(type=device))
    input_tensor.to.return_value = device_tensor
    monkeypatch.setattr(torch, 'from_numpy', lambda _: input_tensor)
    model = Mock(return_value=torch.zeros((2, 1)))
    export = Mock()
    monkeypatch.setattr(mlflow_pytorch, 'log_model', export)
    profile = SimpleNamespace(config={'platform': platform}) if platform else None

    workload.log_model(model, device, profile)

    input_tensor.to.assert_called_once_with(device)
    model.assert_called_once_with(device_tensor)
    args = export.call_args
    assert args.args == (model,)
    example = args.kwargs['input_example']
    if tensor_input:
        assert example is device_tensor
        assert example.device.type == 'cuda'
    else:
        assert isinstance(example, np.ndarray) and example.dtype == np.float32
    assert args.kwargs['signature'].inputs.inputs[0].shape == (-1, 1)
