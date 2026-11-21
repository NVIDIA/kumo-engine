# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional real customer-model lifecycle; needs kumo-pytorch and MLflow."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
