# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from unittest.mock import create_autospec

import pytest
import torch
from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.automl.runtime import AutoTrainingRuntime
from kumo_core.instrumentation import NoOpTrainingInstrumentation
from kumo_core.model_plan import ModelPlan
from kumo_core.modeling.retrieval.mips import (
    FaissUnavailableError,
    MIPSFaiss,
    require_faiss,
)
from kumo_core.resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTrainTable,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import LinkPredOutputType


def test_imports_without_faiss() -> None:
    code = """
import importlib.abc
import sys

class BlockFaiss(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'faiss' or fullname.startswith('faiss.'):
            raise ModuleNotFoundError('FAISS intentionally absent', name='faiss')

sys.meta_path.insert(0, BlockFaiss())
import kumo_core.prediction
import kumo_core.automl.auto_trainer
import kumo_core.train.trainer
from kumo_core.modeling.retrieval.mips import MIPSBruteForce
import torch

keys = torch.tensor([[1., 0.], [0., 1.]])
scores, indices = MIPSBruteForce(keys).search(keys[:1], 1)
assert indices.tolist() == [[0]]
assert scores.tolist() == [[1.]]
assert 'faiss' not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_faiss_rejects_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'faiss', None)
    with pytest.raises(FaissUnavailableError, match='faiss-cpu'):
        MIPSFaiss(torch.zeros(2, 3))


@pytest.mark.parametrize(('task_type', 'module', 'needs_faiss'), [
    (TaskType.STATIC_LINK_PREDICTION, LinkPredOutputType.SHALLOW_RHS, True),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredOutputType.TWO_TOWER, True),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredOutputType.CONTEXT, True),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredOutputType.PAIRWISE, False),
    (TaskType.BINARY_CLASSIFICATION, LinkPredOutputType.SHALLOW_RHS, False),
])
def test_training_faiss_requirement(
    monkeypatch: pytest.MonkeyPatch,
    task_type: TaskType,
    module: LinkPredOutputType,
    needs_faiss: bool,
) -> None:
    monkeypatch.setitem(sys.modules, 'faiss', None)
    task = ResolvedNumericalTask(
        name='links', task_type=task_type,
        metrics=['recall'], graph=ResolvedTaskGraph(
            node_types=('users', 'items'), edge_types=(),
            seed_node_type='users', target_node_type='items'),
        train_table=ResolvedTrainTable('train', 'TARGET', None, None))
    plan = ModelPlan()
    plan.model_architecture.module = module
    runtime = create_autospec(AutoTrainingRuntime, instance=True)
    # Stop at the first runtime call, before filesystem or training setup.
    runtime.default_home_dir.side_effect = RuntimeError('runtime reached')
    if needs_faiss:
        with pytest.raises(FaissUnavailableError, match='faiss-cpu'):
            AutoTrainer(plan, task, runtime, NoOpTrainingInstrumentation())
        runtime.default_home_dir.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match='runtime reached'):
            AutoTrainer(plan, task, runtime, NoOpTrainingInstrumentation())
        runtime.default_home_dir.assert_called_once_with()


def test_cpu_faiss_rejects_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu_faiss = pytest.importorskip('faiss')
    monkeypatch.delattr(cpu_faiss, 'StandardGpuResources', raising=False)
    monkeypatch.delattr(cpu_faiss, 'knn_gpu', raising=False)
    assert require_faiss(torch.device('cpu')) is cpu_faiss
    with pytest.raises(FaissUnavailableError, match='GPU FAISS'):
        require_faiss(torch.device('cuda'))


def test_broken_faiss_is_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.abc

    class BrokenFaiss(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'faiss':
                raise ModuleNotFoundError('Missing numpy', name='numpy')

    monkeypatch.delitem(sys.modules, 'faiss', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [BrokenFaiss(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError) as error:
        require_faiss()
    assert error.value.name == 'numpy'
