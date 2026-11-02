# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise product recovery and immutable model execution without a new fit."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_learning', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


def test_completed_training_allows_partial_batches_and_actual_early_stopping():
    work = dict(
        core_fit_attempts=1, fits=1, completed_fits=1, optimizer_steps=80,
        training_batches=80, training_epochs=[
            dict(batches=20, batch_sizes=[64] * 19 + [32]) for _ in range(4)
        ], early_stopping=dict(stopped_epoch=3, patience=2, wait_count=2))
    nightly.check_training(work, 40)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, optimizer_steps=0), 40)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, early_stopping={}), 40)


def test_candidate_ignores_reference_profile_but_keeps_host_requirement(
        monkeypatch):
    import kumo.engine as kumo

    monkeypatch.setattr(nightly.platform, 'machine', lambda: 'unsupported-host')
    profile = {'nightly-reference-only-test-package': '0.0'}
    with pytest.raises(ValueError, match='one actual CUDA GPU'):
        nightly.verify_runtime(kumo.__version__, profile)
    with pytest.raises(ValueError, match='selected profile'):
        nightly.verify_runtime(kumo.__version__, profile, reference_only=True)


@pytest.mark.parametrize('method', ['predict_v2', 'test'])
@pytest.mark.parametrize('recover', [False, True])
def test_nightly_keeps_product_oom_recovery(monkeypatch, tmp_path, method,
                                            recover):
    from types import SimpleNamespace

    import torch
    from kumo_core.train.trainer import Trainer, handle_gpu_oom

    batches = []

    @handle_gpu_oom
    def predict(self):
        size = self.datamodule.eval_loader_kwargs['batch_size']
        batches.append(size)
        if recover and size > 32:
            raise torch.cuda.OutOfMemoryError('injected allocation failure')
        return 'complete prediction'

    monkeypatch.setattr(Trainer, method, predict)
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(eval_loader_kwargs={'batch_size': 64}),
        pl_model=SimpleNamespace(reset_embedding_hook=lambda: None))
    state = dict(core_fit_attempts=0, stage='predict', native_ingests=[],
                 pseudo_options={}, ledger_path=str(tmp_path / 'ledger.json'))
    with nightly.observe_cuda(state):
        assert getattr(Trainer, method)(trainer) == 'complete prediction'
    assert batches == ([64, 32] if recover else [64])
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 64


def test_installed_wheel_check_rejects_changed_runtime_files(tmp_path):
    import importlib.metadata
    import zipfile

    dist_info = tmp_path / 'example-1.0.dist-info'
    dist_info.mkdir()
    metadata = b'Name: example\nVersion: 1.0\n'
    (dist_info / 'METADATA').write_bytes(metadata)
    source = tmp_path / 'example.py'
    source.write_bytes(b'VALUE = 1\n')
    wheel = tmp_path / 'example-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('example.py', source.read_bytes())
        archive.writestr('example-1.0.dist-info/METADATA', metadata)
        archive.writestr('example-1.0.dist-info/RECORD', b'wheel record')
    distribution = importlib.metadata.Distribution.at(dist_info)
    check = nightly.verify_installed_files
    assert check(wheel, distribution) == 2
    source.write_bytes(b'VALUE = 2\n')
    with pytest.raises(ValueError):
        check(wheel, distribution)
    source.unlink()
    with pytest.raises(ValueError):
        check(wheel, distribution)
