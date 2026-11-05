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


def candidate_work():
    return dict(
        fits=2, completed_fits=2, optimizer_steps=18, training_batches=6,
        training_epochs=[
            dict(batches=2, batch_sizes=[512, 37]),
            dict(batches=1, batch_sizes=[1024]),
            dict(batches=3, batch_sizes=[1024, 71, 9])
        ], early_stopping=dict(stopped_epoch=1, patience=0, wait_count=0),
        checkpoint_sha256='a' * 64, fit_ledger=[
            dict(
                status='PASS', steps=6, cuda_batches=3, epochs=[
                    dict(batches=2, batch_sizes=[512, 37]),
                    dict(batches=1, batch_sizes=[1024])
                ], validation=[dict(epoch=0, metrics={'val/score': .8})],
                checkpoint_sha256='a' * 64),
            dict(status='PASS', steps=12, cuda_batches=3,
                 epochs=[dict(batches=3, batch_sizes=[1024, 71, 9])],
                 validation=[dict(epoch=0,
                                  metrics={'val/score':
                                           .7})], checkpoint_sha256='b' * 64)
        ])


def test_training_accepts_multiple_candidates_and_an_earlier_selected_model():
    work = candidate_work()
    nightly.check_training(work, 4)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, optimizer_steps=0), 4)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, checkpoint_sha256='c' * 64), 4)
    with pytest.raises(ValueError):
        nightly.check_training(work, 8)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, completed_fits=1), 4)


def test_failed_candidate_stays_visible_without_rejecting_a_selected_model():
    work = candidate_work()
    work['fits'] = 3
    work['fit_ledger'].append(
        dict(status='FAIL', steps=0, cuda_batches=0, epochs=[], validation=[],
             error='RuntimeError: candidate allocation failed'))
    nightly.check_training(work, 4)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, checkpoint_sha256='c' * 64), 4)


def test_observer_retains_validation_and_multiple_stopping_callbacks(
        tmp_path, monkeypatch):
    import hashlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.trainer.states import TrainerFn

    monkeypatch.setattr(nightly.smoke, 'observe_cuda',
                        lambda state: nullcontext())
    checkpoint = ModelCheckpoint(dirpath=tmp_path)
    path = tmp_path / 'selected.ckpt'
    path.write_bytes(b'earlier candidate checkpoint')
    checkpoint.best_model_path = str(path)
    callbacks = [
        EarlyStopping(monitor='val/score'),
        EarlyStopping(monitor='val/other'), checkpoint
    ]
    state = dict(stage='fit', fit_ledger=[], training_epochs=[])
    with nightly.observe_cuda(state):
        actual = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                            callbacks=callbacks)
        reloaded = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                              enable_checkpointing=False)
    [proof] = [
        c for c in actual.callbacks
        if (type(c).__name__ == 'Proof'
            and type(c).__module__ != nightly.smoke.__name__)
    ]
    trainer = SimpleNamespace(current_epoch=0, global_step=2,
                              callbacks=actual.callbacks,
                              checkpoint_callback=checkpoint,
                              sanity_checking=False,
                              callback_metrics={'val/score': torch.tensor(.25)
                                                }, should_stop=False,
                              state=actual.state, ckpt_path=str(path))
    trainer.state.fn = TrainerFn.FITTING
    module = SimpleNamespace(lhs_name='users')
    proof.on_fit_start(trainer, module)
    proof.on_train_epoch_start(trainer, module)
    proof.on_train_batch_end(trainer, module, None,
                             {'users': SimpleNamespace(batch_size=17)}, 0)
    proof.on_validation_end(trainer, module)
    proof.on_fit_end(trainer, module)
    [observed] = state['fit_ledger']
    assert observed['epochs'][0]['batch_sizes'] == [17]
    assert observed['validation'][0]['metrics']['val/score'] == .25
    assert len(observed['early_stopping']) == 2
    assert observed['checkpoint_sha256'] == hashlib.sha256(
        path.read_bytes()).hexdigest()
    path.write_bytes(b'checkpoint after default classification calibration')
    trainer.state.fn = TrainerFn.VALIDATING
    proof.on_validation_end(trainer, module)
    assert observed['checkpoint_sha256'] == hashlib.sha256(
        path.read_bytes()).hexdigest()
    [other_proof] = [
        c for c in reloaded.callbacks
        if (type(c).__name__ == 'Proof'
            and type(c).__module__ != nightly.smoke.__name__)
    ]
    other_proof.on_validation_end(trainer, module)
    assert len(observed['validation']) == 2


def test_observer_tracks_a_real_lightning_fit_and_final_validation(
        tmp_path, monkeypatch):
    from contextlib import nullcontext

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.trainer.states import TrainerFn
    from torch.utils.data import DataLoader
    from torch_geometric.data import HeteroData

    monkeypatch.setattr(nightly.smoke, 'observe_cuda',
                        lambda state: nullcontext())

    class Model(pl.LightningModule):
        lhs_name = 'users'

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            return self.layer(batch['users'].x).square().mean()

        def validation_step(self, batch, batch_idx):
            self.log('val/score',
                     self.layer(batch['users'].x).square().mean(), batch_size=2)

        def on_validation_epoch_end(self):
            if self.trainer.state.fn == TrainerFn.VALIDATING:
                checkpoint = torch.load(self.trainer.ckpt_path,
                                        weights_only=False)
                checkpoint['state_dict']['layer.bias'] += .1
                torch.save(checkpoint, self.trainer.ckpt_path)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.01)

    batch = HeteroData()
    batch['users'].x = torch.tensor([[1.], [2.]])
    batch['users'].batch_size = 2
    loader = DataLoader([batch, batch], batch_size=1,
                        collate_fn=lambda items: items[0])
    state = dict(stage='fit', fit_ledger=[], training_epochs=[])
    with nightly.observe_cuda(state):
        trainer = pl.Trainer(
            accelerator='cpu', devices=1, max_epochs=1, logger=False,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[ModelCheckpoint(dirpath=tmp_path, monitor='val/score')])
        model = Model()
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        [candidate] = state['fit_ledger']
        before = candidate['checkpoint_sha256']
        trainer.validate(model, dataloaders=loader,
                         ckpt_path=trainer.checkpoint_callback.best_model_path)
    assert candidate['steps'] == 2
    assert candidate['cuda_batches'] == 2
    assert candidate['checkpoint_sha256'] != before
    assert candidate['checkpoint_sha256'] == nightly.digest(
        trainer.checkpoint_callback.best_model_path)
    assert candidate['validation'][-1]['phase'] == 'candidate-validation'


def test_candidate_ignores_reference_profile_but_keeps_host_requirement(
        monkeypatch):
    import kumo.engine as kumo

    monkeypatch.setattr(nightly.platform, 'machine', lambda: 'unsupported-host')
    profile = {'nightly-reference-only-test-package': '0.0'}
    with pytest.raises(ValueError, match='one actual CUDA GPU'):
        nightly.verify_runtime(kumo.__version__, profile)
    with pytest.raises(ValueError, match='selected profile'):
        nightly.verify_runtime(kumo.__version__, profile, reference_only=True)
    with pytest.raises(ValueError, match='one actual CUDA GPU'):
        nightly.verify_runtime(kumo.__version__, None, reference_only=True)


def test_fit_predictions_do_not_replace_the_four_prediction_checks():
    from types import SimpleNamespace

    import pytorch_lightning as pl
    import torch

    state = dict(stage='fit',
                 prediction_batches={stage: 0
                                     for stage in nightly.PREDICTIONS})
    with nightly.smoke.observe_cuda(state):
        trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                             enable_checkpointing=False)
    [proof] = [c for c in trainer.callbacks if hasattr(c, 'check')]
    proof.on_predict_batch_end(trainer, SimpleNamespace(device='cpu'),
                               torch.tensor([1.]), None, 0)
    assert state['prediction_batches'] == dict.fromkeys(nightly.PREDICTIONS, 0)


@pytest.mark.parametrize('method', ['predict_v2', 'test'])
@pytest.mark.parametrize('recover', [False, True])
def test_nightly_keeps_product_oom_recovery(monkeypatch, method, recover):
    from types import SimpleNamespace

    import torch
    from kumo_core.train.trainer import Trainer, handle_gpu_oom

    batches = []
    allocator = {'num_ooms': 0}
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: allocator.copy())

    @handle_gpu_oom
    def predict(self):
        size = self.datamodule.eval_loader_kwargs['batch_size']
        batches.append(size)
        if recover and size > 32:
            allocator['num_ooms'] += 1
            raise torch.cuda.OutOfMemoryError('injected allocation failure')
        return 'complete prediction'

    monkeypatch.setattr(Trainer, method, predict)
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(eval_loader_kwargs={'batch_size': 64}),
        pl_model=SimpleNamespace(reset_embedding_hook=lambda: None))
    state = dict(stage='predict')
    with nightly.observe_cuda(state):
        assert getattr(Trainer, method)(trainer) == 'complete prediction'
    assert batches == ([64, 32] if recover else [64])
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 64
    assert state['oom_recoveries'] == ([
        dict(stage='predict', method=method, allocator_ooms=1)
    ] if recover else [])


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
