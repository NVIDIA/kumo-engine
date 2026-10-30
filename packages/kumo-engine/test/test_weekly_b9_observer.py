# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the B9 spawn import with the real Lightning runtime on CPU."""

import json
import runpy
from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_b9.py'


class Scores(LightningModule):
    def predict_step(self, batch, batch_idx):
        return {'score': batch}


@pytest.mark.parametrize('operation', ['fit', 'predict'])
def test_spawn_import_rejects_training_and_cpu_fallback(operation, tmp_path,
                                                        monkeypatch):
    monkeypatch.setenv('KUMO_B9_RECEIPTS', str(tmp_path))
    monkeypatch.setenv('KUMO_B9_CHECKPOINT', 'a' * 64)
    original_fit = Trainer.fit
    namespace = runpy.run_path(str(SOURCE), run_name='__mp_main__')
    try:
        trainer = Trainer(accelerator='cpu', logger=False,
                          enable_checkpointing=False)
        if operation == 'fit':
            with pytest.raises(ValueError, match='Training is forbidden'):
                trainer.fit(Scores())
        else:
            with pytest.raises(ValueError, match='fell back to CPU'):
                trainer.predict(Scores(), DataLoader(torch.ones(4)))
        [receipt] = tmp_path.glob('*.json')
        proof = json.loads(receipt.read_text())
        assert proof['status'] == 'FAIL'
        assert proof['fit_attempts'] == (1 if operation == 'fit' else 0)
        assert proof['batches'] == 0
        assert proof['restored_tensors'] == 0
    finally:
        namespace['_SPAWN_OBSERVER'].__exit__(None, None, None)
    assert Trainer.fit is original_fit
