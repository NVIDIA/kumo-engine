# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from kumoapi.model_plan import ModelPlan

from kumo.engine import PredictionExplanations, Trainer


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_local_flag_validated_before_inputs(value):
    with pytest.raises(ValueError, match='local_sensitivity must be a boolean'):
        Trainer(ModelPlan()).predict(local_sensitivity=value)


def test_local_runtime_constraints_validated_first():
    with pytest.raises(ValueError, match='CPU and one scorer'):
        Trainer(ModelPlan(), accelerator='cpu').predict(local_sensitivity=True,
                                                        num_workers=2)


def test_local_readers_and_independent_availability(tmp_path):
    local = tmp_path / 'xai/local'
    (local / 'rows').mkdir(parents=True)
    (local / 'samples').mkdir()
    prediction_id = 'a' * 64
    row = {'prediction_id': prediction_id, 'status': 'zero_signal'}
    (local / 'manifest.json').write_text(json.dumps({'complete': True}))
    (local / 'index.jsonl').write_text(json.dumps(row) + '\n')
    (local / 'rows' / f'{prediction_id}.json').write_text(json.dumps(row))
    sample = {'origin': 'original_forward'}
    (local / 'samples' / f'{prediction_id}.json').write_text(json.dumps(sample))
    xai = PredictionExplanations.load(tmp_path)
    assert xai.local_availability.available
    assert not xai.availability['samples'].available
    assert xai.local_sensitivity_manifest()['complete']
    assert xai.local_sensitivities() == (row, )
    assert xai.local_sensitivity(prediction_id) == row
    assert xai.scoring_sample(prediction_id) == sample
    for reader in (xai.local_sensitivity, xai.scoring_sample):
        with pytest.raises(ValueError, match='SHA256'):
            reader('../escape')
    missing = PredictionExplanations.from_run(tmp_path / 'missing',
                                              local_requested=True)
    assert missing.local_availability.reason == 'failed'


def test_evidence_configuration_requires_explicit_batch_size():
    from kumo.engine import EvidenceConfig, Trainer
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    with pytest.raises(ValueError, match='batch_size=1'):
        trainer.predict(evidence=EvidenceConfig())


def test_evidence_record_policy_does_not_reject_runtime_before_inputs():
    from kumo.engine import EvidenceConfig, Trainer
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    with pytest.raises(TypeError, match='snapshot'):
        trainer.predict(evidence=EvidenceConfig(on_unsupported='record'),
                        num_workers=2)


def test_replay_only_evidence_does_not_require_single_seed():
    from kumo.engine import EvidenceConfig, Trainer
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    with pytest.raises(TypeError, match='snapshot'):
        trainer.predict(evidence=EvidenceConfig(
            scoring_graph=False, numerical_sensitivity=None, replay=True))
