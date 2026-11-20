# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Companion checkpoint, predictor binding and binary masking API contracts."""
from types import SimpleNamespace

import pytest
import torch
from kumo_core.explain.classification_masking import (
    FeasibilityCompanion,
    fit_feasibility,
)

from kumo.engine.evidence.masking import (
    ClassificationMaskingCompanion,
    _load_binary,
    _replay,
    _score,
)

KEY = (1, ('events', 'owner_id', 'owners'))
MODEL = 'a' * 64


def test_owned_checkpoint_roundtrip_integrity_threshold_and_optimizer(tmp_path):
    network = FeasibilityCompanion([KEY], 1, {1: torch.tensor([.25])})
    contexts = {KEY: torch.ones(2, 3)}
    optimizer, curve = fit_feasibility(network, [(contexts, {
        KEY: torch.zeros(2)
    })], epochs=2)
    companion = ClassificationMaskingCompanion(network,
                                               model_bundle_sha256=MODEL,
                                               decision_threshold=.35,
                                               optimizer_state=optimizer,
                                               training={'curve': curve})
    path = tmp_path / 'companion.pt'
    companion.save(path)
    loaded = ClassificationMaskingCompanion.load(path)
    assert loaded._model_sha256 == MODEL and loaded._threshold == .35
    assert loaded._optimizer_state['state']
    assert loaded._training == companion._training
    assert all(
        torch.equal(a, b)
        for a, b in zip(network.state_dict().values(),
                        loaded._network.state_dict().values()))
    with pytest.raises(FileExistsError):
        companion.save(path)
    payload = torch.load(path, weights_only=True)
    payload['decision_threshold'] = .6
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='integrity'):
        ClassificationMaskingCompanion.load(tmp_path / 'tampered.pt')
    payload['decision_threshold'] = .35
    payload['state_dict']['standins.1'].add_(1.)
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='integrity'):
        ClassificationMaskingCompanion.load(tmp_path / 'tampered.pt')


@pytest.mark.parametrize('task', [
    'regression', 'multiclass_classification', 'forecasting', 'link_prediction'
])
def test_other_tasks_refused_before_model_loading(task):
    bundle = SimpleNamespace(
        validate=lambda: None, manifest={
            'native_provenance': {
                'model_reconstruction': {
                    'task': {
                        'task_type': task
                    }
                }
            }
        })
    with pytest.raises(ValueError, match='binary classification only'):
        _load_binary(bundle, 'unused')


def test_calibrated_served_score_and_strict_keepall_replay():
    from walk_binary_fixtures import binary_scoring_module
    module = binary_scoring_module(calibration=(.5, -.3))
    raw = torch.tensor([[1.]])
    scored = _score(module, raw)
    assert scored['raw'] == 1.
    expected = module.loss.to_probability(module.calibrator(raw))[0, 1]
    assert scored['served'] == float(expected)
    assert scored['served'] != float(raw.sigmoid())
    _replay(scored, dict(scored))
    with pytest.raises(ValueError, match='raw replay'):
        _replay({**scored, 'raw': 1.001}, scored)
    with pytest.raises(ValueError, match='served replay'):
        _replay({**scored, 'served': float('nan')}, scored)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='New finite GPU validation not authorized')
def test_cuda_fit_load_and_cumulative_masking(tmp_path, monkeypatch):
    from kumo_core.explain.walk_graph import capture_numerics
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from walk_binary_fixtures import (
        b1_batch,
        b1_forward,
        binary_scoring_module,
    )

    import kumo.engine.evidence.masking as masking

    module = binary_scoring_module(calibration=(.7, -.1))
    x, edges, nodes, counts = b1_batch(dtype=torch.float32)
    module.model = HeteroGNN(1, 3, metadata=(list(x), list(edges)),
                             aggr=['sum'], norm=None)
    module.cuda().eval().requires_grad_(False)
    x = {k: v.cuda() for k, v in x.items()}
    edges = {k: v.cuda() for k, v in edges.items()}
    nodes, counts = ({
        k: v + [0]
        for k, v in data.items()
    } for data in (nodes, counts))
    module.common_step = lambda _: b1_forward(module, (x, edges, nodes, counts))

    class Batch(dict):
        def to(self, _):
            return self

    batch = Batch({
        module.entity_table_name: SimpleNamespace(batch_size=1),
        module.train_table_name: SimpleNamespace(num_nodes=1)
    })
    original = _score(module, module.common_step(batch))
    record = {
        'output': {
            'head_output': original['raw']
        },
        'prediction': original['served'],
        'sample_sha256': 'b' * 64,
        'prediction_id': 'case'
    }
    provenance = {
        'model_reconstruction': {
            'task': {
                'task_type': 'binary_classification'
            }
        },
        'bundle_sha256': MODEL,
        'numerics': capture_numerics()
    }
    sample = {
        'nodes': {
            table: {
                'pkeys': ['repeated'] * len(value)
            }
            for table, value in x.items()
        }
    }
    bundle = SimpleNamespace(validate=lambda: None,
                             manifest={'native_provenance': provenance},
                             numerical_evidence=lambda _: record,
                             iter_numerical_evidence=lambda: iter([record]),
                             scoring_graph=lambda _: sample)
    monkeypatch.setattr(masking, 'load_scoring_module', lambda *_: module)
    monkeypatch.setattr(masking, 'reconstruct_batch', lambda _: batch)
    companion = ClassificationMaskingCompanion.fit(
        bundle, model='trusted',
        standins={i: torch.tensor([.2])
                  for i in (1, 2, 3)}, decision_threshold=.37, epochs=2)
    path = tmp_path / 'companion.pt'
    companion.save(path)
    restored = ClassificationMaskingCompanion.load(path)
    first = restored.mask(bundle, 'case', model='trusted')
    repeated = restored.mask(bundle, 'case', model='trusted')
    assert {
        k: v
        for k, v in first.items() if k != 'timing'
    } == {
        k: v
        for k, v in repeated.items() if k != 'timing'
    }
    assert first['fidelity']['fidelity_pass']
    assert first['decision_threshold'] == .37
    assert first['replaced_messages'] + first['kept_messages'] == first[
        'total_messages']
    assert first['timing']['forward_calls'] == 2 + len(first['attempts'])
    assert all(p.grad is None for p in module.parameters())
    restored._model_sha256 = 'c' * 64
    with pytest.raises(ValueError, match='different predictor'):
        restored.mask(bundle, 'case', model='trusted')


def test_same_forward_context_preserves_occurrences_receivers_and_target():
    from kumo_core.explain.classification_masking import capture_context
    from walk_binary_fixtures import FK, REV, b1_batch, b1_forward, b1_module

    module = b1_module(gnn_bias=True).requires_grad_(False)
    inputs = b1_batch()
    module.common_step = lambda _: b1_forward(module, inputs)
    batch = {
        module.entity_table_name: SimpleNamespace(batch_size=1),
        module.train_table_name: SimpleNamespace(num_nodes=1)
    }
    raw, contexts, occurrences = capture_context(module, batch)
    assert torch.equal(raw, module.common_step(batch))
    # Both reverse edges share a source but retain distinct receiver occurrences.
    assert torch.equal(
        contexts[(1, REV)],
        torch.tensor([[1., 2., 1.], [1., 3., 1.]], dtype=raw.dtype))
    assert occurrences[(1, REV)]['edges'].tolist() == [[0, 0], [0, 1]]
    assert occurrences[(1, REV)]['direction'] == 'pk_to_fk'
    assert occurrences[(2, FK)]['direction'] == 'fk_to_pk'
    assert all(not value.requires_grad for value in contexts.values())
    assert not any(
        hasattr(m, '_message_intervention') for m in module.modules())
    batch[module.entity_table_name].batch_size = 2
    with pytest.raises(ValueError, match='one entity seed'):
        capture_context(module, batch)


def test_checkpoint_owns_network_and_rejects_non_mapping_payload(tmp_path):
    network = FeasibilityCompanion([KEY], 1, {1: torch.tensor([.25])})
    companion = ClassificationMaskingCompanion(network,
                                               model_bundle_sha256=MODEL,
                                               decision_threshold=.35)
    with torch.no_grad():
        network.standins['1'].add_(1.)
        next(network.networks.parameters()).add_(1.)
    assert companion._network.standins['1'].item() == .25
    assert not torch.equal(next(network.networks.parameters()),
                           next(companion._network.networks.parameters()))
    torch.save([], tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='integrity'):
        ClassificationMaskingCompanion.load(tmp_path / 'bad.pt')
