# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classification API refusals and same-capture allocation integration."""
from types import SimpleNamespace

import pytest
import torch

from kumo.engine.evidence.classification import (
    _capture_digest,
    _check_score,
    _summarize,
    explain_classification,
)


@pytest.mark.parametrize('kwargs', [
    dict(top_k=-1),
    dict(top_k=101),
    dict(top_k=True),
    dict(max_walks=0),
    dict(max_walks=10001)
])
def test_bounded_output_refuses_invalid_budget(kwargs):
    with pytest.raises(ValueError, match='must be an integer'):
        explain_classification(None, 'prediction', model='unused', **kwargs)


@pytest.mark.parametrize('task', [
    'regression', 'multiclass_classification', 'link_prediction', 'forecasting'
])
def test_only_binary_classification(task):
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
    with pytest.raises(ValueError, match='Only binary classification'):
        explain_classification(bundle, 'prediction', model='unused')


def test_raw_score_required_before_execution():
    bundle = SimpleNamespace(
        validate=lambda: None, manifest={
            'native_provenance': {
                'model_reconstruction': {
                    'task': {
                        'task_type': 'binary_classification'
                    }
                }
            }
        }, numerical_evidence=lambda _: {'prediction': .5})
    with pytest.raises(ValueError, match='raw head logit'):
        explain_classification(bundle, 'prediction', model='unused')


def test_capture_digest_binds_hidden_values_and_field_identity():
    capture = {'activation': torch.tensor([1., 2.]), 'field': 'amount'}
    assert _capture_digest(capture) == _capture_digest(dict(capture))
    assert _capture_digest(capture) != _capture_digest({
        **capture, 'activation':
        torch.tensor([2., 1.])
    })
    assert _capture_digest(capture) != _capture_digest({
        **capture, 'field': 'age'
    })
    for value in [float('nan'), float('inf'), 1.01]:
        with pytest.raises(ValueError, match='replay mismatch'):
            _check_score(value, 1., name='raw')
    _check_score(1. + 1e-6, 1., name='raw')


@pytest.mark.parametrize('budget', [1, 10000])
@pytest.mark.parametrize('sign', [-1, 1])
def test_summary_preserves_field_accounting_when_paths_are_bounded(
        budget, sign):
    from kumo_core.encoders.tensor_frame_encoder import TensorFrameEncoder
    from kumo_core.explain.encoded_fields import FieldCapture
    from kumo_core.explain.walk_graph import GraphCapture
    from kumoapi.encoder import Numerical
    from torch_frame import TensorFrame
    from walk_binary_fixtures import b1_batch, b1_forward, b1_module

    module = b1_module(dtype=torch.float32)
    x, edges, nodes, counts = b1_batch(dtype=torch.float32)
    encoders = torch.nn.ModuleDict({
        table:
        TensorFrameEncoder(
            table, {'amount': Numerical(scaler=None, na_strategy='separate')},
            None, 1, device='cpu')
        for table in x
    })
    module.encoder = torch.nn.Module()
    module.encoder.enc_dict = encoders
    module.eval()
    with torch.no_grad():
        for encoder in encoders.values():
            for parameter in encoder.parameters():
                parameter.fill_(1.)
        with FieldCapture(module) as fields, GraphCapture(module) as graph:
            encoded = {
                table:
                encoders[table](TensorFrame({'numerical': values},
                                            {'numerical': ['amount']}),
                                torch.zeros(len(values), dtype=torch.long))
                for table, values in x.items()
            }
            score = b1_forward(module, (encoded, edges, nodes, counts))
    computation = graph.computation(preserve_device=True)
    sample = {
        'nodes': {
            table: {
                'pkeys': ['duplicate'] * len(values)
            }
            for table, values in x.items()
        }
    }
    result = _summarize(computation, fields, sample, sign=sign,
                        max_walks=budget, top_k=1)
    assert sum(result['accounting'].values()) == pytest.approx(sign *
                                                               score.item())
    assert sum(row['field_relevance']
               for row in result['records']) == pytest.approx(
                   result['accounting']['fields'])
    assert len(result['walks']) <= 1
    assert len({(r['table'], r['occurrence'])
                for r in result['records']}) == len(result['records'])
    for orientation in ['positive', 'negative']:
        coverage = result['walk_coverage'][orientation]
        if budget == 1:
            assert coverage['omitted'] is None and coverage['total'] is None
        else:
            assert coverage['omitted'] == pytest.approx(coverage['total'] -
                                                        coverage['displayed'])
    for walk in result['walks']:
        assert [step['layer'] for step in walk['steps']] == [1, 2]
        for step in walk['steps']:
            assert step['source']['primary_key'] == 'duplicate'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_binary_api_same_capture_and_mandatory_replay(monkeypatch):
    from kumo_core.encoders.tensor_frame_encoder import TensorFrameEncoder
    from kumo_core.explain import local_sensitivity, perturbation
    from kumo_core.explain.walk_binary import emitted_probabilities
    from kumo_core.explain.walk_graph import capture_numerics
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumoapi.encoder import Numerical
    from torch_frame import TensorFrame
    from walk_binary_fixtures import b1_batch, b1_forward, binary_scoring_module

    module = binary_scoring_module()
    x, edges, nodes, counts = b1_batch(dtype=torch.float32)
    module.model = HeteroGNN(1, 3, metadata=(list(x), list(edges)),
                             aggr=['sum'], norm=None)
    module.encoder = torch.nn.Module()
    module.encoder.enc_dict = torch.nn.ModuleDict({
        table:
        TensorFrameEncoder(
            table, {'amount': Numerical(scaler=None, na_strategy='separate')},
            None, 1, device='cuda')
        for table in x
    })
    module.cuda().eval()
    x = {k: v.cuda() for k, v in x.items()}
    edges = {k: v.cuda() for k, v in edges.items()}
    nodes = {k: v + [0] for k, v in nodes.items()}
    counts = {k: v + [0] for k, v in counts.items()}

    def score(_):
        encoded = {
            table:
            module.encoder.enc_dict[table](
                TensorFrame({'numerical': values}, {'numerical': ['amount']}),
                torch.zeros(len(values), dtype=torch.long, device='cuda'))
            for table, values in x.items()
        }
        return b1_forward(module, (encoded, edges, nodes, counts))

    module.common_step = score
    with torch.no_grad():
        raw = score(None)
        probability = float(emitted_probabilities(module, raw)[0, 1])
    record = {
        'output': {
            'head_output': float(raw.item())
        },
        'prediction': probability,
        'sample_sha256': 'sample'
    }
    provenance = {
        'model_reconstruction': {
            'task': {
                'task_type': 'binary_classification'
            }
        },
        'bundle_sha256': 'model',
        'numerics': capture_numerics()
    }
    sample = {
        'nodes': {
            table: {
                'pkeys': list(range(len(values)))
            }
            for table, values in x.items()
        }
    }
    bundle = SimpleNamespace(validate=lambda: None,
                             manifest={'native_provenance': provenance},
                             numerical_evidence=lambda _: record,
                             scoring_graph=lambda _: sample)

    def load(path, actual_provenance):
        assert path == 'trusted-model' and actual_provenance is provenance
        return module

    monkeypatch.setattr(local_sensitivity, 'load_scoring_module', load)
    monkeypatch.setattr(perturbation, 'reconstruct_batch',
                        lambda _: SimpleNamespace(to=lambda _: None))
    first = explain_classification(bundle, 'prediction', model='trusted-model')
    repeated = explain_classification(bundle, 'prediction',
                                      model='trusted-model')
    assert first == repeated
    assert first['sample_sha256'] == 'sample'
    assert first['model_bundle_sha256'] == 'model'
    assert sum(first['accounting'].values()) == pytest.approx(
        first['selected_logit'])
    assert first['fields'] and first['walks']
    for walk in first['walks']:
        assert [step['layer'] for step in walk['steps']] == [1, 2, 3]
    record['output']['head_output'] += 1.
    with pytest.raises(ValueError, match='raw head replay mismatch'):
        explain_classification(bundle, 'prediction', model='trusted-model')
    record['output']['head_output'] -= 1.
    record['prediction'] = -1.
    with pytest.raises(ValueError, match='served probability replay mismatch'):
        explain_classification(bundle, 'prediction', model='trusted-model')
