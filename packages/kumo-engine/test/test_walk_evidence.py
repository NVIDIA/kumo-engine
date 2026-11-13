# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData
from kumoapi.task import TaskType
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.explain.local_sensitivity import LocalSensitivity
from kumo_core.explain.walk_attribution import WalkAttributionConfig
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import MSELoss
from kumo.engine.evidence import EvidenceConfig
from kumo.engine.evidence.walks import WalkAttributionBundle, attribute_bundle
from kumo.engine.evidence.bundle import EvidenceValidationError


def model():
    m = torch.nn.Module()
    m.entity_table_name = 'product'
    m.train_table_name = 'task'
    m.device = torch.device('cpu')
    m.encoder = torch.nn.Identity()
    m.task_type = TaskType.REGRESSION
    m.loss = MSELoss(TaskType.REGRESSION)
    m.target_encoder = NumericalTargetEncoder(0, 1)
    m.model = HeteroGNN(1, 1, metadata=(['product'], []), aggr='sum', norm=None,
                        bias=True)
    m.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[])
    m.eval()
    with torch.no_grad():
        m.model.root_skips[0].lin_dict['product'].weight.fill_(2)
        m.model.root_skips[0].lin_dict['product'].bias.fill_(1)
        m.head.mlp.lins[0].weight.fill_(3)
        m.head.mlp.lins[0].bias.fill_(4)
    m.common_step = lambda _: m.head(
        m.model({'product': torch.tensor([[2.]])}, {})['product'])
    return m


def test_original_scoring_capture_matches_offline_and_keeps_predictions(
        tmp_path, monkeypatch):
    m = model()
    config = WalkAttributionConfig()
    sample = {
        'nodes': {
            'product': {
                'pkeys': ['p17']
            }
        },
        'origin': 'original_forward'
    }
    batch = HeteroData()
    batch.input_type = 'product'
    batch['product'].batch_size = 1
    batch['product'].input_id = torch.tensor([8])
    collector = LocalSensitivity(str(tmp_path / 'capture'), {
        'run_id': 'r',
        'sampling_plan': {}
    }, numerical_sensitivity=False, walk_attribution=vars(config))
    collector.explain_start(m, SimpleNamespace(base_seed=7, sampling_plan={}),
                            None)
    monkeypatch.setattr(collector, '_sample', lambda *args: sample)
    collector.explain_pre_step(batch, m, None, None, 0)
    logits = m.common_step(batch)
    out = {'logits': logits, 'pkey': np.array(['p17']), 'time': None}
    collector.explain_post_step(batch, m, None, out, None, 0)
    assert logits.item() == 19
    identifier = out['prediction_id'][0]
    record = json.loads(
        (tmp_path / 'capture/rows' / f'{identifier}.json').read_text())
    original = record['walk_attribution']
    assert original['origin'] == 'original_forward_attribution'
    assert original['status'] == 'available'
    assert original['walks'][0]['source_pkey'] == 'p17'
    assert all(not x._forward_hooks for x in m.modules())
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'manifest.json').write_text('{}')
    bundle = SimpleNamespace(
        root=parent, validate=lambda: None,
        manifest={'native_provenance': {
            'bundle_sha256': 'b' * 64
        }}, iter_predictions=lambda: iter([{}]),
        iter_numerical_evidence=lambda: iter([record]),
        scoring_graph=lambda _: sample)
    monkeypatch.setattr(
        'kumo_core.explain.local_sensitivity.load_scoring_module',
        lambda *args: m)
    monkeypatch.setattr('kumo_core.explain.perturbation.reconstruct_batch',
                        lambda _: batch)
    result = attribute_bundle(bundle, model='trusted', config=config,
                              output=tmp_path / 'walks')
    offline = next(result.iter_rows())
    assert offline['origin'] == 'retained_graph_recomputation'
    assert offline['walks'] == original['walks']
    assert result.validate()['rows'] == 1
    with (result.root / 'rows.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(EvidenceValidationError, match='Corrupt'):
        WalkAttributionBundle.load(result.root)


def test_walk_config_requires_graph():
    with pytest.raises(ValueError, match='scoring graphs'):
        EvidenceConfig(scoring_graph=False,
                       walk_attribution=WalkAttributionConfig())
    with pytest.raises(TypeError):
        EvidenceConfig(walk_attribution={})


@pytest.mark.parametrize(
    'fault', ['duplicate', 'counts', 'ledger', 'coordinate', 'completeness'])
def test_resigned_malformed_artifact_rejected(tmp_path, fault):
    from kumo.engine.evidence.bundle import _digest, _json
    row = {
        'prediction_id': 'a' * 64,
        'sample_sha256': 'b' * 64,
        'origin': 'retained_graph_recomputation',
        'status': 'available',
        'method': 'hetero_lrp_gamma_reference_v1',
        'config': vars(WalkAttributionConfig()),
        'gamma': 0.,
        'epsilon': 1e-9,
        'output_index': 0,
        'schema': 'walk_attribution/v1',
        'coordinate': 'encoded_model_output',
        'input_coordinate': 'encoded_node_features',
        'complete': True,
        'truncated': False,
        'score': 3.,
        'bias_relevance': 3.,
        'stabilizer_residual': 0.,
        'omitted_signed_relevance': 0.,
        'accounting_error': 0.,
        'walks': [],
        'expanded_routes': 0
    }
    if fault == 'ledger':
        row['bias_relevance'] = 2.
    if fault == 'coordinate':
        row['coordinate'] = 'decoded_model_output'
    if fault == 'completeness':
        row['complete'] = False
    rows = [row, row] if fault == 'duplicate' else [row]
    (tmp_path / 'rows.jsonl').write_bytes(b''.join(
        _json(r) + b'\n' for r in rows))
    manifest = {
        'schema': 'walk_attribution/v1',
        'config': vars(WalkAttributionConfig()),
        'parent_manifest_sha256': 'c' * 64,
        'model_bundle_sha256': 'd' * 64,
        'generation_closed': True,
        'rows': len(rows),
        'status_counts': {
            'available': 3 if fault == 'counts' else len(rows)
        },
        'rows_sha256': _digest(tmp_path / 'rows.jsonl')
    }
    (tmp_path / 'manifest.json').write_bytes(_json(manifest))
    (tmp_path / '_SUCCESS').write_text(_digest(tmp_path / 'manifest.json'))
    with pytest.raises(EvidenceValidationError):
        WalkAttributionBundle.load(tmp_path)


def test_unsupported_profile_fails_preflight_or_records_without_losing_prediction(
        tmp_path, monkeypatch):
    m = model()
    m.model.act = 'gelu'
    config = vars(WalkAttributionConfig())
    from kumo_core.explain.walk_attribution import UnsupportedWalkAttribution
    data = SimpleNamespace(base_seed=7, sampling_plan={})
    with pytest.raises(UnsupportedWalkAttribution):
        LocalSensitivity(str(tmp_path / 'raise'), {
            'run_id': 'r',
            'sampling_plan': {}
        }, numerical_sensitivity=False,
                         walk_attribution=config).explain_start(m, data, None)
    collector = LocalSensitivity(str(tmp_path / 'record'), {
        'run_id': 'r',
        'sampling_plan': {}
    }, numerical_sensitivity=False, walk_attribution=config,
                                 walk_on_unsupported='record')
    collector.explain_start(m, data, None)
    batch = HeteroData()
    batch.input_type = 'product'
    batch['product'].batch_size = 1
    batch['product'].input_id = torch.tensor([8])
    monkeypatch.setattr(
        collector, '_sample',
        lambda *args: {'nodes': {
            'product': {
                'pkeys': ['p17']
            }
        }})
    collector.explain_pre_step(batch, m, data, None, 0)
    out = {
        'logits': m.common_step(batch),
        'pkey': np.array(['p17']),
        'time': None
    }
    collector.explain_post_step(batch, m, data, out, None, 0)
    collector.explain_end(m, data, {}, None)
    assert out['logits'].item() == 19
    record = json.loads((collector.root / 'rows' /
                         f"{out['prediction_id'][0]}.json").read_text())
    assert record['walk_attribution']['status'] == 'unsupported'
    assert record['walk_attribution']['prediction_id'] == out['prediction_id'][
        0]
    assert not json.loads(
        (collector.root / 'manifest.json').read_text())['complete']
    assert all(not x._forward_hooks for x in m.modules())
