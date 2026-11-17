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


# v1 byte identity (contract sections 4.1 and 12) ##########################
# Uses only APIs that exist at 1909c5b9. `_V1_EVIDENCE_GOLDEN` was produced by
# running `v1_evidence_golden_artifacts` under the installed 1909c5b9 wheel
# without the source overlay (ledger/pods/P2-binary/golden/generate.py).
# The native local manifest is compared without its additive `numerics` key
# (contract C4); every other byte is frozen.

_LEGACY_CONFIG = {'gamma': 0.0, 'epsilon': 1e-9, 'max_walks': 10000,
                  'top_k': 20, 'output_index': 0}


def _golden_capture(root, walk_attribution, *, unsupported=False,
                    numerical_sensitivity=False):
    from typing import Any
    m: Any = model()
    if unsupported:
        setattr(m.model, 'act', 'gelu')
    sample = {
        'nodes': {
            'product': {
                'pkeys': ['p17']
            }
        },
        'origin': 'original_forward'
    }
    batch = HeteroData()
    setattr(batch, 'input_type', 'product')
    batch['product'].batch_size = 1
    batch['product'].input_id = torch.tensor([8])
    collector = LocalSensitivity(
        str(root), {
            'run_id': 'r',
            'sampling_plan': {}
        }, numerical_sensitivity=numerical_sensitivity,
        walk_attribution=walk_attribution,
        walk_on_unsupported='record' if unsupported else 'raise')
    data = SimpleNamespace(base_seed=7, sampling_plan={})
    collector.explain_start(m, data, None)
    setattr(collector, '_sample', lambda *args: sample)
    collector.explain_pre_step(batch, m, data, None, 0)
    out = {'logits': m.common_step(batch), 'pkey': np.array(['p17']),
           'time': None}
    collector.explain_post_step(batch, m, data, out, None, 0)
    collector.explain_end(m, data, {}, None)
    identifier = out['prediction_id'][0]
    return m, batch, sample, identifier


def v1_evidence_golden_artifacts(tmp):
    """Return {name: exact text} for frozen native and offline v1 bytes."""
    from pathlib import Path

    import kumo_core.explain.local_sensitivity as local
    import kumo_core.explain.perturbation as perturbation
    tmp = Path(tmp)
    out = {}
    cases = (('native', dict(_LEGACY_CONFIG), {}),
             ('native_no_walk', None, {}),
             ('native_numerical', dict(_LEGACY_CONFIG),
              {'numerical_sensitivity': True}),
             ('native_unsupported', dict(_LEGACY_CONFIG),
              {'unsupported': True}))
    for name, walk, options in cases:
        root = tmp / name
        m, batch, sample, identifier = _golden_capture(root, walk, **options)
        out[f'{name}_row'] = (root / 'rows' / f'{identifier}.json').read_text()
        out[f'{name}_index'] = (root / 'index.jsonl').read_text()
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.pop('numerics', None)
        out[f'{name}_manifest_without_numerics'] = local._encoded(manifest)
    m, batch, sample, identifier = _golden_capture(tmp / 'replay-source',
                                                   dict(_LEGACY_CONFIG))
    record = json.loads((tmp / 'replay-source' / 'rows' /
                         f'{identifier}.json').read_text())
    parent = tmp / 'parent'
    parent.mkdir()
    (parent / 'manifest.json').write_text('{}')
    bundle = SimpleNamespace(
        root=parent, validate=lambda: None,
        manifest={'native_provenance': {
            'bundle_sha256': 'b' * 64
        }}, iter_predictions=lambda: iter([{}]),
        iter_numerical_evidence=lambda: iter([record]),
        scoring_graph=lambda _: sample)
    saved = local.load_scoring_module, perturbation.reconstruct_batch
    local.load_scoring_module = lambda *args: m
    perturbation.reconstruct_batch = lambda _: batch
    try:
        for name, config in (('offline', WalkAttributionConfig()),
                             ('offline_gamma_025_top_k_1',
                              WalkAttributionConfig(gamma=0.25, top_k=1)),
                             ('offline_max_walks_1',
                              WalkAttributionConfig(max_walks=1))):
            result = attribute_bundle(bundle, model='trusted', config=config,
                                      output=tmp / name)
            out[f'{name}_rows'] = (result.root / 'rows.jsonl').read_text()
            out[f'{name}_manifest'] = (result.root /
                                       'manifest.json').read_text()
            out[f'{name}_success'] = (result.root / '_SUCCESS').read_text()
    finally:
        local.load_scoring_module, perturbation.reconstruct_batch = saved
    # Default training-spec.json bytes (contract C2: unchanged by bias
    # options, which are written only when non-default).
    from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan

    from kumo.engine.trainer._spec import MANIFEST_FILE, write_training_spec
    spec_bundle = tmp / 'spec-bundle'
    spec_bundle.mkdir()
    (spec_bundle / MANIFEST_FILE).write_text(json.dumps({'artifacts': []}))
    write_training_spec(spec_bundle,
                        predictive_query='PREDICT users.churn FOR EACH users.id',
                        model_plan=ModelPlan(), snapshot_id='snap-1',
                        training_table_generation=TrainingTableGenerationPlan(),
                        graph_config={'tables': {}})
    out['training_spec_default'] = (spec_bundle /
                                    'training-spec.json').read_text()
    return out


_V1_EVIDENCE_GOLDEN = {
    'native_index':
    '{"available_cells":0,"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null}\n',
    'native_manifest_without_numerics':
    '{"attempted_rows":1,"batch_size":1,"complete":true,"device":"cpu","failure":null,"generation_closed":true,"graph_failures":0,"input_coordinate":"transformed_numerical_input","loss":"MSELoss","method":"absolute_gradient_decoded_output_wrt_transformed_numerical_input","numerical_sensitivity":false,"origin":"original_forward","output_coordinate":"decoded_model_output","raw_value_coordinate":"numerical_tensor_mapper_output","rows":1,"run_id":"r","sampler_seed":7,"sampling_plan":{},"schema_version":1,"signed_method":"gradient_decoded_output_wrt_transformed_numerical_input","status_counts":{"disabled":1},"target_transform":{"log_transform":false,"offset":0,"scale":1},"temporal_boundary":{"coordinate":"unix_nanoseconds_from_native_sampler","sampler":"at_or_before_seed_time","source_time_columns":{},"strict_validation":"source_time_strictly_before_prediction_cutoff"},"temporal_status_counts":{"not_applicable":1},"walk_status_counts":{"available":1}}',
    'native_no_walk_index':
    '{"available_cells":0,"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null}\n',
    'native_no_walk_manifest_without_numerics':
    '{"attempted_rows":1,"batch_size":1,"complete":true,"device":"cpu","failure":null,"generation_closed":true,"graph_failures":0,"input_coordinate":"transformed_numerical_input","loss":"MSELoss","method":"absolute_gradient_decoded_output_wrt_transformed_numerical_input","numerical_sensitivity":false,"origin":"original_forward","output_coordinate":"decoded_model_output","raw_value_coordinate":"numerical_tensor_mapper_output","rows":1,"run_id":"r","sampler_seed":7,"sampling_plan":{},"schema_version":1,"signed_method":"gradient_decoded_output_wrt_transformed_numerical_input","status_counts":{"disabled":1},"target_transform":{"log_transform":false,"offset":0,"scale":1},"temporal_boundary":{"coordinate":"unix_nanoseconds_from_native_sampler","sampler":"at_or_before_seed_time","source_time_columns":{},"strict_validation":"source_time_strictly_before_prediction_cutoff"},"temporal_status_counts":{"not_applicable":1},"walk_status_counts":{}}',
    'native_no_walk_row':
    '{"available_cells":0,"cells":[],"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null}',
    'native_numerical_index':
    '{"entity":"p17","graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"AttributeError: local sensitivity capture failed","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"failed","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null}\n',
    'native_numerical_manifest_without_numerics':
    '{"attempted_rows":1,"batch_size":1,"complete":false,"device":"cpu","failure":null,"generation_closed":true,"graph_failures":0,"input_coordinate":"transformed_numerical_input","loss":"MSELoss","method":"absolute_gradient_decoded_output_wrt_transformed_numerical_input","numerical_sensitivity":true,"origin":"original_forward","output_coordinate":"decoded_model_output","raw_value_coordinate":"numerical_tensor_mapper_output","rows":1,"run_id":"r","sampler_seed":7,"sampling_plan":{},"schema_version":1,"signed_method":"gradient_decoded_output_wrt_transformed_numerical_input","status_counts":{"failed":1},"target_transform":{"log_transform":false,"offset":0,"scale":1},"temporal_boundary":{"coordinate":"unix_nanoseconds_from_native_sampler","sampler":"at_or_before_seed_time","source_time_columns":{},"strict_validation":"source_time_strictly_before_prediction_cutoff"},"temporal_status_counts":{"not_applicable":1},"walk_status_counts":{"available":1}}',
    'native_numerical_row':
    '{"cells":[],"entity":"p17","graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"AttributeError: local sensitivity capture failed","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"failed","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null,"walk_attribution":{"accounting_error":0.0,"bias_relevance":7.0,"complete":true,"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":10000,"output_index":0,"top_k":20},"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"origin":"original_forward_attribution","output_index":0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":0.0,"status":"available","truncated":false,"walks":[{"relevance":12.0,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}}',
    'native_row':
    '{"available_cells":0,"cells":[],"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null,"walk_attribution":{"accounting_error":0.0,"bias_relevance":7.0,"complete":true,"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":10000,"output_index":0,"top_k":20},"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"origin":"original_forward_attribution","output_index":0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":0.0,"status":"available","truncated":false,"walks":[{"relevance":12.0,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}}',
    'native_unsupported_index':
    '{"available_cells":0,"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null}\n',
    'native_unsupported_manifest_without_numerics':
    '{"attempted_rows":1,"batch_size":1,"complete":false,"device":"cpu","failure":null,"generation_closed":true,"graph_failures":0,"input_coordinate":"transformed_numerical_input","loss":"MSELoss","method":"absolute_gradient_decoded_output_wrt_transformed_numerical_input","numerical_sensitivity":false,"origin":"original_forward","output_coordinate":"decoded_model_output","raw_value_coordinate":"numerical_tensor_mapper_output","rows":1,"run_id":"r","sampler_seed":7,"sampling_plan":{},"schema_version":1,"signed_method":"gradient_decoded_output_wrt_transformed_numerical_input","status_counts":{"disabled":1},"target_transform":{"log_transform":false,"offset":0,"scale":1},"temporal_boundary":{"coordinate":"unix_nanoseconds_from_native_sampler","sampler":"at_or_before_seed_time","source_time_columns":{},"strict_validation":"source_time_strictly_before_prediction_cutoff"},"temporal_status_counts":{"not_applicable":1},"walk_status_counts":{"unsupported":1}}',
    'native_unsupported_row':
    '{"available_cells":0,"cells":[],"entity":"p17","expected_cells":0,"graph_reason":null,"graph_status":"available","origin":"original_forward","output_coordinate":"scalar","prediction":19.0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"numerical_sensitivity_disabled","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"disabled","task_row_id":[8],"temporal_validation":{"reason":"static_task","status":"not_applicable"},"time_ns":null,"walk_attribution":{"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":10000,"output_index":0,"top_k":20},"method":"hetero_lrp_gamma_reference_v1","origin":"original_forward_attribution","prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","reason":"Reference profile requires ReLU/identity and no normalization","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","status":"unsupported"}}',
    'offline_gamma_025_top_k_1_manifest':
    '{"config":{"epsilon":1e-09,"gamma":0.25,"max_walks":10000,"output_index":0,"top_k":1},"generation_closed":true,"model_bundle_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","parent_manifest_sha256":"44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","rows":1,"rows_sha256":"46a15311518f834ad3822122269748355ad6becd037473141cc1aeb8cfe01f77","schema":"walk_attribution/v1","status_counts":{"available":1}}',
    'offline_gamma_025_top_k_1_rows':
    '{"accounting_error":0.0,"bias_relevance":7.0,"complete":true,"config":{"epsilon":1e-09,"gamma":0.25,"max_walks":10000,"output_index":0,"top_k":1},"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.25,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"origin":"retained_graph_recomputation","output_index":0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":0.0,"status":"available","truncated":false,"walks":[{"relevance":12.0,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}\n',
    'offline_gamma_025_top_k_1_success':
    'b6354daaae90ed03fdc2ca2201b0306d2b6b3d0ac9b02a00dd2861d9d982e3db',
    'offline_manifest':
    '{"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":10000,"output_index":0,"top_k":20},"generation_closed":true,"model_bundle_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","parent_manifest_sha256":"44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","rows":1,"rows_sha256":"05bc684e7bdb9c398526038b23dcc7c07758ca81e676db5854289439e71cf49c","schema":"walk_attribution/v1","status_counts":{"available":1}}',
    'offline_max_walks_1_manifest':
    '{"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":1,"output_index":0,"top_k":20},"generation_closed":true,"model_bundle_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","parent_manifest_sha256":"44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","rows":1,"rows_sha256":"35c14d4f2e85a592b1a7c87849f3a11170683ba61ea0f75da53a0aa575845543","schema":"walk_attribution/v1","status_counts":{"truncated":1}}',
    'offline_max_walks_1_rows':
    '{"accounting_error":0.0,"bias_relevance":4.0,"complete":false,"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":1,"output_index":0,"top_k":20},"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":1,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":15.0,"origin":"retained_graph_recomputation","output_index":0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":0.0,"status":"truncated","truncated":true,"walks":[]}\n',
    'offline_max_walks_1_success':
    '12f13c3492d5d69fc0b71af922c83c05bf8f96d10b32f5f81717171ab08101b5',
    'offline_rows':
    '{"accounting_error":0.0,"bias_relevance":7.0,"complete":true,"config":{"epsilon":1e-09,"gamma":0.0,"max_walks":10000,"output_index":0,"top_k":20},"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"origin":"retained_graph_recomputation","output_index":0,"prediction_id":"763d430b6fec7e432c5e398f537ad20548202cd898be3b90b74df96950f4984c","sample_sha256":"d1a961f185a7566aa44d0ad8228c5bbcc761c947aa103869bd1895d9548ebcea","schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":0.0,"status":"available","truncated":false,"walks":[{"relevance":12.0,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}\n',
    'offline_success':
    '927650dac2aac792545b38a15c8d5d5804180e8184281e2ba58830d15fcb6676',
    'training_spec_default':
    '{\n  "graph_config": {\n    "tables": {}\n  },\n  "model_plan": {\n    "column_processing": {\n      "encoder_overrides": null,\n      "na_strategy": null\n    },\n    "model_architecture": {\n      "activation": "???",\n      "aggregation": "???",\n      "channels": "???",\n      "distance_measure": "???",\n      "embedding_norm_reg_coeff": "???",\n      "forecast_type": "???",\n      "handle_new_entities": "???",\n      "handle_new_target_entities": "???",\n      "model": "???",\n      "module": "???",\n      "normalization": "???",\n      "num_post_message_passing_layers": "???",\n      "num_pre_message_passing_layers": "???",\n      "output_embedding_dim": "???",\n      "past_encoder": "???",\n      "prediction_time_encodings": "???",\n      "ranking_embedding_loss_coeff": "???",\n      "target_embedding_mode": "???",\n      "use_seq_id": "???"\n    },\n    "neighbor_sampling": {\n      "adaptive_sampling": false,\n      "max_target_neighbors_per_entity": "???",\n      "num_neighbors": "???",\n      "sample_from_entity_table": "???"\n    },\n    "optimization": {\n      "base_lr": "???",\n      "batch_size": "???",\n      "early_stopping": "???",\n      "lookup_optimization": "???",\n      "loss": "???",\n      "lr_scheduler": "???",\n      "majority_sampling_ratio": "???",\n      "max_epochs": "???",\n      "max_steps_per_epoch": "???",\n      "max_test_steps": "???",\n      "max_val_steps": "???",\n      "min_steps_per_epoch": "???",\n      "optimizer": "???",\n      "overlap_ratio": "???",\n      "target_fraction": "???",\n      "weight_decay": "???",\n      "weight_mode": "???"\n    },\n    "training_job": {\n      "disable_compilation": true,\n      "disable_explain": false,\n      "manual_seed": null,\n      "metrics": "???",\n      "num_experiments": "???",\n      "pruning": "???",\n      "refit_full": false,\n      "refit_trainval": true,\n      "tune_metric": "???"\n    }\n  },\n  "predictive_query": "PREDICT users.churn FOR EACH users.id",\n  "snapshot_id": "snap-1",\n  "spec_version": 1,\n  "timestamp_precision": "seconds",\n  "training_table_generation": {\n    "end_time": "???",\n    "entity_candidate": null,\n    "entity_candidate_aggregation": null,\n    "forecast_length": "???",\n    "lag_timesteps": "???",\n    "split": "inferred",\n    "start_time": "???",\n    "task_path_override": null,\n    "timeframe_step": "???",\n    "train_end_offset": "???",\n    "train_start_offset": "???",\n    "train_table_path_override": null,\n    "weight_col": null,\n    "year_over_year": "???"\n  }\n}\n',
}


def test_v1_evidence_bytes_identical_to_1909c5b9(tmp_path):
    regenerated = v1_evidence_golden_artifacts(tmp_path)
    assert sorted(regenerated) == sorted(_V1_EVIDENCE_GOLDEN)
    for name, text in _V1_EVIDENCE_GOLDEN.items():
        assert regenerated[name] == text, name


# Legacy records, regression v2 and the replay precision record (C4, C9) ####

def test_vars_config_still_writes_legacy_record_bytes(tmp_path):
    # Callers that pass vars(config) (now nine keys) keep v1 record bytes.
    _, _, _, identifier = _golden_capture(tmp_path / 'vars', vars(WalkAttributionConfig()))
    text = (tmp_path / 'vars' / 'rows' / f'{identifier}.json').read_text()
    assert text == _V1_EVIDENCE_GOLDEN['native_row']
    assert 'output' not in json.loads(text) and 'numerics' not in json.loads(text)


def test_regression_v2_rows_match_offline_derivation(tmp_path):
    from walk_binary_fixtures import comparable_row, native_offline_parity
    config = WalkAttributionConfig(feature_vectors=True)
    native, offline, bundle = native_offline_parity(model(), config, tmp_path,
                                                    sample={'nodes': {'product': {
                                                        'pkeys': ['p17']}},
                                                            'origin': 'original_forward'},
                                                    pkey='p17')
    assert native['schema'] == offline['schema'] == 'walk_attribution/v2'
    assert native['status'] == offline['status'] == 'available'
    assert comparable_row(native) == comparable_row(offline)
    assert native['score'] == 19.0 and native['ledger']['bias_relevance'] == pytest.approx(7)
    assert native['sources'][0]['input_relevance'] == pytest.approx([12.0])
    assert offline['numerics']['replay_precision_mode'] is not None
    assert bundle.validate()['schema'] == 'walk_attribution/v2'


def _medium_differs():
    from kumo_core.explain.walk_graph import float32_matmul_precision
    from walk_binary_fixtures import precision_module
    module = precision_module()
    with torch.no_grad(), float32_matmul_precision('medium'):
        medium = module.common_step()
    with torch.no_grad(), float32_matmul_precision('highest'):
        highest = module.common_step()
    return not torch.equal(medium, highest)


def _precision_capture(root, config):
    from kumo_core.explain.walk_graph import float32_matmul_precision
    from walk_binary_fixtures import (
        capture_record,
        precision_module,
        precision_sample,
    )
    module = precision_module()
    with float32_matmul_precision('medium'):
        captured = capture_record(module, root, config, sample=precision_sample(), pkey=1)
    return module, captured


def test_replay_runs_under_the_recorded_mode_and_restores_it(tmp_path):
    from walk_binary_fixtures import derive_offline
    config = WalkAttributionConfig(top_k=1000)
    module, captured = _precision_capture(tmp_path / 'native', config)
    assert captured.manifest['numerics']['precision_mode'] == 'medium'
    observed, step = [], module.common_step

    def watched(batch=None):
        observed.append(torch.get_float32_matmul_precision())
        return step(batch)

    module.common_step = watched
    before = torch.get_float32_matmul_precision()
    rows = list(derive_offline(captured, module, config, tmp_path / 'recorded').iter_rows())
    assert set(observed) == {'medium'} and rows[0]['status'] == 'available'
    assert torch.get_float32_matmul_precision() == before
    observed.clear()
    derive_offline(captured, module, config, tmp_path / 'unrecorded', numerics=None)
    assert set(observed) == {before}  # bundles from before the record: unchanged


@pytest.mark.parametrize('options', [{}, {'feature_vectors': True}])
def test_mkldnn_mismatch_fails_with_replay_precision_unavailable(tmp_path, options):
    from walk_binary_fixtures import derive_offline
    config = WalkAttributionConfig(top_k=1000, **options)
    module, captured = _precision_capture(tmp_path / 'native', config)
    numerics = {**captured.manifest['numerics'], 'mkldnn_matmul_fp32_precision': 'tf32'}
    rows = list(derive_offline(captured, module, config, tmp_path / 'offline',
                               numerics=numerics).iter_rows())
    assert [(r['status'], r['reason']) for r in rows] == [('failed',
                                                           'replay_precision_unavailable')]


REPLAY_SCRIPT = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import torch
from walk_binary_fixtures import derive_offline, precision_module, precision_sample, seed_batch
from kumo_core.explain.walk_attribution import WalkAttributionConfig
payload = json.loads(Path(sys.argv[2]).read_text())
captured = SimpleNamespace(record=payload['record'], manifest=payload['manifest'],
                           sample=precision_sample(), batch=seed_batch('entities'))
before = torch.get_float32_matmul_precision()
out = derive_offline(captured, precision_module(), WalkAttributionConfig(**payload['config']),
                     Path(sys.argv[3]), numerics=payload['numerics'])
rows = list(out.iter_rows())
print(json.dumps({'before': before, 'after': torch.get_float32_matmul_precision(),
                  'statuses': [r['status'] for r in rows],
                  'reasons': [r.get('reason') for r in rows]}))
'''


@pytest.mark.parametrize('options', [{}, {'feature_vectors': True}])
def test_fresh_process_replay_of_a_medium_capture(tmp_path, options):
    """C4: a fresh process starts at `highest`; replay must apply `medium`.

    The fixture's 32-channel linears see 12 rows, past this host's oneDNN
    threshold, so only the recorded mode reproduces the emitted prediction.
    Overriding the record, or a bundle without it (base behavior), fails.
    """
    import subprocess
    import sys
    from pathlib import Path
    config = WalkAttributionConfig(top_k=1000, **options)
    _, captured = _precision_capture(tmp_path / 'native', config)
    assert captured.record['walk_attribution']['status'] == 'available'
    numerics = captured.manifest['numerics']

    def replay(name, recorded):
        payload = tmp_path / f'{name}.json'
        payload.write_text(json.dumps({'record': captured.record,
                                       'manifest': captured.manifest,
                                       'config': config.to_dict(), 'numerics': recorded}))
        process = subprocess.run([sys.executable, '-c', REPLAY_SCRIPT,
                                  str(Path(__file__).parent), str(payload),
                                  str(tmp_path / name)], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr[-3000:]
        return json.loads(process.stdout.strip().splitlines()[-1])

    recorded = replay('recorded', 'recorded')
    assert recorded['before'] == recorded['after'] == 'highest'
    assert recorded['statuses'] == ['available'], recorded
    overridden = replay('overridden', {**numerics, 'precision_mode': 'highest',
                                       'mkldnn_matmul_fp32_precision': 'ieee'})
    unrecorded = replay('unrecorded', None)
    if _medium_differs():
        assert overridden['statuses'] == ['failed'], overridden
        assert unrecorded['statuses'] == ['failed'], unrecorded
    else:
        pytest.skip('medium and highest agree on this host; failure cases not observable')
