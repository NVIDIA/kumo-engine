# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary walk evidence: records, rows, rejections, invariance and binding."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.explain.walk_attribution import (
    METHOD_MODULES,
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
)
from kumo_core.explain.walk_graph import GraphCapture
from kumo_core.explain.walk_schema import validate_v2_row
from kumo_core.modeling.gnn.hetero import (
    HeteroGNN,
    HeteroLinear,
    PrimaryKeyToForeignKeyConv,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import BinaryCrossEntropyLoss
from kumoapi.task import TaskType
from torch_geometric.nn.aggr import MaxAggregation
from walk_binary_fixtures import (
    ART_REV,
    FK,
    b1_sample,
    binary_scoring_module,
    capture_record,
    comparable_row,
    derive_offline,
    emitted_frame,
    native_offline_parity,
)

from kumo.engine.evidence import (
    EvidenceValidationError,
    PredictionEvidenceBundle,
)
from kumo.engine.evidence.bundle import PredictionEvidenceSource

LEGACY = 'hetero_lrp_gamma_reference_v1'


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


@pytest.mark.parametrize('selected,sign', [(None, 1), (True, 1), (False, -1)])
def test_binary_record_and_row(tmp_path, selected, sign):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    config = WalkAttributionConfig(selected_class=selected, feature_vectors=True)
    captured = capture_record(module, tmp_path, config)
    record, manifest = captured.record, captured.manifest
    emitted = emitted_frame(module, captured.logits.detach())
    true_prob, false_prob = float(emitted.true_prob[0]), float(emitted.false_prob[0])
    # The record binds the emitted True_PROB, identified by the raw logit.
    assert record['prediction'] == true_prob
    assert record['output_coordinate'] == 'encoded_logit'
    assert record['prediction_coordinate'] == 'encoded_positive_class_probability'
    assert record['output']['binary']['logit_sign'] == 1  # record: encoded class 1
    row = record['walk_attribution']
    assert row['schema'] == 'walk_attribution/v2' and row['status'] == 'available'
    binary = row['output']['binary']
    assert binary['logit_sign'] == sign and row['score'] == sign * 9.0
    assert binary['encoded_positive_label'] is True and binary['target_encoder_reverse'] is True
    assert binary['emitted_encoded_positive_probability'] == true_prob
    assert binary['emitted_probability_selected'] == (true_prob if sign == 1 else false_prob)
    assert binary['calibration']['kind'] == 'platt'
    assert math.fsum(w['relevance'] for w in row['walks']) == pytest.approx(sign * 9.0)
    assert manifest['task_type'] == 'binary_classification'
    assert manifest['binary_output']['encoded_positive_label'] is True
    assert manifest['loss'] == 'BinaryCrossEntropyLoss' and manifest['target_transform'] is None
    assert set(manifest['numerics']) == {'precision_mode', 'mkldnn_matmul_fp32_precision',
                                         'num_threads', 'cpu_capability', 'torch_version'}
    assert no_hooks(module)


def test_reversed_and_unreversed_mappings_both_record_true_prob(tmp_path):
    for i, counts in enumerate((((True, 123), (False, 37)), ((False, 123), (True, 37)))):
        module = binary_scoring_module(counts=counts)
        captured = capture_record(module, tmp_path / str(i), WalkAttributionConfig())
        true_prob = float(emitted_frame(module, captured.logits.detach()).true_prob[0])
        assert captured.record['prediction'] == true_prob
        assert captured.record['walk_attribution']['output']['binary'][
            'target_encoder_reverse'] is (i == 0)


def test_records_without_walks_still_carry_the_output_block(tmp_path):
    module = binary_scoring_module()
    captured = capture_record(module, tmp_path, None)
    assert 'walk_attribution' not in captured.record
    assert captured.record['output']['binary']['calibration']['kind'] == 'none'
    assert captured.manifest['complete'] is True


def test_numerical_sensitivity_is_regression_only(tmp_path):
    with pytest.raises(ValueError, match='binary evidence requires it off'):
        capture_record(binary_scoring_module(), tmp_path, WalkAttributionConfig(),
                       numerical_sensitivity=True)


def _class(instance, base):
    instance.__class__ = type(f'Other{base.__name__}', (base, ), {})


def _post_operator(module):
    module.model.post_convs[0].layer_dict['customer'][0] = torch.nn.GELU()


def _aggregator(module):
    module.model.hetero_convs[0][FK].aggr = MaxAggregation()


# Section 7 reject matrix: each mutation is applied to a working binary module.
REJECTIONS = {
    'model_type': lambda m: _class(m.model, HeteroGNN),  # also torch.compile wrappers
    'head_type': lambda m: _class(m.head, NodePredHead),
    'prediction_time_encodings': lambda m: setattr(m.head, 'prediction_time_encodings', True),
    'training_mode': lambda m: m.train(),
    'gelu_activation': lambda m: setattr(m.model, 'act', 'gelu'),
    'elu_head_activation': lambda m: setattr(m.head, 'act', 'elu'),
    'leaky_relu_activation': lambda m: setattr(m.model, 'act', 'leaky_relu'),
    'batch_norm': lambda m: setattr(m.model, 'norm', 'batch_norm'),
    'layer_norm_head': lambda m: setattr(m.head, 'norm', 'layer_norm'),
    'unregistered_post_operator': _post_operator,
    'unregistered_root_operator': lambda m: _class(m.model.root_skips[0], HeteroLinear),
    'unregistered_relation_operator': lambda m: _class(m.model.hetero_convs[0][ART_REV],
                                                       PrimaryKeyToForeignKeyConv),
    'max_aggregation': lambda m: setattr(m.model, 'aggr', ['max']),
    'actual_aggregator_differs': _aggregator,
    'multi_aggregation': lambda m: setattr(m.model, 'aggr', ['sum', 'mean']),
    'single_label_mapping': lambda m: setattr(m, 'target_encoder', _single_label()),
    'two_head_outputs': lambda m: setattr(m.head, 'out_channels', 2),
}


def _single_label():
    from walk_binary_fixtures import target_encoder
    return target_encoder([[True, 10]])


@pytest.mark.parametrize('name', sorted(REJECTIONS))
def test_rejections_raise_before_hooks_or_record_without_losing_predictions(tmp_path, name):
    reference = binary_scoring_module()
    expected = float(emitted_frame(reference, reference.common_step().detach()).true_prob[0])
    module = binary_scoring_module()
    REJECTIONS[name](module)
    with pytest.raises(UnsupportedWalkAttribution):
        capture_record(module, tmp_path / 'raise', WalkAttributionConfig())
    assert no_hooks(module)
    module = binary_scoring_module()
    REJECTIONS[name](module)
    captured = capture_record(module, tmp_path / 'record', WalkAttributionConfig(),
                              on_unsupported='record')
    assert captured.record['walk_attribution']['status'] == 'unsupported'
    assert no_hooks(module)
    if name not in ('training_mode', 'unregistered_post_operator', 'two_head_outputs',
                    'actual_aggregator_differs'):
        # The prediction is unchanged by the refusal (where the mutation does
        # not change the forward itself).
        assert captured.record['prediction'] == expected
    assert captured.manifest['complete'] is False


def test_distribution_head_is_rejected_before_hooks(tmp_path):
    module = binary_scoring_module()
    module.head.param_keys = ['mu']
    with pytest.raises(UnsupportedWalkAttribution):
        capture_record(module, tmp_path, WalkAttributionConfig())
    assert no_hooks(module)


@pytest.mark.parametrize('loss,match', [('focal', 'BinaryCrossEntropyLoss')])
def test_task_level_rejections(tmp_path, loss, match):
    module = binary_scoring_module(loss=loss)
    with pytest.raises(UnsupportedWalkAttribution, match=match):
        capture_record(module, tmp_path / 'raise', WalkAttributionConfig())
    captured = capture_record(module, tmp_path / 'record', WalkAttributionConfig(),
                              on_unsupported='record')
    assert captured.record['walk_attribution']['status'] == 'unsupported'


@pytest.mark.parametrize('selected', [1, 'True', 0])
def test_invalid_selected_class(tmp_path, selected):
    module = binary_scoring_module()
    config = WalkAttributionConfig(selected_class=selected)
    with pytest.raises(UnsupportedWalkAttribution, match='not a label'):
        capture_record(module, tmp_path / 'raise', config)
    captured = capture_record(module, tmp_path / 'record', config, on_unsupported='record')
    row = captured.record['walk_attribution']
    assert row['status'] == 'unsupported' and 'not a label' in row['reason']
    # The record keeps its own default orientation and the emitted value.
    assert captured.record['output']['binary']['logit_sign'] == 1


def test_configuration_errors_and_budget_exhaustion(tmp_path):
    with pytest.raises(ValueError, match='no gamma rule'):
        WalkAttributionConfig(method='hetero_gnn_gi_degree_t_v1', gamma=0.25)
    with pytest.raises(ValueError, match='bags=True requires'):
        WalkAttributionConfig(bags=True)
    captured = capture_record(binary_scoring_module(), tmp_path,
                              WalkAttributionConfig(max_walks=6, top_k=2),
                              on_unsupported='record')
    row = captured.record['walk_attribution']
    assert row['status'] == 'incomplete' and row['coverage']['status'] == 'budget_exhausted'
    assert captured.manifest['complete'] is False


def test_multi_edge_gather_is_a_row_level_rejection(tmp_path):
    from walk_binary_fixtures import b1_forward
    module = binary_scoring_module()
    x = {'customer': torch.tensor([[1.]], dtype=torch.float32),
         'transactions': torch.tensor([[2.], [3.]]), 'article': torch.tensor([[4.], [5.]])}
    edges = {FK: torch.tensor([[0, 1], [0, 0]]), ('customer', 'rev_customer_id',
                                                  'transactions'): torch.tensor([[0, 0], [0, 1]]),
             ART_REV: torch.tensor([[0, 1], [0, 0]]),
             ('transactions', 'article_id', 'article'): torch.zeros(2, 0, dtype=torch.long)}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0], 'article': [0, 0, 2]}
    nse = {FK: [2, 0], ('customer', 'rev_customer_id', 'transactions'): [0, 2],
           ART_REV: [0, 2], ('transactions', 'article_id', 'article'): [0, 0]}
    module.common_step = lambda _=None: b1_forward(module, (x, edges, nodes, nse))
    sample = b1_sample()
    sample['nodes']['article'] = {'pkeys': [9, 10], 'node_ids': [90, 100], 'num_nodes': 2}
    for mode in ('raise', 'record'):
        captured = capture_record(module, tmp_path / mode, WalkAttributionConfig(),
                                  sample=sample, on_unsupported=mode)
        row = captured.record['walk_attribution']
        assert row['status'] == 'unsupported' and 'multiple incoming' in row['reason']


def test_every_method_refuses_or_explains_without_fallback(tmp_path):
    """Method-agnostic: placeholder and implemented methods alike refuse an
    out-of-profile module before hooks (raise) or record an `unsupported` row
    of their own (record); on a module in profile, the row is always the
    requested method's own. Predictions are never lost."""
    reference = binary_scoring_module()
    expected = float(emitted_frame(reference, reference.common_step().detach()).true_prob[0])
    for method in METHOD_MODULES:
        config = WalkAttributionConfig(method=method)
        module = binary_scoring_module()
        module.model.act = 'gelu'  # outside the shared operator profile
        with pytest.raises(UnsupportedWalkAttribution):
            capture_record(module, tmp_path / f'{method}-raise', config)
        assert no_hooks(module)
        captured = capture_record(module, tmp_path / f'{method}-record', config,
                                  on_unsupported='record')
        row = captured.record['walk_attribution']
        assert row['status'] == 'unsupported' and row['reason']
        assert row['method'] == method and row['config'] == config.to_dict()
        assert captured.record['prediction'] == expected and no_hooks(module)
        module = binary_scoring_module()
        captured = capture_record(module, tmp_path / f'{method}-in-profile', config,
                                  on_unsupported='record')
        row = captured.record['walk_attribution']
        assert row['method'] == method and row['config'] == config.to_dict()
        assert row['status'] in ('available', 'unsupported')
        if row['status'] == 'available':
            assert row['score'] == pytest.approx(9.0) and row['walks']
            validate_v2_row(row, captured.sample)
        assert captured.record['prediction'] == expected and no_hooks(module)


# Invariance ###################################################################

def test_capture_leaves_outputs_gradients_and_calibrator_unchanged(tmp_path):
    def run(context):
        module = binary_scoring_module(calibration=(0.8, -0.1), gnn_bias=True, head_bias=True)
        calibrator = (module.calibrator.weight_and_bias.detach().clone(),
                      module.calibrator.is_fitted)
        with context(module):
            logits = module.common_step()
            loss = module.loss(logits, torch.ones_like(logits))
        grads = torch.autograd.grad(loss, list(module.model.parameters())
                                    + list(module.head.parameters()), allow_unused=True)
        assert torch.equal(module.calibrator.weight_and_bias, calibrator[0])
        assert module.calibrator.is_fitted == calibrator[1]
        return logits.detach(), grads

    class Plain:
        def __init__(self, module):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    plain = run(Plain)
    typed = run(lambda m: GraphCapture(m, WalkAttributionConfig()))
    assert torch.equal(plain[0], typed[0])
    assert all((a is None and b is None) or torch.equal(a, b)
               for a, b in zip(plain[1], typed[1]))
    # Evidence capture through LocalSensitivity leaves the calibrator as is.
    module = binary_scoring_module(calibration=(0.8, -0.1))
    before = module.calibrator.weight_and_bias.detach().clone()
    capture_record(module, tmp_path, WalkAttributionConfig(feature_vectors=True))
    assert torch.equal(module.calibrator.weight_and_bias, before)
    assert module.calibrator.is_fitted


def test_routes_and_walk_keys_ignore_edge_type_insertion_order(tmp_path):
    from walk_binary_fixtures import ART_FK, REV
    rows = []
    for i, order in enumerate((None, [ART_FK, ART_REV, REV, FK])):
        module = binary_scoring_module(batch_order=order)
        captured = capture_record(module, tmp_path / str(i),
                                  WalkAttributionConfig(feature_vectors=True))
        rows.append(captured.record['walk_attribution'])
    assert rows[0]['walks'] == rows[1]['walks'] and rows[0]['ledger'] == rows[1]['ledger']


# Offline replay ###############################################################

@pytest.mark.parametrize('selected', [True, False])
def test_offline_replay_matches_the_native_row(tmp_path, selected):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    config = WalkAttributionConfig(selected_class=selected, feature_vectors=True, gamma=0.25)
    native, offline, bundle = native_offline_parity(module, config, tmp_path)
    assert comparable_row(native) == comparable_row(offline)
    assert offline['replay_logit_abs_diff'] == 0.0
    assert offline['replay_probability_abs_diff'] == 0.0
    assert bundle.validate()['status_counts'] == {'available': 1}


@pytest.mark.parametrize('schema_options', [{}, {'feature_vectors': True}])
def test_replay_fails_on_a_tampered_graph_calibrator_or_model(tmp_path, schema_options):
    config = WalkAttributionConfig(**schema_options)
    module = binary_scoring_module(calibration=(0.8, -0.1))
    captured = capture_record(module, tmp_path / 'native', config)
    assert list(derive_offline(captured, module, config, tmp_path / 'intact').iter_rows())[
        0]['status'] == 'available'
    # The retained graph changed: one transaction feature differs.
    setattr(captured.batch, 'b1_x', {'customer': [[1.]], 'transactions': [[2.], [3.5]],
                                     'article': [[4.]]})
    rows = list(derive_offline(captured, module, config, tmp_path / 'graph').iter_rows())
    assert rows[0]['status'] == 'failed' and 'does not reproduce' in rows[0]['reason']
    delattr(captured.batch, 'b1_x')
    recalibrated = binary_scoring_module(calibration=(0.9, -0.1))
    rows = list(derive_offline(captured, recalibrated, config,
                               tmp_path / 'calibrator').iter_rows())
    assert rows[0]['status'] == 'failed' and 'does not reproduce' in rows[0]['reason']
    reweighted = binary_scoring_module(calibration=(0.8, -0.1), head_weight=1.5)
    rows = list(derive_offline(captured, reweighted, config, tmp_path / 'model').iter_rows())
    assert rows[0]['status'] == 'failed' and 'does not reproduce' in rows[0]['reason']


# Bundle validation ############################################################

def _bundle(tmp_path, config=WalkAttributionConfig(feature_vectors=True), tamper=None):
    module = binary_scoring_module(calibration=(0.8, -0.1))
    run_root = tmp_path / 'run'
    captured = capture_record(module, run_root / 'xai' / 'local', config)
    emitted = emitted_frame(module, captured.logits.detach())
    frame = pd.DataFrame({
        'PREDICTION_ID': [captured.record['prediction_id']], 'ENTITY': [17],
        'TASK_ROW_ID': [8], 'TARGET_PRED': pd.array([None], dtype='boolean'),
        'False_PROB': np.array([float(emitted.false_prob[0])], dtype=np.float32),
        'True_PROB': np.array([float(emitted.true_prob[0])], dtype=np.float32)})
    if tamper is not None:
        tamper(frame)
    frame.to_parquet(tmp_path / 'predictions.parquet')
    return PredictionEvidenceSource((str(tmp_path / 'predictions.parquet'), ), run_root,
                                    {}).export(tmp_path / 'bundle')


def test_bundle_binds_true_prob_bitwise(tmp_path):
    bundle = _bundle(tmp_path)
    report = PredictionEvidenceBundle.load(bundle.root).validate()
    assert report['evidence_rows'] == 1 and report['valid']


@pytest.mark.parametrize('tamper,match', [
    (lambda f: f.__setitem__('True_PROB', np.nextafter(f['True_PROB'].to_numpy(),
                                                       np.float32(1)).astype(np.float32)),
     'not bound'),
    (lambda f: f.__setitem__('False_PROB', np.array([0.5], dtype=np.float32)), 'sum to one'),
    (lambda f: f.__setitem__('TARGET_PRED', np.array([1.0])), 'null or bool'),
    (lambda f: f.__setitem__('True_PROB', np.array([np.nan], dtype=np.float32)),
     'Invalid emitted'),
])
def test_bundle_rejects_unbound_binary_predictions(tmp_path, tamper, match):
    with pytest.raises(EvidenceValidationError, match=match):
        _bundle(tmp_path, tamper=tamper)


def test_bundle_rejects_a_walk_row_bound_to_another_probability(tmp_path):
    from kumo.engine.evidence.bundle import _json, _read, _seal
    bundle = _bundle(tmp_path)
    rows = list((bundle.root / 'xai/local/rows').glob('*.json'))
    record = json.loads(rows[0].read_text())
    record['walk_attribution']['output']['binary']['emitted_encoded_positive_probability'] = 0.5
    rows[0].write_bytes(_json(record))
    manifest = _read(bundle.root / 'manifest.json')
    _seal(bundle.root, manifest)
    with pytest.raises(EvidenceValidationError):
        PredictionEvidenceBundle.load(bundle.root)


def test_task_type_mismatch_is_not_explained(tmp_path):
    module = binary_scoring_module()
    module.task_type = TaskType.MULTICLASS_CLASSIFICATION
    with pytest.raises(ValueError, match='scalar MSE regression'):
        capture_record(module, tmp_path, WalkAttributionConfig())
    module = binary_scoring_module()
    assert isinstance(module.loss, BinaryCrossEntropyLoss)


def test_eval_mode_dropout_and_identity_activation_are_allowed(tmp_path):
    # Section 7: dropout modules in evaluation mode and identity activations
    # are allowed for LRP rows.
    module = binary_scoring_module()
    for post in module.model.post_convs:
        for layers in post.layer_dict.values():
            layers[-1].p = 0.5
    captured = capture_record(module, tmp_path / 'dropout', WalkAttributionConfig())
    assert captured.record['walk_attribution']['status'] == 'available'
    identity = binary_scoring_module()
    identity.model.act = None  # as HeteroGNN(act=None): no activation module
    for post in identity.model.post_convs:
        for layers in post.layer_dict.values():
            del layers[0]
    captured = capture_record(identity, tmp_path / 'identity', WalkAttributionConfig())
    assert captured.record['walk_attribution']['status'] == 'available'
