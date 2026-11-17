# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-gamma GNN-LRP evidence: native rows, offline replay and validation.

Native rows come from `LocalSensitivity` around one scoring forward and
offline rows from `attribute_bundle`, through P2's frozen fixture helpers
(nothing here patches P2 entry points). The binary fixture is P0's B1 with
GNN biases 0.5 and head bias -0.25 (z = 12.25), so hand values apply.
"""
import copy
import json
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.explain.walk_attribution import (
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
    inspect_walk_capability,
)
from kumo_core.explain.walk_binary import Selection, output_block
from kumo_core.explain.walk_graph import (
    STRUCTURAL_CODES,
    GraphCapture,
    float32_matmul_precision,
    mkldnn_matmul_fp32_precision,
)
from kumo_core.explain.walk_schema import (
    WalkSchemaError,
    explain_capture,
    row_numerics,
    validate_v2_row,
)
from kumo_core.modeling.gnn.hetero import (
    HeteroGNN,
    HeteroLinear,
    PrimaryKeyToForeignKeyConv,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead
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
    precision_module,
    precision_sample,
    target_encoder,
)

from kumo.engine.evidence import PredictionEvidenceBundle
from kumo.engine.evidence.bundle import PredictionEvidenceSource

WEIGHT = 'hetero_gnn_lrp_weight_gamma_v1'
LEGACY = 'hetero_lrp_gamma_reference_v1'
REV = ('customer', 'rev_customer_id', 'transactions')
SELF_C2 = (2, 'self_update', 'customer', '', 'customer', 0)


def config(**options):
    return WalkAttributionConfig(method=WEIGHT, **options)


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


def rel(t, relation, edge):
    return (t, 'relation', *relation, edge)


#: The eight B1 walks and their scores at gamma 0 without biases (sum 9).
B1_SCORES = {
    ((1, 'self_update', 'customer', '', 'customer', 0), SELF_C2): 1.0,
    (rel(1, FK, 0), SELF_C2): 1.0,
    (rel(1, FK, 1), SELF_C2): 1.5,
    ((1, 'self_update', 'transactions', '', 'transactions', 0), rel(2, FK, 0)): 1.0,
    (rel(1, REV, 0), rel(2, FK, 0)): 0.5,
    (rel(1, ART_REV, 0), rel(2, FK, 0)): 2.0,
    ((1, 'self_update', 'transactions', '', 'transactions', 1), rel(2, FK, 1)): 1.5,
    (rel(1, REV, 1), rel(2, FK, 1)): 0.5,
}


def walks_of(row):
    return {tuple((s['mp_layer'], 'relation', *s['relation'], s['edge_occurrence'])
                  if s['kind'] == 'relation' else
                  (s['mp_layer'], 'self_update', s['table'], '', s['table'], s['occurrence'])
                  for s in w['steps']): w['relevance'] for w in row['walks']}


def mass(row):
    ledger = row['ledger']
    return math.fsum([abs(w['relevance']) for w in row['walks']]
                     + [ledger['omitted_display_abs_relevance'],
                        ledger['omitted_search_abs_prefix_mass'],
                        abs(ledger['precision_residual'])]
                     + [abs(v) for v in ledger['bias_relevance_by_kind'].values()]
                     + [abs(v) for v in ledger['stabilizer_residual_by_kind'].values()])


def biased_module(**options):
    return binary_scoring_module(gnn_bias=True, head_bias=True, calibration=(0.8, -0.1),
                                 **options)


# Hand values on native rows ######################################################

@pytest.mark.parametrize('gamma,selected,factor,mp_bias,head_bias', [
    (0.0, True, Fraction(1), Fraction(7, 2), Fraction(-1, 4)),
    (0.25, True, Fraction(245, 246), Fraction(7, 2) * Fraction(245, 246),
     Fraction(-1, 4) * Fraction(98, 123)),
    (0.0, False, Fraction(-1), Fraction(-7, 2), Fraction(1, 4)),
    (0.25, False, Fraction(-196, 195), Fraction(-7, 2) * Fraction(196, 195),
     Fraction(5, 16) * Fraction(196, 195)),
])
def test_native_rows_carry_the_b1_bias_hand_values(tmp_path, gamma, selected, factor,
                                                   mp_bias, head_bias):
    """B1-bias: walks are a uniform factor times the B1 scores. Explaining the
    other class folds its sign into the head row, which at gamma 0.25 gives
    -196/195 rather than the negated -245/246 (fixture B3 at Engine level)."""
    module = biased_module()
    captured = capture_record(module, tmp_path, config(gamma=gamma, selected_class=selected,
                                                       feature_vectors=True))
    row = captured.record['walk_attribution']
    assert row['status'] == 'available', row.get('reason')
    assert row['score'] == (12.25 if selected else -12.25)
    walks = walks_of(row)
    assert set(walks) == set(B1_SCORES)
    for key, score in B1_SCORES.items():
        assert walks[key] == pytest.approx(float(factor) * score, abs=1e-6)
    by_kind = row['ledger']['bias_relevance_by_kind']
    assert by_kind['message_passing'] == pytest.approx(float(mp_bias), abs=1e-6)
    assert by_kind['head'] == pytest.approx(float(head_bias), abs=1e-6)
    assert row['method_settings']['gamma_rule'] == 'weight'
    assert row['exactness']['reasons'] == ['method_is_lrp', 'biases_present']
    assert row['diagnostics'][WEIGHT]['head_output_flipped'] is False
    true_prob = float(emitted_frame(module, captured.logits.detach()).true_prob[0])
    assert captured.record['prediction'] == true_prob
    assert no_hooks(module)


def test_negative_logit_head_inversion_is_reported_natively_and_offline(tmp_path):
    """Head bias -9.5 gives z = -0.5: at gamma 0.25 the output denominator is
    +1.75, every walk changes sign, and both paths report it."""
    module = binary_scoring_module(head_bias=True)
    with torch.no_grad():
        module.head.mlp.lins[0].bias.fill_(-9.5)
    for gamma, flipped in ((0.0, False), (0.25, True)):
        native, offline, _ = native_offline_parity(
            module, config(gamma=gamma, selected_class=True), tmp_path / str(gamma))
        assert native['status'] == offline['status'] == 'available'
        assert comparable_row(native) == comparable_row(offline)
        diagnostics = native['diagnostics'][WEIGHT]
        assert diagnostics['head_output_flipped'] is flipped
        assert all((w['relevance'] < 0) is flipped for w in native['walks'])


# Native versus offline ##########################################################

@pytest.mark.parametrize('gamma', [0.0, 0.25])
@pytest.mark.parametrize('selected', [True, False])
def test_native_and_offline_rows_agree_through_the_frozen_helper(tmp_path, gamma, selected):
    module = biased_module()
    native, offline, bundle = native_offline_parity(
        module, config(gamma=gamma, selected_class=selected, feature_vectors=True,
                       top_k=3), tmp_path)
    assert native['status'] == offline['status'] == 'available'
    assert comparable_row(native) == comparable_row(offline)
    assert offline['replay_logit_abs_diff'] == 0.0
    assert offline['numerics']['replay_precision_mode'] is not None
    assert bundle.validate()['status_counts'] == {'available': 1}
    for row in (native, offline):
        validate_v2_row(json.loads(json.dumps(row, allow_nan=False)), b1_sample())


def _randomize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in list(module.model.parameters()) + list(module.head.parameters()):
            param.copy_(torch.rand(param.shape, generator=generator) * 2 - 1)
    return module


@pytest.mark.parametrize('gamma', [0.0, 0.25])
@pytest.mark.parametrize('selected', [True, False])
def test_save_and_reload_parity(tmp_path, gamma, selected):
    """A native capture derived offline on a module rebuilt from a saved state
    dict gives the same row; an unloaded module fails the replay check."""
    cfg = config(gamma=gamma, selected_class=selected, feature_vectors=True, top_k=5)
    original = _randomize(biased_module(), seed=7)
    captured = capture_record(original, tmp_path / 'native', cfg)
    torch.save(original.state_dict(), tmp_path / 'state.pt')
    # Fresh fixture modules (a deep copy would share the scoring closure).
    reloaded = binary_scoring_module(gnn_bias=True, head_bias=True, calibration=(0.5, 0.3))
    unloaded = binary_scoring_module(gnn_bias=True, head_bias=True, calibration=(0.5, 0.3))
    reloaded.load_state_dict(torch.load(tmp_path / 'state.pt', weights_only=True))
    native = captured.record['walk_attribution']
    offline = next(derive_offline(captured, reloaded, cfg, tmp_path / 'offline').iter_rows())
    assert native['status'] == offline['status'] == 'available', offline.get('reason')
    assert comparable_row(native) == comparable_row(offline)
    stale = next(derive_offline(captured, unloaded, cfg, tmp_path / 'stale').iter_rows())
    assert stale['status'] == 'failed' and 'does not reproduce' in stale['reason']


def test_one_capture_derived_with_both_lrp_rules_coincides(tmp_path):
    """Every input of the B1 fixture is nonnegative, so both gamma rules give
    the same rows within 1e-12 of the absolute mass (contract D5)."""
    module = _randomize(biased_module(), seed=11)

    def lrp(method):
        return WalkAttributionConfig(method=method, gamma=0.25, selected_class=False,
                                     feature_vectors=True, top_k=100)

    captured = capture_record(module, tmp_path / 'native', lrp(WEIGHT))
    rows = {}
    for method in (WEIGHT, LEGACY):
        cfg = lrp(method)
        rows[method] = next(derive_offline(captured, module, cfg,
                                           tmp_path / method).iter_rows())
        assert rows[method]['status'] == 'available'
    new, old = rows[WEIGHT], rows[LEGACY]
    tol = 1e-12 * max(1.0, mass(new))
    a, b = walks_of(new), walks_of(old)
    assert set(a) == set(b) and all(abs(a[key] - b[key]) <= tol for key in a)
    for s, t in zip(new['sources'], old['sources']):
        assert all(abs(p - q) <= tol for p, q in zip(s['input_relevance'],
                                                     t['input_relevance']))
    for item in ('bias_relevance_by_kind', 'stabilizer_residual_by_kind'):
        assert all(abs(new['ledger'][item][k] - old['ledger'][item][k]) <= tol
                   for k in new['ledger'][item])
    assert new['diagnostics'] is not None and old['diagnostics'] is None


# Bundle validation ###############################################################

def _bundle(tmp_path, cfg):
    module = biased_module()
    run_root = tmp_path / 'run'
    captured = capture_record(module, run_root / 'xai' / 'local', cfg)
    emitted = emitted_frame(module, captured.logits.detach())
    frame = pd.DataFrame({
        'PREDICTION_ID': [captured.record['prediction_id']], 'ENTITY': [17],
        'TASK_ROW_ID': [8], 'TARGET_PRED': pd.array([None], dtype='boolean'),
        'False_PROB': np.array([float(emitted.false_prob[0])], dtype=np.float32),
        'True_PROB': np.array([float(emitted.true_prob[0])], dtype=np.float32)})
    frame.to_parquet(tmp_path / 'predictions.parquet')
    return PredictionEvidenceSource((str(tmp_path / 'predictions.parquet'), ), run_root,
                                    {}).export(tmp_path / 'bundle')


def test_evidence_bundles_with_weight_gamma_rows_validate(tmp_path):
    bundle = _bundle(tmp_path, config(gamma=0.25, feature_vectors=True))
    report = PredictionEvidenceBundle.load(bundle.root).validate()
    assert report['valid'] and report['evidence_rows'] == 1
    (record, ) = list(PredictionEvidenceBundle.load(bundle.root).iter_numerical_evidence())
    assert record['walk_attribution']['method'] == WEIGHT


ROW_MUTATIONS = {
    'contribution_rule': lambda r: r['method_settings'].update(gamma_rule='contribution'),
    'degree_t_claim': lambda r: r['exactness'].update(degree_t_claim=True, reasons=[]),
    'bags': lambda r: r.update(bags=[]),
    'missing_biases_present': lambda r: r['exactness'].update(reasons=['method_is_lrp']),
    'no_diagnostics': lambda r: r.update(diagnostics=None),
    'inconsistent_head_flip': lambda r: r['diagnostics'][WEIGHT].update(
        head_output_flipped=True),
}


@pytest.mark.parametrize('name', sorted(ROW_MUTATIONS))
def test_validation_rejects_mutated_native_rows(tmp_path, name):
    captured = capture_record(biased_module(), tmp_path, config(gamma=0.25))
    row = captured.record['walk_attribution']
    validate_v2_row(row, b1_sample())
    broken = copy.deepcopy(row)
    ROW_MUTATIONS[name](broken)
    with pytest.raises(WalkSchemaError):
        validate_v2_row(broken, b1_sample())


# Negative capability ##############################################################

def _swap_class(instance, base):
    instance.__class__ = type(f'Other{base.__name__}', (base, ), {})


def _post_operator(module):
    module.model.post_convs[0].layer_dict['customer'][0] = torch.nn.GELU()


def _aggregator(module):
    module.model.hetero_convs[0][FK].aggr = MaxAggregation()


#: Contract section 7 rejections (each applied to a working binary module).
REJECTIONS = {
    'model_type': lambda m: _swap_class(m.model, HeteroGNN),
    'head_type': lambda m: _swap_class(m.head, NodePredHead),
    'prediction_time_encodings': lambda m: setattr(m.head, 'prediction_time_encodings', True),
    'training_mode': lambda m: m.train(),
    'gelu_activation': lambda m: setattr(m.model, 'act', 'gelu'),
    'elu_head_activation': lambda m: setattr(m.head, 'act', 'elu'),
    'leaky_relu_activation': lambda m: setattr(m.model, 'act', 'leaky_relu'),
    'batch_norm': lambda m: setattr(m.model, 'norm', 'batch_norm'),
    'layer_norm_head': lambda m: setattr(m.head, 'norm', 'layer_norm'),
    'unregistered_post_operator': _post_operator,
    'unregistered_root_operator': lambda m: _swap_class(m.model.root_skips[0], HeteroLinear),
    'unregistered_relation_operator': lambda m: _swap_class(m.model.hetero_convs[0][ART_REV],
                                                            PrimaryKeyToForeignKeyConv),
    'max_aggregation': lambda m: setattr(m.model, 'aggr', ['max']),
    'actual_aggregator_differs': _aggregator,
    'multi_aggregation': lambda m: setattr(m.model, 'aggr', ['sum', 'mean']),
    'single_label_mapping': lambda m: setattr(m, 'target_encoder', target_encoder([[True, 10]])),
    'two_head_outputs': lambda m: setattr(m.head, 'out_channels', 2),
}
#: Mutations that change the forward itself, so the prediction may differ.
CHANGES_FORWARD = {'training_mode', 'unregistered_post_operator', 'two_head_outputs',
                   'actual_aggregator_differs'}


@pytest.mark.parametrize('name', sorted(REJECTIONS))
def test_rejections_raise_before_hooks_or_record_unsupported_rows(tmp_path, name):
    reference = biased_module()
    expected = float(emitted_frame(reference, reference.common_step().detach()).true_prob[0])
    module = biased_module()
    REJECTIONS[name](module)
    with pytest.raises(UnsupportedWalkAttribution):
        capture_record(module, tmp_path / 'raise', config(gamma=0.25))
    assert no_hooks(module)
    module = biased_module()
    REJECTIONS[name](module)
    captured = capture_record(module, tmp_path / 'record', config(gamma=0.25),
                              on_unsupported='record')
    row = captured.record['walk_attribution']
    assert row['status'] == 'unsupported' and row['method'] == WEIGHT
    assert no_hooks(module) and captured.manifest['complete'] is False
    if name not in CHANGES_FORWARD:
        assert captured.record['prediction'] == expected


def test_distribution_head_is_rejected_before_hooks(tmp_path):
    module = biased_module()
    module.head.param_keys = ['mu']  # its forward returns a dict: raise mode only
    with pytest.raises(UnsupportedWalkAttribution):
        capture_record(module, tmp_path, config())
    assert no_hooks(module)


@pytest.mark.parametrize('module_options,cfg_options,match', [
    ({'loss': 'focal'}, {}, 'BinaryCrossEntropyLoss'),
    ({}, {'selected_class': 1}, 'not a label'),
    ({}, {'selected_class': 'True'}, 'not a label'),
])
def test_task_level_rejections(tmp_path, module_options, cfg_options, match):
    module = binary_scoring_module(**module_options)
    with pytest.raises(UnsupportedWalkAttribution, match=match):
        capture_record(module, tmp_path / 'raise', config(**cfg_options))
    captured = capture_record(module, tmp_path / 'record', config(**cfg_options),
                              on_unsupported='record')
    assert captured.record['walk_attribution']['status'] == 'unsupported'
    with pytest.raises(ValueError, match='bags=True requires'):
        config(bags=True)


def test_multi_edge_gather_is_a_row_level_rejection(tmp_path):
    from walk_binary_fixtures import b1_forward
    module = biased_module()
    x = {'customer': torch.tensor([[1.]]), 'transactions': torch.tensor([[2.], [3.]]),
         'article': torch.tensor([[4.], [5.]])}
    edges = {FK: torch.tensor([[0, 1], [0, 0]]), REV: torch.tensor([[0, 0], [0, 1]]),
             ART_REV: torch.tensor([[0, 1], [0, 0]]),
             ('transactions', 'article_id', 'article'): torch.zeros(2, 0, dtype=torch.long)}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0], 'article': [0, 0, 2]}
    counts = {FK: [2, 0], REV: [0, 2], ART_REV: [0, 2],
              ('transactions', 'article_id', 'article'): [0, 0]}
    module.common_step = lambda _=None: b1_forward(module, (x, edges, nodes, counts))
    sample = b1_sample()
    sample['nodes']['article'] = {'pkeys': [9, 10], 'node_ids': [90, 100], 'num_nodes': 2}
    for mode in ('raise', 'record'):
        captured = capture_record(module, tmp_path / mode, config(), sample=sample,
                                  on_unsupported=mode)
        row = captured.record['walk_attribution']
        assert row['status'] == 'unsupported' and 'multiple incoming' in row['reason']


def test_identity_activation_and_eval_mode_dropout_are_allowed(tmp_path):
    module = biased_module()
    for post in module.model.post_convs:
        for layers in post.layer_dict.values():
            layers[-1].p = 0.5
    captured = capture_record(module, tmp_path / 'dropout', config(gamma=0.25))
    assert captured.record['walk_attribution']['status'] == 'available'
    identity = biased_module()
    identity.model.act = None  # as HeteroGNN(act=None): no activation module
    for post in identity.model.post_convs:
        for layers in post.layer_dict.values():
            del layers[0]
    captured = capture_record(identity, tmp_path / 'identity', config(gamma=0.25))
    assert captured.record['walk_attribution']['status'] == 'available'
    report = inspect_walk_capability(identity, config=config())
    assert report['supported'] and report['qualification'] == list(STRUCTURAL_CODES)
    assert no_hooks(identity)


# Precision ######################################################################

def _precision_row(mode):
    module = precision_module()
    cfg = config(gamma=0.25, top_k=100)
    selection = Selection('regression', 0, 1, None)
    with float32_matmul_precision(mode):
        numerics = {'precision_mode': mode,
                    'mkldnn_matmul_fp32_precision': mkldnn_matmul_fp32_precision()}
        with torch.no_grad(), GraphCapture(module, cfg) as cap:
            logits = module.common_step()
        row = explain_capture(cap, config=cfg, selection=selection,
                              output=output_block(selection, module, logits[0]),
                              sample=precision_sample(), origin='original_forward_attribution',
                              prediction_id='a' * 64, sample_sha256='b' * 64,
                              numerics=row_numerics(numerics, 'float32'))
    return numerics, row


def test_capture_precision_shows_in_the_preactivation_mismatch():
    """Twelve events per 32-channel linear: float32 captures differ from the
    float64 recomputation by float32 rounding under `highest` and by the
    bf16 internals under `medium` on hosts that use them; the ledger is
    unaffected (precision_residual is 0 for LRP)."""
    highest = _precision_row('highest')[1]
    numerics, medium = _precision_row('medium')
    for row in (highest, medium):
        assert row['status'] == 'available', row.get('reason')
        assert row['ledger']['precision_residual'] == 0.0
    small = highest['diagnostics'][WEIGHT]['max_preactivation_mismatch']
    assert 0.0 < small < 1e-4
    if numerics['mkldnn_matmul_fp32_precision'] != 'bf16':
        pytest.skip('this host does not run medium-precision matmuls in bf16')
    assert medium['diagnostics'][WEIGHT]['max_preactivation_mismatch'] > 10 * small
