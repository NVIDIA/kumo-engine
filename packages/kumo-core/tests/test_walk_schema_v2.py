# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""walk_attribution/v2 rows: construction, bags and validator mutations."""
import copy
import json
import math

import pytest
import torch
from kumoapi.encoder import Index
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType

from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.explain.walk_attribution import (
    DEGREE_T_METHOD,
    LEGACY_METHOD,
    METHOD_MODULES,
    WEIGHT_GAMMA_METHOD,
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
)
from kumo_core.explain.walk_binary import (
    Selection,
    output_block,
    resolve_selection,
)
from kumo_core.explain.walk_enumerate import Absorbed, enumerate_walks
from kumo_core.explain.walk_graph import (
    GraphCapture,
    architecture_record,
    operations_record,
)
from kumo_core.explain.walk_lrp_reference import ContributionGammaRule
from kumo_core.explain.walk_lrp_reference import summarize as legacy_summarize
from kumo_core.explain.walk_schema import (
    WalkSchemaError,
    absolute_mass,
    bag_signature,
    build_row,
    explain_capture,
    group_bags,
    row_numerics,
    row_step_key,
    tolerance,
    validate_v2_row,
    walk_key,
)
from kumo_core.modeling.calibration import PlatScaling
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import BinaryCrossEntropyLoss
from tests.test_walk_enumerate import PathProductRule
from tests.test_walk_graph import (
    ART_REV,
    FK,
    REV,
    b1_batch,
    b1_forward,
    b1_module,
    capture,
    random_b1,
    toy_t3_module,
)

ID = 'a' * 64
SAMPLE_ID = 'b' * 64
NUMERICS = {'precision_mode': 'highest', 'mkldnn_matmul_fp32_precision': 'ieee',
            'num_threads': 2, 'cpu_capability': 'AVX512', 'torch_version': '2.13.0+cpu'}


def b1_sample():
    return {'nodes': {'customer': {'pkeys': [17], 'node_ids': [170]},
                      'transactions': {'pkeys': [41, 42], 'node_ids': [410, 420]},
                      'article': {'pkeys': [9], 'node_ids': [90]}}}


def binary_b1(dtype=torch.float64, **options):
    module = b1_module(dtype, **options)
    module.task_type = TaskType.BINARY_CLASSIFICATION
    module.loss = BinaryCrossEntropyLoss(TaskType.BINARY_CLASSIFICATION)
    index = Index()
    index._stats = {ColStatType.CATEGORY_COUNTS: [[True, 123], [False, 37]]}
    module.target_encoder = TargetEncoder.from_encoder(TaskType.BINARY_CLASSIFICATION, index)
    module.calibrator = PlatScaling(TaskType.BINARY_CLASSIFICATION)
    with torch.no_grad():
        module.calibrator.weight_and_bias.copy_(torch.tensor([[0.8], [-0.1]]))
    module.calibrator.set_extra_state({'is_fitted': True})
    module.calibrator.to(dtype)
    return module.eval()  # every attached submodule in evaluation mode


def row_for(module, config, *, origin='original_forward_attribution'):
    batch = b1_batch(next(module.parameters()).dtype)
    with GraphCapture(module, config) as cap:
        logits = b1_forward(module, batch)
    if getattr(module, 'task_type', None) == TaskType.BINARY_CLASSIFICATION:
        selection = resolve_selection(module, config)
    else:
        selection = Selection('regression', config.output_index, 1, None)
    output = output_block(selection, module, logits[0])
    numerics = row_numerics(NUMERICS, str(logits.dtype).removeprefix('torch.'))
    return explain_capture(cap, config=config, selection=selection, output=output,
                           sample=b1_sample(), origin=origin, prediction_id=ID,
                           sample_sha256=SAMPLE_ID, numerics=numerics)


@pytest.fixture
def row():
    config = WalkAttributionConfig(feature_vectors=True, top_k=5)
    built = row_for(b1_module(gnn_bias=True, head_bias=True, head_layers=2), config)
    assert built['status'] == 'available', built.get('reason')
    return built


@pytest.fixture
def binary_row():
    # z = -9 gives True_PROB < 0.5, where `1 - True_PROB` in float64 differs
    # from the emitted float32 False_PROB (C7).
    config = WalkAttributionConfig(feature_vectors=True, selected_class=False)
    built = row_for(binary_b1(torch.float32, values={'head.mlp.lins.0.weight': [[-1.0]]}),
                    config)
    assert built['status'] == 'available', built.get('reason')
    return built


def test_valid_rows_are_json_round_trippable(row, binary_row):
    for built in (row, binary_row):
        validate_v2_row(built, b1_sample())
        reloaded = json.loads(json.dumps(built, allow_nan=False))
        validate_v2_row(reloaded, b1_sample())
    assert row['coverage']['status'] == 'complete_top_k'
    assert row['coverage']['total_walks'] == 8 and len(row['walks']) == 5
    assert binary_row['coordinate'] == 'selected_class_logit'
    assert binary_row['score'] == -binary_row['output']['head_output'] == 9.0
    assert binary_row['exactness']['qualification'][-1] == 'single_edge_gathers'
    assert row['ledger']['bias_relevance_by_kind']['head'] != 0.0


def mutate(row, change):
    broken = copy.deepcopy(row)
    change(broken)
    return broken


def first_step(r):
    return r['walks'][0]['steps'][0]


MUTATIONS = {
    'unknown_kind': lambda r: first_step(r).update(kind='teleport'),
    'mp_layer_gap': lambda r: r['walks'][0]['steps'][1].update(mp_layer=3),
    'non_contiguous': lambda r: r['walks'][0].update(
        source_occurrence=r['walks'][0]['source_occurrence'] + 1),
    'not_ending_at_seed': lambda r: r['seed'].update(occurrence=1),
    'too_few_steps': lambda r: r['walks'][0]['steps'].pop(),
    'ledger_mismatch': lambda r: r['ledger'].update(
        bias_relevance=r['ledger']['bias_relevance'] + 1e-3),
    'by_kind_mismatch': lambda r: r['ledger']['bias_relevance_by_kind'].update(head=0.0),
    'accounting_error_too_large': lambda r: (
        r['ledger'].update(accounting_error=0.5,
                           precision_residual=r['ledger']['precision_residual'] - 0.5)),
    'displayed_sum': lambda r: r['ledger'].update(
        displayed_walk_relevance=r['ledger']['displayed_walk_relevance'] + 1),
    'vector_length': lambda r: r['walks'][0]['input_relevance'].append(0.0),
    'vector_sum': lambda r: r['walks'][0]['input_relevance'].__setitem__(0, 7.0),
    'inactive_channel': lambda r: (r['sources'][0]['input_value'].__setitem__(0, 0.0)),
    'source_walk_count': lambda r: r['sources'][0].update(
        walk_count=r['sources'][0]['walk_count'] + 1),
    'source_displayed_count': lambda r: r['sources'][0].update(
        displayed_walk_count=r['sources'][0]['displayed_walk_count'] + 1),
    'source_missing': lambda r: r['sources'].pop(),
    'source_relevance': lambda r: r['sources'][0].update(
        relevance=r['sources'][0]['relevance'] + 1),
    'rank': lambda r: r['walks'][1].update(rank=0),
    'order': lambda r: r['walks'].reverse(),
    'duplicate_walk': lambda r: r['walks'].__setitem__(1, copy.deepcopy(r['walks'][0])),
    'nonfinite': lambda r: r['walks'][0].update(relevance=math.inf),
    'available_but_exhausted': lambda r: r['coverage'].update(status='budget_exhausted'),
    'complete_with_omission': lambda r: (r['coverage'].update(status='complete')),
    'expanded_over_budget': lambda r: r['coverage'].update(expanded_routes=10 ** 6),
    'budget_unit': lambda r: r['coverage'].update(budget_unit='route_expansions'),
    'non_canonical_config': lambda r: r['config'].update(bags=False),
    'method_mismatch': lambda r: r.update(method=WEIGHT_GAMMA_METHOD),
    'unknown_method': lambda r: r.update(method='hetero_lrp_mystery_v1'),
    'reason_on_available': lambda r: r.update(reason='because'),
    'numerics_key': lambda r: r['numerics'].pop('num_threads'),
    'replay_mode_on_original': lambda r: r['numerics'].update(replay_precision_mode='medium'),
    'regression_score': lambda r: r.update(score=r['score'] + 1),
    'relation_operator': lambda r: next(
        s for w in r['walks'] for s in w['steps'] if s['kind'] == 'relation').update(
            operator='sum'),
    'gather_denominator': lambda r: next(
        s for w in r['walks'] for s in w['steps']
        if s['kind'] == 'relation' and s['operator'] == 'gather').update(denominator=2),
    'self_operator': lambda r: next(
        s for w in r['walks'] for s in w['steps'] if s['kind'] == 'self_update').update(
            operator='identity'),
    'lrp_claim': lambda r: r['exactness'].update(degree_t_claim=True, reasons=[]),
    'diagnostics_namespace': lambda r: r.update(diagnostics={'other_method': {}}),
    'diagnostics_not_json': lambda r: r.update(diagnostics={r['method']: {'x': {1, 2}}}),
    'schema': lambda r: r.update(schema='walk_attribution/v1'),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_validator_rejects_mutation(row, name):
    with pytest.raises(WalkSchemaError):
        validate_v2_row(mutate(row, MUTATIONS[name]), b1_sample())


BINARY_MUTATIONS = {
    'score_not_selected_logit': lambda r: r.update(score=-r['score']),
    'logit_sign': lambda r: r['output']['binary'].update(logit_sign=1),
    'false_prob_in_float64': lambda r: r['output']['binary'].update(
        emitted_probability_selected=1.0 - r['output']['binary'][
            'emitted_encoded_positive_probability']),
    'column_name': lambda r: r['output']['binary'].update(emitted_column_selected='True_PROB'),
    'emitted_probability': lambda r: r['output']['binary'].update(
        emitted_encoded_positive_probability=0.5),
    'weight_effective': lambda r: r['output']['binary']['calibration'].update(
        weight_effective=0.5),
    'model_probability': lambda r: r['output']['binary'].update(
        model_probability_selected=0.5),
    'mapping': lambda r: r['output']['binary']['label_mapping'][0].update(encoded_class=0),
    'coordinate': lambda r: r.update(coordinate='encoded_model_output'),
}


@pytest.mark.parametrize('name', sorted(BINARY_MUTATIONS))
def test_validator_rejects_binary_mutation(binary_row, name):
    broken = mutate(binary_row, BINARY_MUTATIONS[name])
    assert broken != binary_row
    with pytest.raises(WalkSchemaError):
        validate_v2_row(broken, b1_sample())


def test_binding_with_the_retained_graph(row):
    sample = b1_sample()
    sample['nodes']['transactions']['pkeys'][0] = 99
    with pytest.raises(WalkSchemaError, match='Bound key'):
        validate_v2_row(row, sample)


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


def test_rows_without_explanations():
    """Method-agnostic: every method refuses an out-of-profile module before
    hooks, and its rows without explanations validate."""
    module = b1_module()
    module.model.act = 'gelu'  # outside the shared operator profile
    for method in METHOD_MODULES:
        config = WalkAttributionConfig(method=method)
        cap = GraphCapture(module, config)
        with pytest.raises(UnsupportedWalkAttribution):
            cap.__enter__()
        assert no_hooks(module)
        other = next(m for m in METHOD_MODULES if m != method)
        for status in ('unsupported', 'failed'):
            row = {'schema': 'walk_attribution/v2', 'status': status,
                   'reason': 'activation outside the walk profile',
                   'origin': 'original_forward_attribution', 'prediction_id': ID,
                   'sample_sha256': SAMPLE_ID, 'method': method, 'config': config.to_dict()}
            validate_v2_row(row)
            for broken in ({**row, 'walks': [{}]}, {**row, 'reason': ''},
                           {**row, 'method': other}):
                with pytest.raises(WalkSchemaError):
                    validate_v2_row(broken)


def test_every_method_row_is_unsupported_on_an_out_of_profile_graph():
    """A multi-edge primary-key gather is refused by the shared capture for
    every method: an `unsupported` row of that method, no fallback, no hooks."""
    module = b1_module()
    x, edges, nodes, sampled = b1_batch()
    edges[ART_REV] = torch.tensor([[0, 0], [0, 0]])  # two gathers into transaction 0
    sampled = {**sampled, ART_REV: [0, 2]}
    cap = GraphCapture(module, WalkAttributionConfig())
    with cap:
        logits = b1_forward(module, (x, edges, nodes, sampled))
    selection = Selection('regression', 0, 1, None)
    for method in METHOD_MODULES:
        config = WalkAttributionConfig(method=method)
        row = explain_capture(cap, config=config, selection=selection,
                              output=output_block(selection, module, logits[0]),
                              sample=b1_sample(), origin='original_forward_attribution',
                              prediction_id=ID, sample_sha256=SAMPLE_ID,
                              numerics=row_numerics(NUMERICS, 'float64'))
        assert row['status'] == 'unsupported' and row['method'] == method
        assert row['config'] == config.to_dict()
        assert row['reason'] == 'PK gather has multiple incoming occurrences'
        assert 'walks' not in row
        assert row['output']['head_output'] == float(logits[0, 0].detach())
        validate_v2_row(row)
        json.dumps(row, allow_nan=False)
    assert no_hooks(module)


def test_budget_exhaustion_reports_display_and_search_omissions():
    config = WalkAttributionConfig(top_k=2, max_walks=8)
    built = row_for(b1_module(), config)
    assert built['status'] == 'incomplete', built.get('reason')
    coverage, ledger = built['coverage'], built['ledger']
    assert coverage['status'] == 'budget_exhausted' and coverage['unexpanded_prefixes'] == 1
    assert coverage['total_walks'] == 5 and coverage['displayed_walks'] == 2
    # m5: display omissions cover the completed walks that were not shown.
    assert ledger['omitted_display_abs_relevance'] > 0
    assert ledger['omitted_search_relevance'] != 0
    assert 'enumeration_incomplete' in built['exactness']['reasons']
    validate_v2_row(built, b1_sample())


def test_complete_rows_show_every_walk_without_omissions():
    built = row_for(b1_module(), WalkAttributionConfig(top_k=8))
    assert built['coverage']['status'] == 'complete'
    assert built['ledger']['omitted_display_relevance'] == 0.0
    assert built['ledger']['omitted_search_relevance'] == 0.0
    assert built['exactness']['reasons'] == ['method_is_lrp']


class BrokenRule(PathProductRule):
    """Absorbs nothing yet reports walks that do not sum to the score."""


def toy_rows(config, *, rule=PathProductRule, score_offset=0.0, summary=None):
    module, run = toy_t3_module()
    cap, logits = capture(module, run)
    computation = cap.computation()
    enumeration = enumerate_walks(computation, rule(computation),
                                  max_walks=config.max_walks)
    total = math.fsum(w.relevance for w in enumeration.walks)
    selection = Selection('regression', 0, 1, None)
    output = output_block(selection, module, logits[0])
    score = output['head_output']
    summary = summary or {
        'method_settings': dict.fromkeys(('gamma', 'gamma_rule', 'bias_rule', 'epsilon',
                                          'stabilizer', 'identity_root_rule',
                                          'sign_folding')),
        'precision_residual': score - total + score_offset,
        'exactness': {'decomposition': 'piecewise_gnn_gi', 'degree_t_claim': False,
                      'reasons': ['precision_residual_exceeded'], 'piece': 'interior',
                      'boundary_neurons': 0, 'mask_disagreements': 0,
                      'relevant_gated_neurons': 3, 'min_abs_gated_preactivation': 1.0,
                      'qualification': []}}
    sample = {'nodes': {'customer': {'pkeys': [5]}, 'transactions': {'pkeys': [6]}}}
    row = build_row(computation, enumeration, summary, config=config, selection=selection,
                    output=output, qualification=['hetero_gnn'],
                    architecture=_architecture(module), operations={}, sample=sample,
                    origin='original_forward_attribution', prediction_id=ID,
                    sample_sha256=SAMPLE_ID, numerics=row_numerics(NUMERICS, 'float64'))
    return row, enumeration, sample


def _architecture(module):
    from kumo_core.explain.walk_graph import architecture_record
    return architecture_record(module)


def test_bags_by_hand_on_the_t3_toy():
    """D9: membership, multiplicities and sums on a T = 3 toy (hand values).

    Weights: self 2, FK 3, REV 5; h0(customer) 1, h0(transaction) 2. The
    eight walks are 8, 24, 24, 30, 24, 30, 30 and 90 (total 260).
    """
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True, top_k=3)
    row, enumeration, sample = toy_rows(config)
    assert row['status'] == 'available', row.get('reason')
    fk = {'kind': 'relation', 'relation': list(FK), 'edge_occurrence': 0}
    rev = {'kind': 'relation', 'relation': list(REV), 'edge_occurrence': 0}
    self_c = {'kind': 'self_update', 'table': 'customer', 'occurrence': 0}
    self_t = {'kind': 'self_update', 'table': 'transactions', 'occurrence': 0}

    def bag(relevance, count, *variables):
        return {'relevance': relevance, 'walk_count': count,
                'variables': [{**v, 'multiplicity': m} for v, m in variables]}

    expected = [
        bag(90.0, 1, (rev, 1), (fk, 2)),                       # repeated edge, alpha 2
        bag(60.0, 2, (rev, 1), (fk, 1), (self_c, 1)),          # multi-walk bag
        bag(30.0, 1, (rev, 1), (fk, 1), (self_t, 1)),
        bag(24.0, 1, (fk, 1), (self_c, 1), (self_t, 1)),
        bag(24.0, 1, (fk, 1), (self_c, 2)),
        bag(24.0, 1, (fk, 1), (self_t, 2)),
        bag(8.0, 1, (self_c, 3)),
    ]
    assert [{k: b[k] for k in ('relevance', 'walk_count', 'variables')}
            for b in row['bags']] == expected
    assert [b['index'] for b in row['bags']] == list(range(7))
    assert row['coverage']['bags'] == 'complete' and row['coverage']['total_walks'] == 8
    # Displayed walks: 90, then the 30s by walk key; each links to its bag.
    assert [w['relevance'] for w in row['walks']] == [90.0, 30.0, 30.0]
    assert [w['bag'] for w in row['walks']] == [0, 1, 2]
    assert row['ledger']['omitted_display_relevance'] == 110.0
    # Bag sums are plain walk sums; 1/alpha! never applies to walk sums.
    groups = group_bags(enumeration.walks)
    assert math.fsum(relevance for _, _, relevance in groups) == 260.0
    validate_v2_row(row, sample, method_checks=False)
    broken = copy.deepcopy(row)
    broken['walks'][1]['bag'] = 2
    with pytest.raises(WalkSchemaError, match='wrong bag'):
        validate_v2_row(broken, sample, method_checks=False)
    broken = copy.deepcopy(row)
    broken['bags'][0]['variables'][1]['multiplicity'] = 1
    with pytest.raises(WalkSchemaError):
        validate_v2_row(broken, sample, method_checks=False)
    broken = copy.deepcopy(row)
    broken['bags'][1]['walk_count'] = 1
    with pytest.raises(WalkSchemaError, match='cover every'):
        validate_v2_row(broken, sample, method_checks=False)


def test_bag_signature_counts_tied_variables():
    module, run = toy_t3_module()
    cap, _ = capture(module, run)
    computation = cap.computation()
    enumeration = enumerate_walks(computation, PathProductRule(computation), max_walks=100)
    signatures = [bag_signature(walk_key(w.steps)) for w in enumeration.walks]
    assert len(signatures) == 8 and len(set(signatures)) == 7
    assert all(sum(m for _, m in s) == 3 for s in signatures)


def test_bags_are_omitted_without_complete_enumeration():
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True, max_walks=3)
    row, _, sample = toy_rows(config, summary={
        'method_settings': dict.fromkeys(('gamma', 'gamma_rule', 'bias_rule', 'epsilon',
                                          'stabilizer', 'identity_root_rule',
                                          'sign_folding')),
        'precision_residual': 0.0,
        'exactness': {'decomposition': 'piecewise_gnn_gi', 'degree_t_claim': False,
                      'reasons': ['enumeration_incomplete', 'precision_residual_exceeded'],
                      'piece': 'interior', 'boundary_neurons': 0, 'mask_disagreements': 0,
                      'relevant_gated_neurons': 3, 'min_abs_gated_preactivation': 1.0,
                      'qualification': []}})
    assert row['status'] in ('incomplete', 'failed')
    if row['status'] == 'incomplete':
        assert row['bags'] is None
        assert row['coverage']['bags'] == 'omitted_incomplete_enumeration'


def test_ledger_that_does_not_close_fails_the_row():
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True)
    row, _, _ = toy_rows(config, score_offset=1.0)
    assert row['status'] == 'failed' and row['reason'] == 'accounting_error_exceeded'
    assert row['ledger']['accounting_error'] == pytest.approx(-1.0)
    assert 'walks' not in row and row['input_coordinate'] == 'encoded_node_features'
    assert row['coordinate'] == 'encoded_model_output' and row['output'] and row['numerics']
    validate_v2_row(row)


def gnn_gi_row():
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True)
    row, _, sample = toy_rows(config)
    validate_v2_row(row, sample, method_checks=False)
    return row, sample


DEGREE_T_MUTATIONS = {
    'interior_with_boundary': lambda r: r['exactness'].update(boundary_neurons=1),
    'boundary_label_without_count': lambda r: r['exactness'].update(piece='boundary'),
    'infinite_margin': lambda r: r['exactness'].update(min_abs_gated_preactivation=math.inf),
    'null_margin_with_gates': lambda r: r['exactness'].update(
        min_abs_gated_preactivation=None),
    'missing_precision_reason': lambda r: r['exactness'].update(reasons=[]),
    'claim_with_incomplete_search': lambda r: (
        r['exactness'].update(degree_t_claim=True, reasons=[],
                              decomposition='exact_degree_t'),
        r['coverage'].update(enumeration_complete=False)),
    'claim_with_biases': lambda r: (
        r['exactness'].update(degree_t_claim=True, reasons=[],
                              decomposition='exact_degree_t'),
        r['architecture'].update(gnn_bias=True)),
    'lrp_reason_on_gnn_gi': lambda r: r['exactness']['reasons'].append('method_is_lrp'),
}


@pytest.mark.parametrize('name', sorted(DEGREE_T_MUTATIONS))
def test_degree_t_fields_are_checked(name):
    row, sample = gnn_gi_row()
    with pytest.raises(WalkSchemaError):
        validate_v2_row(mutate(row, DEGREE_T_MUTATIONS[name]), sample, method_checks=False)


def test_null_margin_without_gated_neurons_is_valid():
    row, sample = gnn_gi_row()
    row['exactness'].update(relevant_gated_neurons=0, min_abs_gated_preactivation=None)
    validate_v2_row(row, sample, method_checks=False)


def test_method_diagnostics_are_namespaced_and_json_safe():
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True)
    module, run = toy_t3_module()
    cap, logits = capture(module, run)
    computation = cap.computation()
    enumeration = enumerate_walks(computation, PathProductRule(computation), max_walks=100)
    selection = Selection('regression', 0, 1, None)
    output = output_block(selection, module, logits[0])
    total = math.fsum(w.relevance for w in enumeration.walks)
    summary = {
        'method_settings': dict.fromkeys(('gamma', 'gamma_rule', 'bias_rule', 'epsilon',
                                          'stabilizer', 'identity_root_rule',
                                          'sign_folding')),
        'precision_residual': output['head_output'] - total,
        'exactness': {'decomposition': 'piecewise_gnn_gi', 'degree_t_claim': False,
                      'reasons': ['precision_residual_exceeded'], 'piece': 'interior',
                      'boundary_neurons': 0, 'mask_disagreements': 0,
                      'relevant_gated_neurons': 3, 'min_abs_gated_preactivation': 1.0,
                      'qualification': ['interior_piece']},
        'diagnostics': {'margins': [1.0, 2.5], 'note': 'test'}}
    sample = {'nodes': {'customer': {'pkeys': [5]}, 'transactions': {'pkeys': [6]}}}
    common = dict(config=config, selection=selection, output=output,
                  qualification=['hetero_gnn'], architecture=_architecture(module),
                  operations={}, sample=sample, origin='original_forward_attribution',
                  prediction_id=ID, sample_sha256=SAMPLE_ID,
                  numerics=row_numerics(NUMERICS, 'float64'))
    row = build_row(computation, enumeration, summary, **common)
    assert row['diagnostics'] == {DEGREE_T_METHOD: {'margins': [1.0, 2.5], 'note': 'test'}}
    # The builder merges qualify codes with the method's row codes (D10).
    assert row['exactness']['qualification'] == ['hetero_gnn', 'single_edge_gathers',
                                                 'interior_piece']
    validate_v2_row(row, sample, method_checks=False)
    with pytest.raises(WalkSchemaError, match='JSON-safe'):
        build_row(computation, enumeration, {**summary, 'diagnostics': {'x': math.nan}},
                  **common)


def test_absorbed_amounts_reach_the_ledger_by_kind():
    module, run = toy_t3_module()
    cap, logits = capture(module, run)
    computation = cap.computation()

    class Absorbing(PathProductRule):
        def start(self):
            return 1.0, Absorbed.of('head', bias=0.25, stabilizer=-0.125)

    enumeration = enumerate_walks(computation, Absorbing(computation), max_walks=100)
    assert enumeration.bias == {'preprocessing': 0.0, 'message_passing': 0.0, 'head': 0.25}
    assert enumeration.stabilizer['head'] == -0.125


@pytest.mark.parametrize('offset', [math.nan, math.inf])
def test_nonfinite_ledger_fails_the_row_with_serializable_fields(offset):
    config = WalkAttributionConfig(method=DEGREE_T_METHOD, bags=True)
    row, _, _ = toy_rows(config, score_offset=offset)
    assert row['status'] == 'failed' and row['reason'] == 'nonfinite_relevance'
    assert 'ledger' not in row and 'walks' not in row
    json.dumps(row, allow_nan=False)  # never a nonfinite number in the row
    validate_v2_row(row)


# The explained class is the row's own configuration (review m1) ##############

SELECTION_MUTATIONS = {
    'other_label': lambda r: r['config'].update(selected_class=True),
    'removed': lambda r: r['config'].pop('selected_class'),
    'type_changed': lambda r: r['config'].update(selected_class=0),
    'default_source': lambda r: r['output']['binary'].update(
        selected_class_source='encoded_positive_default'),
}


@pytest.mark.parametrize('name', sorted(SELECTION_MUTATIONS))
def test_explained_class_is_bound_to_the_row_config(binary_row, name):
    broken = mutate(binary_row, SELECTION_MUTATIONS[name])
    with pytest.raises(WalkSchemaError, match='selected_class'):
        validate_v2_row(broken, b1_sample())


def test_default_selection_and_regression_rows_have_no_configured_class(row):
    default = row_for(binary_b1(torch.float32), WalkAttributionConfig(feature_vectors=True))
    binary = default['output']['binary']
    assert binary['selected_class_source'] == 'encoded_positive_default'
    assert binary['selected_class_label'] is True and binary['logit_sign'] == 1
    validate_v2_row(default, b1_sample())
    for valid in (default, row):
        broken = mutate(valid, lambda r: r['config'].update(selected_class=True))
        with pytest.raises(WalkSchemaError, match='select'):
            validate_v2_row(broken, b1_sample())


# Relation steps bound to the retained edges (review m2, contract 4.10) ######

def b1_retained_sample(counts=True):
    """`b1_sample` plus the edge records Engine retains for B1's batch."""
    sample = b1_sample()
    _, edges, _, sampled = b1_batch()
    sample['edges'] = [{'type': list(kind), 'edge_index': index.tolist(),
                        'num_sampled_edges': list(sampled[kind]) if counts else None,
                        'original_edge_index': None} for kind, index in edges.items()]
    return sample


def edge_record(sample, kind):
    return next(e for e in sample['edges'] if tuple(e['type']) == kind)


def add_edge(sample, kind, source, destination, sampled):
    record = edge_record(sample, kind)
    record['edge_index'][0].append(source)
    record['edge_index'][1].append(destination)
    record['num_sampled_edges'] = sampled


# B1's top walk is article -> transaction 0 (gather, layer 1), then
# transaction 0 -> customer 0 (mean over 2 edges, layer 2).
EDGE_MUTATIONS = {
    'extra_seed_edge': (lambda s: add_edge(s, FK, 1, 0, [3, 0]), 'Mean denominator'),
    'trimmed_at_layer_2': (lambda s: edge_record(s, FK).update(num_sampled_edges=[0, 2]),
                           'outside the edges of its layer'),
    'other_source': (lambda s: edge_record(s, FK)['edge_index'][0].reverse(),
                     'endpoints differ'),
    'gather_fan_in': (lambda s: add_edge(s, ART_REV, 0, 0, [0, 2]), 'one incoming edge'),
    'missing_relation': (lambda s: s['edges'].remove(edge_record(s, FK)), 'absent'),
    'inconsistent_trimming': (lambda s: edge_record(s, FK).update(num_sampled_edges=None),
                              'Inconsistent edge trimming'),
}


def test_relation_steps_validate_on_the_retained_edges(row, binary_row):
    for built in (row, binary_row):
        validate_v2_row(built, b1_retained_sample())
        validate_v2_row(built, b1_retained_sample(counts=False))  # an untrimmed record
    steps = [s for w in binary_row['walks'] for s in w['steps'] if s['kind'] == 'relation']
    assert {(s['operator'], s['denominator']) for s in steps} == {('gather', 1), ('mean', 2)}


@pytest.mark.parametrize('name', sorted(EDGE_MUTATIONS))
def test_retained_edge_mutations_are_rejected(row, name):
    sample = b1_retained_sample()
    change, match = EDGE_MUTATIONS[name]
    change(sample)
    with pytest.raises(WalkSchemaError, match=match):
        validate_v2_row(row, sample)


def test_mean_denominator_is_the_retained_in_degree(row):
    broken = copy.deepcopy(row)
    step = broken['walks'][0]['steps'][1]
    assert (step['operator'], step['denominator']) == ('mean', 2)
    step['denominator'] = 3
    validate_v2_row(broken, b1_sample())  # bounds alone cannot see it
    with pytest.raises(WalkSchemaError, match='Mean denominator'):
        validate_v2_row(broken, b1_retained_sample())


def test_row_step_key_is_the_step_key_of_the_serialized_route():
    """The frozen walk key of serialized steps (n8): equal to the route keys."""
    row, enumeration, _ = toy_rows(WalkAttributionConfig(top_k=10000))
    reloaded = json.loads(json.dumps(row, allow_nan=False))
    serialized = [tuple(row_step_key(step) for step in walk['steps'])
                  for walk in reloaded['walks']]
    assert len(serialized) == len(enumeration.walks) == len(set(serialized))
    assert sorted(serialized) == sorted(walk_key(walk.steps) for walk in enumeration.walks)
    kinds = {key[1] for walk in serialized for key in walk}
    assert kinds == {'relation', 'self_update'}


# Failed rows of a nonfinite model output stay strict JSON (contract 4.2) #####

# A nonfinite layer-2 weight: +inf and nan reach the logit; ReLU turns -inf
# into a finite logit of 0 while the relevance is still not finite.
@pytest.mark.parametrize('value,finite_logit', [(math.inf, False), (math.nan, False),
                                                (-math.inf, True)])
def test_failed_rows_with_nonfinite_values_are_strict_json(value, finite_logit):
    for module, config in ((b1_module(), WalkAttributionConfig()),
                           (binary_b1(), WalkAttributionConfig(selected_class=False))):
        with torch.no_grad():
            module.model.hetero_convs[1][FK].lin.weight.fill_(value)
        row = row_for(module, config)
        assert row['status'] == 'failed' and row['method'] == LEGACY_METHOD
        output = row['output']
        assert (output['head_output'] is not None) == finite_logit
        if output['binary'] is not None:
            assert (output['binary']['selected_logit'] is not None) == finite_logit
        json.dumps(row, allow_nan=False)
        validate_v2_row(row)


# Legacy v2 rows report the head sign inversion (ledger D11) #################

def test_legacy_rows_carry_the_head_inversion_diagnostic(row, binary_row):
    for built in (row, binary_row):
        diagnostics = built['diagnostics'][LEGACY_METHOD]
        assert sorted(diagnostics) == ['head_output_denominator', 'head_output_flipped']
        assert type(diagnostics['head_output_denominator']) is float
        assert diagnostics['head_output_flipped'] is False  # gamma 0: S is the folded logit
        assert diagnostics['head_output_denominator'] == pytest.approx(built['score'],
                                                                       abs=1e-12)
        validate_v2_row(json.loads(json.dumps(built, allow_nan=False)), b1_sample())


LEGACY_DIAGNOSTIC_MUTATIONS = {
    'missing': lambda r: r.update(diagnostics=None),
    'extra_key': lambda r: r['diagnostics'][LEGACY_METHOD].update(margin=1.0),
    'flag_disagrees': lambda r: r['diagnostics'][LEGACY_METHOD].update(head_output_flipped=True),
    'flag_type': lambda r: r['diagnostics'][LEGACY_METHOD].update(head_output_flipped=0),
    'denominator_type': lambda r: r['diagnostics'][LEGACY_METHOD].update(
        head_output_denominator=1),
    'denominator_sign': lambda r: r['diagnostics'][LEGACY_METHOD].update(
        head_output_denominator=-r['diagnostics'][LEGACY_METHOD]['head_output_denominator']),
}


@pytest.mark.parametrize('name', sorted(LEGACY_DIAGNOSTIC_MUTATIONS))
def test_legacy_head_inversion_mutations_are_rejected(row, name):
    with pytest.raises(WalkSchemaError):
        validate_v2_row(mutate(row, LEGACY_DIAGNOSTIC_MUTATIONS[name]), b1_sample())


# Accounting tolerance components (ledger D14) ################################

def components(row):
    ledger = row['ledger']
    return ledger['walk_vector_abs_mass'], ledger['intermediate_state_abs_mass']


def test_rows_record_the_tolerance_components(row, binary_row):
    for built in (row, binary_row):
        vectors, states = components(built)
        assert type(vectors) is float and type(states) is float and states > vectors > 0
        without = copy.deepcopy(built)
        del without['ledger']['walk_vector_abs_mass']
        del without['ledger']['intermediate_state_abs_mass']
        assert absolute_mass(built) == pytest.approx(absolute_mass(without) + vectors + states)
        validate_v2_row(without, b1_sample())  # rows written before D14 stay valid
    # Every walk is displayed with its vector: the mass is exactly theirs.
    assert binary_row['coverage']['total_walks'] == len(binary_row['walks'])
    assert components(binary_row)[0] == pytest.approx(math.fsum(
        abs(v) for w in binary_row['walks'] for v in w['input_relevance']))


TOLERANCE_MUTATIONS = {
    'negative_vectors': lambda r: r['ledger'].update(walk_vector_abs_mass=-1.0),
    'integer_states': lambda r: r['ledger'].update(intermediate_state_abs_mass=3),
    'vectors_below_displayed': lambda r: r['ledger'].update(walk_vector_abs_mass=0.0),
    'states_without_vectors': lambda r: r['ledger'].pop('walk_vector_abs_mass'),
    'null_vectors': lambda r: r['ledger'].update(walk_vector_abs_mass=None),
}


@pytest.mark.parametrize('name', sorted(TOLERANCE_MUTATIONS))
def test_tolerance_component_mutations_are_rejected(binary_row, name):
    with pytest.raises(WalkSchemaError):
        validate_v2_row(mutate(binary_row, TOLERANCE_MUTATIONS[name]), b1_sample())


def cancellation_module(mixing, hidden=1.0):
    """The P3 reviewer's cancellation cases (review E5 and probe 10).

    One customer, T = 1, identity activations, no biases. At gamma 0.25 the
    head's hidden neuron 0 (row (1, -1) on H_1 = (1, 1.25)) has a modified
    total of exactly 0 but nonzero relevance, so its shares are about +-3e8
    against a score below 1. With an identity root they reach the walk
    vector (E5, and V4 with hidden weight 0.7); with the channel-mixing root
    [[0.5, 0.5], [0.625, 0.625]] on x = (1, 1) they cancel at the root step
    and the walk vector stays small (V5 and V6, hidden weights 0.7 and 0.3).
    """
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(2, 1, metadata=(['customer'], []), aggr='mean', norm=None,
                             bias=False, learn_skip=mixing, act=None)
    module.head = NodePredHead(2, 1, hidden_channels=2, num_layers=2, norm=None,
                               param_keys=[], bias=False, act=None)
    module = module.double()
    with torch.no_grad():
        if mixing:
            module.model.root_skips[0].lin_dict['customer'].weight.copy_(
                torch.tensor([[0.5, 0.5], [0.625, 0.625]]))
        module.head.mlp.lins[0].weight.copy_(torch.tensor([[1.0, -1.0], [hidden, 0.0]]))
        module.head.mlp.lins[1].weight.copy_(torch.tensor([[1.0, 1.0]]))
    x = {'customer': torch.tensor([[1.0, 1.0] if mixing else [1.0, 1.25]],
                                  dtype=torch.float64)}
    return module.eval(), x


def cancellation_row(method, mixing, hidden):
    module, x = cancellation_module(mixing, hidden)
    config = WalkAttributionConfig(method=method, gamma=0.25, feature_vectors=True)
    with GraphCapture(module, config) as cap:
        hidden_rows = module.model(dict(x), {}, {'customer': [1, 0]}, {})
        logits = module.head(hidden_rows['customer'][:1]).detach()
    selection = Selection('regression', 0, 1, None)
    return explain_capture(cap, config=config, selection=selection,
                           output=output_block(selection, module, logits[0]),
                           sample={'nodes': {'customer': {'pkeys': [7], 'node_ids': [70]}}},
                           origin='original_forward_attribution', prediction_id=ID,
                           sample_sha256=SAMPLE_ID, numerics=row_numerics(NUMERICS, 'float64'))


@pytest.mark.parametrize('method', [LEGACY_METHOD, WEIGHT_GAMMA_METHOD])
@pytest.mark.parametrize('case,mixing,hidden', [
    ('E5', False, 1.0), ('V4', False, 0.7), ('V5', True, 0.7), ('V6', True, 0.3)])
def test_cancelling_rows_close_their_ledger_with_the_d14_mass(method, case, mixing, hidden):
    row = cancellation_row(method, mixing, hidden)
    assert row['status'] == 'available', row.get('reason')
    validate_v2_row(row)
    error = abs(row['ledger']['accounting_error'])
    vectors, states = components(row)
    mass = absolute_mass(row)
    assert error <= tolerance(mass)
    # Without the D14 components these correct rows would fail ...
    assert error > tolerance(mass - vectors - states)
    if mixing:
        # ... and with only the walk vectors too: the magnitude cancelled
        # before H_0, so only the intermediate states carry it.
        assert error > tolerance(mass - states)
        assert vectors < 2.0 and states > 1e8
    else:
        assert vectors > 1e8


class LeakyRule(ContributionGammaRule):
    """Not conserving: the last child of every transfer loses `leak`."""
    leak = 0.0

    def transfer(self, layer, destination, state, incoming):
        children, absorbed = super().transfer(layer, destination, state, incoming)
        children[-1] = children[-1] * (1.0 - self.leak)
        return children, absorbed


def leaky_row(seed, leak):
    module, run = random_b1(seed, head_layers=2, head_bias=True)
    cap, logits = capture(module, run)
    computation = cap.computation()
    selection = Selection('regression', 0, 1, None)
    config = WalkAttributionConfig(feature_vectors=True, gamma=0.25)
    rule = LeakyRule(computation, selection, config)
    rule.leak = leak
    enumeration = enumerate_walks(computation, rule, max_walks=config.max_walks)
    sample = {'nodes': {table: {'pkeys': list(range(len(h))), 'node_ids': list(range(len(h)))}
                        for table, h in computation.h0.items()}}
    return build_row(computation, enumeration,
                     legacy_summarize(computation, selection, config, enumeration),
                     config=config, selection=selection,
                     output=output_block(selection, module, logits[0].detach()),
                     qualification=[], architecture=architecture_record(module),
                     operations=operations_record(module), sample=sample,
                     origin='original_forward_attribution', prediction_id=ID,
                     sample_sha256=SAMPLE_ID, numerics=row_numerics(NUMERICS, 'float64'))


@pytest.mark.parametrize('seed', range(5))
def test_a_non_conserving_rule_still_fails_on_ordinary_rows(seed):
    exact = leaky_row(seed, 0.0)
    assert exact['status'] == 'available'
    validate_v2_row(exact)
    for leak in (1.0, 1e-5):  # a dropped share, and a leak of 1e-5 of one share
        broken = leaky_row(seed, leak)
        assert broken['status'] == 'failed'
        assert broken['reason'] == 'accounting_error_exceeded'


def test_methods_without_the_hook_record_only_the_vector_mass():
    """A rule without the `absolute_mass` hook (degree-T, the toy rules) gains
    the walk vectors only, and the state component is recorded as null."""
    row, enumeration, _ = toy_rows(WalkAttributionConfig(method=DEGREE_T_METHOD))
    assert enumeration.intermediate_state_abs_mass is None
    assert components(row) == (0.0, None)  # the toy rule has no vectors either
    validate_v2_row(row, method_checks=False)
    # B1 (bias-free, f = 9 on eight walks of nonnegative vectors).
    built = row_for(b1_module(), WalkAttributionConfig(method=DEGREE_T_METHOD,
                                                       feature_vectors=True))
    assert built['status'] == 'available', built.get('reason')
    vectors, states = components(built)
    assert states is None and vectors == pytest.approx(9.0)
    validate_v2_row(built, b1_sample())
