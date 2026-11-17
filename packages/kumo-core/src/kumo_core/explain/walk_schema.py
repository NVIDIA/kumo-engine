# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`walk_attribution/v2` rows: construction and validation (frozen interface).

Rows are built only here, for original and offline paths alike, from a
`WalkComputation`, an `Enumeration` and the method's `summarize` output.
This module ranks and displays walks, binds source identities, aggregates
sources, groups walks into bags from their walk keys and closes the ledger
`score = displayed + omitted_display + omitted_search + bias + stabilizer +
precision_residual + accounting_error`. A conserved ledger is accounting,
not evidence of faithfulness, causality or predictive quality.
"""
from collections import Counter
import math
import re

import numpy as np

from kumo_core.explain.walk_attribution import (
    DEGREE_T_METHOD, METHOD_MODULES, SCHEMA_V2, UnsupportedWalkAttribution,
    WalkAttributionConfig, method_module,
)
from kumo_core.explain.walk_binary import same_label, selected_score, sigmoid
from kumo_core.explain.walk_enumerate import enumerate_walks
from kumo_core.explain.walk_graph import (
    KINDS, architecture_record, operations_record,
)


class WalkSchemaError(ValueError):
    """A walk_attribution/v2 row violates the contract."""


ORIGINS = ('original_forward_attribution', 'retained_graph_recomputation')
STATUSES = ('available', 'incomplete', 'unsupported', 'failed')
DECOMPOSITIONS = ('exact_degree_t', 'piecewise_gnn_gi', 'conservative_lrp')
REASONS = ('method_is_lrp', 'biases_present', 'enumeration_incomplete',
           'boundary_piece', 'mask_disagreement', 'precision_residual_exceeded')
LEDGER_ITEMS = ('displayed_walk_relevance', 'omitted_display_relevance',
                'omitted_search_relevance', 'bias_relevance',
                'stabilizer_residual', 'precision_residual')
SETTINGS_KEYS = ('gamma', 'gamma_rule', 'bias_rule', 'epsilon', 'stabilizer',
                 'identity_root_rule', 'sign_folding')
NUMERICS_KEYS = ('precision_mode', 'replay_precision_mode',
                 'mkldnn_matmul_fp32_precision', 'num_threads', 'cpu_capability',
                 'torch_version', 'model_dtype', 'compute_dtype')
BUDGET_UNIT = 'message_passing_route_expansions'
RELATIVE_TOLERANCE = 1e-9
_ID = re.compile(r'[a-f0-9]{64}\Z')


def tolerance(mass):
    """`tol(A) = 1e-9 * max(1, A)` for an absolute ledger mass `A` (4.10)."""
    return RELATIVE_TOLERANCE * max(1.0, mass)


# Walk keys and bags ##########################################################

def step_key(route):
    """`(mp_layer, kind, source table, relation or '', destination table,
    edge_occurrence or occurrence)` (C6)."""
    if route.kind == 'relation':
        source, relation, destination = route.relation
        return (route.layer, 'relation', source, relation, destination,
                route.edge_occurrence)
    table, occurrence = route.destination
    return (route.layer, 'self_update', table, '', table, occurrence)


def walk_key(steps):
    return tuple(step_key(route) for route in steps)


def row_step_key(step):
    """`step_key` of a serialized step."""
    if step['kind'] == 'relation':
        source, relation, destination = step['relation']
        return (step['mp_layer'], 'relation', source, relation, destination,
                step['edge_occurrence'])
    return (step['mp_layer'], 'self_update', step['table'], '', step['table'],
            step['occurrence'])


def bag_signature(keys):
    """Sorted `(variable, multiplicity)` pairs of a walk's tied variables.

    A variable is a step key without `mp_layer`: `(relation, edge)` or
    `(table, occurrence)` for self updates. Walks with equal multisets share
    a bag; forward and `rev_` edges are distinct variables.
    """
    return tuple(sorted(Counter(key[1:] for key in keys).items()))


def variable_record(variable, multiplicity):
    kind, source, relation, destination, occurrence = variable
    if kind == 'relation':
        return {'kind': 'relation', 'relation': [source, relation, destination],
                'edge_occurrence': occurrence, 'multiplicity': multiplicity}
    return {'kind': 'self_update', 'table': source, 'occurrence': occurrence,
            'multiplicity': multiplicity}


def variable_from_record(record):
    if record['kind'] == 'relation':
        source, relation, destination = record['relation']
        return ('relation', source, relation, destination, record['edge_occurrence'])
    if record['kind'] != 'self_update':
        raise WalkSchemaError('Unknown bag variable kind')
    return ('self_update', record['table'], '', record['table'], record['occurrence'])


def group_bags(walks):
    """All bags of `walks`, sorted by absolute relevance, ties by signature.

    Returns `[(signature, members, relevance)]`; bag relevance is the plain
    sum of its walk scores (no `1/alpha!`, which belongs only to the tied-gate
    derivative form).
    """
    groups = {}
    for walk in walks:
        groups.setdefault(bag_signature(walk_key(walk.steps)), []).append(walk)
    bags = [(signature, members, math.fsum(w.relevance for w in members))
            for signature, members in groups.items()]
    bags.sort(key=lambda bag: (-abs(bag[2]), bag[0]))
    return bags


def rank_walks(walks):
    """Absolute relevance descending, ties by ascending walk key (C6)."""
    return sorted(walks, key=lambda walk: (-abs(walk.relevance), walk_key(walk.steps)))


# Rows ########################################################################

def _pkey(sample, table, occurrence):
    return sample['nodes'][table]['pkeys'][occurrence]


def _node_id(sample, table, occurrence):
    ids = sample['nodes'][table].get('node_ids')
    return None if ids is None else ids[occurrence]


def _step_record(route, sample):
    if route.kind == 'relation':
        source, relation, destination = route.relation
        return {'mp_layer': route.layer, 'kind': 'relation',
                'relation': [source, relation, destination],
                'direction': route.direction, 'operator': route.operator,
                'denominator': route.denominator,
                'edge_occurrence': route.edge_occurrence,
                'source_occurrence': route.source[1],
                'destination_occurrence': route.destination[1],
                'source_pkey': _pkey(sample, source, route.source[1]),
                'destination_pkey': _pkey(sample, destination, route.destination[1])}
    table, occurrence = route.destination
    return {'mp_layer': route.layer, 'kind': 'self_update', 'table': table,
            'operator': route.operator, 'denominator': route.denominator,
            'occurrence': occurrence, 'pkey': _pkey(sample, table, occurrence)}


def coordinate(selection):
    return ('selected_class_logit' if selection.task_type == 'binary_classification'
            else 'encoded_model_output')


def minimal_row(status, reason, *, config, origin, prediction_id, sample_sha256,
                **extra):
    """An `unsupported` or `failed` row: identity, method and config (4.2)."""
    row = {'schema': SCHEMA_V2, 'status': status, 'reason': reason,
           'origin': origin, 'prediction_id': prediction_id,
           'sample_sha256': sample_sha256, 'method': config.method,
           'config': config.to_dict()}
    row.update({key: value for key, value in extra.items() if value is not None})
    return row


def build_row(computation, enumeration, summary, *, config, selection, output,
              qualification, architecture, operations, sample, origin,
              prediction_id, sample_sha256, numerics, extra=None):
    """Assemble a v2 row; a ledger that does not close fails the row."""
    k = selection.output_index
    score = selected_score(output)
    if score != selection.logit_sign * float(computation.head_output[k]):
        raise ValueError('Captured head output differs from the scored logit')
    ranked = rank_walks(enumeration.walks)
    displayed, hidden = ranked[:config.top_k], ranked[config.top_k:]
    complete = enumeration.complete
    bag_index, bag_rows = {}, None
    if not config.bags:
        bag_status = 'not_requested'
    elif not complete:
        bag_status = 'omitted_incomplete_enumeration'
    else:
        bag_status, bag_rows = 'complete', []
        for index, (signature, members, relevance) in enumerate(
                group_bags(enumeration.walks)):
            bag_index[signature] = index
            bag_rows.append({'index': index, 'relevance': relevance,
                             'walk_count': len(members),
                             'variables': [variable_record(v, m) for v, m in signature]})
    walks = []
    for rank, walk in enumerate(displayed):
        table, occurrence = walk.source
        record = {'rank': rank, 'relevance': walk.relevance, 'source_table': table,
                  'source_occurrence': occurrence,
                  'source_pkey': _pkey(sample, table, occurrence),
                  'source_node_id': _node_id(sample, table, occurrence),
                  'bag': (bag_index[bag_signature(walk_key(walk.steps))]
                          if bag_rows is not None else None),
                  'steps': [_step_record(route, sample) for route in walk.steps]}
        if config.feature_vectors:
            record['input_relevance'] = walk.input_relevance.tolist()
        walks.append(record)
    shown = Counter(walk.source for walk in displayed)
    by_source = {}
    for walk in enumeration.walks:
        by_source.setdefault(walk.source, []).append(walk)
    sources = []
    for table, occurrence in sorted(by_source):
        members = by_source[(table, occurrence)]
        entry = {'table': table, 'occurrence': occurrence,
                 'pkey': _pkey(sample, table, occurrence),
                 'node_id': _node_id(sample, table, occurrence),
                 'walk_count': len(members),
                 'displayed_walk_count': shown.get((table, occurrence), 0),
                 'relevance': math.fsum(w.relevance for w in members)}
        if config.feature_vectors:
            entry['input_value'] = computation.h0[table][occurrence].tolist()
            columns = zip(*(w.input_relevance.tolist() for w in members))
            entry['input_relevance'] = [math.fsum(column) for column in columns]
        sources.append(entry)
    masses = [prefix.mass for prefix in enumeration.prefixes]
    bias_terms = [v for kind in KINDS for v in enumeration.bias_terms[kind]]
    stabilizer_terms = [v for kind in KINDS for v in enumeration.stabilizer_terms[kind]]
    precision_residual = float(summary['precision_residual'])
    ledger = {
        'displayed_walk_relevance': math.fsum(w.relevance for w in displayed),
        'omitted_display_relevance': math.fsum(w.relevance for w in hidden),
        'omitted_display_abs_relevance': math.fsum(abs(w.relevance) for w in hidden),
        'omitted_search_relevance': math.fsum(masses),
        'omitted_search_abs_prefix_mass': math.fsum(abs(m) for m in masses),
        'bias_relevance': math.fsum(bias_terms),
        'bias_relevance_by_kind': dict(enumeration.bias),
        'stabilizer_residual': math.fsum(stabilizer_terms),
        'stabilizer_residual_by_kind': dict(enumeration.stabilizer),
        'precision_residual': precision_residual,
        # Score minus every other item, summed exactly from the parts.
        'accounting_error': math.fsum(
            [score, -precision_residual] + [-w.relevance for w in enumeration.walks]
            + [-m for m in masses] + [-v for v in bias_terms]
            + [-v for v in stabilizer_terms]),
    }
    status = 'available' if complete else 'incomplete'
    coverage = {
        'status': ('budget_exhausted' if not complete else
                   'complete_top_k' if hidden else 'complete'),
        'enumeration_complete': complete, 'total_walks': len(enumeration.walks),
        'displayed_walks': len(displayed), 'expanded_routes': enumeration.expanded_routes,
        'unexpanded_prefixes': len(enumeration.prefixes),
        'max_walks': config.max_walks, 'top_k': config.top_k,
        'budget_unit': BUDGET_UNIT, 'bags': bag_status}
    exactness = dict(summary['exactness'])
    exactness['qualification'] = (list(qualification) + ['single_edge_gathers']
                                  + list(exactness.get('qualification') or []))
    diagnostics = summary.get('diagnostics')
    if diagnostics is not None:
        _require(isinstance(diagnostics, dict) and _json_safe(diagnostics),
                 'Method diagnostics must be a JSON-safe dict')
        diagnostics = {config.method: diagnostics}
    seed_table, seed_occurrence = computation.seed
    row = {
        'schema': SCHEMA_V2, 'status': status, 'reason': None, 'origin': origin,
        'prediction_id': prediction_id, 'sample_sha256': sample_sha256,
        'method': config.method, 'config': config.to_dict(),
        'method_settings': dict(summary['method_settings']),
        'coordinate': coordinate(selection),
        'input_coordinate': 'encoded_node_features',
        'train_table_name': computation.train_table_name,
        'entity_table_name': computation.entity_table_name,
        'score': score, 'output': output,
        'seed': {'table': seed_table, 'occurrence': seed_occurrence,
                 'pkey': _pkey(sample, seed_table, seed_occurrence),
                 'node_id': _node_id(sample, seed_table, seed_occurrence)},
        'numerics': numerics, 'architecture': architecture,
        'operations': operations, 'walks': walks, 'sources': sources,
        'bags': bag_rows, 'ledger': ledger, 'coverage': coverage,
        'exactness': exactness,
        # Method-supplied diagnostics, namespaced by method ID (ledger D11).
        'diagnostics': diagnostics,
    }
    row.update(extra or {})
    # Failed rows keep the same context as the failures of `explain_capture`.
    failure = dict(config=config, origin=origin, prediction_id=prediction_id,
                   sample_sha256=sample_sha256, coordinate=row['coordinate'],
                   input_coordinate=row['input_coordinate'], output=output,
                   numerics=numerics, **(extra or {}))
    if not _json_safe(ledger) or not all(math.isfinite(w.relevance)
                                         for w in enumeration.walks):
        # A nonfinite intermediate fails the row (contract 4.2).
        return minimal_row('failed', 'nonfinite_relevance', **failure)
    if not abs(ledger['accounting_error']) <= tolerance(absolute_mass(row)):
        return minimal_row('failed', 'accounting_error_exceeded', ledger=ledger, **failure)
    return row


def absolute_mass(row):
    """`A`: absolute ledger mass of a row (4.10)."""
    ledger = row['ledger']
    return math.fsum(
        [abs(w['relevance']) for w in row['walks']]
        + [ledger['omitted_display_abs_relevance'],
           ledger['omitted_search_abs_prefix_mass'], abs(ledger['precision_residual'])]
        + [abs(v) for v in ledger['bias_relevance_by_kind'].values()]
        + [abs(v) for v in ledger['stabilizer_residual_by_kind'].values()])


def row_numerics(numerics, computation_dtype, *, replay_precision_mode=None):
    """Row `numerics`: settings at capture or replay plus dtypes (4.2)."""
    numerics = dict(numerics or {})
    return {'precision_mode': numerics.get('precision_mode'),
            'replay_precision_mode': replay_precision_mode,
            'mkldnn_matmul_fp32_precision': numerics.get('mkldnn_matmul_fp32_precision'),
            'num_threads': numerics.get('num_threads'),
            'cpu_capability': numerics.get('cpu_capability'),
            'torch_version': numerics.get('torch_version'),
            'model_dtype': computation_dtype, 'compute_dtype': 'float64'}


def explain_capture(capture, *, config, selection, output, sample, origin,
                    prediction_id, sample_sha256, numerics, seed_index=0,
                    extra=None):
    """One v2 row from a completed `GraphCapture`; never raises.

    `numerics` are the row numerics of `row_numerics`. Unsupported graph
    facts give `unsupported` rows and every other error a `failed` row, so a
    prediction is never lost to its explanation.
    """
    identity = dict(config=config, origin=origin, prediction_id=prediction_id,
                    sample_sha256=sample_sha256)
    context = dict(coordinate=coordinate(selection),
                   input_coordinate='encoded_node_features', output=output,
                   numerics=numerics, **(extra or {}))
    try:
        computation = capture.computation(seed_index, numerics)
        method = method_module(config.method)
        rule = method.make_rule(computation, selection, config)
        enumeration = enumerate_walks(computation, rule, max_walks=config.max_walks)
        summary = method.summarize(computation, selection, config, enumeration)
        row = build_row(computation, enumeration, summary, config=config,
                        selection=selection, output=output,
                        qualification=capture.qualification or [],
                        architecture=architecture_record(capture.module),
                        operations=operations_record(capture.module), sample=sample,
                        origin=origin, prediction_id=prediction_id,
                        sample_sha256=sample_sha256, numerics=numerics, extra=extra)
        if row['status'] in ('available', 'incomplete'):
            validate_v2_row(row, sample)
        return row
    except UnsupportedWalkAttribution as error:
        return minimal_row('unsupported', str(error), **identity, **context)
    except WalkSchemaError as error:
        return minimal_row('failed', f'row_validation_failed: {error}', **identity,
                           **context)
    except Exception as error:
        return minimal_row('failed', f'{type(error).__name__}: {error}', **identity,
                           **context)


# Validation ##################################################################

def _require(condition, message):
    if not condition:
        raise WalkSchemaError(message)


def _identifier(value, name):
    _require(isinstance(value, str) and _ID.fullmatch(value) is not None,
             f'{name} must be a lowercase SHA-256')


def _is_int(value):
    return type(value) is int


def _json_safe(value):
    """Plain JSON data with finite numbers only."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item)
                   for key, item in value.items())
    return False


def _check_finite(value, path='row'):
    if isinstance(value, float):
        _require(math.isfinite(value), f'Nonfinite number at {path}')
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f'{path}[{index}]')


def _close(actual, expected, mass):
    return abs(actual - expected) <= tolerance(mass)


def validate_v2_row(row, sample=None, *, method_checks=True):
    """Generic v2 checks (4.10), then the method's `validate_row`.

    With `sample`, bound keys and node ids must match the retained graph.
    Raises `WalkSchemaError`.
    """
    try:
        _validate(row, sample, method_checks)
    except WalkSchemaError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as error:
        raise WalkSchemaError(f'Malformed v2 row: {type(error).__name__}: {error}') from error


def _validate(row, sample, method_checks):
    _require(row.get('schema') == SCHEMA_V2, 'Not a walk_attribution/v2 row')
    status = row['status']
    _require(status in STATUSES, 'Invalid walk status')
    _require(row['origin'] in ORIGINS, 'Invalid walk origin')
    _identifier(row['prediction_id'], 'prediction_id')
    _require(row['method'] in METHOD_MODULES, 'Unregistered walk method')
    config = WalkAttributionConfig(**row['config'])
    _require(config.to_dict() == row['config'], 'Configuration is not canonical')
    _require(config.method == row['method'], 'Configuration and method disagree')
    if status in ('unsupported', 'failed'):
        _require(isinstance(row.get('reason'), str) and row['reason'], 'Missing reason')
        _require(not row.get('walks'), 'Rows without explanations carry no walks')
        if row.get('sample_sha256') is not None:
            _identifier(row['sample_sha256'], 'sample_sha256')
        return
    _identifier(row['sample_sha256'], 'sample_sha256')
    _require(row['reason'] is None, 'Explained rows carry no reason')
    _check_finite(row)
    _require(set(row['method_settings']) == set(SETTINGS_KEYS), 'method_settings keys')
    _require(set(row['numerics']) == set(NUMERICS_KEYS)
             and row['numerics']['compute_dtype'] == 'float64', 'numerics keys')
    if row['origin'] == 'original_forward_attribution':
        _require(row['numerics']['replay_precision_mode'] is None,
                 'Original rows have no replay precision')
    architecture = row['architecture']
    T, channels = architecture['message_passing_layers'], architecture['channels']
    _require(_is_int(T) and T >= 1 and _is_int(channels) and channels >= 1,
             'Invalid architecture shape')
    _validate_output(row, config)
    coverage, ledger, walks = row['coverage'], row['ledger'], row['walks']
    complete = coverage['enumeration_complete']
    _require(type(complete) is bool, 'enumeration_complete must be a bool')
    _validate_coverage(status, coverage, ledger, walks, config)
    keys = _validate_walks(row, walks, architecture, T)
    _validate_sources(row, walks, coverage, ledger, config, channels)
    _validate_bags(row, walks, keys, coverage, ledger, config, complete, T)
    _validate_ledger(row, walks, ledger)
    _validate_exactness(row, architecture, complete)
    diagnostics = row['diagnostics']
    _require(diagnostics is None or (
        isinstance(diagnostics, dict) and list(diagnostics) == [row['method']]
        and isinstance(diagnostics[row['method']], dict) and _json_safe(diagnostics)),
        'diagnostics must be a JSON-safe dict namespaced by the row method')
    if sample is not None:
        _validate_binding(row, sample)
    if method_checks:
        try:
            method_module(row['method']).validate_row(row)
        except WalkSchemaError:
            raise
        except ValueError as error:
            raise WalkSchemaError(f'Method check failed: {error}') from error


def _validate_output(row, config):
    output, score = row['output'], row['score']
    _require(output['output_index'] == config.output_index, 'Output index mismatch')
    _require(type(output['head_output']) is float, 'head_output must be a float')
    _require(row['input_coordinate'] == 'encoded_node_features', 'Input coordinate')
    if output['task_type'] == 'regression':
        _require(output['binary'] is None and row['coordinate'] == 'encoded_model_output',
                 'Regression rows explain the encoded model output')
        _require(score == output['head_output'], 'Score must be the head output')
        return
    _require(output['task_type'] == 'binary_classification', 'Unsupported task type')
    _require(row['coordinate'] == 'selected_class_logit', 'Binary rows explain a logit')
    binary = output['binary']
    sign = binary['logit_sign']
    _require(_is_int(sign) and sign in (1, -1), 'logit_sign must be +1 or -1')
    _require(binary['selected_logit'] == sign * output['head_output'] == score,
             'score == selected_logit == logit_sign * head_output must hold exactly')
    _require(abs(binary['model_probability_selected'] - sigmoid(binary['selected_logit']))
             <= 1e-12, 'model_probability_selected is not sigmoid(selected_logit)')
    mapping = binary['label_mapping']
    _require(len(mapping) == 2 and sorted(e['encoded_class'] for e in mapping) == [0, 1],
             'Binary label mapping must encode two labels as 0 and 1')
    positive = next(e['raw_label'] for e in mapping if e['encoded_class'] == 1)
    _require(same_label(binary['encoded_positive_label'], positive),
             'encoded_positive_label disagrees with the mapping')
    selected = binary['selected_class_label']
    _require(any(same_label(selected, e['raw_label']) for e in mapping),
             'selected_class_label is not a model label')
    _require((sign == 1) == same_label(selected, positive), 'logit_sign disagrees')
    _require(binary['selected_class_source'] in ('config', 'encoded_positive_default'),
             'Invalid selected_class_source')
    float32 = row['numerics']['model_dtype'] != 'float64'
    calibration = binary['calibration']
    _require(calibration['weight_floor'] == 0.001, 'Platt weight floor is 0.001')
    z = output['head_output']
    if calibration['fitted']:
        _require(calibration['kind'] == 'platt', 'Fitted calibration must be Platt')
        raw = calibration['weight_raw']
        expected = (float(np.maximum(np.float32(raw), np.float32(0.001))) if float32
                    else max(raw, 0.001))
        _require(calibration['weight_effective'] == expected, 'weight_effective')
        logit = calibration['weight_effective'] * z + calibration['bias']
    else:
        _require(calibration['kind'] == 'none' and calibration['weight_raw'] is None
                 and calibration['weight_effective'] is None
                 and calibration['bias'] is None, 'Unfitted calibration is kind none')
        logit = z
    positive_probability = binary['emitted_encoded_positive_probability']
    _require(0.0 <= positive_probability <= 1.0
             and abs(positive_probability - sigmoid(logit)) <= 1e-6,
             'Emitted probability disagrees with the calibrated logit')
    if sign == 1:
        _require(binary['emitted_probability_selected'] == positive_probability
                 and binary['emitted_column_selected'] == 'True_PROB',
                 'Selected probability must be the True_PROB column')
    else:
        negative = (float(np.float32(1.0) - np.float32(positive_probability)) if float32
                    else 1.0 - positive_probability)
        _require(binary['emitted_probability_selected'] == negative
                 and binary['emitted_column_selected'] == 'False_PROB',
                 'Selected probability must be the False_PROB column (C7)')


def _validate_coverage(status, coverage, ledger, walks, config):
    _require(coverage['max_walks'] == config.max_walks
             and coverage['top_k'] == config.top_k
             and coverage['budget_unit'] == BUDGET_UNIT, 'Coverage settings')
    expanded = coverage['expanded_routes']
    _require(_is_int(expanded) and 0 <= expanded <= config.max_walks,
             'expanded_routes must stay within max_walks')
    total = coverage['total_walks']
    _require(_is_int(total) and coverage['displayed_walks'] == len(walks)
             == min(config.top_k, total), 'Displayed walk count')
    unexpanded = coverage['unexpanded_prefixes']
    _require(_is_int(unexpanded), 'unexpanded_prefixes must be an int')
    if status == 'available':
        _require(coverage['enumeration_complete'] and unexpanded == 0
                 and coverage['status'] in ('complete', 'complete_top_k'),
                 'Available rows have a complete search')
        _require(ledger['omitted_search_relevance'] == 0
                 and ledger['omitted_search_abs_prefix_mass'] == 0,
                 'A complete search omits nothing')
    else:
        _require(not coverage['enumeration_complete'] and unexpanded >= 1
                 and coverage['status'] == 'budget_exhausted',
                 'Incomplete rows exhausted the budget')
    if total <= config.top_k:
        _require(ledger['omitted_display_relevance'] == 0
                 and ledger['omitted_display_abs_relevance'] == 0,
                 'Nothing was hidden from display')
        _require(coverage['status'] != 'complete_top_k', 'complete_top_k hides walks')
    elif status == 'available':
        _require(coverage['status'] == 'complete_top_k', 'Hidden walks need complete_top_k')


def _validate_walks(row, walks, architecture, T):
    relations = {tuple(r['relation']): (r['direction'], r['operator'])
                 for r in architecture['relations']}
    seed = (row['seed']['table'], row['seed']['occurrence'])
    _require(row['seed']['table'] == row['entity_table_name'], 'Seed table')
    keys = []
    for rank, walk in enumerate(walks):
        _require(walk['rank'] == rank, 'Walk ranks must be 0-based and ordered')
        steps = walk['steps']
        _require(isinstance(steps, list) and len(steps) == T,
                 'A walk has exactly T message-passing steps')
        _require(_is_int(walk['source_occurrence']) and walk['source_occurrence'] >= 0,
                 'Invalid source occurrence')
        current = (walk['source_table'], walk['source_occurrence'])
        for t, step in enumerate(steps, start=1):
            _require(step['mp_layer'] == t, 'Steps must cover mp_layer 1..T in order')
            if step['kind'] == 'relation':
                relation = tuple(step['relation'])
                _require(len(relation) == 3 and relations.get(relation)
                         == (step['direction'], step['operator']),
                         'Relation direction or operator disagrees with the model')
                for name in ('edge_occurrence', 'source_occurrence',
                             'destination_occurrence', 'denominator'):
                    _require(_is_int(step[name]) and step[name] >= 0, f'Invalid {name}')
                _require(step['denominator'] >= 1 and (
                    step['operator'] == 'mean' or step['denominator'] == 1),
                    'Only mean steps have denominators above 1')
                source = (relation[0], step['source_occurrence'])
                destination = (relation[2], step['destination_occurrence'])
            elif step['kind'] == 'self_update':
                _require(_is_int(step['occurrence']) and step['occurrence'] >= 0,
                         'Invalid self occurrence')
                _require(step['operator'] == architecture['root']
                         and step['denominator'] == 1, 'Self updates use the root')
                source = destination = (step['table'], step['occurrence'])
            else:
                raise WalkSchemaError(f'Unknown step kind {step["kind"]!r}')
            _require(source == current, 'Consecutive steps must be contiguous')
            current = destination
        _require(current == seed, 'A walk must end at the seed occurrence')
        keys.append(tuple(row_step_key(step) for step in steps))
    order = [(-abs(w['relevance']), key) for w, key in zip(walks, keys)]
    _require(order == sorted(order), 'Walks must be ranked by |relevance|, then walk key')
    _require(len(set(keys)) == len(keys), 'Duplicate walks')
    return keys


def _validate_sources(row, walks, coverage, ledger, config, channels):
    sources = row['sources']
    keys = [(s['table'], s['occurrence']) for s in sources]
    _require(keys == sorted(keys) and len(set(keys)) == len(keys),
             'Sources must be unique and sorted by (table, occurrence)')
    by_source = dict(zip(keys, sources))
    shown = Counter((w['source_table'], w['source_occurrence']) for w in walks)
    _require(set(shown) <= set(by_source), 'Every walk source is listed')
    for key, source in by_source.items():
        _require(_is_int(source['walk_count']) and source['walk_count'] >= 1,
                 'Source walk counts')
        _require(source['displayed_walk_count'] == shown.get(key, 0),
                 'Source displayed walk counts')
    _require(sum(s['walk_count'] for s in sources) == coverage['total_walks'],
             'Sources must cover every enumerated walk')
    every = math.fsum([ledger['displayed_walk_relevance'],
                       ledger['omitted_display_relevance']])
    mass = math.fsum(abs(s['relevance']) for s in sources)
    _require(_close(math.fsum(s['relevance'] for s in sources), every, mass),
             'Source relevance totals disagree with the walks')
    if not config.feature_vectors:
        _require(all('input_relevance' not in w for w in walks)
                 and all('input_value' not in s and 'input_relevance' not in s
                         for s in sources), 'Vectors were not requested')
        return
    for source in sources:
        _require(len(source['input_value']) == channels
                 and len(source['input_relevance']) == channels, 'Source vector length')
        _check_vector(source['input_relevance'], source['input_value'],
                      source['relevance'])
    for walk in walks:
        vector = walk['input_relevance']
        _require(len(vector) == channels, 'Walk vector length')
        source = by_source[(walk['source_table'], walk['source_occurrence'])]
        _check_vector(vector, source['input_value'], walk['relevance'])


def _check_vector(vector, values, relevance):
    _require(_close(math.fsum(vector), relevance, math.fsum(abs(v) for v in vector)),
             'Vector does not sum to its relevance')
    _require(all(r == 0 for r, v in zip(vector, values) if v == 0),
             'Relevance on an inactive input channel must be exactly 0')


def _validate_bags(row, walks, keys, coverage, ledger, config, complete, T):
    bags = row['bags']
    if not config.bags:
        _require(coverage['bags'] == 'not_requested' and bags is None,
                 'Bags were not requested')
    elif not complete:
        _require(coverage['bags'] == 'omitted_incomplete_enumeration' and bags is None,
                 'Bags need a complete enumeration')
    else:
        _require(coverage['bags'] == 'complete' and isinstance(bags, list),
                 'Complete bags are listed')
    if bags is None:
        _require(all(w['bag'] is None for w in walks), 'Walks name no bag')
        return
    signatures = []
    for index, bag in enumerate(bags):
        _require(bag['index'] == index, 'Bag indices are positions')
        signature = tuple(sorted((variable_from_record(v), v['multiplicity'])
                                 for v in bag['variables']))
        _require(all(_is_int(m) and m >= 1 for _, m in signature)
                 and sum(m for _, m in signature) == T,
                 'Bag multiplicities must sum to T')
        _require(len({v for v, _ in signature}) == len(signature), 'Duplicate variable')
        signatures.append(signature)
    _require(len(set(signatures)) == len(signatures), 'Duplicate bags')
    order = [(-abs(b['relevance']), s) for b, s in zip(bags, signatures)]
    _require(order == sorted(order), 'Bags are ranked by |relevance|, then variables')
    _require(sum(b['walk_count'] for b in bags) == coverage['total_walks'],
             'Bags must cover every enumerated walk')
    every = math.fsum([ledger['displayed_walk_relevance'],
                       ledger['omitted_display_relevance']])
    mass = math.fsum(abs(b['relevance']) for b in bags)
    _require(_close(math.fsum(b['relevance'] for b in bags), every, mass),
             'Bag relevance totals disagree with the walks')
    position = {signature: index for index, signature in enumerate(signatures)}
    for walk, key in zip(walks, keys):
        _require(walk['bag'] == position.get(bag_signature(key)),
                 'A walk is linked to the wrong bag')


def _validate_ledger(row, walks, ledger):
    for name in ('bias_relevance', 'stabilizer_residual'):
        by_kind = ledger[f'{name}_by_kind']
        _require(set(by_kind) == set(KINDS), f'{name}_by_kind must list every kind')
        _require(_close(math.fsum(by_kind.values()), ledger[name],
                        math.fsum(abs(v) for v in by_kind.values())),
                 f'{name}_by_kind does not sum to {name}')
    mass = absolute_mass(row)
    _require(_close(math.fsum(w['relevance'] for w in walks),
                    ledger['displayed_walk_relevance'], mass), 'Displayed relevance sum')
    _require(ledger['omitted_display_abs_relevance'] >= abs(
        ledger['omitted_display_relevance']) - tolerance(mass), 'Display omission mass')
    _require(ledger['omitted_search_abs_prefix_mass'] >= abs(
        ledger['omitted_search_relevance']) - tolerance(mass), 'Search omission mass')
    remainder = math.fsum([row['score']] + [-ledger[item] for item in LEDGER_ITEMS])
    _require(abs(remainder - ledger['accounting_error']) <= tolerance(mass),
             'Ledger identity does not close')
    _require(abs(ledger['accounting_error']) <= tolerance(mass),
             'accounting_error exceeds tol(A)')


def _validate_exactness(row, architecture, complete):
    exactness, method = row['exactness'], row['method']
    _require(exactness['decomposition'] in DECOMPOSITIONS, 'Unknown decomposition')
    claim, reasons = exactness['degree_t_claim'], exactness['reasons']
    _require(type(claim) is bool and isinstance(reasons, list)
             and all(r in REASONS for r in reasons), 'Exactness reasons')
    _require(claim == (not reasons), 'reasons is empty iff the claim holds')
    _require(isinstance(exactness['qualification'], list)
             and all(isinstance(c, str) for c in exactness['qualification']),
             'Qualification codes')
    if not complete:
        _require('enumeration_incomplete' in reasons, 'Incomplete search has a reason')
    if architecture['gnn_bias'] or architecture['head_bias']:
        _require('biases_present' in reasons, 'Biased models carry biases_present')
    if claim:
        _require(method == DEGREE_T_METHOD and exactness['decomposition'] == 'exact_degree_t'
                 and complete and exactness['piece'] == 'interior'
                 and exactness['boundary_neurons'] == 0
                 and exactness['mask_disagreements'] == 0
                 and not architecture['gnn_bias'] and not architecture['head_bias'],
                 'A degree-T claim needs GNN-GI, complete search, an interior piece '
                 'and a bias-free model')
    if method != DEGREE_T_METHOD:
        _require(exactness['decomposition'] == 'conservative_lrp' and not claim
                 and 'method_is_lrp' in reasons
                 and all(exactness[name] is None for name in (
                     'piece', 'boundary_neurons', 'mask_disagreements',
                     'relevant_gated_neurons', 'min_abs_gated_preactivation')),
                 'LRP rows are conservative decompositions')
        return
    _require(exactness['decomposition'] in ('exact_degree_t', 'piecewise_gnn_gi')
             and (exactness['decomposition'] == 'exact_degree_t') == claim
             and 'method_is_lrp' not in reasons, 'GNN-GI decomposition label')
    counts = [exactness[name] for name in ('boundary_neurons', 'mask_disagreements',
                                           'relevant_gated_neurons')]
    _require(all(_is_int(c) and c >= 0 for c in counts), 'GNN-GI neuron counts')
    boundary, disagreements, gated = counts
    _require(exactness['piece'] in ('interior', 'boundary')
             and (exactness['piece'] == 'interior') == (boundary == 0 and disagreements == 0),
             'piece is interior iff no boundary neuron and no mask disagreement')
    margin = exactness['min_abs_gated_preactivation']
    _require((margin is None) == (gated == 0)
             and (margin is None or (type(margin) is float and margin >= 0)),
             'min_abs_gated_preactivation is null exactly when nothing is gated')
    _require(('boundary_piece' in reasons) == (boundary > 0)
             and ('mask_disagreement' in reasons) == (disagreements > 0),
             'Boundary and disagreement reasons follow the counts')
    residual = abs(row['ledger']['precision_residual'])
    _require(('precision_residual_exceeded' in reasons)
             == (residual > precision_limit(row)),
             'precision_residual_exceeded follows the mode tolerance')


def precision_limit(row):
    """GNN-GI claim tolerance on `precision_residual` (4.10): 1e-5 under
    `highest`, 1e-2 under `medium`, `high` or an unrecorded mode, times
    `max(1, |score|)`, for the mode of the forward that produced the score."""
    numerics = row['numerics']
    mode = (numerics['replay_precision_mode']
            if row['origin'] == 'retained_graph_recomputation'
            else numerics['precision_mode'])
    return (1e-5 if mode == 'highest' else 1e-2) * max(1.0, abs(row['score']))


def _validate_binding(row, sample):
    def check(table, occurrence, pkey, node_id=False, expected_id=None):
        _require(sample['nodes'][table]['pkeys'][occurrence] == pkey,
                 'Bound key differs from the retained graph')
        if node_id:
            ids = sample['nodes'][table].get('node_ids')
            _require((None if ids is None else ids[occurrence]) == expected_id,
                     'Bound node id differs from the retained graph')

    seed = row['seed']
    check(seed['table'], seed['occurrence'], seed['pkey'], True, seed['node_id'])
    for walk in row['walks']:
        check(walk['source_table'], walk['source_occurrence'], walk['source_pkey'],
              True, walk['source_node_id'])
        for step in walk['steps']:
            if step['kind'] == 'relation':
                source, _, destination = step['relation']
                check(source, step['source_occurrence'], step['source_pkey'])
                check(destination, step['destination_occurrence'],
                      step['destination_pkey'])
            else:
                check(step['table'], step['occurrence'], step['pkey'])
    for source in row['sources']:
        check(source['table'], source['occurrence'], source['pkey'], True,
              source['node_id'])
