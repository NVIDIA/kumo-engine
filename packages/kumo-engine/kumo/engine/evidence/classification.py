# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted binary classification allocations from one retained execution."""
import dataclasses
import hashlib
import math


def _capture_digest(value):
    """Bind tensor bytes, coordinates and fitted field metadata of a capture."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(
                str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif dataclasses.is_dataclass(item):
            visit({
                f.name: getattr(item, f.name)
                for f in dataclasses.fields(item)
            })
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode() + b'\0')

    visit(value)
    return digest.hexdigest()


def _check_score(actual, expected, *, name, atol=1e-5, rtol=1e-5):
    if (not math.isfinite(actual) or not math.isfinite(expected)
            or not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol)):
        raise ValueError(f'{name} replay mismatch')


def _summarize(computation, fields, sample, *, sign, max_walks, top_k):
    """Allocate all fields independently of the bounded path display."""
    from kumo_core.explain.walk_lrp_epsilon import (
        SMOOTH_FLOOR_POLICY_ID,
        enumerate_canonical_walks,
    )
    walks = enumerate_canonical_walks(computation, sign=sign,
                                      max_walks=max_walks, top_k=top_k,
                                      policy_id=SMOOTH_FLOOR_POLICY_ID)
    relevance = walks.merged
    allocated = fields.allocate(relevance.h0, h0=computation.h0,
                                invalid=relevance.invalid_h0,
                                policy_id=SMOOTH_FLOOR_POLICY_ID)
    if not all(allocated.complete.values()):
        raise ValueError('Incomplete fitted-field allocation')
    ledger = {
        key: float(value)
        for key, value in relevance.ledger().items() if key != 'encoded_inputs'
    }
    ledger['fields'] = math.fsum(float(v) for v in allocated.fields.values())
    for split in allocated.ledgers.values():
        for key, values in (('bias', split.biases), ('constant',
                                                     split.constants)):
            ledger[key] += math.fsum(float(v.sum()) for v in values)
        for key in ('stabilization', 'rejected', 'residual'):
            ledger[key] += float(getattr(split, key).sum())
    if not all(math.isfinite(v) for v in ledger.values()):
        raise ValueError('Nonfinite terminal accounting')
    _check_score(math.fsum(ledger.values()), float(relevance.score),
                 name='terminal accounting', atol=1e-9, rtol=1e-9)

    def identity(node):
        table, occurrence = node
        return {
            'table': table,
            'occurrence': occurrence,
            'primary_key': sample['nodes'][table]['pkeys'][occurrence]
        }

    rows = [{
        **identity((table, occurrence)), 'field': name,
        'relevance': float(value),
        **allocated.metadata[table, occurrence, name]
    } for (table, occurrence, name), value in allocated.fields.items()]
    records = [{
        **identity(node), 'field_relevance':
        math.fsum(row['relevance'] for row in rows
                  if (row['table'], row['occurrence']) == node),
        'constant_relevance':
        float(allocated.constants.get(node, 0.))
    } for node in sorted(relevance.h0)]
    paths = []
    for index in walks.displayed_indices:
        walk = walks.enumeration.walks[index]
        steps = [{
            key: getattr(route, key)
            for key in ('layer', 'kind', 'relation', 'direction', 'operator',
                        'edge_occurrence')
        }
                 | {
                     'source': identity(route.source),
                     'destination': identity(route.destination)
                 } for route in walk.steps]
        paths.append({
            'source': identity(walk.source),
            'steps': steps,
            'encoded_input_relevance': walk.relevance
        })
    return {
        'policy_id':
        SMOOTH_FLOOR_POLICY_ID,
        'selected_logit':
        float(relevance.score),
        'accounting':
        ledger,
        'fields':
        rows,
        'records':
        records,
        'walks':
        paths,
        'walk_coverage':
        walks.coverage,
        'walk_search_complete':
        walks.enumeration.complete,
        'capture_sha256':
        _capture_digest({
            'graph': vars(computation),
            'fitted_fields': fields.tables
        })
    }


def explain_classification(bundle, prediction_id, *, model, selected_class=None,
                           max_walks=10000, top_k=20):
    """Explain one binary prediction using a trusted, matching model bundle.

    Requires CUDA float32, one seed, three sum/gather layers, fitted numerical
    or categorical encoders, linear/ReLU and no normalization or compilation.
    Raises on unsupported profiles, provenance or mandatory replay/accounting
    failures. Returned allocations explain the selected raw logit, not a
    probability change or causal effect. Walk omissions are signed separately.
    """
    import torch
    from kumo_core.explain.encoded_fields import FieldCapture
    from kumo_core.explain.local_sensitivity import load_scoring_module
    from kumo_core.explain.perturbation import reconstruct_batch
    from kumo_core.explain.walk_binary import (
        emitted_probabilities,
        verify_binary_module,
    )
    from kumo_core.explain.walk_graph import (
        GraphCapture,
        capture_numerics,
        float32_matmul_precision,
    )

    from .walks import _apply_recorded_numerics

    if type(max_walks) is not int or not 1 <= max_walks <= 10000:
        raise ValueError('max_walks must be an integer in [1, 10000]')
    if type(top_k) is not int or not 0 <= top_k <= 100:
        raise ValueError('top_k must be an integer in [0, 100]')
    bundle.validate()
    provenance = bundle.manifest['native_provenance']
    if provenance['model_reconstruction']['task'][
            'task_type'] != 'binary_classification':
        raise ValueError('Only binary classification is qualified')
    record = bundle.numerical_evidence(prediction_id)
    original_logit = record.get('output', {}).get('head_output')
    if original_logit is None:
        raise ValueError('Retained raw head logit is required')
    if not torch.cuda.is_available():
        raise ValueError('Classification relevance requires CUDA')
    sample = bundle.scoring_graph(prediction_id)
    module = load_scoring_module(str(model), provenance).cuda().eval()
    selection = verify_binary_module(module, selected_class)
    batch = reconstruct_batch(sample).to('cuda')
    numerics = provenance.get('numerics')
    if not numerics or 'precision_mode' not in numerics:
        raise ValueError('Recorded capture precision is required')
    with float32_matmul_precision(numerics['precision_mode']), torch.no_grad():
        _apply_recorded_numerics(numerics)
        native = module.common_step(batch)
        with FieldCapture(module) as fields, GraphCapture(module) as graph:
            captured = module.common_step(batch)
        if (captured.shape != (1, 1) or captured.dtype != torch.float32
                or captured.device.type != 'cuda'
                or native.shape != captured.shape):
            raise ValueError('One float32 binary seed is required')
        _check_score(float(captured.item()), float(native.item()),
                     name='native capture')
        _check_score(float(captured.item()), original_logit, name='raw head')
        probability = float(emitted_probabilities(module, captured)[0, 1])
        _check_score(probability, record['prediction'],
                     name='served probability', atol=1e-6, rtol=1e-6)
        computation = graph.computation(preserve_device=True)
        if computation.T != 3 or any(
                relation.operator not in ('sum', 'gather')
                for layer in computation.relations.values()
                for relation in layer):
            raise ValueError(
                'Only three-layer sum/gather classification is qualified')
        result = _summarize(computation, fields, sample,
                            sign=selection.logit_sign, max_walks=max_walks,
                            top_k=top_k)
        result.update({
            'prediction_id': prediction_id,
            'origin': 'retained_graph_same_execution_capture',
            'model_bundle_sha256': provenance['bundle_sha256'],
            'sample_sha256': record['sample_sha256'],
            'selected_class': selection.selected_class_label,
            'logit_sign': selection.logit_sign,
            'original_raw_logit': original_logit,
            'captured_raw_logit': float(captured.item()),
            'original_probability': record['prediction'],
            'captured_probability': probability,
            'prediction_device': str(captured.device),
            'allocation_dtype': 'float64',
            'numerics': capture_numerics()
        })
    return result
