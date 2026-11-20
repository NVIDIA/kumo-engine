# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary classification companion fitting and runtime-verified message replacement."""
import copy
import math
import os
import tempfile
import time
from pathlib import Path

import torch
from kumo_core.explain.classification_masking import (
    POLICY,
    FeasibilityCompanion,
    capture_context,
    fidelity,
    fit_feasibility,
    verified_greedy,
)
from kumo_core.explain.local_sensitivity import _digest, load_scoring_module
from kumo_core.explain.message_intervention import DifferentiableMessageMasking
from kumo_core.explain.perturbation import reconstruct_batch
from kumo_core.explain.walk_binary import verify_binary_module
from kumo_core.explain.walk_graph import (
    _relation_key,
    float32_matmul_precision,
    structural_qualification,
)

from .walks import _apply_recorded_numerics


def _load_binary(bundle, model):
    bundle.validate()
    provenance = bundle.manifest['native_provenance']
    if provenance['model_reconstruction']['task'][
            'task_type'] != 'binary_classification':
        raise ValueError('Message masking supports binary classification only')
    if not torch.cuda.is_available():
        raise ValueError('Classification masking requires CUDA')
    module = load_scoring_module(str(model),
                                 provenance).cuda().eval().requires_grad_(False)
    verify_binary_module(module, None)
    structural_qualification(module)
    aggr = module.model.aggr
    if (len(module.model.hetero_convs) != 3 or aggr not in ('sum', ['sum'])
            or any(p.dtype != torch.float32 or p.device.type != 'cuda'
                   for p in module.parameters())):
        raise ValueError('Requires float32 CUDA and three sum/gather layers')
    numerics = provenance.get('numerics')
    if not numerics or 'precision_mode' not in numerics:
        raise ValueError('Recorded prediction precision is required')
    return module, provenance


def _score(module, raw):
    if raw.shape != (1, 1) or not torch.isfinite(raw).all():
        raise ValueError('Require one finite binary logit')
    calibrated = module.calibrator(raw) if module.calibrator.is_fitted else raw
    probability = module.loss.to_probability(calibrated)[0, 1]
    result = {'raw': float(raw.item()), 'served': float(probability.item())}
    if not math.isfinite(result['served']) or not 0 <= result['served'] <= 1:
        raise ValueError('Invalid calibrated served probability')
    return result


def _replay(actual, expected):
    for key, atol, rtol in (('raw', 1e-5, 1e-5), ('served', 1e-6, 1e-6)):
        if (not math.isfinite(expected[key])
                or not math.isfinite(actual[key]) or not math.isclose(
                    actual[key], expected[key], abs_tol=atol, rel_tol=rtol)):
            raise ValueError(f'{key} replay integrity failed')


def _capture(bundle, prediction_id, module):
    record = bundle.numerical_evidence(prediction_id)
    if 'head_output' not in (record.get('output') or {}):
        raise ValueError('Retained raw head logit is required')
    sample = bundle.scoring_graph(prediction_id)
    batch = reconstruct_batch(sample).to('cuda')
    raw, contexts, occurrences = capture_context(module, batch)
    original = _score(module, raw)
    retained = {
        'raw': record['output']['head_output'],
        'served': record['prediction']
    }
    _replay(original, retained)
    return batch, contexts, occurrences, original, retained, sample, record


def _evaluate(module, batch, probabilities, baselines, replacements):
    gates = {
        key: torch.ones_like(value)
        for key, value in probabilities.items()
    }
    for key, edge in replacements:
        gates[key][edge] = 0
    with torch.enable_grad(), DifferentiableMessageMasking(
            module, gates=gates, baselines=baselines) as intervention:
        raw = module.common_step(batch)
    return {
        **_score(module, raw), 'replaced': len(replacements),
        'events': intervention.observations
    }


class ClassificationMaskingCompanion:
    """Fitted gate networks and fixed stand-ins bound to one binary predictor."""
    def __init__(self, network, *, model_bundle_sha256, decision_threshold,
                 optimizer_state=None, training=None):
        if (not isinstance(model_bundle_sha256, str)
                or len(model_bundle_sha256) != 64
                or any(c not in '0123456789abcdef'
                       for c in model_bundle_sha256)):
            raise ValueError('Model bundle SHA256 is required')
        fidelity({
            'raw': 0.,
            'served': .5
        }, {
            'raw': 0.,
            'served': .5
        }, decision_threshold)
        if not isinstance(network, FeasibilityCompanion):
            raise ValueError('Expected a feasibility companion')
        self._network = copy.deepcopy(network).eval()
        self._model_sha256 = model_bundle_sha256
        self._threshold = decision_threshold
        self._optimizer_state = copy.deepcopy(optimizer_state)
        self._training = copy.deepcopy(training or {})
        self._standins_sha256 = _digest(network.standins.state_dict())

    @classmethod
    def fit(cls, bundle, *, model, standins, decision_threshold, epochs=20,
            prediction_ids=None):
        """Fit feasibility gates on retained cases with supplied learned stand-ins.

        Stand-ins stay fixed. Labels are single-replacement fidelity failures,
        not task labels. Adam uses case-balanced BCE, lr .01 and seed 36.
        No validation-set checkpoint selection or stand-in fitting is implied.
        """
        if type(epochs) is not int or not 1 <= epochs <= 120:
            raise ValueError('epochs must be in [1, 120]')
        fidelity({
            'raw': 0.,
            'served': .5
        }, {
            'raw': 0.,
            'served': .5
        }, decision_threshold)
        module, provenance = _load_binary(bundle, model)
        keys = [(layer, _relation_key(relation))
                for layer, convs in enumerate(module.model.hetero_convs, 1)
                for relation in convs]
        with torch.random.fork_rng():
            torch.manual_seed(36)
            network = FeasibilityCompanion(keys, module.model.channels,
                                           standins).cuda()
        predictor_before = _digest(module.state_dict())
        standins_before = _digest(network.standins.state_dict())
        ids = (list(prediction_ids) if prediction_ids is not None else
               [r['prediction_id'] for r in bundle.iter_numerical_evidence()])
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(
                'Fitting requires distinct retained prediction IDs')
        cases, bindings = [], []
        with float32_matmul_precision(provenance['numerics']['precision_mode']):
            _apply_recorded_numerics(provenance['numerics'])
            for prediction_id in ids:
                batch, contexts, occurrences, original, retained, _, record = _capture(
                    bundle, prediction_id, module)
                with torch.no_grad():
                    probabilities, baselines = network(contexts)
                keep_all = _evaluate(module, batch, probabilities, baselines,
                                     ())
                _replay(keep_all, original)
                _replay(keep_all, retained)
                targets = {
                    key: torch.empty_like(value)
                    for key, value in probabilities.items()
                }
                for key, value in probabilities.items():
                    for edge in range(value.numel()):
                        result = _evaluate(module, batch, probabilities,
                                           baselines, ((key, edge), ))
                        targets[key][edge] = float(
                            not fidelity(original, result,
                                         decision_threshold)['fidelity_pass'])
                cases.append((contexts, targets))
                bindings.append({
                    'prediction_id':
                    prediction_id,
                    'sample_sha256':
                    record['sample_sha256'],
                    'context_sha256':
                    _digest({
                        'contexts': contexts,
                        'occurrences': occurrences
                    }),
                    'feasibility_targets': {
                        str(k): v.tolist()
                        for k, v in targets.items()
                    }
                })
            optimizer, curve = fit_feasibility(network, cases, epochs=epochs)
        if (_digest(module.state_dict()) != predictor_before
                or _digest(network.standins.state_dict()) != standins_before):
            raise ValueError('Fitting mutated predictor or fixed stand-ins')
        return cls(
            network, model_bundle_sha256=provenance['bundle_sha256'],
            decision_threshold=decision_threshold, optimizer_state=optimizer,
            training={
                'policy': 'feasibility_distillation_receiver_target_v1',
                'epochs': epochs,
                'seed': 36,
                'learning_rate': .01,
                'curve': curve,
                'cases': bindings
            })

    def save(self, path):
        """Atomically save owned tensor state and bindings without overwriting."""
        payload = {
            'version': 1,
            'policy': POLICY,
            'keys': self._network.keys,
            'channels': self._network.channels,
            'state_dict': {
                k: v.detach().cpu().clone()
                for k, v in self._network.state_dict().items()
            },
            'model_bundle_sha256': self._model_sha256,
            'decision_threshold': self._threshold,
            'standins_sha256': self._standins_sha256,
            'optimizer_state_dict': self._optimizer_state,
            'training': self._training
        }
        payload['content_sha256'] = _digest(payload)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                torch.save(payload, stream)
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    @classmethod
    def load(cls, path):
        """Load a tensor-only companion checkpoint and verify all stored bindings."""
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(payload, dict) or payload.get('version') != 1
                or payload.get('policy') != POLICY
                or payload.get('content_sha256') != _digest({
                    k: v
                    for k, v in payload.items() if k != 'content_sha256'
                })):
            raise ValueError('Companion checkpoint integrity failed')
        state = payload['state_dict']
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite companion checkpoint')
        standins = {
            int(k.split('.')[1]): v
            for k, v in state.items() if k.startswith('standins.')
        }
        with torch.random.fork_rng(devices=[]):
            network = FeasibilityCompanion(payload['keys'], payload['channels'],
                                           standins)
        network.load_state_dict(state, strict=True)
        result = cls(network,
                     model_bundle_sha256=payload['model_bundle_sha256'],
                     decision_threshold=payload['decision_threshold'],
                     optimizer_state=payload['optimizer_state_dict'],
                     training=payload['training'])
        if result._standins_sha256 != payload['standins_sha256']:
            raise ValueError('Stand-in checkpoint identity mismatch')
        return result

    def mask(self, bundle, prediction_id, *, model):
        """Return verified cumulative replacements or keep-all for one CUDA prediction."""
        module, provenance = _load_binary(bundle, model)
        if provenance['bundle_sha256'] != self._model_sha256:
            raise ValueError('Companion belongs to a different predictor')
        network = copy.deepcopy(
            self._network).cuda().eval().requires_grad_(False)
        predictor_before = _digest(module.state_dict())
        companion_before = _digest(network.state_dict())
        if _digest(network.standins.state_dict()) != self._standins_sha256:
            raise ValueError('Fixed stand-ins changed')
        with float32_matmul_precision(provenance['numerics']['precision_mode']):
            _apply_recorded_numerics(provenance['numerics'])
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch, contexts, occurrences, original, retained, sample, record = _capture(
                bundle, prediction_id, module)
            torch.cuda.synchronize()
            captured_at = time.perf_counter()
            with torch.no_grad():
                probabilities, baselines = network(contexts)
            torch.cuda.synchronize()
            ranked_at = time.perf_counter()
            ranked = [(float(value[edge]), key, edge)
                      for key, value in probabilities.items()
                      for edge in range(value.numel())]
            keep_all = _evaluate(module, batch, probabilities, baselines, ())
            _replay(keep_all, original)
            _replay(keep_all, retained)
            result = verified_greedy(
                ranked, original, keep_all, lambda replacements: _evaluate(
                    module, batch, probabilities, baselines, replacements),
                self._threshold)
            torch.cuda.synchronize()
            finished = time.perf_counter()
        if (_digest(module.state_dict()) != predictor_before
                or _digest(network.state_dict()) != companion_before):
            raise ValueError('Masking mutated predictor or companion')
        events = []
        for probability, key, edge in sorted(ranked):
            source, destination = occurrences[key]['edges'][:, edge].tolist()
            relation = key[1]
            events.append({
                'layer':
                key[0],
                'relation':
                list(relation),
                'edge_column':
                edge,
                'direction':
                occurrences[key]['direction'],
                'source_occurrence':
                source,
                'destination_occurrence':
                destination,
                'source_key':
                sample['nodes'][relation[0]]['pkeys'][source],
                'destination_key':
                sample['nodes'][relation[2]]['pkeys'][destination],
                'keep_probability':
                probability
            })
        return {
            **result, 'policy':
            POLICY,
            'prediction_id':
            prediction_id,
            'original':
            original,
            'retained_prediction':
            retained,
            'keep_all':
            keep_all,
            'decision_threshold':
            self._threshold,
            'raw_tolerance':
            .05,
            'served_tolerance':
            .01,
            'model_bundle_sha256':
            self._model_sha256,
            'predictor_state_sha256':
            predictor_before,
            'companion_state_sha256':
            companion_before,
            'standins_sha256':
            self._standins_sha256,
            'sample_sha256':
            record['sample_sha256'],
            'context_sha256':
            _digest({
                'contexts': contexts,
                'occurrences': occurrences
            }),
            'ranking':
            events,
            'total_messages':
            len(ranked),
            'replaced_messages':
            len(result['selected_mask']),
            'kept_messages':
            len(ranked) - len(result['selected_mask']),
            'fidelity':
            fidelity(original, result['selected'], self._threshold),
            'timing': {
                'context_seconds':
                captured_at - started,
                'ranking_seconds':
                ranked_at - captured_at,
                'verification_seconds':
                finished - ranked_at,
                'total_seconds':
                finished - started,
                'forward_calls':
                2 + len(result['attempts']),
                'excludes': [
                    'model_loading', 'sampling', 'integrity_hashing',
                    'serialization'
                ]
            }
        }
