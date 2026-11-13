# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable derived walk explanations; inspection never loads a checkpoint."""
from collections import Counter
from dataclasses import dataclass
import math
import os
import sqlite3
from pathlib import Path
import tempfile

from .bundle import (_digest, _identifier, _json, _lines, _path, _read,
                     EvidenceValidationError)


@dataclass(frozen=True)
class WalkAttributionBundle:
    root: Path

    @classmethod
    def load(cls, root):
        bundle = cls(Path(root).absolute())
        bundle.validate()
        return bundle

    def validate(self):
        manifest = _read(_path(self.root, 'manifest.json'))
        if manifest.get('schema') != 'walk_attribution/v1':
            raise EvidenceValidationError('Unsupported walk attribution schema')
        if _path(self.root, '_SUCCESS').read_text() != _digest(
                _path(self.root, 'manifest.json')):
            raise EvidenceValidationError('Incomplete or corrupt walk manifest')
        if manifest['rows_sha256'] != _digest(_path(self.root, 'rows.jsonl')):
            raise EvidenceValidationError('Corrupt walk attribution rows')
        from kumo_core.explain.walk_attribution import WalkAttributionConfig
        WalkAttributionConfig(**manifest['config'])
        _identifier(manifest['parent_manifest_sha256'])
        _identifier(manifest['model_bundle_sha256'])
        if manifest.get('generation_closed') is not True:
            raise EvidenceValidationError('Incomplete walk generation')
        counts = Counter()
        with tempfile.TemporaryDirectory(prefix='walk-validation-') as scratch:
            with sqlite3.connect(str(Path(scratch) / 'ids.sqlite')) as db:
                db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
                for row in self.iter_rows():
                    _identifier(row['prediction_id'])
                    if row.get('sample_sha256') is not None:
                        _identifier(row['sample_sha256'])
                    elif row['status'] != 'failed':
                        raise EvidenceValidationError('Missing graph identity')
                    try:
                        db.execute('INSERT INTO ids VALUES (?)',
                                   (row['prediction_id'], ))
                    except sqlite3.IntegrityError as error:
                        raise EvidenceValidationError(
                            'Duplicate walk prediction identity') from error
                    if row['origin'] != 'retained_graph_recomputation':
                        raise EvidenceValidationError(
                            'Incorrect offline attribution origin')
                    if row['status'] not in ('available', 'unsupported',
                                             'failed', 'truncated'):
                        raise EvidenceValidationError(
                            'Invalid walk availability')
                    if row.get('config') != manifest['config']:
                        raise EvidenceValidationError(
                            'Row/manifest attribution configuration mismatch')
                    if row['status'] in ('available', 'truncated'):
                        validate_result(row)
                    counts[row['status']] += 1
        if sum(counts.values()) != manifest['rows'] or dict(
                counts) != manifest['status_counts']:
            raise EvidenceValidationError(
                'Incomplete or inconsistent walk attribution rows')
        return manifest

    def iter_rows(self):
        yield from _lines(_path(self.root, 'rows.jsonl'))


def validate_result(row):
    if (row.get('schema') != 'walk_attribution/v1'
            or row.get('coordinate') != 'encoded_model_output'
            or row.get('input_coordinate') != 'encoded_node_features'
            or type(row.get('truncated')) is not bool
            or row.get('complete') is not (not row['truncated'])
            or (row['status'] == 'truncated') != row['truncated']):
        raise EvidenceValidationError(
            'Invalid attribution coordinate or completeness')
    from kumo_core.explain.walk_attribution import WalkAttributionConfig
    if row.get('method') != 'hetero_lrp_gamma_reference_v1':
        raise EvidenceValidationError('Unknown walk attribution method')
    cfg = WalkAttributionConfig(**row['config'])
    if any(
            row.get(key) != getattr(cfg, key)
            for key in ('gamma', 'epsilon', 'output_index')):
        raise EvidenceValidationError(
            'Walk method parameters disagree with configuration')
    if (type(row.get('expanded_routes')) is not int
            or not 0 <= row['expanded_routes'] <= cfg.max_walks
            or len(row['walks']) > cfg.top_k
            or (row['complete'] and row.get('omitted_signed_relevance') != 0)):
        raise EvidenceValidationError('Invalid walk search budget or coverage')
    names = ('score', 'bias_relevance', 'stabilizer_residual',
             'omitted_signed_relevance', 'accounting_error')

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not all(finite(row.get(name)) for name in names):
        raise EvidenceValidationError('Nonfinite or missing attribution ledger')
    for walk in row['walks']:
        if not finite(walk['relevance']) or type(
                walk['source_occurrence']
        ) is not int or walk['source_occurrence'] < 0:
            raise EvidenceValidationError(
                'Invalid walk relevance or occurrence')
        previous_layer = -1
        for step in walk['steps']:
            if type(step.get(
                    'layer')) is not int or step['layer'] <= previous_layer:
                raise EvidenceValidationError('Invalid layer sequence')
            previous_layer = step['layer']
            indices = ('occurrence', ) if step.get('kind') == 'self' else (
                'source_occurrence', 'destination_occurrence',
                'edge_occurrence')
            if any(
                    type(step.get(key)) is not int or step[key] < 0
                    for key in indices):
                raise EvidenceValidationError('Invalid route occurrence')
            if step['kind'] not in ('self', 'relation'):
                raise EvidenceValidationError('Unknown walk route kind')
            if step['kind'] == 'relation' and (
                    len(step['type']) != 3
                    or not all(isinstance(v, str) and v for v in step['type'])):
                raise EvidenceValidationError('Invalid relation type')
    error = row['score'] - sum(w['relevance']
                               for w in row['walks']) - sum(row[k]
                                                            for k in names[1:4])
    if not math.isclose(error, row['accounting_error'], abs_tol=1e-5,
                        rel_tol=1e-5):
        raise EvidenceValidationError('Attribution ledger mismatch')


def attribute_bundle(bundle, *, model, config, output):
    """Recompute on frozen original graphs; model is a trusted loaded Trainer or staged model directory.

    The initial reconstruction adapter deliberately retains scalar MSE/CPU
    limits. Operator qualification is independent and can be narrower.
    """
    import torch
    from kumo_core.explain.local_sensitivity import bind_walk_result, load_scoring_module
    from kumo_core.explain.perturbation import reconstruct_batch
    from kumo_core.explain.walk_attribution import (UnsupportedWalkAttribution,
                                                    WalkAttributionConfig,
                                                    WalkCapture)

    if not isinstance(config, WalkAttributionConfig):
        raise TypeError('config must be WalkAttributionConfig')
    bundle.validate()
    provenance = bundle.manifest['native_provenance']
    if hasattr(model, '_fitted'):
        from kumo_core.artifacts import stage_model_bundle
        training = model._fitted('attribute_walks')
        with tempfile.TemporaryDirectory(prefix='walk-model-') as temp:
            model_dir = stage_model_bundle(training.artifact_paths,
                                           Path(temp) / 'model')
            module = load_scoring_module(str(model_dir), provenance)
    else:
        module = load_scoring_module(str(model), provenance)
    expected_rows = sum(1 for _ in bundle.iter_predictions())
    destination = Path(output).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + '.lock')
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        staging = Path(
            tempfile.mkdtemp(prefix=destination.name + '.partial-',
                             dir=destination.parent))
        counts = Counter()
        with (staging / 'rows.jsonl').open('wb') as stream:
            for record in bundle.iter_numerical_evidence():
                row = {
                    'prediction_id': record['prediction_id'],
                    'origin': 'retained_graph_recomputation',
                    'method': 'hetero_lrp_gamma_reference_v1',
                    'config': vars(config),
                    'sample_sha256': record.get('sample_sha256')
                }
                try:
                    sample = bundle.scoring_graph(record['prediction_id'])
                    batch = reconstruct_batch(sample)
                    with torch.no_grad():
                        logits = module.common_step(batch)
                        decoded = module.target_encoder.decode_tensor(
                            module.loss.to_prediction(logits))
                        if decoded.numel() != 1 or not math.isclose(
                                float(decoded.reshape(
                                    ())), record['prediction'], rel_tol=1e-6,
                                abs_tol=1e-6):
                            raise ValueError(
                                'Retained graph does not reproduce emitted prediction'
                            )
                    with torch.no_grad(), WalkCapture(module,
                                                      config) as capture:
                        captured_logits = module.common_step(batch)
                        if captured_logits.shape != logits.shape or not torch.allclose(
                                captured_logits, logits, atol=1e-6, rtol=1e-6):
                            raise ValueError(
                                'Attribution forward changed the reconstructed prediction'
                            )
                        result = bind_walk_result(capture.result(seed_index=0),
                                                  sample)
                    row.update(result)
                    row.update(origin='retained_graph_recomputation',
                               prediction_id=record['prediction_id'])
                except UnsupportedWalkAttribution as error:
                    row.update(status='unsupported', reason=str(error))
                except Exception as error:
                    row.update(status='failed', reason=type(error).__name__)
                counts[row['status']] += 1
                stream.write(_json(row) + b'\n')
        if sum(counts.values()) != expected_rows:
            raise EvidenceValidationError(
                'Prediction/attribution coverage mismatch')
        manifest = {
            'schema': 'walk_attribution/v1',
            'parent_manifest_sha256': _digest(bundle.root / 'manifest.json'),
            'model_bundle_sha256': provenance['bundle_sha256'],
            'config': vars(config),
            'rows': sum(counts.values()),
            'status_counts': dict(counts),
            'generation_closed': True,
            'rows_sha256': _digest(staging / 'rows.jsonl')
        }
        (staging / 'manifest.json').write_bytes(_json(manifest))
        (staging / '_SUCCESS').write_text(_digest(staging / 'manifest.json'))
        WalkAttributionBundle(staging).validate()
        os.rename(staging, destination)
        return WalkAttributionBundle(destination)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
