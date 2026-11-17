# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable derived walk explanations; inspection never loads a checkpoint.

A derived bundle holds one schema, declared by its manifest:
`walk_attribution/v1` for the legacy rule on regression defaults (format
frozen) or `walk_attribution/v2` otherwise. Offline replay applies the
float32 matmul precision recorded by the native capture (`numerics`), for
both schemas; bundles produced before that record replay at the process
precision, as before.
"""
import math
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import (
    EvidenceValidationError,
    _digest,
    _identifier,
    _json,
    _lines,
    _path,
    _read,
)

SCHEMA_V1 = 'walk_attribution/v1'
SCHEMA_V2 = 'walk_attribution/v2'


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
        schema = manifest.get('schema')
        if schema not in (SCHEMA_V1, SCHEMA_V2):
            raise EvidenceValidationError('Unsupported walk attribution schema')
        if _path(self.root, '_SUCCESS').read_text() != _digest(
                _path(self.root, 'manifest.json')):
            raise EvidenceValidationError('Incomplete or corrupt walk manifest')
        if manifest['rows_sha256'] != _digest(_path(self.root, 'rows.jsonl')):
            raise EvidenceValidationError('Corrupt walk attribution rows')
        from kumo_core.explain.walk_attribution import WalkAttributionConfig
        config = WalkAttributionConfig(**manifest['config'])
        if schema == SCHEMA_V2 and config.to_dict() != manifest['config']:
            raise EvidenceValidationError('Non-canonical walk configuration')
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
                    if row.get('config') != manifest['config']:
                        raise EvidenceValidationError(
                            'Row/manifest attribution configuration mismatch')
                    if schema == SCHEMA_V2:
                        validate_v2(row)
                    else:
                        if row['status'] not in ('available', 'unsupported',
                                                 'failed', 'truncated'):
                            raise EvidenceValidationError(
                                'Invalid walk availability')
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


def validate_v2(row, sample=None):
    """Validate a v2 row; a v1 row in a v2 context is a schema mix."""
    from kumo_core.explain.walk_schema import WalkSchemaError, validate_v2_row
    if row.get('schema') != SCHEMA_V2:
        raise EvidenceValidationError('Mixed walk attribution schemas')
    try:
        validate_v2_row(row, sample)
    except WalkSchemaError as error:
        raise EvidenceValidationError(f'Invalid v2 walk row: {error}') from error


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


class ReplayPrecisionUnavailable(ValueError):
    """The recorded capture precision cannot be reproduced in this process."""


def _apply_recorded_numerics(numerics):
    """Check, inside the recorded mode, that the derived oneDNN setting
    matches the record (`none` counts as `ieee`, contract C9)."""
    from kumo_core.explain.walk_graph import (
        mkldnn_matmul_fp32_precision,
        normalized_mkldnn,
    )
    recorded = (numerics or {}).get('mkldnn_matmul_fp32_precision')
    if recorded is None:
        return
    actual = mkldnn_matmul_fp32_precision()
    if actual is None or normalized_mkldnn(actual) != normalized_mkldnn(recorded):
        raise ReplayPrecisionUnavailable('replay_precision_unavailable')


def _check_replayed_prediction(module, logits, record, binary):
    """The retained graph must reproduce the emitted prediction (1e-6)."""
    from kumo_core.explain.walk_binary import emitted_probabilities
    if binary:
        probability = float(emitted_probabilities(module, logits).reshape(-1, 2)[0, 1])
        if logits.numel() != 1 or not math.isclose(
                probability, record['prediction'], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Retained graph does not reproduce emitted prediction')
        output = record.get('output') or {}
        if 'head_output' in output and not math.isclose(
                float(logits.reshape(())), output['head_output'], rel_tol=1e-6,
                abs_tol=1e-6):
            raise ValueError('Retained graph does not reproduce the head logit')
        return
    decoded = module.target_encoder.decode_tensor(module.loss.to_prediction(logits))
    if decoded.numel() != 1 or not math.isclose(
            float(decoded.reshape(())), record['prediction'], rel_tol=1e-6,
            abs_tol=1e-6):
        raise ValueError('Retained graph does not reproduce emitted prediction')


def attribute_bundle(bundle, *, model, config, output):
    """Recompute on frozen original graphs; model is a trusted loaded Trainer or staged model directory.

    Supports MSE regression and BCE binary classification on CPU. Operator
    qualification is independent and can be narrower. Each row is replayed
    under the native capture's recorded precision, restored afterwards.
    """
    import torch
    from kumo_core.explain.local_sensitivity import (
        bind_walk_result,
        load_scoring_module,
    )
    from kumo_core.explain.perturbation import reconstruct_batch
    from kumo_core.explain.walk_attribution import (
        UnsupportedWalkAttribution,
        WalkAttributionConfig,
        WalkCapture,
        select_schema,
    )
    from kumo_core.explain.walk_graph import float32_matmul_precision

    if not isinstance(config, WalkAttributionConfig):
        raise TypeError('config must be WalkAttributionConfig')
    bundle.validate()
    provenance = bundle.manifest['native_provenance']
    task_type = provenance.get('model_reconstruction', {}).get(
        'task', {}).get('task_type', 'regression')
    schema = select_schema(config, task_type)
    # None for bundles produced before the record existed: process precision.
    numerics = provenance.get('numerics')
    mode = None if numerics is None else numerics.get('precision_mode')
    # The scoring module is duck-typed: any loaded node scorer qualifies.
    module: Any
    if hasattr(model, '_fitted'):
        from kumo_core.artifacts import stage_model_bundle
        training = model._fitted('attribute_walks')
        with tempfile.TemporaryDirectory(prefix='walk-model-') as temp:
            model_dir = stage_model_bundle(training.artifact_paths,
                                           Path(temp) / 'model')
            module = load_scoring_module(str(model_dir), provenance)
    else:
        module = load_scoring_module(str(model), provenance)
    replay = (_V2Replay(module, config, numerics, mode, task_type)
              if schema == SCHEMA_V2 else None)
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
                if replay is not None:
                    row = replay.row(bundle, record)
                    counts[row['status']] += 1
                    stream.write(_json(row) + b'\n')
                    continue
                row = {
                    'prediction_id': record['prediction_id'],
                    'origin': 'retained_graph_recomputation',
                    'method': 'hetero_lrp_gamma_reference_v1',
                    'config': config.to_dict(),
                    'sample_sha256': record.get('sample_sha256')
                }
                try:
                    sample = bundle.scoring_graph(record['prediction_id'])
                    batch = reconstruct_batch(sample)
                    with float32_matmul_precision(mode):
                        _apply_recorded_numerics(numerics)
                        with torch.no_grad():
                            logits = module.common_step(batch)
                            _check_replayed_prediction(module, logits, record,
                                                       binary=False)
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
                except ReplayPrecisionUnavailable as error:
                    row.update(status='failed', reason=str(error))
                except Exception as error:
                    row.update(status='failed', reason=type(error).__name__)
                counts[row['status']] += 1
                stream.write(_json(row) + b'\n')
        if sum(counts.values()) != expected_rows:
            raise EvidenceValidationError(
                'Prediction/attribution coverage mismatch')
        manifest = {
            'schema': schema,
            'parent_manifest_sha256': _digest(bundle.root / 'manifest.json'),
            'model_bundle_sha256': provenance['bundle_sha256'],
            'config': config.to_dict(),
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


class _V2Replay:
    """v2 rows recomputed on retained graphs with the original pipeline."""

    def __init__(self, module: Any, config, numerics, mode, task_type):
        from kumo_core.explain.walk_attribution import (
            UnsupportedWalkAttribution,
            method_module,
        )
        from kumo_core.explain.walk_binary import resolve_selection
        self.module, self.config = module, config
        self.numerics, self.mode = numerics, mode
        self.binary = task_type == 'binary_classification'
        self.selection = self.failure = None
        try:
            self.selection = resolve_selection(module, config)
            method_module(config.method).qualify(module, config)
        except UnsupportedWalkAttribution as error:
            self.failure = str(error)

    def row(self, bundle, record):
        import torch
        from kumo_core.explain.perturbation import reconstruct_batch
        from kumo_core.explain.walk_binary import (
            emitted_probabilities,
            output_block,
        )
        from kumo_core.explain.walk_graph import (
            GraphCapture,
            capture_numerics,
            float32_matmul_precision,
        )
        from kumo_core.explain.walk_schema import (
            explain_capture,
            minimal_row,
            row_numerics,
        )
        identity: dict[str, Any] = dict(
            config=self.config, origin='retained_graph_recomputation',
            prediction_id=record['prediction_id'],
            sample_sha256=record.get('sample_sha256'))
        if self.failure is not None:
            return minimal_row('unsupported', self.failure, **identity)
        try:
            sample = bundle.scoring_graph(record['prediction_id'])
            batch = reconstruct_batch(sample)
            with float32_matmul_precision(self.mode):
                _apply_recorded_numerics(self.numerics)
                replay = capture_numerics()
                with torch.no_grad():
                    logits = self.module.common_step(batch)
                    _check_replayed_prediction(self.module, logits, record,
                                               binary=self.binary)
                with torch.no_grad(), GraphCapture(self.module, self.config) as capture:
                    captured = self.module.common_step(batch)
                if captured.shape != logits.shape or not torch.allclose(
                        captured, logits, atol=1e-6, rtol=1e-6):
                    raise ValueError(
                        'Attribution forward changed the reconstructed prediction')
                output = output_block(self.selection, self.module, captured[0])
                extra = {'replay_logit_abs_diff': None,
                         'replay_probability_abs_diff': None}
                recorded_output = record.get('output') or {}
                if 'head_output' in recorded_output:
                    extra['replay_logit_abs_diff'] = abs(
                        output['head_output'] - recorded_output['head_output'])
                if self.binary:
                    probability = float(emitted_probabilities(self.module, captured)[0, 1])
                    extra['replay_probability_abs_diff'] = abs(
                        probability - record['prediction'])
                numerics = row_numerics(
                    {**replay, 'precision_mode': self.mode},
                    str(captured.dtype).removeprefix('torch.'),
                    replay_precision_mode=replay['precision_mode'])
                return explain_capture(capture, selection=self.selection,
                                       output=output, sample=sample,
                                       numerics=numerics, extra=extra, **identity)
        except ReplayPrecisionUnavailable as error:
            return minimal_row('failed', str(error), **identity)
        except Exception as error:
            return minimal_row('failed', f'{type(error).__name__}: {error}', **identity)
