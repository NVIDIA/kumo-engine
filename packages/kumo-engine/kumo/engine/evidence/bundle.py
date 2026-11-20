# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, relocatable prediction evidence with bounded row readers.

Checksums detect corruption, not a malicious author who rewrites all hashes.
Source files are retained as data; inspection never executes model code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, Mapping

SCHEMA_VERSION = 1
_ID = re.compile(r'[a-f0-9]{64}\Z')
_STATES = {
    'available', 'zero_signal', 'not_sampled', 'unsupported', 'disabled',
    'failed', 'unknown'
}


class EvidenceValidationError(ValueError):
    """Retained evidence is malformed, incomplete, or inconsistent."""


@dataclass(frozen=True)
class EvidenceConfig:
    scoring_graph: bool = True
    numerical_sensitivity: Literal[
        'signed_and_absolute'] | None = 'signed_and_absolute'
    replay: bool = False
    walk_attribution: Any = None
    on_unsupported: Literal['raise', 'record'] = 'raise'

    def __post_init__(self):
        if self.walk_attribution is not None:
            from kumo_core.explain.walk_attribution import (
                WalkAttributionConfig,
            )
            if not isinstance(self.walk_attribution, WalkAttributionConfig):
                raise TypeError(
                    'walk_attribution must be WalkAttributionConfig')
            if not self.scoring_graph:
                raise ValueError(
                    'Walk attribution requires retained scoring graphs')
        if type(self.scoring_graph) is not bool or type(
                self.replay) is not bool:
            raise TypeError('Evidence flags must be boolean')
        if self.numerical_sensitivity not in (None, 'signed_and_absolute'):
            raise ValueError(
                'Unsupported numerical sensitivity coordinate/method')
        if self.on_unsupported not in ('raise', 'record'):
            raise ValueError('on_unsupported must be raise or record')


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise EvidenceValidationError('Duplicate JSON key')
        result[key] = value
    return result


def _loads(text):
    try:
        return json.loads(
            text, object_pairs_hook=_pairs, parse_constant=lambda _:
            (_ for _ in
             ()).throw(EvidenceValidationError('Nonfinite JSON number')))
    except (ValueError, TypeError) as error:
        raise EvidenceValidationError('Malformed evidence JSON') from error


def _read(path):
    return _loads(path.read_bytes())


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PurePosixPath(relative).is_absolute()
            or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise EvidenceValidationError('Unsafe evidence path')
    path = root
    for part in relative.split('/'):
        path = path / part
        if path.is_symlink():
            raise EvidenceValidationError('Evidence symlinks are forbidden')
    if not path.resolve().is_relative_to(root.resolve()):
        raise EvidenceValidationError('Evidence path escapes bundle')
    return path


def _lines(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield _loads(line)


def _files(root):
    for directory, directories, files in os.walk(root):
        for name in directories + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise EvidenceValidationError('Evidence symlinks are forbidden')
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_file():
                raise EvidenceValidationError(
                    'Evidence must contain regular files')
            yield path


def _copy_tree(source, destination):
    if source.is_symlink():
        raise EvidenceValidationError('Evidence symlinks are forbidden')
    for path in _files(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise EvidenceValidationError('Prediction ID must be lowercase SHA256')
    return value


def _validate_graph(graph):
    if graph.get('origin') != 'original_forward':
        raise EvidenceValidationError(
            'Scoring graph is not original-forward evidence')
    nodes = graph['nodes']
    for node in nodes.values():
        if len(node.get('pkeys', [])) != node['num_nodes']:
            raise EvidenceValidationError('Graph primary-key coverage mismatch')
    for edge in graph['edges']:
        src, _, dst = edge['type']
        indices = edge['edge_index']
        if src not in nodes or dst not in nodes or len(indices) != 2:
            raise EvidenceValidationError('Invalid graph relation')
        if len(indices[0]) != len(indices[1]):
            raise EvidenceValidationError('Invalid edge endpoint arrays')
        for table, coordinates in ((src, indices[0]), (dst, indices[1])):
            if any(
                    type(i) is not int or not 0 <= i < nodes[table]['num_nodes']
                    for i in coordinates):
                raise EvidenceValidationError(
                    'Graph endpoint outside node block')


@dataclass(frozen=True)
class PredictionEvidenceSource:
    """Lazy reference to a completed prediction, before portable retention."""
    files: tuple[str, ...]
    run_root: Path | None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def export(self, destination: str | Path, *,
               source_records=None) -> PredictionEvidenceBundle:
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock = destination.with_name(destination.name + '.lock')
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        staging = None
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            staging = Path(
                tempfile.mkdtemp(prefix=destination.name + '.partial-',
                                 dir=destination.parent))
            prediction_root = staging / 'predictions'
            prediction_root.mkdir()
            for index, source in enumerate(self.files):
                path = Path(source)
                if path.is_symlink() or not path.is_file():
                    raise EvidenceValidationError(
                        'Prediction source is not a regular file')
                shutil.copyfile(path, prediction_root / f'{index:08d}.parquet')
            if self.run_root is not None:
                xai = self.run_root / 'xai'
                if xai.exists():
                    _copy_tree(xai, staging / 'xai')
            if source_records is not None:
                with (staging / 'source-records.jsonl').open('wb') as stream:
                    for source_record in source_records:
                        stream.write(_json(dict(source_record)) + b'\n')
            counts = Counter()
            for record in _lines(staging / 'xai/local/index.jsonl'):
                counts[record['status']] += 1
            local = staging / 'xai/local/manifest.json'
            native = _read(local) if local.exists() else {}
            with (staging / 'inventory.jsonl').open('wb') as stream:
                for path in _files(staging):
                    if path.name == 'inventory.jsonl':
                        continue
                    stream.write(
                        _json({
                            'path': path.relative_to(staging).as_posix(),
                            'bytes': path.stat().st_size,
                            'sha256': _digest(path)
                        }) + b'\n')
            manifest = {
                'schema_version':
                SCHEMA_VERSION,
                'kind':
                'prediction_evidence',
                'provenance':
                dict(self.provenance),
                'native_provenance':
                native,
                'generation_closed':
                bool(
                    native.get('generation_closed',
                               native.get('complete', False))),
                'status_counts':
                dict(counts),
                'evidence_complete':
                bool(counts)
                and not any(counts[s]
                            for s in ('failed', 'unknown', 'unsupported'))
                and bool(native.get('complete')),
                'inventory_sha256':
                _digest(staging / 'inventory.jsonl'),
                'raw_value_coordinate':
                'numerical_tensor_mapper_output',
                'source_lineage': ('provided_mapper_verified' if source_records
                                   is not None else 'not_provided'),
            }
            (staging / 'manifest.json').write_bytes(_json(manifest))
            (staging / '_SUCCESS').write_text(_digest(staging /
                                                      'manifest.json'))
            bundle = PredictionEvidenceBundle(staging)
            bundle.validate()
            os.rename(staging, destination)
            return PredictionEvidenceBundle(destination)
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
            # Failed staging remains inspectable, never published as destination.


@dataclass(frozen=True)
class PredictionEvidenceBundle:
    root: Path

    @classmethod
    def from_result(cls, result, *, run_root=None, provenance=None):
        if run_root is None:
            run_root = getattr(getattr(result, 'explanations', None),
                               'run_root', None)
        return PredictionEvidenceSource(tuple(result.files),
                                        Path(run_root) if run_root else None,
                                        dict(provenance or {}))

    @classmethod
    def load(cls, path):
        bundle = cls(Path(path).absolute())
        bundle.validate()
        return bundle

    @property
    def manifest(self):
        return _read(_path(self.root, 'manifest.json'))

    def iter_predictions(self, batch_size=1024) -> Iterator[dict]:
        import pyarrow.parquet as pq
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be positive')
        for entry in _lines(_path(self.root, 'inventory.jsonl')):
            if entry['path'].startswith('predictions/'):
                path = _path(self.root, entry['path'])
                for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=batch_size):
                    yield from batch.to_pylist()

    def iter_numerical_evidence(self):
        for row in _lines(_path(self.root, 'xai/local/index.jsonl')):
            yield self.numerical_evidence(row['prediction_id'])

    def numerical_evidence(self, prediction_id):
        return _read(
            _path(self.root,
                  f'xai/local/rows/{_identifier(prediction_id)}.json'))

    def scoring_graph(self, prediction_id):
        path = _path(self.root,
                     f'xai/local/samples/{_identifier(prediction_id)}.json')
        if not path.exists():
            raise EvidenceValidationError('Original scoring graph not retained')
        graph = _read(path)
        _validate_graph(graph)
        return graph

    def availability(self, prediction_id):
        row = self.numerical_evidence(prediction_id)
        return {
            'walk_attribution':
            row.get('walk_attribution', {}).get('status', 'disabled'),
            'numerical':
            row['status'],
            'reason':
            row.get('reason'),
            'scoring_graph':
            row.get('graph_status',
                    'available' if row.get('sample_sha256') else 'unknown'),
            'graph_reason':
            row.get('graph_reason'),
            'temporal_validation':
            row.get('temporal_validation', {
                'status': 'unavailable',
                'reason': 'not_retained'
            })
        }

    def validate(self):
        if self.root.is_symlink() or not self.root.is_dir():
            raise EvidenceValidationError(
                'Expected a retained bundle directory')
        manifest_path = _path(self.root, 'manifest.json')
        success = _path(self.root, '_SUCCESS')
        if not success.is_file() or success.read_text() != _digest(
                manifest_path):
            raise EvidenceValidationError('Bundle was not completely published')
        manifest = self.manifest
        if (type(manifest.get('schema_version')) is not int
                or manifest['schema_version'] != SCHEMA_VERSION
                or manifest.get('kind') != 'prediction_evidence'):
            raise EvidenceValidationError('Unsupported evidence schema')
        inventory = _path(self.root, 'inventory.jsonl')
        if _digest(inventory) != manifest['inventory_sha256']:
            raise EvidenceValidationError('Inventory checksum mismatch')
        # Disk-backed uniqueness and joins keep validation bounded in row count.
        with tempfile.TemporaryDirectory(
                prefix='evidence-validation-') as scratch:
            with sqlite3.connect(str(Path(scratch) / 'index.sqlite')) as db:
                db.execute('CREATE TABLE files(path TEXT PRIMARY KEY)')
                db.execute(
                    'CREATE TABLE predictions(id TEXT PRIMARY KEY, value REAL, entity TEXT, cutoff INTEGER, task TEXT)'
                )
                db.execute('CREATE TABLE evidence(id TEXT PRIMARY KEY)')
                db.execute('CREATE TABLE intervals(id TEXT PRIMARY KEY)')
                db.execute(
                    'CREATE TABLE sources(key TEXT PRIMARY KEY, value TEXT)')
                try:
                    for entry in _lines(inventory):
                        path = _path(self.root, entry['path'])
                        db.execute('INSERT INTO files VALUES (?)',
                                   (entry['path'], ))
                        if (not path.is_file()
                                or path.stat().st_size != entry['bytes']
                                or _digest(path) != entry['sha256']):
                            raise EvidenceValidationError(
                                'Artifact integrity mismatch')
                    for path in _files(self.root):
                        relative = path.relative_to(self.root).as_posix()
                        if relative not in ('manifest.json', 'inventory.jsonl',
                                            '_SUCCESS'):
                            found_file = db.execute(
                                'SELECT 1 FROM files WHERE path=?',
                                (relative, )).fetchone()
                            if not found_file:
                                raise EvidenceValidationError(
                                    'Unlisted artifact')
                    for source in _lines(
                            _path(self.root, 'source-records.jsonl')):
                        locator = source['source']
                        if (not isinstance(locator['file'], str)
                                or not locator['file']
                                or not _ID.fullmatch(locator['sha256'])
                                or type(locator['row']) is not int
                                or locator['row'] < 1
                                or not isinstance(locator['column'], str)
                                or not locator['column']):
                            raise EvidenceValidationError(
                                'Invalid source record locator')
                        value = source['source_value']
                        if value is not None and (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not math.isfinite(value)):
                            raise EvidenceValidationError(
                                'Invalid source numerical value')
                        key = _json(
                            [source['table'], source['pkey'],
                             source['column']]).decode()
                        db.execute('INSERT INTO sources VALUES (?, ?)',
                                   (key, _json(source).decode()))
                    binary = manifest['native_provenance'].get(
                        'model_reconstruction', {}).get('task', {}).get(
                            'task_type') == 'binary_classification'
                    prediction_count = 0
                    for prediction in self.iter_predictions():
                        prediction_count += 1
                        identifier = prediction.get(
                            'PREDICTION_ID', prediction.get('prediction_id'))
                        if identifier is not None:
                            if binary:
                                # Evidence binds the emitted encoded class-1
                                # probability, never a thresholded label.
                                value = _emitted_class_probability(prediction)
                            else:
                                value = prediction.get('TARGET_PRED')
                                if (isinstance(value, bool)
                                        or not isinstance(value, (int, float))
                                        or not math.isfinite(value)):
                                    raise EvidenceValidationError(
                                        'Nonfinite or nonscalar emitted prediction')
                            import pandas as pd
                            entity = prediction.get('ENTITY',
                                                    prediction.get('ENTITY1'))
                            cutoff = prediction.get('CUTOFF_NS')
                            if cutoff is None and prediction.get(
                                    'TIMESTAMP') is not None:
                                cutoff = pd.Timestamp(
                                    prediction['TIMESTAMP']).value
                            db.execute(
                                'INSERT INTO predictions VALUES (?, ?, ?, ?, ?)',
                                (_identifier(identifier), value,
                                 _json(entity).decode(), cutoff,
                                 _json(prediction.get('TASK_ROW_ID')).decode()))
                    counts = Counter()
                    for row in _lines(_path(self.root,
                                            'xai/local/index.jsonl')):
                        identifier = _identifier(row['prediction_id'])
                        db.execute('INSERT INTO evidence VALUES (?)',
                                   (identifier, ))
                        detail = self.numerical_evidence(identifier)
                        walk = detail.get('walk_attribution')
                        if walk is not None:
                            if walk.get(
                                    'prediction_id') != identifier or walk.get(
                                        'sample_sha256') != detail.get(
                                            'sample_sha256') or walk.get(
                                                'origin'
                                            ) != 'original_forward_attribution':
                                raise EvidenceValidationError(
                                    'Walk attribution identity mismatch')
                            if walk.get('schema') == 'walk_attribution/v2':
                                _validate_v2_walk(self, identifier, detail, walk)
                            elif walk.get('status') in ('available', 'truncated'):
                                from .walks import validate_result
                                validate_result(walk)
                            elif walk.get('status') not in ('unsupported',
                                                            'failed'):
                                raise EvidenceValidationError(
                                    'Invalid walk availability')
                        record_binary = (detail.get('output') or {}).get('binary')
                        if record_binary is not None and record_binary[
                                'emitted_encoded_positive_probability'] != detail[
                                    'prediction']:
                            raise EvidenceValidationError(
                                'Record probability not bound to the emitted '
                                'True_PROB')
                        if any(
                                detail.get(key) != value
                                for key, value in row.items()):
                            raise EvidenceValidationError(
                                'Index and row evidence disagree')
                        if detail.get('origin') != 'original_forward' or detail[
                                'status'] not in _STATES:
                            raise EvidenceValidationError(
                                'Invalid numerical evidence origin/status')
                        found = db.execute(
                            'SELECT value, entity, cutoff, task FROM predictions WHERE id=?',
                            (identifier, )).fetchone()
                        if found is None or found[0] != detail['prediction']:
                            raise EvidenceValidationError(
                                'Evidence not bound to emitted prediction ID/value'
                            )
                        if _loads(found[1]) != detail.get('entity'):
                            raise EvidenceValidationError(
                                'Evidence entity differs from emitted prediction'
                            )
                        if detail.get('time_ns') != (None if found[2] is None
                                                     else [found[2]]):
                            raise EvidenceValidationError(
                                'Evidence cutoff differs from emitted prediction'
                            )
                        if _loads(found[3]) is None or detail.get(
                                'task_row_id') != [_loads(found[3])]:
                            raise EvidenceValidationError(
                                'Evidence task row differs from emitted prediction'
                            )
                        if (manifest['evidence_complete'] and detail.get(
                                'graph_status', 'available'
                                if detail.get('sample_sha256') else 'unknown')
                                != 'available'):
                            raise EvidenceValidationError(
                                'Complete evidence has unavailable graph')
                        if (manifest['evidence_complete'] and detail.get(
                                'temporal_validation', {}).get('status')
                                in {'violated', 'unavailable'}):
                            raise EvidenceValidationError(
                                'Complete evidence has invalid temporal coverage'
                            )
                        if detail.get('sample_sha256'):
                            sample = self.scoring_graph(identifier)
                            if hashlib.sha256(_json(sample)).hexdigest(
                            ) != detail['sample_sha256']:
                                raise EvidenceValidationError(
                                    'Original graph fingerprint mismatch')
                        for cell in detail.get('cells', []):
                            if manifest[
                                    'source_lineage'] == 'provided_mapper_verified':
                                import numpy as np
                                key = _json([
                                    cell['table'], cell['pkey'], cell['column']
                                ]).decode()
                                source = db.execute(
                                    'SELECT value FROM sources WHERE key=?',
                                    (key, )).fetchone()
                                if source is None:
                                    raise EvidenceValidationError(
                                        'Missing source numerical record')
                                raw_source = _loads(source[0])['source_value']
                                mapper = cell.get('mapper_value',
                                                  cell.get('raw_value'))
                                if raw_source is None:
                                    if mapper is not None or not cell.get(
                                            'missing', True):
                                        raise EvidenceValidationError(
                                            'Source missingness mismatch')
                                else:
                                    dtype = cell.get('input_dtype',
                                                     'torch.float32')
                                    if dtype not in ('torch.float32',
                                                     'torch.float64'):
                                        raise EvidenceValidationError(
                                            'Unsupported source mapper dtype')
                                    cast = float(
                                        np.float32(raw_source)
                                    ) if dtype == 'torch.float32' else float(
                                        raw_source)
                                    # Exact comparison after the declared cast, not a loose source tolerance.
                                    if not math.isfinite(
                                            cast) or mapper != cast or cell.get(
                                                'missing', False):
                                        raise EvidenceValidationError(
                                            'Source value disagrees with mapper coordinate'
                                        )
                            value = cell.get('abs_gradient')
                            if value is not None and (isinstance(
                                    value, bool) or not math.isfinite(value)
                                                      or value < 0):
                                raise EvidenceValidationError(
                                    'Invalid gradient magnitude')
                            signed = cell.get('signed_gradient')
                            if signed is not None and (isinstance(
                                    signed, bool) or not math.isfinite(signed)
                                                       or abs(signed) != value):
                                raise EvidenceValidationError(
                                    'Signed gradient/magnitude mismatch')
                        counts[detail['status']] += 1
                    interval_count = 0
                    for interval in self.iter_intervals():
                        identifier = _identifier(interval['prediction_id'])
                        db.execute('INSERT INTO intervals VALUES (?)',
                                   (identifier, ))
                        match = db.execute(
                            'SELECT value FROM predictions WHERE id=?',
                            (identifier, )).fetchone()
                        if match is None or match[0] != interval.get(
                                'prediction'):
                            raise EvidenceValidationError(
                                'Interval has no matching prediction')
                        interval_count += 1
                    if 'intervals' in manifest:
                        if interval_count != prediction_count:
                            raise EvidenceValidationError(
                                'Incomplete interval coverage')
                        from ..uncertainty import PredictionIntervalCalibrator
                        calibrator = PredictionIntervalCalibrator.load(
                            _path(self.root, 'intervals/calibration'))
                        if calibrator.sha256 != manifest['intervals'][
                                'calibration_sha256']:
                            raise EvidenceValidationError(
                                'Calibration identity mismatch')
                        expected = calibrator.apply(
                            self.iter_intervals(),
                            model_sha256=manifest['provenance'].get(
                                'model_sha256',
                                manifest['native_provenance'].get(
                                    'bundle_sha256')),
                            scoring_sha256=manifest['provenance']
                            ['scoring_sha256'])
                        for actual, calculated in zip(self.iter_intervals(),
                                                      expected, strict=True):
                            if actual != calculated:
                                raise EvidenceValidationError(
                                    'Interval disagrees with calibration')
                    response_count = 0
                    for response in _lines(
                            _path(self.root, 'perturbations/responses.jsonl')):
                        identifier = _identifier(response['prediction_id'])
                        match = db.execute(
                            'SELECT value FROM predictions WHERE id=?',
                            (identifier, )).fetchone()
                        if (response.get('origin') != 'perturbation'
                                or match is None or
                                not math.isclose(match[0], response['baseline'],
                                                 abs_tol=1e-6, rel_tol=1e-6)
                                or response['signed_change']
                                != response['baseline'] -
                                response['perturbed_prediction']
                                or response['absolute_change'] != abs(
                                    response['signed_change'])
                                or response.get('resampled') is not False):
                            raise EvidenceValidationError(
                                'Invalid controlled-removal response')
                        if response[
                                'original_sample_sha256'] != self.numerical_evidence(
                                    identifier)['sample_sha256']:
                            raise EvidenceValidationError(
                                'Perturbation graph identity mismatch')
                        response_count += 1
                    if 'perturbations' in manifest and response_count != manifest[
                            'perturbations']['responses']:
                        raise EvidenceValidationError(
                            'Perturbation coverage mismatch')
                    if dict(counts) != manifest['status_counts']:
                        raise EvidenceValidationError(
                            'Evidence status counts mismatch')
                    if manifest['evidence_complete']:
                        if (sum(counts.values()) != prediction_count
                                or any(counts[state]
                                       for state in ('failed', 'unknown',
                                                     'unsupported'))
                                or not manifest['native_provenance'].get(
                                    'complete')
                                or manifest['native_provenance'].get(
                                    'graph_failures', 0)
                                or any(manifest['native_provenance'].get(
                                    'temporal_status_counts', {}).get(state, 0)
                                       for state in ('violated',
                                                     'unavailable'))):
                            raise EvidenceValidationError(
                                'Complete evidence contradicts producer coverage/status'
                            )
                except (sqlite3.IntegrityError, KeyError, TypeError) as error:
                    raise EvidenceValidationError(
                        'Malformed or duplicate evidence record') from error
        return {
            'valid': True,
            'prediction_rows': prediction_count,
            'evidence_rows': sum(counts.values()),
            'evidence_complete': manifest['evidence_complete'],
            'status_counts': dict(counts)
        }

    def explain_classification(self, prediction_id, *, model,
                               selected_class=None, max_walks=10000, top_k=20):
        """Return restricted binary relevance and bounded walks on CUDA."""
        from .classification import explain_classification
        return explain_classification(
            self, prediction_id, model=model, selected_class=selected_class,
            max_walks=max_walks, top_k=top_k)

    def attribute_walks(self, *, model, config, output):
        """Attribute a trusted frozen model on retained graphs, without resampling."""
        from kumo.engine.evidence.walks import attribute_bundle
        return attribute_bundle(self, model=model, config=config, output=output)

    def export(self, destination):
        """Copy all retained bytes, including derived methods, without recomputing."""
        self.validate()
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock = destination.with_name(destination.name + '.lock')
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            staging = Path(
                tempfile.mkdtemp(prefix=destination.name + '.partial-',
                                 dir=destination.parent))
            _copy_tree(self.root, staging)
            PredictionEvidenceBundle(staging).validate()
            os.rename(staging, destination)
            return PredictionEvidenceBundle(destination)
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)

    def export_legacy(self, destination):
        """Write the prototype's xai tree into a new directory, never in place."""
        self.validate()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        _copy_tree(self.root / 'xai', destination / 'xai')
        return destination

    def with_intervals(self, calibrator, *, output, block_column=None,
                       member_column=None):
        """Create an immutable child bundle using a compatible calibrator."""
        self.validate()
        manifest = self.manifest
        native = manifest['native_provenance']
        provenance = manifest['provenance']
        model_hash = provenance.get('model_sha256', native.get('bundle_sha256'))
        scoring_hash = provenance.get('scoring_sha256')
        if not model_hash or not scoring_hash:
            raise EvidenceValidationError(
                'Bundle lacks calibration compatibility identities')
        if (model_hash != calibrator.spec.model_sha256
                or scoring_hash != calibrator.spec.scoring_sha256):
            raise EvidenceValidationError(
                'Calibration model/scoring identity mismatch')
        if calibrator.spec.unit == 'block' and (not block_column
                                                or not member_column):
            raise ValueError(
                'Block intervals require explicit block/member columns')
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
            _copy_tree(self.root, staging)
            interval_root = staging / 'intervals'
            # A calibration revision is immutable, never replaced in place.
            if interval_root.exists():
                raise EvidenceValidationError(
                    'Bundle already contains intervals')
            interval_root.mkdir()
            calibrator.save(interval_root / 'calibration')

            def predictions():
                import pandas as pd
                for row in self.iter_predictions():
                    record = {
                        'prediction_id':
                        row.get('PREDICTION_ID', row.get('prediction_id')),
                        'prediction':
                        row['TARGET_PRED']
                    }
                    timestamp = row.get('TIMESTAMP')
                    if timestamp is not None:
                        record['cutoff_ns'] = pd.Timestamp(timestamp).value
                    if calibrator.spec.unit == 'block':
                        record['block_id'] = str(row[block_column])
                        record['member_id'] = str(row[member_column])
                    yield record

            with (interval_root / 'rows.jsonl').open('wb') as stream:
                for row in calibrator.apply(predictions(),
                                            model_sha256=model_hash,
                                            scoring_sha256=scoring_hash):
                    stream.write(_json(row) + b'\n')
            manifest['parent_manifest_sha256'] = _digest(self.root /
                                                         'manifest.json')
            manifest['intervals'] = {
                'calibration_sha256': calibrator.sha256,
                'rows': 'intervals/rows.jsonl'
            }
            _seal(staging, manifest)
            PredictionEvidenceBundle(staging).validate()
            os.rename(staging, destination)
            return PredictionEvidenceBundle(destination)
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)

    def iter_source_records(self):
        yield from _lines(_path(self.root, 'source-records.jsonl'))

    def iter_intervals(self):
        yield from _lines(_path(self.root, 'intervals/rows.jsonl'))

    def perturb(self, *, model, plan, output, prediction_ids=None):
        """Publish controlled-removal responses in a new immutable bundle.

        The explicit optional ID selection limits computation, never graph capture.
        Invalid groups or a failed baseline check leave an unpublished partial tree.
        """
        from dataclasses import asdict

        from kumo_core.explain.perturbation import RecordRemovalPlan

        from kumo.engine.perturbation import responses_from_trainer

        if not isinstance(plan, RecordRemovalPlan):
            raise TypeError('plan must be RecordRemovalPlan')
        self.validate()
        selected = (None if prediction_ids is None else
                    {_identifier(value)
                     for value in prediction_ids})
        if selected == set():
            raise ValueError('Select at least one prediction')
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
            _copy_tree(self.root, staging)
            root = staging / 'perturbations'
            if root.exists():
                raise EvidenceValidationError(
                    'Bundle already contains perturbations')
            root.mkdir()
            count = 0
            manifest = self.manifest
            with (root / 'responses.jsonl').open('wb') as stream:
                for record in self.iter_numerical_evidence():
                    identifier = record['prediction_id']
                    if selected is not None and identifier not in selected:
                        continue
                    sample = self.scoring_graph(identifier)
                    for response in responses_from_trainer(
                            model, sample, record,
                            manifest['native_provenance'], plan):
                        stream.write(_json(response) + b'\n')
                        count += 1
                    if selected is not None:
                        selected.remove(identifier)
            if selected:
                raise EvidenceValidationError(
                    'Selected prediction IDs were not retained')
            manifest['parent_manifest_sha256'] = _digest(self.root /
                                                         'manifest.json')
            manifest['perturbations'] = {
                'plan': asdict(plan),
                'responses': count,
                'origin': 'perturbation',
                'rows': 'perturbations/responses.jsonl'
            }
            _seal(staging, manifest)
            PredictionEvidenceBundle(staging).validate()
            os.rename(staging, destination)
            return PredictionEvidenceBundle(destination)
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def _emitted_class_probability(prediction):
    """Binary rows emit `False_PROB` and `True_PROB`; return `True_PROB`."""
    probabilities = (prediction.get('True_PROB'), prediction.get('False_PROB'))
    for value in probabilities:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise EvidenceValidationError('Invalid emitted class probability')
    if abs(probabilities[0] + probabilities[1] - 1) > 1e-6:
        raise EvidenceValidationError(
            'Emitted class probabilities do not sum to one')
    label = prediction.get('TARGET_PRED')
    if label is not None and not isinstance(label, bool):
        raise EvidenceValidationError('Binary TARGET_PRED must be null or bool')
    return probabilities[0]


def _validate_v2_walk(bundle, identifier, detail, walk):
    """A native v2 row: valid on its retained graph and bound to the record.

    The row's emitted class-1 probability must equal the record's
    `prediction`, itself bound bitwise to the emitted `True_PROB`.
    """
    from .walks import validate_v2
    sample = (bundle.scoring_graph(identifier) if walk.get('status') in (
        'available', 'incomplete') else None)
    validate_v2(walk, sample)
    output, record_output = walk.get('output'), detail.get('output')
    if output is None or walk.get('status') not in ('available', 'incomplete'):
        return
    if record_output is not None and (
            record_output['head_output'] != output['head_output']):
        raise EvidenceValidationError('Walk and record logits differ')
    binary = output.get('binary')
    if binary is not None and (binary['emitted_encoded_positive_probability']
                               != detail['prediction']):
        raise EvidenceValidationError(
            'Walk probability not bound to the emitted True_PROB')


def _seal(root, manifest):
    with (root / 'inventory.jsonl').open('wb') as stream:
        for path in _files(root):
            if path.relative_to(root).as_posix() in ('manifest.json',
                                                     '_SUCCESS',
                                                     'inventory.jsonl'):
                continue
            stream.write(
                _json({
                    'path': path.relative_to(root).as_posix(),
                    'bytes': path.stat().st_size,
                    'sha256': _digest(path)
                }) + b'\n')
    manifest['inventory_sha256'] = _digest(root / 'inventory.jsonl')
    (root / 'manifest.json').write_bytes(_json(manifest))
    (root / '_SUCCESS').write_text(_digest(root / 'manifest.json'))
