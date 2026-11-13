# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-managed split conformal intervals for frozen scalar predictions.

Coverage is marginal over exchangeable calibration units, not conditional on
an entity. Block maxima target simultaneous coverage within a declared block.
Chronological separation does not establish exchangeability or prevent drift.
This module never fits, loads, or executes a prediction model.
"""

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a scalar prediction or outcome')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Predictions and outcomes must be finite')
    return number


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('Identities must be nonempty strings')
    return value


def _rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict]:
    with tempfile.TemporaryDirectory(prefix='interval-identities-') as temp:
        with sqlite3.connect(str(Path(temp) / 'identities.sqlite')) as database:
            database.execute('PRAGMA cache_size=-1024')
            database.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
            for original in rows:
                row = dict(original)
                key = _identity(row['prediction_id'])
                try:
                    database.execute('INSERT INTO ids VALUES (?)', (key, ))
                except sqlite3.IntegrityError as error:
                    raise ValueError('Duplicate prediction_id') from error
                row['prediction'] = _number(row['prediction'])
                yield row


def _blocks(rows: Iterable[dict], expected: list[str] | None) -> Iterator[dict]:
    if expected is None:
        yield from rows
        return
    with tempfile.TemporaryDirectory(prefix='interval-blocks-') as temp:
        with sqlite3.connect(str(Path(temp) / 'blocks.sqlite')) as database:
            database.execute('PRAGMA cache_size=-1024')
            database.execute('CREATE TABLE members (block TEXT, member TEXT, '
                             'PRIMARY KEY (block, member))')
            for row in rows:
                block, member = _identity(row.get('block_id')), _identity(
                    row.get('member_id'))
                if member not in expected:
                    raise ValueError(
                        'Prediction is outside calibrated block population')
                try:
                    database.execute('INSERT INTO members VALUES (?, ?)',
                                     (block, member))
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        'Duplicate prediction block member') from error
                yield row
            incomplete = database.execute(
                'SELECT block FROM members GROUP BY block HAVING COUNT(*) != ? LIMIT 1',
                (len(expected), )).fetchone()
            if incomplete:
                raise ValueError('Incomplete prediction block population')


@dataclass(frozen=True)
class CalibrationSpec:
    """Frozen calibration policy; hashes include target and sampler semantics.

    ``scoring_sha256`` must identify the task, horizon, preprocessing, sampling
    and output policy. ``fitted_through_ns`` enables strict temporal checks:
    calibration rows then require ``cutoff_ns`` and ``label_available_ns``.
    Operators attest population separation when historical provenance is not
    independently available; that declaration is retained, not verified fact.
    """

    model_sha256: str
    scoring_sha256: str
    coverage: float = 0.9
    unit: str = 'row'
    nonnegative: bool = False
    output_coordinate: str = 'decoded_model_output'
    units: str = 'target_units'
    horizon: str | None = None
    fitted_through_ns: int | None = None

    def __post_init__(self):
        for digest in (self.model_sha256, self.scoring_sha256):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Model and scoring identities must be SHA256')
        if isinstance(self.coverage, bool) or not 0 < self.coverage < 1:
            raise ValueError('coverage must be between zero and one')
        if self.unit not in {'row', 'block'}:
            raise ValueError('unit must be row or block')
        if type(self.nonnegative) is not bool:
            raise ValueError('nonnegative must be boolean')
        if self.output_coordinate != 'decoded_model_output':
            raise ValueError('Only raw decoded model output is supported')
        _identity(self.units)
        if (self.fitted_through_ns is not None
                and type(self.fitted_through_ns) is not int):
            raise ValueError('fitted_through_ns must be integer nanoseconds')


class PredictionIntervalCalibrator:
    """Immutable calibration payload with integrity-checked local retention."""
    def __init__(self, payload: Mapping[str, Any]):
        # Defensive copy also rejects NaN and non-JSON executable objects.
        self._payload = json.loads(_bytes(payload))
        if (type(self._payload.get('schema_version')) is not int
                or self._payload['schema_version'] != 1):
            raise ValueError('Unsupported calibration schema version')
        if self._payload.get('method') != 'split_conformal_absolute_residual':
            raise ValueError('Unsupported calibration method')
        self._spec = CalibrationSpec(**self._payload['spec'])
        residuals = self._payload['unit_residuals']
        if not residuals or any(_number(r) < 0 for r in residuals):
            raise ValueError('Invalid calibration residuals')
        rank = math.ceil((len(residuals) + 1) * self.spec.coverage)
        expected = sorted(residuals)[rank -
                                     1] if rank <= len(residuals) else None
        if self._payload['quantile'] != expected:
            raise ValueError('Calibration quantile does not match residuals')
        _identity(self._payload['population_id'])
        ids = self._payload['prediction_ids']
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Invalid calibration prediction identities')
        for key in ids:
            _identity(key)
        members = self._payload['expected_block_members']
        blocks = self._payload['calibration_block_ids']
        if self.spec.unit == 'row':
            if members is not None or blocks or len(residuals) != len(ids):
                raise ValueError('Inconsistent row calibration payload')
        elif (not isinstance(members, list) or not members
              or len(members) != len(set(members))
              or len(blocks) != len(set(blocks))
              or len(blocks) != len(residuals)
              or len(ids) != len(blocks) * len(members)):
            raise ValueError('Inconsistent block calibration payload')
        for key in (members or []) + blocks:
            _identity(key)
        self._sha256 = _hash(self._payload)

    @property
    def spec(self) -> CalibrationSpec:
        return self._spec

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def availability(self) -> dict:
        return {
            'state': 'available'
            if self._payload['quantile'] is not None else 'not_calibrated',
            'reason': None if self._payload['quantile'] is not None else
            'insufficient_calibration_units',
            'units': len(self._payload['unit_residuals']),
        }

    @property
    def manifest(self) -> dict:
        return json.loads(_bytes(self._payload))

    @classmethod
    def fit(
        cls, predictions: Iterable[Mapping[str,
                                           Any]], outcomes: Mapping[str,
                                                                    float], *,
        spec: CalibrationSpec, population_id: str, separation_declaration: str,
        excluded_population_ids: Iterable[str] = (),
        excluded_prediction_ids: Iterable[str] = (),
        expected_block_members: Iterable[str] | None = None
    ) -> 'PredictionIntervalCalibrator':
        """Calibrate with explicitly aligned, previously unused outcomes.

        Row fields are ``prediction_id`` and ``prediction``. Block mode also
        requires ``block_id`` and ``member_id``; every block must contain the
        same operator-declared member population exactly once.
        """
        _identity(population_id)
        _identity(separation_declaration)
        excluded_populations = sorted(set(excluded_population_ids))
        excluded_ids = set(excluded_prediction_ids)
        if population_id in excluded_populations:
            raise ValueError('Calibration population was previously used')
        rows = list(_rows(predictions))
        ids = {r['prediction_id'] for r in rows}
        if not rows or ids != set(outcomes):
            raise ValueError('Outcomes must cover calibration IDs exactly')
        if ids & excluded_ids:
            raise ValueError('Calibration overlaps previously used rows')
        members = (sorted(set(expected_block_members))
                   if expected_block_members is not None else None)
        if spec.unit == 'block' and not members:
            raise ValueError('Block calibration requires expected members')
        if spec.unit == 'row' and members is not None:
            raise ValueError('Row calibration does not accept block members')
        for member in members or []:
            _identity(member)
        blocks: dict[str, list[float]] = {}
        block_members: dict[str, set[str]] = {}
        residuals = []
        label_end = None
        for row in rows:
            observed = _number(outcomes[row['prediction_id']])
            if spec.nonnegative and observed < 0:
                raise ValueError('Outcome violates nonnegative target support')
            if spec.fitted_through_ns is not None:
                cutoff, available = row.get('cutoff_ns'), row.get(
                    'label_available_ns')
                if (type(cutoff) is not int or type(available) is not int
                        or cutoff <= spec.fitted_through_ns
                        or available < cutoff):
                    raise ValueError('Invalid temporal calibration separation')
                label_end = max(label_end or available, available)
            residual = abs(observed - row['prediction'])
            if not math.isfinite(residual):
                raise ValueError('Residual overflow')
            if spec.unit == 'block':
                block = _identity(row.get('block_id'))
                member = _identity(row.get('member_id'))
                present = block_members.setdefault(block, set())
                if member in present:
                    raise ValueError('Duplicate block member')
                present.add(member)
                blocks.setdefault(block, []).append(residual)
            else:
                residuals.append(residual)
        if spec.unit == 'block':
            if any(present != set(members)
                   for present in block_members.values()):
                raise ValueError('Incomplete calibration block population')
            residuals = [max(blocks[key]) for key in sorted(blocks)]
        rank = math.ceil((len(residuals) + 1) * spec.coverage)
        quantile = sorted(residuals)[rank -
                                     1] if rank <= len(residuals) else None
        return cls({
            'schema_version':
            1,
            'method':
            'split_conformal_absolute_residual',
            'spec':
            asdict(spec),
            'population_id':
            population_id,
            'separation_declaration':
            separation_declaration,
            'excluded_population_ids':
            excluded_populations,
            'excluded_prediction_ids':
            sorted(excluded_ids),
            'prediction_ids':
            sorted(ids),
            'expected_block_members':
            members,
            'calibration_block_ids':
            sorted(blocks),
            'calibration_label_end_ns':
            label_end,
            'input_sha256':
            _hash({
                'rows': rows,
                'outcomes': dict(outcomes)
            }),
            'unit_residuals':
            residuals,
            'quantile':
            quantile,
            'assumption':
            'exchangeable_calibration_units',
            'limitations': [
                'No entity-conditional coverage guarantee',
                'Temporal separation does not establish exchangeability',
                'Population separation includes operator attestation'
            ],
        })

    def apply(self, predictions: Iterable[Mapping[str, Any]], *,
              model_sha256: str, scoring_sha256: str) -> Iterator[dict]:
        """Yield intervals in input order without changing prediction values.

        Consumers must exhaust the iterator before publishing: block
        completeness is checked at its end. Identity validation uses bounded
        SQLite caches, with temporary disk space proportional to row count.
        """
        if (model_sha256 != self.spec.model_sha256
                or scoring_sha256 != self.spec.scoring_sha256):
            raise ValueError('Calibration model or scoring identity mismatch')
        used = set(self._payload['prediction_ids'])
        used.update(self._payload['excluded_prediction_ids'])
        availability = self.availability
        for row in _blocks(_rows(predictions),
                           self._payload['expected_block_members']):
            if row['prediction_id'] in used:
                raise ValueError(
                    'Prediction overlaps calibration or excluded rows')
            if self.spec.unit == 'block':
                if row.get(
                        'block_id') in self._payload['calibration_block_ids']:
                    raise ValueError('Prediction overlaps calibration blocks')
                if row.get('member_id'
                           ) not in self._payload['expected_block_members']:
                    raise ValueError(
                        'Prediction is outside calibrated block population')
            end = self._payload['calibration_label_end_ns']
            if end is not None:
                cutoff = row.get('cutoff_ns')
                if type(cutoff) is not int or cutoff <= end:
                    raise ValueError(
                        'Prediction precedes calibration label availability')
            q = self._payload['quantile']
            lower = upper = raw_lower = None
            empty = False
            if q is not None:
                raw_lower, upper = row['prediction'] - q, row['prediction'] + q
                if not math.isfinite(raw_lower) or not math.isfinite(upper):
                    raise ValueError('Interval endpoint overflow')
                lower = max(0.,
                            raw_lower) if self.spec.nonnegative else raw_lower
                empty = lower > upper
            yield {
                **row, 'lower': None if empty else lower,
                'upper': None if empty else upper,
                'raw_lower': raw_lower,
                'raw_upper': upper,
                'empty': empty,
                'coverage': self.spec.coverage,
                'coverage_unit': self.spec.unit,
                'calibration_sha256': self.sha256,
                'output_coordinate': self.spec.output_coordinate,
                'availability': availability
            }

    def evaluate(self, predictions: Iterable[Mapping[str, Any]],
                 outcomes: Mapping[str, float], *, population_id: str,
                 model_sha256: str, scoring_sha256: str) -> dict:
        """Report forward-population coverage, widths and interval scores."""
        _identity(population_id)
        if population_id in {
                self._payload['population_id'],
                *self._payload['excluded_population_ids']
        }:
            raise ValueError('Evaluation population was previously used')
        rows = list(
            self.apply(predictions, model_sha256=model_sha256,
                       scoring_sha256=scoring_sha256))
        if not rows or {r['prediction_id'] for r in rows} != set(outcomes):
            raise ValueError('Evaluation outcomes must cover IDs exactly')
        if self._payload['quantile'] is None:
            raise ValueError('Insufficient calibration units')
        covered, widths, scores = [], [], []
        blocks: dict[str, list[bool]] = {}
        members: dict[str, set[str]] = {}
        for row in rows:
            y = _number(outcomes[row['prediction_id']])
            if self.spec.nonnegative and y < 0:
                raise ValueError('Outcome violates target support')
            inside = not row['empty'] and row['lower'] <= y <= row['upper']
            covered.append(inside)
            width = 0. if row['empty'] else row['upper'] - row['lower']
            widths.append(width)
            # Score the original real-valued interval; empty support
            # intersections have no standard interval-score endpoints.
            low, high = row['raw_lower'], row['raw_upper']
            scores.append(high - low + 2 / (1 - self.spec.coverage) *
                          (max(low - y, 0) + max(y - high, 0)))
            if self.spec.unit == 'block':
                block = _identity(row.get('block_id'))
                if block in self._payload['calibration_block_ids']:
                    raise ValueError('Evaluation overlaps calibration blocks')
                member = _identity(row.get('member_id'))
                present = members.setdefault(block, set())
                if member in present:
                    raise ValueError('Duplicate evaluation block member')
                present.add(member)
                blocks.setdefault(block, []).append(inside)
        if any(group != set(self._payload['expected_block_members'])
               for group in members.values()):
            raise ValueError('Incomplete evaluation block population')
        ordered = sorted(widths)
        return {
            'population_id':
            population_id,
            'rows':
            len(rows),
            'calibration_sha256':
            self.sha256,
            'input_sha256':
            _hash({
                'rows': rows,
                'outcomes': dict(outcomes)
            }),
            'empirical_coverage':
            sum(covered) / len(rows),
            'block_coverage':
            (sum(all(v)
                 for v in blocks.values()) / len(blocks) if blocks else None),
            'blocks':
            len(blocks),
            'mean_width':
            sum(widths) / len(rows),
            'median_width':
            (ordered[(len(rows) - 1) // 2] + ordered[len(rows) // 2]) / 2,
            'p90_width':
            ordered[math.ceil(.9 * len(rows)) - 1],
            'mean_raw_interval_score':
            sum(scores) / len(rows)
        }

    def save(self, destination: str | Path) -> Path:
        """Publish a new local directory, refusing overwrite or symlinks."""
        target = Path(destination).absolute()
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix='.calibration-', dir=target.parent))
        try:
            (staging / 'calibration.json').write_bytes(_bytes(self._payload))
            (staging / '_SUCCESS').write_text(self.sha256)
            # mkdir reserves destination exclusively; move files only after
            # reservation. Readers require the completion marker written last.
            target.mkdir()
            os.replace(staging / 'calibration.json',
                       target / 'calibration.json')
            os.replace(staging / '_SUCCESS', target / '_SUCCESS')
        finally:
            shutil.rmtree(staging)
        return target

    @classmethod
    def load(cls, source: str | Path) -> 'PredictionIntervalCalibrator':
        root = Path(source)
        paths = [root, root / 'calibration.json', root / '_SUCCESS']
        if any(path.is_symlink() for path in paths):
            raise ValueError('Calibration symlinks are not permitted')
        data = (root / 'calibration.json').read_bytes()
        if hashlib.sha256(data).hexdigest() != (root / '_SUCCESS').read_text():
            raise ValueError('Calibration integrity check failed')
        return cls(json.loads(data))
