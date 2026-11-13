# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration tests shared by source and isolated installed-wheel gates."""

import math
import random

import pytest

from kumo.engine.uncertainty import (
    CalibrationSpec,
    PredictionIntervalCalibrator,
)

IDENTITY = {'model_sha256': 'a' * 64, 'scoring_sha256': 'b' * 64}


def calibrate(n=9, **kwargs):
    rows = [{'prediction_id': str(i), 'prediction': 0.} for i in range(n)]
    return PredictionIntervalCalibrator.fit(
        rows, {str(i): i
               for i in range(n)}, spec=CalibrationSpec(**IDENTITY, **kwargs),
        population_id='cal',
        separation_declaration='Synthetic independent calibration population')


def test_exact_rank_and_insufficient_population():
    cal = calibrate()
    assert cal.manifest['quantile'] == 8
    result, = cal.apply([{
        'prediction_id': 'new',
        'prediction': 10
    }], **IDENTITY)
    assert (result['lower'], result['upper']) == (2, 18)
    assert calibrate(
        8).availability['reason'] == 'insufficient_calibration_units'


def test_save_load_and_corruption(tmp_path):
    cal = calibrate()
    cal.save(tmp_path / 'cal')
    loaded = PredictionIntervalCalibrator.load(tmp_path / 'cal')
    assert loaded.sha256 == cal.sha256
    with pytest.raises(FileExistsError):
        cal.save(tmp_path / 'cal')
    (tmp_path / 'cal/calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        PredictionIntervalCalibrator.load(tmp_path / 'cal')


def test_nonnegative_empty_and_unchanged_raw_prediction():
    cal = calibrate(nonnegative=True)
    row, = cal.apply([{'prediction_id': 'new', 'prediction': -20}], **IDENTITY)
    assert row['prediction'] == -20
    assert row['empty'] and row['lower'] is None and row['upper'] is None
    assert row['raw_upper'] == -12


def test_identity_and_overlap_guards():
    cal = calibrate()
    with pytest.raises(ValueError, match='identity'):
        list(cal.apply([], model_sha256='c' * 64, scoring_sha256='b' * 64))
    with pytest.raises(ValueError, match='overlaps'):
        list(cal.apply([{'prediction_id': '0', 'prediction': 1}], **IDENTITY))
    with pytest.raises(ValueError, match='previously used'):
        cal.evaluate([], {}, population_id='cal', **IDENTITY)


def test_block_maxima_and_complete_members():
    rows = [{
        'prediction_id': f'{b}-{m}',
        'prediction': 0,
        'block_id': str(b),
        'member_id': m
    } for b in range(9) for m in ('a', 'b')]
    outcomes = {
        r['prediction_id']:
        int(r['block_id']) + (10 if r['member_id'] == 'b' else 0)
        for r in rows
    }
    arguments = dict(spec=CalibrationSpec(**IDENTITY,
                                          unit='block'), population_id='cal',
                     separation_declaration='Independent blocks',
                     expected_block_members=['a', 'b'])
    cal = PredictionIntervalCalibrator.fit(rows, outcomes, **arguments)
    assert cal.manifest['quantile'] == 18
    assert cal.availability['units'] == 9
    with pytest.raises(ValueError, match='Identities'):
        list(
            cal.apply([{
                'prediction_id': 'future',
                'prediction': 0,
                'member_id': 'a'
            }], **IDENTITY))
    with pytest.raises(ValueError, match='Incomplete'):
        list(
            cal.apply([{
                'prediction_id': 'future',
                'prediction': 0,
                'block_id': 'future',
                'member_id': 'a'
            }], **IDENTITY))
    with pytest.raises(ValueError, match='Duplicate prediction block'):
        list(
            cal.apply([{
                'prediction_id': f'future{i}',
                'prediction': 0,
                'block_id': 'future',
                'member_id': 'a'
            } for i in range(2)], **IDENTITY))
    with pytest.raises(ValueError, match='Incomplete'):
        PredictionIntervalCalibrator.fit(
            rows[:-1], {
                r['prediction_id']: outcomes[r['prediction_id']]
                for r in rows[:-1]
            }, **arguments)


def test_temporal_label_availability():
    rows = [{
        'prediction_id': str(i),
        'prediction': 0,
        'cutoff_ns': 11 + i,
        'label_available_ns': 20 + i
    } for i in range(9)]
    cal = PredictionIntervalCalibrator.fit(
        rows, {str(i): i
               for i in range(9)}, spec=CalibrationSpec(**IDENTITY,
                                                        fitted_through_ns=10),
        population_id='cal', separation_declaration='Future calibration rows')
    with pytest.raises(ValueError, match='label availability'):
        list(
            cal.apply([{
                'prediction_id': 'new',
                'prediction': 0,
                'cutoff_ns': 28
            }], **IDENTITY))
    assert list(
        cal.apply([{
            'prediction_id': 'new',
            'prediction': 0,
            'cutoff_ns': 29
        }], **IDENTITY))


def test_empirical_independent_coverage():
    rng = random.Random(42)
    rows = [{'prediction_id': f'cal{i}', 'prediction': 0} for i in range(2000)]
    cal = PredictionIntervalCalibrator.fit(
        rows, {r['prediction_id']: rng.gauss(0, 1)
               for r in rows}, spec=CalibrationSpec(**IDENTITY),
        population_id='cal',
        separation_declaration='Independent synthetic draws')
    test = [{'prediction_id': f'test{i}', 'prediction': 0} for i in range(5000)]
    report = cal.evaluate(test,
                          {r['prediction_id']: rng.gauss(0, 1)
                           for r in test}, population_id='future', **IDENTITY)
    assert .88 < report['empirical_coverage'] < .93
    p, n, z = report['empirical_coverage'], len(test), 1.645
    lower = ((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z /
                                                  (4 * n * n))) /
             (1 + z * z / n))
    assert lower > .88
    assert report['mean_width'] > 0
    assert report['mean_raw_interval_score'] >= report['mean_width']


def test_empirical_block_coverage_with_related_rows():
    rng = random.Random(71)

    def population(prefix, count):
        rows, outcomes = [], {}
        for i in range(count):
            shared = rng.gauss(0, 1)
            for member in ('a', 'b'):
                key = f'{prefix}-{i}-{member}'
                rows.append({
                    'prediction_id': key,
                    'prediction': 0,
                    'block_id': f'{prefix}-{i}',
                    'member_id': member
                })
                outcomes[key] = shared + rng.gauss(0, .2)
        return rows, outcomes

    rows, outcomes = population('cal', 2000)
    cal = PredictionIntervalCalibrator.fit(
        rows, outcomes, spec=CalibrationSpec(**IDENTITY,
                                             unit='block'), population_id='cal',
        separation_declaration='Independent synthetic blocks',
        expected_block_members=('a', 'b'))
    test, observed = population('future', 5000)
    report = cal.evaluate(test, observed, population_id='future', **IDENTITY)
    assert report['blocks'] == 5000
    assert report['empirical_coverage'] >= report['block_coverage']
    p, n, z = report['block_coverage'], report['blocks'], 1.645
    lower = ((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z /
                                                  (4 * n * n))) /
             (1 + z * z / n))
    assert lower > .88


@pytest.mark.parametrize('value', [True, float('nan'), float('inf')])
def test_invalid_values(value):
    with pytest.raises(ValueError):
        list(calibrate().apply([{
            'prediction_id': 'new',
            'prediction': value
        }], **IDENTITY))
