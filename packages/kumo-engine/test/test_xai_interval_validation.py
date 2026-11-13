# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent adversarial validation of calibration ownership and blocks."""
import pytest

from kumo.engine.uncertainty import (
    CalibrationSpec,
    PredictionIntervalCalibrator,
)

MODEL = 'a' * 64
POLICY = 'b' * 64


def calibration(unit='row'):
    rows = [{
        'prediction_id': f'{block}-{member}',
        'prediction': 0.,
        'block_id': str(block),
        'member_id': member
    } for block in range(20) for member in ('small', 'large')]
    outcomes = {
        row['prediction_id']: 1. if row['member_id'] == 'small' else 10.
        for row in rows
    }
    return PredictionIntervalCalibrator.fit(
        rows, outcomes, spec=CalibrationSpec(model_sha256=MODEL,
                                             scoring_sha256=POLICY, unit=unit),
        population_id='calibration',
        separation_declaration='Independent fixture',
        **({
            'expected_block_members': ['small', 'large']
        } if unit == 'block' else {}))


def test_reloaded_block_interval_is_joint_and_requires_complete_block(tmp_path):
    original = calibration('block')
    original.save(tmp_path / 'calibration')
    loaded = PredictionIntervalCalibrator.load(tmp_path / 'calibration')
    rows = [{
        'prediction_id': 'fresh-small',
        'prediction': 50.,
        'block_id': 'fresh',
        'member_id': 'small'
    }]
    iterator = loaded.apply(rows, model_sha256=MODEL, scoring_sha256=POLICY)
    first = next(iterator)
    assert first['lower'] == 40. and first['upper'] == 60.
    with pytest.raises(ValueError, match='Incomplete'):
        next(iterator)
    assert loaded.sha256 == original.sha256


def test_interval_identity_cannot_be_relabelled_by_prediction(tmp_path):
    calibrator = calibration()
    with pytest.raises(ValueError, match='identity mismatch'):
        list(
            calibrator.apply([{
                'prediction_id': 'fresh',
                'prediction': 0
            }], model_sha256='c' * 64, scoring_sha256=POLICY))
    with pytest.raises(ValueError, match='overlaps'):
        list(
            calibrator.apply([{
                'prediction_id': '0-small',
                'prediction': 0
            }], model_sha256=MODEL, scoring_sha256=POLICY))


def test_saved_payload_rejects_method_substitution():
    payload = calibration().manifest
    payload['method'] = 'arbitrary_percentage'
    with pytest.raises(ValueError):
        PredictionIntervalCalibrator(payload)
