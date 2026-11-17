# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Class orientation, selection and calibration of binary walk evidence.

Expected orientations come from category counts alone (the fitted mapping is
sorted by count, descending and stable; encoded class 1 is the larger raw
label), never from Engine encode calls or output column names.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from kumoapi.encoder import Index
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType

from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.explain.walk_attribution import (
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
)
from kumo_core.explain.walk_binary import (
    binary_block,
    calibration_record,
    emitted_probabilities,
    output_block,
    resolve_selection,
    same_label,
    selected_score,
    sigmoid,
    verify_binary_module,
)
from kumo_core.modeling.calibration import PlatScaling
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import BinaryCrossEntropyLoss, FocalLoss, MSELoss

BINARY = TaskType.BINARY_CLASSIFICATION


def encoder(counts):
    index = Index()
    index._stats = {ColStatType.CATEGORY_COUNTS: [list(c) for c in counts]}
    return TargetEncoder.from_encoder(BINARY, index)


def module(counts=((True, 123), (False, 37)), calibration=None, loss=None, head=None,
           task=BINARY, target_encoder=None):
    calibrator = PlatScaling(BINARY)
    if calibration is not None:
        with torch.no_grad():
            calibrator.weight_and_bias.copy_(torch.tensor([[calibration[0]],
                                                           [calibration[1]]]))
        calibrator.set_extra_state({'is_fitted': True})
    return SimpleNamespace(
        task_type=task, loss=loss or BinaryCrossEntropyLoss(BINARY),
        target_encoder=target_encoder or encoder(counts),
        head=head or NodePredHead(4, 1, num_layers=1, norm=None, param_keys=[]),
        calibrator=calibrator)


# Hand table: (counts, mapping order, reverse, encoded class of each label).
ORDERINGS = [
    (((True, 123), (False, 37)), [True, False], True, {True: 1, False: 0}),
    (((False, 123), (True, 37)), [False, True], False, {True: 1, False: 0}),
    (((True, 50), (False, 50)), [True, False], True, {True: 1, False: 0}),
    (((False, 50), (True, 50)), [False, True], False, {True: 1, False: 0}),
    (((1, 80), (0, 20)), [1, 0], True, {1: 1, 0: 0}),
    (((0, 80), (1, 20)), [0, 1], False, {1: 1, 0: 0}),
]


@pytest.mark.parametrize('counts,order,reverse,encoded', ORDERINGS)
def test_six_orderings_encode_the_larger_label_as_class_one(counts, order, reverse,
                                                            encoded):
    spec = verify_binary_module(module(counts), None)
    assert [entry['raw_label'] for entry in spec.label_mapping] == order
    assert all(type(entry['raw_label']) is type(order[0]) for entry in spec.label_mapping)
    assert {e['raw_label']: e['encoded_class'] for e in spec.label_mapping} == encoded
    assert [e['mapper_index'] for e in spec.label_mapping] == [0, 1]
    assert spec.target_encoder_reverse is reverse
    positive = max(order)
    assert same_label(spec.encoded_positive_label, positive)
    assert spec.selected_class_source == 'encoded_positive_default'
    assert spec.logit_sign == 1
    other = min(order)
    flipped = verify_binary_module(module(counts), other)
    assert flipped.logit_sign == -1 and flipped.selected_class_source == 'config'


def test_single_label_mapping_is_rejected():
    with pytest.raises(UnsupportedWalkAttribution, match='exactly two labels'):
        verify_binary_module(module(((True, 10), )), None)


def test_three_labels_never_reach_walk_evidence():
    with pytest.raises(ValueError, match='two distinct labels'):
        encoder((('a', 3), ('b', 2), ('c', 1)))


@pytest.mark.parametrize('counts,selected', [
    (((True, 123), (False, 37)), 1), (((True, 123), (False, 37)), 'True'),
    (((1, 80), (0, 20)), True), (((1, 80), (0, 20)), np.int64(1).item() + 1)])
def test_selected_class_is_type_strict(counts, selected):
    with pytest.raises(UnsupportedWalkAttribution, match='not a label'):
        verify_binary_module(module(counts), selected)
    assert not same_label(1, True) and not same_label(True, 1) and same_label(True, True)


@pytest.mark.parametrize('broken,match', [
    ({'loss': FocalLoss(BINARY)}, 'BinaryCrossEntropyLoss'),
    ({'target_encoder': NumericalTargetEncoder(0, 1)}, 'categorical target encoder'),
    ({'head': NodePredHead(4, 2, num_layers=1, norm=None, param_keys=[])}, 'one head'),
])
def test_task_level_rejections(broken, match):
    with pytest.raises(UnsupportedWalkAttribution, match=match):
        verify_binary_module(module(**broken), None)


def test_selection_by_task():
    assert resolve_selection(module(), WalkAttributionConfig()).logit_sign == 1
    assert resolve_selection(module(), WalkAttributionConfig(selected_class=False)).logit_sign == -1
    with pytest.raises(UnsupportedWalkAttribution, match='output_index 0'):
        resolve_selection(module(), WalkAttributionConfig(output_index=1))
    regression = SimpleNamespace(task_type=TaskType.REGRESSION, loss=MSELoss(TaskType.REGRESSION),
                                 target_encoder=NumericalTargetEncoder(0, 1))
    selection = resolve_selection(regression, WalkAttributionConfig(output_index=2))
    assert (selection.task_type, selection.output_index, selection.logit_sign,
            selection.binary) == ('regression', 2, 1, None)
    with pytest.raises(UnsupportedWalkAttribution, match='binary classification'):
        resolve_selection(regression, WalkAttributionConfig(selected_class=True))
    regression.loss = FocalLoss(BINARY)
    with pytest.raises(UnsupportedWalkAttribution, match='MSE'):
        resolve_selection(regression, WalkAttributionConfig())
    with pytest.raises(UnsupportedWalkAttribution, match='does not support'):
        resolve_selection(module(task=TaskType.MULTICLASS_CLASSIFICATION),
                          WalkAttributionConfig())


@pytest.mark.parametrize('selected,logit,model_probability,emitted', [
    (True, 0.5, 0.6224593312018546, 0.574442516811659),
    (False, -0.5, 0.3775406687981454, 0.425557483188341),
])
def test_fixture_b4_platt_numbers(selected, logit, model_probability, emitted):
    m = module(calibration=(0.8, -0.1))
    spec = verify_binary_module(m, selected)
    block = binary_block(spec, m, torch.tensor([[0.5]]))
    assert block['selected_logit'] == logit
    assert block['model_probability_selected'] == pytest.approx(model_probability, abs=1e-15)
    assert block['emitted_probability_selected'] == pytest.approx(emitted, abs=1e-6)
    calibration = block['calibration']
    assert calibration['kind'] == 'platt' and calibration['fitted'] is True
    assert calibration['weight_raw'] == pytest.approx(0.8) and calibration['bias'] == (
        pytest.approx(-0.1))
    assert calibration['weight_effective'] == calibration['weight_raw']
    assert block['emitted_column_selected'] == ('True_PROB' if selected else 'False_PROB')


def test_negative_platt_weight_is_clamped_and_unfitted_is_kind_none():
    clamped = calibration_record(module(calibration=(-0.2, 0.0)))
    assert clamped['weight_raw'] == pytest.approx(-0.2)
    assert clamped['weight_effective'] == float(np.float32(1e-3))
    assert clamped['weight_floor'] == 0.001
    unfitted = calibration_record(module())
    assert unfitted == {'kind': 'none', 'fitted': False, 'weight_raw': None,
                        'weight_effective': None, 'bias': None, 'weight_floor': 0.001}
    m = module()
    block = binary_block(verify_binary_module(m, None), m, torch.tensor([[0.5]]))
    assert block['emitted_encoded_positive_probability'] == float(torch.tensor([0.5]).sigmoid())


def emitted_like_predict(m, logits):
    """`on_predict_batch_end`: calibrate if fitted, then `to_probability`."""
    if m.calibrator.is_fitted:
        logits = m.calibrator(logits)
    return m.loss.to_probability(logits).cpu()


@pytest.mark.parametrize('calibration', [None, (0.8, -0.1), (-0.2, 0.3), (1.7, 2.5)])
def test_emitted_probabilities_bind_bitwise_to_both_columns(calibration):
    m = module(calibration=calibration)
    logits = torch.linspace(-12, 12, 2001).reshape(-1, 1)
    positive = verify_binary_module(m, True)
    negative = verify_binary_module(m, False)
    float64_complement = below_half = 0
    for row in logits:
        reference = emitted_like_predict(m, row.reshape(1, 1))
        true_prob, false_prob = float(reference[0, 1]), float(reference[0, 0])
        block = binary_block(positive, m, row)
        assert block['emitted_encoded_positive_probability'] == true_prob
        assert block['emitted_probability_selected'] == true_prob
        other = binary_block(negative, m, row)
        assert other['emitted_probability_selected'] == false_prob  # C7
        assert other['emitted_probability_selected'] == float(
            np.float32(1.0) - np.float32(true_prob))
        float64_complement += (1.0 - true_prob) != false_prob
        below_half += true_prob < 0.5
    # Why C7 matters: below one half, `1 - True_PROB` in float64 is not the
    # emitted float32 column (above one half the subtraction is exact).
    assert (float64_complement > 0) == (below_half > 0)


def test_output_block_and_score():
    m = module(calibration=(0.8, -0.1))
    selection = resolve_selection(m, WalkAttributionConfig(selected_class=False))
    block = output_block(selection, m, torch.tensor([0.5], requires_grad=True))
    assert block['task_type'] == 'binary_classification' and block['output_index'] == 0
    assert block['head_output'] == 0.5 and selected_score(block) == -0.5
    regression = SimpleNamespace(task_type=TaskType.REGRESSION, loss=MSELoss(TaskType.REGRESSION),
                                 target_encoder=NumericalTargetEncoder(0, 1))
    selection = resolve_selection(regression, WalkAttributionConfig(output_index=1))
    block = output_block(selection, regression, torch.tensor([3.0, -4.0]))
    assert block == {'task_type': 'regression', 'output_index': 1, 'head_output': -4.0,
                     'binary': None}
    assert selected_score(block) == -4.0


def test_recording_does_not_change_the_calibrator():
    m = module(calibration=(0.8, -0.1))
    before = (m.calibrator.weight_and_bias.detach().clone(), m.calibrator.is_fitted)
    emitted_probabilities(m, torch.tensor([[0.5]]))
    calibration_record(m)
    assert torch.equal(m.calibrator.weight_and_bias, before[0])
    assert m.calibrator.is_fitted == before[1]


def test_sigmoid_is_stable():
    assert sigmoid(0.0) == 0.5 and sigmoid(800.0) == 1.0 and sigmoid(-800.0) == 0.0
    assert sigmoid(-2.0) + sigmoid(2.0) == pytest.approx(1.0, abs=1e-15)
