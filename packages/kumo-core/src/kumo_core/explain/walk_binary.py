# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explained coordinate, class orientation and calibration (frozen interface).

The explained scalar is `selected_logit = s * z`, with `z` the raw head
output before calibration and `s = +1` iff the selected class label is the
label the target encoder maps to class 1. Orientation is verified through
the fitted encoder and the loss, never inferred from column names. The
calibrator stays outside every explained function; probabilities are shown
beside the logit, never as relevance.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from kumo_core.explain.walk_attribution import UnsupportedWalkAttribution

PLATT_WEIGHT_FLOOR = 1e-3


@dataclass(frozen=True)
class BinarySpec:
    """Per-model class orientation, verified once per run."""
    loss: str
    label_mapping: tuple
    target_encoder_reverse: bool
    encoded_positive_label: object
    selected_class_label: object
    selected_class_source: str
    logit_sign: int


@dataclass(frozen=True)
class Selection:
    """What a method explains: `logit_sign * z[output_index]`."""
    task_type: str
    output_index: int
    logit_sign: int
    binary: BinarySpec | None


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def same_label(a, b):
    """Type-strict equality, so `1` never matches `True` (contract 2.4)."""
    return type(a) is type(b) and a == b


def verify_binary_module(module, selected_class):
    """Contract 2.4 steps 1-4; raises `UnsupportedWalkAttribution`."""
    import pyarrow as pa

    from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
    from kumo_core.train.loss import BinaryCrossEntropyLoss
    loss = getattr(module, 'loss', None)
    if not isinstance(loss, BinaryCrossEntropyLoss):
        raise UnsupportedWalkAttribution(
            'Binary walk evidence requires BinaryCrossEntropyLoss')
    encoder = getattr(module, 'target_encoder', None)
    if not isinstance(encoder, CategoricalTargetEncoder):
        raise UnsupportedWalkAttribution(
            'Binary walk evidence requires a categorical target encoder')
    head = module.head
    if getattr(head, 'out_channels', None) != 1 or getattr(head, 'param_keys', None):
        raise UnsupportedWalkAttribution('Binary walk evidence requires one head output')
    labels = [_plain(value) for value in encoder.tensor_mapper.mapping]
    if len(labels) != 2:
        raise UnsupportedWalkAttribution(
            f'Binary walk evidence requires exactly two labels (got {len(labels)})')
    mapping = []
    for label in labels:
        index = int(encoder.tensor_mapper.forward(pa.array([label])).reshape(-1)[0])
        encoded = int(encoder.encode_tensor(torch.tensor([[index]])).reshape(-1)[0])
        decoded = _plain(encoder.decode(torch.tensor([[encoded]])).iloc[0])
        if not same_label(decoded, label):
            raise UnsupportedWalkAttribution('Target mapping does not round-trip')
        mapping.append({'raw_label': label, 'mapper_index': index,
                        'encoded_class': encoded})
    if sorted(entry['encoded_class'] for entry in mapping) != [0, 1]:
        raise UnsupportedWalkAttribution('Target mapping is not a binary encoding')
    grid = torch.tensor([[-2.0], [0.0], [2.0]])
    if not torch.equal(loss.to_probability(grid)[..., 1], grid.sigmoid()[..., 0]):
        raise UnsupportedWalkAttribution('Loss probability is not sigmoid(z) for class 1')
    positive = next(entry['raw_label'] for entry in mapping if entry['encoded_class'] == 1)
    if selected_class is None:
        selected, source = positive, 'encoded_positive_default'
    else:
        matches = [entry['raw_label'] for entry in mapping
                   if same_label(entry['raw_label'], selected_class)]
        if not matches:
            raise UnsupportedWalkAttribution(
                f'selected_class {selected_class!r} is not a label of this model')
        selected, source = matches[0], 'config'
    return BinarySpec(loss=type(loss).__name__, label_mapping=tuple(mapping),
                      target_encoder_reverse=bool(encoder.reverse),
                      encoded_positive_label=positive, selected_class_label=selected,
                      selected_class_source=source,
                      logit_sign=1 if same_label(selected, positive) else -1)


def resolve_selection(module, config):
    """The explained coordinate of `module` under `config`, or unsupported."""
    from kumo_core.encoders.target.numerical import NumericalTargetEncoder
    from kumo_core.train.loss import MSELoss
    from kumoapi.task import TaskType
    task = TaskType(module.task_type)
    if task == TaskType.REGRESSION:
        if (not isinstance(getattr(module, 'loss', None), MSELoss)
                or not isinstance(getattr(module, 'target_encoder', None),
                                  NumericalTargetEncoder)):
            raise UnsupportedWalkAttribution(
                'Regression walk evidence requires MSE with a numerical target encoder')
        if config.selected_class is not None:
            raise UnsupportedWalkAttribution('selected_class requires binary classification')
        return Selection('regression', config.output_index, 1, None)
    if task == TaskType.BINARY_CLASSIFICATION:
        if config.output_index != 0:
            raise UnsupportedWalkAttribution('Binary walk evidence explains output_index 0')
        spec = verify_binary_module(module, config.selected_class)
        return Selection('binary_classification', 0, spec.logit_sign, spec)
    raise UnsupportedWalkAttribution(f'Walk evidence does not support {task.value} tasks')


def calibration_record(module):
    """The calibrator applied to emitted probabilities (contract 2.3)."""
    from kumo_core.modeling.calibration import PlatScaling
    calibrator = getattr(module, 'calibrator', None)
    if calibrator is None or not calibrator.is_fitted:
        return {'kind': 'none', 'fitted': False, 'weight_raw': None,
                'weight_effective': None, 'bias': None,
                'weight_floor': PLATT_WEIGHT_FLOOR}
    if not isinstance(calibrator, PlatScaling):
        raise UnsupportedWalkAttribution('Only fitted Platt scaling is recorded')
    weight, bias = calibrator.weight_and_bias.detach().cpu().reshape(2, -1)
    return {'kind': 'platt', 'fitted': True, 'weight_raw': float(weight[0]),
            'weight_effective': float(weight.clamp(min=PLATT_WEIGHT_FLOOR)[0]),
            'bias': float(bias[0]), 'weight_floor': PLATT_WEIGHT_FLOOR}


def emitted_probabilities(module, logits):
    """Exactly the calls of `BaseNodeLightningModule.on_predict_batch_end`."""
    with torch.no_grad():
        if module.calibrator.is_fitted:
            logits = module.calibrator(logits)
        return module.loss.to_probability(logits).cpu()


def sigmoid(value):
    """Float64 logistic function without overflow."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def binary_block(spec, module, logits_row):
    """The `output.binary` block for one seed row of raw head logits."""
    logits_row = logits_row.detach()
    z = float(logits_row.reshape(-1)[0])
    selected_logit = spec.logit_sign * z
    prob = emitted_probabilities(module, logits_row.reshape(1, -1))
    column = 1 if spec.logit_sign == 1 else 0
    return {
        'loss': spec.loss,
        'label_mapping': [dict(entry) for entry in spec.label_mapping],
        'target_encoder_reverse': spec.target_encoder_reverse,
        'encoded_positive_label': spec.encoded_positive_label,
        'selected_class_label': spec.selected_class_label,
        'selected_class_source': spec.selected_class_source,
        'logit_sign': spec.logit_sign,
        'selected_logit': selected_logit,
        'model_probability_selected': sigmoid(selected_logit),
        'calibration': calibration_record(module),
        'emitted_encoded_positive_probability': float(prob[0, 1]),
        'emitted_probability_selected': float(prob[0, column]),
        'emitted_column_selected': 'True_PROB' if column == 1 else 'False_PROB',
    }


def output_block(selection, module, logits_row):
    """`{task_type, output_index, head_output, binary}` for one seed row."""
    row = logits_row.detach().reshape(-1)
    head_output = float(row[selection.output_index])
    return {
        'task_type': selection.task_type,
        'output_index': selection.output_index,
        'head_output': head_output,
        'binary': (None if selection.binary is None else
                   binary_block(selection.binary, module, row[selection.output_index:][:1])),
    }


def selected_score(output):
    """`s * z` from an output block: the explained scalar of the row."""
    if output['binary'] is None:
        return output['head_output']
    return output['binary']['selected_logit']
