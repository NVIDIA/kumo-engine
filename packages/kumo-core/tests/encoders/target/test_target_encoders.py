# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import random
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
import torch
from kumoapi.encoder import (
    Encoder,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    NAStrategy,
    Numerical,
)
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType
from torch_frame.data import MultiNestedTensor
from torch_geometric.testing import withCUDA

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.target.multicategorical import MultiCategoricalTargetEncoder
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.encoders.target.base import TargetEncoder


@withCUDA
def test_numerical_encode_decode(device: torch.device) -> None:
    offset = random.random()
    scale = random.uniform(1.0, 2.0)
    m = NumericalTargetEncoder(offset, scale).to(device)
    assert m.offset == offset
    assert m.scale == scale

    x = torch.randn(7, 1, device=device)
    out = m.encode_tensor(x)
    assert torch.allclose(out, (x - offset) / scale)
    assert torch.allclose(m.decode_tensor(out), x, rtol=1e-4, atol=1e-4)

    # Gracefully handle log(intermediate) where intermediate <= 0:
    offset = 10.0
    scale = 1.0
    x = torch.tensor([1.0, 2.0, 3.0], device=device).abs()
    m = NumericalTargetEncoder(offset, scale, log_transform=True)
    out = m.encode_tensor(x)
    assert not torch.isnan(out).any()
    assert not torch.isnan(m.decode_tensor(out)).any()


@withCUDA
def test_binary_encode_decode(device: torch.device) -> None:
    tensor_mapper = Mock(spec=CategoricalTensorMapper)
    tensor_mapper.mapping = pd.Index(data=[True, False], name='Name')
    m = CategoricalTargetEncoder(TaskType.BINARY_CLASSIFICATION,
                                 tensor_mapper).to(device)
    x = torch.randint(0, 2, (7, 1), device=device)
    out = m.encode_tensor(x)
    assert out.equal(1 - x)
    assert torch.allclose(m.decode_tensor(m.encode_tensor(x)), x)


@withCUDA
def test_categorical_encode_decode(device: torch.device) -> None:
    tensor_mapper = Mock(spec=CategoricalTensorMapper)
    tensor_mapper.mapping = pd.Index(data=['A', 'B', 'C'], name='Name')
    m = CategoricalTargetEncoder(TaskType.MULTICLASS_CLASSIFICATION,
                                 tensor_mapper).to(device)
    x = torch.randint(0, 3, (7, 1), device=device)
    assert torch.allclose(m.decode_tensor(m.encode_tensor(x)), x)


@withCUDA
def test_multicategorical_encode_decode(device: torch.device) -> None:
    tensor_mapper = Mock(spec=MultiCategoricalTensorMapper)
    tensor_mapper.mapping = pd.Index(data=['A', 'B', 'C'], name='Name')
    m = MultiCategoricalTargetEncoder(tensor_mapper=tensor_mapper).to(device)
    assert m.num_categories == 3
    x = MultiNestedTensor.from_tensor_mat([
        [torch.tensor([1])],
        [torch.tensor([], dtype=torch.long)],
        [torch.tensor([0, 2])],
        [torch.tensor([0, 1])],
    ]).to(device)
    out_encoded = m.encode_tensor(x)
    assert torch.allclose(
        out_encoded,
        torch.tensor(
            [
                [0., 1., 0.],  # [1]
                [0., 0., 0.],  # []
                [1., 0., 1.],  # [0, 2]
                [1., 1., 0.],  # [0, 1]
            ],
            device=device),
    )
    out_decoded = m.decode_tensor(out_encoded)
    assert torch.allclose(out_decoded.offset, x.offset)
    assert torch.allclose(out_decoded.values, x.values)


@pytest.mark.parametrize('na_strategy', [e for e in NAStrategy])
@pytest.mark.parametrize(
    'get_cfg, task_type, stats, expected_attrs',
    [
        pytest.param(
            lambda: Numerical(scaler=None),
            TaskType.REGRESSION,
            {},
            dict(offset=0.0, scale=1.0),
            id='numerical',
        ),
        pytest.param(
            lambda: Numerical(scaler='robust'),
            TaskType.REGRESSION,
            {ColStatType.QUANTILES: [-0.123, 0.0, 0.123]},
            dict(offset=0.0, scale=0.246),
            id='numerical-robust',
        ),
        pytest.param(
            lambda: Numerical(scaler='standard'),
            TaskType.REGRESSION,
            {
                ColStatType.MEAN: 0.0,
                ColStatType.STD: 1.0
            },
            dict(offset=0.0, scale=1.0),
            id='numerical-standard',
        ),
        pytest.param(
            lambda: MaxLogNumerical(),
            TaskType.REGRESSION,
            {ColStatType.MIN: -10.0},
            dict(offset=-11.0, scale=1.0),
            id='maxlog',
        ),
        pytest.param(
            lambda: MinLogNumerical(),
            TaskType.REGRESSION,
            {ColStatType.MAX: 10.0},
            dict(offset=11.0, scale=-1.0),
            id='minlog',
        ),
        pytest.param(
            lambda: Index(),
            TaskType.BINARY_CLASSIFICATION,
            {ColStatType.CATEGORY_COUNTS: [('A', 10), ('B', 3)]},
            dict(num_categories=2),
            id='categorical',
        ),
        pytest.param(
            lambda: Index(),
            TaskType.BINARY_CLASSIFICATION,
            {ColStatType.CATEGORY_COUNTS: [('B', 3), ('A', 10)]},
            dict(num_categories=2),
            id='categorical-unordered',
        ),
        pytest.param(
            lambda: MultiCategorical(),
            TaskType.MULTILABEL_CLASSIFICATION,
            {
                ColStatType.MULTI_CATEGORY_COUNTS: [('A', 10), ('B', 3),
                                                    ('C', 1)],
                ColStatType.MULTI_CATEGORIES_SEPARATOR: '|',
            },
            dict(num_categories=3),
            id='multicategorical',
        ),
        pytest.param(
            lambda: MultiCategorical(),
            TaskType.MULTILABEL_CLASSIFICATION,
            {
                ColStatType.MULTI_CATEGORY_COUNTS: [('C', 1), ('A', 10),
                                                    ('B', 3)],
                ColStatType.MULTI_CATEGORIES_SEPARATOR: '|',
            },
            dict(num_categories=3),
            id='multicategorical-unordered',
        ),
    ],
)
def test_from_encoder(
    get_cfg: Callable[[], Encoder],
    task_type: TaskType,
    stats: dict[ColStatType, Any],
    expected_attrs: dict[str, Any],
    na_strategy: NAStrategy,
) -> None:
    cfg = get_cfg()
    cfg._stats = stats
    cfg.na_strategy = na_strategy
    target_encoder = TargetEncoder.from_encoder(task_type, cfg)
    for key, value in expected_attrs.items():
        assert getattr(target_encoder, key) == value

    if isinstance(cfg, Index):
        assert isinstance(target_encoder.tensor_mapper,
                          CategoricalTensorMapper)
        assert target_encoder.tensor_mapper.mapping.to_list() == ['A', 'B']
    elif isinstance(cfg, MultiCategorical):
        assert isinstance(target_encoder.tensor_mapper,
                          MultiCategoricalTensorMapper)
        assert target_encoder.tensor_mapper.mapping.to_list() == [
            'A', 'B', 'C'
        ]


def test_from_encoder_unsupported() -> None:
    cfg = MagicMock(spec=Encoder)
    with pytest.raises(ValueError, match='Unsupported encoder provided'):
        TargetEncoder.from_encoder(TaskType.REGRESSION, cfg)
