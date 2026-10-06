# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import random

from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Image,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    NAStrategy,
    Numerical,
    NumericalList,
    Scaler,
    TextEmbedding,
)
from kumoapi.task import TaskType
from kumoapi.typing import Stype


def sample_numerical_encoder() -> Encoder:
    na_strategy = random.choice([NAStrategy.ZERO, NAStrategy.MEAN])
    return random.choice([
        Numerical(scaler=None, na_strategy=na_strategy),
        Numerical(scaler=Scaler.STANDARD, na_strategy=na_strategy),
        Numerical(scaler=Scaler.MINMAX, na_strategy=na_strategy),
        Numerical(scaler=Scaler.ROBUST, na_strategy=na_strategy),
        MinLogNumerical(na_strategy),
        MaxLogNumerical(na_strategy),
    ])


def sample_categorical_encoder() -> Encoder:
    min_occ = random.randint(1, 10)
    num_components = random.randint(1, 10)
    return random.choice([
        Index(min_occ, na_strategy=NAStrategy.ZERO),
        Index(min_occ, na_strategy=NAStrategy.SEPARATE),
        Index(min_occ, na_strategy=NAStrategy.MOST_FREQUENT),
        Hash(num_components, na_strategy=NAStrategy.SEPARATE),
        Hash(num_components, na_strategy=NAStrategy.MOST_FREQUENT),
    ])


def sample_multicategorical_encoder() -> Encoder:
    min_occ = random.randint(1, 10)
    na_strategy = random.choice([
        NAStrategy.ZERO,
        NAStrategy.SEPARATE,
        NAStrategy.MOST_FREQUENT,
    ])
    return MultiCategorical(min_occ, na_strategy)


def get_encoder(stype: Stype) -> Encoder:
    if stype == Stype.numerical:
        return sample_numerical_encoder()
    if stype in {Stype.categorical, Stype.ID}:
        return sample_categorical_encoder()
    if stype == Stype.multicategorical:
        return sample_multicategorical_encoder()
    if stype == Stype.timestamp:
        return Datetime()
    if stype == Stype.text:
        return TextEmbedding('glove.test-10d')
    if stype == Stype.sequence:
        return NumericalList()
    if stype == Stype.image:
        return Image()
    raise NotImplementedError


def get_target_encoder(task_type: TaskType) -> Encoder | None:
    if task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
    }:
        return Index()
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        return sample_numerical_encoder()
    if task_type in {
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
    }:
        return MultiCategorical(na_strategy=NAStrategy.ZERO)
    return None
