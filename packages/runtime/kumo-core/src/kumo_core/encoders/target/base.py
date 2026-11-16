# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd
from kumoapi.encoder import (
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Numerical,
)
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType
from torch import Tensor
from torch.nn import Module
from torch_frame.data import MultiNestedTensor

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, TensorMapper
from kumo_core.encoders.scaling import get_offset_and_scale

T = TypeVar('T', Tensor, MultiNestedTensor)


class TargetEncoder(Module, Generic[T], ABC):
    """Target encoder that instantiates the correct target encoder based on
    the encoder config and attaches the tensor mapper.
    """
    def __init__(self) -> None:
        super().__init__()
        self.tensor_mapper: TensorMapper

    @abstractmethod
    def encode_tensor(self, tensor: T) -> Tensor:
        pass

    @abstractmethod
    def decode_tensor(self, tensor: Tensor) -> T:
        pass

    def decode(self, tensor: Tensor) -> pd.Series:
        return self.tensor_mapper.backward(self.decode_tensor(tensor))

    @classmethod
    def from_encoder(
        cls,
        task_type: TaskType,
        encoder: Index | MultiCategorical | Numerical | MaxLogNumerical
        | MinLogNumerical,
    ) -> 'TargetEncoder':
        from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
        from kumo_core.encoders.target.multicategorical import MultiCategoricalTargetEncoder
        from kumo_core.encoders.target.numerical import NumericalTargetEncoder

        if isinstance(encoder, Index):
            category_counts = encoder._stats[ColStatType.CATEGORY_COUNTS]
            return CategoricalTargetEncoder(
                task_type,
                CategoricalTensorMapper(category_counts=category_counts),
            )

        elif isinstance(encoder, MultiCategorical):
            category_counts = encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS]
            return MultiCategoricalTargetEncoder(
                MultiCategoricalTensorMapper(
                    category_counts=category_counts,
                    sep=encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR],
                ))

        elif isinstance(encoder, Numerical):
            return NumericalTargetEncoder(*get_offset_and_scale(encoder))

        elif isinstance(encoder, MaxLogNumerical):
            return NumericalTargetEncoder(
                *get_offset_and_scale(encoder),
                log_transform=True,
            )

        elif isinstance(encoder, MinLogNumerical):
            return NumericalTargetEncoder(
                *get_offset_and_scale(encoder),
                log_transform=True,
            )

        raise ValueError(f"Unsupported encoder provided to 'TargetEncoder'. "
                         f"Got '{encoder}'.")
