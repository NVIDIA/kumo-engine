# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kumoapi.task import TaskType
from torch import Tensor

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper
from kumo_core.encoders.target.base import TargetEncoder


class CategoricalTargetEncoder(TargetEncoder[Tensor]):
    """Target encoder for categorical columns. Returns the tensor as it is.

    Args:
        task_type: The task type of the target column.
        tensor_mapper: The categorical tensor mapper to use for decoding.
    """
    def __init__(
        self,
        task_type: TaskType,
        tensor_mapper: CategoricalTensorMapper,
    ) -> None:
        super().__init__()
        self.tensor_mapper = tensor_mapper
        self.task_type = task_type

        # For binary classification, we need to make sure that the `mapping` is
        # sorted by values rather than category count, such that `False` is
        # mapped to index `0`, and `True` is mapped to index `1`:
        self.reverse = False
        if task_type == TaskType.BINARY_CLASSIFICATION:
            num_classes = len(tensor_mapper.mapping)
            if num_classes > 2:
                raise ValueError(f"Binary classification requires two "
                                 f"distinct labels (got {num_classes})")
            self.reverse = not tensor_mapper.mapping.is_monotonic_increasing

    @property
    def num_categories(self) -> int:
        assert isinstance(self.tensor_mapper, CategoricalTensorMapper)
        return len(self.tensor_mapper.mapping)

    def encode_tensor(self, tensor: Tensor) -> Tensor:
        return tensor if not self.reverse else 1 - tensor

    def decode_tensor(self, tensor: Tensor) -> Tensor:
        return tensor if not self.reverse else 1 - tensor
