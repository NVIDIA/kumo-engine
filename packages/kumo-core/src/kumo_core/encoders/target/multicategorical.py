# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from torch_frame.data import MultiNestedTensor
from torch_geometric.utils.sparse import index2ptr

from kumo_core.encoders.tensor_mapper import MultiCategoricalTensorMapper
from kumo_core.encoders.target.base import TargetEncoder


class MultiCategoricalTargetEncoder(TargetEncoder[MultiNestedTensor]):
    """Encodes multi-categorical target tensors.

    Args:
        tensor_mapper: The multi-categorical tensor mapper to use for decoding.
    """
    def __init__(self, tensor_mapper: MultiCategoricalTensorMapper) -> None:
        super().__init__()
        self.tensor_mapper = tensor_mapper

    @property
    def num_categories(self) -> int:
        assert isinstance(self.tensor_mapper, MultiCategoricalTensorMapper)
        return len(self.tensor_mapper.mapping)

    def encode_tensor(self, tensor: MultiNestedTensor) -> Tensor:
        """Encodes a tuple of offset and values to a multi-hot tensor.

        Example:
            >>> import torch
            >>> from kumo_core.encoders.target import MultiCategoricalTargetEncoder
            >>>
            >>> target_encoder = MultiCategoricalTargetEncoder(tensor_mapper)
            >>> input = MultiNestedTensor(
            ...     num_rows=4,
            ...     num_cols=1,
            ...     values=torch.tensor([1, 0, 2, 0, 1]),
            ...     offset=torch.tensor([0, 1, 1, 3, 5]),
            ... )
            >>>
            >>> out = target_encoder.encode_tensor(input)
            >>> expected = torch.tensor([
            ...     [0., 1., 0.],  # [1]
            ...     [0., 0., 0.],  # []
            ...     [1., 0., 1.],  # [0, 2]
            ...     [1., 1., 0.],  # [0, 1]
            ... ])
            >>> assert torch.allclose(out, expected)
        """
        count = tensor.offset.diff()
        batch = torch.arange(count.numel(), device=count.device)
        batch = batch.repeat_interleave(
            count,
            output_size=tensor.values.numel(),
        )
        out = torch.zeros(
            (count.numel(), self.num_categories),
            device=count.device,
        )
        out[batch, tensor.values] = 1.
        return out

    def decode_tensor(self, tensor: Tensor) -> MultiNestedTensor:
        """Decodes multi-hot tensor to offset and values.

        Example:
            >>> import torch
            >>> from kumo_core.encoders.target import MultiCategoricalTargetEncoder

            >>> tensor = torch.tensor([
            ...     [0., 1., 0.],  # [1]
            ...     [0., 0., 0.],  # []
            ...     [1., 0., 1.],  # [0, 2]
            ...     [1., 1., 0.],  # [0, 1]
            ... ])
            >>> target_encoder = MultiCategoricalTargetEncoder(tensor_mapper)
            >>> offset, values = target_encoder.decode_tensor(tensor)
            >>> assert torch.allclose(offset, torch.tensor([0, 1, 1, 3, 5]))
            >>> assert torch.allclose(values, torch.tensor([1, 0, 2, 0, 1]))
        """
        batch, index = tensor.nonzero().t()
        rowptr = index2ptr(batch, size=tensor.size(0))
        return MultiNestedTensor(
            num_rows=rowptr.numel() - 1,
            num_cols=1,
            values=index,
            offset=rowptr,
        )
