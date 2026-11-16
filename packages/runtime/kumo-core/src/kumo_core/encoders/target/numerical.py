# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from torch import Tensor

from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumo_core.encoders.target.base import TargetEncoder


class NumericalTargetEncoder(TargetEncoder[Tensor]):
    """Target encoder that scales and optionally log-transforms numerical
    values.

    Args:
        offset: The offset to subtract from the input tensor before scaling.
        scale: The scaling factor to divide the input tensor by.
        log_transform: Whether to apply a log transformation to the input
            tensor after scaling.
    """
    def __init__(
        self,
        offset: float,
        scale: float,
        log_transform: bool = False,
    ) -> None:
        super().__init__()
        self.offset = offset
        self.scale = scale
        self.log_transform = log_transform
        self.tensor_mapper = NumericalTensorMapper()

    def encode_tensor(self, tensor: Tensor) -> Tensor:
        """Applies scaling and optionally log transformation to the input.

        Example:
            import torch
            from kumo_core.encoders.target import NumericalTargetEncoder

            target_encoder = NumericalTargetEncoder(offset=1, scale=2)
            x = torch.tensor([0., 1., 2., 3., 4.])

            out = target_encoder.encode_tensor(x)
            expected = torch.tensor([-0.5, 0., 0.5, 1., 1.5])
            assert torch.allclose(out, expected)
        """
        out = (tensor - self.offset) / self.scale
        if self.log_transform:
            out = out.clamp(min=1).log()
        return out

    def decode_tensor(self, tensor: Tensor) -> Tensor:
        """Decodes the input tensor by reversing the scaling and log
        transformation.

        Example:
            import torch
            from kumo_core.encoders.target import NumericalTargetEncoder

            target_encoder = NumericalTargetEncoder(offset=1, scale=2)
            x = torch.tensor([0, 1, 2, 3, 4])

            out = target_encoder.encode_tensor(x)
            decoded = target_encoder.decode_tensor(out)
            assert torch.allclose(decoded, x)
        """
        if self.log_transform:
            tensor = tensor.exp()
        return tensor * self.scale + self.offset
