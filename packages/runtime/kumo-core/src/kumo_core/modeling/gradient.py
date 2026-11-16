# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

from torch import Tensor


class GradientMixin(ABC):
    @property
    def saved_tensors(self) -> tuple[Tensor | None, ...]:
        r"""Intermediate data to compute gradients with
        respect to inputs.
        """
        return self._saved_tensors

    def save_for_backward(self, *tensors: Tensor | None) -> None:
        r"""Saves any data required for computing gradients with respect to
        the inputs.
        """
        for tensor in tensors:
            if tensor is not None and tensor.is_floating_point():
                if not tensor.requires_grad:
                    tensor.requires_grad = True
                tensor.retain_grad()

        self._saved_tensors = tensors

    def get_grad_wrt_input(
        self,
        retain_grad: bool = False,
        raise_error: bool = True,
    ) -> Tensor | None:
        r"""Returns gradients with respect to the inputs of the forward pass.

        Args:
            retain_grad: If set to :obj:`True`, will retain the gradient of the
                output of the module. This should only ever be used whenever
                gradients w.r.t. the inputs need to be requested twice.
            raise_error: If set to :obj:`False`, will not raise an error if
                gradients are not available.
        """
        if getattr(self, '_saved_tensors', None) is None:
            if not raise_error:
                return None
            raise ValueError(f"Did not store required data for gradient "
                             f"computation in '{self.__class__.__name__}'. "
                             f"You either (1) forgot to call the module, "
                             f"(2) did not wrap its call with "
                             f"`with kumo_core.modeling.context.explaining()`, or (3) "
                             f"requested input gradients twice.")

        for tensor in self.saved_tensors:
            if (tensor is not None and tensor.is_floating_point()
                    and tensor.grad is None):
                if not raise_error:
                    return None
                raise ValueError(f"Gradients not yet computed for "
                                 f"'{self.__class__.__name__}'. Did you "
                                 f"forget to call `backward()`?")

        out = self._get_grad_wrt_input()

        if not retain_grad:
            del self._saved_tensors

        return out

    @abstractmethod
    def _get_grad_wrt_input(self) -> Tensor:
        raise NotImplementedError
