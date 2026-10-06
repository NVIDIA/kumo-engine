# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import group_argsort

import kumo_core.modeling.context as modeling_context
from kumo_core.modeling.gradient import GradientMixin
from kumo_core.features.time import get_rel_time


class RelTimeEncoder(torch.nn.Module, GradientMixin):
    r"""Relative time encoder to encode not only time column's relative
    time but also relative time for other timestamp columns, having the
    :obj:`DateTime` encoder.

    Args:
        num_cols: Number of timestamp columns. The time column should be also
            counted into this.
        out_channels: Output channels.
        enc_mode: Encode mode.
        use_seq_id: Whether also encode the sequence order of timestamps.
        device: The device to use.
    """
    def __init__(
        self,
        num_cols: int,
        out_channels: int,
        enc_mode: Literal["fuse", "column"],
        use_seq_id: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_cols = num_cols

        self.out_channels = out_channels
        self.enc_mode = enc_mode
        if enc_mode not in {'fuse', 'column'}:
            raise ValueError(f"Unsupported encoder mode '{enc_mode}'.")

        # Parameters that handle relative time
        self.time_lin = Linear(
            self.num_cols * out_channels,
            out_channels,
            device=device,
        )
        self.time_enc = PositionalEncoding(out_channels, device=device)
        self.use_seq_id = use_seq_id

        if self.use_seq_id:
            self.seq_enc = PositionalEncoding(out_channels, device=device)
            self.seq_lin = Linear(
                self.num_cols * out_channels,
                out_channels,
                device=device,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.time_lin.reset_parameters()
        if self.use_seq_id:
            self.seq_lin.reset_parameters()

    @staticmethod
    def get_seq_id(time: Tensor, batch: Tensor) -> Tensor:
        r"""Computes the sequence order of timestamps belonging to the same
        seed node/example.
        """
        # TODO This implementation is far from perfect, since it will assign a
        # sequence ID to all sampled facts within a single disjoint subgraph,
        # which may lead to misleading ordering for subgraphs with more than
        # two hops, e.g., we will assign ID=0 to the last transaction of a
        # store, but we are more interested in assigning ID=0 to the last
        # transaction of the entity we care about. Since we currently limit
        # subgraphs to two hops by default, this is not a problem, but will
        # become problematic once we move to deeper subgraphs.
        assert time.dim() == 2
        outs = []
        for i in range(time.size(1)):
            outs.append(group_argsort(time[:, i], batch, descending=True))
        out = torch.stack(outs, dim=1).to(torch.get_default_dtype())
        out[time == -1] = float('NaN')
        return out

    def forward(
        self,
        time: Tensor,
        seed_time: Tensor,
        batch: Tensor,
    ) -> Tensor:
        num_rows, num_cols = time.size()

        rel_time = get_rel_time(time, seed_time[batch])

        seq_id: Tensor | None = None
        if self.use_seq_id:
            seq_id = self.get_seq_id(time, batch)

        if modeling_context.is_explaining():
            self.save_for_backward(rel_time, seq_id)

        time_enc = self.time_enc(rel_time)
        time_enc = time_enc.nan_to_num()
        time_enc = time_enc.view(num_rows, self.out_channels * self.num_cols)

        if self.enc_mode == "fuse":
            out = self.time_lin(time_enc)
        else:
            out = torch.einsum(
                'ijk,jkl->ijl',
                time_enc.view(num_rows, num_cols, self.out_channels),
                self.time_lin.weight.view(
                    self.num_cols,
                    self.out_channels,
                    self.out_channels,
                ),
            ) + self.time_lin.bias

        if seq_id is not None:
            seq_enc = self.seq_enc(seq_id)
            seq_enc = seq_enc.nan_to_num()
            seq_enc = seq_enc.view(num_rows, self.out_channels * self.num_cols)

            if self.enc_mode == "fuse":
                out = out + self.seq_lin(seq_enc)
            else:
                out = out + torch.einsum(
                    'ijk,jkl->ijl',
                    seq_enc.view(num_rows, num_cols, self.out_channels),
                    self.seq_lin.weight.view(
                        self.num_cols,
                        self.out_channels,
                        self.out_channels,
                    ),
                ) + self.seq_lin.bias

        return out

    def _get_grad_wrt_input(self) -> Tensor:
        rel_time, seq_id = self.saved_tensors

        assert rel_time is not None and rel_time.grad is not None
        # Ignore gradient from relative time in case it does not hold
        # information (e.g., in static queries on entities with timestamps):
        if (rel_time == 0).all():
            grad = torch.zeros_like(rel_time)
        else:
            grad = rel_time.grad.abs().nan_to_num()
        if seq_id is not None:
            assert seq_id.grad is not None
            grad = grad + seq_id.grad.abs().nan_to_num()

        return grad
