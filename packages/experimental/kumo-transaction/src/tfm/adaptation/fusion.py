# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint fusion of a sequence embedding with tabular features (nuFormer-style).

nuFormer's strongest adaptation trains a DCNv2 over periodic (PLR) embeddings
of existing tabular features together with the transformer's embedding, end to
end. That is the pattern fraud teams ask about first: keep the features they
already have, and let the foundation model add what those features miss.

- `PeriodicEmbedding` maps each numeric feature through learned sinusoids,
  then a per-feature linear layer and ReLU (PLR; Gorishniy et al., 2022).
- `JointFusionHead` concatenates the PLR features with the layer-normed
  embedding, applies DCNv2 cross layers, then an MLP to one logit.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from tfm.contracts.errors import IncompatibleCompositionError


class PeriodicEmbedding(nn.Module):
    """PLR: `ReLU(Linear([sin(2π c x), cos(2π c x)]))` per numeric feature.

    Features are standardized with statistics set by `fit_statistics` from the
    training rows only, so no evaluation value shapes the transform.
    """

    mean: Tensor
    scale: Tensor

    def __init__(
        self, n_features: int, *, frequencies: int = 8, dim: int = 8, sigma: float = 1.0
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise IncompatibleCompositionError("PeriodicEmbedding needs at least one feature.")
        self.n_features = n_features
        self.dim = dim
        self.coefficients = nn.Parameter(torch.randn(n_features, frequencies) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * frequencies, dim))
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))

    @property
    def out_features(self) -> int:
        return self.n_features * self.dim

    def fit_statistics(self, features: Tensor) -> None:
        """Set the standardization from training rows `[N, F]`."""
        self.mean.copy_(features.float().mean(dim=0))
        self.scale.copy_(features.float().std(dim=0, unbiased=False).clamp(min=1e-6))

    def forward(self, features: Tensor) -> Tensor:
        x = (features.float() - self.mean) / self.scale
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.coefficients
        periodic = torch.cat([angles.sin(), angles.cos()], dim=-1)
        projected = torch.einsum("bfp,fpd->bfd", periodic, self.weight) + self.bias
        return torch.relu(projected).flatten(start_dim=1)


class CrossLayer(nn.Module):
    """DCNv2 full-rank cross: `x_{l+1} = x_0 * (W x_l + b) + x_l`."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0: Tensor, xl: Tensor) -> Tensor:
        return x0 * self.linear(xl) + xl


class JointFusionHead(nn.Module):
    """One logit from a sequence embedding and raw tabular features, trained jointly."""

    def __init__(
        self,
        embedding_dim: int,
        n_features: int,
        *,
        cross_layers: int = 2,
        frequencies: int = 8,
        feature_dim: int = 8,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.numeric = PeriodicEmbedding(n_features, frequencies=frequencies, dim=feature_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        width = embedding_dim + self.numeric.out_features
        self.cross = nn.ModuleList([CrossLayer(width) for _ in range(cross_layers)])
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, embeddings: Tensor, features: Tensor) -> Tensor:
        x0 = torch.cat([self.embedding_norm(embeddings.float()), self.numeric(features)], dim=-1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.mlp(x).squeeze(-1)
