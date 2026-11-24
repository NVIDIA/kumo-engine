# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two- and three-dimensional projections of sequence embeddings, for inspection only.

A scatter plot of pooled embeddings is a qualitative sanity check, not evidence:
neighborhood structure in two dimensions does not establish that a
representation carries signal. Use the probes in `tfm.evaluation.fraud` for
that.

PCA is the default because it is deterministic, linear, and needs no extra.
UMAP matches the published NVIDIA Blueprint notebook 04 but is non-linear and
seeded, so it is an optional extra and its output must not be read as distance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tfm.contracts.errors import UnsupportedCombinationError

PROJECTIONS = ("pca", "umap")


def project(
    vectors: NDArray[np.floating],
    *,
    dims: int = 2,
    method: str = "pca",
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> NDArray[np.float32]:
    """Project `[N, D]` embeddings to `[N, dims]`, for a 2-D plot or a 3-D view.

    `dims` is 2 or 3. With fewer features than `dims`, the missing columns
    are zero. `n_neighbors` and `min_dist` apply to UMAP only. UMAP requires at
    least three rows and clamps `n_neighbors` to `N - 1`.
    """
    if dims not in (2, 3):
        raise UnsupportedCombinationError(f"A projection has 2 or 3 dims, got dims={dims}.")
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise UnsupportedCombinationError(
            f"project expects a [N, D] array, got shape {array.shape}."
        )
    name = method.strip().lower()
    if array.shape[0] == 0:
        return np.zeros((0, dims), dtype=np.float32)
    if name == "pca":
        return _pca(array, dims=dims, seed=seed)
    if name == "umap":
        return _umap(array, dims=dims, seed=seed, n_neighbors=n_neighbors, min_dist=min_dist)
    raise UnsupportedCombinationError(f"Unknown projection {method!r}; use one of {PROJECTIONS}.")


def project_2d(
    vectors: NDArray[np.floating],
    *,
    method: str = "pca",
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> NDArray[np.float32]:
    """Project `[N, D]` embeddings to `[N, 2]` (`project` with `dims=2`)."""
    return project(
        vectors, dims=2, method=method, seed=seed, n_neighbors=n_neighbors, min_dist=min_dist
    )


def _pca(array: NDArray[np.float32], *, dims: int, seed: int) -> NDArray[np.float32]:
    from sklearn.decomposition import PCA

    components = min(dims, array.shape[0], array.shape[1])
    projected = PCA(n_components=components, random_state=seed).fit_transform(array)
    return _pad(np.asarray(projected, dtype=np.float32), dims)


def _umap(
    array: NDArray[np.float32], *, dims: int, seed: int, n_neighbors: int, min_dist: float
) -> NDArray[np.float32]:
    from tfm.extras.capability import require_extra

    require_extra("umap", "umap")
    from umap.umap_ import UMAP  # type: ignore[import-not-found]

    if array.shape[0] < 3:
        raise UnsupportedCombinationError(
            f"UMAP needs at least 3 rows to build a neighbor graph, got {array.shape[0]}. "
            "Use method='pca' on tiny smoke batches."
        )
    reducer: Any = UMAP(
        n_components=dims,
        n_neighbors=min(n_neighbors, array.shape[0] - 1),
        min_dist=min_dist,
        random_state=seed,
    )
    return _pad(np.asarray(reducer.fit_transform(array), dtype=np.float32), dims)


def _pad(projected: NDArray[np.float32], dims: int) -> NDArray[np.float32]:
    """Widen a projection with fewer components than `dims` so callers always get `[N, dims]`."""
    if projected.shape[1] >= dims:
        return projected[:, :dims]
    pad = np.zeros((projected.shape[0], dims - projected.shape[1]), dtype=np.float32)
    return np.concatenate([projected, pad], axis=1)
