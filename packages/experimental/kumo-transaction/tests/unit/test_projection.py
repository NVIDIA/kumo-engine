# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two- and three-dimensional embedding projections used by the teaching notebooks."""

from __future__ import annotations

import numpy as np
import pytest

from tfm.contracts.errors import ExtraUnavailableError, UnsupportedCombinationError
from tfm.evaluation.representation import PROJECTIONS, project, project_2d
from tfm.extras.capability import extra_available


def _vectors(rows: int = 12, dim: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(rows, dim)).astype(np.float32)


def test_pca_is_the_default_and_is_deterministic() -> None:
    vectors = _vectors()

    first = project_2d(vectors)
    second = project_2d(vectors, method="pca", seed=0)

    assert first.shape == (12, 2)
    assert first.dtype == np.float32
    np.testing.assert_allclose(first, second)


def test_a_single_feature_still_yields_two_columns() -> None:
    projected = project_2d(_vectors(dim=1))

    assert projected.shape == (12, 2)
    assert np.all(projected[:, 1] == 0.0)


def test_empty_input_projects_to_an_empty_plane() -> None:
    assert project_2d(np.zeros((0, 5), dtype=np.float32)).shape == (0, 2)


def test_non_matrix_input_is_rejected() -> None:
    with pytest.raises(UnsupportedCombinationError, match=r"\[N, D\]"):
        project_2d(np.zeros((4,), dtype=np.float32))


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(UnsupportedCombinationError, match="tsne"):
        project_2d(_vectors(), method="tsne")


def test_projection_names_are_advertised() -> None:
    assert PROJECTIONS == ("pca", "umap")


def test_umap_requires_the_extra() -> None:
    if extra_available("umap"):
        projected = project_2d(_vectors(), method="umap", seed=0)
        assert projected.shape == (12, 2)
        return
    with pytest.raises(ExtraUnavailableError, match="umap"):
        project_2d(_vectors(), method="umap")


def test_project_reaches_three_dims_and_project_2d_is_its_first_case() -> None:
    vectors = _vectors()

    projected = project(vectors, dims=3)

    assert projected.shape == (12, 3)
    assert projected.dtype == np.float32
    np.testing.assert_allclose(project(vectors, dims=2), project_2d(vectors))


def test_three_dims_from_two_features_pad_the_last_column() -> None:
    projected = project(_vectors(dim=2), dims=3)

    assert projected.shape == (12, 3)
    assert np.all(projected[:, 2] == 0.0)


def test_empty_input_projects_to_an_empty_volume() -> None:
    assert project(np.zeros((0, 5), dtype=np.float32), dims=3).shape == (0, 3)


@pytest.mark.parametrize("dims", [1, 4])
def test_dims_other_than_two_or_three_are_rejected(dims: int) -> None:
    with pytest.raises(UnsupportedCombinationError, match="2 or 3 dims"):
        project(_vectors(), dims=dims)


@pytest.mark.skipif(not extra_available("umap"), reason="umap-learn not installed")
def test_umap_projects_to_three_dims() -> None:
    assert project(_vectors(), dims=3, method="umap", seed=0).shape == (12, 3)


@pytest.mark.skipif(not extra_available("umap"), reason="umap-learn not installed")
def test_umap_refuses_a_batch_too_small_for_a_neighbor_graph() -> None:
    with pytest.raises(UnsupportedCombinationError, match="at least 3 rows"):
        project_2d(_vectors(rows=2), method="umap")


def test_three_dimensions_for_an_interactive_view() -> None:
    vectors = _vectors()
    projected = project(vectors, dims=3)
    assert projected.shape == (12, 3)
    assert projected.dtype == np.float32
    # PCA's first two components do not depend on how many are kept.
    np.testing.assert_allclose(projected[:, :2], project_2d(vectors), rtol=1e-5, atol=1e-5)


def test_fewer_features_than_dimensions_are_padded() -> None:
    projected = project(_vectors(dim=2), dims=3)
    assert projected.shape == (12, 3)
    assert np.all(projected[:, 2] == 0.0)
    assert project(np.zeros((0, 5), dtype=np.float32), dims=3).shape == (0, 3)


def test_only_two_or_three_dimensions() -> None:
    with pytest.raises(UnsupportedCombinationError, match="dims"):
        project(_vectors(), dims=4)


def test_umap_in_three_dimensions_requires_the_extra() -> None:
    if extra_available("umap"):
        assert project(_vectors(), dims=3, method="umap", seed=0).shape == (12, 3)
        return
    with pytest.raises(ExtraUnavailableError, match="umap"):
        project(_vectors(), dims=3, method="umap")
