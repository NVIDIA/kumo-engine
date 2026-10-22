# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from kumo_core.modeling.retrieval.mips import MIPSBruteForce, MIPSFaiss
from torch_geometric.testing import noMac, withCUDA


def require_backend(device: torch.device) -> None:
    faiss = pytest.importorskip('faiss')
    if device.type == 'cuda' and not hasattr(faiss, 'StandardGpuResources'):
        pytest.skip('CUDA retrieval requires GPU FAISS')


@noMac
@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('num_lhs', [10, 20])
@pytest.mark.parametrize('num_rhs', [100, 1_000, 10_000])
@pytest.mark.parametrize('dim', [16])
@pytest.mark.parametrize('k', [2])
@pytest.mark.parametrize('with_exclude', [False, True])
@pytest.mark.parametrize('requires_grad', [False, True])
def test_mips_faiss(
    dtype: torch.dtype,
    device: torch.device,
    num_lhs: int,
    num_rhs: int,
    dim: int,
    k: int,
    with_exclude: bool,
    requires_grad: bool,
) -> None:
    require_backend(device)
    lhs = torch.randn(
        (num_lhs, dim),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    rhs = torch.randn(num_rhs, dim, dtype=dtype, device=device)

    mat = lhs @ rhs.t()

    mips_structure = MIPSFaiss(rhs)
    assert str(mips_structure) == f'MIPSFaiss({num_rhs}, {dim})'
    assert mips_structure.num_keys == num_rhs
    assert mips_structure.embedding_dim == dim
    assert mips_structure.dtype == dtype
    assert mips_structure.device == device

    exclude = None
    if with_exclude:  # Filter out the `top-k` values:
        threshold = mat.view(-1).topk(1000)[0].min()
        mask = mat >= threshold
        nnz = mask.nonzero(as_tuple=True)
        exclude = (nnz[0], nnz[1])

    score, index = mips_structure.search(lhs, k, exclude=exclude)
    assert score.device == device and index.device == device
    assert score.requires_grad == requires_grad

    if exclude:
        mat[exclude[0], exclude[1]] = -float('inf')
    expected = mat.argsort(dim=-1, descending=True)[:, :k]

    # Only require non-inf prediction to match:
    notinf = ~torch.isinf(score)
    assert torch.allclose(index[notinf], expected[notinf])


@withCUDA
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('num_lhs', [10, 20])
@pytest.mark.parametrize('num_rhs', [100, 1_000, 10_000])
@pytest.mark.parametrize('dim', [16])
@pytest.mark.parametrize('k', [2])
@pytest.mark.parametrize('with_exclude', [False, True])
@pytest.mark.parametrize('requires_grad', [False, True])
def test_mips_brute_force(
    device: torch.device,
    dtype: torch.dtype,
    num_lhs: int,
    num_rhs: int,
    dim: int,
    k: int,
    with_exclude: bool,
    requires_grad: bool,
) -> None:
    lhs = torch.randn(
        (num_lhs, dim),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    rhs = torch.randn(num_rhs, dim, dtype=dtype, device=device)

    mat = lhs @ rhs.t()

    mips_structure = MIPSBruteForce(rhs)
    assert str(mips_structure) == f'MIPSBruteForce({num_rhs}, {dim})'
    assert mips_structure.num_keys == num_rhs
    assert mips_structure.embedding_dim == dim
    assert mips_structure.dtype == dtype
    assert mips_structure.device == device

    exclude = None
    if with_exclude:  # Filter out the `top-k` values:
        threshold = mat.view(-1).topk(1000)[0].min()
        mask = mat >= threshold
        nnz = mask.nonzero(as_tuple=True)
        exclude = (nnz[0], nnz[1])

    score, index = mips_structure.search(lhs, k, exclude=exclude)
    assert score.device == device and index.device == device
    assert score.requires_grad == requires_grad

    if exclude:
        mat[exclude[0], exclude[1]] = -float('inf')
    expected = mat.argsort(dim=-1, descending=True)[:, :k]

    # Only require non-inf prediction to match:
    notinf = ~torch.isinf(score)
    assert torch.allclose(index[notinf], expected[notinf])


@noMac
@withCUDA
@pytest.mark.parametrize('cls', [MIPSBruteForce, MIPSFaiss])
def test_empty(
    cls: type[MIPSBruteForce | MIPSFaiss],
    device: torch.device,
) -> None:
    if cls is MIPSFaiss:
        require_backend(device)
    lhs = torch.randn(4, 8, device=device)
    rhs = torch.randn(0, 8, device=device)
    mips_structure = cls(rhs)
    assert mips_structure.search(lhs, k=2)[0].size() == (4, 0)
    assert mips_structure.search(lhs, k=2)[1].size() == (4, 0)

    lhs = torch.randn(0, 8, device=device)
    rhs = torch.randn(4, 8, device=device)
    mips_structure = cls(rhs)
    assert mips_structure.search(lhs, k=2)[0].size() == (0, 2)
    assert mips_structure.search(lhs, k=2)[1].size() == (0, 2)
