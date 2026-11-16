# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from kumo_core.modeling.retrieval.mips import (
    MAX_K,
    FaissUnavailableError,
    MIPSBruteForce,
    MIPSFaiss,
    require_faiss,
)


def require_backend(device: torch.device) -> None:
    try:
        faiss = require_faiss()
    except FaissUnavailableError as error:
        pytest.skip(str(error))
    if device.type == 'cuda' and not hasattr(faiss, 'StandardGpuResources'):
        pytest.skip('CUDA retrieval requires GPU FAISS')


@pytest.mark.parametrize('error', [
    ModuleNotFoundError('No FAISS', name='faiss'),
    ModuleNotFoundError('Missing numpy', name='numpy'),
    ImportError('Broken FAISS shared library'),
])
def test_backend_import_errors(monkeypatch: pytest.MonkeyPatch,
                               error: ImportError) -> None:
    def broken_import(name: str) -> None:
        raise error

    monkeypatch.setattr(
        'kumo_core.modeling.retrieval.mips.importlib.import_module',
        broken_import)
    expected = (pytest.skip.Exception if isinstance(error, ModuleNotFoundError)
                and error.name == 'faiss' else type(error))
    with pytest.raises(expected):
        require_backend(torch.device('cpu'))


@pytest.fixture(params=[
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA unavailable')),
    pytest.param('mps', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='MPS unavailable')),
])
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param if request.param == 'cpu'
                        else f'{request.param}:0')


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

    # Compare scores so equivalent near-tied neighbors need not share IDs.
    notinf = ~torch.isinf(score)
    torch.testing.assert_close(mat.gather(1, index)[notinf],
                               mat.gather(1, expected)[notinf],
                               rtol=1e-5, atol=1e-5)


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


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason='MPS unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16,
                                   torch.bfloat16])
@pytest.mark.parametrize('with_exclude', [False, True])
def test_mips_metal_scores_and_gradients(
    dtype: torch.dtype,
    with_exclude: bool,
) -> None:
    # Transposing twice with a copy gives non-contiguous input tensors.
    rhs = torch.tensor([
        [1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4],
        [-1, -1, -1, -1],
    ], device='mps', dtype=dtype).t().contiguous().t()
    lhs = torch.tensor([[1, 2, 3, 4], [7, 5, 2, 1]],
                       device='mps', dtype=dtype).t().contiguous().t()
    lhs.requires_grad_()
    expected_lhs = lhs.detach().clone().requires_grad_()
    exclude = None
    if with_exclude:
        exclude = (torch.tensor([0, 1], device='mps'),
                   torch.tensor([3, 1], device='mps'))
    mips = MIPSFaiss(rhs)
    assert mips._metal_index is not None
    # Reuse the index across searches, including inference and autograd.
    inference_score, inference_index = mips.search(lhs.detach(), 2, exclude)
    score, index = mips.search(lhs, 2, exclude)
    expected_score, expected_index = MIPSBruteForce(rhs).search(
        expected_lhs, 2, exclude)
    assert score.device == index.device == lhs.device
    assert score.dtype == dtype
    assert index.dtype == torch.long
    torch.testing.assert_close(score, expected_score)
    torch.testing.assert_close(index, expected_index)
    torch.testing.assert_close(inference_score, score.detach())
    torch.testing.assert_close(inference_index, index)
    score.sum().backward()
    expected_score.sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad)


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason='MPS unavailable')
@pytest.mark.parametrize('compile_options, num_gpus', [
    ('OPTIMIZE', 1), ('OPTIMIZE MAC_METAL', 0),
])
def test_mips_mps_requires_metal_faiss(
    monkeypatch: pytest.MonkeyPatch,
    compile_options: str,
    num_gpus: int,
) -> None:
    import faiss

    monkeypatch.setattr(faiss, 'get_compile_options', lambda: compile_options)
    monkeypatch.setattr(faiss, 'get_num_gpus', lambda: num_gpus)
    with pytest.raises(RuntimeError, match='Metal-enabled Faiss'):
        MIPSFaiss(torch.ones(2, 4, device='mps'))


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason='MPS unavailable')
@pytest.mark.parametrize('k', [1, MAX_K])
def test_mips_metal_large_topk(k: int) -> None:
    rhs = torch.arange(MAX_K + 3, device='mps', dtype=torch.float32).view(-1, 1)
    lhs = torch.tensor([[1.0], [2.0]], device='mps')
    score, index = MIPSFaiss(rhs).search(lhs, k)
    expected_index = torch.arange(rhs.size(0) - 1, rhs.size(0) - k - 1, -1,
                                  device='mps').expand(2, -1)
    torch.testing.assert_close(index, expected_index)
    torch.testing.assert_close(score, expected_index * lhs)


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason='MPS unavailable')
def test_mips_metal_all_keys_excluded() -> None:
    rhs = torch.eye(4, device='mps')
    lhs = torch.ones(1, 4, device='mps', requires_grad=True)
    exclude = (torch.zeros(4, device='mps', dtype=torch.long),
               torch.arange(4, device='mps'))
    score, index = MIPSFaiss(rhs).search(lhs, 2, exclude)
    assert score.device == index.device == lhs.device
    assert torch.isneginf(score).all()
    score.masked_fill(torch.isinf(score), 0).sum().backward()
    torch.testing.assert_close(lhs.grad, torch.zeros_like(lhs))


@pytest.mark.parametrize('with_exclude', [False, True])
def test_macos_cpu_retrieval_does_not_use_faiss_search(
    monkeypatch: pytest.MonkeyPatch,
    with_exclude: bool,
) -> None:
    require_backend(torch.device('cpu'))
    monkeypatch.setattr(sys.modules[MIPSFaiss.__module__], 'sys',
                        SimpleNamespace(platform='darwin'))
    import faiss

    def fail_search(*args, **kwargs):
        raise AssertionError('macOS CPU retrieval must not use Faiss search')

    monkeypatch.setattr(faiss, 'knn', fail_search)
    rhs = torch.eye(4, dtype=torch.float64)
    lhs = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64,
                       requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_()
    exclude = (torch.tensor([0]), torch.tensor([3])) if with_exclude else None
    index = MIPSFaiss(rhs)
    assert isinstance(index._cpu_index, MIPSBruteForce)
    score, ids = index.search(lhs, 2, exclude)
    expected_score, expected_ids = MIPSBruteForce(rhs).search(
        expected_lhs, 2, exclude)
    torch.testing.assert_close(score, expected_score)
    torch.testing.assert_close(ids, expected_ids)
    score.sum().backward()
    expected_score.sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad)


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS dispatch')
@pytest.mark.parametrize('first_device', ['cpu', 'mps'])
def test_macos_cpu_and_metal_retrieval_in_one_process(
    first_device: str,
) -> None:
    if first_device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    # A subprocess turns a native OpenMP abort/segfault into a test failure.
    script = '''
import sys
import torch
from kumo_core.modeling.retrieval.mips import MIPSBruteForce, MIPSFaiss

torch.manual_seed(123)
devices = ['cpu']
if torch.backends.mps.is_available():
    devices = (['cpu', 'mps', 'cpu'] if sys.argv[1] == 'cpu'
               else ['mps', 'cpu', 'mps'])
for device in devices:
    rhs = torch.randn(1024, 32, device=device)
    lhs = torch.randn(128, 32, device=device, requires_grad=True)
    index = MIPSFaiss(rhs)
    scores, ids = index.search(lhs, 5)
    expected_scores, expected_ids = MIPSBruteForce(rhs).search(lhs, 5)
    torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ids, expected_ids)
    scores.sum().backward()
    assert torch.isfinite(lhs.grad).all()
    layer = torch.nn.Linear(32, 16)
    layer(torch.randn(512, 32)).sum().backward()
print('CPU and Metal retrieval passed')
'''
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(('OMP_', 'KMP_', 'MKL_', 'VECLIB_'))
    }
    result = subprocess.run([sys.executable, '-c', script, first_device], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
