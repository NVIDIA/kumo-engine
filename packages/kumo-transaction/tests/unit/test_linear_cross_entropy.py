# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from tfm.kernels import cross_entropy
from tfm.kernels.cross_entropy import IGNORE_INDEX, linear_cross_entropy

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(rows: int, vocab: int, dim: int, device: str, seed: int = 0, ignored: bool = False):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, dim, generator=gen).to(device).requires_grad_()
    # The offset moves each row's mean logit away from 0, where label smoothing's term
    # would vanish.
    weight = (torch.randn(vocab, dim, generator=gen) * 0.5 + 0.3).to(device).requires_grad_()
    targets = torch.randint(0, vocab, (rows,), generator=gen)
    if ignored:
        targets[::3] = -100
    return hidden, weight, targets.to(device)


def _grads(loss_fn, hidden, weight, targets, smoothing):
    hidden.grad = weight.grad = None
    loss = loss_fn(hidden, weight, targets, smoothing)
    loss.backward()
    assert hidden.grad is not None and weight.grad is not None
    return loss.detach(), hidden.grad.clone(), weight.grad.clone()


def _fused(hidden, weight, targets, smoothing):
    return linear_cross_entropy(hidden, weight, targets, label_smoothing=smoothing)


def _eager(hidden, weight, targets, smoothing):
    return F.cross_entropy(hidden @ weight.t(), targets, label_smoothing=smoothing)


@pytest.mark.parametrize("ignored", [False, True], ids=["all-rows", "ignored-rows"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_matches_eager_over_several_chunks_on_cpu(monkeypatch, smoothing, ignored):
    """50 rows in chunks of 7: the last chunk is partial. Every third row's target is
    -100 when `ignored`."""
    vocab = 37
    monkeypatch.setattr(cross_entropy, "CHUNK_BYTES", 7 * vocab * 4)
    hidden, weight, targets = _inputs(50, vocab, 16, "cpu", ignored=ignored)
    got = _grads(_fused, hidden, weight, targets, smoothing)
    want = _grads(_eager, hidden, weight, targets, smoothing)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want.float()).norm() / want.float().norm())


@CUDA
@pytest.mark.parametrize("ignored", [False, True], ids=["all-rows", "ignored-rows"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("vocab", [1000, 9001])
def test_matches_eager_under_autocast_on_cuda(monkeypatch, smoothing, vocab, ignored):
    """bf16 hidden and fp32 weights, as a model under autocast feeds the heads. 9,001
    columns take three of the kernel's row steps, the last partial; 1,000 take one.
    Both paths round to bf16, so they are compared on relative error against fp32."""
    monkeypatch.setattr(cross_entropy, "CHUNK_BYTES", 300 * vocab * 2)
    hidden, weight, targets = _inputs(1000, vocab, 64, "cuda", ignored=ignored)
    with torch.autocast("cuda", torch.bfloat16):
        got = _grads(lambda h, *a: _fused(h.bfloat16(), *a), hidden, weight, targets, smoothing)
        bf16 = _grads(lambda h, *a: _eager(h.bfloat16(), *a), hidden, weight, targets, smoothing)
    exact = _grads(_eager, hidden, weight, targets, smoothing)
    for g, b, x in zip(got, bf16, exact, strict=True):
        assert _relative_error(g, x) <= 2 * _relative_error(b, x) + 1e-4


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_custom_op_registration(device):
    hidden, weight, targets = _inputs(20, 30, 8, device, ignored=True)
    torch.library.opcheck(
        torch.ops.tfm.linear_cross_entropy.default, (hidden, weight, targets, 0.1)
    )


@pytest.mark.parametrize("target", [-1, 30], ids=["negative", "vocab"])
def test_a_target_outside_the_vocabulary_raises_on_cpu(target):
    """Only -100 is ignored; -1 and `vocab` are errors, as in `F.cross_entropy`."""
    hidden, weight, targets = _inputs(20, 30, 8, "cpu", ignored=True)
    targets[1] = target
    with pytest.raises((IndexError, RuntimeError)):
        linear_cross_entropy(hidden, weight, targets)


@CUDA
@pytest.mark.parametrize("target", [-1, 30], ids=["negative", "vocab"])
def test_a_target_outside_the_vocabulary_fails_a_device_assert_on_cuda(target):
    """Without `TRITON_DEBUG`, in a child process: a failed device assert leaves the CUDA
    context unusable."""
    script = (
        "import torch\n"
        "from tfm.kernels import linear_cross_entropy\n"
        "h, w = torch.randn(8, 16, device='cuda'), torch.randn(30, 16, device='cuda')\n"
        f"t = torch.tensor([0, -100, {target}, 1, 2, 3, 4, 5], device='cuda')\n"
        "linear_cross_entropy(h, w, t)\n"
        "torch.cuda.synchronize()\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        env={k: v for k, v in os.environ.items() if k != "TRITON_DEBUG"},
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "neither in [0, vocab)" in run.stderr


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_all_ignored_targets_give_zero_loss_and_gradients(monkeypatch, device):
    """50 rows in chunks of 7, every target `IGNORE_INDEX`: 0, not NaN, so a batch that
    is all padding adds nothing to the step."""
    monkeypatch.setattr(cross_entropy, "CHUNK_BYTES", 7 * 37 * 4)
    hidden, weight, targets = _inputs(50, 37, 16, device)
    targets.fill_(IGNORE_INDEX)
    loss, grad_hidden, grad_weight = _grads(_fused, hidden, weight, targets, 0.1)
    assert loss.item() == 0
    assert not grad_hidden.any() and not grad_weight.any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_strided_targets_match_eager(monkeypatch, device):
    """Targets read as every other element of a longer tensor: each row gets its own
    target, not the next one in memory."""
    monkeypatch.setattr(cross_entropy, "CHUNK_BYTES", 7 * 37 * 4)
    hidden, weight, targets = _inputs(50, 37, 16, device, ignored=True)
    strided = torch.stack([targets, targets.roll(1)], dim=1)[:, 0]
    assert not strided.is_contiguous()
    got = _grads(_fused, hidden, weight, strided, 0.1)
    want = _grads(_eager, hidden, weight, targets, 0.1)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_ignored_cpu_float16_overflow_has_zero_loss_and_gradients(smoothing):
    hidden = torch.full((3, 8), 30000.0, dtype=torch.float16, requires_grad=True)
    weight = torch.full((5, 8), 30000.0, dtype=torch.float16, requires_grad=True)
    targets = torch.full((3,), IGNORE_INDEX, dtype=torch.long)
    loss = linear_cross_entropy(hidden, weight, targets, label_smoothing=smoothing)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
    assert torch.equal(weight.grad, torch.zeros_like(weight))
