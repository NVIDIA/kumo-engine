# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every attention kernel must agree with the eager baseline.

AGENTS.md requires a correct eager baseline for each optimized path, so these
are equivalence contracts rather than behavior tests of one kernel.
"""

from __future__ import annotations

import pytest
import torch

from tfm.attention import (
    EagerAttention,
    EagerAttentionKernel,
    SdpaAttention,
    SdpaAttentionKernel,
    advertised_attention,
    build_attention,
)
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.protocols import AttentionSpec
from tfm.data.packing import pack_token_rows

KERNELS = (EagerAttention(), SdpaAttention())


def _qkv(batch: int = 2, heads: int = 2, seq: int = 6, dim: int = 8):
    torch.manual_seed(0)
    shape = (batch, heads, seq, dim)
    return torch.randn(shape), torch.randn(shape), torch.randn(shape)


@pytest.mark.parametrize("spec", KERNELS, ids=lambda s: s.name)
def test_every_kernel_satisfies_the_attention_spec(spec) -> None:
    assert isinstance(spec, AttentionSpec)
    assert spec.capabilities()["causal"] is True
    assert spec.capabilities()["bidirectional"] is True


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
def test_sdpa_matches_eager_with_padding(causal: bool) -> None:
    query, key, value = _qkv()
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])

    want = EagerAttentionKernel()(query, key, value, mask, causal=causal)
    got = SdpaAttentionKernel()(query, key, value, mask, causal=causal)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_sdpa_matches_eager_on_packed_segments() -> None:
    query, key, value = _qkv(batch=1, seq=5)
    packed, _stats = pack_token_rows([[1, 2, 3], [4, 5]], max_tokens=5)
    assert packed.segment_ids is not None

    want = EagerAttentionKernel()(
        query, key, value, packed.attention_mask, causal=True, segment_ids=packed.segment_ids
    )
    got = SdpaAttentionKernel()(
        query, key, value, packed.attention_mask, causal=True, segment_ids=packed.segment_ids
    )

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_sdpa_zeros_fully_masked_rows_like_eager() -> None:
    """A padded query row sees nothing; softmax must not produce NaN."""
    query, key, value = _qkv(batch=1, seq=4)
    mask = torch.tensor([[0, 0, 0, 0]])

    want = EagerAttentionKernel()(query, key, value, mask, causal=False)
    got = SdpaAttentionKernel()(query, key, value, mask, causal=False)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sdpa_keeps_short_sequences_off_cudnn() -> None:
    query, key, value = (t.cuda().bfloat16() for t in _qkv(batch=64, heads=8, seq=10, dim=32))
    mask = torch.ones(64, 10, dtype=torch.long, device="cuda")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        SdpaAttentionKernel()(query, key, value, mask, causal=False)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kernels
    assert not [k for k in kernels if "cudnn" in k.lower()]


def test_registry_resolves_config_names() -> None:
    assert advertised_attention() == ("eager", "sdpa")
    assert build_attention("eager").name == "eager"
    assert build_attention("SDPA").name == "sdpa"
    # Configs and the capability registry spell it either way.
    assert build_attention("eager_attention").name == "eager"


def test_registry_rejects_an_unknown_kernel() -> None:
    with pytest.raises(UnsupportedCombinationError, match="flash"):
        build_attention("flash")
