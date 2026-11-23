# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed layouts carry kernel-ready varlen metadata at both nesting levels."""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

from kumo.pytorch import api
from tests.test_diskgraph_backend import (
    _fixture_rows,
    _pragma_contract,
    _requests,
    _schema,
    _write_parquet,
)


def _two_level_contract() -> api.DataLoaderSpec:
    base = _pragma_contract(_schema())
    history = base.batch.inputs["history"]
    layout = api.Nested(fields=api.Packed(), events=api.Packed())
    inputs = {**base.batch.inputs, "history": replace(history, layout=layout)}
    return replace(base, batch=replace(base.batch, inputs=inputs))


def _check(history: dict) -> None:
    for prefix, node in (("field_", history), ("", history["sequence"])):
        offsets = node[f"{prefix}offsets"]
        cu = node[f"{prefix}cu_seqlens"]
        assert cu.dtype == torch.int32
        assert cu.tolist() == offsets.tolist()
        assert isinstance(node[f"{prefix}max_seqlen"], int)
        assert node[f"{prefix}max_seqlen"] == int((offsets[1:] - offsets[:-1]).max())


def test_diskgraph_model_path_carries_the_same_metadata(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    parquet = _write_parquet(_fixture_rows(), tmp_path / "parquet")
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with api.prepare(_two_level_contract(), parquet, engine) as plan:
        task = plan.materialize(_requests(), fit_split="train")
        batch = next(iter(task.batches(_requests(), batch_size=2, output_format="model")))
        _check(batch.inputs["history"])


@pytest.fixture
def native_packed_batch(tmp_path, native_anchor_delivery):
    pytest.importorskip("diskgraph")
    parquet = _write_parquet(_fixture_rows(), tmp_path / "parquet")
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    contract = _two_level_contract()
    rows = [
        {"user": r.entity_id, "at": r.as_of_time, "label": r.targets[0].value} for r in _requests()
    ]
    source = _write_parquet({"anchors": rows}, tmp_path / "anchors").tables["anchors"]
    anchors = api.AnchorTable(source, "user", "at", targets={"transactions_next_7d": "label"})
    fixed = replace(contract, batch=replace(contract.batch, batching=api.Batching(max_samples=2)))
    with api.prepare(fixed, parquet, engine) as plan:
        with plan.materialize(anchors) as native:
            return next(native_anchor_delivery(native))


def test_native_two_level_packing_carries_cu_seqlens(native_packed_batch):
    _check(native_packed_batch.inputs["history"])


@pytest.mark.skipif(
    find_spec("torch.nn.attention.varlen") is None,
    reason=f"torch {torch.__version__} has no torch.nn.attention.varlen",
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="varlen attention needs a BF16-capable CUDA device",
)
def test_varlen_attention_consumes_the_metadata_directly(native_packed_batch) -> None:
    from torch.nn.attention.varlen import varlen_attn

    history = native_packed_batch.to("cuda").inputs["history"]
    tokens = history["field_ids"].numel()
    q = torch.randn(tokens, 2, 64, device="cuda", dtype=torch.bfloat16)
    cu, longest = history["field_cu_seqlens"], history["field_max_seqlen"]
    out = varlen_attn(q, q, q, cu, cu, longest, longest)
    assert out.shape == q.shape
