# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `ddp` backend across real processes (backend contracts).

The shared backend suite runs `ddp` single-process, where it is the eager loop.
These tests launch two Gloo ranks on CPU so gradient averaging actually runs,
and repeat on NCCL with two and four GPUs when the host has them:

- With the same shard on both ranks, averaged gradients equal each rank's own,
  so the run must match eager exactly.
- With different shards, the replicas must still end identical, which only
  holds if gradients were averaged.

The `llama` case is the HuggingFace backbone the TabFormer recipe trains with
`training.backend: ddp`; it is skipped without the `huggingface` extra.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from tfm import Decoder, Flat, TrxFM
from tfm.config import OptimizerConfig, RecipeConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.experimental import ProfileEventDecoder, compose_blueprint_v2
from tfm.extras.capability import extra_available
from tfm.features.tokenizers import KeyValueTimeTokenizer
from tfm.training.ddp import DdpTrainer
from tfm.training.trainer import TrainableBatch, build_trainer

STEPS = 6
VOCAB = 32


def _launch_target(device: str, world_size: int):
    marks = []
    if device == "cuda":
        marks.append(
            pytest.mark.skipif(
                torch.cuda.device_count() < world_size,
                reason=f"NCCL case needs {world_size} CUDA devices",
            )
        )
    return pytest.param(device, world_size, id=f"{device}-{world_size}", marks=marks)


LAUNCH_TARGETS = [_launch_target("cpu", 2), _launch_target("cuda", 2), _launch_target("cuda", 4)]


def _flat_batch(offset: int) -> TransactionBatch:
    ids = torch.tensor([[3 + (offset + r + i) % 20 for i in range(10)] for r in range(4)])
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    return TransactionBatch(input_ids=ids, attention_mask=mask, labels=labels)


def _kvt_sequences(offset: int) -> list[TransactionSequence]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    seqs = []
    for u in range(4):
        events = tuple(
            TransactionEvent(
                entity_id=f"u{u}",
                event_id=f"u{u}-{i}",
                event_timestamp=start + timedelta(hours=3 * i + u),
                categorical={"source": "transaction", "state": f"s{(offset + u + i) % 4}"},
            )
            for i in range(5)
        )
        seqs.append(
            TransactionSequence(
                entity_id=f"u{u}",
                events=events,
                cutoff_time=events[-1].event_timestamp,
                observation_window=None,
            )
        )
    return seqs


def _kvt_batch(offset: int) -> TrainableBatch:
    # One vocabulary for every offset: a tokenizer fitted per batch numbers values
    # in encounter order, which would make shifted batches collate identically.
    tok = KeyValueTimeTokenizer()
    tok.fit([seq for o in range(10) for seq in _kvt_sequences(o)])
    return tok.collate(_kvt_sequences(offset), max_length=8)


def _llama() -> nn.Module:
    from tfm.runtime.pipeline import build_model

    config = RecipeConfig.model_validate(
        {
            "seed": 0,
            "data": {
                "adapter": "synthetic",
                "max_tokens": 10,
                "temporal_split": {
                    "train_end": "2024-01-04T00:00:00Z",
                    "validation_end": "2024-01-07T00:00:00Z",
                },
            },
            "model": {
                "backbone": "huggingface",
                "composition": {
                    "position": "rope",
                    "granularity": {
                        "decoder": {
                            "depth": 1,
                            "hidden_size": 32,
                            "num_heads": 4,
                            "max_seq_len": 10,
                            "dropout": 0.0,
                            "mask": "causal",
                        }
                    },
                },
            },
        }
    )
    torch.manual_seed(0)
    return build_model(config, VOCAB)


def _model(case: str) -> nn.Module:
    torch.manual_seed(0)
    if case == "llama":
        return _llama()
    if case == "flat":
        return TrxFM(
            granularity=Flat(
                Decoder(depth=1, hidden_size=32, num_heads=4, vocab_size=VOCAB, dropout=0.0)
            )
        )
    spec = ProfileEventDecoder(
        hidden_size=32,
        num_heads=4,
        profile_depth=1,
        event_depth=1,
        decoder_depth=1,
        vocab_size=64,
    )
    return compose_blueprint_v2(granularity=spec)


def _batches(case: str, offsets: list[int]) -> list[TrainableBatch]:
    build = _kvt_batch if case == "blueprint_v2" else _flat_batch
    return [build(o) for o in offsets]


def _fit(
    name: str, model: nn.Module, batches: list[TrainableBatch], device: str = "cpu"
) -> dict[str, float | str]:
    torch.manual_seed(1)  # MLM masking draws from the global generator.
    return build_trainer(name).fit(
        model,
        batches,
        max_steps=STEPS,
        learning_rate=1e-2,
        seed=0,
        device=torch.device(device),
        optimizer=OptimizerConfig(),
    )


def _rank_main(
    rank: int,
    port: int,
    world_size: int,
    device: str,
    case: str,
    offsets: list[int],
    out: str,
) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    torch.set_num_threads(1)
    model = _model(case)
    stats = _fit("ddp", model, _batches(case, offsets), device)
    checksum = float(sum(p.detach().double().sum() for p in model.parameters()))
    params_device = str(next(model.parameters()).device)
    Path(out, f"rank{rank}.json").write_text(
        json.dumps({**stats, "checksum": checksum, "params_device": params_device}),
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _launch(
    case: str, offsets: list[int], tmp_path: Path, *, device: str, world_size: int
) -> list[dict[str, float]]:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    if device == "cpu" and not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    if device == "cuda" and not torch.distributed.is_nccl_available():
        pytest.skip("NCCL is unavailable")
    mp.spawn(
        _rank_main,
        args=(_free_port(), world_size, device, case, offsets, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    return [
        json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8"))
        for r in range(world_size)
    ]


def _require(case: str) -> None:
    if case == "llama" and not extra_available("transformers"):
        pytest.skip("the llama case needs the huggingface extra")


@pytest.mark.parametrize(("device", "world_size"), LAUNCH_TARGETS)
@pytest.mark.parametrize("case", ["flat", "blueprint_v2", "llama"])
def test_ranks_on_the_same_shard_match_eager(
    case: str, device: str, world_size: int, tmp_path: Path
) -> None:
    _require(case)
    # Batches are dealt round-robin, so [a] * W + [b] * W gives each rank [a, b].
    ranks = _launch(
        case, [0] * world_size + [5] * world_size, tmp_path, device=device, world_size=world_size
    )
    eager_model = _model(case)
    eager = _fit("eager", eager_model, _batches(case, [0, 5]), device)
    checksum = float(sum(p.detach().double().sum() for p in eager_model.parameters()))
    for rank, stats in enumerate(ranks):
        assert stats["world_size"] == world_size
        assert stats["batches_per_rank"] == 2
        assert stats["final_loss"] == pytest.approx(float(eager["final_loss"]), rel=1e-5)
        assert stats["checksum"] == pytest.approx(checksum, rel=1e-6)
        if device == "cuda":
            assert stats["params_device"] == f"cuda:{rank}"


@pytest.mark.parametrize(("device", "world_size"), LAUNCH_TARGETS)
@pytest.mark.parametrize("case", ["flat", "blueprint_v2", "llama"])
def test_replicas_stay_identical_on_different_shards(
    case: str, device: str, world_size: int, tmp_path: Path
) -> None:
    _require(case)
    # The key-value-time states repeat every 4 offsets, so each rank's first
    # batch takes a distinct offset modulo 4.
    offsets = [0, 5, 2, 7, 1, 3, 8, 6][: 2 * world_size]
    shards = _batches(case, offsets)
    firsts = [
        shards[r].value_ids if case == "blueprint_v2" else shards[r].input_ids  # type: ignore[attr-defined]
        for r in range(world_size)
    ]
    for a in range(world_size):
        for b in range(a + 1, world_size):
            assert not torch.equal(firsts[a], firsts[b]), "every rank's shard must differ"
    ranks = _launch(case, offsets, tmp_path, device=device, world_size=world_size)
    for stats in ranks[1:]:
        assert stats["checksum"] == ranks[0]["checksum"]
        # Reported losses are averaged over ranks, so every rank reports the same value.
        assert stats["final_loss"] == ranks[0]["final_loss"]


@pytest.mark.parametrize("case", ["blueprint_v2", "llama"])
def test_single_process_ddp_is_the_eager_loop(case: str) -> None:
    _require(case)
    batches = _batches(case, [0, 5])
    ddp = _fit("ddp", _model(case), batches)
    eager = _fit("eager", _model(case), batches)
    assert ddp["world_size"] == 1.0
    assert ddp["final_loss"] == eager["final_loss"]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ),
    ],
)
def test_blueprint_v2_trains_in_bf16_close_to_fp32(device: str) -> None:
    batches = _batches("blueprint_v2", [0, 5])
    fp32_model = _model("blueprint_v2")
    bf16_model = _model("blueprint_v2")
    fp32 = _fit("ddp", fp32_model, batches, device)
    torch.manual_seed(1)
    bf16 = build_trainer("ddp").fit(
        bf16_model,
        batches,
        max_steps=STEPS,
        learning_rate=1e-2,
        seed=0,
        device=torch.device(device),
        optimizer=OptimizerConfig(),
        precision="bf16",
    )
    assert bf16["precision"] == "bf16"
    assert bf16["final_loss"] == pytest.approx(float(fp32["final_loss"]), rel=5e-2)
    assert all(p.dtype == torch.float32 for p in bf16_model.parameters())


def test_ddp_replicates_rather_than_shards() -> None:
    capabilities = DdpTrainer.capabilities
    assert capabilities.multi_process
    assert not capabilities.shards_parameters
    assert not capabilities.requires_cuda
    assert "key_value_time" in capabilities.batch_layouts
    assert "eager" in capabilities.backbones


class _Stream:
    """A sized loader that can only be iterated."""

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[TransactionBatch]:
        return iter([])


def test_ddp_refuses_a_loader_it_cannot_index() -> None:
    with pytest.raises(UnsupportedCombinationError, match="_Stream, which can only be iterated"):
        DdpTrainer().fit(
            nn.Linear(1, 1),
            _Stream(),  # type: ignore[arg-type]
            max_steps=1,
            learning_rate=1e-3,
            seed=0,
            device=torch.device("cpu"),
        )
