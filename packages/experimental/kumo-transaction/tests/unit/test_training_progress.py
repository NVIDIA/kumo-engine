# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`TrainingProgress`: what it prints, when, and the curve every run records."""

from __future__ import annotations

import types
import warnings
from collections.abc import Iterator

import pytest
import torch

import tfm.training.progress as progress_module
from tfm import Decoder, Flat, TrxFM
from tfm.config import TrainingConfig
from tfm.contracts.batch import TransactionBatch
from tfm.training.progress import TrainingProgress
from tfm.training.trainer import real_tokens, train_model


def _run(progress: TrainingProgress, steps: int, *, grad_norm: float = 1.0) -> None:
    progress.start(max_steps=steps, batches_per_epoch=2)
    for step in range(1, steps + 1):
        progress.step(step, loss=1.0 / step, lr=1e-3, grad_norm=grad_norm, tokens=8.0)


def test_prints_every_log_every_steps_and_the_last_step() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=2, write=lines.append, rank=0)

    _run(progress, steps=5)

    assert lines[0] == "training 5 steps, 2 minibatches per epoch"
    assert [line.split(" | ")[0] for line in lines[1:]] == ["step 2/5", "step 4/5", "step 5/5"]
    assert "epoch 2.50" in lines[-1]
    assert "grad norm 1.000" in lines[-1]


def test_the_token_rate_includes_the_readback_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """The readback waits for the GPU; the interval must end after it."""
    clock = [0.0]
    monkeypatch.setattr(
        progress_module, "time", types.SimpleNamespace(perf_counter=lambda: clock[0])
    )
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=0)
    progress.start(max_steps=1, batches_per_epoch=1)
    finish = progress.finish

    def waits() -> None:
        clock[0] += 1.0
        finish()

    monkeypatch.setattr(progress, "finish", waits)
    progress.step(1, loss=1.0, lr=1e-3, grad_norm=1.0, tokens=2000.0)
    assert "| 2k tok/s |" in lines[-1]


def test_records_every_step_without_printing_when_log_every_is_none() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=None, write=lines.append, rank=0)

    _run(progress, steps=3)

    assert lines == []
    assert [row["step"] for row in progress.curve] == [1.0, 2.0, 3.0]
    assert progress.curve[1]["loss"] == pytest.approx(0.5)
    assert progress.curve[1]["epoch"] == pytest.approx(1.0)
    assert set(progress.curve[0]) == {
        "step",
        "epoch",
        "loss",
        "lr",
        "grad_norm",
        "tokens",
        "seconds",
    }


def test_only_the_first_rank_prints() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=1)

    _run(progress, steps=2)

    assert lines == []
    assert len(progress.curve) == 2


def test_an_unclipped_backend_omits_the_gradient_norm() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=0)

    _run(progress, steps=1, grad_norm=float("nan"))

    assert "grad norm" not in lines[-1]


def test_several_processes_are_named_in_the_header() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=1, write=lines.append, rank=0)

    progress.start(max_steps=4, batches_per_epoch=3, world_size=2)

    assert lines == ["training 4 steps on 2 processes, 3 minibatches per epoch per process"]


def test_log_every_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="log_every must be >= 1"):
        TrainingProgress(log_every=0)


def test_config_log_every_defaults_to_silent_and_rejects_zero() -> None:
    assert TrainingConfig().log_every is None
    assert TrainingConfig.model_validate({"log_every": 50}).log_every == 50
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        TrainingConfig.model_validate({"log_every": 0})


def test_real_tokens_counts_unpadded_positions() -> None:
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    )
    assert real_tokens(batch) == 5.0


def test_tensor_values_are_read_back_when_a_line_is_due() -> None:
    lines: list[str] = []
    progress = TrainingProgress(log_every=2, write=lines.append, rank=0)
    progress.start(max_steps=2, batches_per_epoch=1)

    progress.step(1, loss=torch.tensor(0.5), lr=1e-3, grad_norm=torch.tensor(2.0), tokens=8.0)
    assert isinstance(progress.curve[0]["loss"], torch.Tensor)
    progress.step(2, loss=torch.tensor(0.25), lr=1e-3, grad_norm=torch.tensor(1.0), tokens=8.0)

    assert [row["loss"] for row in progress.curve] == [0.5, 0.25]
    assert "loss 0.2500" in lines[-1]
    assert "grad norm 1.000" in lines[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tensor_values_on_different_devices_are_read_back() -> None:
    progress = TrainingProgress(log_every=None, rank=0)
    progress.start(max_steps=1, batches_per_epoch=1)
    loss = torch.tensor(0.5, device="cuda")
    progress.step(1, loss=loss, lr=1e-3, grad_norm=torch.tensor(2.0), tokens=8.0)
    progress.finish()
    assert (progress.curve[0]["loss"], progress.curve[0]["grad_norm"]) == (0.5, 2.0)


def test_silent_progress_reads_tensor_values_back_every_100_steps() -> None:
    progress = TrainingProgress(log_every=None, rank=0)
    progress.start(max_steps=250, batches_per_epoch=1)

    for step in range(1, 151):
        progress.step(step, loss=torch.tensor(1.0), lr=1e-3, grad_norm=1.0, tokens=1.0)

    assert all(isinstance(row["loss"], float) for row in progress.curve[:100])
    assert all(isinstance(row["loss"], torch.Tensor) for row in progress.curve[100:])
    progress.finish()
    assert all(isinstance(row["loss"], float) for row in progress.curve)


def _flat_model_and_batch(device: torch.device) -> tuple[TrxFM, TransactionBatch]:
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=2, vocab_size=16, dropout=0.0))
    )
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        labels=torch.tensor([[2, 3, -100, -100], [5, -100, -100, -100]], dtype=torch.long),
    )
    return model, batch.to(device)


def _trainer_syncs(steps: int) -> int:
    """Synchronizing calls made from the trainer or progress code during a CUDA run."""
    model, batch = _flat_model_and_batch(torch.device("cuda"))
    progress = TrainingProgress(log_every=None, rank=0)
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            train_model(
                model,
                [batch],
                max_steps=steps,
                learning_rate=1e-3,
                seed=0,
                device=torch.device("cuda"),
                progress=progress,
            )
        finally:
            torch.cuda.set_sync_debug_mode(0)
    return sum(
        "synchroniz" in str(w.message) and w.filename.endswith(("trainer.py", "progress.py"))
        for w in caught
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_model_adds_no_host_sync_per_step() -> None:
    assert _trainer_syncs(steps=2) == _trainer_syncs(steps=6)


def test_train_model_feeds_every_step_to_progress() -> None:
    model, batch = _flat_model_and_batch(torch.device("cpu"))
    lines: list[str] = []
    progress = TrainingProgress(log_every=2, write=lines.append, rank=0)

    stats = train_model(
        model,
        [batch],
        max_steps=3,
        learning_rate=1e-3,
        seed=0,
        device=torch.device("cpu"),
        progress=progress,
    )

    assert stats["steps"] == 3.0
    assert [row["step"] for row in progress.curve] == [1.0, 2.0, 3.0]
    assert all(row["tokens"] == 5.0 for row in progress.curve)
    assert progress.curve[-1]["loss"] == pytest.approx(stats["final_loss"])
    assert len(lines) == 3


class _Stream:
    """A loader that can only be iterated, yielding `batch` twice per pass."""

    def __init__(self, batch: TransactionBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[TransactionBatch]:
        return iter([self.batch, self.batch])


def test_train_model_reads_progress_back_from_an_iterable_loader() -> None:
    model, batch = _flat_model_and_batch(torch.device("cpu"))
    progress = TrainingProgress(log_every=None, rank=0)

    train_model(
        model,
        _Stream(batch),
        max_steps=3,
        learning_rate=1e-3,
        seed=0,
        device=torch.device("cpu"),
        progress=progress,
    )

    assert [row["step"] for row in progress.curve] == [1.0, 2.0, 3.0]
    assert all(isinstance(row[key], float) for row in progress.curve for key in row)
