# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only rank 0 writes run artifacts, and a sharded run points at its HF export.

Regression: an 8-rank `torchrun` job had every rank write `train_stats.json`
concurrently, and the interleaved per-rank content was invalid JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch import nn

from tfm.runtime.common import is_primary_process, save_run_checkpoint


def test_non_primary_rank_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")

    save_run_checkpoint(nn.Linear(2, 2), tmp_path, {"seed": 0}, {"world_size": 8.0, "rank": 3.0})

    assert not is_primary_process()
    assert list(tmp_path.iterdir()) == []


def test_sharded_primary_points_at_the_consolidated_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    stats = {
        "backend": "fsdp2",
        "world_size": 8.0,
        "rank": 0.0,
        "hf_checkpoint": str(tmp_path / "hf_checkpoint"),
    }

    save_run_checkpoint(nn.Linear(2, 2), tmp_path, {"seed": 0}, stats)

    assert not (tmp_path / "model.pt").exists()
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["path"] == stats["hf_checkpoint"]
    assert manifest["format"] == "huggingface"
    assert json.loads((tmp_path / "train_stats.json").read_text(encoding="utf-8")) == stats


def test_replicated_primary_keeps_the_eager_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DDP rank holds the whole model, so rank 0 writes `model.pt`."""
    monkeypatch.setenv("RANK", "0")
    stats = {"backend": "ddp", "world_size": 8.0, "rank": 0.0, "hf_checkpoint": ""}

    save_run_checkpoint(nn.Linear(2, 2), tmp_path, {"seed": 0}, stats)

    assert (tmp_path / "model.pt").exists()
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["path"] == str(tmp_path / "model.pt")


def test_single_process_keeps_the_eager_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RANK", raising=False)

    save_run_checkpoint(nn.Linear(2, 2), tmp_path, {"seed": 0}, {"final_loss": 1.0})

    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "train_stats.json").exists()
