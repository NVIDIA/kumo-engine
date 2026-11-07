# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e import runs


@pytest.fixture(scope="module")
def xs(tmp_path_factory: pytest.TempPathFactory) -> HMDataset:
    return generate("XS", 7, tmp_path_factory.mktemp("hm-xs"))


@pytest.fixture
def trained(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The flow commands ``run`` starts, recorded instead of training."""

    commands: list[list[str]] = []

    def train(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        Path(command[command.index("--work") + 1]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runs.subprocess, "run", train)
    monkeypatch.delenv("KUMO_PYTORCH_E2E_REUSE_ONLY", raising=False)
    return commands


def sampled(xs: HMDataset, root: Path, fractions: Mapping[str, float], seed: int) -> HMDataset:
    return sample_anchors(xs, fractions, seed, root / "sampled")


def test_a_run_is_reused_only_on_the_same_anchor_sample(
    xs: HMDataset, tmp_path: Path, trained: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs.run(sampled(xs, tmp_path, {"train": 0.25}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 1

    runs.run(sampled(xs, tmp_path, {"train": 0.25}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 1, "the identical sample retrained"

    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 2, "another sample fraction reused the old run"

    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 1), tmp_path, "gnn", 0)
    assert len(trained) == 3, "another sample seed reused the old run"

    monkeypatch.setenv("KUMO_PYTORCH_E2E_REUSE_ONLY", "1")
    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 1), tmp_path, "gnn", 0)
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.25}, 1), tmp_path, "gnn", 0)
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.5}, 2), tmp_path, "gnn", 0)
    assert len(trained) == 3


def test_a_run_on_the_full_anchors_is_not_reused_for_a_sample(
    xs: HMDataset, tmp_path: Path, trained: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs.run(xs, tmp_path, "pragma", 1)
    runs.run(xs, tmp_path, "pragma", 1)
    assert len(trained) == 1

    monkeypatch.setenv("KUMO_PYTORCH_E2E_REUSE_ONLY", "1")
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.5}, 0), tmp_path, "pragma", 1)
