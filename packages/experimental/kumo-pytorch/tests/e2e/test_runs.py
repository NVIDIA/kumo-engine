# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e import runs


def test_loader_stage_observer_scopes_calls_and_restores_patches():
    from tests.e2e.loader_timing import stage_observer

    original = str.upper
    target = SimpleNamespace(call=original)

    def broken_setup(observer):
        observer.wrap(target, "call", "first")
        observer.wrap(target, "missing", "second")

    with pytest.raises(AttributeError, match="missing"), stage_observer(broken_setup):
        pytest.fail("observer setup should fail")
    assert target.call is original

    with stage_observer() as observer:
        observer.wrap(target, "call", lambda value: value)
        with pytest.raises(RuntimeError, match="already active"), stage_observer():
            pytest.fail("overlapping observers should be rejected")
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(target.call, "other").result() == "OTHER"
        assert target.call("owner") == "OWNER"
    assert target.call is original
    assert set(observer.stages) == {"owner"}
    assert observer.stages["owner"]["calls"] == 1
    assert observer.stages["owner"]["elapsed_ns"] > 0


def test_fused_native_profile_counts_the_interval_once():
    from tests.e2e.loader_timing import _native_totals

    fused = {"operation": "runtime", "fused": True, "sampling_ns": 7, "retrieval_ns": 7}
    assert _native_totals([[fused]]) == {"sample_and_retrieve_ns": 7}
    assert _native_totals([[fused], []]) is None
    assert _native_totals([[fused], [dict(fused, rows_decoded=4)]]) == {
        "sample_and_retrieve_ns": 14
    }


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
