# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache boundary/report checks; real kernel acceptance uses validate_cache."""

import io
import json
from test.kumo_pytorch.benchmarking import BatchCounts, Workload, run
from test.kumo_pytorch.benchmarking.file_cache import (
    FilePageCache,
    condition_files,
)
from test.kumo_pytorch.benchmarking.recording import Recorder

import pytest


def test_unsupported_platform_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.file_cache.sys.platform", "unsupported"
    )
    with pytest.raises(RuntimeError, match="Linux"):
        condition_files([], "cold")


def test_unverified_advisory_eviction_does_not_pass(monkeypatch):
    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.file_cache.condition_files",
        lambda files, mode, **kwargs: {"status": "unavailable", "mode": mode},
    )
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="not verified"):
        FilePageCache(list)(0, Recorder(stream))
    assert json.loads(stream.getvalue())["status"] == "unavailable"


@pytest.mark.parametrize("changed", (None, "inode", "size", "mtime_ns"))
def test_only_verified_matching_files_establish_cache_pair(
    monkeypatch, tmp_path, changed
):
    def conditions(files, mode, *, release_mappings=False):
        identity = {"device": 1, "inode": 2, "size": 4096, "mtime_ns": 10}
        if changed is not None and mode == "warm":
            identity[changed] += 1
        return {
            "mode": mode,
            "status": "verified",
            "files": [identity],
        }

    monkeypatch.setattr(
        "test.kumo_pytorch.benchmarking.file_cache.condition_files", conditions
    )

    def factory(config, directory):
        return Workload(
            identity={"dataset": "cache-fixture", "workload": "read"},
            prepare=lambda recorder: None,
            batches=lambda epoch: iter([[1]]),
            count=lambda batch: BatchCounts(1, 0),
            before_epoch=FilePageCache(list),
        )

    result = run(factory, {}, tmp_path / "run", repetitions=1, epochs=2)
    assert result["execution_status"] == "completed"
    assert result["experiments"]["cold_and_warm_caches"]["status"] == (
        "measured" if changed is None else "pending"
    )
    assert result["status"] == "incomplete"
    assert len(result["cache_conditions"]) == 2


def test_first_epoch_length_cache_is_disclosed_outside_fixed_selection(monkeypatch, tmp_path):
    original = tmp_path / "index.bin"
    original.write_bytes(b"index")
    seen = []

    def condition(files, mode, **kwargs):
        seen.append(set(files))
        return {"status": "verified", "mode": mode}

    monkeypatch.setattr("test.kumo_pytorch.benchmarking.file_cache.condition_files", condition)
    cache = FilePageCache(lambda: tmp_path.iterdir())
    stream = io.StringIO()
    cache(0, Recorder(stream))
    lengths = tmp_path / "lengths-stage_0.bin"
    lengths.write_bytes(b"lengths")
    cache(1, Recorder(stream))
    assert seen == [{original}, {original}]
    cold, warm = map(json.loads, stream.getvalue().splitlines())
    assert cold["unconditioned_new_files"] == []
    assert warm["unconditioned_new_files"] == [{"path": str(lengths), "size": 7}]


def test_disappearing_selected_file_is_not_silently_removed(monkeypatch, tmp_path):
    original = tmp_path / "index.bin"
    original.touch()

    def condition(files, mode, **kwargs):
        for path in files:
            path.stat()
        return {"status": "verified", "mode": mode}

    monkeypatch.setattr("test.kumo_pytorch.benchmarking.file_cache.condition_files", condition)
    cache = FilePageCache(lambda: tmp_path.iterdir())
    stream = io.StringIO()
    cache(0, Recorder(stream))
    original.unlink()
    with pytest.raises(FileNotFoundError):
        cache(1, Recorder(stream))
    assert json.loads(stream.getvalue().splitlines()[-1])["status"] == "unavailable"
