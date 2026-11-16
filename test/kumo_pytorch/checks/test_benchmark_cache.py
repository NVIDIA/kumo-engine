"""Cache boundary/report checks; real kernel acceptance uses validate_cache."""

import io
import json

import pytest

from test.kumo_pytorch.benchmarking import BatchCounts, Workload, run
from test.kumo_pytorch.benchmarking.file_cache import FilePageCache, condition_files
from test.kumo_pytorch.benchmarking.recording import Recorder


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


@pytest.mark.parametrize("same_files", (True, False))
def test_only_verified_matching_files_establish_cache_pair(
    monkeypatch, tmp_path, same_files
):
    def conditions(files, mode, *, release_mappings=False):
        return {
            "mode": mode,
            "status": "verified",
            "files": [
                {
                    "device": 1,
                    "inode": 2 if same_files or mode == "cold" else 3,
                    "size": 4096,
                    "mtime_ns": 10,
                }
            ],
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
        "measured" if same_files else "pending"
    )
    assert result["status"] == "incomplete"
    assert len(result["cache_conditions"]) == 2
