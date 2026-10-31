# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native gateway ownership and startup failure cleanup."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from kumo_core.diskgraph.sampling_service import (
    SamplingService,
    SamplingServiceConfig,
)


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Use local process fixtures without application services."""
    return False


def test_missing_gateway_fails_before_process_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing executable produces an actionable error without starting a child."""
    (tmp_path / "index_metadata.json").write_text("{}")
    monkeypatch.setenv("DISKGRAPH_TASK_GATEWAY", str(tmp_path / "missing"))
    process = MagicMock()
    monkeypatch.setattr(
        "kumo_core.diskgraph.sampling_service.subprocess.Popen", process
    )
    with pytest.raises(FileNotFoundError, match="gateway not found"):
        SamplingService(SamplingServiceConfig(tmp_path))
    process.assert_not_called()


def test_failed_startup_reaps_process_and_retains_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An early process failure retains its diagnostic log and closes ownership."""
    (tmp_path / "index_metadata.json").write_text("{}")
    binary = tmp_path / "gateway"
    binary.touch()
    monkeypatch.setenv("DISKGRAPH_TASK_GATEWAY", str(binary))
    previous_logs = set(tmp_path.parent.glob("sampling-gateway-*.log"))
    process = MagicMock()
    process.poll.return_value = 1
    monkeypatch.setattr(
        "kumo_core.diskgraph.sampling_service.subprocess.Popen",
        MagicMock(return_value=process),
    )
    with pytest.raises(RuntimeError, match="inspect"):
        SamplingService(SamplingServiceConfig(tmp_path))
    assert len(set(tmp_path.parent.glob("sampling-gateway-*.log")) - previous_logs) == 1
    process.kill.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="Unix parent ownership")
def test_gateway_exits_when_its_python_owner_dies(tmp_path: Path) -> None:
    """Abrupt worker exit cannot leave a native listener running indefinitely."""
    import subprocess
    import sys
    import time

    import orjson
    import psutil

    (tmp_path / "index_metadata.json").write_text("{}")
    record = tmp_path / "owner.json"
    script = """
import os, sys
from pathlib import Path
import orjson
from kumo_core.diskgraph.sampling_service import SamplingService, SamplingServiceConfig
service = SamplingService(SamplingServiceConfig(Path(sys.argv[1])))
Path(sys.argv[2]).write_bytes(orjson.dumps(service.health()))
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(record)],
        check=True,
        timeout=30,
    )
    pid = orjson.loads(record.read_bytes())["pid"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.05)
    process.kill()
    raise AssertionError("Native sampling process survived its owning worker")
