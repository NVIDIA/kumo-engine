# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tfm.data.adapters.pragmatiq import write_pragmatiq_dataset

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/kumo-engine"
CONFIG = ROOT / "recipes/blueprint_v2/config.smoke.yaml"


def test_kumo_engine_example_builds_and_reuses_index(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    pytest.importorskip("kumo.pytorch.api")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    fields_by_source = {
        "transaction": {"amount": "5.0", "currency": "GBP"},
        "app": {"screen": "home", "action": "open"},
        "trading": {"instrument": "example", "quantity": "1.0", "price": "2.0"},
        "communication": {"comm_event": "sent", "channel": "email"},
    }
    source = write_pragmatiq_dataset(
        tmp_path / "source",
        {
            "profiles": [
                {"user_id": "u1", "as_of": start, "attributes": {"country": "GB"}, "lifelong": []}
            ],
            "events": [
                {
                    "user_id": "u1",
                    "ts": start + timedelta(hours=hour),
                    "source": source,
                    "fields": fields,
                }
                for hour, (source, fields) in enumerate(fields_by_source.items(), start=1)
            ],
        },
    )
    data = tmp_path / "data"
    scratch = tmp_path / "scratch"
    base_options = ("--config", str(CONFIG), "--threads", "1")

    def run(script: str, *options: str) -> str:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE / script), *base_options, *options],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run("seed_data.py", "--source", str(source), "--output", str(data))
    options = ("--scratch-dir", str(scratch), "--device", "cpu", "--steps", "1")
    fresh = tmp_path / "fresh"
    output = run("run.py", *options, "--data", str(data), "--output", str(fresh))
    (index,) = (scratch / "indexes").iterdir()
    assert f"DiskGraph index: {index}" in output
    assert (fresh / "model.pt").is_file()

    (data / "tables").rename(tmp_path / "offline-tables")
    reused = tmp_path / "reused"
    output = run(
        "run.py",
        *options,
        "--data",
        str(data),
        "--index",
        str(index),
        "--anchors",
        str(data / "anchors.parquet"),
        "--output",
        str(reused),
    )
    assert f"DiskGraph index: {index}" in output
    assert (reused / "model.pt").is_file()
    assert list((scratch / "indexes").iterdir()) == [index]
