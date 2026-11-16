# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from common_utils.kumo_timing import (
    kumo_timing,
    kumo_timing_enabled,
    kumo_timing_event,
)


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_kumo_timing_disabled_by_default(tmp_path, monkeypatch):
    timing_file = tmp_path / "timing.jsonl"
    monkeypatch.delenv("KUMO_TIMING", raising=False)
    monkeypatch.setenv("KUMO_TIMING_FILE", str(timing_file))

    with kumo_timing("unit.disabled"):
        pass
    kumo_timing_event("unit.disabled.event")

    assert not kumo_timing_enabled()
    assert not timing_file.exists()


def test_kumo_timing_records_structured_redacted_jsonl(tmp_path, monkeypatch):
    timing_file = tmp_path / "timing.jsonl"
    monkeypatch.setenv("KUMO_TIMING", "1")
    monkeypatch.setenv("KUMO_TIMING_FILE", str(timing_file))

    with kumo_timing(
        "unit.operation",
        api_key="secret",
        count=3,
        file_path="/tmp/raw",
        nested={"download_url": "http://example.invalid/raw"},
    ):
        pass

    records = _read_jsonl(timing_file)

    assert [record["name"] for record in records] == [
        "unit.operation.start",
        "unit.operation.done",
    ]
    assert records[0]["event"] == "kumo_timing"
    assert records[0]["fields"]["api_key"] == "<redacted>"
    assert records[0]["fields"]["file_path"] == "<redacted>"
    assert records[0]["fields"]["nested"]["download_url"] == "<redacted>"
    assert records[0]["fields"]["count"] == 3
    assert records[1]["status"] == "done"
    assert records[1]["elapsed_seconds"] >= 0
    assert "timestamp_utc" in records[1]
    assert "process_name" in records[1]
    assert "thread_id" in records[1]
