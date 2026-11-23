# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tests.e2e import throughput_resources as resources
from tests.e2e.throughput import BATCH_SIZE, _window


def test_timed_window_counts_batches_even_when_a_batch_has_no_labels():
    weight = torch.nn.Parameter(torch.ones(()))
    batch = SimpleNamespace(
        batch_size=BATCH_SIZE,
        metadata={"is_padding": torch.zeros(BATCH_SIZE, dtype=torch.bool)},
        inputs={},
    )
    stream = iter([batch, batch, batch])
    losses = iter([None, weight.square()])
    model = SimpleNamespace(batch_loss=lambda batch, device: next(losses))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    result = _window(stream, 2, model, optimizer, torch.device("cpu"))
    assert (result["batches"], result["samples"], result["optimizer_updates"]) == (2, 64, 1)
    assert weight.item() < 1 and next(stream) is batch


def test_loader_window_counts_only_valid_history_tokens_without_a_model():
    batch = SimpleNamespace(
        batch_size=BATCH_SIZE,
        metadata={"is_padding": torch.zeros(BATCH_SIZE, dtype=torch.bool)},
        inputs={"history": {"valid": torch.tensor([[True, False]] * BATCH_SIZE)}},
    )
    result = _window(iter([batch, batch]), 2)
    assert result["samples"] == result["history_tokens"] == 2 * BATCH_SIZE
    assert "optimizer_updates" not in result


def test_resource_totals_exclude_other_workloads_and_count_shared_blocks_once(
    tmp_path, monkeypatch
):
    pin = tmp_path / "manifest.json"
    pin.write_text('{"objects":{"anchors":{"key":"anchors"}}}')
    monkeypatch.setattr(resources, "pin_path", lambda workload: pin)
    counters = iter([{"read_bytes": 12, "write_bytes": 30}, {"read_bytes": 16, "write_bytes": 50}])
    monkeypatch.setattr(resources, "_process_io", lambda: next(counters))
    staged = tmp_path / "anchors"
    staged.write_bytes(b"a" * 8192)
    (tmp_path / "unrelated").write_bytes(b"b" * 16384)
    index = tmp_path / "index"
    with resources.workload_resources("M04", tmp_path, index) as report:
        index.mkdir()
        (index / "linked-anchors").hardlink_to(staged)
        derived = index / "nodes"
        derived.write_bytes(b"c" * 8192)
    assert report["staged_bytes"] == staged.stat().st_blocks * 512
    assert report["peak_sampled_scratch_bytes"] == sum(
        path.stat().st_blocks * 512 for path in (staged, index, derived)
    )
    assert (report["process_read_bytes"], report["process_write_bytes"]) == (4, 20)
    assert report["peak_rss_bytes"] > 0


def test_training_gpu_reports_device_mean_and_empty_window(monkeypatch):
    nvml, sampled, values = Mock(), Event(), []
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="123")
    )

    def read(handle):
        values.append(20 if not values else 80)
        if len(values) >= 2:
            sampled.set()
        return SimpleNamespace(gpu=values[-1])

    nvml.nvmlDeviceGetUtilizationRates.side_effect = read
    with resources.training_gpu_window(torch.device("cuda")) as report:
        assert sampled.wait(5)
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-123")
    nvml.nvmlShutdown.assert_called_once()
    assert report["samples"] == len(values) and report["mean_percent"] == sum(values) / len(values)
    monkeypatch.setattr(resources, "_poll", lambda interval, sample: nullcontext())
    with resources.training_gpu_window(torch.device("cuda")) as empty:
        pass
    assert empty["samples"] == 0 and empty["mean_percent"] is None
