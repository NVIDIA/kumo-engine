"""CUDA reference loader validation; no model training or engine-parity oracle."""

import json
import os
from collections.abc import Mapping

import pytest
import torch

from test.kumo_pytorch.aml_benchmark import counts, factory
from test.kumo_pytorch.benchmarking import run
from test.kumo_pytorch.benchmarking.campaign import compare_states

from .test_benchmark_aml import CASES, source  # noqa: F401 -- shared local source fixture


@pytest.mark.parametrize("workload,family", CASES)
def test_gpu_loader_timings_and_state_reuse(source, workload, family):  # noqa: F811 -- imported fixture
    if not torch.cuda.is_available():
        if os.environ.get("KUMO_GPU_REFERENCE_REQUIRED") == "1":
            pytest.fail("GPU reference validation requires CUDA on the nightly runner")
        pytest.skip("CUDA reference runs in the nightly benchmark job")
    config = {
        "data_dir": str(source),
        "family": family,
        "workload": workload,
        "anchors_per_split": 1,
        "batch_size": 1,
        "threads": 1,
        "backend": "gpu",
        "training": False,
        "device": "cuda:0",
        "save_fitted_state": str(source / "state"),
    }

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from tensors(item)

    def adapter(config, workdir):
        result = factory(config, workdir)

        def count(batch):
            payload = list(tensors(batch.inputs)) + list(tensors(batch.targets))
            assert payload and all(t.device.type == "cuda" for t in payload)
            return counts(batch)

        result.count = count
        return result

    fresh = run(
        adapter,
        config,
        source / "fresh",
        repetitions=1,
        epochs=2,
        measurement_profile="timing",
    )
    assert fresh["execution_status"] == "completed", fresh["errors"]
    for name in (
        "reference.load",
        "reference.fit",
        "reference.plan",
        "reference.batch",
    ):
        assert fresh["measurements"][name + ".elapsed_ns"]
    assert not fresh["measurements"].get("training.step.elapsed_ns")
    assert not fresh["gpu_observations"]
    coverage = json.loads((source / "fresh/coverage.json").read_text())["metrics"]
    assert {name for name, row in coverage.items() if row["status"] != "complete"} == {
        "later_batch_latency_p50_ns",
        "later_batch_latency_p95_ns",
    }
    config["load_fitted_state"] = config.pop("save_fitted_state")
    reused = run(
        adapter,
        config,
        source / "reused",
        repetitions=1,
        epochs=2,
        measurement_profile="timing",
    )
    assert reused["execution_status"] == "completed", reused["errors"]
    assert compare_states(fresh, reused)["status"] == "measured"
    assert all(row["metrics"]["real_samples"] == 1 for row in reused["epochs"])
