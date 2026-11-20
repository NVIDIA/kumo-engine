# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny native fitting checks; simulated collectors are not Linux memory evidence."""

from test.kumo_pytorch.benchmarking import fitting_memory as fitting
from test.kumo_pytorch.contracts import build_contract

import pyarrow as pa
import pyarrow.parquet as pq

from .fixtures import CUTOFF, build_fixture
from .test_run_aml import mapped  # noqa: F401 - pytest fixture


def tiny_spec(tmp_path):
    fixture = build_fixture()
    dataset = fixture.parquet_dataset(tmp_path / "sources")
    contract = build_contract(fixture.schema, "nuformer", native=False, max_events=11)
    path = tmp_path / "contract.json"
    path.write_text(contract.to_json())
    anchors = {}
    for count in (1, 2):
        anchor_path = tmp_path / f"anchors-{count}.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": [str(i) for i in range(count)],
                    "entity": list(range(1, count + 1)),
                    "clock": [int(CUTOFF.timestamp())] * count,
                    "split": ["train"] * count,
                    "label": [float(i % 2) for i in range(count)],
                }
            ),
            anchor_path,
        )
        anchors[str(count)] = {
            "path": str(anchor_path),
            "sha256": fitting.digest(anchor_path),
        }
    spec = {
        "id": "tiny",
        "contract": str(path),
        "contract_fingerprint": contract.fingerprint,
        "sources": {name: source.path for name, source in dataset.tables.items()},
        "index": str(tmp_path / "index"),
        "threads": 1,
        "anchors": anchors,
        "anchor_columns": {
            "entity_id": "entity",
            "as_of_time": "clock",
            "sample_id": "id",
            "split": "split",
            "targets": {"label": "label"},
        },
        "fit_split": "train",
        "encoder_cutoff_seconds": int(CUTOFF.timestamp()) - 1,
    }
    spec_path = tmp_path / "spec.json"
    fitting.save(spec_path, spec)
    return spec_path


def test_native_worker_builds_once_and_fits_both_counts_without_reingesting(
    tmp_path, monkeypatch
):
    spec = tiny_spec(tmp_path)
    assert fitting.worker(spec, tmp_path / "build")["status"] == "passed"
    monkeypatch.setenv("KUMO_BENCHMARK_CGROUP", str(tmp_path))
    monkeypatch.setattr(fitting.SampledCgroupMemory, "validate", lambda self: None)
    monkeypatch.setattr(fitting.SampledCgroupMemory, "_read_current", lambda self: 4096)
    (tmp_path / "memory.stat").write_text("anon 2048\nfile 4096\n")
    before = {
        p.relative_to(tmp_path / "index"): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in (tmp_path / "index").rglob("*")
        if p.is_file()
    }
    for count in (1, 2):
        report = fitting.worker(spec, tmp_path / f"fit-{count}", count=count)
        assert report["status"] == "passed", report
        assert report["actual_count"] == count and report["peak_host_bytes"] == 4096
        assert report["peak_anonymous_bytes"] == 2048
        assert (
            report["memory"]["details"]["method"] == "sampled_cgroup_v2_memory_current"
        )
    after = {
        p.relative_to(tmp_path / "index"): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in (tmp_path / "index").rglob("*")
        if p.is_file()
    }
    assert before == {name: after[name] for name in before}
    # Materialization can populate column-statistics caches without reingestion.
    assert all(name.parts[0] == "col_stats" for name in after.keys() - before.keys())
