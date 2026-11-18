# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny native fitting checks; simulated collectors are not Linux memory evidence."""

import json
from test.kumo_pytorch.aml_fitting_memory import make_spec
from test.kumo_pytorch.benchmarking import fitting_memory as fitting
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.population_anchors import project_population

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .fixtures import CUTOFF, build_fixture
from .test_aml import CASES
from .test_aml_inputs import anchor_file
from .test_population import source_anchors
from .test_run_aml import mapped  # noqa: F401 - pytest fixture


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
def test_real_anchor_spec_is_training_only_and_nested(
    mapped, selection, family, tmp_path
):
    dataset, identities, selections = mapped
    source = source_anchors(selections, selection, tmp_path)
    table = pq.read_table(source)
    table = table.set_column(
        table.schema.get_field_index("involvement"),
        "involvement",
        table["involvement"].cast(pa.int32()),
    )
    pq.write_table(table, source)
    spec_path = make_spec(
        dataset,
        identities,
        source,
        selection,
        family,
        tmp_path / "inputs",
        counts=(1, 2),
        capacity=5,
    )
    spec = json.loads(spec_path.read_text())
    small, large = [
        pq.read_table(spec["anchors"][str(n)]["path"]).to_pylist() for n in (1, 2)
    ]
    assert small == large[:1] and all(row["split"] == "train" for row in large)
    assert (
        spec["encoder_cutoff_seconds"]
        == spec["anchors"]["2"]["population"]["encoder_cutoff_seconds"]
    )
    assert spec["threads"] == 1 and spec["fit_split"] == "train"


def test_insufficient_training_population_fails_without_repeating_rows(tmp_path):
    source = anchor_file(tmp_path)
    with pytest.raises(ValueError, match="Insufficient"):
        project_population(
            source,
            tmp_path / "projected.parquet",
            {"leading-0007": 42},
            native=True,
            graph=False,
            limit=400000,
            splits=("train",),
        )


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


def test_missing_isolation_is_a_failure_not_a_local_memory_fallback(
    tmp_path, monkeypatch
):
    spec = tiny_spec(tmp_path)
    monkeypatch.delenv("KUMO_BENCHMARK_CGROUP", raising=False)
    report = fitting.worker(spec, tmp_path / "fit", count=1)
    assert report["status"] == "failed" and "dedicated Linux cgroup" in report["error"]
