# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.compact_graph_benchmarks import NAMES, _conditions, compact


def test_compact_manifest_preserves_latency_and_hash_without_inventing_old_peaks(tmp_path):
    raw = {
        "summary": [
            {
                "arm": "gnn_training",
                "sample_count": 2,
                "before_median_ms": {"cycle_ms": 2.0},
                "after_median_ms": {"cycle_ms": 1.0},
            }
        ],
        "runs": [
            {
                "source_revision": revision,
                "source_dirty": False,
                "results": [{"arm": "gnn_training", "final_embeddings": [[1.25], [2.5]]}],
            }
            for revision in ("85f47bb-original", "1e73e84-original")
        ],
    }
    path = tmp_path / "graph_execution_views_l4.json"
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    result = compact(path, "https://example.invalid/private-artifacts")
    assert path.read_bytes() == before
    assert result["raw_artifact"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert "runs" not in result and "final_embeddings" not in json.dumps(result)
    assert "samples" not in result["conditions"][0]["workloads"][0]
    assert result["conditions"][0]["workloads"][0]["root_count"] == 2
    row = result["summary"][0]
    assert row["before_median_ms"] == raw["summary"][0]["before_median_ms"]
    assert row["throughput"]["before"]["value"] == 1000.0
    assert row["throughput"]["after"]["value"] == 2000.0
    assert row["throughput"]["before"]["unit"] == "roots/s"
    assert row["peak_memory"]["before"]["allocated_mib"] is None
    assert "not measured" in row["peak_memory"]["after"]["status"]


def test_compaction_refuses_unmatched_resource_provenance(tmp_path):
    raw = {
        "sources": {"baseline": {"revision": "actual-source"}},
        "summary": [
            {
                "arm": "gnn_training",
                "effective_settings": {"root_count": 2},
                "stages": {"baseline": {"cycle_ms": {"median_of_run_medians": 2.0}}},
            }
        ],
        "runs": [],
        "preflight": [],
    }
    path = tmp_path / "graph_constructor_checks_l4.json"
    path.write_text(json.dumps(raw))
    resources = tmp_path / "resources.json"
    resources.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "source_revision": "wrong-source",
                        "arm": "gnn_training",
                        "peak_memory": {"allocated_mib": 42},
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Expected one measured resource summary"):
        compact(path, "https://example.invalid/raw", resources, "https://example.invalid/resources")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("gpu", "different GPU"),
        ("config", {"seed": 1}),
        ("driver_sha256", "different-driver"),
        ("source_dirty", True),
        ("precision", "bf16"),
        ("warmup", 10),
        ("steps", 30),
        ("results", [{"arm": "gnn_training", "samples": 3}]),
    ],
)
def test_conditions_reject_mixed_repetitions(field, replacement):
    first = {
        "source_revision": "same-source",
        "source_dirty": False,
        "gpu": "original GPU",
        "config": {"seed": 0},
        "driver_sha256": "original-driver",
        "precision": "fp32",
        "warmup": 5,
        "steps": 20,
        "results": [{"arm": "gnn_training", "samples": 2}],
    }
    second = copy.deepcopy(first)
    second[field] = replacement
    with pytest.raises(ValueError, match="Inconsistent benchmark conditions.*same-source"):
        _conditions([first, second])


def test_conditions_consolidate_identical_repetitions_without_grouping_timings():
    first = {
        "source_revision": "same-source",
        "results": [{"arm": "validation", "samples": {"cycle_ms": [1.0, 2.0]}}],
    }
    second = copy.deepcopy(first)
    second["results"][0]["samples"]["cycle_ms"] = [3.0, 4.0]
    original = copy.deepcopy([first, second])
    result = _conditions([first, second])
    assert result == [
        {"source_revision": "same-source", "workloads": [{"arm": "validation"}], "repetitions": 2}
    ]
    assert [first, second] == original


@pytest.mark.parametrize("collision", ["raw_dir", "resource", "symlink", "hardlink", "outputs"])
def test_cli_rejects_all_output_collisions_before_writing(tmp_path, collision):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for name in NAMES:
        (raw_dir / f"{name}.json").write_bytes(b"untouched raw input")
        (output_dir / f"{name}.json").write_bytes(b"untouched existing output")
    extra_args = []
    if collision == "raw_dir":
        output_dir = raw_dir
    elif collision == "resource":
        resource = output_dir / "resources.json"
        resource.write_bytes(b"untouched resource input")
        extra_args = ["--resources", str(resource), "--resources-url", "https://example.invalid/r"]
    elif collision == "outputs":
        resource = tmp_path / f"{NAMES[-1]}.json"
        resource.write_bytes(b"untouched resource input")
        extra_args = ["--resources", str(resource), "--resources-url", "https://example.invalid/r"]
    else:
        destination = output_dir / f"{NAMES[-1]}.json"
        destination.unlink()
        source = raw_dir / f"{NAMES[0]}.json"
        if collision == "symlink":
            destination.symlink_to(source)
        else:
            destination.hardlink_to(source)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    script = Path(__file__).resolve().parents[2] / "scripts/compact_graph_benchmarks.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--artifact-base-url",
            "https://example.invalid/raw",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "would overwrite input or output" in result.stderr
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == before
