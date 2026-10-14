# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(("script", "arguments", "required"), [
    ("scripts/run_validation.sh", ["example trainingjob-example"], "--bucket"),
    ("tools/validate_ttgen.sh", ["example", "trainingjob-example"], "--bucket"),
    ("diskgraph-datagenerator/scripts/cloud_generate_dataset.sh", ["example.json"],
     "DISKGRAPH_CLOUD_BUCKET"),
])
def test_validation_requires_bucket_before_starting(script, arguments, required, tmp_path):
    marker = tmp_path / "started"
    for command in ("aws", "cargo", "curl"):
        stub = tmp_path / command
        stub.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / script), *arguments], cwd=tmp_path,
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, capture_output=True, text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert required in result.stdout + result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(("script", "jobs"), [
    ("diskgraph-datagenerator/scripts/download_ingested_graph.sh",
     ["example", "trainingjob-example"]),
    ("scripts/run_validation.sh", ["example trainingjob-example"]),
])
def test_wrapper_preserves_storage_arguments(script, jobs, tmp_path):
    output = tmp_path / "arguments"
    stub = tmp_path / "cargo"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{output}"\n')
    stub.chmod(0o755)
    subprocess.run(
        ["bash", str(ROOT / script),
         "--bucket", "example-bucket", "--customer-prefix", "example-prefix",
         *jobs],
        cwd=tmp_path, env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        check=True, timeout=10,
    )
    args = output.read_text().splitlines()
    assert args[args.index("--bucket") + 1] == "example-bucket"
    assert args[args.index("--customer-prefix") + 1] == "example-prefix"
    assert args[-len(jobs):] == jobs


def test_benchmark_paths_use_selected_data_root(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"tables": [{"source": {
        "type": "parquet", "paths": ["example/items/part-00000.parquet"],
    }}]}))
    output = tmp_path / "output.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/benchmarks/rewrite_config.py"), str(source),
        "--data-root", str(tmp_path / "data"), "--output", str(output),
    ], check=True)
    assert json.loads(output.read_text())["tables"][0]["source"]["paths"] == [
        str(tmp_path / "data/example/items/part-00000.parquet"),
    ]


@pytest.mark.parametrize("config", sorted(
    (ROOT / "scripts/benchmarks/benchmark_configs").glob("*.json")), ids=lambda p: p.stem)
def test_benchmark_configuration_has_no_fixed_storage(config):
    for table in json.loads(config.read_text())["tables"]:
        for path in table["source"]["paths"]:
            assert not path.startswith("/") and "://" not in path


def test_cloud_generator_uses_chosen_bucket(tmp_path):
    output = tmp_path / "arguments"
    for name, body in {
        "curl": "exit 0",
        "cargo": f'printf "%s\\n" "$@" > "{output}"',
    }.items():
        stub = tmp_path / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o755)
    subprocess.run([
        "bash", str(ROOT / "diskgraph-datagenerator/scripts/cloud_generate_dataset.sh"),
        "example.json",
    ], cwd=tmp_path, env={
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "DISKGRAPH_CLOUD_BUCKET": "example-bucket",
    }, check=True, timeout=10)
    args = output.read_text().splitlines()
    assert args[args.index("--output") + 1] == (
        "gs://example-bucket/diskgraph-datagenerator/example"
    )
