# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from e2e.datasets.staging import verify


def test_stage_pinned_datasets(tmp_path, monkeypatch):
    root = tmp_path / "staged"
    payload = None

    def fetch(command, **kwargs):
        assert payload is not None, "local-only verification attempted a download"
        assert command[:3] == ["aws", "s3api", "get-object"]
        assert command[command.index("--expected-bucket-owner") + 1] == "123456789012"
        assert command[command.index("--key") + 1] == "one"
        Path(command[-1]).write_bytes(payload)

    manifests = []
    for key in ("one", "two"):
        pin = {"key": key, "size": 2, "sha256": hashlib.sha256(b"ok").hexdigest()}
        spec = {
            "version": 1,
            "bucket": "test-bucket",
            "region": "us-west-2",
            "expected_bucket_owner": "123456789012",
            "objects": {"table": pin},
        }
        if key == "two":
            spec.update(include_objects_from="one.json", objects={"second": pin})
        manifest = tmp_path / f"{key}.json"
        manifest.write_text(json.dumps(spec))
        manifests.append(manifest)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", fetch)
        with pytest.raises(FileNotFoundError):
            verify(root, manifests[0])
        payload = b"no"
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify(root, manifests[0], download=True)
        assert not (root / "one").exists()
        payload = b"ok"
        assert verify(root, manifests[0], download=True) == {"table": root / "one"}
    (root / "two").write_bytes(payload)
    assert verify(root, manifests[1]) == {"table": root / "one", "second": root / "two"}
    command = [sys.executable, "-m", "e2e.datasets.staging", *manifests, "--root", root]
    # Fully cached inputs work with no AWS executable.
    monkeypatch.setenv("PATH", "")
    subprocess.run(command, check=True)
    for bad, message in ((b"no", "SHA-256"), (b"longer", "Size")):
        (root / "two").write_bytes(bad)
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode and f"{message} mismatch: two" in failed.stderr


def test_verify_source_tables(tmp_path):
    def write(key, data):
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"key": key, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    shard = write("data/sample/events/part.parquet", b"events")
    files = [
        {"path": "sample/events/part.parquet", "bytes": shard["size"], "sha256": shard["sha256"]}
    ]
    source = {"tables": {"events": {"files": files}}, "ground_truth": {"hidden": {}}}
    manifest = write("data/sample/manifest.json", json.dumps(source).encode())
    marker = dict(
        ok=True, violations=0, checksums_verified=True, manifest_sha256=manifest["sha256"]
    )
    pin = tmp_path / "pin.json"

    def save_marker():
        validation = write("data/sample/_VALIDATED.json", json.dumps(marker).encode())
        objects = {"manifest": manifest, "validation": validation}
        pin.write_text(json.dumps(dict(version=1, table_source_prefix="data", objects=objects)))

    save_marker()
    assert verify(tmp_path, pin)["events/part.parquet"] == tmp_path / shard["key"]
    write("data/sample/events/extra.parquet", b"extra")
    with pytest.raises(ValueError, match="inventory for events"):
        verify(tmp_path, pin)
    marker["manifest_sha256"] = "invalid"
    save_marker()
    with pytest.raises(ValueError, match="Validation marker"):
        verify(tmp_path, pin)
