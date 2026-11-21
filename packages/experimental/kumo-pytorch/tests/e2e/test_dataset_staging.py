# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from e2e.datasets.staging import verify


def _pins(directory, name, key, data):
    pin = {"key": key, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    spec = {
        "version": 1,
        "bucket": "test-bucket",
        "region": "us-west-2",
        "expected_bucket_owner": "123456789012",
        "objects": {name: pin},
    }
    manifest = directory / f"{name}.json"
    manifest.write_text(json.dumps(spec))
    return manifest


def test_verify_and_reuse_pinned_download(tmp_path):
    data = b"input"
    key = "source/table"
    manifest = _pins(tmp_path, "table", key, data)
    root = tmp_path / "staged"
    path = root / key
    with patch("e2e.datasets.staging.subprocess.run") as get_object:
        with pytest.raises(FileNotFoundError):
            verify(root, manifest)
        get_object.assert_not_called()

        def download(command, **options):
            assert command[:3] == ["aws", "s3api", "get-object"]
            assert command[command.index("--expected-bucket-owner") + 1] == "123456789012"
            assert command[command.index("--key") + 1] == key
            Path(command[-1]).write_bytes(payload)

        get_object.side_effect = download
        payload = b"wrong"
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify(root, manifest, download=True)
        assert not path.exists()

        payload = data
        assert verify(root, manifest, download=True) == {"table": path}
        get_object.reset_mock()
        assert verify(root, manifest, download=True) == {"table": path}
        path.write_bytes(b"truncated")
        with pytest.raises(ValueError, match="Size mismatch"):
            verify(root, manifest, download=True)
        get_object.assert_not_called()


def test_cli_verifies_every_manifest(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    manifests = []
    for name in ("first", "second"):
        data = name.encode()
        (root / name).write_bytes(data)
        manifests.append(_pins(tmp_path, name, name, data))
    command = [sys.executable, "-m", "e2e.datasets.staging", *manifests, "--root", root]
    subprocess.run(command, check=True)
    (root / "second").write_bytes(b"broken")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode and "SHA-256 mismatch: second" in failed.stderr
