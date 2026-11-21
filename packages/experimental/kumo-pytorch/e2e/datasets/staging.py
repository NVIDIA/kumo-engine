# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify pinned local inputs, or stage missing objects with the AWS CLI."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def _check(path: Path, pin: dict) -> None:
    if path.stat().st_size != pin["size"]:
        raise ValueError(f"Size mismatch: {pin['key']}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != pin["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {pin['key']}")


def _stage(root: Path, spec: dict, pins: dict, download: bool) -> dict[str, Path]:
    files = {}
    for name, pin in pins.items():
        key = pin["key"]
        path = root / key
        if Path(key).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Object key outside dataset root: {key}")
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=path.parent) as tmp:
                pending = Path(tmp) / "download"
                subprocess.run(
                    [
                        "aws",
                        "s3api",
                        "get-object",
                        "--region",
                        spec["region"],
                        "--expected-bucket-owner",
                        spec["expected_bucket_owner"],
                        "--bucket",
                        spec["bucket"],
                        "--key",
                        key,
                        str(pending),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                _check(pending, pin)
                pending.replace(path)
        else:
            _check(path, pin)
        files[name] = path
    return files


def verify(root: Path, manifest: Path, *, download: bool = False) -> dict[str, Path]:
    """Return verified files by logical name. By default, only read local files."""
    spec = json.loads(manifest.read_text())
    if spec["version"] != 1 or not spec["objects"]:
        raise ValueError(f"Invalid dataset manifest: {manifest}")
    if shared := spec.get("include_objects_from"):
        shared_objects = json.loads(manifest.with_name(shared).read_text())["objects"]
        spec["objects"] = shared_objects | spec["objects"]
    root = root.resolve()
    files = _stage(root, spec, spec["objects"], download)
    if prefix := spec.get("table_source_prefix"):
        marker = json.loads(files["validation"].read_text())
        if not (
            marker["ok"]
            and marker["checksums_verified"]
            and marker["violations"] == 0
            and marker["manifest_sha256"] == spec["objects"]["manifest"]["sha256"]
        ):
            raise ValueError("Validation marker does not attest the pinned source manifest")
        source = json.loads(files["manifest"].read_text())
        for table, metadata in source["tables"].items():
            pins = {
                f"{table}/{Path(f['path']).name}": {
                    "key": f"{prefix}/{f['path']}",
                    "size": f["bytes"],
                    "sha256": f["sha256"],
                }
                for f in metadata["files"]
            }
            shards = _stage(root, spec, pins, download)
            directory = files["manifest"].parent / table
            if set(directory.rglob("*.parquet")) != set(shards.values()):
                raise ValueError(f"Parquet files differ from pinned inventory for {table}")
            files.update(shards)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for manifest in args.manifests:
        verify(args.root, manifest, download=True)
