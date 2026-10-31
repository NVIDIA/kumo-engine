#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate resolved package graphs for the shipped artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "third_party" / "artifact-manifest.json"
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
)
ARTIFACTS = (
    {
        "id": "diskgraph-gateway",
        "label": "bin/diskgraph-gateway",
        "package": "diskgraph-gateway",
        "manifest": "diskgraph-gateway/Cargo.toml",
        "features": [],
    },
    {
        "id": "diskgraph-extension",
        "label": "diskgraph._c.abi3.so",
        "package": "diskgraph-pybind",
        "manifest": "diskgraph-pybind/Cargo.toml",
        "features": ["extension-module", "pyo3/abi3-py311"],
    },
    {
        "id": "diskgraph-ingest",
        "label": "bin/ingest",
        "package": "diskgraph-ingest",
        "manifest": "diskgraph-ingest/Cargo.toml",
        "features": [],
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cargo_metadata(artifact: dict[str, Any], target: str,
                   root: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
        "--filter-platform",
        target,
        "--manifest-path",
        artifact["manifest"],
    ]
    if artifact["features"]:
        command.extend(("--features", ",".join(artifact["features"])))
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def cargo_tree_packages(artifact: dict[str, Any], target: str,
                        root: Path) -> set[tuple[str, str]]:
    command = [
        "cargo",
        "tree",
        "--locked",
        "--offline",
        "--target",
        target,
        "--package",
        artifact["package"],
        "--edges",
        "normal",
        "--prefix",
        "none",
        "--format",
        "{p}",
    ]
    if artifact["features"]:
        command.extend(("--features", ",".join(artifact["features"])))
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    packages: set[tuple[str, str]] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[1].startswith("v"):
            raise ValueError(f"cannot parse cargo tree package line: {line!r}")
        packages.add((fields[0], fields[1][1:]))
    return packages


def stable_source(package: dict[str, Any], root: Path) -> tuple[str, str]:
    source = package["source"]
    if source is not None:
        return "registry-or-git", source
    manifest = Path(package["manifest_path"])
    relative = manifest.resolve().relative_to(root.resolve()).as_posix()
    if relative.startswith("diskgraph-ptrhash/"):
        return "vendored-third-party", relative
    return "workspace-first-party", relative


def package_record(package: dict[str, Any], root: Path) -> dict[str, Any]:
    source_kind, source = stable_source(package, root)
    return {
        "name": package["name"],
        "version": package["version"],
        "license": package["license"],
        "source_kind": source_kind,
        "source": source,
    }


def generate(root: Path = ROOT) -> bytes:
    root = root.resolve()
    artifacts = []
    for definition in ARTIFACTS:
        artifact_packages = None
        for target in TARGETS:
            metadata = cargo_metadata(definition, target, root)
            by_name_version: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for package in metadata["packages"]:
                key = (package["name"], package["version"])
                by_name_version.setdefault(key, []).append(package)
            resolved_packages = []
            for key in cargo_tree_packages(definition, target, root):
                candidates = by_name_version.get(key, [])
                if len(candidates) != 1:
                    raise ValueError("expected one metadata package for "
                                     f"{key}, found {len(candidates)}")
                resolved_packages.append(candidates[0])
            package_records = sorted(
                (package_record(package, root)
                 for package in resolved_packages),
                key=lambda item:
                (item["name"].lower(), item["version"], item["source"]),
            )
            if artifact_packages is None:
                artifact_packages = package_records
            elif artifact_packages != package_records:
                raise ValueError(
                    f"{definition['id']} dependency graph differs for "
                    f"{target}; represent the target-specific graph before "
                    "shipping it")
        artifacts.append({
            "id": definition["id"],
            "label": definition["label"],
            "root": definition["package"],
            "targets": list(TARGETS),
            "default_features": True,
            "features": definition["features"],
            "dependency_kinds": ["normal"],
            "packages": artifact_packages,
        })
    document = {
        "schema": 1,
        "cargo_lock_sha256": sha256(root / "Cargo.lock"),
        "artifacts": artifacts,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help=("resolve a DiskGraph checkout other than this script's "
              "repository"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output",
                      type=Path,
                      help="write the resolved manifest")
    mode.add_argument("--check",
                      type=Path,
                      help="compare against a committed manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generated = generate(args.repo_root)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(generated)
            print(f"wrote {args.output}")
            return 0
        actual = args.check.read_bytes() if args.check.is_file() else b""
        if actual != generated:
            script = Path(__file__).resolve().relative_to(ROOT)
            print(
                f"{args.check} is stale; regenerate it with "
                f"{script} --output {args.check}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.check}: current")
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"artifact manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
