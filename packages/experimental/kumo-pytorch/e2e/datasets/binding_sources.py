# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the verified source tables required for a selected full-population replay."""

import argparse
import json
from pathlib import Path, PurePosixPath

from e2e.datasets.staging import _stage, admit, verify

PINS = Path(__file__).parent / "pins"
SYNTHETIC = {"S01": "weekly-account", "S02": "weekly-customer", "S04": "ltv"}
SYNTHETIC_TABLES = (
    "accounts",
    "customers",
    "card_txns",
    "transfers",
    "logins",
    "profile_changes",
    "disputes",
    "lifelong_events",
    "plan_events",
)


def _sources(binding):
    if binding in SYNTHETIC:
        return ("synthetic-medium", "synthetic-" + SYNTHETIC[binding]), SYNTHETIC_TABLES
    if binding in ("M03", "M04"):
        return (("mbd-m04", "mbd-m03") if binding == "M03" else ("mbd-m04",)), (
            "client_split",
            "detail/trx",
        )
    if binding in ("E01", "E02", "E03", "E04", "E05", "L01"):
        return ("aml-l01" if binding == "L01" else "aml-e01",), ()
    raise ValueError(f"Unknown full-population binding: {binding}")


def full_source(binding, root=None, *, download=False):
    names, tables = _sources(binding)
    manifests = [PINS / (name + ".json") for name in names]
    specs = [json.loads(p.read_text()) for p in manifests]
    owner = specs[0]
    source = PurePosixPath(owner["objects"]["source_manifest"]["key"]).parent
    if root is None:
        bucket = owner["bucket"]
        keys = {obj["key"] for spec in specs for obj in spec["objects"].values()}
        prefixes = tuple(str(source / table) + "/" for table in tables)
        keys = {key for key in keys if not key.startswith(prefixes)} | {p + "*" for p in prefixes}
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": [f"arn:aws:s3:::{bucket}/{key}" for key in sorted(keys)],
                }
            ],
        }
    # M03 already includes M04's source pins.
    selected = manifests[-1:] if binding == "M03" else manifests
    if download:
        root.mkdir(parents=True, exist_ok=True)
        admit(root, selected, minimum_free_gib=128 if tables else 0)
    files = {}
    for manifest in selected:
        files.update(verify(root, manifest, download=download))
    if not tables:
        return files
    published = json.loads(files["source_manifest"].read_text())["files"]
    if binding in SYNTHETIC:
        entries = ((name, digest, None) for name, digest in published.items())
    else:
        entries = ((row["path"], row["sha256"], row["bytes"]) for row in published)
    shards = {}
    found = set()
    folds = {table: set() for table in tables}
    for name, digest, size in entries:
        table = next((t for t in tables if name.startswith(t + "/")), None)
        if table is None or not name.endswith(".parquet"):
            continue
        path = PurePosixPath(name)
        if ".." in path.parts or path.is_absolute():
            raise ValueError(f"Invalid source shard location: {name}")
        if binding in ("M03", "M04"):
            fold = path.parent.name
            if fold == "fold=-1":
                continue
            if fold not in {f"fold={n}" for n in range(5)}:
                raise ValueError(f"Invalid registered MBD fold: {name}")
            folds[table].add(fold)
        found.add(table)
        shards[name] = {"key": str(source / path), "size": size, "sha256": digest}
    if found != set(tables):
        raise ValueError(f"Missing published tables: {set(tables) - found}")
    if binding.startswith("M") and any(len(parts) != 5 for parts in folds.values()):
        raise ValueError("Expected five registered folds in each MBD table")
    return files | _stage(root.resolve(), owner, shards, download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--policy", action="store_true")
    group.add_argument("--root", type=Path)
    args = parser.parse_args()
    result = full_source(args.binding, args.root, download=bool(args.root))
    if args.policy:
        print(json.dumps(result, separators=(",", ":")))
