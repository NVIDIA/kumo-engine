# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch and time pinned dataset/anchor files without importing Arena or Kumo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "datasets" / "inputs.json"


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    if PurePosixPath(value).is_absolute() or any(
        p in ("", ".", "..") for p in value.split("/")
    ):
        raise ValueError(f"Invalid relative path: {value!r}")
    return value


def safe_path(root: Path, relative: str) -> Path:
    path = root.absolute() / relative_path(relative)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink in dataset path: {path}")
    return path


def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("datasets"):
        raise ValueError("Expected a version 1 dataset manifest")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", manifest["bucket"]):
        raise ValueError("Invalid S3 bucket")
    if not re.fullmatch(r"[0-9]{12}", manifest["expected_bucket_owner"]):
        raise ValueError("Invalid expected bucket owner")
    identities = {}
    for entry in manifest["datasets"].values():
        if not entry["files"]:
            raise ValueError("Empty file inventory")
        keys = set()
        for item in entry["files"]:
            key = relative_path(item["key"])
            if key in keys:
                raise ValueError(f"Duplicate key: {key}")
            keys.add(key)
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("Expected lowercase SHA-256")
            size = item["bytes"]
            if type(size) is not int or size < 0:
                raise ValueError("Expected nonnegative file size")
            previous = identities.setdefault(item["key"], item)
            if previous != item:
                raise ValueError(f"Conflicting identity: {item['key']}")
    return manifest


def verify_file(path: Path, item: dict) -> int:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    size = path.stat().st_size
    if size != item["bytes"]:
        raise ValueError(f"Byte count differs: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"SHA-256 differs: {path}")
    return size


def acquire(
    item: dict,
    manifest: dict,
    root: Path,
    *,
    fetch: bool,
    source_dir: Path | None = None,
    client=None,
) -> dict:
    """Publish only verified files; existing corrupt files fail without replacement."""
    row = dict(
        path=item["key"],
        sha256=item["sha256"],
        status="running",
        transfer_seconds=0.0,
        verification_seconds=0.0,
        transferred_bytes=0,
    )
    started = time.perf_counter()
    temporary = None
    phase = "verify"
    try:
        destination = safe_path(root, item["key"])
        if destination.exists() or not fetch:
            clock = time.perf_counter()
            try:
                row["bytes"] = verify_file(destination, item)
            finally:
                row["verification_seconds"] = time.perf_counter() - clock
            row["status"] = "verified_existing"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".download-", delete=False
            ) as f:
                temporary = Path(f.name)
            phase = "transfer"
            clock = time.perf_counter()
            try:
                if source_dir is not None:
                    shutil.copyfile(safe_path(source_dir, item["key"]), temporary)
                else:
                    if client is None:
                        raise ValueError("S3 fetch requires an authenticated client")
                    # No bucket listing, writes, extraction or mutable 'latest' lookup.
                    response = client.get_object(
                        Bucket=manifest["bucket"],
                        Key=item["key"],
                        ExpectedBucketOwner=manifest["expected_bucket_owner"],
                    )
                    body = response["Body"]
                    try:
                        with temporary.open("wb") as out:
                            shutil.copyfileobj(body, out, length=1024 * 1024)
                    finally:
                        body.close()
            finally:
                row["transfer_seconds"] = time.perf_counter() - clock
                row["transferred_bytes"] = temporary.stat().st_size
            phase = "verify"
            clock = time.perf_counter()
            try:
                row["bytes"] = verify_file(temporary, item)
            finally:
                row["verification_seconds"] = time.perf_counter() - clock
            phase = "publish"
            try:
                os.link(temporary, destination)
            except FileExistsError:
                verify_file(safe_path(root, item["key"]), item)
            row["status"] = "downloaded"
    except Exception as exc:
        row.update(status="failed", phase=phase, error=f"{type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        row["elapsed_seconds"] = time.perf_counter() - started
    return row


def inventory(manifest: dict, dataset: str) -> dict:
    entry = manifest["datasets"][dataset]
    return dict(
        dataset=dataset,
        dataset_id=entry["dataset_id"],
        dataset_version=entry["dataset_version"],
        population=entry["population"],
        anchor_versions=entry["anchor_versions"],
        files=len(entry["files"]),
        source_bytes=entry["source_bytes"],
        coverage="file_integrity_only",
    )


def execute(
    command: str,
    manifest: dict,
    dataset: str,
    root: Path,
    *,
    source_dir: Path | None = None,
    client=None,
    jobs: int = 4,
) -> dict:
    started = time.perf_counter()
    files = manifest["datasets"][dataset]["files"]

    def one(item):
        return acquire(
            item,
            manifest,
            root,
            fetch=command == "fetch",
            source_dir=source_dir,
            client=client,
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, files))
    elapsed = time.perf_counter() - started
    transferred = sum(r["transferred_bytes"] for r in rows)
    return {
        **inventory(manifest, dataset),
        "command": command,
        "jobs": jobs,
        "status": "failed" if any(r["status"] == "failed" for r in rows) else "passed",
        "elapsed_seconds": elapsed,
        "transferred_bytes": transferred,
        "downloaded_files": sum(r["status"] == "downloaded" for r in rows),
        "cached_files": sum(r["status"] == "verified_existing" for r in rows),
        "failed_files": sum(r["status"] == "failed" for r in rows),
        "transfer_seconds_sum": sum(r["transfer_seconds"] for r in rows),
        "verification_seconds_sum": sum(r["verification_seconds"] for r in rows),
        "effective_bytes_per_second": transferred / elapsed if elapsed else None,
        "timing_scope": "Wall time includes transfers, hashing and publication. Per-file sums overlap across workers.",
        "results": rows,
    }


def s3_client(profile: str | None, region: str, jobs: int):
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=profile).client(
        "s3",
        region_name=region,
        config=Config(
            max_pool_connections=jobs,
            connect_timeout=15,
            read_timeout=60,
            retries={"mode": "standard", "max_attempts": 3},
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "fetch", "verify"))
    parser.add_argument("--dataset", default="hi-small")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument(
        "--source-dir", type=Path, help="Local mirror of S3 keys; no AWS needed"
    )
    parser.add_argument(
        "--profile", help="AWS profile; otherwise use the normal credential chain"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--report", type=Path, help="Full JSON report; stdout prints a summary"
    )
    args = parser.parse_args(argv)
    manifest = load_manifest()
    if args.dataset not in manifest["datasets"]:
        parser.error(f"Unknown dataset; choose {', '.join(manifest['datasets'])}")
    if not 1 <= args.jobs <= 32:
        parser.error("--jobs must be between 1 and 32")
    if args.command == "list":
        print(
            json.dumps([inventory(manifest, d) for d in manifest["datasets"]], indent=2)
        )
        return 0
    if args.report:
        roots = [args.data_dir] + ([args.source_dir] if args.source_dir else [])
        protected = {MANIFEST.resolve()} | {
            safe_path(root, f["key"]).resolve()
            for root in roots
            for v in manifest["datasets"].values()
            for f in v["files"]
        }
        if args.report.resolve() in protected:
            parser.error("Report would overwrite an input or manifest")
    try:
        client = None
        if args.command == "fetch" and args.source_dir is None:
            client = s3_client(args.profile, manifest["region"], args.jobs)
        report = execute(
            args.command,
            manifest,
            args.dataset,
            args.data_dir,
            source_dir=args.source_dir,
            client=client,
            jobs=args.jobs,
        )
    except Exception as exc:
        report = {
            **inventory(manifest, args.dataset),
            "status": "failed",
            "phase": "setup",
            "error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }
    report["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for row in report["results"]:
        if row["status"] == "failed":
            print(f"FAILED {row['path']}: {row['error']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
