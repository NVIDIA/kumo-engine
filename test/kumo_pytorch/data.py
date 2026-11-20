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
        location = {
            key: entry.get(key, manifest[key])
            for key in ("bucket", "expected_bucket_owner")
        }
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", location["bucket"]):
            raise ValueError("Invalid dataset S3 bucket")
        if not re.fullmatch(r"[0-9]{12}", location["expected_bucket_owner"]):
            raise ValueError("Invalid dataset bucket owner")
        if "inventory" in manifest:
            if type(entry["file_count"]) is not int or entry["file_count"] < 1:
                raise ValueError("Expected a positive file count")
            continue
        shared = entry.get("file_sets", ())
        unknown = set(shared) - manifest.get("file_sets", {}).keys()
        if unknown:
            raise ValueError(f"Unknown file sets: {sorted(unknown)}")
        entry["files"] = [
            item for name in shared for item in manifest["file_sets"][name]
        ] + entry["files"]
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
            previous = identities.setdefault(item["key"], (location, item))
            if previous != (location, item):
                raise ValueError(f"Conflicting identity: {item['key']}")
    return manifest


def resolve_inventory(manifest, root, *, fetch=False, source_dir=None, client=None):
    """Load the checksum-pinned inventory using the same verified download path."""
    if "inventory" not in manifest:
        return manifest
    pin = manifest["inventory"]
    result = acquire(pin, pin, root, fetch=fetch, source_dir=source_dir, client=client)
    if result["status"] == "failed":
        raise ValueError(f"Pinned input inventory: {result['error']}")
    resolved = load_manifest(safe_path(root, pin["key"]))
    if "inventory" in resolved:
        raise ValueError("Nested inventory pointers are not supported")
    # Keep offline catalog metadata consistent with the pinned file inventory.
    catalog = {k: v for k, v in resolved.items() if k != "file_sets"}
    catalog["datasets"] = {
        name: {
            **{k: v for k, v in entry.items() if k not in ("files", "file_sets")},
            "file_count": len(entry["files"]),
        }
        for name, entry in resolved["datasets"].items()
    }
    if catalog != {k: v for k, v in manifest.items() if k != "inventory"}:
        raise ValueError("Dataset catalog differs from the pinned input inventory")
    return resolved


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
        files=len(entry["files"]) if "files" in entry else entry["file_count"],
        source_bytes=entry["source_bytes"],
        coverage="file_integrity_only",
    )


def storage_preflight(root: Path, min_free_gib: int) -> dict:
    """Measure the filesystem used by both downloads and campaign output."""
    if type(min_free_gib) is not int or min_free_gib < 0:
        raise ValueError("min_free_gib must be a nonnegative integer")
    total, used, free = shutil.disk_usage(root)
    required = min_free_gib * 1024**3
    return {
        "status": "passed" if free >= required else "failed",
        "path": str(root.resolve()),
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "required_free_bytes": required,
        "scope": "Admission floor; actual prepared-table and index growth is not yet measured.",
    }


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
    manifest = resolve_inventory(
        manifest, root, fetch=command == "fetch", source_dir=source_dir, client=client
    )
    entry = manifest["datasets"][dataset]
    files = entry["files"]
    location = {
        **manifest,
        **{k: entry[k] for k in ("bucket", "expected_bucket_owner") if k in entry},
    }

    def one(item):
        return acquire(
            item,
            location,
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
    pin = manifest.get("inventory")
    if args.report:
        roots = [args.data_dir] + ([args.source_dir] if args.source_dir else [])
        if args.report.resolve() == MANIFEST.resolve() or any(
            args.report.resolve().is_relative_to(root.resolve()) for root in roots
        ):
            parser.error("Keep reports outside input directories and manifests")
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
    if pin:
        report["inventory_sha256"] = pin["sha256"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for row in report["results"]:
        if row["status"] == "failed":
            print(f"FAILED {row['path']}: {row['error']}")
    return 0 if report["status"] == "passed" else 1


def verify_dataset(data_dir, binding):
    """Verify the selected inventory before any conversion or anchor selection."""
    from .bindings import dataset_definition

    definition = dataset_definition(binding)
    manifest = resolve_inventory(load_manifest(), Path(data_dir))
    entry = manifest["datasets"][definition["inventory"]]
    if any(
        entry[key] != binding["selector"][key]
        for key in ("dataset_id", "dataset_version", "population")
    ):
        raise ValueError("Pinned inventory identity differs from the selected dataset")
    files = {item["key"]: item for item in entry["files"]}
    for key, item in files.items():
        verify_file(safe_path(Path(data_dir), key), item)
    # Optional source-format metadata supplements the universal file pins.
    manifest_format = definition.get("anchor_manifest_format")
    if manifest_format not in (None, "arena"):
        raise ValueError(f"Unknown anchor manifest format: {manifest_format}")
    for version in entry.get("anchor_versions", ()) if manifest_format else ():
        import pyarrow.parquet as pq

        key = definition["anchor_path"].format(
            population=entry["population"], version=version
        )
        manifest = json.loads(
            safe_path(
                Path(data_dir), str(PurePosixPath(key).with_name("manifest.json"))
            ).read_text()
        )
        expected = {
            "dataset_id": entry["dataset_id"],
            "dataset_version": entry["dataset_version"],
            "task_id": entry["population"],
            "anchor_version": version,
        }
        if any(manifest["definition"][k] != v for k, v in expected.items()):
            raise ValueError("Anchor manifest identity differs from the binding")
        item = files[key]
        if manifest["files"] != [
            {
                "path": "anchors.parquet",
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        ]:
            raise ValueError("Anchor payload differs from the pinned manifest")
        if (
            pq.read_metadata(safe_path(Path(data_dir), key)).num_rows
            != manifest["statistics"]["rows"]
        ):
            raise ValueError("Anchor row count differs from its manifest")
    return files


def schema_for(dataset, binding):
    import pyarrow.dataset as ds

    from kumo.pytorch.api import (
        NON_TEMPORAL,
        DataSchema,
        RelationSpec,
        TableSpec,
    )

    from .bindings import dataset_definition
    from .contracts import schema_columns

    definition = dataset_definition(binding)
    return DataSchema(
        tables=tuple(
            TableSpec(
                name,
                entry["key"],
                schema_columns(ds.dataset(source.path, format="parquet").schema),
                time_column=entry.get("time", NON_TEMPORAL),
            )
            for name, source in dataset.tables.items()
            for entry in (definition["tables"][name],)
        ),
        relations=tuple(RelationSpec(*entry) for entry in definition["relations"]),
    )


def identity_map(data_dir, definition, files):
    import pyarrow.parquet as pq

    paths = definition.get("identity_paths") or [definition["identity_path"]]
    for path in paths:
        verify_file(safe_path(data_dir, path), files[path])
    names = [
        value
        for path in paths
        for value in pq.ParquetFile(data_dir / path)
        .read(columns=[definition["identity_column"]])
        .column(0)
        .to_pylist()
    ]
    if definition.get("identity_encoding") == "canonical_integer":
        if any(type(name) is not int or not 0 <= name < 2**63 for name in names):
            raise ValueError("Canonical identities require nonnegative int64 values")
        names = [str(name) for name in names]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Identity mapping requires unique, nonempty canonical strings")
    if definition.get("identity_id_column"):
        ids = (
            pq.read_table(
                [data_dir / path for path in paths],
                columns=[definition["identity_id_column"]],
            )
            .column(0)
            .to_pylist()
        )
        if len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
            raise ValueError("Mapped entity IDs must be unique integers")
        identities = dict(zip(names, ids, strict=True))
    else:
        identities = {
            name: i
            for i, name in enumerate(
                sorted(names, key=lambda name: name.encode("utf-8"))
            )
        }
    return identities


def prepare_binding(
    data_dir,
    directory,
    binding,
    files,
    *,
    limit,
    prepared_dir=None,
    recorder=None,
    splits=("train", "valid", "test"),
    full_sources=False,
):
    """Shared preparation for arbitrary datasets with declared source/anchor schemas."""
    from collections import Counter
    from contextlib import nullcontext

    import pyarrow.parquet as pq

    from kumo.pytorch.api import Dataset, Parquet

    from .bindings import dataset_definition, graph_layout, profile, resolve
    from .population_anchors import project_binding, projected_rows

    definition = dataset_definition(binding)
    with recorder.span("source.read") if recorder else nullcontext():
        if definition.get("anchor_generation"):
            from .cohort_anchors import generate

            source_anchor, identities, anchor_identity = generate(
                data_dir, directory, definition, files
            )
        else:
            identities = identity_map(data_dir, definition, files)
            selector = binding["selector"]
            anchor_path = definition["anchor_path"].format(
                population=selector["population"], version=selector["anchor_version"]
            )
            source_anchor = safe_path(data_dir, anchor_path)
            verify_file(source_anchor, files[anchor_path])
            anchor_identity = files[anchor_path]
    projected = directory / "anchors.parquet"
    with recorder.span("anchors.prepare") if recorder else nullcontext():
        anchors, population = project_binding(
            source_anchor,
            projected,
            identities,
            binding,
            limit=limit,
            splits=splits,
        )
    roots = (
        None
        if limit is None or full_sources
        else {
            row["registered_entity_id"]
            for split in population["counts"]
            for row in projected_rows(projected, split)
        }
    )
    prepared_dir = Path(prepared_dir) if prepared_dir else directory / "prepared"
    span = (
        "source.read"
        if prepared_dir.exists() or not definition.get("prepare")
        else "source.convert"
    )
    with recorder.span(span) if recorder else nullcontext():
        if definition.get("prepare"):
            dataset, mapping = resolve(definition["prepare"])(
                data_dir, prepared_dir, definition, files, roots
            )
        else:
            from .cohort_anchors import table_files

            sources = {
                name: table_files(data_dir, entry, files)
                for name, entry in definition["tables"].items()
            }
            dataset = Dataset(
                {
                    name: Parquet(
                        str(
                            safe_path(
                                data_dir,
                                entry.get("path") or entry["prefix"].rstrip("/"),
                            )
                        )
                    )
                    for name, entry in definition["tables"].items()
                }
            )
            mapping = {
                "scope": {"kind": "full_source"},
                "sources": {
                    name: items[0] if "path" in definition["tables"][name] else items
                    for name, items in sources.items()
                },
                "tables": {
                    name: {
                        "rows": sum(
                            pq.read_metadata(item["path"]).num_rows for item in items
                        ),
                        "bytes": sum(item["bytes"] for item in items),
                    }
                    for name, items in sources.items()
                },
            }
    mapping = {
        **mapping,
        "table_paths": {n: str(source.path) for n, source in dataset.tables.items()},
        "anchor_source": {**anchor_identity, "path": str(source_anchor)},
    }
    sampling_limit = profile(binding, "input").get("sampling_limit")
    if sampling_limit is not None:
        if type(sampling_limit) is not int or sampling_limit < 1:
            raise ValueError("sampling_limit must be a positive integer")
        mapping["sampling_limit"] = sampling_limit
        return dataset, anchors, population, mapping, sampling_limit
    with recorder.span("source.read") if recorder else nullcontext():
        bounds = {}

        def degree(table, column):
            key = (table, column)
            if key not in bounds:
                counts = Counter()
                for batch in pq.ParquetFile(dataset.tables[table].path).iter_batches(
                    columns=[column]
                ):
                    counts.update(batch.column(0).to_pylist())
                bounds[key] = max(counts.values(), default=1)
            return bounds[key]

        bound = definition.get("history_bound")
        inputs = profile(binding, "input")
        relations = {row[0]: row for row in definition["relations"]}
        capacity = 0
        for _, path, _, _ in inputs.get("families", ()):
            table, population_bound = inputs["root_table"], 1
            for relation in path:
                edge = relations[relation]
                if table == edge[1]:
                    table = edge[3]
                elif table == edge[3]:
                    table = edge[1]
                else:
                    raise ValueError(f"Disconnected history path at {relation}")
                population_bound *= pq.read_metadata(
                    dataset.tables[table].path
                ).num_rows
            direct = relations[path[0]] if len(path) == 1 else None
            if (
                direct
                and direct[1] == inputs["root_table"]
                and (
                    not bound
                    or (direct[3] == bound["table"] and direct[4] == bound["entity"])
                )
            ):
                capacity += degree(direct[3], direct[4])
            else:
                # Row products bound every path occurrence, including repeated
                # families and alternate joins; they may conservatively overcount.
                capacity += population_bound
        graph_capacity = 1
        if binding["representation"] in {"graphsage", "combined"}:
            for relation, node in graph_layout(inputs)[1].items():
                edge = relations[relation]
                if edge[1] != inputs["root_table"] or edge[3] != node:
                    raise ValueError(
                        f"Graph profile requires a direct root-to-node relation: {relation}"
                    )
                graph_capacity = max(graph_capacity, degree(edge[3], edge[4]))
    return dataset, anchors, population, mapping, max(graph_capacity, capacity, 1)


if __name__ == "__main__":
    raise SystemExit(main())
