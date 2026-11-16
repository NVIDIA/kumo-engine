"""Atomic, versioned prepared-store reuse; anchor and fitted state stay separate."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
from kumo.pytorch.contract.schema import table_clock

from .parquet import local_parquet_files

FORMAT_VERSION = 1


def _files(dataset):
    return {
        name: [
            dict(  # noqa: C408 -- preserve copied source
                path=path,
                size=(stat := Path(path).stat()).st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino,
                device=stat.st_dev,
            )
            for path in local_parquet_files(source.path)
        ]
        for name, source in sorted(dataset.tables.items())
    }


def identity(dataset, schema, backend):
    return {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "backend_version": importlib.metadata.version(backend),
        "arrow_version": pa.__version__,
        "sources": _files(dataset),
        "implementation": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).parent / backend / "store.py",
                Path(__file__).parent / "parquet.py",
            )
        },
        "schema": {
            "tables": [
                {
                    "name": t.name,
                    "primary_key": t.primary_key,
                    "columns": {
                        name: asdict(dtype) if dtype is not None else None
                        for name, dtype in t.columns.items()
                    },
                    "clock": table_clock(t),
                }
                for t in schema.tables
            ],
            "relations": [
                {
                    "name": r.name,
                    "left_table": r.left_table,
                    "left_key": r.left_key,
                    "right_table": r.right_table,
                    "right_key": r.right_key,
                }
                for r in schema.relations
            ],
        },
    }


def _inventory(root):
    return {
        str(p.relative_to(root)): {
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _metadata(store):
    return {
        "description": store.describe(),
        "schemas": {
            name: base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")
            for name, schema in store._schemas.items()
        },
    }


def decode_schemas(metadata):
    return {
        name: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))
        for name, value in metadata["schemas"].items()
    }


@contextmanager
def _lock(path):
    # Independent opens may read the finished artifact; construction is serialized.
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def open_prepared(
    store_type, dataset, schema, *, backend, index=None, ingest="if_missing", **options
):
    if ingest not in {"if_missing", "never"}:
        raise ValueError("ingest must be 'if_missing' or 'never'")
    if index is None:
        if ingest == "never":
            raise ValueError("ingest='never' requires an existing index directory")
        store = store_type(dataset, schema, **options)
        store._description["artifact"] = {"status": "ephemeral", "path": None}
        return store
    destination = Path(index).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError("index must be a prepared-store directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")
    with _lock(lock_path):
        selected = identity(dataset, schema, backend)
        if destination.exists():
            manifest_path = destination / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"prepared index is incomplete or invalid: {destination}"
                ) from error
            if not isinstance(manifest, dict) or not isinstance(
                manifest.get("metadata"), dict
            ):
                raise ValueError("prepared index manifest has invalid metadata")
            metadata = manifest["metadata"]
            if manifest.get("metadata_sha256") != _digest(metadata):
                raise ValueError(
                    "prepared index metadata changed or is incomplete; choose a new index"
                )
            if manifest.get("identity") != selected:
                raise ValueError(
                    "prepared index belongs to a different source/schema/backend version; choose a new index"
                )
            if manifest.get("inventory") != _inventory(destination):
                raise ValueError(
                    "prepared index files changed or are incomplete; choose a new index"
                )
            store = store_type(
                dataset,
                schema,
                artifact_dir=destination,
                reuse=manifest["metadata"],
                **options,
            )
            status = "reused"
        else:
            if ingest == "never":
                raise FileNotFoundError(f"prepared index does not exist: {destination}")
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=destination.name + ".building-", dir=destination.parent
                )
            )
            store = None
            try:
                store = store_type(dataset, schema, artifact_dir=temporary, **options)
                metadata = _metadata(store)
                store.close()
                store = None
                if identity(dataset, schema, backend) != selected:
                    raise ValueError("source files changed during index preparation")
                payload = {
                    "identity": selected,
                    "metadata": metadata,
                    "metadata_sha256": _digest(metadata),
                    "inventory": _inventory(temporary),
                }
                (temporary / "manifest.json").write_text(
                    json.dumps(payload, indent=2) + "\n"
                )
                os.replace(temporary, destination)
                store = store_type(
                    dataset, schema, artifact_dir=destination, reuse=metadata, **options
                )
                status = "built"
            except BaseException:
                if store is not None:
                    store.close()
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        store._description["artifact"] = {
            "status": status,
            "path": str(destination),
            "format_version": FORMAT_VERSION,
            "identity_sha256": _digest(selected),
        }
        return store
