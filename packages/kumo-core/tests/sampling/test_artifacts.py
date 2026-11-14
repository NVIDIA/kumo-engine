# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download integrity and worker-local metadata relocation."""

import hashlib
import json
from pathlib import Path

import pytest
from kumo_core.diskgraph.sampling_artifacts import (
    ArtifactBundle,
    ArtifactFile,
    download_artifact,
    relocate_artifacts,
)
from pyarrow.fs import LocalFileSystem


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "", "."])
def test_unsafe_names_are_refused(tmp_path: Path, name: str) -> None:
    """Downloads cannot leave their local artifact directory."""
    with pytest.raises(ValueError):
        download_artifact(
            LocalFileSystem(), "unused", ArtifactFile(name, 0, ""), tmp_path
        )


@pytest.mark.parametrize(
    "corruption", [b"short", b"wrong bytes", b"too many bytes here"]
)
def test_corrupt_download_is_never_installed(tmp_path: Path, corruption: bytes) -> None:
    """A truncated, same-size or oversized object cannot replace a valid local file."""
    source = tmp_path / "source"
    (source / "files").mkdir(parents=True)
    expected = b"right bytes"
    entry = ArtifactFile("payload", len(expected), hashlib.sha256(expected).hexdigest())
    (source / "files/payload").write_bytes(corruption)
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "payload").write_bytes(b"previous")

    with pytest.raises(ValueError):
        download_artifact(LocalFileSystem(), str(source / "files"), entry, replica)

    assert (replica / "payload").read_bytes() == b"previous"
    assert list(replica.glob("*.partial"))


def test_relocates_paths_and_refreshes_checksums(tmp_path: Path) -> None:
    """Copied TT metadata and both integrity inventories bind to the new graph."""
    source = tmp_path / "source"
    source.mkdir()
    metadata = {
        "graph": str(source),
        "rows": [str(source / "rows.bin")],
        "query": "SELECT /source/rows",
    }
    (source / "metadata.json").write_bytes(json.dumps(metadata).encode("utf-8"))
    (source / "checksums.json").write_bytes(
        json.dumps({"metadata.json_sha256": "old"}).encode("utf-8")
    )
    (source / "_MANIFEST.json").write_bytes(
        json.dumps({"files": [{"rel_path": "metadata.json", "size_bytes": 0}]}).encode(
            "utf-8"
        )
    )
    (source / "_SUCCESS").write_bytes(
        json.dumps({"index_path": str(source)}).encode("utf-8")
    )
    local = tmp_path / "worker-with-different-length-root"
    local.mkdir()
    entries = []
    for path in source.iterdir():
        data = path.read_bytes()
        (local / path.name).write_bytes(data)
        entries.append(
            ArtifactFile(path.name, len(data), hashlib.sha256(data).hexdigest())
        )
    bundle = ArtifactBundle(str(source), tuple(entries))

    relocate_artifacts(bundle, local)

    value = json.loads((local / "metadata.json").read_bytes())
    assert value == {**metadata, "graph": str(local), "rows": [str(local / "rows.bin")]}
    assert json.loads((local / "checksums.json").read_bytes()) == {
        "metadata.json_sha256": hashlib.sha256(
            (local / "metadata.json").read_bytes()
        ).hexdigest()
    }
    manifest = json.loads((local / "_MANIFEST.json").read_bytes())
    assert (
        manifest["files"][0]["size_bytes"] == (local / "metadata.json").stat().st_size
    )
    assert json.loads((local / "_SUCCESS").read_bytes())["index_path"] == str(local)


def test_relocation_rebases_sibling_source_paths(tmp_path: Path) -> None:
    """An index may reference staged source files outside its own directory."""
    producer = tmp_path / "producer"
    replica = tmp_path / "replica"
    index = replica / "snapshots/graph/version"
    index.mkdir(parents=True)
    metadata = {
        "index": str(producer / "snapshots/graph/version"),
        "source": str(producer / "snapshots/graph/source_tables/users.parquet"),
    }
    data = json.dumps(metadata).encode("utf-8")
    (index / "diskgraph_config.json").write_bytes(data)
    entry = ArtifactFile(
        "diskgraph_config.json", len(data), hashlib.sha256(data).hexdigest()
    )

    relocate_artifacts(
        ArtifactBundle(str(producer), (entry,)), index, destination_root=replica
    )

    assert json.loads((index / "diskgraph_config.json").read_bytes()) == {
        "index": str(index),
        "source": str(replica / "snapshots/graph/source_tables/users.parquet"),
    }
