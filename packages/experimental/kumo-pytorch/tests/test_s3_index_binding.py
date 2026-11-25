# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The object snapshot handed to DiskGraph also defines reuse of its local index."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.fake_ingest import FakeDiskGraph, recording
from kumo.pytorch import api, materialization


@pytest.fixture
def remote_dataset(tmp_path, monkeypatch):
    source = "s3://data-bucket/users/"
    objects = [
        {"path": source + part, "size": 123, "e_tag": '"abc"'}
        for part in ("part-0.parquet", "part-1.parquet")
    ]
    objects[0]["version"] = "original"
    local = tmp_path / "footer.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), local)
    monkeypatch.setattr(materialization, "parquet_objects", lambda path: [dict(o) for o in objects])
    monkeypatch.setattr(materialization, "open_parquet_file", lambda obj: pq.ParquetFile(local))
    schema = api.DataSchema(
        (api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),)
    )
    return schema, api.Dataset({"users": api.Parquet(source)}), objects


@pytest.mark.parametrize("change", ["etag", "version", "add", "remove"])
def test_index_reuse_fingerprints_the_exact_object_manifest(remote_dataset, tmp_path, change):
    schema, dataset, objects = remote_dataset
    dest = tmp_path / "index"
    fake = FakeDiskGraph()
    with recording(fake):
        artifact = materialization.ensure_diskgraph_index(schema, dataset, dest)
        assert materialization.ensure_diskgraph_index(schema, dataset, dest) == artifact
        assert fake.calls[0].config["tables"][0]["source"] == {
            "type": "object_store_parquet",
            "auth_profile": "ambient",
            "objects": objects,
        }
        assert (
            json.loads((dest / "index_manifest.json").read_text())["s3_objects"]["users"] == objects
        )
        if change == "etag":
            objects[0]["e_tag"] = '"def"'
        elif change == "version":
            objects[0]["version"] = "next"
        elif change == "add":
            objects.append({**objects[0], "path": objects[0]["path"].replace("-0.", "-2.")})
        else:
            objects.pop()
        with pytest.raises(ValueError, match="different dataset"):
            materialization.ensure_diskgraph_index(schema, dataset, dest)
        assert len(fake.calls) == 1


def test_s3_change_during_ingest_leaves_no_reusable_index(remote_dataset, tmp_path):
    schema, dataset, objects = remote_dataset
    dest = tmp_path / "indexes" / "index"

    def overwrite(_):
        objects[0]["e_tag"] = '"same-size-overwrite"'

    with recording(FakeDiskGraph(overwrite)), pytest.raises(ValueError, match="changed while"):
        materialization.ensure_diskgraph_index(schema, dataset, dest)
    assert list(dest.parent.iterdir()) == []
    with recording(FakeDiskGraph()):
        artifact = materialization.ensure_diskgraph_index(schema, dataset, dest)
    assert artifact.manifest["s3_objects"]["users"] == objects
