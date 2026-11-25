# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native DiskGraph ingest and anchor scheduling against an HTTP S3 endpoint."""

import io
import json
from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.e2e.s3_endpoint import local_s3
from tests.test_candidate_neighborhoods import REQUESTS, assert_graphs, contract, parquet_inputs
from tests.test_scored_walks import star, walk_answer


def parquet_bytes(data):
    buffer = io.BytesIO()
    pq.write_table(data, buffer, compression=None, use_dictionary=False)
    return buffer.getvalue()


def train_outputs(task):
    answers = []
    for batch in task.batches(
        split="train", batch_size=2, seed=7, shuffle=True, output_format="columnar"
    ):
        assert_graphs(batch)
        answers.append((tuple(batch.metadata["sample_ids"]), walk_answer([batch])))
    return answers


def test_s3_parquet_index_and_anchor_revisions(tmp_path):
    diskgraph = pytest.importorskip("diskgraph")
    pytest.importorskip("moto.server")
    schema, rows = star()
    decl = contract(schema)
    requests = (REQUESTS[2], REQUESTS[1], REQUESTS[0], replace(REQUESTS[0], sample_id="held_out"))
    local_dataset, anchors = parquet_inputs(schema, rows, requests, tmp_path)
    data = pq.read_table(anchors.source.path).append_column(
        "split", pa.array(["train"] * 3 + ["test"])
    )
    pq.write_table(data, anchors.source.path)
    anchors = replace(anchors, split="split")
    # Warehouse IDs and split assignments can have different physical names remotely.
    data = data.set_column(
        data.schema.get_field_index(anchors.entity_id),
        "remote_entity",
        data.column(anchors.entity_id).cast(pa.int32()),
    )
    data = data.rename_columns(
        ["remote_split" if name == "split" else name for name in data.column_names]
    )
    engine = api.DiskGraphEngine(num_threads=1)
    bucket = "omnisage-fixture"
    uri = f"s3://{bucket}/"
    pins = pq.read_table(local_dataset.tables["pins"].path)
    members = pq.read_table(local_dataset.tables["membership"].path)

    with local_s3() as s3, patch.object(diskgraph, "ingest", wraps=diskgraph.ingest) as ingest:

        def put(key, payload):
            s3.put_object(Bucket=bucket, Key=key, Body=payload)

        s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
        original = parquet_bytes(pins.rename_columns(["remote_id"]))
        pins_key = "tables/pins #1% é.parquet"
        put(pins_key, original)
        for part in (1, 0):
            put(
                f"tables/membership/day=1/part-{part}.parquet",
                parquet_bytes(members.slice(2 * part, 2)),
            )
            put(f"anchors/day=1/part-{part}.parquet", parquet_bytes(data.slice(2 * part, 2)))
        put("tables/membership/_SUCCESS", b"")
        dataset = api.Dataset(
            {
                **local_dataset.tables,
                "pins": api.Parquet(uri + pins_key, {"remote_id": "id"}),
                "membership": api.Parquet(uri + "tables/membership/"),
            }
        )
        remote = replace(
            anchors,
            source=api.Parquet(
                uri + "anchors/", {"remote_entity": anchors.entity_id, "remote_split": "split"}
            ),
        )
        with api.prepare(decl, dataset, engine) as plan:
            with plan.materialize(anchors, fit_split="train") as local:
                expected = train_outputs(local)
            with plan.materialize(remote, fit_split="train") as task:
                assert train_outputs(task) == expected
                fingerprints = dict(task.fingerprints)
        assert sorted(sid for ids, _ in expected for sid in ids) == ["early", "empty", "late"]
        manifest_path = engine.index / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        objects = manifest["s3_objects"]
        assert set(objects) == {"pins", "membership"}
        assert [p["path"] for p in objects["membership"]] == [
            uri + f"tables/membership/day=1/part-{i}.parquet" for i in (0, 1)
        ]
        head = s3.head_object(Bucket=bucket, Key=pins_key)
        assert objects["pins"] == [
            {
                "path": uri + pins_key,
                "e_tag": head["ETag"],
                "version": head["VersionId"],
                "size": len(original),
            }
        ]

        # Reopening unchanged inputs reuses the same base. Replacing the first
        # anchor part moves "late" into the held-out split.
        with api.prepare(decl, dataset, engine) as plan:
            with plan.materialize(remote, fit_split="train") as task:
                assert train_outputs(task) == expected
                assert task.fingerprints == fingerprints
            replacement = data.slice(0, 2).set_column(
                data.schema.get_field_index("remote_split"),
                "remote_split",
                pa.array(["test", "train"]),
            )
            put("anchors/day=1/part-0.parquet", parquet_bytes(replacement))
            with plan.materialize(remote, fit_split="train") as changed:
                assert sorted(sid for ids, _ in train_outputs(changed) for sid in ids) == [
                    "early",
                    "empty",
                ]
                assert changed.fingerprints["anchors"] != fingerprints["anchors"]
                assert changed.fingerprints["dataset"] == fingerprints["dataset"]

        checkpoint = {
            path: path.stat().st_mtime_ns for path in engine.index.rglob("*") if path.is_file()
        }
        modified = parquet_bytes(pa.table({"remote_id": [1, 2, 3, 9]}))
        assert len(modified) == len(original) and modified != original
        put(pins_key, modified)
        with pytest.raises(ValueError, match="different dataset"):
            api.prepare(decl, dataset, engine)
        assert {
            path: path.stat().st_mtime_ns for path in engine.index.rglob("*") if path.is_file()
        } == checkpoint
        assert [call.kwargs.get("parent_index_path") for call in ingest.call_args_list].count(
            None
        ) == 1
