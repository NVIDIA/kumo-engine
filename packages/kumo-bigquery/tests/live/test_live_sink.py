# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier for the sink: Parquet loaded into real BigQuery tables."""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

if os.environ.get("KUMO_BIGQUERY_LIVE") != "1":
    pytest.skip("Set KUMO_BIGQUERY_LIVE=1 to run live BigQuery tests.",
                allow_module_level=True)

PROJECT = os.environ.get("KUMO_BIGQUERY_PROJECT", "").strip()
STAGING_BUCKET = os.environ.get("KUMO_BIGQUERY_STAGING", "").strip().rstrip("/")
if not PROJECT or not STAGING_BUCKET:
    pytest.fail(
        "Live BigQuery tests require explicit KUMO_BIGQUERY_PROJECT and "
        "KUMO_BIGQUERY_STAGING (gs://bucket[/prefix]).", pytrace=False)

import google.auth  # noqa: E402

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# Explicit live runs fail when ADC is unavailable instead of silently skipping.
CREDENTIALS, _ = google.auth.default(scopes=[SCOPE])
CREDENTIALS = CREDENTIALS.with_quota_project(PROJECT)

RUN = uuid.uuid4().hex[:8]
DATASET = f"kumo_conn_test_{RUN}"
STAGING_URI = f"{STAGING_BUCKET}/sink/{RUN}"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from google.api_core.exceptions import NotFound  # noqa: E402
from google.cloud import bigquery, storage  # noqa: E402
from kumo_bigquery.sink import BigQuerySinkBackend  # noqa: E402
from kumo_bigquery.write import OutputSchemaMismatchError  # noqa: E402
from kumo_core.sources.errors import (  # noqa: E402
    PermanentSourceError,
    SourceAccessDeniedError,
    SourceNotFoundError,
    UnsupportedReadModeError,
)
from kumo_core.sources.principals import store  # noqa: E402
from kumo_core.sources.types import ReadMode  # noqa: E402
from kumoapi.data_source import (  # noqa: E402
    BigQueryConnectorResourceConfig,
    BigQueryIdentityConfig,
    CreateIdentityArgs,
    GoogleAmbientAuthentication,
)
from kumoapi.jobs import WriteMode  # noqa: E402

IDENTITY = f"bq-sink-live-{RUN}"


def connector(dataset=DATASET, location="US", project=PROJECT):
    return SimpleNamespace(
        id="live-sink", identity=IDENTITY,
        config=BigQueryConnectorResourceConfig(project_id=project,
                                               dataset_id=dataset,
                                               location=location,
                                               billing_project_id=PROJECT))


@pytest.fixture(scope="module")
def client():
    store.create(
        CreateIdentityArgs(
            display_name=IDENTITY, slug=IDENTITY, config=BigQueryIdentityConfig(
                authentication=GoogleAmbientAuthentication())))
    bq = bigquery.Client(project=PROJECT, credentials=CREDENTIALS)
    bq.create_dataset(bigquery.Dataset(f"{PROJECT}.{DATASET}"))
    try:
        yield bq
    finally:
        bq.delete_dataset(f"{PROJECT}.{DATASET}", delete_contents=True,
                          not_found_ok=True)
        bucket, _, prefix = STAGING_URI.removeprefix("gs://").partition("/")
        gcs = storage.Client(project=PROJECT, credentials=CREDENTIALS)
        for blob in gcs.list_blobs(bucket, prefix=prefix):
            blob.delete()


@pytest.fixture(params=["staged", "streamed"])
def sink(request):
    if request.param == "streamed":
        return BigQuerySinkBackend()
    return BigQuerySinkBackend(staging_uri=STAGING_URI, mode=ReadMode.STAGED)


def staged_objects():
    bucket, _, prefix = STAGING_URI.removeprefix("gs://").partition("/")
    gcs = storage.Client(project=PROJECT, credentials=CREDENTIALS)
    return [blob.name for blob in gcs.list_blobs(bucket, prefix=prefix)]


def shards(tmp_path, sizes, start=0, **columns):
    paths = []
    for index, size in enumerate(sizes):
        ids = np.arange(start, start + size)
        start += size
        frame = pd.DataFrame({
            "ENTITY":
            ids,
            "SCORE":
            ids / 10 if index else np.full(size, np.nan),
            "ANCHOR":
            pd.Timestamp("2024-01-01 00:00:00.123456789", tz="UTC") +
            pd.to_timedelta(ids, unit="D"),
            "EMBEDDING": [np.array([i, 1.0], dtype=np.float32) for i in ids],
        })
        for name, values in columns.items():
            frame[name] = values[index]
        path = tmp_path / f"part-{index}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(str(path))
    return paths


def table(client, name):
    return client.get_table(f"{PROJECT}.{DATASET}.{name}")


def rows(client, name):
    # Table metadata lags rows committed through the Storage Write API.
    query = f"SELECT COUNT(*) AS n FROM `{PROJECT}.{DATASET}.{name}`"
    return next(iter(client.query(query).result())).n


def schema(client, name):
    return {(f.name, f.field_type, f.mode) for f in table(client, name).schema}


def test_multi_shard_overwrite_then_append(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    files = shards(tmp_path, [5, 7, 0, 9])

    written = sink.write_files(connector(), name, files, WriteMode.OVERWRITE)

    assert written == f"{PROJECT}.{DATASET}.{name}"
    assert rows(client, name) == 21
    assert schema(client, name) == {
        ("ENTITY", "INTEGER", "NULLABLE"),
        ("SCORE", "FLOAT", "NULLABLE"),
        ("ANCHOR", "TIMESTAMP", "NULLABLE"),
        ("EMBEDDING", "FLOAT", "REPEATED"),
    }
    loaded = {
        row.ENTITY: row
        for row in client.query(f"SELECT * FROM `{written}`").result()
    }
    assert loaded[1].EMBEDDING == [1.0, 1.0]
    assert loaded[1].SCORE is None and loaded[6].SCORE == pytest.approx(.6)
    assert loaded[1].ANCHOR == pd.Timestamp("2024-01-02 00:00:00.123456",
                                            tz="UTC")

    sink.write_files(connector(), name, files[1:], WriteMode.APPEND)
    assert rows(client, name) == 37

    sink.write_files(connector(), name, files[1:2], WriteMode.OVERWRITE)
    assert rows(client, name) == 7
    assert not staged_objects()


def test_overwrite_replaces_schema_and_append_creates(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    sink.write_files(connector(), name, shards(tmp_path, [3]), WriteMode.APPEND)
    assert rows(client, name) == 3

    other = tmp_path / "other.parquet"
    pd.DataFrame({"ENTITY": ["a", "b"]}).to_parquet(other)
    sink.write_files(connector(), name, [str(other)], WriteMode.OVERWRITE)
    assert schema(client, name) == {("ENTITY", "STRING", "NULLABLE")}
    assert rows(client, name) == 2


@pytest.mark.parametrize("change", ["extra column", "changed type"])
def test_append_schema_mismatch_leaves_table(client, sink, tmp_path, change):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    sink.write_files(connector(), name, shards(tmp_path, [4]),
                     WriteMode.OVERWRITE)
    path = tmp_path / "mismatch.parquet"
    frame = pd.read_parquet(shards(tmp_path, [2])[0])
    if change == "extra column":
        frame["CLASS"] = "x"
    else:
        frame["ENTITY"] = frame["ENTITY"].astype(str)
    frame.to_parquet(path)

    with pytest.raises(OutputSchemaMismatchError,
                       match="Cannot APPEND") as error:
        sink.write_files(connector(), name, [str(path)], WriteMode.APPEND)
    expected = ("CLASS (STRING NULLABLE) is not a table column"
                if change == "extra column" else
                "ENTITY is STRING NULLABLE but the table column is INTEGER")
    assert expected in str(error.value)
    assert rows(client, name) == 4


def test_append_widens_and_accepts_all_null(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    client.create_table(
        bigquery.Table(
            f"{PROJECT}.{DATASET}.{name}", schema=[
                bigquery.SchemaField("entity", "NUMERIC"),
                bigquery.SchemaField("score", "FLOAT"),
                bigquery.SchemaField("label", "DATETIME"),
            ]))
    path = tmp_path / "widen.parquet"
    pq.write_table(
        pa.table({
            "ENTITY": pa.array([1, 2], pa.int32()),
            "SCORE": [3, 4],
            "LABEL": pa.nulls(2),
        }), path)
    sink.write_files(connector(), name, [str(path)], WriteMode.APPEND)
    loaded = sorted(
        (row.entity, row.score, row.label)
        for row in client.query(f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`"))
    assert [(int(e), s, l) for e, s, l in loaded] == [(1, 3.0, None),
                                                      (2, 4.0, None)]


def test_overwrite_keeps_column_descriptions(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    files = shards(tmp_path, [3])
    sink.write_files(connector(), name, files)
    current = table(client, name)
    current.schema = [
        bigquery.SchemaField(f.name, f.field_type, mode=f.mode,
                             description=f"about {f.name}")
        for f in current.schema
    ]
    client.update_table(current, ["schema"])

    sink.write_files(connector(), name, files, WriteMode.OVERWRITE)
    assert {
        f.name: f.description
        for f in table(client, name).schema
    } == {
        "ENTITY": "about ENTITY",
        "SCORE": "about SCORE",
        "ANCHOR": "about ANCHOR",
        "EMBEDDING": "about EMBEDDING",
    }


@pytest.mark.parametrize("mode", [WriteMode.OVERWRITE, WriteMode.APPEND])
def test_failed_load_leaves_table(client, sink, tmp_path, mode):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    sink.write_files(connector(), name, shards(tmp_path, [4]),
                     WriteMode.OVERWRITE)
    before = schema(client, name)
    files = shards(tmp_path, [3, 2], start=10)
    # BigQuery TIMESTAMP ends at 9999-12-31.
    bad = pq.read_table(files[1])
    position = bad.schema.get_field_index("ANCHOR")
    bad = bad.set_column(
        position, "ANCHOR",
        pa.array([253402300800000000] * 2, pa.timestamp("us", tz="UTC")))
    pq.write_table(bad, files[1])

    with pytest.raises(PermanentSourceError):
        sink.write_files(connector(), name, files, mode)
    assert rows(client, name) == 4
    assert schema(client, name) == before
    assert not staged_objects()
    assert not [
        t.table_id for t in client.list_tables(f"{PROJECT}.{DATASET}")
        if t.table_id.startswith("_kumo_export_")
    ]


def test_extra_columns_are_stamped(client, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    BigQuerySinkBackend().write_files(connector(), name,
                                      shards(tmp_path, [2, 3]),
                                      extra_columns={"RUN": "r1"})
    values = {
        row.RUN
        for row in client.query(
            f"SELECT RUN FROM `{PROJECT}.{DATASET}.{name}`").result()
    }
    assert values == {"r1"}


def test_write_table(client, sink):
    name = f"frame_{uuid.uuid4().hex[:6]}"
    sink.write_table(connector(), name, pd.DataFrame({"ENTITY": [1, 2]}))
    assert rows(client, name) == 2


def test_preflight_accepts_writable_destination(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    sink.preflight_write_files(connector(), name, WriteMode.OVERWRITE)
    sink.write_files(connector(), name, shards(tmp_path, [2]))
    sink.preflight_write_files(connector(), name, WriteMode.APPEND)
    assert not [
        t.table_id for t in client.list_tables(f"{PROJECT}.{DATASET}")
        if t.table_id.startswith("__kumo_write_probe_")
    ]
    assert not staged_objects()


def test_preflight_rejects_unusable_destinations(client):
    sink = BigQuerySinkBackend()
    with pytest.raises(SourceNotFoundError, match="does not exist"):
        sink.preflight_write_files(connector(dataset=f"missing_{RUN}"), "preds")
    with pytest.raises(SourceNotFoundError, match="does not exist"):
        sink.write_files(connector(dataset=f"missing_{RUN}"), "preds", [])
    with pytest.raises(PermanentSourceError, match="location='US'"):
        sink.preflight_write_files(connector(location="EU"), "preds")
    with pytest.raises(SourceAccessDeniedError, match="create tables"):
        sink.preflight_write_files(
            connector(dataset="usa_names", project="bigquery-public-data"),
            "preds")
    with pytest.raises(ValueError, match="unqualified"):
        sink.preflight_write_files(connector(), "other.preds")
    denied = "gs://gcp-public-data-landsat/kumo"
    with pytest.raises(UnsupportedReadModeError, match="offers streamed"):
        BigQuerySinkBackend(staging_uri=denied, mode=ReadMode.STAGED).\
            preflight_write_files(connector(), "preds")
    fallback = BigQuerySinkBackend(staging_uri=denied)
    fallback.preflight_write_files(connector(), "preds")
    assert fallback.write_modes(connector()) == (ReadMode.STREAMED, )
    with pytest.raises(NotFound):
        client.get_table(f"{PROJECT}.{DATASET}.preds")


@pytest.mark.parametrize("mode", [WriteMode.OVERWRITE, WriteMode.APPEND])
def test_partitioned_clustered_destination(client, sink, tmp_path, mode):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    files = shards(tmp_path, [3, 4])
    sink.write_files(connector(), name, files)
    current = table(client, name)
    current.clustering_fields = ["ENTITY"]
    client.update_table(current, ["clustering_fields"])
    partitioned = f"{name}_by_day"
    client.query(f"CREATE TABLE `{PROJECT}.{DATASET}.{partitioned}` "
                 f"PARTITION BY DATE(ANCHOR) CLUSTER BY ENTITY AS "
                 f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`").result()

    for destination in (name, partitioned):
        sink.write_files(connector(), destination, files, mode)
        assert rows(client,
                    destination) == (14 if mode == WriteMode.APPEND else 7)
    assert table(client, partitioned).time_partitioning.field == "ANCHOR"


def test_shards_with_different_columns(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    files = shards(tmp_path, [2, 3])
    pd.DataFrame({
        "ENTITY": [9],
        "CLASS": ["a"]
    }).to_parquet(tmp_path / "c.parquet")
    sink.write_files(connector(), name, files + [str(tmp_path / "c.parquet")])
    assert rows(client, name) == 6
    assert ("CLASS", "STRING", "NULLABLE") in schema(client, name)


def test_thousands_of_shards(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    files = []
    for index in range(1500):
        path = tmp_path / f"part-{index:05d}.parquet"
        pq.write_table(
            pa.table({
                "ENTITY": [2 * index, 2 * index + 1],
                "EMBEDDING": [[float(index)], [1.0]],
            }), path)
        files.append(str(path))
    sink.write_files(connector(), name, files)
    assert rows(client, name) == 3000
    assert not staged_objects()


def test_null_embedding_lands_empty(client, sink, tmp_path):
    name = f"preds_{uuid.uuid4().hex[:6]}"
    path = tmp_path / "nulls.parquet"
    pq.write_table(
        pa.table({
            "ENTITY": [1, 2],
            "EMBEDDING": pa.array([[1.0], None], pa.list_(pa.float32())),
        }), path)
    sink.write_files(connector(), name, [str(path)])
    embeddings = {
        row.ENTITY: row.EMBEDDING
        for row in client.query(
            f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`").result()
    }
    assert embeddings == {1: [1.0], 2: []}
