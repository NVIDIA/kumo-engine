# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier: the reader against real BigQuery objects, row-verified."""

from __future__ import annotations

import os
import uuid

import pytest

if os.environ.get("KUMO_BIGQUERY_LIVE") != "1":
    pytest.skip("Set KUMO_BIGQUERY_LIVE=1 to run live BigQuery tests.",
                allow_module_level=True)

BILLING_PROJECT = os.environ.get("KUMO_BIGQUERY_PROJECT", "").strip()
STAGING_BUCKET = os.environ.get("KUMO_BIGQUERY_STAGING", "").strip().rstrip("/")
if not BILLING_PROJECT or not STAGING_BUCKET:
    pytest.fail(
        "Live BigQuery tests require explicit KUMO_BIGQUERY_PROJECT and "
        "KUMO_BIGQUERY_STAGING (gs://bucket[/prefix]).",
        pytrace=False,
    )

import google.auth  # noqa: E402

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# An explicitly enabled run must fail if ADC cannot be resolved, not skip.
CREDENTIALS, _ = google.auth.default(scopes=[SCOPE])
CREDENTIALS = CREDENTIALS.with_quota_project(BILLING_PROJECT)

from kumo_bigquery.identity import hour_version  # noqa: E402
from kumo_core.sources.types import ReadMode  # noqa: E402
from kumoapi.typing import Dtype  # noqa: E402
from sources.conformance import SourceReaderConformance  # noqa: E402
from sources.conformance import parquet_rows  # noqa: E402

PUBLIC_PROJECT, PUBLIC_DATASET = "bigquery-public-data", "usa_names"
PUBLIC_TABLE = "usa_1910_2013"
KNOWN_ROWS = 5552452
COLUMNS = {"state": Dtype.string, "number": Dtype.int64}

RUN = uuid.uuid4().hex[:8]
VIEW_DATASET = f"kumo_conn_test_{RUN}"
VIEW_TABLE = "usa_names_view"
STAGING_URI = f"{STAGING_BUCKET}/conformance/{RUN}"

_STATE: dict = {}


def _cached(key, build):
    if key not in _STATE:
        _STATE[key] = build()
    return _STATE[key]


def bigquery_client():
    from google.cloud import bigquery

    def build():
        return bigquery.Client(project=BILLING_PROJECT, credentials=CREDENTIALS)

    return _cached("bigquery", build)


def read_client():
    from google.cloud import bigquery_storage
    return _cached(
        "read",
        lambda: bigquery_storage.BigQueryReadClient(credentials=CREDENTIALS))


def storage_client():
    from google.cloud import storage

    def build():
        return storage.Client(project=BILLING_PROJECT, credentials=CREDENTIALS)

    return _cached("storage", build)


def _reader(key, project, dataset):
    from kumo_bigquery import BigQuerySourceReader
    return _cached(
        key, lambda: BigQuerySourceReader(bigquery_client(
        ), project=project, dataset=dataset, read_client=read_client(
        ), storage_client=storage_client(), staging_uri=STAGING_URI))


def table_reader():
    return _reader("table_reader", PUBLIC_PROJECT, PUBLIC_DATASET)


def view_reader():
    return _reader("view_reader", BILLING_PROJECT, VIEW_DATASET)


def counted_view_rows() -> int:
    """Row count established by the warehouse, not by the reader."""
    def count():
        job = bigquery_client().query(
            f"SELECT COUNT(*) FROM `{BILLING_PROJECT}`.`{VIEW_DATASET}`"
            f".`{VIEW_TABLE}`")
        return list(job.result())[0][0]

    return _cached("view_rows", count)


def staging_objects():
    location = STAGING_URI[len("gs://"):]
    bucket, _, prefix = location.partition("/")
    return list(storage_client().list_blobs(bucket, prefix=f"{prefix}/"))


@pytest.fixture(scope="module", autouse=True)
def warehouse_fixtures():
    """Create the view this tier needs, and leave the warehouse as found."""
    from google.cloud import bigquery
    client = bigquery_client()
    client.create_dataset(bigquery.Dataset(f"{BILLING_PROJECT}.{VIEW_DATASET}"))
    view = bigquery.Table(f"{BILLING_PROJECT}.{VIEW_DATASET}.{VIEW_TABLE}")
    view.view_query = (f"SELECT * FROM `{PUBLIC_PROJECT}`.`{PUBLIC_DATASET}`"
                       f".`{PUBLIC_TABLE}`")
    client.create_table(view)
    try:
        yield
    finally:
        client.delete_dataset(f"{BILLING_PROJECT}.{VIEW_DATASET}",
                              delete_contents=True, not_found_ok=True)
        for blob in staging_objects():
            blob.delete()


class TestBigQueryTableConformance(SourceReaderConformance):
    """The shared contract against a physical table, in both read modes."""
    def reader(self):
        return table_reader()

    def table(self) -> str:
        return PUBLIC_TABLE

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return KNOWN_ROWS


class TestBigQueryViewConformance(SourceReaderConformance):
    """The same contract against a logical view, which has no version.

    This is the reader that exercises the unstampable path: no read session
    can open on a view, and nothing about it can be proven unchanged.
    """
    def reader(self):
        return view_reader()

    def table(self) -> str:
        return VIEW_TABLE

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return counted_view_rows()


class TestBigQueryQueryConformance(SourceReaderConformance):
    """The same contract against a query rather than a table."""
    def reader(self):
        return table_reader()

    def table(self) -> str:
        return "custom_sql_live_conformance"

    def sql(self) -> str:
        projection = ", ".join(f"`{name}`" for name in COLUMNS)
        return (f"SELECT {projection} FROM "
                f"`{PUBLIC_PROJECT}`.`{PUBLIC_DATASET}`.`{PUBLIC_TABLE}`")

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return KNOWN_ROWS


def test_the_fixtures_are_the_table_and_the_view_this_tier_claims():
    info = bigquery_client().get_table(
        f"{PUBLIC_PROJECT}.{PUBLIC_DATASET}.{PUBLIC_TABLE}")
    assert info.num_rows == KNOWN_ROWS
    assert counted_view_rows() == KNOWN_ROWS


def test_a_physical_table_is_dated_but_a_logical_view_is_not():
    version = table_reader().data_version(PUBLIC_TABLE)
    assert version.isdigit()
    assert version != hour_version()
    assert view_reader().data_version(VIEW_TABLE) == hour_version()


def test_the_probe_finds_the_staging_bucket_writable_and_leaves_it_clean():
    """The measured cost of this answer is a second, no stages and no rows."""
    from kumo_bigquery import Capability
    assert table_reader().probe_staging() is Capability.ALLOWED
    assert staging_objects() == []


def test_a_read_session_cannot_be_opened_on_a_view():
    """The query-job route is a requirement of the API, not a preference."""
    from google.api_core.exceptions import InvalidArgument
    from google.cloud.bigquery_storage import types
    with pytest.raises(InvalidArgument) as failure:
        read_client().create_read_session(
            parent=f"projects/{BILLING_PROJECT}",
            read_session=types.ReadSession(
                table=(f"projects/{BILLING_PROJECT}/datasets/{VIEW_DATASET}"
                       f"/tables/{VIEW_TABLE}"),
                data_format=types.DataFormat.ARROW), max_stream_count=0)
    assert "non-table entities" in str(failure.value)


def test_a_staged_read_scans_only_the_projected_columns_and_clears_up_after(
        tmp_path):
    """A projection that is not pushed down scans the whole table."""
    info = bigquery_client().get_table(
        f"{PUBLIC_PROJECT}.{PUBLIC_DATASET}.{PUBLIC_TABLE}")
    staged = table_reader().prepare(PUBLIC_TABLE, str(tmp_path / "projected"),
                                    columns=COLUMNS, mode=ReadMode.STAGED,
                                    expected_rows=KNOWN_ROWS)
    assert staged.scanned_bytes < info.num_bytes * 0.75
    assert staging_objects() == []


def test_a_repeated_staged_read_reuses_the_prefix_safely(tmp_path):
    """Each export rewrites the object names the previous one wrote."""
    for attempt in range(2):
        staged = table_reader().prepare(PUBLIC_TABLE,
                                        str(tmp_path / f"repeat{attempt}"),
                                        columns=COLUMNS, mode=ReadMode.STAGED,
                                        expected_rows=KNOWN_ROWS)
        assert parquet_rows(staged.directory) == KNOWN_ROWS
