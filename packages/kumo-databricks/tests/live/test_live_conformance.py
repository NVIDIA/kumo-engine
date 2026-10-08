# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier: the reader against a real Databricks table, row-verified."""

from __future__ import annotations

import json
import os
import uuid

import pytest

if os.environ.get("KUMO_DATABRICKS_LIVE") != "1":
    pytest.skip("Set KUMO_DATABRICKS_LIVE=1 to run live Databricks tests.",
                allow_module_level=True)

CREDENTIALS = os.environ.get("KUMO_DATABRICKS_CREDS", "").strip()
if not CREDENTIALS:
    pytest.fail("Live Databricks tests require explicit KUMO_DATABRICKS_CREDS.",
                pytrace=False)

# Validate configuration before importing the provider; never echo credentials.
try:
    with open(CREDENTIALS, encoding="utf-8") as handle:
        _SECRETS = json.load(handle)
except (OSError, ValueError):
    pytest.fail("KUMO_DATABRICKS_CREDS must name a readable UTF-8 JSON file.",
                pytrace=False)

if not isinstance(_SECRETS, dict) or any(
        not isinstance(_SECRETS.get(key), str) or not _SECRETS[key].strip()
        for key in ("host", "client_id", "client_secret", "warehouse_id")):
    pytest.fail(
        "KUMO_DATABRICKS_CREDS must contain nonempty string fields: "
        "host, client_id, client_secret, warehouse_id.",
        pytrace=False,
    )

from kumo_core.sources.errors import PermanentSourceError  # noqa: E402
from kumo_core.sources.types import ReadMode  # noqa: E402
from kumoapi.typing import Dtype  # noqa: E402
from sources.conformance import SourceReaderConformance  # noqa: E402

CATALOG, SCHEMA, TABLE = "samples", "tpch", "customer"
KNOWN_ROWS = 750_000
STAGING_CATALOG, STAGING_SCHEMA = "kumo_test_catalogue", "amazon_10k"
TOKEN_LIFETIME_SECONDS = 3600
COLUMNS = {
    "c_custkey": Dtype.int64,
    "c_name": Dtype.string,
    "c_nationkey": Dtype.int64,
    "c_acctbal": Dtype.float,
    "c_mktsegment": Dtype.string,
}

_STATE: dict = {}


def secrets() -> dict:
    return _SECRETS


def workspace():
    if "workspace" not in _STATE:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
        found = secrets()
        _STATE["workspace"] = WorkspaceClient(config=Config(
            host=found["host"], client_id=found["client_id"],
            client_secret=found["client_secret"], auth_type="oauth-m2m"))
    return _STATE["workspace"]


def access_token() -> str:
    """A short-lived token, minted for this run and revoked in teardown."""
    if "token" not in _STATE:
        _STATE["token"] = workspace().tokens.create(
            lifetime_seconds=TOKEN_LIFETIME_SECONDS,
            comment="kumo-connector-e2e-test")
    return _STATE["token"].token_value


def volume_path() -> str:
    """A managed volume created for this run and deleted in teardown."""
    if "volume" not in _STATE:
        from databricks.sdk.service.catalog import VolumeType
        _STATE["volume"] = workspace().volumes.create(
            catalog_name=STAGING_CATALOG, schema_name=STAGING_SCHEMA,
            name=f"kumo_connector_test_{uuid.uuid4().hex[:8]}",
            volume_type=VolumeType.MANAGED)
    created = _STATE["volume"]
    return (f"/Volumes/{created.catalog_name}/{created.schema_name}"
            f"/{created.name}/out")


def connection():
    if "connection" not in _STATE:
        import databricks.sql
        found = secrets()
        _STATE["connection"] = databricks.sql.connect(
            server_hostname=found["host"].replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{found['warehouse_id']}",
            access_token=access_token(), use_cloud_fetch=True,
            enable_query_result_lz4_compression=True)
    return _STATE["connection"]


def live_reader():
    if "reader" not in _STATE:
        from kumo_databricks import DatabricksSourceReader
        _STATE["reader"] = DatabricksSourceReader(
            connection(), catalog=CATALOG, schema=SCHEMA,
            workspace_client=workspace(), staging_volume_path=volume_path())
    return _STATE["reader"]


def counted_rows() -> int:
    """Row count established by the warehouse, not by the reader."""
    if "rows" not in _STATE:
        cursor = connection().cursor()
        try:
            cursor.execute(
                f"SELECT COUNT(*) FROM `{CATALOG}`.`{SCHEMA}`.`{TABLE}`")
            _STATE["rows"] = cursor.fetchall()[0][0]
        finally:
            cursor.close()
    return _STATE["rows"]


@pytest.fixture(scope="module", autouse=True)
def release_what_this_run_created():
    yield
    opened = _STATE.pop("connection", None)
    if opened is not None:
        opened.close()
    created = _STATE.pop("volume", None)
    if created is not None:
        workspace().volumes.delete(name=created.full_name)
    minted = _STATE.pop("token", None)
    if minted is not None:
        workspace().tokens.delete(token_id=minted.token_info.token_id)


class TestDatabricksConformance(SourceReaderConformance):
    """The shared contract, unmodified, with both read modes available."""
    def reader(self):
        return live_reader()

    def table(self) -> str:
        return TABLE

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return counted_rows()


class TestDatabricksQueryConformance(SourceReaderConformance):
    """The same contract against a query rather than a table."""
    def reader(self):
        return live_reader()

    def table(self) -> str:
        return "custom_sql_live_conformance"

    def sql(self) -> str:
        projection = ", ".join(f"`{name}`" for name in COLUMNS)
        return (f"SELECT {projection} FROM "
                f"`{CATALOG}`.`{SCHEMA}`.`{TABLE}`")

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return counted_rows()


def test_the_fixture_table_and_both_modes_are_what_this_tier_claims():
    """A volume that failed to appear would silently skip the mode comparison."""
    assert counted_rows() == KNOWN_ROWS
    assert live_reader().read_modes(TABLE) == (ReadMode.STAGED,
                                               ReadMode.STREAMED)


def test_a_delta_table_is_stamped_with_its_version_and_commit_time():
    from kumo_databricks.identity import hour_version
    version = live_reader().data_version(TABLE)
    delta, _, committed = version.partition("_")
    assert delta.isdigit() and committed.isdigit()
    assert version != hour_version()


def test_a_view_falls_back_to_the_hour_window():
    from kumo_databricks.identity import data_version, hour_version
    cursor = connection().cursor()
    view = f"kumo_connector_view_{uuid.uuid4().hex[:8]}"
    source = f"`{STAGING_CATALOG}`.`{STAGING_SCHEMA}`.`{view}`"
    try:
        cursor.execute(f"CREATE VIEW {source} AS SELECT `c_custkey` FROM "
                       f"`{CATALOG}`.`{SCHEMA}`.`{TABLE}` LIMIT 10")
        assert data_version(cursor, source) == hour_version()
    finally:
        cursor.execute(f"DROP VIEW IF EXISTS {source}")
        cursor.close()


def test_an_absent_source_is_reported_not_quietly_dated():
    from kumo_databricks.identity import data_version
    cursor = connection().cursor()
    absent = f"kumo_connector_absent_{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(PermanentSourceError):
            data_version(cursor,
                         f"`{STAGING_CATALOG}`.`{STAGING_SCHEMA}`.`{absent}`")
    finally:
        cursor.close()


def test_the_staging_volume_is_left_empty(tmp_path):
    """The table's own directory above a read's attempt may outlive the read
    -- deleting it is what would reach an overlapping read's attempt -- so
    what is asserted is that nothing is left under it."""
    from databricks.sdk.errors import NotFound
    reader = live_reader()
    reader.prepare(TABLE, str(tmp_path / "cleanup"), columns=COLUMNS,
                   mode=ReadMode.STAGED)
    try:
        left = list(workspace().files.list_directory_contents(
            f"{reader.staging_volume_path}/{TABLE}"))
    except NotFound:
        left = []
    assert left == []


def test_a_streamed_read_needs_no_volume(tmp_path):
    """The mode a customer without write access falls back to."""
    from kumo_databricks import DatabricksSourceReader
    reader = DatabricksSourceReader(connection(), catalog=CATALOG,
                                    schema=SCHEMA)
    assert reader.read_modes(TABLE) == (ReadMode.STREAMED, )
    staged = reader.prepare(TABLE, str(tmp_path / "ungranted"),
                            columns={"c_custkey":
                                     Dtype.int64}, mode=ReadMode.STREAMED)
    assert staged.num_rows == KNOWN_ROWS


def test_a_wide_decimal_is_cast_at_the_source(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_databricks import DatabricksSourceReader
    wide = f"kumo_connector_wide_{uuid.uuid4().hex[:8]}"
    source = f"`{STAGING_CATALOG}`.`{STAGING_SCHEMA}`.`{wide}`"
    cursor = connection().cursor()
    try:
        cursor.execute(f"CREATE TABLE {source} AS SELECT "
                       f"CAST(`c_acctbal` AS DECIMAL(38,2)) AS `balance` "
                       f"FROM `{CATALOG}`.`{SCHEMA}`.`{TABLE}` LIMIT 100")
        reader = DatabricksSourceReader(connection(), catalog=STAGING_CATALOG,
                                        schema=STAGING_SCHEMA)
        staged = reader.prepare(wide, str(tmp_path / "wide"),
                                columns={"balance":
                                         Dtype.float}, mode=ReadMode.STREAMED)
        assert pq.ParquetFile(
            staged.paths[0]).schema_arrow.field("balance").type == pa.float64()
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {source}")
        cursor.close()
