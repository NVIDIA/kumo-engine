# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom SQL Table schema + stype inference (PR-4).

Exercises the engine-side inference path for a Custom SQL Table: dtype from the
query's described result schema (never a physical-table lookup), Option A
stype resolution from the bounded query sample, the warehouse scan-byte cost
gate, and the unsupported-output-dtype error. The warehouse is mocked — a small
``Connector`` subclass supplies ``get_query_metadata`` / ``sample_query`` /
``get_query_scan_estimate_bytes`` and records the calls it receives.
"""
from typing import List, Optional
from unittest.mock import Mock

import pandas as pd
import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import TableMetadataConfig
from kumo.connector.connector import Connector, SourceTableNotFoundError
from kumo.data.custom_sql import custom_sql_synthetic_name
from kumo.graph.column import Column, PrimaryKey
from kumo.graph.table import SQL_SAMPLE_MAX_SCAN_BYTES, Table

_SQL = "SELECT CONCAT(store, '-', sku) AS item_key, price * qty AS total_cost "\
       "FROM sales"


class _FakeSQLConnector(Connector):
    """Minimal SQL-capable connector for inference tests.

    Only the Custom SQL query methods carry behavior; the physical-read
    surface exists to satisfy the ABC and records any (unexpected) use so a
    test can assert the SQL path never resolves the synthetic name physically.
    """
    def __init__(self, name: str, query_cols: List[BaseColumnConfig],
                 sample_df: pd.DataFrame,
                 scan_estimate_bytes: Optional[int] = None,
                 gates_query_scan: bool = False):
        super().__init__(name=name)
        self._query_cols = query_cols
        self._sample_df = sample_df
        self._scan_estimate_bytes = scan_estimate_bytes
        # Only a warehouse that gates is asked for an estimate.
        self.gates_query_scan = gates_query_scan
        # Namespace attributes used by each warehouse's Custom SQL scope gate.
        self.database = 'ANALYTICS'
        self.catalog = 'main'
        self.project_id = 'my-project'
        self.schema_name = 'PUBLIC'
        self.dataset_id = 'Events'
        self.get_query_metadata_calls: List[str] = []
        self.sample_query_calls: List[str] = []
        self.physical_metadata_calls: List[List[str]] = []

    # -- Custom SQL query surface (the path under test) ------------------
    def get_query_metadata(self, sql: str) -> TableMetadataConfig:
        self.get_query_metadata_calls.append(sql)
        return TableMetadataConfig(table_name='', source_name='',
                                   data_source=self.name,
                                   cols=list(self._query_cols))

    def sample_query(self, sql: str, num_rows: int) -> pd.DataFrame:
        self.sample_query_calls.append(sql)
        return self._sample_df.head(num_rows)

    def get_query_scan_estimate_bytes(self, sql: str) -> Optional[int]:
        return self._scan_estimate_bytes

    # -- Physical surface: must NOT be hit for a SQL table ---------------
    def _get_table_metadata(self, table_names, *args, **kwargs):
        self.physical_metadata_calls.append(list(table_names))
        raise AssertionError(
            "physical get_table_metadata must not run for a SQL table")

    def _get_connector_type(self) -> str:
        return "fake_sql"

    def _open(self, credentials=None):
        return True

    def _close(self, context):
        pass

    def list_tables(self, schema_name: Optional[str] = None):
        return []

    def _read_table(self, context, table_name, *args, **kwargs):
        raise AssertionError("physical read must not run for a SQL table")

    def _read_table_chunk(self, context, table_name, *args, **kwargs):
        raise AssertionError("physical read must not run for a SQL table")

    def _read_table_sample(self, context, table_name, *args, **kwargs):
        raise AssertionError("physical read must not run for a SQL table")

    def _get_table_num_rows(self, table_name, data_version=None):
        return None

    def _get_table_source_size(self, table_name, data_version=None):
        return None


def _sql_table(connector: Connector, cols: List[Column],
               pkey: Optional[str] = None) -> Table:
    table = Table(
        table_name="NA",
        data_source="",
        source_name="sales_summary",  # synthetic alias
        cols=cols,
        pkey=PrimaryKey(pkey) if pkey else None,
        custom_sql=_SQL,
    )
    table.connector = connector
    return table


def test_dtype_inferred_from_query_schema_for_computed_columns():
    """dtype comes from the query's described schema for computed/aliased
    output columns, with no physical-table resolution."""
    query_cols = [
        BaseColumnConfig(name="item_key", dtype=Dtype.string),
        BaseColumnConfig(name="total_cost", dtype=Dtype.float32),
    ]
    sample = pd.DataFrame({
        "item_key": [f"s-{i}" for i in range(50)],
        "total_cost": [float(i) for i in range(50)],
    })
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = _sql_table(
        connector,
        cols=[Column("item_key"), Column("total_cost")],
        pkey="item_key",
    )

    table.infer_metadata()

    assert table["item_key"].dtype == Dtype.string
    assert table["total_cost"].dtype == Dtype.float32
    # Computed/aliased pkey resolves to ID from the pkey role, not a lookup.
    assert table.pkey.stype == Stype.ID
    # total_cost has many unique float values -> numerical from the sample.
    assert table["total_cost"].stype == Stype.numerical
    assert connector.get_query_metadata_calls == [_SQL]
    assert connector.physical_metadata_calls == []


def test_option_a_resolves_ambiguous_stype_from_sample():
    """A stats-dependent (ambiguous) column is resolved from the bounded query
    sample — Option A — not from name/dtype heuristics."""
    query_cols = [BaseColumnConfig(name="bucket", dtype=Dtype.int64)]
    # Two distinct values across many rows -> categorical (num_unique <= 3).
    sample = pd.DataFrame({"bucket": pd.Series([1, 2] * 40, dtype="int64")})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = _sql_table(connector, cols=[Column("bucket")])

    table.infer_metadata()

    assert table["bucket"].dtype == Dtype.int64
    assert table["bucket"].stype == Stype.categorical
    assert connector.sample_query_calls == [_SQL]  # sample actually executed


def test_cost_gate_defers_sample_and_falls_back_to_heuristics():
    """When the warehouse scan-byte estimate exceeds the gate, the stype
    sample is skipped and inference falls back to free name/dtype
    heuristics (int -> numerical), never executing the query."""
    query_cols = [BaseColumnConfig(name="bucket", dtype=Dtype.int64)]
    sample = pd.DataFrame({"bucket": pd.Series([1, 2] * 40, dtype="int64")})
    connector = _FakeSQLConnector(
        "wh", query_cols, sample,
        scan_estimate_bytes=SQL_SAMPLE_MAX_SCAN_BYTES + 1,
        gates_query_scan=True)
    table = _sql_table(connector, cols=[Column("bucket")])

    table.infer_metadata()

    # Heuristic fallback for an int column is numerical (not categorical).
    assert table["bucket"].stype == Stype.numerical
    assert connector.sample_query_calls == []  # gate blocked the scan


def test_empty_query_sample_keeps_described_columns_with_heuristic_stypes():
    query_cols = [
        BaseColumnConfig(name="bucket", dtype=Dtype.int64),
        BaseColumnConfig(name="label", dtype=Dtype.string),
    ]
    sample = pd.DataFrame({
        "bucket": pd.Series(dtype="int64"),
        "label": pd.Series(dtype="object"),
    })
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = _sql_table(
        connector,
        cols=[Column("bucket"), Column("label")],
    )

    table.infer_metadata()

    assert table["bucket"].stype == Stype.numerical
    assert table["label"].stype == Stype.categorical
    assert connector.sample_query_calls == [_SQL]


def test_requested_column_absent_from_query_raises_actionable_error():
    """A designated column the query does not produce is rejected with an
    actionable message (the SQL analog of the physical existence check),
    rather than surfacing later as a dtype-less column."""
    query_cols = [BaseColumnConfig(name="item_key", dtype=Dtype.string)]
    sample = pd.DataFrame({"item_key": ["a"]})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = _sql_table(connector,
                       cols=[Column("item_key"),
                             Column("not_selected")], pkey="item_key")

    with pytest.raises(ValueError, match="not produced by the SQL query"):
        table.infer_metadata()


def test_cost_gate_also_defers_timestamp_unit_sample():
    """The scan-byte gate covers the timestamp-unit sample too: a gated query
    keeps the declared time dtype without executing `sample_query`."""
    query_cols = [BaseColumnConfig(name="event_time", dtype=Dtype.time)]
    sample = pd.DataFrame({"event_time": pd.Series([1, 2] * 5, dtype="int64")})
    connector = _FakeSQLConnector(
        "wh", query_cols, sample,
        scan_estimate_bytes=SQL_SAMPLE_MAX_SCAN_BYTES + 1,
        gates_query_scan=True)
    table = _sql_table(connector, cols=[Column("event_time")])

    table.infer_metadata()
    # Stype resolved by heuristic (time -> timestamp), dtype kept, no scan.
    assert table["event_time"].stype == Stype.timestamp
    table.infer_dtypes_from_stypes()
    assert connector.sample_query_calls == []  # gate blocked every scan


def test_a_cost_the_warehouse_would_not_estimate_is_over_budget():
    query_cols = [BaseColumnConfig(name="bucket", dtype=Dtype.int64)]
    sample = pd.DataFrame({"bucket": pd.Series([1, 2] * 40, dtype="int64")})
    connector = _FakeSQLConnector("wh", query_cols, sample,
                                  scan_estimate_bytes=None,
                                  gates_query_scan=True)
    table = _sql_table(connector, cols=[Column("bucket")])

    table.infer_metadata()

    assert table["bucket"].stype == Stype.numerical
    assert connector.sample_query_calls == []


def test_a_warehouse_that_does_not_cost_queries_still_samples():
    query_cols = [BaseColumnConfig(name="bucket", dtype=Dtype.int64)]
    sample = pd.DataFrame({"bucket": pd.Series([1, 2] * 40, dtype="int64")})
    connector = _FakeSQLConnector("wh", query_cols, sample,
                                  scan_estimate_bytes=None,
                                  gates_query_scan=False)
    table = _sql_table(connector, cols=[Column("bucket")])

    table.infer_metadata()

    # Two distinct values over many rows: only the sample can say this.
    assert table["bucket"].stype == Stype.categorical
    assert connector.sample_query_calls == [_SQL]


def test_unsupported_output_dtype_raises_actionable_error():
    """An unsupported output dtype yields an actionable cast-in-SQL error
    rather than a silent warning (as physical tables get)."""
    query_cols = [
        BaseColumnConfig(name="item_key", dtype=Dtype.string),
        BaseColumnConfig(name="weird", dtype=Dtype.unsupported),
    ]
    sample = pd.DataFrame({"item_key": ["a"], "weird": ["?"]})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = _sql_table(connector, cols=[Column("item_key"),
                                        Column("weird")], pkey="item_key")

    with pytest.raises(ValueError, match="unsupported output type"):
        table.infer_metadata()


def test_unsupported_output_dtype_not_masked_by_user_override():
    """A user dtype override must not hide an unsupported warehouse output
    type: the check reads the query's authoritative dtype, not the overridden
    column dtype."""
    query_cols = [
        BaseColumnConfig(name="item_key", dtype=Dtype.string),
        BaseColumnConfig(name="weird", dtype=Dtype.unsupported),
    ]
    sample = pd.DataFrame({"item_key": ["a"], "weird": ["?"]})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    # User "casts" the unsupported column to string in the definition — the
    # warehouse output is still unsupported, so this must not slip through.
    table = _sql_table(
        connector,
        cols=[Column("item_key"),
              Column("weird", dtype=Dtype.string)], pkey="item_key")

    with pytest.raises(ValueError, match="unsupported output type"):
        table.infer_metadata()


def test_sql_schema_preserves_order_and_rejects_duplicates():
    sample = pd.DataFrame({
        "id": range(20),
        "value": range(20),
        "event_ts": pd.date_range("2024-01-01", periods=20)
    })
    cols = [
        BaseColumnConfig(name="id", dtype=Dtype.int64),
        BaseColumnConfig(name="value", dtype=Dtype.int64),
        BaseColumnConfig(name="event_ts", dtype=Dtype.time)
    ]
    connector = _FakeSQLConnector("wh", cols, sample)
    table = Table(table_name="NA", data_source="", source_name="result",
                  pkey="id", time_col="event_ts", custom_sql=_SQL)
    table.connector = connector

    table.infer_metadata()

    assert [col.name for col in table.data_cols] == ["id", "value", "event_ts"]
    assert table.time_col.dtype == Dtype.time

    connector._query_cols.append(cols[1])
    with pytest.raises(ValueError, match="unique alias"):
        table.infer_metadata()


def test_binary_and_list_realized_samples_infer_without_text_decoding():
    cols = [
        BaseColumnConfig(name="payload", dtype=Dtype.binary),
        BaseColumnConfig(name="labels", dtype=Dtype.string)
    ]
    sample = pd.DataFrame({
        "payload": [b"\xff\x00", b"\x80\x01"],
        "labels": [["a", "b"], ["b"]]
    })
    table = _sql_table(_FakeSQLConnector("wh", cols, sample),
                       [Column("payload"), Column("labels")])

    table.infer_metadata()

    assert table["payload"].dtype == Dtype.binary
    assert table["labels"].dtype == Dtype.stringlist
    assert table["labels"].stype == Stype.multicategorical
    assert table["labels"].multicategorical_delimiter is None


# -- Stype inheritance pre-pass (PR-7) -----------------------------------------

_INHERIT_SQL = ("SELECT user_id, amount AS spend, price * qty AS total "
                "FROM orders")


def _inherit_connector() -> _FakeSQLConnector:
    """A warehouse whose query result has a pass-through column (``user_id``),
    an aliased pass-through (``spend`` <- ``amount``), and a computed column
    (``total``), each with many distinct sample values so the *heuristic* would
    resolve them to numerical — making an inherited stype observable."""
    query_cols = [
        BaseColumnConfig(name="user_id", dtype=Dtype.int64),
        BaseColumnConfig(name="spend", dtype=Dtype.float32),
        BaseColumnConfig(name="total", dtype=Dtype.float32),
    ]
    sample = pd.DataFrame({
        "user_id": list(range(50)),
        "spend": [float(i) for i in range(50)],
        "total": [float(i) for i in range(50)],
    })
    return _FakeSQLConnector("wh", query_cols, sample)


def _inherit_table(connector: _FakeSQLConnector, cols: List[Column]) -> Table:
    table = Table(table_name="NA", data_source="", source_name="orders_summary",
                  cols=cols, custom_sql=_INHERIT_SQL)
    table.connector = connector
    return table


def test_stype_inherited_from_referenced_kumo_table():
    """A pass-through output column (direct or aliased) inherits its stype from
    the referenced Kumo table; a computed column falls back to the heuristic."""
    connector = _inherit_connector()
    table = _inherit_table(
        connector,
        [Column("user_id"), Column("spend"),
         Column("total")])
    # ``orders`` is independently a Kumo table; a distinctive ``categorical``
    # stype (the heuristic would pick numerical here) makes inheritance visible.
    table.set_sql_reference_stypes(
        ConnectorType.SNOWFLAKE,
        {"ORDERS": {
            "USER_ID": Stype.categorical,
            "AMOUNT": Stype.categorical
        }})

    table.infer_metadata()

    assert table["user_id"].stype == Stype.categorical  # direct pass-through
    assert table["spend"].stype == Stype.categorical  # aliased pass-through
    assert table["total"].stype == Stype.numerical  # computed -> heuristic


def test_user_stype_override_wins_over_inheritance():
    """An explicitly-set stype is untouched by the inheritance pre-pass."""
    connector = _inherit_connector()
    table = _inherit_table(
        connector,
        [
            Column("user_id"),
            Column("spend", stype=Stype.numerical),  # user override
            Column("total")
        ])
    table.set_sql_reference_stypes(ConnectorType.SNOWFLAKE,
                                   {"ORDERS": {
                                       "AMOUNT": Stype.categorical
                                   }})

    table.infer_metadata()

    # User's numerical stays; inheritance did not overwrite it with categorical.
    assert table["spend"].stype == Stype.numerical


def test_unparseable_sql_inheritance_falls_back_to_heuristic():
    """When the SQL cannot be parsed the inheritance pre-pass resolves nothing
    (no raise) and inference degrades to the heuristic."""
    connector = _inherit_connector()
    table = Table(table_name="NA", data_source="", source_name="orders_summary",
                  cols=[Column("user_id"),
                        Column("spend"),
                        Column("total")], custom_sql="this is (not sql")
    table.connector = connector
    table.set_sql_reference_stypes(ConnectorType.SNOWFLAKE,
                                   {"ORDERS": {
                                       "AMOUNT": Stype.categorical
                                   }})

    table.infer_metadata()  # must not raise

    # Nothing traced, so every column is the heuristic numerical.
    assert table["spend"].stype == Stype.numerical
    assert table["total"].stype == Stype.numerical


def test_incompatible_inherited_stype_is_skipped():
    """An inherited stype incompatible with the query's output dtype is not
    applied (the column falls back to the heuristic instead)."""
    query_cols = [BaseColumnConfig(name="label", dtype=Dtype.string)]
    sample = pd.DataFrame({"label": [f"v{i}" for i in range(50)]})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = Table(table_name="NA", data_source="", source_name="orders_summary",
                  cols=[Column("label")], custom_sql="SELECT label FROM orders")
    table.connector = connector
    # numerical does not support a string dtype, so it must not be inherited.
    table.set_sql_reference_stypes(ConnectorType.SNOWFLAKE,
                                   {"ORDERS": {
                                       "LABEL": Stype.numerical
                                   }})

    table.infer_metadata()

    assert table["label"].stype != Stype.numerical


def test_scalar_multicategorical_inheritance_requires_delimiter():
    query_cols = [BaseColumnConfig(name="label", dtype=Dtype.string)]
    sample = pd.DataFrame({"label": [f"v{i}" for i in range(50)]})
    connector = _FakeSQLConnector("wh", query_cols, sample)
    table = Table(table_name="NA", data_source="", source_name="summary",
                  cols=[Column("label")], custom_sql="SELECT label FROM orders")
    table.connector = connector
    table.set_sql_reference_stypes(
        ConnectorType.SNOWFLAKE, {"ORDERS": {
            "LABEL": Stype.multicategorical
        }})

    table.infer_metadata()

    assert table["label"].stype != Stype.multicategorical
    assert table["label"].multicategorical_delimiter is None


# -- data_version resolution for graph serialization (validate/train) ----------
#
# ``graph.config()`` serializes every table's ``data_version``. For a Custom SQL
# Table that must NOT become a physical ``connector.get_data_version()`` probe of
# the synthetic ``custom_sql_<md5>`` name (which is a SELECT, not a warehouse
# table): the probe 404s on BigQuery and blocks training. The signal is the
# ``is_sql`` marker, which is carried on ``TableConfig`` and so survives the
# validate/train activity's graph round-trip. See ``kumo.graph.Table.data_version``.


class _VersionRecordingConnector(_FakeSQLConnector):
    """SQL connector that records ``get_data_version`` calls so a test can
    assert whether the physical version lookup ran, and returns a sentinel
    keyed by the requested name."""
    def __init__(self, name: str = "wh"):
        super().__init__(name, query_cols=[], sample_df=pd.DataFrame())
        self.get_data_version_calls: List[str] = []

    def get_data_version(self, table_name: str, skip_cache: bool = False,
                         save_to_cache: bool = True) -> str:
        self.get_data_version_calls.append(table_name)
        return f"physical:{table_name}"


def test_sql_table_data_version_is_definition_identity_without_connector():
    """A Custom SQL Table (``is_sql``) resolves ``data_version`` from its SQL
    definition identity -- the synthetic name, which is exactly what the ingest
    ``SqlTableFingerprint`` path-id is built from -- and never probes the
    connector for that synthetic name."""
    connector = _VersionRecordingConnector()
    synthetic = custom_sql_synthetic_name(_SQL, ConnectorType.SNOWFLAKE)
    table = Table(table_name="NA", data_source="", source_name=synthetic,
                  cols=[Column("item_key")], custom_sql=_SQL)
    table.connector = connector

    assert table.is_sql
    assert table.data_version == synthetic
    assert connector.get_data_version_calls == []  # no physical probe


def test_sql_marker_survives_config_roundtrip_without_sql_text():
    """The reported BigQuery crash path: the validate/train activity rebuilds
    the graph via ``Table.from_config``. ``TableConfig`` carries the ``is_sql``
    marker (a bool) but never the SQL text, so the round-tripped table stays
    ``is_sql=True`` with ``custom_sql=None`` and ``data_version`` resolves from
    the synthetic identity with no physical connector probe."""
    synthetic = custom_sql_synthetic_name(_SQL, ConnectorType.BIGQUERY)
    original = Table(table_name="NA", data_source="", source_name=synthetic,
                     cols=[Column("item_key",
                                  dtype=Dtype.string)], custom_sql=_SQL)

    config = original.config()
    # The durable marker is serialized; the raw SQL text is not.
    assert config.is_sql is True
    assert not hasattr(config, "custom_sql")

    roundtripped = Table.from_config(config)
    assert roundtripped.is_sql is True
    assert roundtripped.custom_sql is None  # SQL text intentionally dropped
    assert roundtripped.source_name == synthetic

    # Mirror the pre-materialize activity state: no resolved content version.
    roundtripped._data_version = None
    connector = _VersionRecordingConnector()
    roundtripped.connector = connector

    assert roundtripped.data_version == synthetic
    assert connector.get_data_version_calls == []  # no BigQuery 404 probe


def test_physical_table_config_carries_false_is_sql_marker():
    """A physical table serializes ``is_sql=False`` and round-trips unchanged
    (still probes the connector), so the marker never misclassifies it."""
    original = Table(table_name="NA", data_source="", source_name="real_sales",
                     cols=[Column("item_key", dtype=Dtype.string)])
    config = original.config()
    assert config.is_sql is False

    roundtripped = Table.from_config(config)
    assert roundtripped.is_sql is False
    roundtripped._data_version = None
    connector = _VersionRecordingConnector()
    roundtripped.connector = connector
    assert roundtripped.data_version == "physical:real_sales"
    assert connector.get_data_version_calls == ["real_sales"]


def test_physical_table_data_version_still_probes_connector():
    """A physical (non-Custom-SQL) table is unchanged: ``data_version`` is the
    connector's physical lookup on its real source name."""
    connector = _VersionRecordingConnector()
    table = Table(table_name="NA", data_source="", source_name="real_sales",
                  cols=[Column("item_key")])
    table.connector = connector

    assert not table.is_sql
    assert table.data_version == "physical:real_sales"
    assert connector.get_data_version_calls == ["real_sales"]


def test_missing_physical_table_has_typed_safe_error():
    connector = _VersionRecordingConnector("warehouse")
    connector.get_data_version = Mock(
        side_effect=SourceTableNotFoundError("real_sales", "warehouse"))
    table = Table(table_name="NA", data_source="", source_name="real_sales",
                  cols=[Column("item_key")])
    table.connector = connector

    with pytest.raises(SourceTableNotFoundError) as error:
        _ = table.data_version

    assert "real_sales" in error.value.message
    assert "/internal/path" not in error.value.message
