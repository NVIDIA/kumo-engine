# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.types import ConnectorType
from kumoapi.typing import Dtype

from kumo.data.custom_sql import (
    CustomSqlOutputSchemaError,
    advisory_read_only_check,
    custom_sql_data_version,
    custom_sql_stats_version,
    custom_sql_synthetic_name,
    normalize_sql,
    normalized_sql_identifier,
    normalized_sql_identifier_leaf,
    referenced_tables,
    referenced_tables_or_none,
    source_table_for_col,
    sqlglot_dialect,
    subquery_safe_sql,
    validate_referenced_table_scope,
    validate_sql_output_contract,
)

SF = ConnectorType.SNOWFLAKE
BQ = ConnectorType.BIGQUERY
COLS = {"a": Dtype.int, "b": Dtype.string}


def test_dialect_only_for_supported_warehouses():
    assert sqlglot_dialect(ConnectorType.SNOWFLAKE) == "snowflake"
    assert sqlglot_dialect(ConnectorType.DATABRICKS) == "databricks"
    assert sqlglot_dialect(ConnectorType.BIGQUERY) == "bigquery"
    # File and object-store connectors do not expose a SQL dialect.
    for unsupported in (ConnectorType.S3, ConnectorType.GCS,
                        ConnectorType.FILE):
        with pytest.raises(ValueError):
            sqlglot_dialect(unsupported)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a, b FROM sales",
        "WITH t AS (SELECT id FROM raw) SELECT * FROM t",
        "SELECT 1 UNION SELECT 2",
        # Advisory is fail-open: an unparseable-by-sqlglot query is NOT blocked
        # here — the scoped read-only identity + the warehouse describe are the
        # authoritative guard, so it must reach the warehouse.
        "SELECT FROM WHERE ) ( GROUP",
        "this is (not sql",
    ])
def test_advisory_read_only_check_allows_reads_and_unparseable(sql):
    advisory_read_only_check(sql, SF)  # does not raise


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE t",  # multi-statement
        "DROP TABLE t",
        "CREATE TABLE t AS SELECT 1",
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "WITH x AS (DELETE FROM t) SELECT * FROM x",
    ])
def test_advisory_read_only_check_rejects_obvious_writes(sql):
    with pytest.raises(ValueError):
        advisory_read_only_check(sql, SF)


@pytest.mark.parametrize('sql', [
    'SELECT id, t.name, * FROM sales t',
    'SELECT price * quantity AS total FROM sales',
    'SELECT * EXCEPT(raw) FROM sales',
])
def test_sql_output_contract_accepts_named_flat_columns(sql):
    validate_sql_output_contract(sql, BQ)


@pytest.mark.parametrize('sql', [
    'SELECT 1 + 2',
    'SELECT AS STRUCT 1 AS id',
    'SELECT AS VALUE STRUCT(1 AS id)',
])
def test_sql_output_contract_rejects_implicit_or_value_outputs(sql):
    with pytest.raises(CustomSqlOutputSchemaError):
        validate_sql_output_contract(sql, BQ)


@pytest.mark.parametrize(('sql', 'dialect'), [('SELECT :x', SF),
                                              ('SELECT @x', BQ)])
def test_unbound_parameter_forms_are_rejected(sql, dialect):
    with pytest.raises(ValueError, match='fully bound'):
        advisory_read_only_check(sql, dialect)


def test_referenced_tables_excludes_ctes_and_aliases_keeps_qualified():
    sql = """
        WITH recent AS (SELECT id FROM raw.events)
        SELECT s.id
        FROM DB1.SCH.sales AS s
        JOIN recent ON recent.id = s.id
        JOIN (SELECT id FROM other.dim) sub ON sub.id = s.id
    """
    # CTE ``recent`` and derived-table alias ``sub`` are excluded; the physical
    # tables (including fully-qualified cross-hierarchy names) are captured.
    assert referenced_tables(
        sql, SF) == ["DB1.SCH.sales", "other.dim", "raw.events"]
    assert referenced_tables("SELECT 1", SF) == []


def test_referenced_tables_preserves_case_and_disambiguates_quotes():
    # Case is preserved, not folded, so a case-only table swap reads as a
    # distinct table (conservative — BigQuery table names are case-sensitive).
    assert referenced_tables("SELECT * FROM Sales", SF) == ["Sales"]
    assert referenced_tables("SELECT * FROM SALES", SF) == ["SALES"]
    # A quoted identifier containing a dot stays one identifier, not two parts.
    assert referenced_tables('SELECT * FROM "a.b"."c"', SF) == ['"a.b"."c"']


def test_data_version_stable_under_formatting_but_sensitive_to_changes():
    versions = {"sales": "v1", "raw.events": "v9"}
    canonical = custom_sql_data_version(
        "select A, b from Sales -- note\n where x = 1", SF, versions, COLS)

    # Formatting/comment/case-only edits do not change the fingerprint.
    reformatted = custom_sql_data_version(
        "SELECT a, b\nFROM sales\nWHERE x = 1", SF, versions, COLS)
    assert canonical == reformatted

    # A semantic SQL edit changes it.
    assert canonical != custom_sql_data_version(
        "SELECT a, b FROM sales WHERE x = 2", SF, versions, COLS)

    # A referenced-table version change changes it.
    assert canonical != custom_sql_data_version(
        "select A, b from Sales -- note\n where x = 1", SF, {
            "sales": "v2",
            "raw.events": "v9"
        }, COLS)

    # A cols/dtype-override change changes it (mandatory: the export casts per
    # cols and the fingerprint path-id excludes cols).
    assert canonical != custom_sql_data_version(
        "select A, b from Sales -- note\n where x = 1", SF, versions, {
            "a": Dtype.float,
            "b": Dtype.string
        })
    # Renaming/adding/removing an output column changes it too.
    assert canonical != custom_sql_data_version(
        "select A, b from Sales -- note\n where x = 1", SF, versions,
        {"a": Dtype.int})

    # Ordering of the referenced-version map and cols is irrelevant.
    assert canonical == custom_sql_data_version(
        "select A, b from Sales -- note\n where x = 1", SF, {
            "raw.events": "v9",
            "sales": "v1"
        }, {
            "b": Dtype.string,
            "a": Dtype.int
        })

    # Length-framing keeps a referenced-table version from being confused with
    # SQL text at the version/SQL boundary (unframed concatenation would collide).
    assert (custom_sql_data_version("Y", SF, {"t": "x"}, {})
            != custom_sql_data_version("t=xY", SF, {}, {}))


def test_data_version_max_staleness_bucket_is_isolated_from_table_keyspace():
    versions = {"sales": "v1"}
    base = custom_sql_data_version("SELECT a FROM sales", SF, versions, COLS)

    # Omitting the bucket (``None``) is byte-identical to not having the field:
    # backward-compatible for any caller that imposes no staleness floor.
    assert base == custom_sql_data_version("SELECT a FROM sales", SF, versions,
                                           COLS, max_staleness_bucket=None)

    # A bucket changes the fingerprint, and each distinct window is distinct —
    # this is what makes a window rollover force one re-ingest.
    b100 = custom_sql_data_version("SELECT a FROM sales", SF, versions, COLS,
                                   max_staleness_bucket=100)
    b101 = custom_sql_data_version("SELECT a FROM sales", SF, versions, COLS,
                                   max_staleness_bucket=101)
    assert len({base, b100, b101}) == 3  # all three pairwise-distinct

    # Structural isolation (the reason the bucket is NOT a pseudo-table key): no
    # ``referenced_table_versions`` entry can forge the staleness field. A table
    # literally named ``__max_staleness__`` (the old sentinel key) with a matching
    # version is a DIFFERENT input from the dedicated bucket, so a real warehouse
    # table can never shadow or be conflated with the staleness floor.
    via_forged_key = custom_sql_data_version(
        "SELECT a FROM sales", SF, {
            "sales": "v1",
            "__max_staleness__": "stalebucket:100"
        }, COLS)
    assert via_forged_key != b100
    # The named table and an independent bucket coexist as separate inputs.
    with_both = custom_sql_data_version("SELECT a FROM sales", SF, {
        "sales": "v1",
        "__max_staleness__": "stalebucket:100"
    }, COLS, max_staleness_bucket=100)
    assert with_both != via_forged_key and with_both != b100


def test_stats_version_is_definition_identity_not_data_version():
    canonical = custom_sql_stats_version(
        "select A, b from Sales -- note\n where x = 1", SF, COLS)

    # Formatting/comment/case-only edits reuse the key (stats survive).
    assert canonical == custom_sql_stats_version(
        "SELECT a, b\nFROM sales\nWHERE x = 1", SF, COLS)

    # A semantic SQL edit resets the key.
    assert canonical != custom_sql_stats_version(
        "SELECT a, b FROM sales WHERE x = 2", SF, COLS)

    # A cols/dtype edit resets the key.
    assert canonical != custom_sql_stats_version(
        "select A, b from Sales -- note\n where x = 1", SF, {
            "a": Dtype.float,
            "b": Dtype.string
        })
    assert canonical != custom_sql_stats_version(
        "select A, b from Sales -- note\n where x = 1", SF, {"a": Dtype.int})

    # Unlike custom_sql_data_version, referenced-table versions are NOT part of
    # the key: upstream data churn must not orphan the persisted preview stats.
    # (custom_sql_stats_version has no referenced-version argument at all.)


def test_synthetic_name_stable_under_formatting_only_edits():
    canonical = custom_sql_synthetic_name(
        "select A, b from Sales -- note\n where x = 1", SF)
    # Formatting/comment/case-only edits keep the same synthetic name (so the
    # fingerprint path-id does not churn).
    assert canonical == custom_sql_synthetic_name(
        "SELECT a, b\nFROM sales\nWHERE x = 1", SF)
    # A semantic edit yields a new name (a new definition / staging path).
    assert canonical != custom_sql_synthetic_name(
        "SELECT a, b FROM sales WHERE x = 2", SF)
    assert canonical.startswith("custom_sql_")


def test_bigquery_synthetic_name_preserves_table_identifier_case():
    assert custom_sql_synthetic_name("SELECT id FROM Sales",
                                     BQ) != custom_sql_synthetic_name(
                                         "SELECT id FROM SALES", BQ)


def test_referenced_tables_or_none_distinguishes_parse_failure():
    # A dependency-free query is [] (stable hash); an unparseable query is None
    # (dependencies unknown → callers force a conservative re-materialization).
    assert referenced_tables_or_none("SELECT 1", SF) == []
    assert referenced_tables_or_none("SELECT * FROM sales", SF) == ["sales"]
    assert referenced_tables_or_none("this is (not sql", SF) is None
    # The best-effort variant collapses both empty and unparseable to [].
    assert referenced_tables("this is (not sql", SF) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT a FROM t", "SELECT a FROM t"),
        ("SELECT a FROM t;", "SELECT a FROM t"),
        ("SELECT a FROM t ;  ", "SELECT a FROM t"),
        ("SELECT a FROM t; -- trailing comment", "SELECT a FROM t"),
        ("SELECT a FROM t;\n/* tail */", "SELECT a FROM t"),
        ("SELECT /* h */ a FROM t /* tail */", "SELECT /* h */ a FROM t"),
        # Body (incl. string literals with -- or ; and interior comments) is left
        # byte-identical — the lexer never mistakes them for a trailing terminator.
        ("SELECT ';' AS c FROM t", "SELECT ';' AS c FROM t"),
        ("SELECT '--x' AS c FROM t", "SELECT '--x' AS c FROM t"),
    ])
def test_subquery_safe_sql_strips_only_trailing_terminator(raw, expected):
    assert subquery_safe_sql(raw, SF) == expected


def test_normalize_sql_falls_back_when_unparseable():
    garbage = "this is (not sql"
    assert normalize_sql(garbage, SF) == "this is (not sql"


def test_source_table_for_col_traces_passthrough_columns():
    # Aliased and unqualified pass-through columns resolve to their base table
    # and source column, normalized to the warehouse's case rule.
    sql = "SELECT t.id AS uid, ts FROM transactions t"
    assert source_table_for_col(sql, SF, "uid") == ("TRANSACTIONS", "ID")
    assert source_table_for_col(sql, SF, "ts") == ("TRANSACTIONS", "TS")
    # A join resolves each column to the correct side.
    joined = "SELECT a.id, b.name FROM users a JOIN accounts b ON a.id = b.uid"
    assert source_table_for_col(joined, SF, "name") == ("ACCOUNTS", "NAME")


@pytest.mark.parametrize(
    "sql, col",
    [
        ("SELECT price * qty AS total FROM sales", "total"),  # computed
        ("SELECT UPPER(name) AS name FROM users", "name"),  # function
        ("SELECT * FROM transactions", "id"),  # star
        ("WITH c AS (SELECT * FROM t) SELECT id FROM c", "id"),  # CTE source
        ("SELECT a.id, b.id FROM x a JOIN y b", "id"),  # ambiguous output name
        ("this is (not sql", "id"),  # unparseable
    ])
def test_source_table_for_col_returns_none_when_untraceable(sql, col):
    assert source_table_for_col(sql, SF, col) is None


def test_normalized_sql_identifier_leaf_uses_table_case_rules():
    # Leaf normalization follows per-dialect *table* case rules consistently for
    # both bare and qualified names (Snowflake folds upper, BigQuery preserves,
    # Databricks folds lower), so a qualified reference reconciles with a bare
    # stored name.
    assert normalized_sql_identifier_leaf('Orders', SF) == 'ORDERS'
    assert normalized_sql_identifier_leaf('db.public.Orders', SF) == 'ORDERS'
    assert normalized_sql_identifier_leaf('Orders', BQ) == 'Orders'
    assert normalized_sql_identifier_leaf('proj.ds.Orders', BQ) == 'Orders'
    assert normalized_sql_identifier_leaf('Orders',
                                          ConnectorType.DATABRICKS) == 'orders'


@pytest.mark.parametrize('sql,connector_type,namespace', [
    ('SELECT * FROM orders', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM public.orders', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM analytics.public.orders', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT analytics.public.mask_email(email) FROM orders', SF,
     ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM `my-project.Events.Users`', BQ, ('my-project', 'Events')),
    ('SELECT `my-project.Events.mask_email`(email) FROM Users', BQ,
     ('my-project', 'Events')),
    ("SELECT KEYS.NEW_KEYSET('AEAD_AES_GCM_256') FROM Users", BQ,
     ('my-project', 'Events')),
    ("SELECT AEAD.DECRYPT_STRING(b'key', b'ciphertext', '') FROM Users", BQ,
     ('my-project', 'Events')),
    ('SELECT * FROM main.sales.orders', ConnectorType.DATABRICKS,
     ('main', 'sales')),
    ('SELECT main.sales.mask_email(email) FROM orders',
     ConnectorType.DATABRICKS, ('main', 'sales')),
])
def test_referenced_table_scope_accepts_only_home_namespace(
        sql, connector_type, namespace):
    assert validate_referenced_table_scope(sql, connector_type, namespace)


@pytest.mark.parametrize('sql,connector_type,namespace', [
    ('SELECT * FROM private.orders', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM other.public.orders', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM `my-project.events.Users`', BQ, ('my-project', 'Events')),
    ('SELECT * FROM other.sales.orders', ConnectorType.DATABRICKS,
     ('main', 'sales')),
    ('SELECT other.private.read_secret()', SF, ('ANALYTICS', 'PUBLIC')),
    ('SELECT `other-project.private.read_secret`()', BQ,
     ('my-project', 'Events')),
    ('SELECT `AEAD`.read_secret()', BQ, ('my-project', 'Events')),
    ('SELECT other.private.read_secret()', ConnectorType.DATABRICKS,
     ('main', 'sales')),
])
def test_referenced_table_scope_rejects_cross_namespace_reads(
        sql, connector_type, namespace):
    with pytest.raises(ValueError, match='outside'):
        validate_referenced_table_scope(sql, connector_type, namespace)


def test_referenced_table_scope_fails_closed_when_dependencies_are_unknown():
    with pytest.raises(ValueError, match='cannot be safely verified'):
        validate_referenced_table_scope('this is (not sql', SF,
                                        ('ANALYTICS', 'PUBLIC'))


@pytest.mark.parametrize('sql,connector_type,namespace', [
    ("SELECT * FROM EXTERNAL_QUERY('connection', 'SELECT 1')", BQ,
     ('my-project', 'Events')),
    ("SELECT * FROM read_files('/mnt/private/*.parquet')",
     ConnectorType.DATABRICKS, ('main', 'sales')),
    ('SELECT * FROM TABLE(other.schema.private_function())', SF,
     ('ANALYTICS', 'PUBLIC')),
    ('SELECT * FROM @private_stage/path', SF, ('ANALYTICS', 'PUBLIC')),
])
def test_referenced_table_scope_fails_closed_for_hidden_data_sources(
        sql, connector_type, namespace):
    with pytest.raises(ValueError, match='cannot be safely verified'):
        validate_referenced_table_scope(sql, connector_type, namespace)


def test_source_table_for_col_matches_normalized_reference_name():
    # The traced table name matches ``normalized_sql_identifier`` of the raw
    # reference, so lineage reconciles against referenced Kumo tables per
    # dialect (Snowflake upper-folds; BigQuery table names stay case-sensitive).
    traced = source_table_for_col("SELECT id FROM db.public.transactions", SF,
                                  "id")
    assert traced == (normalized_sql_identifier("db.public.transactions",
                                                SF), "ID")
    bq = source_table_for_col("SELECT userId FROM `proj.ds.Users`", BQ,
                              "userId")
    assert bq == (normalized_sql_identifier("proj.ds.Users", BQ), "userid")
