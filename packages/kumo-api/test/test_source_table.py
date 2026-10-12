# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from kumoapi.common import ValidationError as ApiValidationError
from kumoapi.data_source import ConnectorType, DataSourceType
from kumoapi.jobs import CustomTrainingTable
from kumoapi.json_serde import from_json, to_json
from kumoapi.source_table import (
    ADLSRelativeSource,
    ADLSSourceTable,
    ADLSURISource,
    CustomSQLSourceTable,
    FileType,
    GCSRelativeSource,
    GCSSourceTable,
    GCSURISource,
    LocalFileSourceTable,
    S3RelativeSource,
    S3SourceTable,
    S3URISource,
    SnowflakeSourceTable,
    SourceTableConfigResponse,
)
from kumoapi.table import (
    ColumnMetadataRequest,
    SqlTableResolveRequest,
    SqlTableResolveResponse,
    TableDefinition,
)
from kumoapi.train import TrainingTableSpec


def test_s3_source_table_uses_explicit_relative_coordinate():
    source_table = S3SourceTable(
        connector_id="connector_0123456789abcdef0123456789abcdef",
        file_type=FileType.PARQUET,
        source=S3RelativeSource(path="2026/07/orders.parquet"),
    )

    assert source_table.table == "orders"
    assert not hasattr(source_table, "s3_path")


def test_s3_source_table_uses_canonical_full_uri_coordinate():
    source_table = S3SourceTable(
        connector_id="connector_0123456789abcdef0123456789abcdef",
        file_type=FileType.PARQUET,
        source=S3URISource(
            uri="s3://company-data/deep/2026/orders%20archive.parquet"),
    )

    assert isinstance(source_table.source, S3URISource)
    assert source_table.source.uri == (
        "s3://company-data/deep/2026/orders%20archive.parquet")
    assert source_table.table == "orders archive"


def test_s3_source_rejects_unknown_fields():
    with pytest.raises(ValidationError, match='checksum_typo'):
        from_json(
            {
                'mode': 'relative',
                'path': '2026/orders.parquet',
                'checksum_typo': 'sha256:example',
            },
            S3RelativeSource,
        )


@pytest.mark.parametrize("source_type", [GCSSourceTable, ADLSSourceTable])
def test_other_object_store_source_table_uses_connector_relative_name(
        source_type):
    relative_type = (GCSRelativeSource
                     if source_type is GCSSourceTable else ADLSRelativeSource)
    source_table = source_type(
        connector_id="connector_0123456789abcdef0123456789abcdef",
        source=relative_type(path="2026/07/orders.parquet"),
        file_type=FileType.PARQUET,
    )

    assert source_table.table == "orders"
    assert not hasattr(source_table, "gcs_path")
    assert not hasattr(source_table, "adls_path")


@pytest.mark.parametrize(
    'source_type',
    [S3SourceTable, GCSSourceTable, ADLSSourceTable],
)
@pytest.mark.parametrize(
    'coordinate,expected_table',
    [
        ('v1.2_events', 'v1.2_events'),
        ('tenant/deep/v1.2_events', 'v1.2_events'),
        ('tenant/deep/v1.2_events/', 'v1.2_events'),
        ('dataset.csv/', 'dataset.csv'),
        ('dataset.parquet/', 'dataset.parquet'),
        ('tenant/deep/dataset.csv/', 'dataset.csv'),
        ('tenant/deep/dataset.parquet/', 'dataset.parquet'),
        ('tenant/deep/v1.2_events/orders.csv', 'orders'),
        ('tenant/deep/v1.2_events/orders.PARQUET', 'orders'),
        ('tenant/deep/v1.2_events/orders.jsonl', 'orders.jsonl'),
    ],
)
def test_object_store_table_name_strips_only_recognized_file_suffixes(
    source_type,
    coordinate,
    expected_table,
):
    kwargs = {
        'connector_id': 'connector_0123456789abcdef0123456789abcdef',
        'file_type': FileType.PARQUET,
    }
    if source_type is S3SourceTable:
        kwargs['source'] = S3RelativeSource(path=coordinate)
    else:
        relative_type = (GCSRelativeSource if source_type is GCSSourceTable
                         else ADLSRelativeSource)
        kwargs['source'] = relative_type(path=coordinate)

    source_table = source_type(**kwargs)

    assert source_table.table == expected_table


def test_s3_uri_table_name_distinguishes_suffix_file_from_directory():
    connector_id = 'connector_0123456789abcdef0123456789abcdef'
    file_source = S3SourceTable(
        connector_id=connector_id,
        source=S3URISource(uri='s3://company-data/deep/dataset.parquet'),
        file_type=FileType.PARQUET,
    )
    directory_source = S3SourceTable(
        connector_id=connector_id,
        source=S3URISource(uri='s3://company-data/deep/dataset.parquet/'),
        file_type=FileType.PARQUET,
    )

    assert file_source.table == 'dataset'
    assert isinstance(directory_source.source, S3URISource)
    assert directory_source.source.uri.endswith('/')
    assert directory_source.table == 'dataset.parquet'


@pytest.mark.parametrize('source_type', [GCSSourceTable, ADLSSourceTable])
@pytest.mark.parametrize(
    'coordinate',
    [
        '',
        '/absolute/table.csv',
        'one//table.csv',
        'one/./table.csv',
        'one/../table.csv',
        'one\\table.csv',
        'one/%2f/table.csv',
        'one/%2e%2e/table.csv',
        'one/%252f/table.csv',
        'one/table.csv?version=1',
        'one/table.csv#fragment',
        'one/\x00/table.csv',
        'one/\x1f/table.csv',
        'one/\x7f/table.csv',
    ],
)
def test_gcs_and_adls_source_coordinates_reject_unsafe_values(
    source_type,
    coordinate,
):
    relative_type = (GCSRelativeSource
                     if source_type is GCSSourceTable else ADLSRelativeSource)
    with pytest.raises((ValidationError, ValueError)):
        source_type(
            connector_id='connector-id',
            source=relative_type(path=coordinate),
            file_type=FileType.PARQUET,
        )


@pytest.mark.parametrize('source_type', [GCSSourceTable, ADLSSourceTable])
def test_gcs_and_adls_source_coordinates_preserve_valid_deep_paths(
        source_type):
    relative_type = (GCSRelativeSource
                     if source_type is GCSSourceTable else ADLSRelativeSource)
    source_table = source_type(
        connector_id='connector-id',
        source=relative_type(
            path='tenant/year=2026/month=07/v1.2_events/data.parquet'),
        file_type=FileType.PARQUET,
    )

    assert source_table.source_table_name == (
        'tenant/year=2026/month=07/v1.2_events/data.parquet')
    assert source_table.table == 'data'


@pytest.mark.parametrize('source_type', [GCSSourceTable, ADLSSourceTable])
def test_gcs_and_adls_source_coordinates_preserve_directory_marker(
        source_type):
    relative_type = (GCSRelativeSource
                     if source_type is GCSSourceTable else ADLSRelativeSource)
    source_table = source_type(
        connector_id='connector-id',
        source=relative_type(path='tenant/deep/dataset.csv/'),
        file_type=FileType.CSV,
    )

    assert source_table.source_table_name == 'tenant/deep/dataset.csv/'
    assert source_table.table == 'dataset.csv'


def test_object_store_source_table_config_response_json_roundtrip():
    response = SourceTableConfigResponse(source_table=ADLSSourceTable(
        connector_id="connector_0123456789abcdef0123456789abcdef",
        file_type=FileType.PARQUET,
        source=ADLSRelativeSource(path="datasets/orders.parquet"),
    ))

    assert from_json(to_json(response), SourceTableConfigResponse) == response


@pytest.mark.parametrize(
    'source_table',
    [
        GCSSourceTable(
            connector_id='connector-gcs',
            file_type=FileType.PARQUET,
            source=GCSRelativeSource(path='deep/orders.parquet'),
        ),
        GCSSourceTable(
            connector_id='connector-gcs-any',
            file_type=FileType.PARQUET,
            source=GCSURISource(
                uri='gs://company-data/deep/orders%20archive.parquet'),
        ),
        ADLSSourceTable(
            connector_id='connector-adls',
            file_type=FileType.CSV,
            source=ADLSRelativeSource(path='deep/orders.csv'),
        ),
        ADLSSourceTable(
            connector_id='connector-adls-any',
            file_type=FileType.CSV,
            source=ADLSURISource(
                uri=('abfss://curated@companylake.dfs.core.windows.net/'
                     'deep/orders%20archive.csv')),
        ),
    ],
)
def test_gcs_and_adls_source_unions_roundtrip(source_table):
    restored = from_json(to_json(source_table), type(source_table))
    assert restored == source_table
    assert restored.table in {'orders', 'orders archive'}


@pytest.mark.parametrize(
    'source_type,payload',
    [
        (GCSSourceTable, {
            'mode': 'uri',
            'uri': 'abfss://curated@companylake.dfs.core.windows.net/a.csv',
        }),
        (ADLSSourceTable, {
            'mode': 'uri',
            'uri': 'gs://company-data/a.csv',
        }),
    ],
)
def test_gcs_and_adls_source_unions_reject_provider_or_mode_mismatch(
        source_type, payload):
    with pytest.raises((ValidationError, ValueError)):
        from_json(
            {
                'connector_id':
                'connector-id',
                'file_type':
                'CSV',
                'source':
                payload,
                'data_source_type':
                ('GCS' if source_type is GCSSourceTable else 'ADLS'),
            }, source_type)


@pytest.mark.parametrize(
    'source_type,payload,expected_source',
    [
        (GCSSourceTable, {
            'connector_id': 'connector-gcs',
            'source_table_name': 'legacy/orders.parquet',
            'file_type': 'PARQUET',
            'data_source_type': 'GCS',
        }, GCSRelativeSource(path='legacy/orders.parquet')),
        (ADLSSourceTable, {
            'connector_id': 'connector-adls',
            'source_table_name': 'legacy/orders.parquet',
            'file_type': 'PARQUET',
            'data_source_type': 'ADLS',
        }, ADLSRelativeSource(path='legacy/orders.parquet')),
    ],
)
def test_legacy_rooted_source_table_records_migrate_to_relative_union(
        source_type, payload, expected_source):
    original_payload = dict(payload)
    restored = from_json(payload, source_type)
    assert restored.source == expected_source
    assert restored.source_table_name == expected_source.path
    assert payload == original_payload


@pytest.mark.parametrize(
    'payload',
    [
        {
            'mode': 'relative',
            'path': 's3://company-data/table.csv'
        },
        {
            'mode': 'relative',
            'path': '/absolute/table.csv'
        },
        {
            'mode': 'relative',
            'path': 'one/../table.csv'
        },
        {
            'mode': 'relative',
            'path': 'one/%2F/table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3a://company-data/table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data'
        },
        {
            'mode': 'uri',
            'uri': 's3://user@company-data/table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data:443/table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/a/../table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/a//table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/a/%2F/table.csv'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/table.csv?signature=x'
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/table.csv#fragment'
        },
        {
            'mode': 'uri',
            'path': 'wrong-field'
        },
    ],
)
def test_s3_source_coordinate_rejects_unsafe_or_incomplete_payload(payload):
    with pytest.raises((ValidationError, ValueError)):
        from_json(
            {
                'connector_id': 'connector-id',
                'file_type': 'CSV',
                'source': payload,
                'data_source_type': 'S3',
            }, S3SourceTable)


@pytest.mark.parametrize(
    'source',
    [
        {
            'mode': 'relative',
            'path': 'valid/table.csv',
            'uri': 's3://company-data/typo.csv',
        },
        {
            'mode': 'uri',
            'uri': 's3://company-data/valid.csv',
            'path': 'typo.csv',
        },
    ],
)
def test_s3_source_rejects_unknown_coordinates(source):
    with pytest.raises(ValidationError, match='(uri|path)'):
        from_json(
            {
                'connector_id': 'connector-id',
                'file_type': 'CSV',
                'source': source,
                'data_source_type': 'S3',
            }, S3SourceTable)


def test_custom_sql_source_table_config_response_json_roundtrip():
    source_table = CustomSQLSourceTable(
        connector_id="connector-id",
        connector_type=ConnectorType.SNOWFLAKE,
        sql="SELECT store_id, price * qty AS total FROM sales",
        table="high_value_sales",
        referenced_tables=["sales"],
    )
    response = SourceTableConfigResponse(source_table=source_table)

    restored = from_json(to_json(response), SourceTableConfigResponse)

    assert restored == response
    assert isinstance(restored.source_table, CustomSQLSourceTable)
    # The union tags Custom SQL on the plain string "CUSTOM_SQL", which is
    # intentionally not a DataSourceType member.
    assert restored.source_table.data_source_type == "CUSTOM_SQL"
    assert restored.source_table.connector_type == ConnectorType.SNOWFLAKE
    assert restored.source_table.sql == source_table.sql
    assert restored.source_table.table == "high_value_sales"
    assert restored.source_table.referenced_tables == ["sales"]


def test_custom_sql_source_table_defaults_server_owned_fields():
    # ``connector_type`` and ``referenced_tables`` are server-owned: a client
    # supplies only ``connector_id``, ``sql``, and ``table``. Omitting the
    # server-owned fields must succeed, defaulting to None / [].
    source_table = CustomSQLSourceTable(
        connector_id="connector-id",
        sql="SELECT 1",
        table="synthetic",
    )

    assert source_table.connector_type is None
    assert source_table.referenced_tables == []
    assert source_table.data_source_type == "CUSTOM_SQL"

    # A minimal object still resolves through the discriminated union and
    # round-trips with the server-owned fields at their defaults.
    response = SourceTableConfigResponse(source_table=source_table)
    restored = from_json(to_json(response), SourceTableConfigResponse)

    assert restored == response
    assert isinstance(restored.source_table, CustomSQLSourceTable)
    assert restored.source_table.connector_type is None
    assert restored.source_table.referenced_tables == []


def test_custom_sql_resolve_contract_roundtrip():
    request = SqlTableResolveRequest(
        source_table=CustomSQLSourceTable("connector-id", "SELECT id",
                                          "pending"),
        column_hints=[ColumnMetadataRequest("id")],
        pkey="id",
        preview_rows=25,
    )
    restored = from_json(to_json(request), SqlTableResolveRequest)
    assert restored == request
    assert SqlTableResolveRequest(
        source_table=request.source_table).column_hints is None
    response = from_json(
        {
            "errors": [{
                "title": "Invalid Custom SQL table",
                "message": "Use an aliased output column.",
            }],
            "future_field":
            True,
        }, SqlTableResolveResponse)
    assert response.table_definition is None
    assert response.errors[0].title == "Invalid Custom SQL table"


def test_custom_sql_resolve_contract_rejects_invalid_states():
    source = CustomSQLSourceTable("connector-id", "SELECT id", "pending")
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveRequest(source_table=source, column_hints=[])
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveRequest(source_table=source, preview_rows=0)
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveRequest(source_table=source, preview_rows=1001)
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveResponse()
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveResponse(
            errors=[
                ApiValidationError("Invalid Custom SQL table",
                                   "Invalid output")
            ],
            table_definition=TableDefinition([], source),
        )
    with pytest.raises((ValidationError, ValueError)):
        SqlTableResolveResponse(
            errors=[
                ApiValidationError("Invalid Custom SQL table",
                                   "Invalid output")
            ],
            preview_rows=[["value"]],
        )


def test_custom_sql_source_table_wire_payload_may_omit_server_owned_fields():
    # A client wire payload that carries only the discriminator tag and the
    # required client input must deserialize, defaulting the server-owned
    # fields. Only ``data_source_type`` is needed for the union to resolve.
    response = from_json(
        {
            "source_table": {
                "connector_id": "connector-id",
                "sql": "SELECT 1",
                "table": "synthetic",
                "data_source_type": "CUSTOM_SQL",
            }
        },
        SourceTableConfigResponse,
    )

    assert isinstance(response.source_table, CustomSQLSourceTable)
    assert response.source_table.connector_type is None
    assert response.source_table.referenced_tables == []
    assert response.source_table.connector_id == "connector-id"
    assert response.source_table.sql == "SELECT 1"
    assert response.source_table.table == "synthetic"


def test_custom_sql_source_table_type_is_decoupled_from_data_source_enum():
    # CUSTOM_SQL is not part of the DataSourceType taxonomy...
    assert not hasattr(DataSourceType, "CUSTOM_SQL")

    # ...yet the discriminated union still tags a Custom SQL table on the wire
    # value "CUSTOM_SQL" and records the backing warehouse separately.
    response = SourceTableConfigResponse(source_table=CustomSQLSourceTable(
        connector_id="connector-id",
        connector_type=ConnectorType.DATABRICKS,
        sql="SELECT 1",
        table="synthetic",
        referenced_tables=["sales"],
    ))
    wire = json.loads(to_json(response))["source_table"]

    assert wire["data_source_type"] == "CUSTOM_SQL"
    assert wire["connector_type"] == "databricks"

    restored = from_json(to_json(response), SourceTableConfigResponse)
    assert isinstance(restored.source_table, CustomSQLSourceTable)
    assert restored.source_table.connector_type == ConnectorType.DATABRICKS


def test_source_table_union_json_schema_export():
    # The mixed enum-literal + plain-string discriminator must still emit a
    # valid JSON schema (OpenAPI export).
    from pydantic import TypeAdapter
    schema = TypeAdapter(SourceTableConfigResponse).json_schema()

    schema_text = json.dumps(schema)
    assert "CUSTOM_SQL" in schema_text  # the decoupled string tag
    assert "connector_type" in schema_text  # the backing-warehouse field
    assert "SNOWFLAKE" in schema_text  # a physical variant tag, still present


def test_snowflake_source_table_config_response_json_roundtrip():
    response = SourceTableConfigResponse(source_table=SnowflakeSourceTable(
        snowflake_connector_id="connector-id",
        database="DB",
        schema_name="SCHEMA",
        table="orders",
    ))

    restored = from_json(to_json(response), SourceTableConfigResponse)

    assert restored == response
    assert isinstance(restored.source_table, SnowflakeSourceTable)
    assert (restored.source_table.data_source_type == DataSourceType.SNOWFLAKE)


def test_a_local_file_source_stays_out_of_the_shared_union():
    custom = CustomTrainingTable(
        source_table=LocalFileSourceTable(path='/tmp/rows.parquet'),
        table_mod_spec=TrainingTableSpec(weight_col='weight'))
    wire = to_json(custom.source_table)

    assert isinstance(
        from_json(to_json(custom), CustomTrainingTable).source_table,
        LocalFileSourceTable)
    with pytest.raises(ValidationError):
        from_json(f'{{"cols": [], "source_table": {wire}}}', TableDefinition)


def test_a_local_file_source_resolves_the_path_it_was_given():
    source = LocalFileSourceTable(path='  rows.parquet ')

    assert source.path.startswith('/')
    assert source.table == 'rows'
