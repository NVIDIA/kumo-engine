// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ADLSSourceTable,
    BigQuerySourceTable,
    DatabricksSourceTable,
    GCSSourceTable,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinitionSourceTable
} from "gen/v0/types";
import { ListResponse } from "gen/v1/types";
import { describe, it } from "vitest";

import { ConnectorConfig } from "types/connectors";
import { DataSourceType, TableKind, TableResponse } from "types/tables";

import { warehouseSource } from "utils/Constants";
import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    parseDate,
    timeDifference
} from "utils/timeConverter";

import {
    getConnectorIDFromTableDefSource,
    getConnectorIDFromTableSource,
    getSourceInfoFromConnectorConfig,
    getSourceNameFromTableDefSource,
    getSourceNameFromTableSource,
    tableFromTableListResp,
    tableFromTableResponse,
    tableListFromListResp
} from "../tablesConverter";

import { mockGetTablesDetails } from "./MockResponse/mockResponses";

const bigQuerySource: BigQuerySourceTable = {
    data_source_type: DataSourceType.BigQuery,
    bigquery_connector_id: "bq-123",
    table_name: "bigquery_table",
    project_id: "project-id",
    dataset_id: "dataset-id"
};

const databricksSource: DatabricksSourceTable = {
    data_source_type: DataSourceType.Databricks,
    databricks_connector_id: "db-456",
    table: "databricks_table"
};

const snowflakeSource: SnowflakeSourceTable = {
    data_source_type: DataSourceType.SnowFlake,
    snowflake_connector_id: "sf-789",
    database: "sf_db",
    schema_name: "sf_schema",
    table: "snowflake_table"
};

const s3SourceWithID: S3SourceTable = {
    data_source_type: DataSourceType.S3,
    file_type: "CSV",
    source: { mode: "relative", path: "s3_table" },
    connector_id: "s3-001"
};

const gcsSourceWithID: GCSSourceTable = {
    data_source_type: DataSourceType.GCS,
    file_type: "PARQUET",
    source: { mode: "relative", path: "deep/gcs_table" },
    connector_id: "gcs-001"
};

const adlsSourceWithID: ADLSSourceTable = {
    data_source_type: DataSourceType.ADLS,
    file_type: "CSV",
    source: { mode: "relative", path: "deep/adls_table.csv" },
    connector_id: "adls-001"
};

function withRequiredConnectorDefaults(
    partial: Record<string, unknown>
): ConnectorConfig {
    return {
        bucket: "",
        location: "",
        project_id: "",
        dataset_id: "",
        database: "",
        schema_name: "",
        warehouse: "",
        cluster_id: "",
        warehouse_id: "",
        catalog: "",
        ...partial
    } as unknown as ConnectorConfig;
}

describe("tableFromTableResponse", () => {
    it("Should return parsed table info with fullTimeStamp = false", () => {
        let result = tableFromTableResponse(
            mockGetTablesDetails as unknown as TableResponse,
            false
        );

        expect(result).toMatchObject({
            id: mockGetTablesDetails.id,
            name_alias: mockGetTablesDetails.name_alias,
            pkey: mockGetTablesDetails.table.pkey,
            timeCol: mockGetTablesDetails.table.time_col,

            endTimeCol: mockGetTablesDetails.table.end_time_col,
            num_cols: parseNumbers(mockGetTablesDetails.table.cols?.length, {
                millionSystem: true
            }) as string | number,
            num_rows: (mockGetTablesDetails?.stats?.num_rows > 0
                ? parseNumbers(mockGetTablesDetails?.stats?.num_rows, {
                      millionSystem: true
                  })
                : "") as string | number,
            size:
                mockGetTablesDetails?.stats?.size_bytes > 0
                    ? bytesToSize(mockGetTablesDetails?.stats?.size_bytes)
                    : "",
            connectorId: mockGetTablesDetails.connector.id,

            earliestFactOrigin: mockGetTablesDetails.time_range.min,
            latestFactOrigin: mockGetTablesDetails.time_range.max
        });

        expect(result.created).toEqual(
            parseDate(mockGetTablesDetails.timestamps.created_at)
        );
        expect(result.updated).toEqual(
            timeDifference(mockGetTablesDetails.timestamps.updated_at)
        );
        //table info with fullTimeStamp = true
        result = tableFromTableResponse(
            mockGetTablesDetails as unknown as TableResponse,
            true
        );
        expect(result.created).toEqual(
            formatToShortLocalDateString(
                mockGetTablesDetails.timestamps.created_at
            )
        );
        expect(result.updated).toEqual(
            formatToLongLocalDateString(
                mockGetTablesDetails.timestamps.updated_at
            )
        );
        expect(result.dateCreated).toEqual(
            formatToLongLocalDateString(
                mockGetTablesDetails.timestamps.created_at
            )
        );
        expect(result.dateUpdated).toEqual(
            formatToLongLocalDateString(
                mockGetTablesDetails.timestamps.updated_at
            )
        );
        //time_range with approx = false
        expect(result.earliestFact).toBe(
            parseDate(mockGetTablesDetails.time_range.min)
        );
        expect(result.latestFact).toBe(
            parseDate(mockGetTablesDetails.time_range.max)
        );
        expect(result.timeRange?.trim()).toBe(
            `${parseDate(mockGetTablesDetails.time_range.min)}  to ${parseDate(mockGetTablesDetails.time_range.max)}`
        );
        //time_range with approx = true
        const approxTrueMock = {
            ...mockGetTablesDetails,
            time_range: {
                ...mockGetTablesDetails.time_range,
                approx: true
            }
        };

        result = tableFromTableResponse(
            approxTrueMock as unknown as TableResponse,
            false
        );

        expect(result.earliestFact).toBe("-");
        expect(result.latestFact).toBe("-");
        expect(result.timeRange).toBeUndefined();
    });
});

describe("SourceTableData utility functions", () => {
    describe("getConnectorIDFromTableSource", () => {
        it.each([
            {
                description: "Should return BigQuery connector ID",
                source: bigQuerySource,
                expected: "bq-123"
            },
            {
                description: "Should return Databricks connector ID",
                source: databricksSource,
                expected: "db-456"
            },
            {
                description: "Should return Snowflake connector ID",
                source: snowflakeSource,
                expected: "sf-789"
            },
            {
                description: "Should return S3 connector ID when present",
                source: s3SourceWithID,
                expected: "s3-001"
            },
            {
                description: "Should return GCS connector ID",
                source: gcsSourceWithID,
                expected: "gcs-001"
            },
            {
                description: "Should return ADLS connector ID",
                source: adlsSourceWithID,
                expected: "adls-001"
            },
            {
                description: "Should return empty string for null source",
                source: null,
                expected: ""
            }
        ])("$description", ({ source, expected }) => {
            expect(getConnectorIDFromTableSource(source)).toBe(expected);
        });

        it("Should return the generic connector_id for a Custom SQL source", () => {
            expect(
                getConnectorIDFromTableSource({
                    data_source_type: "CUSTOM_SQL",
                    connector_id: "csql-001",
                    sql: "SELECT 1"
                } as unknown as Parameters<
                    typeof getConnectorIDFromTableSource
                >[0])
            ).toBe("csql-001");
        });
    });

    describe("getSourceNameFromTableSource", () => {
        it.each([
            {
                description: "Should return BigQuery table name",
                source: bigQuerySource,
                expected: "bigquery_table"
            },
            {
                description: "Should return Databricks table name",
                source: databricksSource,
                expected: "databricks_table"
            },
            {
                description: "Should return Snowflake table name",
                source: snowflakeSource,
                expected: "snowflake_table"
            },
            {
                description: "Should return S3 table name with connector ID",
                source: s3SourceWithID,
                expected: "s3_table"
            },
            {
                description: "Should return GCS table name",
                source: gcsSourceWithID,
                expected: "deep/gcs_table"
            },
            {
                description: "Should return ADLS table name",
                source: adlsSourceWithID,
                expected: "deep/adls_table.csv"
            },
            {
                description: "Should return empty string for null source",
                source: null,
                expected: ""
            }
        ])("$description", ({ source, expected }) => {
            expect(getSourceNameFromTableSource(source)).toBe(expected);
        });
    });
});

describe("getSourceInfoFromConnectorConfig", () => {
    it.each([
        {
            description:
                "Should returns correct source info for BigQuery config",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.BigQuery,
                project_id: "project-id",
                dataset_id: "dataset-id",
                location: "US"
            }),
            expected: {
                bigquery_connector_id: "bq-123",
                table_name: "",
                project_id: "project-id",
                dataset_id: "dataset-id",
                data_source_type: DataSourceType.BigQuery
            },
            connectorId: "bq-123"
        },
        {
            description:
                "Should returns correct source info for Databricks config",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.Databricks,
                cluster_id: "db_id",
                warehouse_id: "db_warehouse",
                catalog: "db_catalog",
                schema_name: "db_schema"
            }),
            expected: {
                databricks_connector_id: "db-456",
                table: "",
                data_source_type: DataSourceType.Databricks
            },
            connectorId: "db-456"
        },
        {
            description:
                "Should returns correct source info for Snowflake config",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.SnowFlake,
                database: "sf_db",
                schema_name: "sf_schema",
                warehouse: "sf_warehouse"
            }),
            expected: {
                snowflake_connector_id: "sf-789",
                database: "sf_db",
                schema_name: "sf_schema",
                table: "",
                data_source_type: DataSourceType.SnowFlake
            },
            connectorId: "sf-789"
        },
        {
            description:
                "Should returns correct source info for S3 config with connector ID",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.S3,
                target: {
                    mode: "rooted",
                    bucket: "bucket",
                    prefix: "file.csv"
                }
            }),
            expected: {
                connector_id: "s3-001",
                source: { mode: "relative", path: "" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.S3
            },
            connectorId: "s3-001"
        },
        {
            description: "Should return correct source info for GCS",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.GCS,
                target: {
                    mode: "rooted",
                    bucket: "bucket",
                    prefix: "root"
                }
            }),
            expected: {
                connector_id: "gcs-001",
                source: { mode: "relative", path: "" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.GCS
            },
            connectorId: "gcs-001"
        },
        {
            description: "Should return correct source info for ADLS",
            config: withRequiredConnectorDefaults({
                type: warehouseSource.ADLS,
                target: {
                    mode: "rooted",
                    storage_account: "account",
                    filesystem: "filesystem",
                    prefix: "root"
                }
            }),
            expected: {
                connector_id: "adls-001",
                source: { mode: "relative", path: "" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.ADLS
            },
            connectorId: "adls-001"
        }
    ])("$description", ({ config, expected, connectorId }) => {
        const result = getSourceInfoFromConnectorConfig(config, connectorId);
        expect(result).toEqual(expected);
    });

    it("Should returns default S3 structure for unknown connector type", () => {
        const config = withRequiredConnectorDefaults({
            type: "UnknownType"
        });

        const expected = {
            connector_id: "unknown-001",
            source: { mode: "relative", path: "" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        };

        const result = getSourceInfoFromConnectorConfig(config, "unknown-001");
        expect(result).toEqual(expected);
    });

    it("Should returns default S3 structure when error is thrown", () => {
        const faultyConfig = {
            get type() {
                throw new Error("Unexpected access error");
            }
        } as unknown as ConnectorConfig;

        const expected = {
            connector_id: "faulty-001",
            source: { mode: "relative", path: "" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        };

        const result = getSourceInfoFromConnectorConfig(
            faultyConfig,
            "faulty-001"
        );
        expect(result).toEqual(expected);
    });
});

describe("getConnectorIDFromTableDefSource", () => {
    it.each([
        {
            description: "Should return BigQuery connector ID",
            source: {
                data_source_type: DataSourceType.BigQuery,
                bigquery_connector_id: "bq-123",
                table_name: "bigquery_table",
                project_id: "project-id",
                dataset_id: "dataset-id"
            },
            expected: "bq-123"
        },
        {
            description: "Should return Databricks connector ID",
            source: {
                data_source_type: DataSourceType.Databricks,
                databricks_connector_id: "db-456",
                table: "databricks_table"
            },
            expected: "db-456"
        },
        {
            description: "Should return Snowflake connector ID",
            source: {
                data_source_type: DataSourceType.SnowFlake,
                snowflake_connector_id: "sf-789",
                database: "sf_db",
                schema_name: "sf_schema",
                table: "snowflake_table"
            },
            expected: "sf-789"
        },
        {
            description: "Should return S3 connector ID when present",
            source: {
                data_source_type: DataSourceType.S3,
                file_type: "CSV",
                source: { mode: "relative", path: "s3_table" },
                connector_id: "s3-001"
            },
            expected: "s3-001"
        },
        {
            description: "Should return GCS connector ID",
            source: gcsSourceWithID,
            expected: "gcs-001"
        },
        {
            description: "Should return ADLS connector ID",
            source: adlsSourceWithID,
            expected: "adls-001"
        },
        {
            description:
                "Should return empty string for S3 without connector ID",
            source: {
                data_source_type: DataSourceType.S3,
                file_type: "CSV",
                source: { mode: "relative", path: "s3_table" }
            },
            expected: ""
        },
        {
            description: "Should return empty string for undefined source",
            source: undefined,
            expected: ""
        }
    ])("$description", ({ source, expected }) => {
        expect(
            getConnectorIDFromTableDefSource(
                source as TableDefinitionSourceTable | undefined
            )
        ).toBe(expected);
    });

    it.each([
        {
            description: "Should return empty string for unknown source type",
            source: {
                data_source_type: "UnknownType",
                some_field: "value"
            },
            expected: ""
        }
    ])("$description", ({ source, expected }) => {
        expect(
            getConnectorIDFromTableDefSource(
                source as unknown as TableDefinitionSourceTable
            )
        ).toBe(expected);
    });
});

describe("getSourceNameFromTableDefSource", () => {
    it.each([
        {
            description: "Should return BigQuery table name",
            source: {
                data_source_type: DataSourceType.BigQuery,
                bigquery_connector_id: "bq-123",
                table_name: "bigquery_table",
                project_id: "project-id",
                dataset_id: "dataset-id"
            },
            expected: "bigquery_table"
        },
        {
            description: "Should return Databricks table name",
            source: {
                data_source_type: DataSourceType.Databricks,
                databricks_connector_id: "db-456",
                table: "databricks_table"
            },
            expected: "databricks_table"
        },
        {
            description: "Should return Snowflake table name",
            source: {
                data_source_type: DataSourceType.SnowFlake,
                snowflake_connector_id: "sf-789",
                database: "sf_db",
                schema_name: "sf_schema",
                table: "snowflake_table"
            },
            expected: "snowflake_table"
        },
        {
            description: "Should return S3 source table name",
            source: {
                data_source_type: DataSourceType.S3,
                file_type: "CSV",
                source: { mode: "relative", path: "s3_table" },
                connector_id: "s3-001"
            },
            expected: "s3_table"
        },
        {
            description: "Should return GCS source table name",
            source: gcsSourceWithID,
            expected: "deep/gcs_table"
        },
        {
            description: "Should return ADLS source table name",
            source: adlsSourceWithID,
            expected: "deep/adls_table.csv"
        },
        {
            description:
                "Should return empty string for S3 without source table name",
            source: {
                data_source_type: DataSourceType.S3,
                file_type: "CSV"
            },
            expected: ""
        },
        {
            description: "Should return empty string for undefined source",
            source: undefined,
            expected: ""
        }
    ])("$description", ({ source, expected }) => {
        expect(
            getSourceNameFromTableDefSource(
                source as TableDefinitionSourceTable | undefined
            )
        ).toBe(expected);
    });

    it.each([
        {
            description: "Should return empty string for unknown source type",
            source: {
                data_source_type: "UnknownType",
                some_field: "value"
            },
            expected: ""
        }
    ])("$description", ({ source, expected }) => {
        expect(
            getSourceNameFromTableDefSource(
                source as unknown as TableDefinitionSourceTable
            )
        ).toBe(expected);
    });
});

describe("tableListFromListResp", () => {
    it("Should map array of ListResponse to TableListResponse", () => {
        const mockListResponses = [
            {
                id: "table-1",
                name: "Users",
                timestamps: {
                    created_at: "2024-01-01T00:00:00Z",
                    updated_at: "2024-01-02T00:00:00Z"
                },
                connector_id: "conn-1",
                source_table_name: "users_source"
            },
            {
                id: "table-2",
                name: "Orders",
                timestamps: {
                    created_at: "2024-02-01T00:00:00Z",
                    updated_at: "2024-02-02T00:00:00Z"
                },
                connector_id: "conn-2",
                source_table_name: "orders_source"
            }
        ];

        const result = tableListFromListResp(
            mockListResponses as unknown as Array<ListResponse>
        );

        expect(result).toHaveLength(2);
        expect(result[0]).toMatchObject({
            id: "table-1",
            name: "Users",
            connector_id: "conn-1",
            source_table_name: "users_source"
        });
        expect(result[1]).toMatchObject({
            id: "table-2",
            name: "Orders",
            connector_id: "conn-2",
            source_table_name: "orders_source"
        });
    });

    it("Should return empty array when given empty array", () => {
        const result = tableListFromListResp([]);
        expect(result).toEqual([]);
    });
});

describe("tableFromTableListResp", () => {
    it("Should correctly transform ListResponse to TableListResponse", () => {
        const mockListResponse = {
            id: "table-1",
            name: "Users",
            timestamps: {
                created_at: "2024-01-01T00:00:00Z",
                updated_at: "2024-01-02T00:00:00Z"
            },
            connector_id: "conn-1",
            source_table_name: "users_source"
        };

        const result = tableFromTableListResp(
            mockListResponse as unknown as ListResponse
        );

        expect(result).toMatchObject({
            id: "table-1",
            name: "Users",
            connector_id: "conn-1",
            source_table_name: "users_source"
        });
        expect(result.dateCreated).toBe(
            formatToShortLocalDateString("2024-01-01T00:00:00Z")
        );
        expect(result.timestamps).toEqual(mockListResponse.timestamps);
    });

    it("Should handle missing name field", () => {
        const mockListResponse = {
            id: "table-1",
            name: null,
            timestamps: {
                created_at: "2024-01-01T00:00:00Z",
                updated_at: "2024-01-02T00:00:00Z"
            },
            connector_id: "conn-1",
            source_table_name: "users_source"
        };

        const result = tableFromTableListResp(
            mockListResponse as unknown as ListResponse
        );

        expect(result.name).toBe("");
    });

    it("Should classify a Custom SQL table from the wire table_kind", () => {
        const mockListResponse = {
            id: "table-1",
            name: "Custom SQL",
            timestamps: {
                created_at: "2024-01-01T00:00:00Z",
                updated_at: "2024-01-02T00:00:00Z"
            },
            connector_id: "conn-1",
            source_table_name: "sql_source",
            // Backend now stamps this (matches rest TableKind.SQL wire value).
            table_kind: "sql"
        };

        const result = tableFromTableListResp(
            mockListResponse as unknown as ListResponse
        );

        expect(result.table_kind).toBe(TableKind.Sql);
    });

    it("Should default table_kind to Raw when the field is absent", () => {
        const mockListResponse = {
            id: "table-1",
            name: "Users",
            timestamps: {
                created_at: "2024-01-01T00:00:00Z",
                updated_at: "2024-01-02T00:00:00Z"
            },
            connector_id: "conn-1",
            source_table_name: "users_source"
        };

        const result = tableFromTableListResp(
            mockListResponse as unknown as ListResponse
        );

        expect(result.table_kind).toBe(TableKind.Raw);
    });

    it("Should classify a Custom SQL table from its custom_sql_ source name when table_kind is absent", () => {
        // The list route (`ListResponse`) carries no `source_table`, so with a
        // backend that doesn't yet stamp `table_kind` the only SQL signal is the
        // server-owned synthetic `source_table_name` (`custom_sql_<hash>`).
        const mockListResponse = {
            id: "table-1",
            name: "account_sql",
            timestamps: {
                created_at: "2024-01-01T00:00:00Z",
                updated_at: "2024-01-02T00:00:00Z"
            },
            connector_id: "conn-1",
            source_table_name: "custom_sql_9810c14892831cd297f65c0308388983"
        };

        const result = tableFromTableListResp(
            mockListResponse as unknown as ListResponse
        );

        expect(result.table_kind).toBe(TableKind.Sql);
    });
});
