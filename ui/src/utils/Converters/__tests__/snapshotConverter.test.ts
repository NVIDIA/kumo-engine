// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    BigQuerySourceTable,
    DatabricksSourceTable,
    GraphSnapshotResponse,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinition,
    TableDefinitionSourceTable,
    TableSnapshotResponse
} from "gen/v0/types";
import { ColumnKey } from "gen/v1/types";
import { describe, expect, it } from "vitest";

import { ColumnGroup } from "types/graphs";
import { DataSourceType } from "types/tables";

import { jobStatus } from "utils/Constants";

import {
    graphSnapshotToGraphInformationTables,
    graphSnapshotToJSGraph,
    tableLocationFromSourceTable,
    tableSnapshotToJSTable
} from "../snapshotConverter";

describe("tableLocationFromSourceTable", () => {
    it.each([
        {
            description: "Should return the path for an S3 source",
            source: {
                data_source_type: DataSourceType.S3,
                connector_id: "s3-123",
                file_type: "CSV",
                source: { mode: "relative", path: "table1" }
            } as S3SourceTable,
            expected: "table1"
        },
        {
            description:
                "Should returns snowflake_connector_id for Snowflake source",
            source: {
                snowflake_connector_id: "sf-123",
                database: "my_db",
                schema_name: "public",
                table: "users",
                data_source_type: DataSourceType.SnowFlake
            } as SnowflakeSourceTable,
            expected: "sf-123"
        },
        {
            description:
                "Should returns databricks_connector_id for Databricks source",
            source: {
                databricks_connector_id: "db-456",
                table: "sales",
                data_source_type: DataSourceType.Databricks
            } as DatabricksSourceTable,
            expected: "db-456"
        },
        {
            description:
                "Should returns bigquery_connector_id for BigQuery source",
            source: {
                bigquery_connector_id: "bq-789",
                table_name: "events",
                project_id: "project-id",
                dataset_id: "dataset-id",
                data_source_type: DataSourceType.BigQuery
            } as BigQuerySourceTable,
            expected: "bq-789"
        },
        {
            description: "Should returns null for undefined input",
            source: undefined as unknown as
                | S3SourceTable
                | SnowflakeSourceTable
                | DatabricksSourceTable
                | BigQuerySourceTable,
            expected: null
        }
    ])("$description", ({ source, expected }) => {
        expect(tableLocationFromSourceTable(source)).toBe(expected);
    });
});

describe("graphSnapshotToGraphInformationTables", () => {
    it("Should correctly transform tables from graphSnapshot", () => {
        const colKey1: ColumnKey = {
            table_name: "table1",
            col_name: "colA"
        };

        const colGroup1: ColumnGroup = {
            columns: [colKey1]
        };

        const ingestStart = "2024-01-01T00:00:00Z";
        const ingestEnd = "2024-01-01T01:00:00Z";
        const tableDef: TableDefinition = {
            pkey: "id",
            time_col: "created_at",
            end_time_col: "updated_at",
            cols: [
                { name: "id", dtype: "int", stype: "ID" },
                { name: "created_at", stype: "timestamp", dtype: "date" }
            ],
            source_table: {
                data_source_type: DataSourceType.S3,
                connector_id: "s3-123",
                file_type: "PARQUET",
                source: { mode: "relative", path: "trans" }
            } as TableDefinitionSourceTable
        };

        const graphSnapshot: GraphSnapshotResponse = {
            snapshot_resource: {
                id: "",
                table_ids: {},
                stages: {
                    ["INGEST"]: {
                        status: jobStatus.Running,
                        start_time: ingestStart,
                        end_time: ingestEnd,
                        warnings: [],
                        error: ""
                    },
                    ["MATERIALIZE"]: {
                        status: jobStatus.Running,
                        start_time: ingestStart,
                        end_time: ingestEnd,
                        warnings: [],
                        error: ""
                    }
                }
            },
            definition: {
                tables: { table1: tableDef },
                col_groups: [colGroup1]
            }
        };

        const result = graphSnapshotToGraphInformationTables(graphSnapshot);

        expect(result).toEqual([
            {
                id: "table1",
                name_alias: "table1",
                pkey: "id",
                s3Connector: {},
                timeCol: "created_at",
                endTimeCol: "updated_at",
                earliestFact: "-",
                latestFact: "-",
                latestFactOrigin: "",
                earliestFactOrigin: "",
                columns: [
                    { name: "id", dtype: "int", stype: "ID" },
                    { name: "created_at", dtype: "date", stype: "timestamp" }
                ],
                sortCreated: "",
                sortUpdated: "",
                allData: null,
                description: "",
                sourceTable: "",
                source_table: {
                    connector_id: "s3-123",
                    data_source_type: "S3",
                    file_type: "PARQUET",
                    source: { mode: "relative", path: "trans" }
                },
                table: {
                    cols: [
                        {
                            name: "id",
                            dtype: "int",
                            stype: "ID"
                        },
                        {
                            name: "created_at",
                            dtype: "date",
                            stype: "timestamp"
                        }
                    ],
                    end_time_col: "updated_at",
                    pkey: "id",
                    source_table: {
                        connector_id: "s3-123",
                        data_source_type: "S3",
                        file_type: "PARQUET",
                        source: { mode: "relative", path: "trans" }
                    },
                    time_col: "created_at"
                },
                connector: null,
                connectorId: null,
                table_location: null,
                name_link: null,
                graphs_original: [],
                graphs: ""
            }
        ]);
    });

    it("Should return empty array when graphSnapshot is null", () => {
        const result = graphSnapshotToGraphInformationTables(
            null as unknown as GraphSnapshotResponse
        );
        expect(result).toEqual([]);
    });

    it("Should return empty array when definition.tables is undefined", () => {
        const graphSnapshot = {
            snapshot_resource: {
                id: "",
                table_ids: {},
                stages: {}
            },
            definition: {
                col_groups: []
            }
        } as unknown as GraphSnapshotResponse;
        const result = graphSnapshotToGraphInformationTables(graphSnapshot);
        expect(result).toEqual([]);
    });
});

describe("graphSnapshotToJSGraph", () => {
    it("Should return default object when graphSnapshot is null", () => {
        const result = graphSnapshotToJSGraph(
            "snapshot-123",
            null as unknown as GraphSnapshotResponse
        );
        expect(result).toEqual({
            id: "snapshot-123",
            totalLinks: 0,
            totalTables: 0
        });
    });

    it("Should correctly transform graphSnapshot to JSGraph", () => {
        const colKey1: ColumnKey = {
            table_name: "table1",
            col_name: "col1"
        };
        const colKey2: ColumnKey = {
            table_name: "table2",
            col_name: "col2"
        };

        const colGroup1: ColumnGroup = {
            columns: [colKey1, colKey2]
        };

        const ingestStart = "2024-01-01T00:00:00Z";
        const ingestEnd = "2024-01-01T01:00:00Z";

        const graphSnapshot: GraphSnapshotResponse = {
            snapshot_resource: {
                id: "snap-1",
                table_ids: { table1: "fpid1", table2: "fpid2" },
                stages: {
                    INGEST: {
                        status: jobStatus.Running,
                        start_time: ingestStart,
                        end_time: ingestEnd,
                        warnings: [],
                        error: ""
                    }
                }
            },
            definition: {
                tables: {},
                col_groups: [colGroup1]
            }
        };

        const result = graphSnapshotToJSGraph("snapshot-123", graphSnapshot);

        expect(result.id).toBe("snapshot-123");
        expect(result.name).toBe("snapshot-123");
        expect(result.totalTables).toBe(2);
        expect(result.totalLinks).toBe(2);
        expect(result.table_name_to_fpid_mapping).toEqual({
            table1: "fpid1",
            table2: "fpid2"
        });
        expect(result.groups).toHaveLength(1);
        expect(result.groups?.[0].columns).toEqual([
            { table_name: "table1", col_name: "col1" },
            { table_name: "table2", col_name: "col2" }
        ]);
    });
});

describe("tableSnapshotToJSTable", () => {
    it("Should return empty object when snapshot is null", () => {
        const tableSnapshot = {
            snapshot: null
        } as unknown as TableSnapshotResponse;

        const result = tableSnapshotToJSTable(tableSnapshot);
        expect(result).toEqual({});
    });

    it("Should correctly transform tableSnapshot with fullTimeStamp=false", () => {
        const tableSnapshot = {
            snapshot: {
                id: "table-1",
                name_alias: "users",
                table_definition: {
                    pkey: "id",
                    time_col: "created_at",
                    end_time_col: "updated_at",
                    cols: [
                        { name: "id", dtype: "int", stype: "ID" },
                        { name: "name", dtype: "string", stype: "categorical" }
                    ],
                    source_table: {
                        data_source_type: DataSourceType.S3,
                        connector_id: "s3-123",
                        file_type: "CSV",
                        source: { mode: "relative", path: "users" }
                    } as TableDefinitionSourceTable
                },
                connector: {},
                stats: {
                    num_rows: 1000,
                    size_bytes: 50000
                },
                time_range: {
                    min: "2024-01-01T00:00:00Z",
                    max: "2024-12-31T23:59:59Z",
                    approx: false
                },
                timestamps: {
                    created_at: "2024-01-01T00:00:00Z",
                    updated_at: "2024-01-02T00:00:00Z"
                }
            }
        } as unknown as TableSnapshotResponse;

        const result = tableSnapshotToJSTable(tableSnapshot, false);

        expect(result.id).toBe("table-1");
        expect(result.name_alias).toBe("users");
        expect(result.pkey).toBe("id");
        expect(result.timeCol).toBe("created_at");
        expect(result.endTimeCol).toBe("updated_at");
        expect(result.num_rows).toBeDefined();
        expect(result.size).toBeDefined();
        expect(result.num_cols).toBeDefined();
        expect(result.earliestFact).not.toBe("-");
        expect(result.latestFact).not.toBe("-");
        expect(result.timeRange).toBeDefined();
        expect(result.created).toBeDefined();
        expect(result.updated).toBeDefined();
    });

    it("Should correctly transform tableSnapshot with fullTimeStamp=true", () => {
        const tableSnapshot = {
            snapshot: {
                id: "table-1",
                name_alias: "users",
                table_definition: {
                    pkey: "id",
                    time_col: "created_at",
                    cols: [{ name: "id", dtype: "int", stype: "ID" }],
                    source_table: {
                        data_source_type: DataSourceType.S3,
                        connector_id: "s3-123",
                        file_type: "CSV",
                        source: { mode: "relative", path: "users" }
                    } as TableDefinitionSourceTable
                },
                connector: {},
                stats: {},
                time_range: {
                    approx: true
                },
                timestamps: {
                    created_at: "2024-01-01T00:00:00Z",
                    updated_at: "2024-01-02T00:00:00Z"
                }
            }
        } as unknown as TableSnapshotResponse;

        const result = tableSnapshotToJSTable(tableSnapshot, true);

        expect(result.created).toBeDefined();
        expect(result.dateCreated).toBeDefined();
        expect(result.updated).toBeDefined();
        expect(result.dateUpdated).toBeDefined();
    });

    it("Should handle approx=true in time_range", () => {
        const tableSnapshot = {
            snapshot: {
                id: "table-1",
                table_definition: {
                    pkey: "id",
                    cols: [],
                    source_table: {
                        data_source_type: DataSourceType.S3,
                        connector_id: "s3-123",
                        file_type: "CSV",
                        source: { mode: "relative", path: "table" }
                    } as TableDefinitionSourceTable
                },
                connector: {},
                stats: {},
                time_range: {
                    min: "2024-01-01T00:00:00Z",
                    max: "2024-12-31T23:59:59Z",
                    approx: true
                },
                timestamps: {
                    created_at: "2024-01-01T00:00:00Z",
                    updated_at: "2024-01-02T00:00:00Z"
                }
            }
        } as unknown as TableSnapshotResponse;

        const result = tableSnapshotToJSTable(tableSnapshot);

        expect(result.earliestFact).toBe("-");
        expect(result.latestFact).toBe("-");
        expect(result.timeRange).toBeUndefined();
    });
});
