// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableDefinitionSourceTable } from "gen/v0/types";
import {
    KumoapiSourceTableFileType,
    KumoapiTableTableDefinitionSourceTable,
    ListResponse,
    ResourceTimestamps,
    TableResourceResponse
} from "gen/v1/types";

import { CustomError } from "types/common";
import { ConnectorConfig } from "types/connectors";
import {
    DataSourceType,
    JSTable,
    SourceTableData,
    TableConnector,
    TableDetails,
    TableKind,
    TableListResponse,
    TableResponse
} from "types/tables";

import { warehouseSource } from "utils/Constants";
import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import {
    isAnyUriObjectStoreConnector,
    sourcePathFromObjectStoreSource
} from "utils/objectStoreTablePath";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    parseDate,
    timeDifference
} from "utils/timeConverter";

/**
 * Helper function to convert a Pydantic TableResponse dataclass returned
 * from the REST API into a Table object that can be used in Javascript
 * @param tableResponseJson API response of get Table API
 * @param newTab  Bool flag set open link in new tab or same tab
 */
type AnyTableResponse = TableResponse | TableResourceResponse;

/**
 * Custom SQL field the backend will add but the generated kumo-api types don't
 * carry yet. Read defensively so the scaffold threads `table_kind` end-to-end
 * and lights up automatically once the API supplies it. The SQL definition's
 * "last updated" reuses the standard entry timestamp (Q3), so there is no
 * distinct definition field.
 */
type SqlTableExtras = {
    table_kind?: TableKind;
};

const readTableKind = (kind?: TableKind): TableKind =>
    kind === TableKind.Sql ? TableKind.Sql : TableKind.Raw;

/**
 * The list endpoint (`ListResponse`) has no `source_table`/`data_source_type`,
 * so when the wire omits `table_kind` we classify off the one SQL-only signal it
 * does carry: a Custom SQL table's server-owned synthetic `source_table_name`,
 * which is always prefixed `custom_sql_` (backend `custom_sql_synthetic_name`).
 * Physical tables keep their real source name, so they stay `Raw`. This keeps the
 * Tables "Kind" filter correct even against a backend that doesn't yet stamp
 * `table_kind` on the list rows; the explicit wire value wins when present.
 */
const CUSTOM_SQL_SOURCE_NAME_PREFIX = "custom_sql_";

const readListTableKind = (
    kind?: TableKind,
    sourceTableName?: string
): TableKind =>
    kind === TableKind.Sql ||
    (sourceTableName?.startsWith(CUSTOM_SQL_SOURCE_NAME_PREFIX) ?? false)
        ? TableKind.Sql
        : TableKind.Raw;

/** Identity, schema, and connector fields. */
function assignCoreFields(tableInfo: JSTable, json: AnyTableResponse): void {
    const table = json?.table;
    tableInfo.id = json?.id;
    tableInfo.name_alias = json?.name_alias ?? undefined;
    tableInfo.table = table as TableDetails;
    tableInfo.pkey = table?.pkey;
    tableInfo.timeCol = table?.time_col;
    tableInfo.endTimeCol = table?.end_time_col;
    tableInfo.connectorId = getConnectorIDFromTableSource(
        table?.source_table as SourceTableData
    );
    tableInfo.s3Connector = json?.connector as TableConnector;
    tableInfo.source_table = table?.source_table as SourceTableData;

    const extras = json as SqlTableExtras;
    tableInfo.table_kind = readTableKind(extras?.table_kind);
    // The defining SELECT lives on the Custom SQL source table; surfaced in the
    // detail panel via the repurposed `viewSql` field.
    tableInfo.viewSql = (table?.source_table as { sql?: string })?.sql;
}

/** Row count, byte size, and column count. */
function assignStats(tableInfo: JSTable, json: AnyTableResponse): void {
    const stats = json?.stats;
    const table = json?.table as TableDetails;
    tableInfo.num_rows = (
        stats?.num_rows && stats?.num_rows > 0
            ? parseNumbers(stats?.num_rows, { millionSystem: true })
            : ""
    ) as string | number;
    tableInfo.size =
        stats?.size_bytes && stats?.size_bytes > 0
            ? bytesToSize(stats?.size_bytes)
            : "";
    tableInfo.num_cols = parseNumbers(table.cols?.length, {
        millionSystem: true
    }) as string | number;
}

/** Earliest/latest fact timestamps from the (non-approximate) time range. */
function assignTimeRange(tableInfo: JSTable, json: AnyTableResponse): void {
    tableInfo.earliestFact = "-";
    tableInfo.earliestFactOrigin = "-";
    tableInfo.latestFact = "-";
    tableInfo.latestFactOrigin = "-";
    const tr = json?.time_range;
    if (tr && tr.approx === false) {
        tableInfo.earliestFact = parseDate(tr.min as string);
        tableInfo.earliestFactOrigin = tr.min as string;
        tableInfo.latestFact = parseDate(tr.max as string);
        tableInfo.latestFactOrigin = tr.max as string;
        tableInfo.timeRange = `${parseDate(tr.min as string)}  to ${parseDate(
            tr.max as string
        )} `;
        tableInfo.timeRange = `${tableInfo.earliestFact as string}  to ${
            tableInfo.latestFact as string
        } `;
    }
}

/** Created/updated timestamps, formatted per the fullTimeStamp flag. */
function assignTimestamps(
    tableInfo: JSTable,
    json: AnyTableResponse,
    fullTimeStamp: boolean
): void {
    const ts = json?.timestamps;
    if (fullTimeStamp) {
        tableInfo.created = formatToShortLocalDateString(ts?.created_at);
        tableInfo.dateCreated = formatToLongLocalDateString(ts?.created_at);
        tableInfo.updated = formatToLongLocalDateString(ts?.updated_at);
        tableInfo.dateUpdated = formatToLongLocalDateString(ts?.updated_at);
    } else {
        tableInfo.created = parseDate(ts?.created_at);
        tableInfo.updated = timeDifference(ts?.updated_at);
        tableInfo.updatedOrigin = ts?.updated_at as string;
    }
}

export function tableFromTableResponse(
    tableResponseJson: AnyTableResponse,
    fullTimeStamp: boolean = false
): JSTable {
    const tableInfo: JSTable = {} as JSTable;
    assignCoreFields(tableInfo, tableResponseJson);
    assignStats(tableInfo, tableResponseJson);
    assignTimeRange(tableInfo, tableResponseJson);
    assignTimestamps(tableInfo, tableResponseJson, fullTimeStamp);
    return tableInfo;
}

export function getConnectorIDFromTableSource(
    sourceTable: SourceTableData | null
) {
    // A Custom SQL table's backing connector is its generic `connector_id`
    // (mirrors `getConnectorIDFromTableDefSource` and the server's
    // `get_connector_id`). The generated source-table union doesn't carry this
    // variant yet, so match its discriminator before the typed switch rather
    // than adding an uncomparable `case`.
    if ((sourceTable?.data_source_type as string) === "CUSTOM_SQL") {
        return (sourceTable as { connector_id?: string }).connector_id ?? "";
    }
    switch (sourceTable?.data_source_type) {
        case DataSourceType.BigQuery:
            return sourceTable.bigquery_connector_id;
        case DataSourceType.SnowFlake:
            return sourceTable.snowflake_connector_id;
        case DataSourceType.Databricks:
            return sourceTable.databricks_connector_id;
        case DataSourceType.S3:
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            return sourceTable.connector_id ?? "";
        default:
            return "";
    }
}

export function getConnectorIDFromTableDefSource(
    sourceTable:
        | TableDefinitionSourceTable
        | KumoapiTableTableDefinitionSourceTable
        | undefined
) {
    if (!sourceTable) return "";
    // A Custom SQL table's backing connector is its `connector_id` (mirrors the
    // server's `get_connector_id`). The generated source-table union doesn't
    // carry this variant yet, so match its discriminator before the typed
    // switch rather than adding an uncomparable `case`.
    if ((sourceTable.data_source_type as string) === "CUSTOM_SQL") {
        return (sourceTable as { connector_id?: string }).connector_id ?? "";
    }
    switch (sourceTable?.data_source_type) {
        case DataSourceType.BigQuery:
            return sourceTable.bigquery_connector_id;
        case DataSourceType.SnowFlake:
            return sourceTable.snowflake_connector_id;
        case DataSourceType.Databricks:
            return sourceTable.databricks_connector_id;
        case DataSourceType.S3:
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            return sourceTable.connector_id ?? "";
        default:
            return "";
    }
}

export function getSourceNameFromTableSource(
    sourceTable: SourceTableData | null
) {
    switch (sourceTable?.data_source_type) {
        case DataSourceType.BigQuery:
            return sourceTable.table_name;
        case DataSourceType.SnowFlake:
            return sourceTable.table;
        case DataSourceType.Databricks:
            return sourceTable.table;
        case DataSourceType.S3:
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            return "source" in sourceTable
                ? sourcePathFromObjectStoreSource(sourceTable.source)
                : "";
        default:
            return "";
    }
}

export function getSourceNameFromTableDefSource(
    sourceTable:
        | TableDefinitionSourceTable
        | KumoapiTableTableDefinitionSourceTable
        | undefined
) {
    if (!sourceTable) return "";
    switch (sourceTable?.data_source_type) {
        case DataSourceType.BigQuery:
            return sourceTable.table_name;
        case DataSourceType.SnowFlake:
            return sourceTable.table;
        case DataSourceType.Databricks:
            return sourceTable.table;
        case DataSourceType.S3:
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            return "source" in sourceTable
                ? sourcePathFromObjectStoreSource(sourceTable.source)
                : "";
        default:
            return "";
    }
}

export function getSourceInfoFromConnectorConfig(
    config: ConnectorConfig | null,
    connectorId: string
) {
    try {
        switch (config?.type) {
            case warehouseSource.S3: {
                return {
                    connector_id: connectorId,
                    source:
                        config.target.mode === "any_uri"
                            ? { mode: "uri" as const, uri: "" }
                            : { mode: "relative" as const, path: "" },
                    file_type: KumoapiSourceTableFileType.PARQUET,
                    data_source_type: DataSourceType.S3
                };
            }
            case warehouseSource.GCS: {
                return {
                    connector_id: connectorId,
                    source: isAnyUriObjectStoreConnector(config)
                        ? { mode: "uri" as const, uri: "" }
                        : { mode: "relative" as const, path: "" },
                    file_type: KumoapiSourceTableFileType.PARQUET,
                    data_source_type: DataSourceType.GCS
                };
            }
            case warehouseSource.ADLS: {
                return {
                    connector_id: connectorId,
                    source: isAnyUriObjectStoreConnector(config)
                        ? { mode: "uri" as const, uri: "" }
                        : { mode: "relative" as const, path: "" },
                    file_type: KumoapiSourceTableFileType.PARQUET,
                    data_source_type: DataSourceType.ADLS
                };
            }
            case warehouseSource.SnowFlake:
                return {
                    snowflake_connector_id: connectorId,
                    database: config?.database,
                    schema_name: config.schema_name,
                    table: "",
                    data_source_type: DataSourceType.SnowFlake
                };
            case warehouseSource.BigQuery:
                return {
                    bigquery_connector_id: connectorId,
                    table_name: "",
                    project_id: config?.project_id,
                    dataset_id: config?.dataset_id,
                    data_source_type: DataSourceType.BigQuery
                };
            case warehouseSource.Databricks:
                return {
                    databricks_connector_id: connectorId,
                    table: "",
                    data_source_type: DataSourceType.Databricks
                };
            default:
                return {
                    connector_id: connectorId,
                    source: { mode: "relative", path: "" },
                    file_type: KumoapiSourceTableFileType.PARQUET,
                    data_source_type: DataSourceType.S3
                };
        }
    } catch (error) {
        console.log(error as CustomError);
        return {
            connector_id: connectorId,
            source: { mode: "relative", path: "" },
            file_type: KumoapiSourceTableFileType.PARQUET,
            data_source_type: DataSourceType.S3
        };
    }
}

export function tableListFromListResp(
    tables: Array<ListResponse>
): Array<TableListResponse> {
    return tables.map((table) => tableFromTableListResp(table));
}
/**
 * This method is to create new object<JsConnector> from the response of
 * connector API, that is to be used in the UI Components
 *
 * @param connector Connector response
 * @returns `JsConnector`
 */
export function tableFromTableListResp(table: ListResponse): TableListResponse {
    const extras = table as ListResponse & SqlTableExtras;
    return {
        id: table.id as string,
        name: table.name ?? "",
        dateCreated: formatToShortLocalDateString(
            table.timestamps?.created_at as string
        ),
        timestamps: table.timestamps as ResourceTimestamps,
        connector_id: table.connector_id as string,
        source_table_name: table.source_table_name as string,
        table_kind: readListTableKind(
            extras?.table_kind,
            table.source_table_name ?? undefined
        )
    };
}
