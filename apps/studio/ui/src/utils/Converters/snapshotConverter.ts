// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ADLSSourceTable,
    BigQuerySourceTable,
    DatabricksSourceTable,
    GCSSourceTable,
    GraphSnapshotResponse,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinition,
    TableSnapshotResponse
} from "gen/v0/types";
import { ColumnKey } from "gen/v1/types";

import { ColumnGroup, JSGraph } from "types/graphs";
import { DataSourceType, JSTable, TableConnector } from "types/tables";
import { GraphSnapshotID } from "types/training_job";

import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    parseDate
} from "utils/timeConverter";

import { sourcePathFromS3Source } from "../objectStoreTablePath";

import { getConnectorIDFromTableDefSource } from "./tablesConverter";

export const tableLocationFromSourceTable = (
    sourceTable:
        | S3SourceTable
        | GCSSourceTable
        | ADLSSourceTable
        | SnowflakeSourceTable
        | DatabricksSourceTable
        | BigQuerySourceTable
) => {
    switch (sourceTable?.data_source_type) {
        case DataSourceType.S3:
            return sourcePathFromS3Source(sourceTable.source);
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            return "uri" in sourceTable.source
                ? sourceTable.source.uri
                : sourceTable.source.path;
        case DataSourceType.SnowFlake:
            return sourceTable.snowflake_connector_id;
        case DataSourceType.Databricks:
            return sourceTable.databricks_connector_id;
        case DataSourceType.BigQuery:
            return sourceTable.bigquery_connector_id;
    }
    return null;
};

export const graphSnapshotToJSGraph = (
    snapshotId: GraphSnapshotID,
    graphSnapshot: GraphSnapshotResponse
): JSGraph => {
    if (!graphSnapshot)
        return { id: snapshotId, totalLinks: 0, totalTables: 0 };

    const { snapshot_resource: snapshot, definition } = graphSnapshot;
    const ingest = snapshot?.stages?.INGEST;
    return {
        id: snapshotId,
        name: snapshotId,
        tables: definition?.tables,
        totalTables: Object.keys(snapshot?.table_ids)?.length,
        totalLinks: definition?.col_groups?.reduce(
            (acc: number, colGroup: ColumnGroup) =>
                acc + colGroup?.columns?.length,
            0
        ),
        created: parseDate(ingest?.start_time as string),
        updated: parseDate(ingest?.end_time as string),
        dateCreated: parseDate(ingest?.start_time as string),
        dateUpdated: parseDate(ingest?.end_time as string),
        table_name_to_fpid_mapping: snapshot?.table_ids,
        groups: definition?.col_groups?.map((colGroup: ColumnGroup) => ({
            columns: colGroup?.columns?.map((columnKey: ColumnKey) => ({
                table_name: columnKey?.table_name,
                col_name: columnKey?.col_name
            })),
            state: "saved",
            updated: false
        }))
    };
};

export function tableSnapshotToJSTable(
    { snapshot }: TableSnapshotResponse,
    fullTimeStamp: boolean = false
): JSTable {
    if (!snapshot) return {} as JSTable;

    const {
        id,
        name_alias,
        table_definition,
        connector,
        stats,
        time_range,
        timestamps
    } = snapshot;
    const num_rows = parseNumbers(stats?.num_rows, { millionSystem: true });
    const num_cols = parseNumbers(table_definition?.cols.length, {
        millionSystem: true
    });

    const tableInfo: JSTable = {
        id,
        ...(name_alias && { name_alias }),
        table: table_definition,
        pkey: table_definition?.pkey,
        timeCol: table_definition?.time_col,
        endTimeCol: table_definition?.end_time_col,
        connectorId: getConnectorIDFromTableDefSource(
            table_definition?.source_table
        ),
        s3Connector: connector,
        source_table: table_definition?.source_table,
        ...(num_rows != null && { num_rows }),
        ...(stats?.size_bytes != undefined && {
            size: bytesToSize(stats.size_bytes)
        }),
        ...(num_cols != null && { num_cols }),
        earliestFact: "-",
        earliestFactOrigin: "-",
        latestFact: "-",
        latestFactOrigin: "-"
    } as JSTable;

    if (time_range?.approx === false) {
        const { min, max } = time_range;
        tableInfo.earliestFact = parseDate(min as string);
        tableInfo.earliestFactOrigin = min as string;
        tableInfo.latestFact = parseDate(max as string);
        tableInfo.latestFactOrigin = max as string;
        // tableInfo.timeRange = `${extractDateOnlyFromUTC(min as string)} to ${extractDateOnlyFromUTC(max as string)}`;
        tableInfo.timeRange = `${tableInfo.earliestFact as string} to ${tableInfo.latestFact as string}`;
    }

    if (timestamps) {
        const { created_at, updated_at } = timestamps;
        if (fullTimeStamp) {
            tableInfo.created = formatToShortLocalDateString(created_at);
            tableInfo.dateCreated = formatToLongLocalDateString(created_at);
            tableInfo.updated = formatToLongLocalDateString(updated_at);
            tableInfo.dateUpdated = formatToLongLocalDateString(updated_at);
        } else {
            tableInfo.created = parseDate(created_at);
            tableInfo.updated = parseDate(updated_at);
        }
    }

    return tableInfo;
}

export const graphSnapshotToGraphInformationTables = (
    graphSnapshot: GraphSnapshotResponse
): JSTable[] => {
    if (!graphSnapshot) return [];
    return graphSnapshot.definition?.tables
        ? Object.entries(graphSnapshot.definition?.tables).map(
              ([tableName, table]: [string, TableDefinition]) =>
                  ({
                      id: tableName,
                      name_alias: tableName,
                      pkey: table?.pkey,
                      timeCol: table?.time_col,
                      table: table as TableDefinition,
                      source_table: table?.source_table,
                      s3Connector: {} as TableConnector,
                      endTimeCol: table?.end_time_col,
                      earliestFact: "-",
                      latestFact: "-",
                      latestFactOrigin: "",
                      earliestFactOrigin: "",
                      columns: table.cols,
                      sortCreated: "",
                      sortUpdated: "",
                      allData: null,
                      description: "",
                      sourceTable: "",
                      connector: null,
                      connectorId: null,
                      table_location: null,
                      name_link: null,
                      graphs_original: [],
                      graphs: ""
                  }) as JSTable
          )
        : [];
};
