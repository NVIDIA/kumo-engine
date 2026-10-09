// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ADLSSourceTable,
    BigQuerySourceTable,
    DatabricksSourceTable,
    ErrWarnResponse,
    GCSSourceTable,
    ResourceTimestamps,
    S3SourceTable,
    SnowflakeSourceTable
} from "gen/v0/types";
import { GraphLink, KumoapiTableTableDefinition } from "gen/v1/types";

import { Nullable } from "utils/Helpers/typeGuards";

import { ConnectorID } from "./connectors";
import { GraphID } from "./ids";
import { ExtendedColStatsResponse } from "./snapshots";
import { Statistics } from "./stats";

type FactTableTimeRange = {
    min?: string | null;
    max?: string | null;
    approx: boolean;
};

export type TableResponse = {
    id: TableID;
    name_alias?: TableName;
    table: TableDetails;
    timestamps: ResourceTimestamps;
    time_range: FactTableTimeRange | null;
    stats: TableStats;
    connector?: TableConnector;
};

export type TableConnector = {
    id: string;
    name: string;
};

export type SourceTableData =
    | S3SourceTable
    | GCSSourceTable
    | ADLSSourceTable
    | SnowflakeSourceTable
    | DatabricksSourceTable
    | BigQuerySourceTable;

export type TableColData = {
    dtype?: string;
    name?: string;
    stype?: string;
    timestamp_format?: string;
    multicategorical_delimiter?: "," | "|" | ";" | "\t" | null;
};
export type TableDetails = {
    cols: Array<TableColData>;
    source_table: SourceTableData;
    pkey?: string;
    time_col?: string;
    end_time_col?: string;
};

/**
 * Distinguishes a raw warehouse/file table from a Custom SQL table (rows come
 * from a user-defined read-only SELECT). Threaded end-to-end so the list, graph
 * picker, and detail panel can badge/filter SQL tables. Until the generated
 * kumo-api types carry the backend `table_kind`, the converters default to
 * `raw`.
 */
export enum TableKind {
    Raw = "raw",
    Sql = "sql"
}

export type TableListResponse = {
    name: string;
    timestamps: ResourceTimestamps;
    id: string;
    dateCreated?: string;
    connector_id?: string;
    source_table_name?: string;
    table_kind?: TableKind;
};

type TableStats = {
    size_bytes: number;
    num_rows: number;
};

export enum FlowType {
    CREATE = "create",
    EDIT = "edit",
    DUPLICATE = "duplicate",
    AS_TEMPLATE = "use_as_template"
}

export type SourceColumn = {
    name: string;
    stype: Nullable<string>;
    dtype: Nullable<string>;
    is_primary: boolean;
};

export type GetColumnsResponse = [string, SourceColumn[]];

export type CreateTableArgs = {
    cols: Array<TableColData>;
    source_table: SourceTableData;
    pkey?: string;
    time_col?: string;
    end_time_col?: string;
};

export type JSTable = {
    id: TableID;
    name_alias?: TableName;
    table: TableDetails;
    source_table: SourceTableData;
    s3Connector: TableConnector;

    description: string;
    sourceTable: string;
    connector: string | null;
    connectorId: ConnectorID | null;
    table_location: string | null;
    name_link: string | null;
    graphs_original: Array<GraphLink>;
    graphs: Array<GraphLink> | string;

    name_new_tab?: boolean;

    baseConnectorId?: ConnectorID;
    // Repurposed to carry a Custom SQL table's defining SELECT for the
    // view/copy surface in the detail panel.
    viewSql?: string;
    baseTableIds?: Array<string>;
    isView?: boolean;

    table_kind?: TableKind;

    graph_dict?: { [key: string]: GraphID };

    earliestFact?: string;
    earliestFactOrigin?: string;
    latestFact?: string;
    latestFactOrigin?: string;
    timeRange?: string;

    size?: string;
    size_original?: number;
    num_rows?: string | number;
    num_rows_original?: number;

    columns?: Array<ColumnArgs>;
    num_cols?: string | number;
    num_cols_original?: number;

    created?: string;
    dateCreated?: string;
    sortCreated: string;
    updated?: string;
    updatedOrigin?: string;
    dateUpdated?: string;
    sortUpdated: string;
    pkey: Nullable<string>;
    timeCol: Nullable<string>;
    endTimeCol: Nullable<string>;

    tableType?: string;
    output_path?: string;
} & Partial<ErrWarnResponse>;

export type TableID = string;

export type TableName = string;

export type ColumnStatsAndSampleRows = {
    columnStats?: ExtendedColStatsResponse;
    loadingStats: boolean;
    gotError?: boolean;
};

export type ColumnArgs = {
    id?: string;
    name?: string;
    value?: string | number;
    stype?: string;
    dtype?: string;
    sTypeLabel?: string;
    dTypeLabel?: string;
    index?: number;
    statistics?: Statistics;
    timestamp_format?: string;
    multicategorical_delimiter?: "," | "|" | ";" | "\t" | null;
    isActive?: boolean;
    isPrimary?: boolean;
    isTimeColumn?: boolean;
    isEndTimeColumn?: boolean;
    noOfRows?: number;
    nonEmpty?: string | null;
    isSampleAvailable?: boolean;
    userUpdatedType?: boolean;
    disableSelection?: boolean;
    tooltip?: string;
    grad_score?: number;

    sample_rows?: string | null | Array<string | null>;
    properties?: React.JSX.Element;
};

export type TableDefinition = {
    cols: Array<ColumnArgs>;
    pkey?: string;
    time_col?: string;
    end_time_col?: string;
    source_table: SourceTableData;
};

export enum DataSourceType {
    S3 = "S3",
    GCS = "GCS",
    ADLS = "ADLS",
    SnowFlake = "SNOWFLAKE",
    Databricks = "DATABRICKS",
    BigQuery = "BIGQUERY"
}

/** Route state carried into the Add/Edit Table flow. */
export interface ConnectTablesLocationState {
    islocal?: boolean;
    createtableargs?: Partial<KumoapiTableTableDefinition>;
    copytableargs?: Partial<KumoapiTableTableDefinition>;
}

/** Deployment flags that gate which source types are offered. */
export interface SourceTypeDeploymentFlags {
    IS_DATABRICKS_DEPLOY?: boolean;
}
