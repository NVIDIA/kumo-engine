// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ColumnGroup,
    GraphFlowInfo,
    GraphLinkSetupState,
    GraphTables
} from "types/graphs";
import { DataSourceType, FlowType } from "types/tables";

export const baseBackwardLinks = {
    transactions: [
        {
            tableName: "transactions",
            columnName: "article_id",
            pkeyName: "article_id"
        }
    ]
};

export const baseGroups: ColumnGroup[] = [
    {
        columns: [
            {
                table_name: "transactions",
                col_name: "article_id"
            },
            {
                table_name: "articles",
                col_name: "article_id"
            }
        ],
        state: "saved",
        updated: true
    }
];

export const baseGraphFlowInfo: GraphFlowInfo = {
    name: "",
    id: "",
    description: "",
    flow_type: FlowType.CREATE,
    selected_tables: ["transactions", "articles"],
    selected_table_ids: ["", ""],
    links_state: GraphLinkSetupState.EDIT_LINKS,
    tables: {
        transactions: {
            cols: [
                {
                    name: "article_id",
                    stype: "ID",
                    dtype: "string",
                    timestamp_format: ""
                },
                {
                    name: "invoiceNo",
                    stype: "ID",
                    dtype: "string",
                    timestamp_format: ""
                }
            ],
            source_table: {
                connector_id: "test-connector",
                source: { mode: "relative", path: "transactions" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.S3
            }
        },
        articles: {
            pkey: "article_id",
            cols: [
                {
                    name: "article_id",
                    stype: "ID",
                    dtype: "string",
                    timestamp_format: ""
                }
            ],
            source_table: {
                connector_id: "test-connector",
                source: { mode: "relative", path: "articles" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.S3
            }
        }
    }
};

export const baseAllTables: GraphTables = {
    transactions: {
        cols: [
            {
                name: "article_id",
                stype: "ID",
                dtype: "string",
                timestamp_format: ""
            },
            {
                name: "invoiceNo",
                stype: "ID",
                dtype: "string",
                timestamp_format: ""
            }
        ],
        source_table: {
            connector_id: "test-connector",
            source: { mode: "relative", path: "transactions" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        },
        time_col: undefined,
        end_time_col: undefined
        // no pkey
    },
    articles: {
        cols: [
            {
                name: "article_id",
                stype: "ID",
                dtype: "string",
                timestamp_format: ""
            }
        ],
        source_table: {
            connector_id: "test-connector",
            source: { mode: "relative", path: "articles" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        },
        pkey: "article_id",
        time_col: undefined,
        end_time_col: undefined
    }
};

export const mockTablesWithNoPKey: GraphTables = {
    table1: {
        pkey: "", // no primary key
        cols: [],
        source_table: {
            connector_id: "connector-1",
            source: { mode: "relative", path: "table1" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        }
    },
    table2: {
        pkey: "",
        cols: [],
        source_table: {
            connector_id: "connector-2",
            source: { mode: "relative", path: "table2" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        }
    }
};
