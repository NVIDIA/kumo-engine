// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GraphTables } from "types/graphs";
import { DataSourceType } from "types/tables";

export const mockListResponse = [
    {
        name: "customer",
        id: "3ba4c08a282125063d56cf3c2c65a7ad",
        dateCreated: "2025-01-23",
        timestamps: {
            created_at: "2025-01-23T12:11:37.556163+00:00",
            updated_at: "2025-04-07T10:17:33.089461+00:00"
        },
        connector_id: "customerltv_integ_test",
        source_table_name: "customer"
    },
    {
        id: "18e2b1a6e499fff45a6f249f3024662f",
        name: "transaction_ltv1",
        dateCreated: "2025-02-13",
        timestamps: {
            created_at: "2025-02-13T12:11:37.556163+00:00",
            updated_at: "2025-05-07T10:17:33.089461+00:00"
        },
        connector_id: "customerltv_integ_test",
        source_table_name: "transaction_ltv1"
    }
];

export const mockAllTables: GraphTables = {
    transaction_ltv1: {
        cols: [
            {
                name: "InvoiceNo",
                stype: "categorical",
                dtype: "string",
                timestamp_format: ""
            },
            {
                name: "StockCode",
                stype: "ID",
                dtype: "string",
                timestamp_format: ""
            },
            {
                name: "Description",
                stype: "categorical",
                dtype: "string",
                timestamp_format: ""
            },
            {
                name: "Quantity",
                stype: "numerical",
                dtype: "int64",
                timestamp_format: ""
            },
            {
                name: "InvoiceDate",
                stype: "multicategorical",
                dtype: "string",
                timestamp_format: ""
            },
            {
                name: "UnitPrice",
                stype: "numerical",
                dtype: "float64",
                timestamp_format: ""
            },
            {
                name: "CustomerID",
                stype: "numerical",
                dtype: "float64",
                timestamp_format: ""
            },
            {
                name: "Country",
                stype: "categorical",
                dtype: "string",
                timestamp_format: ""
            }
        ],
        source_table: {
            connector_id: "ltv_test",
            source: { mode: "relative", path: "transaction" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        },
        pkey: "",
        time_col: "InvoiceDate",
        end_time_col: ""
    },
    customer: {
        cols: [
            {
                name: "CustomerID",
                stype: "ID",
                dtype: "float64",
                timestamp_format: ""
            }
        ],
        source_table: {
            connector_id: "ltv_test",
            source: { mode: "relative", path: "customer" },
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3
        },
        pkey: "CustomerID",
        time_col: "",
        end_time_col: ""
    }
};

export const mockColumnStats = [
    {
        statistics_available: true,
        columns: [
            {
                name: "CustomerID",
                stype: "ID",
                dtype: "Float64",
                sample_rows: ["16685.0", "15970.0", "16875.0"],
                statistics: {
                    counts: {
                        total: 100,
                        missing_percentage: 0.0,
                        num_unique: 100,
                        invalid_percentage: 0.0
                    },
                    summary_stats: [],
                    distribution_points: null,
                    distribution_category: null,
                    distribution_date: null,
                    bp_stats: null,
                    approx: false
                }
            }
        ],
        approx: false,
        rows_used: 100
    },
    false
];
