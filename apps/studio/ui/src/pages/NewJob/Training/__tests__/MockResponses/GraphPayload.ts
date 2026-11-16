// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KumoapiGraphGraphDefinition as GraphDefinition } from "gen/v1/types";

export const expectedGraphPostPayload: GraphDefinition = {
    col_groups: [
        {
            columns: [
                { col_name: "StockCode", table_name: "stock" },
                { col_name: "StockCode", table_name: "transaction_ltv1" }
            ]
        }
    ],
    tables: {
        "transaction_ltv1": {
            cols: [
                {
                    dtype: "string",
                    name: "InvoiceNo",
                    stype: "categorical",
                    timestamp_format: null
                },
                {
                    dtype: "string",
                    name: "StockCode",
                    stype: "ID",
                    timestamp_format: null
                },
                {
                    dtype: "string",
                    name: "Description",
                    stype: "categorical",
                    timestamp_format: null
                },
                {
                    dtype: "int64",
                    name: "Quantity",
                    stype: "numerical",
                    timestamp_format: null
                },
                {
                    dtype: "date",
                    name: "InvoiceDate",
                    stype: "timestamp",
                    timestamp_format: null
                },
                {
                    dtype: "float64",
                    name: "UnitPrice",
                    stype: "numerical",
                    timestamp_format: null
                },
                {
                    dtype: "float64",
                    name: "CustomerID",
                    stype: "numerical",
                    timestamp_format: null
                },
                {
                    dtype: "string",
                    name: "Country",
                    stype: "categorical",
                    timestamp_format: null
                }
            ],
            end_time_col: null,
            pkey: null,
            time_col: null,
            source_table: {
                connector_id: "ltv_test",
                data_source_type: "S3",
                file_type: "PARQUET",
                source: { mode: "relative", path: "transaction" }
            }
        },
        "stock": {
            "cols": [
                {
                    "dtype": "string",
                    "name": "StockCode",
                    "stype": "ID",
                    "timestamp_format": null
                }
            ],
            "end_time_col": null,
            "pkey": "StockCode",
            "source_table": {
                "connector_id": "ltv_test",
                "data_source_type": "S3",
                "file_type": "PARQUET",
                source: { mode: "relative", path: "stock" }
            }
        }
    }
};
