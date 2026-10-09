// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GraphDefinition } from "types/graphs";

export const mockNewGraphResponse = {
    "col_groups": [
        {
            "columns": [
                {
                    "table_name": "stock",
                    "col_name": "StockCode"
                },
                {
                    "table_name": "transaction_ltv1",
                    "col_name": "StockCode"
                }
            ]
        }
    ],
    "tables": {
        "stock": {
            "cols": [
                {
                    "name": "StockCode",
                    "stype": "ID",
                    "dtype": "string",
                    "timestamp_format": null
                }
            ],
            "source_table": {
                "connector_id": "ltv_test",
                source: { mode: "relative", path: "stock" },
                "file_type": "PARQUET",
                "data_source_type": "S3"
            },
            "pkey": "StockCode",
            "end_time_col": null
        },
        "transaction_ltv1": {
            "cols": [
                {
                    "name": "InvoiceNo",
                    "stype": "categorical",
                    "dtype": "string",
                    "timestamp_format": null
                },
                {
                    "name": "StockCode",
                    "stype": "ID",
                    "dtype": "string",
                    "timestamp_format": null
                },
                {
                    "name": "Description",
                    "stype": "categorical",
                    "dtype": "string",
                    "timestamp_format": null
                },
                {
                    "name": "Quantity",
                    "stype": "numerical",
                    "dtype": "int64",
                    "timestamp_format": null
                },
                {
                    "name": "InvoiceDate",
                    "stype": "timestamp",
                    "dtype": "date",
                    "timestamp_format": null
                },
                {
                    "name": "UnitPrice",
                    "stype": "numerical",
                    "dtype": "float64",
                    "timestamp_format": null
                },
                {
                    "name": "CustomerID",
                    "stype": "numerical",
                    "dtype": "float64",
                    "timestamp_format": null
                },
                {
                    "name": "Country",
                    "stype": "categorical",
                    "dtype": "string",
                    "timestamp_format": null
                }
            ],
            "source_table": {
                "connector_id": "ltv_test",
                source: { mode: "relative", path: "transaction" },
                "file_type": "PARQUET",
                "data_source_type": "S3"
            },
            "pkey": null,
            "time_col": null,
            "end_time_col": null
        }
    }
};

export const mockNewEditGraphResponse = {
    "col_groups": [
        {
            "columns": [
                {
                    "col_name": "customer_id",
                    "table_name": "transactions"
                },
                {
                    "col_name": "customer_id",
                    "table_name": "customers"
                }
            ]
        }
    ],
    "tables": {
        "customers": {
            "cols": [
                {
                    "dtype": "string",
                    "name": "customer_id",
                    "stype": "ID",
                    "timestamp_format": null
                },
                {
                    "dtype": "float64",
                    "name": "FN",
                    "stype": "categorical",
                    "timestamp_format": null
                },
                {
                    "dtype": "float64",
                    "name": "Active",
                    "stype": "categorical",
                    "timestamp_format": null
                },
                {
                    "dtype": "string",
                    "name": "club_member_status",
                    "stype": "categorical",
                    "timestamp_format": null
                },
                {
                    "dtype": "string",
                    "name": "fashion_news_frequency",
                    "stype": "categorical",
                    "timestamp_format": null
                },
                {
                    "dtype": "float64",
                    "name": "age",
                    "stype": "numerical",
                    "timestamp_format": null
                },
                {
                    "dtype": "string",
                    "name": "postal_code",
                    "stype": "ID",
                    "timestamp_format": null
                }
            ],
            "end_time_col": null,
            "pkey": "customer_id",
            "source_table": {
                "connector_id": "hnm",
                "data_source_type": "S3",
                "file_type": "PARQUET",
                source: { mode: "relative", path: "customers" }
            },
            "time_col": null
        },
        "transactions": {
            "cols": [
                {
                    "dtype": "date",
                    "name": "t_dat",
                    "stype": "timestamp",
                    "timestamp_format": null
                },
                {
                    "dtype": "string",
                    "name": "customer_id",
                    "stype": "ID",
                    "timestamp_format": null
                },
                {
                    "dtype": "int64",
                    "name": "article_id",
                    "stype": "ID",
                    "timestamp_format": null
                },
                {
                    "dtype": "float64",
                    "name": "price",
                    "stype": "numerical",
                    "timestamp_format": null
                },
                {
                    "dtype": "int64",
                    "name": "sales_channel_id",
                    "stype": "ID",
                    "timestamp_format": null
                }
            ],
            "end_time_col": null,
            "pkey": null,
            "source_table": {
                "connector_id": "hnm_downsampled",
                "data_source_type": "S3",
                "file_type": "PARQUET",
                source: { mode: "relative", path: "transactions" }
            },
            "time_col": "t_dat"
        }
    }
} as unknown as GraphDefinition;

export const mockNewDeuplicateGraphResponse = {
    "col_groups": [
        {
            "columns": [
                {
                    "col_name": "customer_id",
                    "table_name": "transactions"
                },
                {
                    "col_name": "customer_id",
                    "table_name": "customers"
                }
            ]
        }
    ],
    "tables": {
        "customers": mockNewEditGraphResponse.tables.customers,
        "transactions": {
            "cols": [
                {
                    "dtype": "date",
                    "name": "t_dat",
                    "stype": "timestamp",
                    "timestamp_format": null
                },
                {
                    "dtype": "string",
                    "name": "customer_id",
                    "stype": "ID",
                    "timestamp_format": null
                },
                {
                    "dtype": "int64",
                    "name": "article_id",
                    "stype": "ID",
                    "timestamp_format": null
                },
                {
                    "dtype": "float64",
                    "name": "price",
                    "stype": "numerical",
                    "timestamp_format": null
                },
                {
                    "dtype": "int64",
                    "name": "sales_channel_id",
                    "stype": "ID",
                    "timestamp_format": null
                }
            ],
            "end_time_col": null,
            "pkey": null,
            "source_table": {
                "connector_id": "hnm_downsampled",
                "data_source_type": "S3",
                "file_type": "PARQUET",
                source: { mode: "relative", path: "transactions" }
            },
            "time_col": "t_dat"
        }
    }
} as unknown as GraphDefinition;
