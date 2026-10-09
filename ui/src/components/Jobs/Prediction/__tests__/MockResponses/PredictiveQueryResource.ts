// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PQueryResource } from "gen/v0/types";

export const PQueryResourceResponse: PQueryResource = {
    "query_string":
        "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID WHERE FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'",
    "graph": {
        "tables": {
            "FINANCE_TRANS": {
                "cols": [
                    {
                        "name": "TRANS_ID",
                        "stype": "ID",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "ACCOUNT_ID",
                        "stype": "ID",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "DATE",
                        "stype": "timestamp",
                        "dtype": "date",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "TYPE",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "OPERATION",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "AMOUNT",
                        "stype": "numerical",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "BALANCE",
                        "stype": "numerical",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "K_SYMBOL",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "BANK",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "ACCOUNT",
                        "stype": "numerical",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    }
                ],
                "source_table": {
                    "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                    source: { mode: "relative", path: "TRANS" },
                    "file_type": "PARQUET",
                    "data_source_type": "S3"
                },
                "pkey": "TRANS_ID",
                "time_col": "DATE",
                "end_time_col": undefined
            },
            "FINANCE_ACCOUNT": {
                "cols": [
                    {
                        "name": "ACCOUNT_ID",
                        "stype": "ID",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "DISTRICT_ID",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "FREQUENCY",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "DATE",
                        "stype": "timestamp",
                        "dtype": "date",
                        "timestamp_format": undefined
                    }
                ],
                "source_table": {
                    "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                    source: { mode: "relative", path: "ACCOUNT" },
                    "file_type": "PARQUET",
                    "data_source_type": "S3"
                },
                "pkey": "ACCOUNT_ID",
                "time_col": "DATE",
                "end_time_col": undefined
            },
            "FINANCE_ORDERS": {
                "cols": [
                    {
                        "name": "ORDERS_ID",
                        "stype": "ID",
                        "dtype": "int32",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "ACCOUNT_ID",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "BANK_TO",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "ACCOUNT_TO",
                        "stype": "numerical",
                        "dtype": "int64",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "AMOUNT",
                        "stype": "numerical",
                        "dtype": "float64",
                        "timestamp_format": undefined
                    },
                    {
                        "name": "K_SYMBOL",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": undefined
                    }
                ],
                "source_table": {
                    "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                    source: { mode: "relative", path: "ORDERS" },
                    "file_type": "PARQUET",
                    "data_source_type": "S3"
                },
                "pkey": "ORDERS_ID",
                "time_col": undefined,
                "end_time_col": undefined
            }
        },
        "col_groups": [
            {
                "columns": [
                    {
                        "table_name": "FINANCE_TRANS",
                        "col_name": "ACCOUNT_ID"
                    },
                    {
                        "table_name": "FINANCE_ORDERS",
                        "col_name": "ACCOUNT_ID"
                    },
                    {
                        "table_name": "FINANCE_ACCOUNT",
                        "col_name": "ACCOUNT_IsD"
                    }
                ]
            }
        ]
    },
    "name": "pquery",
    "desc": ""
};
