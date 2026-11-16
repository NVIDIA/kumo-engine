// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RestConfigFileTypeFileType } from "gen/v1/types";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import { DataSourceType } from "types/tables";

export const tableWithNoConnector: KumoapiTableTableDefinition = {
    "cols": [
        {
            "name": "InvoiceNo",
            "stype": "categorical",
            "dtype": "string",
            "timestamp_format": undefined
        },
        {
            "name": "StockCode",
            "stype": "ID",
            "dtype": "string",
            "timestamp_format": undefined
        },
        {
            "name": "Description",
            "stype": "categorical",
            "dtype": "string",
            "timestamp_format": undefined
        },
        {
            "name": "Quantity",
            "stype": "numerical",
            "dtype": "int64",
            "timestamp_format": undefined
        },
        {
            "name": "InvoiceDate",
            "stype": "timestamp",
            "dtype": "time",
            "timestamp_format": undefined
        },
        {
            "name": "UnitPrice",
            "stype": "numerical",
            "dtype": "float64",
            "timestamp_format": undefined
        },
        {
            "name": "CustomerID",
            "stype": "numerical",
            "dtype": "float64",
            "timestamp_format": undefined
        },
        {
            "name": "Country",
            "stype": "categorical",
            "dtype": "string",
            "timestamp_format": undefined
        }
    ],
    "source_table": {
        "connector_id": "connector_missing",
        source: { mode: "relative", path: "" },
        "file_type": RestConfigFileTypeFileType.PARQUET,
        "data_source_type": DataSourceType.S3
    },
    "pkey": "",
    "time_col": "InvoiceDate",
    "end_time_col": ""
};
