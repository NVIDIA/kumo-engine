// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const mockGetTablesDetail = {
    "id": "18e2b1a6e499fff45a6f249f3024662f",
    "table": {
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
    },
    "name_alias": "transaction_ltv1",
    "timestamps": {
        "created_at": "2025-02-13T04:37:03.949636+00:00",
        "updated_at": "2025-02-13T04:38:11.683211+00:00"
    },
    "stats": {
        "size_bytes": 3560810,
        "num_rows": 541909
    },
    "time_range": {
        "min": "2010-12-01T08:26:00",
        "max": "2011-12-09T12:50:00",
        "approx": false
    },
    "connector": {
        "name": "ltv_test",
        "id": "ltv_test"
    },
    "table_args": null
};

export const mockGetTablesDetailStock = {
    "connector": {
        "name": "ltv_test",
        "id": "ltv_test"
    },
    "id": "NA.NA-b656de36982dfe8ea68ab7ab1ebac6a1",
    "name_alias": "stock",
    "stats": {
        "size_bytes": 21310,
        "num_rows": 4109
    },
    "table": {
        "cols": [
            {
                "name": "StockCode",
                "stype": "ID",
                "dtype": "string",
                "timestamp_format": null
            }
        ],
        "pkey": "StockCode",
        "end_time_col": null,
        "source_table": {
            "connector_id": "ltv_test",
            "data_source_type": "S3",
            "file_type": "PARQUET",
            source: { mode: "relative", path: "stock" }
        }
    },
    "time_col": null,
    "time_range": null,
    "timestamps": {
        "created_at": "2025-02-13T04:37:03.949636+00:00",
        "updated_at": "2025-02-13T04:38:11.683211+00:00"
    }
};

export const mockGetTablesDetailWithAllFields = {
    "connector": {
        "name": "ltv_test",
        "id": "ltv_test"
    },
    "id": "NA.NA-d5828d994e8679119b05ec8f07de920a",
    "name_alias": "transactions_no_time",
    "stats": {
        "size_bytes": 100000,
        "num_rows": 5000
    },
    "table": {
        "cols": [
            {
                "name": "id",
                "stype": "ID",
                "dtype": "string",
                "timestamp_format": null
            },
            {
                "name": "created_at",
                "stype": "timestamp",
                "dtype": "datetime",
                "timestamp_format": null
            },
            {
                "name": "updated_at",
                "stype": "timestamp",
                "dtype": "datetime",
                "timestamp_format": null
            }
        ],
        "pkey": "id",
        "time_col": "created_at",
        "end_time_col": "updated_at",
        "source_table": {
            "connector_id": "ltv_test",
            "data_source_type": "S3",
            "file_type": "PARQUET",
            source: { mode: "relative", path: "transactions" }
        }
    },
    "time_range": {
        "min": "2020-01-01T00:00:00",
        "max": "2024-12-31T23:59:59",
        "approx": false
    },
    "timestamps": {
        "created_at": "2025-02-01T04:22:08.025190+00:00",
        "updated_at": "2025-02-01T07:05:59.215589+00:00"
    }
};
