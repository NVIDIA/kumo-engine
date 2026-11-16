// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const mockNewTable = {
    "cols": [
        {
            "name": "InvoiceNo",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "StockCode",
            "dtype": "string",
            "stype": "ID"
        },
        {
            "name": "Description",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "Quantity",
            "dtype": "int64",
            "stype": "numerical"
        },
        {
            "name": "InvoiceDate",
            "dtype": "date",
            "stype": "timestamp"
        },
        {
            "name": "UnitPrice",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "CustomerID",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "Country",
            "dtype": "string",
            "stype": "categorical"
        }
    ],
    "end_time_col": "",
    "pkey": "",
    "source_table": {
        "connector_id": "ltv_test",
        "data_source_type": "S3",
        "file_type": "PARQUET",
        source: { mode: "relative", path: "transaction" }
    },
    "time_col": "InvoiceDate"
};

export const mockNewEditTable = {
    "cols": [
        {
            "name": "InvoiceNo",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "StockCode",
            "dtype": "string",
            "stype": "ID"
        },
        {
            "name": "Description",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "Quantity",
            "dtype": "int64",
            "stype": "numerical"
        },
        {
            "name": "InvoiceDate",
            "dtype": "date",
            "stype": "timestamp"
        },
        {
            "name": "UnitPrice",
            "dtype": "float32",
            "stype": "numerical"
        },
        {
            "name": "Country",
            "dtype": "string",
            "stype": "categorical"
        }
    ],
    "end_time_col": "",
    "pkey": "",
    "source_table": {
        "connector_id": "ltv_test",
        "data_source_type": "S3",
        "file_type": "PARQUET",
        source: { mode: "relative", path: "transaction" }
    },
    "time_col": "InvoiceDate"
};

export const mockEditTablePreservingRows = {
    "cols": [
        {
            "name": "InvoiceNo",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "StockCode",
            "dtype": "string",
            "stype": "ID"
        },
        {
            "name": "Description",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "Quantity",
            "dtype": "int64",
            "stype": "numerical"
        },
        {
            "name": "InvoiceDate",
            "dtype": "date",
            "stype": "timestamp"
        },
        {
            "name": "UnitPrice",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "CustomerID",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "Country",
            "dtype": "string",
            "stype": "categorical"
        }
    ],
    "end_time_col": "",
    "pkey": "",
    "source_table": {
        "connector_id": "ltv_test",
        "data_source_type": "S3",
        "file_type": "PARQUET",
        source: { mode: "relative", path: "transaction" }
    },
    "time_col": "InvoiceDate"
};

export const mockNewDuplicateTable = {
    "cols": [
        {
            "name": "InvoiceNo",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "StockCode",
            "dtype": "string",
            "stype": "ID"
        },
        {
            "name": "Description",
            "dtype": "string",
            "stype": "categorical"
        },
        {
            "name": "Quantity",
            "dtype": "int64",
            "stype": "numerical"
        },
        {
            "name": "InvoiceDate",
            "dtype": "date",
            "stype": "timestamp"
        },
        {
            "name": "UnitPrice",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "CustomerID",
            "dtype": "float64",
            "stype": "numerical"
        },
        {
            "name": "Country",
            "dtype": "string",
            "stype": "categorical"
        }
    ],
    "end_time_col": "",
    "pkey": "",
    "source_table": {
        "connector_id": "ltv_test",
        "data_source_type": "S3",
        "file_type": "PARQUET",
        source: { mode: "relative", path: "transaction" }
    },
    "time_col": "InvoiceDate"
};

export const mockNewCustomerTableResponse = {
    "cols": [
        {
            "name": "CustomerID",
            "dtype": "float64",
            "stype": "ID"
        }
    ],
    "source_table": {
        "connector_id": "ltv_test",
        source: { mode: "relative", path: "customer" },
        "file_type": "PARQUET",
        "data_source_type": "S3"
    },
    "pkey": "CustomerID",
    "time_col": "",
    "end_time_col": ""
};
