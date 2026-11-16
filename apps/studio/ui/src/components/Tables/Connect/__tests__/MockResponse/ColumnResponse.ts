// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const mockGetColumnResponse = {
    "source_cols": [
        {
            "name": "InvoiceNo",
            "stype": "categorical",
            "dtype": "string",
            "is_primary": false
        },
        {
            "name": "StockCode",
            "stype": "ID",
            "dtype": "string",
            "is_primary": false
        },
        {
            "name": "Description",
            "stype": "categorical",
            "dtype": "string",
            "is_primary": false
        },
        {
            "name": "Quantity",
            "stype": "numerical",
            "dtype": "int64",
            "is_primary": false
        },
        {
            "name": "InvoiceDate",
            "stype": "multicategorical",
            "dtype": "string",
            "is_primary": false
        },
        {
            "name": "UnitPrice",
            "stype": "numerical",
            "dtype": "float64",
            "is_primary": false
        },
        {
            "name": "CustomerID",
            "stype": "numerical",
            "dtype": "float64",
            "is_primary": false
        },
        {
            "name": "Country",
            "stype": "categorical",
            "dtype": "string",
            "is_primary": false
        }
    ],
    "validation_message": ""
};

export const mockGetCustomerColumnResponse = {
    "source_cols": [
        {
            "name": "CustomerID",
            "stype": "numerical",
            "dtype": "float64",
            "is_primary": false
        }
    ],
    "validation_message": ""
};
