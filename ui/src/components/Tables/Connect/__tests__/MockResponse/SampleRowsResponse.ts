// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const mockGetSampleRows = {
    "columns": [
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
    "rows": [
        [
            "536365",
            "85123A",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "6",
            "12/1/2010 8:26",
            "2.55",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "71053",
            "WHITE METAL LANTERN",
            "6",
            "12/1/2010 8:26",
            "3.39",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "84406B",
            "CREAM CUPID HEARTS COAT HANGER",
            "8",
            "12/1/2010 8:26",
            "2.75",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "84029G",
            "KNITTED UNION FLAG HOT WATER BOTTLE",
            "6",
            "12/1/2010 8:26",
            "3.39",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "84029E",
            "RED WOOLLY HOTTIE WHITE HEART.",
            "6",
            "12/1/2010 8:26",
            "3.39",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "22752",
            "SET 7 BABUSHKA NESTING BOXES",
            "2",
            "12/1/2010 8:26",
            "7.65",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536365",
            "21730",
            "GLASS STAR FROSTED T-LIGHT HOLDER",
            "6",
            "12/1/2010 8:26",
            "4.25",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536366",
            "22633",
            "HAND WARMER UNION JACK",
            "6",
            "12/1/2010 8:28",
            "1.85",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536366",
            "22632",
            "HAND WARMER RED POLKA DOT",
            "6",
            "12/1/2010 8:28",
            "1.85",
            "17850.0",
            "United Kingdom"
        ],
        [
            "536367",
            "84879",
            "ASSORTED COLOUR BIRD ORNAMENT",
            "32",
            "12/1/2010 8:34",
            "1.69",
            "13047.0",
            "United Kingdom"
        ]
    ],
    "sample_df_serialized": null,
    "response_token": ""
};

export const mockGetCustomerSampleRow = {
    "columns": [
        {
            "name": "CustomerID",
            "stype": "numerical",
            "dtype": "float64",
            "is_primary": false
        }
    ],
    "rows": [
        ["13267.0"],
        ["13894.0"],
        ["13233.0"],
        ["14229.0"],
        ["13174.0"],
        ["14055.0"],
        ["12569.0"],
        ["14965.0"],
        ["16647.0"],
        ["15171.0"]
    ],
    "sample_df_serialized": null,
    "response_token": ""
};
