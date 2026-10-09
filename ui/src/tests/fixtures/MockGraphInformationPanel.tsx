// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnGroup, GraphFlowInfo, GraphTables, JSGraph } from "types/graphs";
import { DataSourceType, FlowType, JSTable } from "types/tables";

import { tablesFromGraphResponse } from "utils/Converters/graphsConverter";

export const graphInfo: JSGraph = {
    "id": "graph-eb2650cd1f7c522fc8d513b7965b3f51",
    "name": "customer_graph",
    "tables": {
        "stock": {
            "cols": [
                {
                    "name": "StockCode",
                    "stype": "ID",
                    "dtype": "string",
                    "timestamp_format": undefined
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "stock" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "StockCode",
            "time_col": undefined,
            "end_time_col": undefined
        },
        "transaction": {
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
                    "dtype": "date",
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
                "connector_id": "test",
                source: { mode: "relative", path: "transaction" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": undefined,
            "time_col": "InvoiceDate",
            "end_time_col": undefined
        },
        "customer": {
            "cols": [
                {
                    "name": "CustomerID",
                    "stype": "ID",
                    "dtype": "float64",
                    "timestamp_format": undefined
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "customer" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "CustomerID",
            "time_col": undefined,
            "end_time_col": undefined
        }
    },
    "groups": [
        {
            "columns": [
                {
                    "table_name": "stock",
                    "col_name": "StockCode"
                },
                {
                    "table_name": "transaction",
                    "col_name": "StockCode"
                }
            ]
        },
        {
            "columns": [
                {
                    "table_name": "transaction",
                    "col_name": "CustomerID"
                },
                {
                    "table_name": "customer",
                    "col_name": "CustomerID"
                }
            ]
        }
    ],
    "totalLinks": 2,
    "totalTables": 3,
    "table_name_to_fpid_mapping": {
        "stock": "NA.NA-c8e37d6b963bfcfc5dd691522eca30f6",
        "transaction": "NA.NA-4b1de19e99d03ba91b2f937b2a4292e1",
        "customer": "NA.NA-039bf1683552e2e227a7590644451b7f"
    },
    "created": "2 months ago",
    "dateCreated": "Tue Mar 18, 2025, 12:11:37 PM",
    "updated": "1 month ago",
    "dateUpdated": "Sat Apr 26, 2025, 10:27:51 AM"
};

export const invalidGraphInfo: JSGraph = {
    "id": "graph-eb2650cd1f7c522fc8d513b7965b3f51",
    "name": "customer_graph",
    "tables": {
        "stock": {
            "cols": [
                {
                    "name": "StockCode",
                    "stype": "ID",
                    "dtype": "string",
                    "timestamp_format": undefined
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "stock" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "invalidpkey",
            "time_col": "invalidtimecol",
            "end_time_col": "invalidendtimecol"
        },
        "transaction": {
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
                    "dtype": "date",
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
                "connector_id": "test",
                source: { mode: "relative", path: "transaction" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "invalidpkey",
            "time_col": "invalidtimecol",
            "end_time_col": "invalidendtimecol"
        },
        "customer": {
            "cols": [
                {
                    "name": "CustomerID",
                    "stype": "ID",
                    "dtype": "float64",
                    "timestamp_format": undefined
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "customer" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "invalidpkey",
            "time_col": "invalidtimecol",
            "end_time_col": "invalidendtimecol"
        }
    },
    "groups": [
        {
            "columns": [
                {
                    "table_name": "stock",
                    "col_name": "StockCode"
                },
                {
                    "table_name": "transaction",
                    "col_name": "StockCode"
                }
            ]
        },
        {
            "columns": [
                {
                    "table_name": "transaction",
                    "col_name": "CustomerID"
                },
                {
                    "table_name": "customer",
                    "col_name": "CustomerID"
                }
            ]
        }
    ],
    "totalLinks": 2,
    "totalTables": 3,
    "table_name_to_fpid_mapping": {
        "stock": "NA.NA-c8e37d6b963bfcfc5dd691522eca30f6",
        "transaction": "NA.NA-4b1de19e99d03ba91b2f937b2a4292e1",
        "customer": "NA.NA-039bf1683552e2e227a7590644451b7f"
    },
    "created": "2 months ago",
    "dateCreated": "Tue Mar 18, 2025, 12:11:37 PM",
    "updated": "1 month ago",
    "dateUpdated": "Sat Apr 26, 2025, 10:27:51 AM"
};

export const tables: JSTable[] = tablesFromGraphResponse(graphInfo);

export const allTables: GraphTables = {
    "stock": {
        "cols": [
            {
                "name": "StockCode",
                "dtype": "string",
                "stype": "ID",
                "timestamp_format": ""
            }
        ],
        "source_table": {
            "connector_id": "test",
            source: { mode: "relative", path: "stock" },
            "file_type": "PARQUET",
            "data_source_type": DataSourceType.S3
        },
        "pkey": "StockCode",
        "time_col": "",
        "end_time_col": ""
    },
    "transaction": {
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
                "dtype": "date",
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
            "connector_id": "test",
            source: { mode: "relative", path: "transaction" },
            "file_type": "PARQUET",
            "data_source_type": DataSourceType.S3
        },
        "pkey": undefined,
        "time_col": "InvoiceDate",
        "end_time_col": undefined
    },
    "customer": {
        "cols": [
            {
                "name": "CustomerID",
                "stype": "ID",
                "dtype": "float64",
                "timestamp_format": ""
            }
        ],
        "source_table": {
            "connector_id": "test",
            source: { mode: "relative", path: "customer" },
            "file_type": "PARQUET",
            "data_source_type": DataSourceType.S3
        },
        "pkey": "CustomerID",
        "time_col": undefined,
        "end_time_col": undefined
    }
};

export const groups: ColumnGroup[] = [
    {
        "columns": [
            {
                "table_name": "stock",
                "col_name": "StockCode"
            },
            {
                "table_name": "transaction",
                "col_name": "StockCode"
            }
        ],
        "updated": true,
        "state": "saved"
    },
    {
        "columns": [
            {
                "table_name": "transaction",
                "col_name": "CustomerID"
            },
            {
                "table_name": "customer",
                "col_name": "CustomerID"
            }
        ],
        "updated": true,
        "state": "saved"
    }
];

export const mockGfc: GraphFlowInfo = {
    "name": "customer_graph",
    "id": "graph-eb2650cd1f7c522fc8d513b7965b3f51",
    "description": "",
    "tables": {
        "stock": {
            "cols": [
                {
                    "name": "StockCode",
                    "stype": "ID",
                    "dtype": "string",
                    "timestamp_format": undefined
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "stock" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "StockCode",
            "time_col": undefined,
            "end_time_col": undefined
        },
        "transaction": {
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
                    "dtype": "date",
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
                "connector_id": "test",
                source: { mode: "relative", path: "transaction" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": undefined,
            "time_col": "InvoiceDate",
            "end_time_col": undefined
        },
        "customer": {
            "cols": [
                {
                    "name": "CustomerID",
                    "stype": "ID",
                    "dtype": "float64",
                    "timestamp_format": ""
                }
            ],
            "source_table": {
                "connector_id": "test",
                source: { mode: "relative", path: "customer" },
                "file_type": "PARQUET",
                "data_source_type": DataSourceType.S3
            },
            "pkey": "CustomerID",
            "time_col": undefined,
            "end_time_col": undefined
        }
    },
    "selected_tables": ["stock", "transaction", "customer"],
    "selected_table_ids": [],
    "flow_type": FlowType.CREATE,
    "links_state": undefined
};

export const mockTableIds = {
    "transaction_ltv1": "NA.NA-18e2b1a6e499fff45a6f249f3024662f",
    "transactions_hnm_demo": "NA.NA-6cc581a46a0e8e89685e941c52e91dcf",
    "articles_v2": "NA.NA-5201df92c4c549b35aead90f4320b2c5",
    "ratings": "NA.NA-ddb30a643eb247a08212b8bf9960cc49",
    "customer": "NA.NA-3ba4c08a282125063d56cf3c2c65a7ad",
    "stock": "NA.NA-b656de36982dfe8ea68ab7ab1ebac6a1",
    "articles_test_v2_hnm_demo": "NA.NA-69adae35a5b098d0e4a8787636fc3f10",
    "transactions_no_time": "NA.NA-d5828d994e8679119b05ec8f07de920a",
    "customer_1": "NA.NA-9c6faa6d2bca7f7ba8cadb931d7fb2c5",
    "transactions": "NA.NA-d5828d994e8679119b05ec8f07de920z",
    "customers": "NA.NA-d5828d994e8679119b05ec8f07de920y",
    "articles": "NA.NA-d5828d994e8679119b05ec8f07de920x"
};
