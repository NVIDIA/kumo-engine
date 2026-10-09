// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const graphDetail = {
    "id": "graph-4973156df602a3f4346186713bd6782d",
    "graph": {
        "tables": {
            "transactions": {
                "cols": [
                    {
                        "name": "t_dat",
                        "stype": "timestamp",
                        "dtype": "date",
                        "timestamp_format": null
                    },
                    {
                        "name": "customer_id",
                        "stype": "ID",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "article_id",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "price",
                        "stype": "numerical",
                        "dtype": "float64",
                        "timestamp_format": null
                    },
                    {
                        "name": "sales_channel_id",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    }
                ],
                "source_table": {
                    "connector_id": "hnm_downsampled",
                    source: { mode: "relative", path: "transactions" },
                    "file_type": "PARQUET" as const,
                    "data_source_type": "S3" as const
                },
                "pkey": null,
                "time_col": "t_dat",
                "end_time_col": null
            },
            "customers": {
                "cols": [
                    {
                        "name": "customer_id",
                        "stype": "ID",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "FN",
                        "stype": "categorical",
                        "dtype": "float64",
                        "timestamp_format": null
                    },
                    {
                        "name": "Active",
                        "stype": "categorical",
                        "dtype": "float64",
                        "timestamp_format": null
                    },
                    {
                        "name": "club_member_status",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "fashion_news_frequency",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "age",
                        "stype": "numerical",
                        "dtype": "float64",
                        "timestamp_format": null
                    },
                    {
                        "name": "postal_code",
                        "stype": "ID",
                        "dtype": "string",
                        "timestamp_format": null
                    }
                ],
                "source_table": {
                    "connector_id": "hnm",
                    source: { mode: "relative", path: "customers" },
                    "file_type": "PARQUET" as const,
                    "data_source_type": "S3" as const
                },
                "pkey": "customer_id",
                "time_col": null,
                "end_time_col": null
            },
            "articles": {
                "cols": [
                    {
                        "name": "article_id",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "product_code",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "prod_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "product_type_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "product_type_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "product_group_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "graphical_appearance_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "graphical_appearance_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "colour_group_code",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "colour_group_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "perceived_colour_value_id",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "perceived_colour_value_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "perceived_colour_master_id",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "perceived_colour_master_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "department_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "department_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "index_code",
                        "stype": "ID",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "index_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "index_group_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "index_group_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "section_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "section_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "garment_group_no",
                        "stype": "ID",
                        "dtype": "int64",
                        "timestamp_format": null
                    },
                    {
                        "name": "garment_group_name",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    },
                    {
                        "name": "detail_desc",
                        "stype": "categorical",
                        "dtype": "string",
                        "timestamp_format": null
                    }
                ],
                "source_table": {
                    "connector_id": "hnm",
                    source: { mode: "relative", path: "articles" },
                    "file_type": "PARQUET" as const,
                    "data_source_type": "S3" as const
                },
                "pkey": "article_id",
                "time_col": null,
                "end_time_col": null
            }
        },
        "col_groups": [
            {
                "columns": [
                    {
                        "table_name": "transactions",
                        "col_name": "customer_id"
                    },
                    {
                        "table_name": "customers",
                        "col_name": "customer_id"
                    }
                ]
            },
            {
                "columns": [
                    {
                        "table_name": "articles",
                        "col_name": "article_id"
                    },
                    {
                        "table_name": "transactions",
                        "col_name": "article_id"
                    }
                ]
            }
        ]
    },
    "name_alias": "ltv_graph",
    "timestamps": {
        "created_at": "2025-02-18T04:22:18.698114+00:00",
        "updated_at": "2025-03-03T05:51:12.778660+00:00"
    },
    "table_name_to_fpid_mapping": {
        "transactions": "b09f45922a7a21d657310bfc00c5ed7c",
        "customers": "767722274fd4fa073bdb3fec5040895e",
        "articles": "8ae4c70469b933c48d884c34d95a3cc5"
    },
    "num_tables": 3,
    "num_links": 2
};
