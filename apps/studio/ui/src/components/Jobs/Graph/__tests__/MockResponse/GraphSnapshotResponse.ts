// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const failed_res = {
    "snapshot_resource": {
        "id": "graph-0005b000000000000000000000000000@00033000000000000000000000000000",
        "stages": {
            "INGEST": {
                "start_time": "2025-05-13T05:54:17.523669+00:00",
                "end_time": "2025-05-13T05:54:16.623171+00:00",
                "status": "FAILED",
                "error":
                    'table-1e113db640484f22c29f9100ccc34844@1680056762 has error: Errors:\nIngested table empty. Ingested table transactions has no rows. 31788324 rows were dropped during ingestion due to one of the following issues. For CSV connector, malformed rows are dropped. For all connectors, if a "Create date" column is specified, any rows with invalid timestamps in that column will be dropped. Please check the connector, source data, and "Create date" column selection.',
                "warnings": [
                    "Warning: Some data may be incomplete",
                    "Warning: Performance might be affected"
                ]
            }
        },
        "table_ids": {
            "articles_HM": "table-bc150313117da0324bfe31ca4ff499e5@1680056672",
            "customers_3": "table-ca638577dc299a59a9257747df49b5a5@1680056672",
            "transactions_2":
                "table-1e113db640484f22c29f9100ccc34844@1680056762"
        }
    },
    "definition": {
        "col_groups": [
            {
                "columns": [
                    {
                        "col_name": "customer_id",
                        "table_name": "customers_3"
                    },
                    {
                        "col_name": "customer_id",
                        "table_name": "transactions_2"
                    }
                ]
            },
            {
                "columns": [
                    {
                        "col_name": "article_id",
                        "table_name": "transactions_2"
                    },
                    {
                        "col_name": "article_id",
                        "table_name": "articles_HM"
                    }
                ]
            }
        ],
        "tables": {
            "articles_HM": {
                "pkey": "article_id",
                "cols": [
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "product_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    { "name": "prod_name", "stype": "text", "dtype": "string" },
                    {
                        "name": "product_type_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "product_type_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "product_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "graphical_appearance_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "graphical_appearance_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_value_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_value_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_master_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_master_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "department_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "department_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "index_code", "stype": "ID", "dtype": "string" },
                    {
                        "name": "index_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "index_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "index_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "section_no", "stype": "ID", "dtype": "int64" },
                    {
                        "name": "section_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "garment_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "garment_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "detail_desc",
                        "stype": "text",
                        "dtype": "string"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "articles" }
                }
            },
            "customers_3": {
                "pkey": "customer_id",
                "cols": [
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "FN", "stype": "categorical", "dtype": "int64" },
                    {
                        "name": "Active",
                        "stype": "categorical",
                        "dtype": "int64"
                    },
                    {
                        "name": "club_member_status",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "fashion_news_frequency",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "age", "stype": "numerical", "dtype": "float64" },
                    { "name": "postal_code", "stype": "ID", "dtype": "string" }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "customers" }
                }
            },
            "transactions_2": {
                "pkey": null,
                "time_col": "sales_channel_id",
                "cols": [
                    {
                        "name": "t_dat",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "price",
                        "stype": "numerical",
                        "dtype": "float64"
                    },
                    {
                        "name": "sales_channel_id",
                        "stype": "timestamp",
                        "dtype": "date"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "transactions" }
                }
            }
        }
    }
};

export const queued_res = {
    "snapshot_resource": {
        "id": "graph-0005b000000000000000000000000000@00033000000000000000000000000000",
        "stages": {
            "INGEST": {
                "start_time": "2025-05-13T05:54:17.523669+00:00",
                "end_time": "2025-05-13T05:54:16.623171+00:00",
                "status": "QUEUED",
                "warnings": []
            }
        },
        "table_ids": {
            "articles_HM": "table-bc150313117da0324bfe31ca4ff499e5@1680056672",
            "customers_3": "table-ca638577dc299a59a9257747df49b5a5@1680056672",
            "transactions_2":
                "table-1e113db640484f22c29f9100ccc34844@1680056762"
        }
    },
    "definition": {
        "col_groups": [
            {
                "columns": [
                    {
                        "col_name": "customer_id",
                        "table_name": "customers_3"
                    },
                    {
                        "col_name": "customer_id",
                        "table_name": "transactions_2"
                    }
                ]
            },
            {
                "columns": [
                    {
                        "col_name": "article_id",
                        "table_name": "transactions_2"
                    },
                    {
                        "col_name": "article_id",
                        "table_name": "articles_HM"
                    }
                ]
            }
        ],
        "tables": {
            "articles_HM": {
                "pkey": "article_id",
                "cols": [
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "product_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    { "name": "prod_name", "stype": "text", "dtype": "string" },
                    {
                        "name": "product_type_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "product_type_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "product_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "graphical_appearance_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "graphical_appearance_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_value_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_value_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_master_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_master_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "department_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "department_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "index_code", "stype": "ID", "dtype": "string" },
                    {
                        "name": "index_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "index_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "index_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "section_no", "stype": "ID", "dtype": "int64" },
                    {
                        "name": "section_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "garment_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "garment_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "detail_desc",
                        "stype": "text",
                        "dtype": "string"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "articles" }
                }
            },
            "customers_3": {
                "pkey": "customer_id",
                "cols": [
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "FN", "stype": "categorical", "dtype": "int64" },
                    {
                        "name": "Active",
                        "stype": "categorical",
                        "dtype": "int64"
                    },
                    {
                        "name": "club_member_status",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "fashion_news_frequency",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "age", "stype": "numerical", "dtype": "float64" },
                    { "name": "postal_code", "stype": "ID", "dtype": "string" }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "customers" }
                }
            },
            "transactions_2": {
                "pkey": null,
                "time_col": "sales_channel_id",
                "cols": [
                    {
                        "name": "t_dat",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "price",
                        "stype": "numerical",
                        "dtype": "float64"
                    },
                    {
                        "name": "sales_channel_id",
                        "stype": "timestamp",
                        "dtype": "date"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "transactions" }
                }
            }
        }
    }
};

export const cancelled_res = {
    "snapshot_resource": {
        "id": "graph-0005b000000000000000000000000000@00033000000000000000000000000000",
        "stages": {
            "INGEST": {
                "start_time": "2025-05-13T05:54:17.523669+00:00",
                "end_time": "2025-05-13T05:54:16.623171+00:00",
                "status": "CANCELLED",
                "warnings": []
            }
        },
        "table_ids": {
            "articles_HM": "table-bc150313117da0324bfe31ca4ff499e5@1680056672",
            "customers_3": "table-ca638577dc299a59a9257747df49b5a5@1680056672",
            "transactions_2":
                "table-1e113db640484f22c29f9100ccc34844@1680056762"
        }
    },
    "definition": {
        "col_groups": [
            {
                "columns": [
                    {
                        "col_name": "customer_id",
                        "table_name": "customers_3"
                    },
                    {
                        "col_name": "customer_id",
                        "table_name": "transactions_2"
                    }
                ]
            },
            {
                "columns": [
                    {
                        "col_name": "article_id",
                        "table_name": "transactions_2"
                    },
                    {
                        "col_name": "article_id",
                        "table_name": "articles_HM"
                    }
                ]
            }
        ],
        "tables": {
            "articles_HM": {
                "pkey": "article_id",
                "cols": [
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "product_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    { "name": "prod_name", "stype": "text", "dtype": "string" },
                    {
                        "name": "product_type_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "product_type_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "product_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "graphical_appearance_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "graphical_appearance_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_value_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_value_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_master_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_master_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "department_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "department_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "index_code", "stype": "ID", "dtype": "string" },
                    {
                        "name": "index_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "index_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "index_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "section_no", "stype": "ID", "dtype": "int64" },
                    {
                        "name": "section_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "garment_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "garment_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "detail_desc",
                        "stype": "text",
                        "dtype": "string"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "articles" }
                }
            },
            "customers_3": {
                "pkey": "customer_id",
                "cols": [
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "FN", "stype": "categorical", "dtype": "int64" },
                    {
                        "name": "Active",
                        "stype": "categorical",
                        "dtype": "int64"
                    },
                    {
                        "name": "club_member_status",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "fashion_news_frequency",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "age", "stype": "numerical", "dtype": "float64" },
                    { "name": "postal_code", "stype": "ID", "dtype": "string" }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "customers" }
                }
            },
            "transactions_2": {
                "pkey": null,
                "time_col": "sales_channel_id",
                "cols": [
                    {
                        "name": "t_dat",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "price",
                        "stype": "numerical",
                        "dtype": "float64"
                    },
                    {
                        "name": "sales_channel_id",
                        "stype": "timestamp",
                        "dtype": "date"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "transactions" }
                }
            }
        }
    }
};

export const running_res = {
    "snapshot_resource": {
        "id": "graph-0005b000000000000000000000000000@00033000000000000000000000000000",
        "stages": {
            "INGEST": {
                "start_time": "2025-05-13T05:54:17.523669+00:00",
                "end_time": "2025-05-13T05:54:16.623171+00:00",
                "status": "RUNNING",
                "warnings": []
            }
        },
        "table_ids": {
            "articles_HM": "table-bc150313117da0324bfe31ca4ff499e5@1680056672",
            "customers_3": "table-ca638577dc299a59a9257747df49b5a5@1680056672",
            "transactions_2":
                "table-1e113db640484f22c29f9100ccc34844@1680056762"
        }
    },
    "definition": {
        "col_groups": [
            {
                "columns": [
                    {
                        "col_name": "customer_id",
                        "table_name": "customers_3"
                    },
                    {
                        "col_name": "customer_id",
                        "table_name": "transactions_2"
                    }
                ]
            },
            {
                "columns": [
                    {
                        "col_name": "article_id",
                        "table_name": "transactions_2"
                    },
                    {
                        "col_name": "article_id",
                        "table_name": "articles_HM"
                    }
                ]
            }
        ],
        "tables": {
            "articles_HM": {
                "pkey": "article_id",
                "cols": [
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "product_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    { "name": "prod_name", "stype": "text", "dtype": "string" },
                    {
                        "name": "product_type_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "product_type_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "product_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "graphical_appearance_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "graphical_appearance_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_code",
                        "stype": "ID",
                        "dtype": "string"
                    },
                    {
                        "name": "colour_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_value_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_value_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "perceived_colour_master_id",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "perceived_colour_master_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "department_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "department_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "index_code", "stype": "ID", "dtype": "string" },
                    {
                        "name": "index_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "index_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "index_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "section_no", "stype": "ID", "dtype": "int64" },
                    {
                        "name": "section_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "garment_group_no",
                        "stype": "ID",
                        "dtype": "int64"
                    },
                    {
                        "name": "garment_group_name",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "detail_desc",
                        "stype": "text",
                        "dtype": "string"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "articles" }
                }
            },
            "customers_3": {
                "pkey": "customer_id",
                "cols": [
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "FN", "stype": "categorical", "dtype": "int64" },
                    {
                        "name": "Active",
                        "stype": "categorical",
                        "dtype": "int64"
                    },
                    {
                        "name": "club_member_status",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    {
                        "name": "fashion_news_frequency",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "age", "stype": "numerical", "dtype": "float64" },
                    { "name": "postal_code", "stype": "ID", "dtype": "string" }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "customers" }
                }
            },
            "transactions_2": {
                "pkey": null,
                "time_col": "sales_channel_id",
                "cols": [
                    {
                        "name": "t_dat",
                        "stype": "categorical",
                        "dtype": "string"
                    },
                    { "name": "customer_id", "stype": "ID", "dtype": "string" },
                    { "name": "article_id", "stype": "ID", "dtype": "string" },
                    {
                        "name": "price",
                        "stype": "numerical",
                        "dtype": "float64"
                    },
                    {
                        "name": "sales_channel_id",
                        "stype": "timestamp",
                        "dtype": "date"
                    }
                ],
                "source_table": {
                    "connector_id": "HM",
                    "data_source_type": "S3",
                    "file_type": "CSV",
                    source: { mode: "relative", path: "transactions" }
                }
            }
        }
    }
};
