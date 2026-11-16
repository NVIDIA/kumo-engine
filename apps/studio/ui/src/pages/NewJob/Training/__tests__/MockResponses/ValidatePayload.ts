// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const validatePayloadDeatails = {
    "pquery_resource": {
        "graph": {
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
                },
                {
                    "columns": [
                        {
                            "col_name": "article_id",
                            "table_name": "articles"
                        },
                        {
                            "col_name": "article_id",
                            "table_name": "transactions"
                        }
                    ]
                }
            ],
            "tables": {
                "articles": {
                    "cols": [
                        {
                            "dtype": "int64",
                            "name": "article_id",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "product_code",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "prod_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "product_type_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "product_type_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "product_group_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "graphical_appearance_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "graphical_appearance_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "colour_group_code",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "colour_group_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "perceived_colour_value_id",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "perceived_colour_value_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "perceived_colour_master_id",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "perceived_colour_master_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "department_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "department_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "index_code",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "index_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "index_group_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "index_group_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "section_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "section_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "garment_group_no",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "garment_group_name",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "detail_desc",
                            "stype": "categorical",
                            "timestamp_format": null
                        }
                    ],
                    "end_time_col": null,
                    "pkey": "article_id",
                    "source_table": {
                        "connector_id": "hnm",
                        "data_source_type": "S3",
                        "file_type": "PARQUET",
                        source: { mode: "relative", path: "articles" }
                    },
                    "time_col": null
                },
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
        },
        "name": expect.stringMatching(/^test_pq_\d+$/),
        "query_string": "PREDICT customers.age\nFOR EACH customers.customer_id"
    },
    "model_plan": `training_table_generation:
  split: RandomSplit(ratio=[0.8, 0.1, 0.1])
training_job:
  num_experiments: 4
  metrics:
  - mae
  - mse
  - rmse
  tune_metric: mae
  refit_trainval: false
  refit_full: false
column_processing:
  encoder_overrides:
neighbor_sampling:
  num_neighbors:
  - hop1:
      default: 12
    hop2:
      default: 12
  sample_from_entity_table: true
optimization:
  max_epochs: 8
  min_steps_per_epoch: 30
  max_steps_per_epoch: 1000
  max_val_steps: 1000
  max_test_steps: 2000
  loss:
  - huber
  base_lr:
  - 0.0001
  - 0.0005
  - 0.001
  - 0.005
  - 0.01
  - 0.05
  weight_decay:
  - 0.0
  - 5.0e-08
  - 5.0e-07
  - 5.0e-06
  batch_size:
  - 512
  - 1024
  early_stopping:
  - min_delta: 0.0
    patience: 3
  lr_scheduler:
  - name: cosine_with_warmup_restarts
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: constant_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: linear_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
  - name: cosine_with_warmup
    interval: step
    kwargs:
      warmup_ratio_or_steps: 0.1
model_architecture:
  model:
  - name: GNN
    channels:
    - 64
    - 128
    - 256
    aggregation:
    - [sum, mean, max]
    - [sum, mean, min, max, std]
  num_pre_message_passing_layers:
  - 0
  - 1
  - 2
  num_post_message_passing_layers:
  - 1
  - 2
  activation:
  - relu
  - leaky_relu
  - elu
  - gelu
  normalization:
  - layer_norm
  use_seq_id:
  - false`
};
