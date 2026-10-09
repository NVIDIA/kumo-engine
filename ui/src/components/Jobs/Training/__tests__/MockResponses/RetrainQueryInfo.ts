// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const retrainQueryInfo = {
    "copytrainingargs": {
        "description": "",
        "enable_baselines": false,
        "graph": {
            "groups": [
                {
                    "columns": [
                        {
                            "col_name": "ACCOUNT_ID",
                            "table_name": "FINANCE_TRANS"
                        },
                        {
                            "col_name": "ACCOUNT_ID",
                            "table_name": "FINANCE_ORDERS"
                        },
                        {
                            "col_name": "ACCOUNT_ID",
                            "table_name": "FINANCE_ACCOUNT"
                        }
                    ]
                }
            ],
            "id": "graph-d445eeb8b2aa1b79a142406239fa74e8",
            "isActive": true,
            "name": "",
            "tables": {
                "FINANCE_ACCOUNT": {
                    "cols": [
                        {
                            "dtype": "int32",
                            "name": "ACCOUNT_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "DISTRICT_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "FREQUENCY",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "date",
                            "name": "DATE",
                            "stype": "timestamp",
                            "timestamp_format": null
                        }
                    ],
                    "end_time_col": null,
                    "pkey": "ACCOUNT_ID",
                    "source_table": {
                        "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                        "data_source_type": "S3",
                        "file_type": "PARQUET",
                        source: { mode: "relative", path: "ACCOUNT" }
                    },
                    "time_col": "DATE"
                },
                "FINANCE_ORDERS": {
                    "cols": [
                        {
                            "dtype": "int32",
                            "name": "ORDERS_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "ACCOUNT_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "BANK_TO",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int64",
                            "name": "ACCOUNT_TO",
                            "stype": "numerical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "float64",
                            "name": "AMOUNT",
                            "stype": "numerical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "K_SYMBOL",
                            "stype": "categorical",
                            "timestamp_format": null
                        }
                    ],
                    "end_time_col": null,
                    "pkey": "ORDERS_ID",
                    "source_table": {
                        "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                        "data_source_type": "S3",
                        "file_type": "PARQUET",
                        source: { mode: "relative", path: "ORDERS" }
                    },
                    "time_col": null
                },
                "FINANCE_TRANS": {
                    "cols": [
                        {
                            "dtype": "int32",
                            "name": "TRANS_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int32",
                            "name": "ACCOUNT_ID",
                            "stype": "ID",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "date",
                            "name": "DATE",
                            "stype": "timestamp",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "TYPE",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "OPERATION",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int32",
                            "name": "AMOUNT",
                            "stype": "numerical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int32",
                            "name": "BALANCE",
                            "stype": "numerical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "K_SYMBOL",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "string",
                            "name": "BANK",
                            "stype": "categorical",
                            "timestamp_format": null
                        },
                        {
                            "dtype": "int32",
                            "name": "ACCOUNT",
                            "stype": "numerical",
                            "timestamp_format": null
                        }
                    ],
                    "end_time_col": null,
                    "pkey": "TRANS_ID",
                    "source_table": {
                        "connector_id": "FINANCIAL_DOWNSAMPLED_DATASET_V2",
                        "data_source_type": "S3",
                        "file_type": "PARQUET",
                        source: { mode: "relative", path: "TRANS" }
                    },
                    "time_col": "DATE"
                }
            },
            "totalLinks": 0,
            "totalTables": 3
        },
        "graph_id": "graph-d445eeb8b2aa1b79a142406239fa74e8",
        "graph_name": "",
        "model_planner_yaml": `training_table_generation:
  split: ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])
  train_start_offset: inferred
  train_end_offset: inferred
  timeframe_step: inferred
  forecast_length: '???'
  lag_timesteps: '???'
  year_over_year: '???'
  entity_candidate: null
  entity_candidate_aggregation: null
  task_path_override: null
  train_table_path_override: null
training_job:
  num_experiments: 2
  metrics:
    - auroc
    - ap
    - auprc
  tune_metric: auroc
  disable_compilation: true
  refit_trainval: false
  refit_full: false
  disable_explain: false
  manual_seed: null
  enable_baselines: false
column_processing:
  encoder_overrides: null
neighbor_sampling:
  max_target_neighbors_per_entity: '???'
  num_neighbors:
    - hop1:
        default: 12
        FINANCE_ACCOUNT.ACCOUNT_ID->FINANCE_TRANS.ACCOUNT_ID: inferred
      hop2:
        default: 12
      hop3: null
      hop4: null
      hop5: null
      hop6: null
  sample_from_entity_table: '???'
optimization:
  max_epochs: 8
  min_steps_per_epoch: 30
  max_steps_per_epoch: 1000
  max_val_steps: 1000
  max_test_steps: 2000
  loss:
    - binary_cross_entropy
  base_lr:
    - 0.0001
    - 0.0005
    - 0.001
    - 0.005
    - 0.01
    - 0.05
  weight_decay:
    - 0
    - 5.e-8
    - 5.e-7
    - 0.000005
  batch_size:
    - 512
    - 1024
  early_stopping:
    - min_delta: 0
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
  majority_sampling_ratio:
    - null
  weight_mode:
    - sample
model_architecture:
  channels:
    - 64
    - 128
    - 256
  num_pre_message_passing_layers:
    - 0
    - 1
    - 2
  num_post_message_passing_layers:
    - 1
    - 2
  aggregation:
    - - sum
      - mean
      - max
    - - sum
      - mean
      - min
      - max
      - std
  activation:
    - relu
    - leaky_relu
    - elu
    - gelu
  normalization:
    - layer_norm
  module: '???'
  handle_new_target_entities: '???'
  target_embedding_mode: '???'
  output_embedding_dim: '???'
  ranking_embedding_loss_coeff: '???'
  distance_measure: '???'
  use_seq_id:
    - false
  prediction_time_encodings:
    - false
  past_encoder: '???'
  handle_new_entities: '???'
  forecast_type: '???'
`,
        "name": "trainingjob-cd40820089504c91825ec0b413e97cbb_copy",
        "query_dialect":
            "PREDICT COUNT(FINANCE_TRANS.*, 0, 30,days)=0\nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
        "query_yaml":
            "PREDICT COUNT(FINANCE_TRANS.*, 0, 30,days)=0\nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
        "run_mode": "fast"
    },
    "isSDKJob": false,
    "taskType": "binary_classification"
};
