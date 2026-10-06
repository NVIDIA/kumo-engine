// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const ModelPlan = `training_table_generation:
  split: >-
    TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.8,
    0.1, 0.1], indices=None, strict_split_boundaries=True)
  train_start_offset: '???'
  train_end_offset: '???'
  start_time: inferred
  end_time: inferred
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
  pruning:
    min_epochs: 2
    k: 3
    min_delta: 0
    patience: 1
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
        customers_hnm_downsampled.customer_id->transactions_2020_hnm_downsampled.customer_id: inferred
      hop2:
        default: 12
      hop3: null
      hop4: null
      hop5: null
      hop6: null
  sample_from_entity_table: '???'
  adaptive_sampling: false
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
    - null
  target_fraction: '???'
  overlap_ratio: '???'
model_architecture:
  model:
    - name: GNN
      channels:
        - 64
        - 128
        - 256
      aggregation:
        - - sum
          - mean
          - max
        - - sum
          - mean
          - min
          - max
          - std
  channels: '???'
  num_pre_message_passing_layers:
    - 0
    - 1
    - 2
  num_post_message_passing_layers:
    - 1
    - 2
  aggregation: '???'
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
`;
