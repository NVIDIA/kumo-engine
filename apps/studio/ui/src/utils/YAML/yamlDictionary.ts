// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HighlightTypes, HoverHintsType } from "types/codeEditor";

export const yamlDictionary: Record<string, { type: HighlightTypes }> = {
    "encoder_overrides": { type: HighlightTypes.KEYWORD },
    "activation": { type: HighlightTypes.KEYWORD },
    "aggregation": { type: HighlightTypes.KEYWORD },
    "channels": { type: HighlightTypes.KEYWORD },
    "handle_new_target_entities": { type: HighlightTypes.KEYWORD },
    "module": { type: HighlightTypes.KEYWORD },
    "normalization": { type: HighlightTypes.KEYWORD },
    "num_post_message_passing_layers": { type: HighlightTypes.KEYWORD },
    "num_pre_message_passing_layers": { type: HighlightTypes.KEYWORD },
    "ranking_embedding_loss_coeff": { type: HighlightTypes.KEYWORD },
    "output_embedding_dim": { type: HighlightTypes.KEYWORD },
    "target_embedding_mode": { type: HighlightTypes.KEYWORD },
    "use_seq_id": { type: HighlightTypes.KEYWORD },
    "max_target_neighbors_per_entity": { type: HighlightTypes.KEYWORD },
    "num_neighbors": { type: HighlightTypes.KEYWORD },
    "sample_from_entity_table": { type: HighlightTypes.KEYWORD },
    "base_lr": { type: HighlightTypes.KEYWORD },
    "batch_size": { type: HighlightTypes.KEYWORD },
    "early_stopping": { type: HighlightTypes.KEYWORD },
    "lr_scheduler": { type: HighlightTypes.KEYWORD },
    "name": { type: HighlightTypes.STRING },
    "interval": { type: HighlightTypes.STRING },
    "kwargs": { type: HighlightTypes.STRING },
    "majority_sampling_ratio": { type: HighlightTypes.KEYWORD },
    "max_epochs": { type: HighlightTypes.KEYWORD },
    "max_steps_per_epoch": { type: HighlightTypes.KEYWORD },
    "min_steps_per_epoch": { type: HighlightTypes.KEYWORD },
    "max_test_steps": { type: HighlightTypes.KEYWORD },
    "min_test_steps": { type: HighlightTypes.KEYWORD },
    "max_val_steps": { type: HighlightTypes.KEYWORD },
    "min_val_steps": { type: HighlightTypes.KEYWORD },
    "weight_decay": { type: HighlightTypes.KEYWORD },
    "weight_mode": { type: HighlightTypes.KEYWORD },
    "refit_full": { type: HighlightTypes.KEYWORD },
    "split": { type: HighlightTypes.KEYWORD },
    "entity_candidate_aggregation": { type: HighlightTypes.KEYWORD },
    "refit_trainval": { type: HighlightTypes.KEYWORD },
    "metrics": { type: HighlightTypes.KEYWORD },
    "tune_metric": { type: HighlightTypes.KEYWORD },
    "auroc_per_label": { type: HighlightTypes.VARIABLE_NAME },
    "auroc_macro": { type: HighlightTypes.VARIABLE_NAME },
    "auroc_micro": { type: HighlightTypes.VARIABLE_NAME },
    "auroc": { type: HighlightTypes.VARIABLE_NAME },
    "acc": { type: HighlightTypes.VARIABLE_NAME },
    "auprc_macro": { type: HighlightTypes.VARIABLE_NAME },
    "auprc_micro": { type: HighlightTypes.VARIABLE_NAME },
    "auprc": { type: HighlightTypes.VARIABLE_NAME },
    "ap_macro": { type: HighlightTypes.VARIABLE_NAME },
    "ap_micro": { type: HighlightTypes.VARIABLE_NAME },
    "ap_per_label": { type: HighlightTypes.VARIABLE_NAME },
    "ap": { type: HighlightTypes.VARIABLE_NAME },
    "recall@100": { type: HighlightTypes.VARIABLE_NAME },
    "recall@10": { type: HighlightTypes.VARIABLE_NAME },
    "recall@1": { type: HighlightTypes.VARIABLE_NAME },
    "recall@25": { type: HighlightTypes.VARIABLE_NAME },
    "recall": { type: HighlightTypes.VARIABLE_NAME },
    "precision@25": { type: HighlightTypes.VARIABLE_NAME },
    "precision@100": { type: HighlightTypes.VARIABLE_NAME },
    "precision@10": { type: HighlightTypes.VARIABLE_NAME },
    "precision@1": { type: HighlightTypes.VARIABLE_NAME },
    "precision": { type: HighlightTypes.VARIABLE_NAME },
    "ndcg": { type: HighlightTypes.VARIABLE_NAME },
    "loss": { type: HighlightTypes.KEYWORD },
    "mrr": { type: HighlightTypes.VARIABLE_NAME },
    "mae": { type: HighlightTypes.VARIABLE_NAME },
    "mape": { type: HighlightTypes.VARIABLE_NAME },
    "smape": { type: HighlightTypes.VARIABLE_NAME },
    "mse": { type: HighlightTypes.VARIABLE_NAME },
    "rmse": { type: HighlightTypes.VARIABLE_NAME },
    "map@100": { type: HighlightTypes.VARIABLE_NAME },
    "map@10": { type: HighlightTypes.VARIABLE_NAME },
    "map@25": { type: HighlightTypes.VARIABLE_NAME },
    "map@1": { type: HighlightTypes.VARIABLE_NAME },
    "map": { type: HighlightTypes.VARIABLE_NAME },
    "f1@100": { type: HighlightTypes.VARIABLE_NAME },
    "f1@25": { type: HighlightTypes.VARIABLE_NAME },
    "f1@10": { type: HighlightTypes.VARIABLE_NAME },
    "f1@1": { type: HighlightTypes.VARIABLE_NAME },
    "f1": { type: HighlightTypes.VARIABLE_NAME },
    "num_experiments": { type: HighlightTypes.KEYWORD },
    "forecast_length": { type: HighlightTypes.KEYWORD },
    "lag_timesteps": { type: HighlightTypes.KEYWORD },
    "year_over_year": { type: HighlightTypes.KEYWORD },
    "timeframe_step": { type: HighlightTypes.KEYWORD },
    "train_end_offset": { type: HighlightTypes.KEYWORD },
    "train_start_offset": { type: HighlightTypes.KEYWORD },
    "start_time": { type: HighlightTypes.KEYWORD },
    "end_time": { type: HighlightTypes.KEYWORD },
    "prediction_time_encodings": { type: HighlightTypes.KEYWORD },
    "handle_new_entities": { type: HighlightTypes.KEYWORD },
    "training_table_generation": { type: HighlightTypes.TYPE_NAME },
    "training_job": { type: HighlightTypes.TYPE_NAME },
    "column_processing": { type: HighlightTypes.TYPE_NAME },
    "neighbor_sampling": { type: HighlightTypes.TYPE_NAME },
    "optimization": { type: HighlightTypes.TYPE_NAME },
    "model_architecture": { type: HighlightTypes.TYPE_NAME },
    "0": { type: HighlightTypes.NUMBER },
    "1": { type: HighlightTypes.NUMBER },
    "2": { type: HighlightTypes.NUMBER },
    "3": { type: HighlightTypes.NUMBER },
    "4": { type: HighlightTypes.NUMBER },
    "5": { type: HighlightTypes.NUMBER },
    "6": { type: HighlightTypes.NUMBER },
    "7": { type: HighlightTypes.NUMBER },
    "8": { type: HighlightTypes.NUMBER },
    "9": { type: HighlightTypes.NUMBER },
    "e-": { type: HighlightTypes.NUMBER },
    "-": { type: HighlightTypes.NUMBER },
    ".": { type: HighlightTypes.NUMBER }
};

export const yamlDict: HoverHintsType = {
    "min_steps_per_epoch": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "min_test_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "min_val_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "auroc_per_label": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_macro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_micro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "acc": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_macro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_micro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_macro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_micro": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_per_label": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@K": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@K": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ndcg": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "loss": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "mrr": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mae": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mape": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "smape": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mse": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "rmse": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@K": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@K": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "prediction_time_encodings": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "training_table_generation": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "training_job": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "column_processing": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "neighbor_sampling": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "optimization": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "model_architecture": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "0": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "1": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "2": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "3": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "4": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "5": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "6": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "7": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "8": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "9": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "e-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    ".": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    }
};
