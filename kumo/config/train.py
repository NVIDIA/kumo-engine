# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict

from kumo_core.types import TaskType

PRPR_ONLY_FIELDS = [
    "disable_explain", "entity_candidate", "entity_candidate_aggregation",
    "refit_full"
]

FIELDS_PER_GROUP = {
    'Training Data Generation': [
        'split',
        'train_start_offset',
        'train_end_offset',
        'timeframe_step',
    ],
    'Training Job': ['num_experiments', 'tune_metric', 'extra_metrics'],
    'Column Processing': ['encoder_overrides'],
    'Neighbor Sampling': ['max_target_neighbors_per_entity', 'num_neighbors'],
    'Optimization': [
        'max_epochs',
        'max_steps_per_epoch',
        'majority_sampling_ratio',
        'base_lr',
        'weight_decay',
        'refit_trainval',
    ],
    'Model Architecture': [
        'channels',
        'num_pre_mp_layers',
        'num_post_mp_layers',
        'aggr',
        'module',
        'handle_new_target_entities',
        'lp_target_embedding_mode',
        'lp_embedding_dim',
        'lp_ranking_emb_loss_coeff',
    ]
}
TASK_SPECIFIC_FIELDS = {
    TaskType.BINARY_CLASSIFICATION: ['majority_sampling_ratio'],
    TaskType.TEMPORAL_LINK_PREDICTION: [
        'module', 'handle_new_target_entities', 'lp_target_embedding_mode',
        'lp_embedding_dim', 'lp_ranking_emb_loss_coeff'
    ],
    TaskType.STATIC_LINK_PREDICTION: [
        'module', 'handle_new_target_entities', 'lp_target_embedding_mode',
        'lp_embedding_dim'
    ],
}

# Necessary but not sufficient validation since these depend on task type as
# well
ALLOWED_METRICS = [
    'acc', 'ap', 'auroc', 'auprc', 'ap_macro', 'auroc_macro', 'auprc_macro',
    'ap_micro', 'auroc_micro', 'auprc_micro', 'ap_per_label', 'auroc_per_label',
    'auprc_per_label', 'ndcg', 'smape', 'wape', 'mae', 'mse', 'rmse', 'map',
    'precision', 'recall', 'f1'
]

ALLOWED_LR_CLASSES = [
    'cosine_with_warmup_restarts', 'constant_with_warmup', 'linear_with_warmup',
    'cosine_with_warmup'
]

ALLOWED_AGGREGATIONS = ['sum', 'mean', 'min', 'max', 'std', 'var']

TYPE_MAP = {
    'int': int,
    'float': float,
    'bool': bool,
    'Any': Any,
    'Enum': Enum,
    'str': str,
    'Dict': Dict,
}
