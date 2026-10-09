# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from io_utils import kumo_dataclass


@kumo_dataclass
class DeprecatedAdvancedAutoTrainerOptions:
    """DEPRECATED: DO NOT USE"""

    run_mode: Optional[str] = None
    max_steps_per_epoch: Optional[int] = None
    max_val_steps: Optional[int] = None
    max_test_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    num_experiments: Optional[int] = None
    refit_trainval: bool = False
    refit_full: bool = False
    glove: bool = True
    disable_explain: bool = False
    tune_metric: Optional[str] = None
    extra_metrics: List[str] = field(default_factory=list)
    num_ensembles: int = 1
    min_num_hops: int = 2
    num_neighbors: Optional[List[int]] = None
    max_target_neighbors_per_entity: Optional[int] = None
    use_seq_id: bool = False
    timeframe_step: Optional[int] = None
    forecast_length: Optional[int] = None
    lag_timesteps: Optional[int] = None
    train_start_offset: Optional[int] = None
    train_end_offset: Optional[int] = None
    majority_sampling_ratio: Optional[float] = None
    weight_mode: str = ""
    entity_candidate: Optional[Union[bool, str]] = None
    entity_candidate_aggregation: Optional[str] = None
    channels: List[int] = field(default_factory=list)
    num_pre_mp_layers: List[int] = field(default_factory=list)
    num_post_mp_layers: List[int] = field(default_factory=list)
    activation: List[str] = field(default_factory=list)
    base_lr: List[float] = field(default_factory=list)
    batch_size: List[int] = field(default_factory=list)
    weight_decay: List[float] = field(default_factory=list)
    aggr: List[List[str]] = field(default_factory=list)
    lr_scheduler: List[List[Any]] = field(default_factory=list)
    early_stopping: List[List[Any]] = field(default_factory=list)
    seed: Optional[int] = None
    task_path_override: Optional[str] = None
    module: str = ""
    probabilistic_modeling: Optional[str] = None
    handle_new_target_entities: bool = False
    lp_target_embedding_mode: str = ""
    lp_embedding_dim: int = 32
    lp_ranking_emb_loss_coeff: float = 0.0
    split: Optional[str] = None
    num_target_to_entity_nodes: Optional[int] = None
    use_lp_retrieval_metrics: Optional[bool] = None
    id_awareness: Optional[bool] = None
    use_retrieval_metrics: Optional[bool] = None
    infer_batch_size: Optional[bool] = None
    train_table_path_override: Optional[str] = None
    encoder_overrides: Optional[Dict[str, str]] = field(default_factory=dict)
