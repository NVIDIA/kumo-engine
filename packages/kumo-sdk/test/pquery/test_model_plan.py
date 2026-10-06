# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.model_plan import (
    EarlyStoppingConfig,
    GNNModelPlan,
    IntervalType,
    LRSchedulerConfig,
    MissingType,
    ModelArchitecturePlan,
    ModelPlan,
    NeighborSamplingPlan,
    OptimizationPlan,
    PruningConfig,
    TaskType,
    TrainingJobPlan,
)
from kumoapi.pquery import QueryType

from kumoai_studio.client.pquery import filter_model_plan


def test_filter_model_plan() -> None:
    plan = ModelPlan(
        training_job=TrainingJobPlan(
            num_experiments=1,
            metrics=['auc'],
            tune_metric='auc',
            pruning=PruningConfig(
                min_epochs=2,
                k=3,
                min_delta=0.0,
                patience=1,
            ),
        ),
        neighbor_sampling=NeighborSamplingPlan(
            max_target_neighbors_per_entity=[128],
            num_neighbors=[[16, 16]],
        ),
        optimization=OptimizationPlan(
            max_epochs=10,
            min_steps_per_epoch=30,
            max_steps_per_epoch=2000,
            max_val_steps=1000,
            max_test_steps=2000,
            loss=['binary_cross_entropy'],
            base_lr=[0.01, 0.001],
            weight_decay=[0.0],
            batch_size=[512, 1024],
            lr_scheduler=[
                LRSchedulerConfig(
                    name='constant_with_warmup',
                    interval=IntervalType.STEP,
                    kwargs=dict(warmup_ratio_or_steps=0.1),
                ),
                LRSchedulerConfig(
                    name='cosine_with_warmup_restarts',
                    interval=IntervalType.STEP,
                ),
            ],
            early_stopping=[EarlyStoppingConfig(min_delta=0, patience=3)],
            majority_sampling_ratio=[None],
            weight_mode=MissingType.VALUE,
        ),
        model_architecture=ModelArchitecturePlan(
            model=[
                GNNModelPlan(
                    name='GNN',
                    channels=[
                        64,
                        128,
                        256,
                    ],
                    aggregation=[
                        ['sum', 'mean', 'max'],
                        ['sum', 'mean', 'min', 'max', 'std'],
                    ],
                ),
            ],
            num_pre_message_passing_layers=[0, 1, 2],
            num_post_message_passing_layers=[1, 2],
            activation=['relu', 'leaky_relu'],
            normalization=['layer_norm'],
            use_seq_id=[False],
            prediction_time_encodings=[False],
            past_encoder=MissingType.VALUE,
        ),
    )
    plan = filter_model_plan(
        plan=plan,
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_weight_col=False,
    )

    assert isinstance(plan.model_architecture.model[0], GNNModelPlan)
    assert plan.model_architecture.model[0].channels == [64, 128, 256]

    with pytest.raises(AttributeError):
        getattr(plan.model_architecture, 'module')

    with pytest.raises(AttributeError):
        getattr(plan.model_architecture, 'past_encoder')

    with pytest.raises(AttributeError):
        getattr(plan.model_architecture, 'weight_mode')

    assert repr(plan).startswith(('ModelPlan(\n'
                                  '  training_job=TrainingJobPlan(\n'
                                  '    num_experiments=1,'))

    assert 'disable_explain' not in repr(plan)  # Do not print hidden options.
    assert 'num_neighbors=[\n      [16, 16],' in repr(plan)
    assert 'EarlyStopping(min_delta=0.0, patience=3)' in repr(plan)
