# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Set

from kumo_core.model_plan import (
    ModelPlan,
    PredictiveQueryAdvancedOptions,
    TrainingTableGenerationPlan,
    inferred_or_missing_type_to_false,
    inferred_or_missing_type_to_none,
)
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType

from kumo.config.model_plan_adapter import to_predictive_query_kwargs
from kumo.config.model_plan_yaml import model_plan_to_yaml
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.pquery.predictive_query import PredictiveQueryV2
from rest.utils.pquery import (
    make_predictive_config_with_graph,
    resolve_train_table_options,
)

VALID_QUERY_YAML = ('PREDICT LOAN_FIXTURE.STATUS\n'
                    'FOR EACH LOAN_FIXTURE.LOAN_ID')

VALID_BINARY_QUERY = ('PREDICT LOAN_FIXTURE.STATUS = "A"\n'
                      'FOR EACH LOAN_FIXTURE.LOAN_ID')

VALID_OPTIONS_YAML = 'run_mode: FAST\nglove: True\nnum_ensembles: 1'
DATE_OFFSET_SPLIT_YAML = ('\nsplit: DateOffsetSplit([-60, -30], unit="days")')
TEMPORAL_SPLIT_YAML = '\nsplit: TimeSplit(fractions=[0.7, 0.2, 0.1])'
RANDOM_SPLIT_YAML = '\nsplit: RandomSplit([0.7, 0.2, 0.1])'
TIME_RANGE_SPLIT_YAML = ('\nsplit: TimeRangeSplit([("2022-04-01", '
                         '"2022-06-01"), ("2022-06-01", "2022-07-01"), '
                         '("2022-07-01", "2022-12-02")])')
LR_SCHEDULER_YAML = (
    "\nlr_scheduler: [['cosine_with_warmup_restarts', {'gamma': 0.9}, 'step']]")

VALID_MODEL_PLAN = """# Training Data Generation
split: TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.8, 0.1, 0.1], indices=None, strict_split_boundaries=True)
start_time: null
end_time: null
timeframe_step: null

# Training Job
num_experiments: 2
tune_metric: auroc
extra_metrics: []

# Column Processing
encoder_overrides:

# Neighbor Sampling
max_target_neighbors_per_entity: null
num_neighbors: [8, 8]

# Optimization
max_epochs: 2
max_steps_per_epoch: 25
majority_sampling_ratio: null
base_lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
weight_decay: [0.0, 5e-08, 5e-07, 5e-06]
refit_trainval: False

# Model Architecture
channels: [64, 128, 256]
num_pre_mp_layers: [0, 1, 2]
num_post_mp_layers: [1, 2]
aggr: [['sum', 'mean', 'max'], ['sum', 'mean', 'min', 'max', 'std']]
"""  # noqa: E501

VALID_MODEL_PLAN_FORMATTED = """split: TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.8, 0.1, 0.1], indices=None, strict_split_boundaries=True)
start_time: null
end_time: null
timeframe_step: null
num_experiments: 2
tune_metric: auroc
extra_metrics: []
encoder_overrides: null
max_target_neighbors_per_entity: null
num_neighbors: [8, 8]
max_epochs: 2
max_steps_per_epoch: 25
majority_sampling_ratio: null
base_lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
weight_decay: [0.0, 5e-08, 5e-07, 5e-06]
refit_trainval: False
channels: [64, 128, 256]
num_pre_mp_layers: [0, 1, 2]
num_post_mp_layers: [1, 2]
aggr: [['sum', 'mean', 'max'], ['sum', 'mean', 'min', 'max', 'std']]


"""  # noqa: E501

INVALID_MODEL_PLANNER_YAML = """
training_table_generation:
  split: TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.54805, 0.1, 0.35195], indices=None, strict_split_boundaries=True)
training_job:
  num_experiments: 8
  metrics:
  - auroc
  - ap
  - auprc
  tune_metric: auroc
  refit_trainval: false
  refit_full: false
column_processing:
  encoder_overrides:
    upc_1234.promo_upc_flag: Null()
    upc_1234.similar_cat_flag: Null()
    orders_casted_1234.cat_nbr: Null()
    orders_casted_1234.brand_descr: Null()
    orders_casted_1234.vendor_nm: Null()
    orders_casted_1234.key_cat_nbr: Null()
    orders_casted_1234.prod_typ_descr: Null()
    orders_casted_1234.prod_subtype_descr: Null()
    orders_casted_1234.order_network_id
    target_static_1234.target: Null()
neighbor_sampling:
  num_neighbors:
  - [24, 42, 42]
optimization:
  max_epochs: 16
  min_steps_per_epoch: 30
  max_steps_per_epoch: 2000
  max_val_steps: 2000
  max_test_steps: 4000
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
  majority_sampling_ratio:
  -
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
  - [sum, mean, max]
  - [sum, mean, min, max, std]
  activation:
  - relu
  - leaky_relu
  - elu
  - gelu
  normalization:
  - layer_norm
  use_seq_id:
  - false
"""

MODEL_PLANNER_YAML_COL_DNE = """
training_table_generation:
  split: TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.54805, 0.1, 0.35195], indices=None, strict_split_boundaries=True)
training_job:
  num_experiments: 8
  metrics:
  - auroc
  - ap
  - auprc
  tune_metric: auroc
  refit_trainval: false
  refit_full: false
column_processing:
  encoder_overrides:
    upc_1234.promo_upc_flag: Null()
    upc_1234.similar_cat_flag: Null()
    orders_casted_1234.cat_nbr: Null()
    orders_casted_1234.brand_descr: Null()
    orders_casted_1234.vendor_nm: Null()
    orders_casted_1234.key_cat_nbr: Null()
    orders_casted_1234.prod_typ_descr: Null()
    orders_casted_1234.prod_subtype_descr: Null()
    target_static_1234.target: Null()
neighbor_sampling:
  num_neighbors:
  - [24, 42, 42]
optimization:
  max_epochs: 16
  min_steps_per_epoch: 30
  max_steps_per_epoch: 2000
  max_val_steps: 2000
  max_test_steps: 4000
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
  majority_sampling_ratio:
  -
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
  - [sum, mean, max]
  - [sum, mean, min, max, std]
  activation:
  - relu
  - leaky_relu
  - elu
  - gelu
  normalization:
  - layer_norm
  use_seq_id:
  - false
"""


def _get_split_fpid(value: str | None) -> Optional[FPID]:
    from kumo.task.split import Split
    split: Optional[Split] = TrainingTableGenerationPlan.get_split(value)
    if split is None:
        return None
    return split.fpid


def test_training_table_generation_options_in_all_configs():
    """
    When adding a new TrainingTableGeneration option, there are a lot of places
    that need to be updated where the options are passed. This test ensures
    that we don't miss any of them.
    """
    def get_fields(cls):
        return set(cls.__dataclass_fields__)

    def assert_fields_match(a: Set[str], b: Set[str]):
        assert a == b

    # The first place where we add an option is in the
    # TrainingTableGenerationPlan. We'll consider this the source of truth for
    # the fields we need to track.
    # Note that we remove special fields we don't expect to carry over
    fields = get_fields(TrainingTableGenerationPlan)
    fields.remove('task_path_override')
    fields.remove('train_table_path_override')

    # PredictiveQueryV2Config
    pq_v2_config_fields = get_fields(PredictiveQueryV2Config)
    pq_v2_config_fields.remove('name')
    pq_v2_config_fields.remove('graph')
    pq_v2_config_fields.remove('list_distinct_target_is_array')
    pq_v2_config_fields.remove('lp_bp_layout')
    pq_v2_config_fields.remove('drop_earliest_timeframe')
    pq_v2_config_fields.remove('validated_predictive_query')
    assert_fields_match(
        fields - {'entity_candidate', 'entity_candidate_aggregation'},
        pq_v2_config_fields)

    # PredictiveQueryAdvancedOptions
    pq_kwargs_fields = get_fields(PredictiveQueryAdvancedOptions)
    assert_fields_match(
        fields - {'entity_candidate', 'entity_candidate_aggregation'},
        pq_kwargs_fields)


def test_training_table_generation_options_translations(create_node_pred_graph,
                                                        tempdir):
    """
    When adding a new TrainingTableGeneration option, there are a lot of places
    that need to be updated where the options are passed. This test ensures
    that we don't miss any of them.
    """
    def get_fields(cls):
        return set(cls.__dataclass_fields__)

    # The first place where we add an option is in the
    # TrainingTableGenerationPlan. We'll consider this the source of truth for
    # the fields we need to track.
    # Note that we remove special fields we don't expect to carry over
    fields = get_fields(TrainingTableGenerationPlan)
    fields.remove('task_path_override')
    fields.remove('train_table_path_override')
    # TODO(frg100): Do we need to add these back?
    fields.remove('entity_candidate')
    fields.remove('entity_candidate_aggregation')

    # Split is handled a bit differently since it's sometimes a string,
    # sometimes a SplitConfig, and sometimes a Split. That logic is tested
    # elsewhere
    fields.remove('split')

    # Temporarily remove start_time/end_time since they are not supported
    # when _USE_DECOUPLED_SPLITS is False
    fields.remove('start_time')
    fields.remove('end_time')

    # ModelPlan -> PredictiveQueryAdvancedOptions
    for task_type in TaskType:
        for query_type in QueryType:
            for has_train_table_time_col in [True, False]:
                default_model_plan = ModelPlan.from_default(
                    task_type,
                    query_type,
                    has_train_table_time_col=has_train_table_time_col,
                    run_mode=RunMode.FAST,
                )

                # Check values match
                pq_kwargs = to_predictive_query_kwargs(default_model_plan)
                for field in fields:
                    mp_value = inferred_or_missing_type_to_none(
                        getattr(default_model_plan.training_table_generation,
                                field))
                    if field == 'year_over_year':
                        mp_value = inferred_or_missing_type_to_false(
                            getattr(
                                default_model_plan.training_table_generation,
                                field))
                    pq_value = getattr(pq_kwargs, field)
                    assert mp_value == pq_value, f"{field} does not match"

    # Test make_predictive_config_with_graph since this is used to generate
    # the PredictiveQueryV2Config in the Batch Prediction codebase
    graph = create_node_pred_graph('simple_graph', root_dir=tempdir.name)
    query = ('PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) '
             'FOR EACH USERS.USER_ID')
    default_model_plan = ModelPlan.from_default(
        TaskType.REGRESSION,
        QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    from copy import deepcopy
    model_plan = deepcopy(default_model_plan)
    for field in fields:
        if field == 'year_over_year':
            model_plan.training_table_generation.year_over_year = True
        if field == 'forecast_length':
            model_plan.training_table_generation.forecast_length = 10
        if field == 'lag_timesteps':
            model_plan.training_table_generation.lag_timesteps = 11
        if field == 'timeframe_step':
            model_plan.training_table_generation.timeframe_step = 12
        if field == 'train_start_offset':
            model_plan.training_table_generation.train_start_offset = 13
        if field == 'train_end_offset':
            model_plan.training_table_generation.train_end_offset = 14
        if field == 'timeframe_step':
            model_plan.training_table_generation.timeframe_step = 15
        if field == 'weight_col':
            model_plan.training_table_generation.weight_col = 'TRANSACTIONS.AMOUNT'
    # This test covers the YAML translation path specifically, so it parses
    # the plan exactly the way the production callers do.
    train_options, train_options_provided = resolve_train_table_options(
        model_plan_to_yaml(
            model_plan,
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
        ))
    pq_config = make_predictive_config_with_graph(
        name='test',
        query_yaml=query,
        graph=graph,
        train_options=train_options,
        train_options_provided=train_options_provided,
    )
    for field in fields:
        if field == 'weight_col':
            # For temporal node prediction/regression flows, weight_col is
            # intentionally hidden and not propagated from model_plan YAML.
            assert getattr(pq_config, field) is None
            continue
        assert getattr(pq_config, field) != getattr(
            default_model_plan.training_table_generation,
            field), f"{field} was not overridden"
        assert getattr(pq_config,
                       field) == getattr(model_plan.training_table_generation,
                                         field), f"{field} does not match"

    # Test PredictiveQueryAdvancedOptions.apply_overrides
    default_options, default_options_provided = resolve_train_table_options(
        model_plan_to_yaml(
            default_model_plan,
            task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL,
        ))
    pq_v2_config = make_predictive_config_with_graph(
        name='test',
        query_yaml=query,
        graph=graph,
        train_options=default_options,
        train_options_provided=default_options_provided,
    )
    overrides = PredictiveQueryAdvancedOptions(
        forecast_length=10,
        lag_timesteps=10,
        timeframe_step=10,
        train_start_offset=10,
        train_end_offset=10,
    )
    pq_config_with_overrides = overrides.apply_overrides(pq_v2_config)
    for field in fields:
        assert getattr(pq_config_with_overrides,
                       field) == getattr(overrides,
                                         field), f"{field} does not match"

    # Test PredictiveQueryConfig.to_predictive_query_v2_config
    for field in fields:
        assert getattr(pq_v2_config,
                       field) == getattr(overrides,
                                         field), f"{field} does not match"

    # PredictiveQueryV2.from_config and PredictiveQueryV2.to_config
    pq_v2 = PredictiveQueryV2.from_config(pq_v2_config)
    for field in fields:
        assert getattr(pq_v2,
                       field) == getattr(overrides,
                                         field), f"{field} does not match"
    pq_v2_config_from_pq_v2 = pq_v2.config()
    for field in fields:
        assert getattr(pq_v2_config_from_pq_v2,
                       field) == getattr(overrides,
                                         field), f"{field} does not match"
