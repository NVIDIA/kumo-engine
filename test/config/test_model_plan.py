# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import yaml
from kumo_core.model_plan import (
    ColumnProcessingPlan,
    GNNModelPlan,
    GraphTransformerModelPlan,
    ModelPlan,
    TrainingTableGenerationPlan,
)
from kumo_core.types import TaskType
from kumoapi.encoder import Index, Numerical, TextEmbedding
from kumoapi.model_plan import (
    AggregationType,
    LinkPredOutputType,
    MissingType,
    PastEncoderType,
    RunMode,
)
from kumoapi.pquery import QueryType
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import (
    ColumnConfig,
    ForeignKeyConfig,
    GraphConfig,
    PrimaryKeyConfig,
    TableConfig,
)
from kumo.config.model_plan_yaml import (
    model_plan_from_yaml,
    model_plan_to_yaml,
    update_model_plan_from_yaml,
)
from kumo.graph.graph import Graph


@pytest.fixture
def auto_setup_kumo_backend():
    return False


@pytest.fixture
def data_backend(tmp_path):
    """Initialize local data state for tests that build a real
    ``Graph`` and call ``.config()``/``.infer_stypes()``.

    Most model-plan tests disable backend setup via ``auto_setup_kumo_backend``
    (they operate on ``GraphConfig`` directly), but the ``create_graph`` fixture
    yields a live ``Graph`` whose ``data_version`` resolution reaches
    ``get_active_cache()``. The persist dir is derived from deploy_env()
    (kumo.data.get_persist_dir); just drop the lazy cache around the test.
    """
    import kumo.data as data

    data.reset_active_cache_for_test()
    try:
        yield
    finally:
        data.reset_active_cache_for_test()


def test_model_plan_basic():
    plan = ModelPlan()

    assert plan.training_table_generation.weight_col is None

    out = model_plan_to_yaml(
        plan,
        TaskType.BINARY_CLASSIFICATION,
        QueryType.TEMPORAL,
    )

    assert out == """
training_table_generation:
  split: inferred
  train_start_offset: ???
  train_end_offset: ???
  timeframe_step: ???
training_job:
  num_experiments: ???
  metrics: ???
  tune_metric: ???
  pruning: ???
  refit_trainval: true
  refit_full: false
column_processing:
  encoder_overrides:
  na_strategy:
neighbor_sampling:
  num_neighbors: ???
  adaptive_sampling: false
optimization:
  max_epochs: ???
  min_steps_per_epoch: ???
  max_steps_per_epoch: ???
  max_val_steps: ???
  max_test_steps: ???
  loss: ???
  optimizer: ???
  base_lr: ???
  weight_decay: ???
  batch_size: ???
  early_stopping: ???
  lr_scheduler: ???
  majority_sampling_ratio: ???
  weight_mode: ???
model_architecture:
  model: ???
  num_pre_message_passing_layers: ???
  num_post_message_passing_layers: ???
  activation: ???
  normalization: ???
  use_seq_id: ???
  prediction_time_encodings: ???"""[1:]

    assert plan == model_plan_from_yaml("")
    assert plan == model_plan_from_yaml("training_job:\n  num_experiments: ???")
    assert plan == model_plan_from_yaml(out)


def test_model_plan_from_default():
    graph_config = GraphConfig(tables=[
        TableConfig(
            table_name="users",
            pkey=PrimaryKeyConfig(name="user_id"),
        ),
        TableConfig(
            table_name="transactions", fkeys=[
                ForeignKeyConfig(
                    name="user_id",
                    dst_table="users",
                    dst_key="user_id",
                ),
            ]),
    ])
    target_to_entity_paths = [[("transactions", "user_id", "users")]]
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )

    out = model_plan_to_yaml(
        plan,
        TaskType.TEMPORAL_LINK_PREDICTION,
        QueryType.TEMPORAL,
    )
    assert out == """
training_table_generation:
  split: inferred
  train_start_offset: inferred
  train_end_offset: inferred
  timeframe_step: inferred
  weight_col:
training_job:
  num_experiments: 4
  metrics:
  - map@1
  - map@10
  - map@100
  - precision@1
  - precision@10
  - precision@100
  - recall@1
  - recall@10
  - recall@100
  - f1@1
  - f1@10
  - f1@100
  tune_metric: map@10
  pruning:
    min_epochs: 2
    k: 3
    min_delta: 0.0
    patience: 1
  refit_trainval: false
  refit_full: false
column_processing:
  encoder_overrides:
  na_strategy:
neighbor_sampling:
  num_neighbors:
  - hop1:
      default: 12
      users.user_id->transactions.user_id: inferred
    hop2:
      default: 12
  adaptive_sampling: false
optimization:
  max_epochs: 8
  min_steps_per_epoch: 30
  max_steps_per_epoch: 1000
  max_val_steps: 1000
  max_test_steps: 2000
  loss:
  - cross_entropy
  optimizer:
  - Adam
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
  lookup_optimization:
  - inferred
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
    dropout:
    - 0.0
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
  module: context
  handle_new_target_entities: false
  target_embedding_mode:
  - lookup
  output_embedding_dim:
  - 32
  ranking_embedding_loss_coeff:
  - 0.0
  embedding_norm_reg_coeff:
  - 0.0
  distance_measure:
  - dot_product
  use_seq_id:
  - false"""[1:]

    plan.validate(
        TaskType.TEMPORAL_LINK_PREDICTION,
        QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )


def test_temporal_two_tower_is_first_class():
    # two_tower temporal LP is first-class: the TT-NG temporal-MIPS RHS
    # seed_time path is the only backend, so validation accepts it (there is
    # no legacy gate to reject it anymore).
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.model_architecture.module = LinkPredOutputType.TWO_TOWER

    plan.validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )


def test_model_plan_forecasting():
    plan = ModelPlan.from_default(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    plan.validate(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'

    with pytest.raises(ValueError, match="Forecasting requires a "):
        plan.validate(
            task_type=TaskType.FORECASTING,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


def test_model_plan_hidden():
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    yaml = model_plan_to_yaml(
        plan,
        TaskType.BINARY_CLASSIFICATION,
        QueryType.TEMPORAL,
    )

    assert 'manual_seed' not in yaml
    plan.training_job.manual_seed = 12345
    assert 'disable_compilation' not in yaml
    plan.training_job.disable_compilation = False

    yaml = model_plan_to_yaml(
        plan,
        TaskType.BINARY_CLASSIFICATION,
        QueryType.TEMPORAL,
    )

    assert 'manual_seed' in yaml
    assert 'disable_compilation' in yaml


@pytest.mark.parametrize('task_type', set(TaskType) - {TaskType.INVALID})
@pytest.mark.parametrize('query_type', list(QueryType))
@pytest.mark.parametrize('has_train_table_time_col', [True, False])
@pytest.mark.parametrize('has_train_table_weight_col', [True, False])
@pytest.mark.parametrize('run_mode', list(RunMode))
def test_from_yaml(
    task_type: TaskType,
    query_type: QueryType,
    has_train_table_time_col: bool,
    has_train_table_weight_col: bool,
    run_mode: RunMode,
):
    if (task_type == TaskType.TEMPORAL_LINK_PREDICTION
            and query_type == QueryType.STATIC):
        return
    if (task_type == TaskType.FORECASTING and query_type == QueryType.STATIC):
        return
    if (task_type == TaskType.STATIC_LINK_PREDICTION
            and query_type == QueryType.TEMPORAL):
        return

    plan = ModelPlan.from_default(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        run_mode=run_mode,
        has_train_table_weight_col=has_train_table_weight_col,
    )
    yaml = model_plan_to_yaml(plan, task_type, query_type,
                              has_train_table_weight_col)

    plan = model_plan_from_yaml(yaml)
    plan.validate(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        run_mode=run_mode,
        has_train_table_weight_col=has_train_table_weight_col,
    )


@pytest.mark.parametrize('split_text', [
    'TimeRangeSplit(['
    '("2024-02-20", "2024-07-09"), '
    '("2024-07-09 23:53:03", "2024-07-16 23:53:03"), '
    '("2024-08-28 23:53:03", "2024-09-04 23:53:03")'
    '])',
    'TimeRangeSplit(['
    '("2024-02-20 23:53:03", "2024-07-09 23:53:03"), '
    '("2024-07-09 23:53:03", "2024-07-16 23:53:03"), '
    '("2024-08-28 23:53:03", "2024-09-04 23:53:03")'
    '])',
    'TimeRangeSplit(['
    '("2024-02-20 10:00:00", "2024-07-09 10:00:00"), '
    '("2024-07-09 10:00:00", "2024-07-16 10:00:00")'
    '])',
    'TimeRangeSplit(['
    '("2024-02-20", "2024-07-09 23:53:03"), '
    '("2024-07-09 23:53:03", "2024-07-16"), '
    '("2024-08-28 23:53:03", "2024-09-04 23:53:03")'
    '])',
    'TimeRangeSplit(['
    '("2024-02-20 00:00:00", "2024-07-09"), '
    '("2024-07-09", "2024-07-16"), '
    '("2024-07-16", "2024-07-23")'
    '])',
    'TimeRangeSplit(['
    '("2024-02-20", "2024-07-09"), '
    '("2024-07-09", "2024-07-16"), '
    '("2024-08-28 23:53:03", "2024-09-04 23:53:03")'
    '])',
    'TimeSplit(timestamp_boundaries=['
    '("2024-02-20", "2024-07-09"), '
    '("2024-07-09 23:53:03", "2024-07-16 23:53:03"), '
    '("2024-08-28 23:53:03", "2024-09-04 23:53:03")'
    '])',
])
def test_split_literals(split_text):
    task_type = TaskType.BINARY_CLASSIFICATION
    query_type = QueryType.TEMPORAL
    plan = ModelPlan.from_default(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    for use_single_quotes in [True, False]:
        if use_single_quotes:
            split_text = split_text.replace('"', "'")
        plan.training_table_generation.split = split_text

        parsed_plan = model_plan_from_yaml(
            model_plan_to_yaml(plan, task_type, query_type))
        assert parsed_plan.training_table_generation.split == (
            plan.training_table_generation.split)


def test_invalid_yaml():
    yaml = """
    - training_table_generation
    """
    with pytest.raises(ValueError,
                       match="Could not parse the Model Plan as valid YAML"):
        model_plan_from_yaml(yaml)

    yaml = """
    training_table_generation:
      split:TemporalSplit([0.8, 0.2])
      train_start_offset:   10
    model_architecture:
      channels:
      -    16
      use_seq_id:
      -true
    """[1:]
    plan = model_plan_from_yaml(yaml)

    assert plan.training_table_generation.split == 'TemporalSplit([0.8, 0.2])'
    assert plan.training_table_generation.train_start_offset == 10
    assert plan.model_architecture.channels == [16]
    assert plan.model_architecture.use_seq_id == [True]

    yaml = """
    training_table_generation:
      train_table_path_override:s3://example-bucket
    """[1:]
    plan = model_plan_from_yaml(yaml)
    assert (plan.training_table_generation.train_table_path_override ==
            's3://example-bucket')

    yaml = """
    training_table_generation:
      train_table_path_override: s3://example-bucket
    """[1:]
    plan = model_plan_from_yaml(yaml)
    assert (plan.training_table_generation.train_table_path_override ==
            's3://example-bucket')


def test_model_plan_validator():
    yaml = """
    training_table_generation:
      train_start_offset: -1
    """[1:]
    with pytest.raises(ValueError,
                       match="Input should be greater than or equal to"):
        model_plan_from_yaml(yaml)

    yaml = """
    training_table_generation:
      train_end_offset: -1
    """[1:]
    with pytest.raises(ValueError,
                       match="Input should be greater than or equal to"):
        model_plan_from_yaml(yaml)

    yaml = """
    neighbor_sampling:
      max_target_neighbors_per_entity:
      - -2
    """[1:]
    with pytest.raises(ValueError,
                       match="Input should be greater than or equal to"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lr_scheduler:
      - name: cosine_with_warmup_restarts
        interval: full_run
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      channels: [test]
    """[1:]
    with pytest.raises(ValueError, match="Input should be a valid integer"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      activation: [super_relu]
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lr_scheduler:
      - name: exponential_lr
        interval: epoch
    """[1:]
    with pytest.raises(ValueError, match="is not a valid learning rate"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      channels: []
    """[1:]
    with pytest.raises(ValueError, match="List should have at least 1"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      channels: [0]
    """[1:]
    with pytest.raises(ValueError,
                       match="Input should be greater than or equal to"):
        model_plan_from_yaml(yaml)

    yaml = """
    training_table_generation:
      split: foo
    """[1:]
    with pytest.raises(ValueError, match="is not a valid split"):
        model_plan_from_yaml(yaml)

    yaml = """
    training_table_generation:
      split: TemporalSplit([0.8, 0.1, 0.1]
    """[1:]
    with pytest.raises(ValueError, match="is not a valid split"):
        model_plan_from_yaml(yaml)

    yaml = """
    column_processing:
      encoder_overrides:
        COL_1: Null()
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match="is not a valid column name"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )

    yaml = """
    column_processing:
      encoder_overrides:
        COL_1: Null()
        COL_2: Null()
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match="are not valid column names"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )

    yaml = """
    column_processing:
      encoder_overrides:
        TABLE_1.COL_1: Foo()
    """[1:]
    with pytest.raises(ValueError, match="is not a valid encoder"):
        model_plan_from_yaml(yaml)

    yaml = """
    column_processing:
      encoder_overrides:
        TABLE_1.COL_1: Foo()
        TABLE_1.COL_2: Bar()
    """[1:]
    with pytest.raises(ValueError, match="are not valid encoders"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lr_scheduler:
      - name: cosine_with_warmup_restarts
        interval: step
        kwargs:
          foo: 0.1
    """[1:]
    with pytest.raises(ValueError, match="is not a valid learning rate"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lr_scheduler:
      - name: exponential_lr
        interval: epoch
      - name: cosine_with_warmup_restarts
        interval: step
        kwargs:
          foo: 0.1
    """[1:]
    with pytest.raises(ValueError, match="are not valid learning rate"):
        model_plan_from_yaml(yaml)

    yaml = """
    neighbor_sampling:
      max_target_neighbors_per_entity:
      - 513
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 512"):
        model_plan_from_yaml(yaml)

    yaml = """
    neighbor_sampling:
      num_neighbors:
      - [16, 16, 16, 16, 16, 16, 17]
    """[1:]
    with pytest.raises(ValueError, match="at most 6 items"):
        model_plan_from_yaml(yaml)

    yaml = """
    neighbor_sampling:
      num_neighbors:
      - [129]
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 128"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      batch_size:
      - 2049
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 2048"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      channels:
      - 513
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 512"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      output_embedding_dim:
      - 257
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 256"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      channels:
      - 55
    """[1:]
    with pytest.raises(ValueError, match="requires an even number"):
        model_plan_from_yaml(yaml)

    yaml = """
    training_table_generation:
      timeframe_step: 0
    """[1:]
    with pytest.raises(ValueError, match="greater than 0"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      min_steps_per_epoch: 4001
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 4000"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      num_pre_message_passing_layers:
      - 5
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 4"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      num_post_message_passing_layers:
      - 9
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 8"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      model:
      - name: GNN
        channels:
        - 4
        - 8
    """[1:]
    res = model_plan_from_yaml(yaml)
    assert res.model_architecture.model[0].name == 'GNN'
    assert res.model_architecture.model[0].channels == [4, 8]

    yaml = """
    model_architecture:
      model:
      - name: GNN
        channels:
        - 65
    """[1:]
    with pytest.raises(ValueError, match="requires an even number"):
        model_plan_from_yaml(yaml)

    yaml = """
    model_architecture:
      model:
      - name: GNN
        aggregation:
        - [cosmic_max]
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    neighbor_sampling:
      sample_from_entity_table: false
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match="not supported for temporal"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    optimization:
      loss:
      - name: huber
        delta: 0
    """[1:]
    with pytest.raises(ValueError, match="greater than 0"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      loss:
      - name: focal
        gamma: 0.5
    """[1:]
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      loss:
      - name: quantile
        q: 1.1
    """[1:]
    with pytest.raises(ValueError, match="less than or equal to 1"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      loss:
      - name: quantile
        q: -0.1
    """[1:]
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      loss:
      - multi_quantile
    """[1:]
    plan = model_plan_from_yaml(yaml)
    plan.validate(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert plan.optimization.loss[0] == 'multi_quantile'

    yaml = """
    optimization:
      loss:
      - mae
      - mse
      - binary_cross_entropy
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match=r"\[mae, mse\] are not valid"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    optimization:
      loss:
      - name: quantile
        q: 0.8
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match=r"\['quantile'\] are not valid"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    optimization:
      loss:
      - multi_quantile
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match=r"\[multi_quantile\] are not valid"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    optimization:
      loss:
      - name: ordinal_log
        alpha: 0.12
    """[1:]
    plan = model_plan_from_yaml(yaml)
    plan.validate(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert plan.optimization.loss[0].name == 'ordinal_log'
    assert plan.optimization.loss[0].alpha == 0.12

    yaml = """
    optimization:
      loss:
      - name: ordinal_log
        alpha: 0.12
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match=r"\['ordinal_log'\] are not valid"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    optimization:
      lr_scheduler:
      - name: reduce_lr_on_plateau
        interval: epoch
    """[1:]
    with pytest.raises(ValueError, match="is not a valid learning rate"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      loss:
      - wape
    """[1:]
    plan = model_plan_from_yaml(yaml)
    plan.validate(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert plan.optimization.loss == ['wape']

    yaml = """
    optimization:
      loss:
      - wape
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(Exception, match=r"are not valid"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


def test_model_plan_max_target_neighbors_per_entity():
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert 'max_target_neighbors_per_entity' not in model_plan_to_yaml(
        plan,
        TaskType.BINARY_CLASSIFICATION,
        QueryType.STATIC,
    )
    assert isinstance(plan.neighbor_sampling.max_target_neighbors_per_entity,
                      MissingType)
    plan.neighbor_sampling.max_target_neighbors_per_entity = [8]
    with pytest.raises(ValueError, match=r"not supported for static binary"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert 'max_target_neighbors_per_entity' not in model_plan_to_yaml(
        plan,
        TaskType.STATIC_LINK_PREDICTION,
        QueryType.STATIC,
    )
    assert isinstance(plan.neighbor_sampling.max_target_neighbors_per_entity,
                      MissingType)

    plan.neighbor_sampling.max_target_neighbors_per_entity = [8]
    assert 'max_target_neighbors_per_entity' in model_plan_to_yaml(
        plan,
        TaskType.STATIC_LINK_PREDICTION,
        QueryType.STATIC,
    )
    plan.validate(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )


def test_model_plan_validate_incompatible_option():
    # Setting `majority_sampling_ratio` for incompatible task:
    plan = ModelPlan.from_default(
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.optimization.majority_sampling_ratio = [100]

    with pytest.raises(ValueError, match=r"supported for temporal multilabel"):
        plan.validate(
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    # `RandomSplit` is not supported for temporal link prediction problem for
    # now
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'RandomSplit([0.8, 0.1, 0.1])'
    with pytest.raises(
            ValueError,
            match=r"Temporal link prediction requires a temporal split"):
        plan.validate(
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    # `RandomLinkSplit` is not supported for temporal link prediction problem
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'RandomLinkSplit([0.8, 0.1, 0.1])'
    with pytest.raises(
            ValueError,
            match=r"Temporal link prediction requires a temporal split"):
        plan.validate(
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.weight_col = 'users.weight'
    with pytest.raises(ValueError, match=r"not supported"):
        plan.validate(
            task_type=TaskType.FORECASTING,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    # `TemporalSplit` is not supported for static problems without timestamp
    # column in target
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'

    with pytest.raises(ValueError,
                       match="the target table doesn't have a 'Create Date'"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )

    # `TemporalSplit` is not supported for static link problems
    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'
    with pytest.raises(ValueError,
                       match="the target table doesn't have a 'Create Date'"):
        plan.validate(
            task_type=TaskType.STATIC_LINK_PREDICTION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )
    # GraphTransformerModelPlan is not supported for link prediction
    for task_type in [
            TaskType.STATIC_LINK_PREDICTION, TaskType.TEMPORAL_LINK_PREDICTION
    ]:
        plan = ModelPlan.from_default(
            task_type=task_type,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )
        plan.model_architecture.model = [
            GraphTransformerModelPlan(
                channels=[16, 32],
                num_heads=[4],
                num_layers=[2],
            )
        ]
        with pytest.raises(
                ValueError,
                match=r"GraphTransformer models are not supported for "
                r"Link Prediction"):
            plan.validate(
                task_type=task_type,
                query_type=QueryType.STATIC,
                has_train_table_time_col=False,
                run_mode=RunMode.FAST,
            )

    # Setting `channels` and `aggregation` at both model level and model plan
    # level is not allowed:
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.model_architecture.model = [
        GNNModelPlan(
            channels=[16],
            aggregation=[['sum', 'mean']],
        ),
        GraphTransformerModelPlan(channels=[16]),
    ]
    plan.model_architecture.aggregation = [['sum', 'mean']]
    plan.model_architecture.channels = [16]
    _, warnings = plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert len(warnings) == 3
    assert "deprecated 'model_architecture.channels" in warnings[0].message
    assert "deprecated 'model_architecture.aggregation" in warnings[1].message
    assert "deprecated 'model_architecture.channels" in warnings[2].message

    # Setting `forecast_length` for incompatible task:
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.forecast_length = 2

    with pytest.raises(ValueError,
                       match=r"not supported for temporal link prediction"):
        plan.validate(
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.forecast_length = 2

    with pytest.raises(ValueError, match=r"supported for link prediction"):
        plan.validate(
            task_type=TaskType.LINK_PREDICTION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.forecast_length = 2

    with pytest.raises(ValueError,
                       match=r"not supported for static regression"):
        plan.validate(
            task_type=TaskType.REGRESSION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.lag_timesteps = 10
    plan.model_architecture.past_encoder = [PastEncoderType.MLP]

    with pytest.raises(ValueError,
                       match=r"not supported for static regression"):
        plan.validate(
            task_type=TaskType.REGRESSION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


# ---------------------------------------------------------------------------
# `ManualSplit`: the caller assigns every row to a stage themselves.
#
# `kumo.task.split.ManualSplit.is_temporal_split` is `False`, and correctly
# so: the property is documented as "derived from `table.time_col`", and a
# manual split derives nothing from time. The gates below must not read a
# split's provenance as a proxy for the question they actually ask.
#
# Note these gates read `TrainingTableGenerationPlan.SplitDescriptor`, a
# name-based property of its own -- not `Split.is_temporal_split`.
# ---------------------------------------------------------------------------

MANUAL_SPLIT = ("ManualSplit(col='SPLIT', train_value=0, val_value=1, "
                "test_value=2)")


def test_manual_split_descriptor_is_row_assigned_not_time_derived():
    descriptor = TrainingTableGenerationPlan.get_split(MANUAL_SPLIT)

    assert descriptor is not None and descriptor.name == 'ManualSplit'
    # "Not random" is not the same question as "derived from time". A
    # `ManualSplit` reads its stage out of a column the caller filled in, so
    # it is neither.
    assert not descriptor.is_temporal_split
    assert descriptor.assigns_stages_explicitly


@pytest.mark.parametrize(('split', 'is_temporal'), [
    ('ApproxDateOffsetSplit([0.8, 0.1, 0.1])', True),
    ('DateOffsetSplit([-2, 0])', True),
    ('TemporalSplit([0.8, 0.1, 0.1])', True),
    ("TimeRangeSplit([['2024-01-01', '2024-02-01']])", True),
    ('TimeSplit(fractions=[0.8, 0.1, 0.1])', True),
    ('RandomSplit([0.8, 0.1, 0.1])', False),
    ('RandomLinkSplit([0.8, 0.1, 0.1])', False),
])
def test_split_descriptor_temporal_answer_is_unchanged(split, is_temporal):
    # Enumerating the time-derived families rather than negating the random
    # ones must not move the answer for any other family.
    descriptor = TrainingTableGenerationPlan.get_split(split)

    assert descriptor is not None
    assert descriptor.is_temporal_split is is_temporal
    assert not descriptor.assigns_stages_explicitly


def test_temporal_link_prediction_accepts_a_manual_split():
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = MANUAL_SPLIT

    # Placing every row in a stage *is* the assurance the temporal-link gate
    # exists to obtain, and it is the combination a supplied training table
    # with a split column exists to serve.
    plan.validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )


@pytest.mark.parametrize(
    'split',
    ['RandomSplit([0.8, 0.1, 0.1])', 'RandomLinkSplit([0.8, 0.1, 0.1])'])
def test_temporal_link_prediction_still_rejects_a_random_split(split):
    # A random split carries no such assurance; accepting `ManualSplit` must
    # not widen the gate to the families the TODO above it is about.
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = split

    with pytest.raises(
            ValueError,
            match=r"Temporal link prediction requires a temporal split"):
        plan.validate(
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


@pytest.mark.parametrize('has_train_table_time_col', [False, True])
@pytest.mark.parametrize('query_type', [QueryType.STATIC, QueryType.TEMPORAL])
def test_non_temporal_task_still_accepts_a_manual_split(
        query_type, has_train_table_time_col):
    # Neither the "is this time-derived split missing its time column?" gate
    # nor the static-with-timestamp family list may catch a `ManualSplit`: it
    # derives nothing from time, and it has already answered the leakage
    # question both of them exist to ask.
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = MANUAL_SPLIT

    plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        run_mode=RunMode.FAST,
    )


def test_a_temporal_split_without_a_time_column_is_still_rejected():
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'

    with pytest.raises(ValueError,
                       match=r"the target table doesn't have a 'Create Date'"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )


def test_model_plan_validate_weight_col_against_graph_config():
    graph_config = GraphConfig(tables=[
        TableConfig(
            table_name="users",
            pkey=PrimaryKeyConfig(name="user_id"),
            cols=[ColumnConfig(name="user_weight", dtype=Dtype.float)],
        ),
        TableConfig(
            table_name="transactions",
            fkeys=[
                ForeignKeyConfig(
                    name="user_id",
                    dst_table="users",
                    dst_key="user_id",
                ),
            ],
            cols=[
                ColumnConfig(name="amount", dtype=Dtype.float),
                ColumnConfig(name="status", dtype=Dtype.string),
            ],
        ),
    ])
    target_to_entity_paths = [[("transactions", "user_id", "users")]]
    validate_kwargs = dict(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
        graph=graph_config,
        entity_table_names=["users"],
        target_to_entity_paths=target_to_entity_paths,
    )

    unsupported_plan = ModelPlan.from_default(**validate_kwargs)
    unsupported_plan.training_table_generation.weight_col = "transactions.amount"
    with pytest.raises(
            ValueError, match=r"Weighted training-table generation is "
            r"not yet supported on DiskGraph"):
        unsupported_plan.validate(**validate_kwargs)


def test_model_plan_validate_warnings():
    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.neighbor_sampling.num_neighbors = [[1, 1, 1], [3, 3, 3, 3]]
    plan.model_architecture.model = [
        GNNModelPlan(
            channels=[256, 512],
            aggregation=[['sum', 'mean', 'max', 'std']],
        ),
    ]
    plan.model_architecture.output_embedding_dim = [256]
    errors, warnings = plan.validate(
        TaskType.STATIC_LINK_PREDICTION,
        QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
        graph=MagicMock(),
        rhs_table_name_rows=("rhs_table_name", 5_000_001),
        raise_on_error=False,
    )
    assert len(errors) == 0
    assert len(warnings) == 3
    assert "cause a GPU out-of-memory error" in str(warnings[0])
    assert "cause a GPU out-of-memory error" in str(warnings[1])
    assert "cause a GPU out-of-memory error" in str(warnings[2])


def test_model_plan_catch_all_errors_and_warnings():
    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )

    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'
    plan.training_job.metrics = ['asdf']
    plan.column_processing.encoder_overrides = {
        'csvv.LOAN.AMOUNT': 'Numerical()',
    }
    plan.neighbor_sampling.max_target_neighbors_per_entity = [8]
    plan.optimization.loss = ["mae"]
    plan.model_architecture.handle_new_target_entities = True
    plan.model_architecture.target_embedding_mode = ['lookup']
    plan.model_architecture.model[0].channels = [256, 512]
    plan.model_architecture.model[0].aggregation = [[
        'sum', 'mean', 'max', 'std'
    ]]

    errors, warnings = plan.validate(
        TaskType.STATIC_LINK_PREDICTION,
        QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
        graph=MagicMock(),
        raise_on_error=False,
    )
    assert len(errors) == 5
    assert len(warnings) == 1


def test_model_plan_validate_models():
    def validate_plan(this_plan):
        return this_plan.validate(
            TaskType.STATIC_LINK_PREDICTION,
            QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
            graph=MagicMock(),
            rhs_table_name_rows=("rhs_table_name", 5_000),
            raise_on_error=False,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.neighbor_sampling.num_neighbors = [[1, 1, 1]]
    plan.model_architecture.model[0].channels = [32]
    plan.model_architecture.model[0].aggregation = [['sum', 'mean']]
    plan.model_architecture.output_embedding_dim = [32]
    # No errors or warnings
    errors, warnings = validate_plan(plan)
    assert len(errors) == 0
    assert len(warnings) == 0

    # Error: GT are not for link pred
    plan.model_architecture.model = [
        GraphTransformerModelPlan(
            channels=[16],
            num_heads=[4],
        ),
    ]
    errors, warnings = validate_plan(plan)
    assert len(errors) == 1
    assert len(warnings) == 0
    assert "Link Prediction" in str(errors[0])

    # Error: channels is not divisible by num_heads, GT are not for link pred
    plan.model_architecture.model = [
        GraphTransformerModelPlan(
            channels=[16],
            num_heads=[3],
            num_layers=[1],
            dropout=[0.1],
        ),
    ]
    errors, warnings = validate_plan(plan)
    assert len(errors) == 2
    assert len(warnings) == 0
    assert "must be divisible" in str(errors[0])

    # Error: Warning about large number of aggregation operators
    plan.model_architecture.model = [
        GNNModelPlan(
            channels=[512],
            aggregation=[['sum', 'mean', 'max', 'std']],
        ),
    ]
    errors, warnings = validate_plan(plan)
    assert len(errors) == 0
    assert len(warnings) == 1
    assert "aggregation" in str(warnings[0])

    # GT are not for link pred, everything else is good
    plan.model_architecture.model = [
        GraphTransformerModelPlan(
            channels=[16, 32],
            num_heads=[4],
            num_layers=[2],
            dropout=[0.2],
        ),
        GNNModelPlan(
            channels=[16],
            aggregation=[['sum', 'mean']],
        ),
    ]
    errors, warnings = validate_plan(plan)
    assert len(errors) == 1
    assert len(warnings) == 0
    assert "Link Prediction" in str(errors[0])


def test_model_plan_serialization():
    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.neighbor_sampling.num_neighbors = [[1, 1, 1]]
    plan.model_architecture.output_embedding_dim = [32]
    plan.model_architecture.model = [
        GraphTransformerModelPlan(
            channels=[16, 32],
            num_heads=[4],
            num_layers=[2],
            dropout=[0.2],
        ),
        GNNModelPlan(
            channels=[16],
            aggregation=[['sum', 'mean']],
        ),
    ]

    target_plan = model_plan_from_yaml(
        model_plan_to_yaml(
            plan,
            TaskType.STATIC_LINK_PREDICTION,
            QueryType.STATIC,
        ))
    errors, warnings = target_plan.validate(
        TaskType.STATIC_LINK_PREDICTION,
        QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
        graph=MagicMock(),
        rhs_table_name_rows=("rhs_table_name", 5_000),
        raise_on_error=False,
    )
    assert len(errors) == 1
    assert len(warnings) == 0
    assert "Link Prediction" in str(errors[0])


def test_model_plan_metric_assignment():
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert isinstance(plan.training_job.metrics, list)
    plan.training_job.metrics.append('map@100')
    with pytest.raises(ValueError, match="Unsupported metric 'map@'"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_job.tune_metric = 'precision@100'
    with pytest.raises(ValueError, match=r"tune metric 'precision@100'"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


def test_model_plan_split_assignment():
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )

    plan.training_table_generation.split = 'RandomSplit([0.8, 0.1, 0.1])'
    plan.validate(
        TaskType.BINARY_CLASSIFICATION,
        QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )

    plan.training_table_generation.split = 'TemporalSplit([0.8, 0.1, 0.1])'
    with pytest.raises(ValueError,
                       match="the target table doesn't have a 'Create Date'"):
        plan.validate(
            TaskType.BINARY_CLASSIFICATION,
            QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )

    plan = ModelPlan.from_default(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = 'RandomLinkSplit([0.8, 0.1, 0.1])'
    plan.validate(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )

    # Raw `TimeSplit` is an internal representation and is now rejected as a
    # user-supplied split on temporal binary classification. Users must use a
    # public split family such as `TemporalSplit(ratio=[...])`. This covers
    # timestamp_boundaries / indices / fractions.
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    def _assert_raw_time_split_rejected(split_repr: str) -> None:
        plan.training_table_generation.split = split_repr
        errors, _ = plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
            raise_on_error=False,
        )
        assert any("internal representation" in str(err)
                   and 'TemporalSplit(ratio=' in str(err) for err in errors), (
                       f"expected raw TimeSplit rejection for {split_repr!r}, "
                       f"got errors={errors}")

    # TimeSplit with timestamp boundaries
    _assert_raw_time_split_rejected('TimeSplit(timestamp_boundaries=['
                                    '("2024-02-20", "2024-07-09"), '
                                    '("2024-07-09", "2024-07-16"), '
                                    '("2024-07-16", "2024-07-23")'
                                    '])')

    # TimeSplit with indices
    _assert_raw_time_split_rejected('TimeSplit('
                                    'indices=[(-180, -120), (-120, -60), '
                                    '(-60, 0)], '
                                    'unit="days", '
                                    'timeframe_size=7, '
                                    'anchor_start_offset=0'
                                    ')')

    # TimeSplit with fractions
    _assert_raw_time_split_rejected('TimeSplit('
                                    'fractions=[0.7, 0.2, 0.1], '
                                    'unit="days", '
                                    'timeframe_size=7, '
                                    'anchor_start_offset=0'
                                    ')')

    # A public split family (e.g. `TemporalSplit`) validates on temporal
    # queries (it only warns about possible leakage, it does not error).
    plan.training_table_generation.split = (
        'TemporalSplit(ratio=[0.7, 0.2, 0.1])')
    plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Test invalid TimeSplit - invalid time unit
    with pytest.raises(ValueError, match="'years' is not a valid TimeUnit"):
        plan.training_table_generation.split = (
            'TimeSplit('
            'indices=[(-180, -120), (-120, -60), (-60, 0)], '
            'unit="years", '  # Invalid unit
            'timeframe_size=7, '
            'anchor_start_offset=0'
            ')')

    # Test invalid TimeSplit - overlapping boundaries
    with pytest.raises(ValueError,
                       match="must be strictly smaller than the start"):
        plan.training_table_generation.split = (
            'TimeSplit(timestamp_boundaries=['
            '("2024-02-20", "2024-07-09"), '
            '("2024-07-08", "2024-07-16"), '  # Overlaps with previous
            '("2024-07-16", "2024-07-23")'
            '])')


_RAW_TIME_SPLIT = ('TimeSplit(fractions=[0.7, 0.2, 0.1], unit="days", '
                   'timeframe_size=7, anchor_start_offset=0)')


def _validate_split(task_type, query_type, has_time_col, split_repr):
    plan = ModelPlan.from_default(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_time_col,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = split_repr
    errors, _ = plan.validate(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_time_col,
        run_mode=RunMode.FAST,
        raise_on_error=False,
    )
    return errors


def _assert_raw_time_split_error(errors):
    assert any("internal representation" in str(err)
               and 'TemporalSplit(ratio=' in str(err)
               for err in errors), f"expected raw TimeSplit rejection, "\
        f"got errors={errors}"


@pytest.mark.parametrize(
    'task_type,query_type,has_time_col',
    [
        # Temporal node prediction / regression.
        (TaskType.REGRESSION, QueryType.TEMPORAL, True),
        (TaskType.BINARY_CLASSIFICATION, QueryType.TEMPORAL, True),
        # Temporal link prediction.
        (TaskType.TEMPORAL_LINK_PREDICTION, QueryType.TEMPORAL, True),
        # Static node prediction.
        (TaskType.REGRESSION, QueryType.STATIC, True),
        # Forecasting.
        (TaskType.FORECASTING, QueryType.TEMPORAL, True),
    ])
def test_raw_time_split_rejected_uniformly(task_type, query_type, has_time_col):
    """Raw user-supplied `TimeSplit` is rejected for every query type."""
    errors = _validate_split(task_type, query_type, has_time_col,
                             _RAW_TIME_SPLIT)
    _assert_raw_time_split_error(errors)


def test_model_plan_target_embedding_mode_assignment():
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    plan.model_architecture.handle_new_target_entities = True

    with pytest.raises(ValueError, match="requires 'feature' target emb"):
        plan.validate(
            TaskType.TEMPORAL_LINK_PREDICTION,
            QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )


@pytest.mark.parametrize('encoder_strings', [True, False])
def test_encoder_overrides(data_backend, create_graph, encoder_strings):
    graph: Graph = create_graph(name='FINANCIAL')

    plan = ModelPlan.from_default(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )

    if encoder_strings:
        plan.column_processing.encoder_overrides = {
            'csv.LOAN.AMOUNT': 'Numerical()',
            'csvv.LOAN.AMOUNT': 'Numerical()',
            'train_table.TARGET': 'Index()',
            'csv.LOOAN.AMOUNT': 'Numerical()',
            'DISTRICTT.A10': 'Numerical()',
            'DISTRICT.A17': 'Numerical()',
        }
    else:
        plan.column_processing.encoder_overrides = {
            'csv.LOAN.AMOUNT': Numerical(),
            'csvv.LOAN.AMOUNT': Numerical(),
            'train_table.TARGET': Index(),
            'csv.LOOAN.AMOUNT': Numerical(),
            'DISTRICTT.A10': Numerical(),
            'DISTRICT.A17': Numerical(),
        }

    with pytest.raises(
            ValueError,
            match=("'csvv.LOAN.AMOUNT', 'csv.LOOAN.AMOUNT', 'DISTRICTT.A10'"
                   " and 'DISTRICT.A17'"),
    ):
        plan.validate(
            TaskType.MULTICLASS_CLASSIFICATION,
            QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
            graph=graph.config(),
        )


@pytest.mark.parametrize('encoder_strings', [True, False])
def test_encoder_overrides_incompatible_encoders(
    data_backend,
    create_graph,
    encoder_strings,
):
    graph: Graph = create_graph(name='FINANCIAL')
    graph.infer_stypes()

    plan = ModelPlan.from_default(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    if encoder_strings:
        plan.column_processing.encoder_overrides = {
            'csv.LOAN.AMOUNT': 'Index()',
            'csv.LOAN.STATUS': 'Numerical()',
        }
    else:
        plan.column_processing.encoder_overrides = {
            'csv.LOAN.AMOUNT': Index(),
            'csv.LOAN.STATUS': Numerical(),
        }

    with pytest.raises(
            ValueError,
            match=(r"'csv.LOAN.AMOUNT'.*'Index\(.*\)'.*'numerical'.*"
                   r"'csv.LOAN.STATUS'.*'Numerical\(.*\)'.*'categorical'"),
    ):
        plan.validate(
            TaskType.MULTICLASS_CLASSIFICATION,
            QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
            graph=graph.config(),
        )


def test_encoder_overrides_removed_encoders():
    plan = ModelPlan.from_default(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    with pytest.raises(ValueError, match=r"'OneHot\(\)' has been removed"):
        plan.column_processing.encoder_overrides = {
            'csv.LOAN.DURATION': 'OneHot()',
        }


@pytest.mark.parametrize('encoder_strings', [True, False])
def test_encoder_overrides_multiple_text_models(encoder_strings):
    graph_config = GraphConfig(tables=[
        TableConfig(
            table_name="table0", cols=[
                ColumnConfig(name="text0", stype=Stype.text),
                ColumnConfig(name="text1", stype=Stype.text),
            ]),
        TableConfig(table_name="table1", cols=[
            ColumnConfig(name="text0", stype=Stype.text),
        ])
    ])

    plan = ModelPlan.from_default(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    if encoder_strings:
        plan.column_processing.encoder_overrides = {
            'table0.text0': 'TextEmbedding(model="glove.6B-50d")',
            'table0.text1': 'TextEmbedding(model="glove.6B-100d")',
            'table1.text0': 'TextEmbedding(model="glove.6B-50d")',
        }
    else:
        plan.column_processing.encoder_overrides = {
            'table0.text0': TextEmbedding('glove.6B-50d'),
            'table0.text1': TextEmbedding('glove.6B-100d'),
            'table1.text0': TextEmbedding('glove.6B-50d'),
        }
    errors, warnings = plan.validate(
        TaskType.MULTICLASS_CLASSIFICATION,
        QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
        graph=graph_config,
        raise_on_error=False,
    )
    assert len(errors) == 0
    assert len(warnings) == 1
    assert "GPU out-of-memory" in str(warnings[0])


def test_encoder_overrides_with_encoder_classes():
    plan = ModelPlan()
    plan.column_processing.encoder_overrides = {
        'COL_1': Index(min_occ=10),
    }

    out = model_plan_to_yaml(
        plan,
        TaskType.BINARY_CLASSIFICATION,
        QueryType.TEMPORAL,
    )
    assert "COL_1: Index(min_occ=10, na_strategy=separate)" in out

    # Check that we can read back in encoder string representation.
    model_plan_from_yaml(out)


def test_parse_text_encoder_overrides():
    encoder_overrides_model_plan = """
    column_processing:
      encoder_overrides:
        table.column: Index(min_occ=10, na_strategy=separate)
    """
    ColumnProcessingPlan.get_encoder_overrides(
        yaml.safe_load(encoder_overrides_model_plan)['column_processing']
        ['encoder_overrides'])


@pytest.mark.parametrize('adaptive_sampling', [True, False])
def test_num_neighbors(adaptive_sampling: bool):
    graph_config = GraphConfig(tables=[  # type: ignore
        TableConfig(
            table_name="users",
            pkey=PrimaryKeyConfig(name="user_id"),
        ),
        TableConfig(
            table_name="transactions", fkeys=[
                ForeignKeyConfig(name="user_id", dst_table="users",
                                 dst_key="user_id"),
            ]),
    ])
    target_to_entity_paths = [[("transactions", "user_id", "users")]]

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        entity_table_names=['users'],
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )

    yaml = f"""
    neighbor_sampling:
      adaptive_sampling: {str(adaptive_sampling).lower()}
      num_neighbors:
      - hop1:
          default: 12
          users.user_id->transactions.user_id: 32
        hop2:
          default: 12
    """[1:]

    update_model_plan_from_yaml(plan, yaml).validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        entity_table_names=['users'],
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )

    yaml = f"""
    neighbor_sampling:
      adaptive_sampling: {str(adaptive_sampling).lower()}
      num_neighbors:
      - hop1:
          default: 12
        hop3:
          default: 12
    """[1:]

    with pytest.raises(ValueError, match="hop definition"):
        update_model_plan_from_yaml(plan, yaml).validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
            entity_table_names=['users'],
            graph=graph_config,
            target_to_entity_paths=target_to_entity_paths,
        )

    yaml = f"""
    neighbor_sampling:
      adaptive_sampling: {str(adaptive_sampling).lower()}
      num_neighbors:
      - hop1:
          default: 12
          USERS.user_id->transactions.user_id: 24
          users.user_id->transactions.USER_ID: 24
    """[1:]
    with pytest.raises(ValueError, match="do not exist in the graph"):
        update_model_plan_from_yaml(plan, yaml).validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
            entity_table_names=['users'],
            graph=graph_config,
            target_to_entity_paths=target_to_entity_paths,
        )

    yaml = f"""
    neighbor_sampling:
      adaptive_sampling: {str(adaptive_sampling).lower()}
      num_neighbors:
      - hop1:
          default: 12
          transactions.user_id->users.user_id: inferred
    """[1:]
    with pytest.raises(ValueError, match="The 'inferred' option"):
        update_model_plan_from_yaml(plan, yaml).validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
            entity_table_names=['users'],
            graph=graph_config,
            target_to_entity_paths=target_to_entity_paths,
        )

    yaml = f"""
    neighbor_sampling:
      adaptive_sampling: {str(adaptive_sampling).lower()}
      num_neighbors:
      - hop1:
          default: 12
          users.user_id->transactions.user_id: inferred
        hop2:
          default: 12
          users.user_id->transactions.user_id: inferred
    """[1:]
    with pytest.raises(ValueError, match="only allowed in the first hop"):
        update_model_plan_from_yaml(plan, yaml).validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
            entity_table_names=['users'],
            graph=graph_config,
            target_to_entity_paths=target_to_entity_paths,
        )


@pytest.mark.parametrize('array_config, num_neighbors, expect_oom',
                         [(False, 500, False), (False, 512, True),
                          (True, [128, 4, 1, 1], False),
                          (True, [128, 8, 1, 1], True)])
def test_num_neighbors_adaptive_sampling_oom(
    array_config: bool,
    num_neighbors: int | list[int],
    expect_oom: bool,
):
    graph_config = GraphConfig(tables=[  # type: ignore
        TableConfig(
            table_name="users",
            pkey=PrimaryKeyConfig(name="user_id"),
        ),
        TableConfig(
            table_name="articles",
            pkey=PrimaryKeyConfig(name="article_id"),
        ),
        TableConfig(
            table_name="transactions", fkeys=[
                ForeignKeyConfig(name="user_id", dst_table="users",
                                 dst_key="user_id"),
                ForeignKeyConfig(name="article_id", dst_table="articles",
                                 dst_key="article_id"),
            ]),
        TableConfig(
            table_name="views", fkeys=[
                ForeignKeyConfig(name="user_id", dst_table="users",
                                 dst_key="user_id"),
                ForeignKeyConfig(name="article_id", dst_table="articles",
                                 dst_key="article_id"),
            ]),
    ])
    target_to_entity_paths = [[("transactions", "user_id", "users")]]

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )

    if array_config:
        # Due to the lack of finegrained control, we cannot exclude views
        # from the 3rd and 5th hop, so we just decrease the number of hops
        yaml = f"""
        neighbor_sampling:
          adaptive_sampling: true
          num_neighbors:
          - {num_neighbors}
        """[1:]
    else:
        yaml = f"""
        neighbor_sampling:
          adaptive_sampling: true
          num_neighbors:
          - hop1:
              default: 12
              users.user_id->transactions.user_id: {num_neighbors}
              users.user_id->views.user_id: {num_neighbors}
            hop2:
              default: 1
            hop3:
              default: 1
              articles.article_id->views.article_id: 0
            hop4:
              default: 1
            hop5:
              default: 1
              users.user_id->views.user_id: 0
            hop6:
              default: 1
        """[1:]

    errors, warnings = update_model_plan_from_yaml(plan, yaml).validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        entity_table_names=['users', 'articles'],
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )
    assert len(errors) == 0
    if not expect_oom:
        assert len(warnings) == 0
    else:
        assert len(warnings) == 1
        assert warnings[0].title == 'GPU Memory: Large Sample Size'
        assert 'produce graph samples of size ' in warnings[0].message


def test_num_neighbors_adaptive_sampling_neg_1():
    graph_config = GraphConfig(tables=[  # type: ignore
        TableConfig(
            table_name="users",
            pkey=PrimaryKeyConfig(name="user_id"),
        ),
        TableConfig(
            table_name="articles",
            pkey=PrimaryKeyConfig(name="article_id"),
        ),
        TableConfig(
            table_name="transactions", fkeys=[
                ForeignKeyConfig(name="user_id", dst_table="users",
                                 dst_key="user_id"),
                ForeignKeyConfig(name="article_id", dst_table="articles",
                                 dst_key="article_id"),
            ]),
        TableConfig(
            table_name="views", fkeys=[
                ForeignKeyConfig(name="user_id", dst_table="users",
                                 dst_key="user_id"),
                ForeignKeyConfig(name="article_id", dst_table="articles",
                                 dst_key="article_id"),
            ]),
    ])
    target_to_entity_paths = [[("transactions", "user_id", "users")]]

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        graph=graph_config,
        target_to_entity_paths=target_to_entity_paths,
    )

    yaml1 = """
        neighbor_sampling:
          adaptive_sampling: true
          num_neighbors:
          - [-1, -1]
        """[1:]
    yaml2 = """
    neighbor_sampling:
      adaptive_sampling: true
      num_neighbors:
      - hop1:
          default: -1
          users.user_id->transactions.user_id: 1
          users.user_id->views.user_id: 2
        hop2:
          default: 1
    """[1:]
    yaml3 = """
    neighbor_sampling:
      adaptive_sampling: true
      num_neighbors:
      - hop1:
          default: 12
          users.user_id->transactions.user_id: -1
          users.user_id->views.user_id: 2
        hop2:
          default: -1
    """[1:]

    for yaml_conf in [yaml1, yaml2, yaml3]:
        with pytest.raises(ValueError, match='Negative neighbor numbers'):
            update_model_plan_from_yaml(plan, yaml_conf).validate(
                task_type=TaskType.BINARY_CLASSIFICATION,
                query_type=QueryType.TEMPORAL,
                has_train_table_time_col=True,
                run_mode=RunMode.FAST,
                entity_table_names=['users', 'articles'],
                graph=graph_config,
                target_to_entity_paths=target_to_entity_paths,
            )


def test_model_plan_decoupled_splits(monkeypatch):
    """Test ModelPlan behavior with decoupled splits enabled/disabled."""
    from kumoapi.model_plan import InferredType, MissingType

    # Test with _USE_DECOUPLED_SPLITS = True
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'true')

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Verify start_time is set and train_start_offset is missing
    assert plan.training_table_generation.start_time == InferredType.VALUE
    assert (
        plan.training_table_generation.train_start_offset == MissingType.VALUE)

    # Verify validation fails if train_start_offset is set
    plan.training_table_generation.train_start_offset = 10
    with pytest.raises(ValueError, match="field is not supported"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    # Test with _USE_DECOUPLED_SPLITS = False
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'false')

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Verify train_start_offset is set and start_time is missing
    assert (
        plan.training_table_generation.train_start_offset == InferredType.VALUE)
    assert plan.training_table_generation.start_time == MissingType.VALUE

    # Verify validation names the supported relative-bound alternative.
    plan.training_table_generation.start_time = "2024-01-01"
    plan.training_table_generation.end_time = "2024-02-01"
    with pytest.raises(ValueError) as error:
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )
    assert str(error.value).count(
        "Use 'train_start_offset' and 'train_end_offset' instead.") == 2

    # Static queries cannot use either absolute or relative bounds.
    static_plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    static_plan.training_table_generation.start_time = "2024-01-01"
    with pytest.raises(ValueError) as error:
        static_plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
        )
    assert "The 'start_time' field is not supported" in str(error.value)
    assert "train_start_offset" not in str(error.value)

    # Case 1: use_decoupled_split=False and static query with TimeSplit raises
    # error
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'false')

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Set a TimeSplit with timestamp boundaries as split
    plan.training_table_generation.split = (
        'TimeSplit(timestamp_boundaries=[("2024-01-01", "2024-03-01"), '
        '("2024-03-01", "2024-05-01"), '
        '("2024-05-01", "2024-07-01")])')

    errors, _ = plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        raise_on_error=False,
    )

    # Should have an error about TimeSplit not being in acceptable_splits
    # TimeSplit is not in acceptable_splits when _USE_DECOUPLED_SPLITS=False
    assert any("contains a timestamp" in str(err) and "TimeSplit" in str(err)
               for err in errors)
    assert any(
        all(split_cls in str(err)
            for split_cls in ["RandomSplit", "TemporalSplit", "TimeRangeSplit"])
        for err in errors)

    # Case 2: raw user-supplied TimeSplit is now rejected uniformly,
    # regardless of _USE_DECOUPLED_SPLITS. The flag-gated acceptance was
    # removed: a raw TimeSplit on a static query with USE_DECOUPLED_SPLITS=True
    # would otherwise be silently dropped by the DG producer. Users should use
    # a public split family such as `TemporalSplit(ratio=[...])` instead.
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'true')

    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Set a TimeSplit with timestamp boundaries as split
    plan.training_table_generation.split = (
        'TimeSplit(timestamp_boundaries=[("2024-01-01", "2024-03-01"), '
        '("2024-03-01", "2024-05-01"), '
        '("2024-05-01", "2024-07-01")])')

    errors, _ = plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        raise_on_error=False,
    )

    # Raw TimeSplit is rejected even with _USE_DECOUPLED_SPLITS=True.
    assert any("internal representation" in str(err)
               and 'TemporalSplit(ratio=' in str(err) for err in errors)

    # Case 3: use_decoupled_split=False and forecasting with TimeSplit raises
    # error
    monkeypatch.setattr('rest.settings._USE_DECOUPLED_SPLITS', False)

    plan = ModelPlan.from_default(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Set a TimeSplit with timestamp boundaries as split
    plan.training_table_generation.split = (
        'TimeSplit(timestamp_boundaries=[("2024-01-01", "2024-03-01"), '
        '("2024-03-01", "2024-05-01"), '
        '("2024-05-01", "2024-07-01")])')

    errors, _ = plan.validate(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        raise_on_error=False,
    )

    # Should have an error about TimeSplit not being in acceptable_splits for
    # forecasting
    assert any("Forecasting requires a" in str(err) for err in errors)
    assert any(
        all(split_cls in str(err)
            for split_cls in ["DateOffsetSplit", "TimeRangeSplit"])
        for err in errors)

    # Case 4: raw user-supplied TimeSplit on forecasting is rejected
    # uniformly, regardless of _USE_DECOUPLED_SPLITS (flag-gated acceptance
    # removed).
    monkeypatch.setattr('rest.settings._USE_DECOUPLED_SPLITS', True)

    plan = ModelPlan.from_default(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # Set a TimeSplit with timestamp boundaries as split
    plan.training_table_generation.split = (
        'TimeSplit(timestamp_boundaries=[("2024-01-01", "2024-03-01"), '
        '("2024-03-01", "2024-05-01"), '
        '("2024-05-01", "2024-07-01")])')

    errors, _ = plan.validate(
        task_type=TaskType.FORECASTING,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
        raise_on_error=False,
    )

    # Raw TimeSplit is rejected even with _USE_DECOUPLED_SPLITS=True.
    assert any("internal representation" in str(err)
               and 'TemporalSplit(ratio=' in str(err) for err in errors)


def test_model_plan_unify() -> None:
    plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    expected_aggs = [
        [
            AggregationType.SUM,
            AggregationType.MEAN,
            AggregationType.MAX,
        ],
        [
            AggregationType.SUM,
            AggregationType.MEAN,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.STD,
        ],
    ]

    # checks that default values are created in the new API
    assert plan.model_architecture.model[0].name == "GNN"
    assert plan.model_architecture.model[0].channels == [64, 128, 256]
    assert plan.model_architecture.model[0].aggregation == expected_aggs

    # checks that unify on the new API doesn't introduce any change
    plan = plan.unify()
    assert plan.model_architecture.model[0].name == "GNN"
    assert plan.model_architecture.model[0].channels == [64, 128, 256]
    assert plan.model_architecture.model[0].aggregation == expected_aggs

    # sets values for the old API without deleting new API
    plan.model_architecture.channels = [4, 8, 32]
    plan = plan.unify()
    assert plan.model_architecture.model[0].name == "GNN"
    assert plan.model_architecture.model[0].channels == [4, 8, 32]
    assert plan.model_architecture.model[0].aggregation == expected_aggs

    plan.model_architecture.aggregation = [[AggregationType.MAX]]
    plan = plan.unify()
    assert plan.model_architecture.model[0].name == "GNN"
    assert plan.model_architecture.model[0].channels == [4, 8, 32]
    assert plan.model_architecture.model[0].aggregation == [[
        AggregationType.MAX
    ]]


def test_static_lp_validator() -> None:
    yaml = """
    optimization:
      target_fraction:
      - 0.5
    model_architecture:
      module: embedding
    """[1:]

    _, warnings = model_plan_from_yaml(yaml).validate(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert "Ignored option 'target_fraction'" in warnings[0].title

    yaml = """
    optimization:
      overlap_ratio:
      - 0.5
    model_architecture:
      module: embedding
    """[1:]

    _, warnings = model_plan_from_yaml(yaml).validate(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert "Ignored option 'overlap_ratio'" in warnings[0].title


def test_temporal_lp_validator() -> None:
    yaml = """
    model_architecture:
      module: embedding
      ranking_embedding_loss_coeff:
      - 1.0
    """[1:]

    _, warnings = model_plan_from_yaml(yaml).validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert "loss_coeff' will be ignored" in warnings[0].message

    yaml = """
    model_architecture:
      module: ranking
      ranking_embedding_loss_coeff:
      - 1.0
    """[1:]

    _, warnings = model_plan_from_yaml(yaml).validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert len(warnings) == 0


@pytest.mark.parametrize('hop_key,error_match', [
    ('customer.CustomerID->transaction.CustomerID->: 32', "too many '->'"),
    ('customer.CustomerIDtransaction.CustomerID: 32',
     "missing the '->' separator"),
    ('->transaction.CustomerID: 32', 'empty source'),
    ('customer.CustomerID->: 32', 'empty destination'),
    ('customerCustomerID->transaction.CustomerID: 32',
     'source is missing the dot separator'),
    ('customer.CustomerID->transactionCustomerID: 32',
     'destination is missing the dot separator'),
])
def test_hop_key_format_errors(hop_key: str, error_match: str):
    yaml_str = f"""
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default: 12
          {hop_key}
    """[1:]
    with pytest.raises(ValueError, match=error_match):
        model_plan_from_yaml(yaml_str)


@pytest.mark.parametrize('hop_key', [
    '.CustomerID->transaction.CustomerID: 32',
    'customer.->transaction.CustomerID: 32',
    'customer.CustomerID->.CustomerID: 32',
    'customer.CustomerID->transaction.: 32',
])
def test_hop_key_empty_parts(hop_key: str):
    graph_config = GraphConfig(tables=[
        TableConfig(table_name="customer", pkey=PrimaryKeyConfig(
            name="CustomerID")),
        TableConfig(table_name="transaction", pkey=PrimaryKeyConfig(
            name="CustomerID")),
    ])
    yaml_str = f"""
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default: 12
          {hop_key}
    """[1:]
    plan = model_plan_from_yaml(yaml_str)
    with pytest.raises(ValueError, match="empty"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
            graph=graph_config,
        )


def test_hop_key_valid():
    yaml = """
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default: 12
          customer.CustomerID->transaction.CustomerID: 32
    """[1:]
    plan = model_plan_from_yaml(yaml)
    assert plan.neighbor_sampling.num_neighbors[0].hop1.__pydantic_extra__[
        'customer.CustomerID->transaction.CustomerID'] == 32


def test_yaml_missing_space_after_colon():
    yaml1 = """
    training_table_generation:
      split:RandomSplit([0.8, 0.1, 0.1])
    """[1:]
    plan1 = model_plan_from_yaml(yaml1)
    assert plan1.training_table_generation.split == 'RandomSplit([0.8, 0.1, 0.1])'

    yaml2 = """
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default:12
    """[1:]
    plan2 = model_plan_from_yaml(yaml2)
    assert plan2.neighbor_sampling.num_neighbors[0].hop1.default == 12


def test_yaml_valid_with_colons_in_timestamps():
    yaml = """
    training_table_generation:
      split: TimeRangeSplit([('2024-01-01 20:00:00', '2024-10-12 20:00:00'), ('2024-10-12 20:00:00', '2024-10-19 20:00:00')])
    """[1:]
    plan = model_plan_from_yaml(yaml)
    assert 'TimeRangeSplit' in plan.training_table_generation.split


def test_yaml_invalid_structure():
    yaml = """
    training_table_generation
      split: RandomSplit([0.8, 0.1, 0.1])
    """[1:]
    with pytest.raises(ValueError,
                       match="Invalid syntax in model plan|Could not parse"):
        model_plan_from_yaml(yaml)


@pytest.mark.parametrize('invalid_yaml', [
    'run_mode: invalid\ntraining_job:\n  num_experiments: 2',
    'invalid_field:\n  nested: value\ntraining_job:\n  num_experiments: 2',
])
def test_model_plan_invalid_top_level_field(invalid_yaml: str):
    with pytest.raises(ValueError, match="top-level field is not allowed"):
        model_plan_from_yaml(invalid_yaml)


@pytest.mark.parametrize('hop_key,tables', [
    ('nonexistent.table->users.user_id: 32', [
        TableConfig(table_name="users", pkey=PrimaryKeyConfig(name="user_id")),
    ]),
    ('users.user_id->transactions.nonexistent_col: 32', [
        TableConfig(table_name="users", pkey=PrimaryKeyConfig(name="user_id")),
        TableConfig(table_name="transactions",
                    pkey=PrimaryKeyConfig(name="trans_id")),
    ]),
])
def test_model_plan_invalid_graph_references(hop_key: str, tables: list):
    graph_config = GraphConfig(tables=tables)
    yaml_str = f"""
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default: 12
          {hop_key}
    """[1:]
    plan = model_plan_from_yaml(yaml_str)
    with pytest.raises(ValueError, match="do not exist in the graph"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.STATIC,
            has_train_table_time_col=False,
            run_mode=RunMode.FAST,
            graph=graph_config,
        )


def test_model_plan_valid_structure():
    yaml = """
    training_table_generation:
      split: RandomSplit([0.8, 0.1, 0.1])
    training_job:
      num_experiments: 2
    column_processing:
      encoder_overrides:
    neighbor_sampling:
      num_neighbors:
      - [12, 12]
    optimization:
      max_epochs: 8
    model_architecture:
      channels:
      - 64
    """[1:]
    plan = model_plan_from_yaml(yaml)
    assert plan.training_job.num_experiments == 2
    assert plan.optimization.max_epochs == 8 or plan.optimization.max_epochs == [
        8
    ]


def test_error_message_format():
    yaml = """
    neighbor_sampling:
      num_neighbors:
      - hop1:
          default: 12
          customer.CustomerID->: 32
    """[1:]
    with pytest.raises(ValueError) as exc_info:
        model_plan_from_yaml(yaml)
    error_msg = str(exc_info.value)
    assert "Invalid syntax in model plan" in error_msg
    assert "Fix by" in error_msg or "fix by" in error_msg.lower()


def test_validate_optimizer():
    yaml = """
    optimization:
      optimizer:
        - Muon
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lookup_optimization:
        - optimizer: Adagrad
    """[1:]
    with pytest.raises(ValueError, match="Field required"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lookup_optimization:
        - optimizer: Adagrad
          base_lr: 0.0
    """[1:]
    with pytest.raises(ValueError, match="Input should be greater than 0"):
        model_plan_from_yaml(yaml)

    yaml = """
    optimization:
      lookup_optimization:
        - optimizer: Adagrad
          base_lr: 0.2
    model_architecture:
      module: embedding
    """[1:]
    plan = model_plan_from_yaml(yaml)
    _, warnings = plan.validate(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.FAST,
    )
    assert len(warnings) == 1
    assert "'lookup_optimization' will be ignored" in warnings[0].message

    yaml = """
    optimization:
      lookup_optimization:
        - optimizer: Adagrad
          base_lr: 0.2
    model_architecture:
      target_embedding_mode:
      - feature
    """[1:]
    plan = model_plan_from_yaml(yaml)
    _, warnings = plan.validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    assert len(warnings) == 1
    assert "'lookup_optimization' will be ignored" in warnings[0].message


def test_na_strategy():
    yaml = """
    column_processing:
      na_strategy:
        number: separate
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    column_processing:
      na_strategy:
        numerical: test
    """[1:]
    with pytest.raises(ValueError, match="Input should be"):
        model_plan_from_yaml(yaml)

    yaml = """
    column_processing:
      na_strategy:
        numerical: most_frequent
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match="is not a supported N/A"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    column_processing:
      na_strategy:
        numerical: most_frequent
        categorical: mean
    """[1:]
    plan = model_plan_from_yaml(yaml)
    with pytest.raises(ValueError, match="hold unsupported N/A"):
        plan.validate(
            task_type=TaskType.BINARY_CLASSIFICATION,
            query_type=QueryType.TEMPORAL,
            has_train_table_time_col=True,
            run_mode=RunMode.FAST,
        )

    yaml = """
    column_processing:
      na_strategy:
        numerical: separate
        categorical: most_frequent
    """[1:]
    plan = model_plan_from_yaml(yaml)
    plan.validate(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
