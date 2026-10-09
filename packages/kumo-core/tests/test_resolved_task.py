# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import kumo_core.automl.auto_trainer as auto_trainer_module
import pytest
from kumo_core.automl.auto_trainer import (
    AutoTrainer,
    _resolved_hop_edge_type,
    infer_link_pred_module_type,
)
from kumo_core.automl.runtime import (
    apply_encoder_overrides,
    apply_na_strategy,
    collect_resolved_encoders,
)
from kumo_core.model_plan import (
    LinkPredOutputType,
    ModelPlan,
    inferred_or_missing_type_to_false,
)
from kumo_core.resolved_task import (
    ResolvedEncoderInput,
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTaskSampling,
    ResolvedTaskSplit,
    ResolvedTrainTable,
    default_metrics,
    plan_validation_facts,
)
from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.train.autoregression import (
    FUTURE_LABELS_COL,
    PADDING_SIZE,
    YOY_COL,
)
from kumo_core.training.config import LinkPredModuleType
from kumo_core.types import TaskType
from kumoapi.encoder import Null, Numerical
from kumoapi.model_plan import MissingType
from kumo_core.modeling.heads.year_over_year import YearOverYearEncoder


def _task(task_type: TaskType) -> ResolvedNumericalTask:
    edge = ("events", "customer", "customers")
    return ResolvedNumericalTask(
        name="retention",
        task_type=task_type,
        metrics=default_metrics(task_type),
        graph=ResolvedTaskGraph(
            node_types=("events", "customers"),
            edge_types=(edge, ),
            seed_node_type="customers",
            target_node_type="events",
        ),
        train_table=ResolvedTrainTable(
            name="train_table",
            target_column="TARGET",
            time_column="TIMESTAMP",
            weight_column="WEIGHT",
        ),
        split=ResolvedTaskSplit(
            stage_counts={
                "train": 80,
                "val": 10,
                "test": 10
            },
            stage_seed_times={"train": (1, 2)},
        ),
        sampling=ResolvedTaskSampling(
            default_num_neighbors=(32, 16),
            num_neighbors_per_edge_type={edge: (8, 4)},
            replace=True,
            disjoint=False,
            adaptive=True,
            temporal_strategy="last",
            max_target_neighbors_per_entity=12,
        ),
        target_to_entity_paths=[[edge]],
        exclude_cols_dict={"customers": ["internal_id"]},
        unix_start_offset=1,
        unix_end_offset=7,
        unix_step_size=1,
        forecast_length=3,
        lag_timesteps=4,
    )


def test_resolved_task_round_trip_preserves_numerical_contract() -> None:
    task = _task(TaskType.FORECASTING)
    assert ResolvedNumericalTask.from_dict(task.to_dict()) == task
    assert task.get_train_table_stats(after_split=True) is None


@pytest.mark.parametrize(
    ("task_type", "time_column", "output_type", "expected_module"),
    [
        (
            TaskType.STATIC_LINK_PREDICTION,
            "PSEUDO_TIMESTAMP",
            LinkPredOutputType.RANKING,
            LinkPredModuleType.SHALLOW_RHS,
        ),
        (
            TaskType.STATIC_LINK_PREDICTION,
            "PSEUDO_TIMESTAMP",
            LinkPredOutputType.EMBEDDING,
            LinkPredModuleType.TWO_TOWER,
        ),
        (
            TaskType.TEMPORAL_LINK_PREDICTION,
            None,
            LinkPredOutputType.RANKING,
            LinkPredModuleType.HYBRID,
        ),
        (
            TaskType.TEMPORAL_LINK_PREDICTION,
            None,
            LinkPredOutputType.EMBEDDING,
            LinkPredModuleType.SHALLOW_RHS,
        ),
        (
            TaskType.LINK_PREDICTION,
            "TIMESTAMP",
            LinkPredOutputType.RANKING,
            LinkPredModuleType.HYBRID,
        ),
    ],
)
def test_link_pred_module_uses_task_type_not_time_column(
    task_type: TaskType,
    time_column: str | None,
    output_type: LinkPredOutputType,
    expected_module: LinkPredModuleType,
) -> None:
    task = _task(task_type)
    task.train_table = ResolvedTrainTable(
        name=task.train_table.name,
        target_column=task.train_table.target_column,
        time_column=time_column,
        weight_column=task.train_table.weight_column,
    )

    assert infer_link_pred_module_type(task, output_type) == expected_module


@pytest.mark.parametrize(
    ("task_type", "time_column", "is_temporal"),
    [
        (TaskType.STATIC_LINK_PREDICTION, "PSEUDO_TIMESTAMP", False),
        (TaskType.TEMPORAL_LINK_PREDICTION, None, True),
        (TaskType.REGRESSION, "TIMESTAMP", True),
        (TaskType.REGRESSION, None, False),
    ],
)
def test_resolved_task_temporality_preserves_link_prediction_semantics(
    task_type: TaskType,
    time_column: str | None,
    is_temporal: bool,
) -> None:
    task = _task(task_type)
    task.train_table = ResolvedTrainTable(
        name=task.train_table.name,
        target_column=task.train_table.target_column,
        time_column=time_column,
        weight_column=task.train_table.weight_column,
    )

    assert task.is_temporal is is_temporal


@pytest.mark.parametrize(
    ("edge_type", "src_table", "dst_table"),
    [
        (
            ("users", "rev_USER_ID", "events"),
            "events",
            "users",
        ),
        (
            ("events", "USER_ID", "users"),
            "users",
            "events",
        ),
    ],
)
def test_resolved_hop_edge_type_uses_materialized_graph_direction(
    edge_type: tuple[str, str, str],
    src_table: str,
    dst_table: str,
) -> None:
    graph = ResolvedTaskGraph(
        node_types=("events", "users"),
        edge_types=(edge_type, ),
        seed_node_type="users",
        target_node_type="events",
    )
    assert _resolved_hop_edge_type(
        graph,
        src_table,
        "USER_ID",
        dst_table,
        "USER_ID",
    ) == edge_type


def test_resolved_hop_edge_type_handles_authored_only_edge_metadata() -> None:
    """A fact->dim hop resolves when only the authored FK is listed."""
    graph = ResolvedTaskGraph(
        node_types=("impressions", "users"),
        edge_types=(("impressions", "CONSUMER_ID", "users"), ),
        seed_node_type="users",
        target_node_type="impressions",
    )
    assert _resolved_hop_edge_type(
        graph,
        "impressions",
        "CONSUMER_ID",
        "users",
        "CONSUMER_ID",
    ) == ("users", "rev_CONSUMER_ID", "impressions")


def test_resolved_hop_edge_type_keeps_dim_to_fact_hops_unchanged() -> None:
    """A dim->fact hop keeps resolving as before."""
    graph = ResolvedTaskGraph(
        node_types=("clicks", "users"),
        edge_types=(("clicks", "CONSUMER_ID", "users"), ),
        seed_node_type="users",
        target_node_type="clicks",
    )
    assert _resolved_hop_edge_type(
        graph,
        "users",
        "CONSUMER_ID",
        "clicks",
        "CONSUMER_ID",
    ) == ("clicks", "CONSUMER_ID", "users")


def test_resolved_hop_edge_type_names_the_graph_when_it_cannot_resolve(
) -> None:
    """``got []`` on its own located nothing; say what was tried."""
    graph = ResolvedTaskGraph(
        node_types=("events", "users"),
        edge_types=(("events", "OTHER_ID", "users"), ),
        seed_node_type="users",
        target_node_type="events",
    )
    with pytest.raises(ValueError, match="OTHER_ID"):
        _resolved_hop_edge_type(graph, "events", "USER_ID", "users", "USER_ID")


def test_all_training_task_families_have_metrics() -> None:
    for task_type in TaskType.get_node_pred_tasks(
    ) + TaskType.get_link_pred_tasks():
        assert default_metrics(task_type)


def test_trainer_task_type_normalization_remains_mutable() -> None:
    task = _task(TaskType.MULTICLASS_CLASSIFICATION)
    task.task_type = TaskType.BINARY_CLASSIFICATION
    assert task.task_type == TaskType.BINARY_CLASSIFICATION


def test_round_trip_rejects_schema_drift() -> None:
    payload = _task(TaskType.REGRESSION).to_dict()
    payload["legacy_task"] = "not accepted"
    try:
        ResolvedNumericalTask.from_dict(payload)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("schema drift must be rejected")


def test_manifest_constructs_service_free_training_task() -> None:
    manifest = DiskgraphTaskManifest(
        manifest_path="/bundle/manifest.json",
        runtime_manifest_path="/bundle/runtime_manifest.json",
        manifest={},
        metadata={
            "task_type": "static_link_prediction",
            "seed_node_type": "users",
            "target_node_type": "products",
            "train_table_name": "train_table",
            "train_table_columns": {
                "target_column": "TARGET",
                "time_column": "PSEUDO_TIMESTAMP",
                "weight_column": "WEIGHT",
                "foreign_key_columns": ["ENTITY", "TARGET"],
            },
        },
        feature_schema={},
        graph_runtime_metadata={
            "node_types": ["users", "products"],
            "edge_types": [{
                "src": "products",
                "rel": "purchased_by",
                "dst": "users",
            }],
        },
    )
    task = ResolvedNumericalTask.from_manifest(manifest, name="recommend")
    assert task.task_type == TaskType.STATIC_LINK_PREDICTION
    assert task.target_to_entity_paths == [[("products", "purchased_by",
                                             "users")]]
    assert task.metrics == default_metrics(task.task_type)
    assert task.train_table.foreign_key_columns == ("ENTITY", "TARGET")
    assert task.is_temporal is False
    assert infer_link_pred_module_type(
        task, LinkPredOutputType.RANKING) == LinkPredModuleType.SHALLOW_RHS


def test_resolved_encoder_inputs_materialize_original_encoder_configs() -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            ResolvedEncoderInput(
                column_name="age",
                dtype="int64",
                stype="numerical",
                encoder={
                    "name": "Numerical",
                    "scaler": "standard"
                },
                stats={
                    "MEAN": 42.0,
                    "STD": 3.0
                },
            )
        ]
    }
    encoders = collect_resolved_encoders(task)
    assert encoders["customers"]["age"].scaler.value == "standard"
    assert encoders["customers"]["age"]._stats


def test_auto_trainer_applies_encoder_overrides_before_persisting(
    monkeypatch, ) -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            ResolvedEncoderInput(
                column_name="age",
                dtype="int64",
                stype="numerical",
                encoder={
                    "name": "Numerical",
                    "scaler": "standard"
                },
                stats={
                    "MEAN": 42.0,
                    "STD": 3.0
                },
            )
        ]
    }
    persisted = []
    trainer = object.__new__(AutoTrainer)
    trainer.task = task
    trainer.root_dir = "/unused"
    trainer.checkpoint_path = None
    trainer._encoders = None
    trainer.runtime = SimpleNamespace(
        collect_encoders=collect_resolved_encoders,
        trainer_runtime_kwargs=dict)
    trainer.space = ModelPlan()
    trainer.space.column_processing.encoder_overrides = {
        "customers.age": "Null()"
    }
    monkeypatch.setattr(auto_trainer_module, "load_encoders", lambda path: None)
    monkeypatch.setattr(auto_trainer_module, "save_encoders",
                        lambda path, value: persisted.append(value))

    assert isinstance(trainer.encoders["customers"]["age"], Null)
    assert persisted == [trainer.encoders]


def _encoder_input(name: str, stype: str,
                   encoder: dict) -> ResolvedEncoderInput:
    stats = ({
        "MEAN": 3.0,
        "STD": 1.0
    } if stype == "numerical" else {
        "CATEGORY_COUNTS": [["a", 3]]
    })
    return ResolvedEncoderInput(column_name=name, dtype="float64",
                                stype=stype, encoder=encoder, stats=stats)


@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "cached"])
def test_global_na_strategy_applies_and_column_override_wins(
        monkeypatch, cached: bool) -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            _encoder_input(name, "numerical", {"name": "Numerical"})
            for name in ("age", "spend")
        ]
    }
    trainer = object.__new__(AutoTrainer)
    trainer.task = task
    trainer.root_dir = "/unused"
    trainer.checkpoint_path = None
    trainer._encoders = None
    trainer.runtime = SimpleNamespace(
        collect_encoders=collect_resolved_encoders,
        trainer_runtime_kwargs=dict)
    trainer.space = ModelPlan()
    trainer.space.column_processing.na_strategy = {"numerical": "zero"}
    trainer.space.column_processing.encoder_overrides = {
        "customers.spend": "Numerical(na_strategy='mean')"
    }
    stale = collect_resolved_encoders(task) if cached else None
    monkeypatch.setattr(auto_trainer_module, "load_encoders",
                        lambda path: stale)
    monkeypatch.setattr(auto_trainer_module, "save_encoders",
                        lambda path, value: None)

    encoders = trainer.encoders["customers"]
    assert encoders["age"].na_strategy == "zero"
    assert encoders["spend"].na_strategy == "mean"


def test_global_na_strategy_skips_unsupported_encoders_with_warning(
) -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            _encoder_input("city", "categorical", {
                "name": "Hash",
                "num_components": 7
            })
        ]
    }
    with pytest.warns(UserWarning,
                      match="'customers.city' \\(Hash, allowed: separate"):
        encoders = apply_na_strategy(task, collect_resolved_encoders(task),
                                     {"categorical": "zero"})
    assert encoders["customers"]["city"].na_strategy == "separate"


def test_encoder_override_uses_the_materialized_column_stats() -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            ResolvedEncoderInput(
                column_name="age",
                dtype="int64",
                stype="numerical",
                encoder={"name": "Numerical"},
                stats={
                    "MEAN": 42.0,
                    "STD": 3.0
                },
            )
        ]
    }
    encoders = collect_resolved_encoders(task)

    apply_encoder_overrides(
        task,
        encoders,
        {"customers.age": Numerical(scaler="standard")},
    )

    overridden = encoders["customers"]["age"]
    assert overridden.scaler.value == "standard"
    assert {
        stat.value: value
        for stat, value in overridden._stats.items()
    } == {
        "MEAN": 42.0,
        "STD": 3.0,
    }


def test_stale_encoder_override_preserves_legacy_warn_and_skip() -> None:
    task = _task(TaskType.REGRESSION)
    encoders = collect_resolved_encoders(task)

    with pytest.warns(UserWarning, match="not a valid table name"):
        result = apply_encoder_overrides(
            task,
            encoders,
            {"renamed_table.age": Null()},
        )

    assert result is encoders


def test_target_encoder_override_supports_the_train_table_alias() -> None:
    task = _task(TaskType.REGRESSION)
    task.target_encoder_input = ResolvedEncoderInput(
        column_name="TARGET",
        dtype="float64",
        stype="numerical",
        encoder={"name": "Numerical"},
        stats={
            "MEAN": 0.5,
            "STD": 0.25
        },
    )
    encoders = collect_resolved_encoders(task)

    apply_encoder_overrides(
        task,
        encoders,
        {"train_table.TARGET": Null()},
    )

    assert encoders.target is not None
    assert isinstance(encoders.target.encoder, Null)


def test_resolved_encoder_materialization_rejects_missing_stats() -> None:
    task = _task(TaskType.REGRESSION)
    task.encoder_inputs = {
        "customers": [
            ResolvedEncoderInput(
                column_name="age",
                dtype="int64",
                stype="numerical",
                encoder={"name": "Numerical"},
                stats={},
            )
        ]
    }
    with pytest.raises(ValueError, match="MEAN"):
        collect_resolved_encoders(task)


def _link_pred_manifest(target_to_entity_paths):
    """One link-prediction manifest over `users <- orders -> products`."""
    return DiskgraphTaskManifest(
        manifest_path="manifest.json",
        runtime_manifest_path="runtime_manifest.json",
        manifest={},
        metadata={
            "task_type": "temporal_link_prediction",
            "seed_node_type": "users",
            "target_node_type": "products",
            "target_to_entity_paths": target_to_entity_paths,
            "train_table_name": "train_table",
            "train_table_columns": {
                "time_column": "TIMESTAMP",
                "weight_column": None,
                "foreign_key_columns": ["ENTITY", "TARGET"],
            },
        },
        feature_schema={},
        graph_runtime_metadata={
            "node_types": ["users", "orders", "products"],
            "edge_types": [
                {
                    "src": "orders",
                    "rel": "user_id",
                    "dst": "users"
                },
                {
                    "src": "orders",
                    "rel": "product_id",
                    "dst": "products"
                },
            ],
        },
    )


def test_plan_facts_read_the_recorded_target_to_entity_paths():
    # The graph's edge list cannot reproduce these: `target_node_type` is
    # the RHS table, so matching against it looks for a `products -> users`
    # edge and misses the `orders -> users` fact edge the platform names.
    manifest = _link_pred_manifest([
        [["orders", "user_id", "users"]],
        [["orders", "product_id", "products"]],
    ])

    facts = plan_validation_facts(manifest)

    assert facts["target_to_entity_paths"] == [
        [("orders", "user_id", "users")],
        [("orders", "product_id", "products")],
    ]
    # The seed first, the RHS last: metric validation resolves a
    # `diversity[col]@k` column against the last name, and the adaptive
    # sample-size estimate seeds off the first.
    assert facts["entity_table_names"] == ["users", "products"]


def test_plan_facts_fall_back_for_a_bundle_without_recorded_paths():
    manifest = _link_pred_manifest(None)
    manifest.metadata.pop("target_to_entity_paths")

    assert plan_validation_facts(manifest)["target_to_entity_paths"] == []


def test_node_prediction_entity_tables_name_the_seed_alone():
    manifest = _link_pred_manifest([])
    manifest.metadata["target_node_type"] = "users"

    assert plan_validation_facts(manifest)["entity_table_names"] == ["users"]


def _sequence_spec(column_name: str, length: int) -> dict:
    return {
        "column_name": column_name,
        "dtype": "floatlist",
        "stype": "sequence",
        "encoder": {
            "name": "NumericalList"
        },
        "stats": {
            "SEQUENCE_MIN_LENGTH": length,
            "SEQUENCE_MAX_LENGTH": length,
        },
    }


def _forecasting_manifest(column_specs: list[dict]) -> DiskgraphTaskManifest:
    """One forecasting manifest whose train table carries `column_specs`."""
    return DiskgraphTaskManifest(
        manifest_path="manifest.json",
        runtime_manifest_path="runtime_manifest.json",
        manifest={},
        metadata={
            "task_type": "forecasting",
            "seed_node_type": "stores",
            "target_node_type": "stores",
            "train_table_name": "train_table",
            "train_table_columns": {
                "target_column": "TARGET",
                "time_column": "TIMESTAMP",
                "weight_column": None,
                "foreign_key_columns": ["ENTITY"],
            },
        },
        feature_schema={
            "node_types": {
                "train_table": {
                    "feature_columns":
                    [spec["column_name"] for spec in column_specs],
                    "mapper_key":
                    "train_table",
                    "time_column":
                    None,
                    "column_specs":
                    column_specs,
                },
            },
            "target_encoder_input": None,
        },
        graph_runtime_metadata={
            "node_types": ["stores"],
            "edge_types": [],
        },
    )


def _forecasting_task(column_specs: list[dict]) -> ResolvedNumericalTask:
    return ResolvedNumericalTask.from_manifest(
        _forecasting_manifest(column_specs),
        name="sales",
        forecast_length=4,
    )


def test_year_over_year_labels_are_read_off_the_built_train_table():
    task = _forecasting_task([
        _sequence_spec(FUTURE_LABELS_COL, 4),
        _sequence_spec(YOY_COL, 4 + 2 * PADDING_SIZE),
    ])

    assert task.has_year_over_year_labels


def test_no_year_over_year_labels_when_the_train_table_omits_the_column():
    task = _forecasting_task([_sequence_spec(FUTURE_LABELS_COL, 4)])

    assert not task.has_year_over_year_labels


def test_year_over_year_labels_ignore_a_like_named_graph_column():
    # Only the train table is compiled with the label column; a graph table
    # is free to carry a feature of the same name without turning the
    # forecasting input on.
    task = _forecasting_task([_sequence_spec(FUTURE_LABELS_COL, 4)])
    task.encoder_inputs["stores"] = [
        ResolvedEncoderInput(**_sequence_spec(YOY_COL, 24))
    ]

    assert not task.has_year_over_year_labels


def test_year_over_year_encoder_follows_the_task_over_an_unset_plan():
    # An unset request is the plan's own default, and `_forecast_options`
    # reads it as off -- so a plan cannot say what the compiler emitted.
    # Here it emitted the column.
    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.task = _forecasting_task([
        _sequence_spec(FUTURE_LABELS_COL, 4),
        _sequence_spec(YOY_COL, 4 + 2 * PADDING_SIZE),
    ])
    plan = ModelPlan()
    plan.training_table_generation.year_over_year = MissingType.VALUE
    trainer.space = plan
    assert not inferred_or_missing_type_to_false(
        plan.training_table_generation.year_over_year)

    assert isinstance(trainer._get_year_over_year_encoder(),
                      YearOverYearEncoder)


def test_year_over_year_encoder_follows_the_task_over_a_plan_asking_for_it():
    # The mirror case: a plan may request the feature for a table that was
    # built without it, and encoding a column that is not there would fail
    # at the first batch rather than at plan resolution.
    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.task = _forecasting_task([_sequence_spec(FUTURE_LABELS_COL, 4)])
    plan = ModelPlan()
    plan.training_table_generation.year_over_year = True
    trainer.space = plan

    assert trainer._get_year_over_year_encoder() is None
