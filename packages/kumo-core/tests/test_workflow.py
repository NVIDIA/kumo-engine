# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import kumo_core.training.workflow as workflow
import pytest
from kumo_core.model_plan import (
    ModelPlan,
    NeighborSamplingPlan,
    OptimizationPlan,
    TrainingJobPlan,
    TrainingTableGenerationPlan,
    inferred_or_missing_type_to_none,
)
from kumo_core.pquery import (
    NativeGraph,
    NativeGraphColumn,
    NativeGraphForeignKey,
    NativeGraphTable,
    ValidatedNativeQuery,
    to_plan_graph,
)
from kumo_core.resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTaskSampling,
    ResolvedTrainTable,
)
from kumo_core.training.search import AutoMLSearchOptions
from kumo_core.training.workflow import (
    ConnectorPreparation,
    GNNTrainingWorkflow,
    PreparedGraph,
    PublishedModel,
    TrainingWorkflowRequest,
)
from kumo_core.types.enums import TaskType
from kumoapi.model_plan import HopConfig, NumNeighborsConfig, RunMode
from kumoapi.pquery import QueryType
from kumoapi.typing import TimeUnit

USERS = NativeGraphTable(
    name="users",
    source_table_name="main.schema.users",
    columns=(NativeGraphColumn("id", "int64", "ID"), ),
    primary_key="id",
    time_column=None,
    end_time_column=None,
    foreign_keys=(),
)


def request(tmp_path):
    return TrainingWorkflowRequest(
        query=ValidatedNativeQuery(
            query=SimpleNamespace(
                get_exclude_cols_dict=lambda: {}),
            task_type="binary_classification",
            task_family="node_prediction",
            entity_table="users",
            entity_column="id",
            target_stype="categorical",
            query_type="static",
            rhs_node_type=None,
            rhs_is_temporal=False,
            list_distinct_target_is_array=False,
            forecast_length=1,
        ),
        graph=NativeGraph((USERS, )),
        prepared_graph=PreparedGraph("graph/version", str(tmp_path / "index")),
        run_id="run-1",
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
        model_plan=ModelPlan(optimization=OptimizationPlan(max_epochs=1)),
    )


def test_request_from_spec_validates_plain_graph_and_query(
        monkeypatch, tmp_path):
    graph = NativeGraph(())
    query = request(tmp_path).query
    calls = []
    monkeypatch.setattr(
        workflow,
        "normalize_graph",
        lambda value: calls.append(("graph", value)) or graph,
    )
    monkeypatch.setattr(
        workflow,
        "parse_and_validate_query",
        lambda value, resolved: calls.append(
            ("query", value, resolved)) or query,
    )

    result = TrainingWorkflowRequest.from_spec(
        graph={"tables": []},
        predictive_query="PREDICT users.label FOR EACH users.id",
        prepared_graph=PreparedGraph("graph/version", "/tmp/index"),
        run_id="run-2",
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert result.graph is graph
    assert result.query is query
    assert calls == [
        ("graph", {
            "tables": []
        }),
        ("query", "PREDICT users.label FOR EACH users.id", graph),
    ]


def test_request_rejects_a_model_plan_mapping(tmp_path):
    model_plan = ModelPlan()
    value = replace(request(tmp_path), model_plan=model_plan)

    assert value.model_plan is model_plan
    with pytest.raises(TypeError, match="ModelPlan"):
        replace(value, model_plan={"optimization": {"max_epochs": 1}})


def resolved_plan():
    """A plan already through `resolve_model_plan`, as far as metrics go."""
    return ModelPlan(training_job=TrainingJobPlan(metrics=["mae"]))


def resolve(model_plan, search=None, *, task_type=TaskType.REGRESSION,
            query_type=QueryType.TEMPORAL, has_train_table_time_col=True):
    """Resolve one plan against a materialized regression task's facts."""
    return workflow.resolve_model_plan(
        model_plan,
        search,
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        entity_table_names=["articles", "events", "sessions"],
        target_to_entity_paths=[],
        has_train_table_weight_col=False,
    )


def test_resolve_model_plan_accepts_a_supplied_split_for_temporal_link_pred():
    # The other half of the same case: `build_training_task_plan` does not
    # validate, so a supplied split column that the plan validator refuses is
    # refused only here -- after the training table has already been built.
    # A caller who placed every row in a stage has given the temporal-link
    # gate the assurance it asks for, so this must resolve.
    plan = ModelPlan(training_table_generation=TrainingTableGenerationPlan(
        split="ManualSplit(col='SPLIT', train_value=0, val_value=1, "
        "test_value=2)"))

    resolved = resolve(plan, task_type=TaskType.TEMPORAL_LINK_PREDICTION,
                       query_type=QueryType.TEMPORAL,
                       has_train_table_time_col=True)

    assert resolved.training_table_generation.split == (
        "ManualSplit(col='SPLIT', train_value=0, val_value=1, test_value=2)")


def test_resolve_model_plan_rejects_a_model_plan_mapping():
    with pytest.raises(TypeError, match="ModelPlan"):
        resolve({"optimization": {"max_epochs": 1}})


def test_resolve_model_plan_fills_defaults_around_the_caller_options():
    caller = ModelPlan(training_job=TrainingJobPlan(metrics=["mae", "mse"]))
    before = asdict(caller)

    result = resolve(caller)

    assert result is not caller
    assert asdict(caller) == before  # The caller's plan stays untouched.
    assert result.training_job.metrics == ["mae", "mse"]
    # The shared task default remains the tuning objective.
    assert result.training_job.tune_metric == "mae"
    assert result.training_job.num_experiments == 8  # Filled from the default.
    assert result.optimization == ModelPlan.from_default(
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.NORMAL,
        entity_table_names=["articles", "events", "sessions"],
        target_to_entity_paths=[],
    ).optimization


def test_resolve_model_plan_does_not_replace_the_default_tune_metric():
    result = resolve(
        ModelPlan(training_job=TrainingJobPlan(metrics=["mse", "mae"])))

    assert result.training_job.metrics == ["mse", "mae"]
    assert result.training_job.tune_metric == "mae"


def test_resolve_model_plan_rejects_metrics_that_drop_the_default_objective():
    with pytest.raises(ValueError, match="tune metric"):
        resolve(
            ModelPlan(training_job=TrainingJobPlan(metrics=["mse", "rmse"])))


def test_resolve_model_plan_keeps_an_explicit_tune_metric():
    # `loss` is a valid tuning objective that is never in `metrics`; nothing
    # may quietly retune the caller onto something else.
    result = resolve(
        ModelPlan(training_job=TrainingJobPlan(
            metrics=["mae", "mse"],
            tune_metric="loss",
        )))

    assert result.training_job.tune_metric == "loss"


def test_resolve_model_plan_rejects_a_tune_metric_it_cannot_compute():
    with pytest.raises(ValueError, match="tune metric"):
        resolve(
            ModelPlan(training_job=TrainingJobPlan(
                metrics=["mae"],
                tune_metric="auroc",
            )))


def test_resolve_model_plan_reads_a_static_query_over_timestamps_as_static():
    # Fraud on transactions.created_at, churn on loan applications: the query
    # is static, but every row carries an event time, so the train table has
    # a time column. Deriving the query type from that column reads the run
    # as temporal and resolves temporal-only defaults for it.
    result = resolve(ModelPlan(), task_type=TaskType.BINARY_CLASSIFICATION,
                     query_type=QueryType.STATIC, has_train_table_time_col=True)

    # Resolved for a static run; left unset when the run is read as temporal.
    assert result.neighbor_sampling.sample_from_entity_table is True
    # Temporal-only, and not applicable to a static run.
    assert inferred_or_missing_type_to_none(
        result.model_architecture.prediction_time_encodings) is None


def test_resolve_model_plan_leaves_an_inferred_split_inferred():
    # `inferred` means "resolve this from the data later", which is what
    # `_split_options` does from the query and graph. Filling it from the
    # task-type default here would record a split the train table never used.
    result = resolve(ModelPlan())

    assert result.training_table_generation.split == "inferred"


def test_resolve_model_plan_keeps_a_temporal_split_on_the_plan():
    caller = ModelPlan(training_table_generation=TrainingTableGenerationPlan(
        split="DateOffsetSplit(offsets=[-1, 0], unit='days')"))

    result = resolve(caller)

    assert result.training_table_generation.split == (
        "DateOffsetSplit(offsets=[-1, 0], unit='days')")


def test_resolve_model_plan_preserves_per_edge_neighbor_override():
    result = resolve(
        ModelPlan(neighbor_sampling=NeighborSamplingPlan(num_neighbors=[
            NumNeighborsConfig(
                hop1=HopConfig(
                    default=16,
                    **{"sessions.session->events.session": 64},
                ),
                hop2=HopConfig(default=16),
            )
        ])))

    neighbors = result.neighbor_sampling.num_neighbors[0]
    assert neighbors.hop1.default == 16
    assert neighbors.hop1.__pydantic_extra__ == {
        "sessions.session->events.session": 64,
    }
    assert neighbors.hop2.default == 16


def test_task_options_project_the_caller_neighbor_plan(tmp_path):
    value = replace(
        request(tmp_path),
        model_plan=ModelPlan(neighbor_sampling=NeighborSamplingPlan(
            num_neighbors=[
                NumNeighborsConfig(
                    hop1=HopConfig(default=8),
                    hop2=HopConfig(default=4),
                )
            ],
            max_target_neighbors_per_entity=[64],
            adaptive_sampling=True,
        )),
    )

    sampling = workflow._resolved_task_options(value,
                                               resolved_plan())["sampling"]

    assert sampling.default_num_neighbors == (8, 4)
    assert sampling.max_target_neighbors_per_entity == 64
    assert sampling.adaptive is True
    assert sampling.temporal_strategy == "last"


def test_task_options_of_an_unset_neighbor_plan_keep_the_task_defaults(
        tmp_path):
    value = replace(request(tmp_path), model_plan=ModelPlan())

    options = workflow._resolved_task_options(value, resolved_plan())

    assert options["sampling"] == ResolvedTaskSampling()


@pytest.mark.parametrize(
    "plan",
    [
        ModelPlan(),  # `metrics` still the unset sentinel.
        ModelPlan(training_job=TrainingJobPlan(
            metrics=[],
            tune_metric="loss",
        )),
    ])
def test_task_options_reject_a_plan_that_names_no_metrics(plan, tmp_path):
    # `MissingType` is a `str` enum, so an unresolved plan would otherwise
    # yield `['?', '?', '?']`; an empty list is falsy, so `from_manifest`
    # would quietly swap in `default_metrics` and put the divergence back.
    with pytest.raises(ValueError, match="non-empty metric list"):
        workflow._resolved_task_options(request(tmp_path), plan)


def test_forecasting_task_options_derive_temporal_training_metadata(tmp_path):
    value = replace(
        request(tmp_path),
        query=replace(
            request(tmp_path).query,
            query=SimpleNamespace(
                get_combined_date_offset_range=lambda: SimpleNamespace(
                    start=0,
                    end=1,
                    unit=TimeUnit.DAYS,
                ),
                get_exclude_cols_dict=lambda: {},
            ),
            task_type="forecasting",
            task_family="forecasting",
            query_type="temporal",
            forecast_length=2,
        ),
        model_plan=ModelPlan(
            training_table_generation=TrainingTableGenerationPlan(
                timeframe_step=1,
                lag_timesteps=2,
            )),
    )

    options = workflow._resolved_task_options(
        value, ModelPlan(training_job=TrainingJobPlan(metrics=["mae"])))

    assert options == {
        "name": "run-1",
        "metrics": ["mae"],
        "forecast_length": 2,
        "lag_timesteps": 2,
        "unix_start_offset": 0,
        "unix_end_offset": 86_400_000_000_000,
        "unix_step_size": 86_400_000_000_000,
        "sampling": ResolvedTaskSampling(),
        "exclude_cols_dict": {},
    }


def test_prepare_training_request_uses_core_connector_contract(
        monkeypatch, tmp_path):
    calls = []
    value = request(tmp_path)

    class Connector:
        def prepare(self, **kwargs):
            calls.append(kwargs)
            return ConnectorPreparation(
                prepared_graph=value.prepared_graph,
                scratch_dir=value.scratch_dir,
                artifact_dir=value.artifact_dir,
            )

    monkeypatch.setattr(workflow, "normalize_graph", lambda _value: value.graph)
    monkeypatch.setattr(
        workflow,
        "parse_and_validate_query",
        lambda _query, _graph: value.query,
    )
    result = workflow.prepare_training_request(
        connector=Connector(),
        graph={"tables": []},
        predictive_query="query",
        run_id="run-1",
        model_plan=value.model_plan,
    )

    assert result.prepared_graph is value.prepared_graph
    assert calls == [{
        "graph": {
            "tables": []
        },
        "predictive_query": "query",
        "model_plan": value.model_plan,
        "custom_training_table": None,
        "run_id": "run-1",
    }]


def test_materialize_training_task_composes_the_activity_chain(
        monkeypatch, tmp_path):
    calls = []
    plan = SimpleNamespace(
        train_table=SimpleNamespace(tt_id="tt-1"),
        task_dataset="task-dataset",
    )
    generated = SimpleNamespace(staging_dir="staging")
    materialized = SimpleNamespace(bundle_dir="bundle")
    runtime = object()
    monkeypatch.setattr(
        workflow,
        "build_training_task_plan",
        lambda value: calls.append(("plan", value)) or plan,
    )
    monkeypatch.setattr(
        workflow,
        "generate_train_table",
        lambda value, **kwargs: calls.append(
            ("generate", value, kwargs)) or generated,
    )
    monkeypatch.setattr(
        workflow,
        "materialize_generated_training_task",
        lambda *args, **kwargs: calls.append(
            ("materialize", args, kwargs)) or materialized,
    )
    value = request(tmp_path)

    result = workflow.materialize_training_task(value, runtime=runtime)

    assert result is materialized
    assert calls == [
        ("plan", value),
        ("generate", plan.train_table, {
            "runtime": runtime
        }),
        ("materialize", (value, "task-dataset", generated), {
            "tt_id": "tt-1",
            "runtime": runtime
        }),
    ]


def test_materialized_task_takes_its_metric_list_from_the_resolved_plan(
        monkeypatch, tmp_path):
    # Built from the manifest alone, the task would carry `default_metrics`,
    # which knows the task type and nothing else. Materialization resolves
    # the caller's plan against the same manifest and builds the task with
    # the list that plan named, so the task computes what training tunes on.
    loaded = []
    manifest = SimpleNamespace(
        task_type="binary_classification",
        train_table_name="train_table",
        train_table_columns={
            "target_column": "TARGET",
            "foreign_key_columns": ("users_id", ),
        },
        graph_runtime_metadata={
            "node_types": ("users", ),
            "edge_types": (),
        },
        feature_schema={},
        load_column_stats=lambda: loaded.append(True) or {},
        seed_node_type="users",
        target_node_type="users",
        target_to_entity_paths=None,
        stage_counts={"train": 8},
        stage_seed_times={},
    )
    value = replace(
        request(tmp_path),
        model_plan=ModelPlan(
            training_job=TrainingJobPlan(metrics=["ap"], tune_metric="ap")),
    )
    caller_plan = asdict(value.model_plan)

    @dataclass(frozen=True)
    class _FakeTaskDatasetSpec:
        tt_manifest_path: str
        task_type: str = "binary_classification"
        task_family: str = "node_prediction"

    task_dataset = _FakeTaskDatasetSpec("original-manifest")
    monkeypatch.setattr(
        workflow,
        "ingest_train_table",
        lambda **kwargs: SimpleNamespace(extension_path="extension",
                                         manifest_path="manifest"),
    )
    monkeypatch.setattr(
        workflow,
        "materialize_task_dataset",
        lambda **kwargs: SimpleNamespace(
            manifest_path="manifest",
            runtime_manifest_path="runtime-manifest",
            bundle_dir="bundle",
            storage_kind="local",
        ),
    )
    monkeypatch.setattr(workflow, "DiskgraphTaskManifest",
                        SimpleNamespace(load=lambda path: manifest))

    result = workflow.materialize_generated_training_task(
        value,
        task_dataset,
        SimpleNamespace(staging_dir="staging"),
        tt_id="tt-1",
    )
    assert loaded == [True]

    assert result.model_plan.training_job.metrics == ["ap"]
    assert result.task.metrics == ["ap"]
    # Whatever the plan tunes on is a metric the task actually computes.
    assert result.model_plan.training_job.tune_metric in result.task.metrics
    # The caller's own plan is left alone.
    assert asdict(value.model_plan) == caller_plan


def test_materialize_generated_training_task_with_a_tt_id_override_reads_the_new_extension(
        monkeypatch, tmp_path):
    r"""The task-dataset manifest placeholder must always be replaced by the
    manifest produced by this call's selected-index ingest."""
    manifest = SimpleNamespace(
        task_type="binary_classification",
        train_table_name="train_table",
        train_table_columns={
            "target_column": "TARGET",
            "foreign_key_columns": ("users_id", ),
        },
        graph_runtime_metadata={
            "node_types": ("users", ),
            "edge_types": (),
        },
        feature_schema={},
        load_column_stats=lambda: {},
        seed_node_type="users",
        target_node_type="users",
        target_to_entity_paths=None,
        stage_counts={"train": 8},
        stage_seed_times={},
    )
    value = request(tmp_path)

    @dataclass(frozen=True)
    class _FakeTaskDatasetSpec:
        tt_manifest_path: str
        task_type: str = "binary_classification"
        task_family: str = "node_prediction"

    original_spec = _FakeTaskDatasetSpec(
        tt_manifest_path=str(tmp_path / "index/train_tables/original-tt/"
                             "manifest.json"))
    monkeypatch.setattr(
        workflow,
        "ingest_train_table",
        lambda **kwargs: SimpleNamespace(
            extension_path=str(tmp_path / "index/train_tables/fresh-tt"),
            manifest_path=str(tmp_path /
                              "index/train_tables/fresh-tt/manifest.json"),
        ),
    )
    materialize_calls = []
    monkeypatch.setattr(
        workflow,
        "materialize_task_dataset",
        lambda **kwargs: materialize_calls.append(kwargs) or SimpleNamespace(
            manifest_path="manifest",
            runtime_manifest_path="runtime-manifest",
            bundle_dir="bundle",
            storage_kind="local",
        ),
    )
    monkeypatch.setattr(workflow, "DiskgraphTaskManifest",
                        SimpleNamespace(load=lambda path: manifest))

    result = workflow.materialize_generated_training_task(
        value,
        original_spec,
        SimpleNamespace(staging_dir="staging"),
        tt_id="fresh-tt",
    )

    assert len(materialize_calls) == 1
    consumed_spec = materialize_calls[0]["spec"]
    assert consumed_spec.tt_manifest_path == str(
        tmp_path / "index/train_tables/fresh-tt/manifest.json")
    assert consumed_spec.tt_manifest_path != original_spec.tt_manifest_path
    assert result.train_table_extension_path == str(
        tmp_path / "index/train_tables/fresh-tt")


def test_train_prepared_graph_uses_one_materialized_training_input(
        monkeypatch, tmp_path):
    calls = []
    bindings = object()
    training_result = object()
    value = request(tmp_path)
    materialized = SimpleNamespace(
        task=ResolvedNumericalTask(
            name="run-1",
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=["auroc"],
            graph=ResolvedTaskGraph(
                node_types=("users", ),
                edge_types=(),
                seed_node_type="users",
                target_node_type="users",
            ),
            train_table=ResolvedTrainTable(
                name="train_table",
                target_column="TARGET",
                time_column=None,
                weight_column=None,
            ),
        ),
        bundle_dir="bundle",
        model_plan=ModelPlan(training_job=TrainingJobPlan(metrics=["auroc"])),
    )
    monkeypatch.setattr(
        workflow,
        "materialize_training_task",
        lambda *args, **kwargs: calls.append(
            ("materialize", args, kwargs)) or materialized,
    )
    monkeypatch.setattr(
        workflow,
        "run_callable_training",
        lambda *args, **kwargs: calls.append(
            ("train", args, kwargs)) or training_result,
    )

    result = workflow.train_prepared_graph(
        value,
        home_dir=str(tmp_path / "home"),
        bindings_factory=lambda item: bindings,
    )

    assert result is training_result
    assert calls[0][0] == "materialize"
    assert calls[1][0] == "train"
    assert calls[1][2]["bindings"] is bindings
    training_input = calls[1][1][0]
    assert training_input.task is materialized.task
    assert training_input.home_dir == str(tmp_path / "home")
    assert training_input.task_dataset_bundle_dir == "bundle"
    # The plan reaching training is the one materialization resolved, not
    # the caller's and not a second resolution of it.
    assert training_input.model_plan is materialized.model_plan


def test_configured_workflow_has_two_argument_training_surface(
        monkeypatch, tmp_path):
    value = request(tmp_path)
    training = SimpleNamespace(
        metrics={"val_acc": 0.82},
        artifact_paths={"checkpoint": "/model"},
    )
    calls = []
    monkeypatch.setattr(
        workflow,
        "prepare_training_request",
        lambda **kwargs: calls.append(("prepare", kwargs)) or value,
    )
    monkeypatch.setattr(
        workflow,
        "train_prepared_graph",
        lambda *args, **kwargs: calls.append(
            ("train", args, kwargs)) or training,
    )
    monkeypatch.setattr(
        workflow.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="run-1"),
    )

    class Publisher:
        def publish(self, name, result):
            calls.append(("publish", name, result))
            return PublishedModel(name, "V_1", "models:/churn/V_1")

    configured = GNNTrainingWorkflow(
        connector=object(),
        graph={"tables": []},
        bindings_factory=lambda _input: object(),
        publisher=Publisher(),
        model_plan=value.model_plan,
    )
    result = configured.train_gnn(
        "PREDICT churn FOR EACH user",
        "KUMO_GNN_CHURN",
    )

    assert result.run_id == "run-1"
    assert result.model_name == "KUMO_GNN_CHURN"
    assert result.version == "V_1"
    assert result.val_metric == 0.82
    assert [call[0] for call in calls] == ["prepare", "train", "publish"]
    assert calls[0][1]["run_id"] == "run-1"
    assert calls[1][2]["home_dir"] == value.scratch_dir


def test_search_options_control_original_model_plan() -> None:
    result = resolve(
        ModelPlan(),
        AutoMLSearchOptions(run_mode="best", num_trials=13, seed=42),
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
    )

    assert result.training_job.num_experiments == 13
    assert result.training_job.manual_seed == 42
    # `run_mode` selects the defaults every unset option is filled from.
    assert result.model_architecture == ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.BEST,
        entity_table_names=["articles", "events", "sessions"],
        target_to_entity_paths=[],
    ).model_architecture


def test_search_options_reject_invalid_trials_and_seed() -> None:
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            AutoMLSearchOptions(num_trials=value)
    with pytest.raises(TypeError):
        AutoMLSearchOptions(seed=True)


def link_pred_graph():
    """Two entity tables and the link fact whose primary fanout is inferred."""
    users = NativeGraphTable(
        name="users",
        source_table_name="main.schema.users",
        columns=(NativeGraphColumn("user_id", "int64", "ID"),
                 NativeGraphColumn("age", "int64", "numerical")),
        primary_key="user_id",
        time_column=None,
        end_time_column=None,
        foreign_keys=(),
    )
    products = NativeGraphTable(
        name="products",
        source_table_name="main.schema.products",
        columns=(NativeGraphColumn("product_id", "int64", "ID"), ),
        primary_key="product_id",
        time_column=None,
        end_time_column=None,
        foreign_keys=(),
    )
    orders = NativeGraphTable(
        name="orders",
        source_table_name="main.schema.orders",
        columns=(
            NativeGraphColumn("user_id", "int64", "ID"),
            NativeGraphColumn("product_id", "int64", "ID"),
            NativeGraphColumn("ts", "date", "timestamp"),
        ),
        primary_key=None,
        time_column="ts",
        end_time_column=None,
        foreign_keys=(
            NativeGraphForeignKey("user_id", "users", "user_id"),
            NativeGraphForeignKey("product_id", "products", "product_id"),
        ),
    )
    return NativeGraph((users, products, orders))


LINK_PRED_FACTS = dict(
    task_type=TaskType.TEMPORAL_LINK_PREDICTION,
    query_type=QueryType.TEMPORAL,
    has_train_table_time_col=True,
    entity_table_names=["users"],
    target_to_entity_paths=[[("orders", "user_id", "users")]],
    has_train_table_weight_col=False,
)

TARGET_EDGE = "users.user_id->orders.user_id"
RHS_EDGE = "products.product_id->orders.product_id"
TWO_TARGET_PATHS = [
    [("orders", "user_id", "users")],
    [("orders", "product_id", "products")],
]


def resolve_link_pred(model_plan=None, **kwargs):
    return workflow.resolve_model_plan(
        model_plan if model_plan is not None else ModelPlan(),
        None,
        **LINK_PRED_FACTS,
        **kwargs,
    )


def test_resolve_model_plan_without_a_graph_never_marks_the_target_edge():
    # The state this fix moved away from: no graph, so `from_default` never
    # creates the per-target-edge entry and link prediction samples the flat
    # run-mode default on the edge the platform treats specially.
    result = resolve_link_pred()

    assert result.neighbor_sampling.num_neighbors[0].hop1.__pydantic_extra__ \
        == {}


def test_resolve_model_plan_infers_the_target_edge_fanout_from_the_graph():
    result = resolve_link_pred(
        graph=to_plan_graph(link_pred_graph()),
        inferred_max_target_neighbors=96,
    )

    hop1 = result.neighbor_sampling.num_neighbors[0].hop1
    # `AutoTrainer` resolves the sentinel from the task's own estimate.
    assert inferred_or_missing_type_to_none(
        hop1.__pydantic_extra__[TARGET_EDGE]) is None
    assert hop1.default == 16


def test_resolve_model_plan_defaults_only_the_primary_link_target_path():
    facts = {
        **LINK_PRED_FACTS,
        "target_to_entity_paths": TWO_TARGET_PATHS,
    }

    result = workflow.resolve_model_plan(
        ModelPlan(),
        None,
        graph=to_plan_graph(link_pred_graph()),
        inferred_max_target_neighbors=96,
        **facts,
    )

    hop1 = result.neighbor_sampling.num_neighbors[0].hop1
    assert set(hop1.__pydantic_extra__) == {TARGET_EDGE}


def test_resolve_model_plan_preserves_explicit_rhs_edge_fanout():
    facts = {
        **LINK_PRED_FACTS,
        "target_to_entity_paths": TWO_TARGET_PATHS,
    }
    caller = ModelPlan(
        **{
            "neighbor_sampling": {
                "num_neighbors": [{
                    "hop1": {
                        "default": 16,
                        RHS_EDGE: 32,
                    },
                    "hop2": {
                        "default": 16,
                    },
                }],
            },
        })

    result = workflow.resolve_model_plan(
        caller,
        None,
        graph=to_plan_graph(link_pred_graph()),
        **facts,
    )

    assert result.neighbor_sampling.num_neighbors[0].hop1.__pydantic_extra__ \
        == {RHS_EDGE: 32}


def test_resolve_model_plan_falls_back_to_the_hop_default_without_an_estimate():
    # A sentinel nothing can resolve would trip `AutoTrainer`'s assertion, so
    # the entry is dropped and the hop's own default stays in force.
    result = resolve_link_pred(graph=to_plan_graph(link_pred_graph()))

    hop1 = result.neighbor_sampling.num_neighbors[0].hop1
    assert hop1.__pydantic_extra__ == {}
    assert hop1.default == 16


def test_resolve_model_plan_keeps_an_explicit_target_edge_fanout():
    result = resolve_link_pred(
        ModelPlan(
            **{
                "neighbor_sampling": {
                    "num_neighbors": [{
                        "hop1": {
                            "default": 16,
                            TARGET_EDGE: 32,
                        },
                        "hop2": {
                            "default": 16
                        },
                    }],
                },
            }),
        graph=to_plan_graph(link_pred_graph()),
    )

    assert result.neighbor_sampling.num_neighbors[0].hop1.__pydantic_extra__ \
        == {TARGET_EDGE: 32}


def test_resolve_model_plan_resolves_a_legacy_inferred_target_list():
    result = resolve_link_pred(
        ModelPlan(
            **{
                "neighbor_sampling": {
                    "num_neighbors": [{
                        "hop1": {
                            "default": 12
                        },
                        "hop2": {
                            "default": 12
                        },
                    }],
                    "max_target_neighbors_per_entity": ["inferred"],
                },
            }),
        graph=to_plan_graph(link_pred_graph()),
    )

    assert result.neighbor_sampling.max_target_neighbors_per_entity == [12]


def test_infer_max_target_neighbors_reads_the_ingested_row_counts(monkeypatch):
    # The degree policy itself is `target_fanout`'s and is tested there; what
    # this wrapper owns is feeding it the DiskGraph row counts.
    monkeypatch.setattr(
        workflow,
        "table_row_count",
        lambda _index, table: {
            "orders": 40_000,
            "users": 100
        }.get(table),
    )

    # 40000 / 100 -> degree 400 -> ceil(sqrt(400) * 8) == 160.
    assert workflow.infer_max_target_neighbors_per_entity(
        "/index", [[("orders", "user_id", "users")]]) == 160


@pytest.mark.parametrize("rows, paths", [
    ({
        "orders": 400,
        "users": 100
    }, []),
    ({
        "orders": 400
    }, [[("orders", "user_id", "users")]]),
])
def test_infer_max_target_neighbors_reports_no_estimate(monkeypatch, rows,
                                                        paths):
    monkeypatch.setattr(workflow, "table_row_count",
                        lambda _index, table: rows.get(table))

    assert workflow.infer_max_target_neighbors_per_entity("/index",
                                                          paths) is None


def test_task_options_carry_the_inferred_target_fanout(tmp_path):
    value = replace(
        request(tmp_path),
        model_plan=ModelPlan(**{
            "neighbor_sampling": {
                "max_target_neighbors_per_entity": [64],
            },
        }),
    )

    # The estimate is what `AutoTrainer` writes into the `inferred` entries,
    # so it wins over the number the caller typed -- which reaches sampling
    # through the plan, untouched.
    sampling = workflow._resolved_task_options(value, resolved_plan(),
                                               160)["sampling"]

    assert sampling.max_target_neighbors_per_entity == 160


def test_resolve_model_plan_survives_a_hopless_neighbor_config():
    # `_default_num_neighbors` returns () for a config with no hops, which
    # used to be indexed straight off the end.
    result = resolve_link_pred(
        ModelPlan(
            **{
                "neighbor_sampling": {
                    "num_neighbors": [{}],
                    "max_target_neighbors_per_entity": ["inferred"],
                },
            }),
        graph=to_plan_graph(link_pred_graph()),
    )

    assert result.neighbor_sampling.max_target_neighbors_per_entity == [10]


def test_effective_target_neighbors_prefers_the_data_estimate(tmp_path):
    value = replace(
        request(tmp_path),
        model_plan=ModelPlan(**{
            "neighbor_sampling": {
                "max_target_neighbors_per_entity": [128],
            },
        }),
    )

    assert workflow.effective_max_target_neighbors(value, 160) == 160
    # With no estimate the caller's own number is what `AutoTrainer` would
    # fill from, so the per-edge entry must survive rather than be dropped.
    assert workflow.effective_max_target_neighbors(value, None) == 128
    assert workflow.effective_max_target_neighbors(request(tmp_path),
                                                   None) is None


def test_the_target_edge_survives_on_a_caller_supplied_fanout(tmp_path):
    # The estimate is unavailable, but the caller named a number: dropping
    # the entry here would silently sample the hop default instead of it.
    caller = ModelPlan(
        **{"neighbor_sampling": {
            "max_target_neighbors_per_entity": [128]
        }})
    value = replace(request(tmp_path), model_plan=caller)
    effective = workflow.effective_max_target_neighbors(value, None)

    result = resolve_link_pred(caller, graph=to_plan_graph(link_pred_graph()),
                               inferred_max_target_neighbors=effective)

    assert TARGET_EDGE in \
        result.neighbor_sampling.num_neighbors[0].hop1.__pydantic_extra__


def test_static_link_fit_accepts_the_link_split_over_a_physical_timestamp(
        monkeypatch, tmp_path):
    manifest = SimpleNamespace(
        task_type="static_link_prediction",
        train_table_name="train_table",
        train_table_columns={
            "target_column": "TARGET",
            "time_column": "TIMESTAMP",
            "weight_column": None,
            "foreign_key_columns": ("ENTITY", "TARGET"),
        },
        graph_runtime_metadata={
            "node_types": ("users", "products", "orders"),
            "edge_types": (),
        },
        feature_schema={},
        load_column_stats=lambda: {},
        seed_node_type="users",
        target_node_type="products",
        target_to_entity_paths=TWO_TARGET_PATHS,
        stage_counts={"train": 8},
        stage_seed_times={},
    )
    value = replace(
        request(tmp_path),
        query=replace(
            request(tmp_path).query,
            task_type="static_link_prediction",
            task_family="link_prediction",
            entity_column="user_id",
            rhs_node_type="products",
            list_distinct_target_is_array=True,
        ),
        graph=link_pred_graph(),
        model_plan=ModelPlan(training_table_generation=TrainingTableGenerationPlan(
            split="RandomLinkSplit([0.8, 0.1, 0.1])")),
    )

    @dataclass(frozen=True)
    class _FakeTaskDatasetSpec:
        tt_manifest_path: str
        task_type: str = "static_link_prediction"
        task_family: str = "link_prediction"

    monkeypatch.setattr(
        workflow,
        "ingest_train_table",
        lambda **kwargs: SimpleNamespace(extension_path="extension",
                                         manifest_path="manifest"),
    )
    monkeypatch.setattr(
        workflow,
        "materialize_task_dataset",
        lambda **kwargs: SimpleNamespace(
            manifest_path="manifest",
            runtime_manifest_path="runtime-manifest",
            bundle_dir="bundle",
            storage_kind="local",
        ),
    )
    monkeypatch.setattr(workflow, "DiskgraphTaskManifest",
                        SimpleNamespace(load=lambda path: manifest))
    monkeypatch.setattr(workflow, "infer_max_target_neighbors_per_entity",
                        lambda *args: None)

    result = workflow.materialize_generated_training_task(
        value,
        _FakeTaskDatasetSpec("original-manifest"),
        SimpleNamespace(staging_dir="staging"),
        tt_id="tt-1",
    )

    assert result.model_plan.training_table_generation.split == (
        "RandomLinkSplit([0.8, 0.1, 0.1])")
