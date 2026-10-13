# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streamlined training workflow over the parity-preserving Kumo modules.

The lower-level modules retain the original Kumo numerical implementation.
This module composes their service-free boundaries for runtimes that already
have a prepared DiskGraph snapshot. Platform packages remain responsible for
source-table staging, graph ingest, durable storage, and model registration.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import (
    Any,
    Callable,
    ClassVar,
    Mapping,
    Protocol,
    runtime_checkable,
)

from kumoapi.pquery import QueryType

from ..automl.target_fanout import (
    TargetToEntityPaths,
    infer_target_edge_fanout,
)
from ..io.json import from_json_dict, to_json_dict
from ..model_plan import (
    ModelPlan,
    NeighborSamplingPlan,
    NumNeighborsConfig,
    TrainingTableGenerationPlan,
    _is_inferred,
    inferred_or_missing_type_to_none,
)
from ..pquery import (
    NativeGraph,
    NativePqueryTaskPlan,
    PlanGraph,
    ValidatedNativeQuery,
    build_native_pquery_task_plan,
    normalize_graph,
    parse_and_validate_query,
    query_is_temporal,
    resolve_query_forecast,
    table_row_count,
    to_plan_graph,
)
from ..resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskSampling,
    plan_validation_facts,
)
from ..task_dataset import (
    DiskgraphTaskManifest,
    TaskDatasetMaterializeSpec,
    materialize_task_dataset,
)
from ..types import TaskType
from ..utils.time import compute_unix_step
from . import search as search_options
from .orchestration import (
    CallableTrainingOptions,
    CallableTrainingResult,
    ResolvedTrainingInput,
    TrainingBindings,
    run_callable_training,
)
from .table import (
    CustomTrainingTable,
    GeneratedTrainTableResult,
    PreparedCustomTrainingTable,
    generate_custom_train_table,
    generate_train_table,
    ingest_train_table,
)


@dataclass(frozen=True)
class PreparedGraph:
    """Identity and local path of one fully ingested DiskGraph snapshot."""

    index_id: str
    index_path: str

    def __post_init__(self) -> None:
        if not self.index_id:
            raise ValueError("prepared graph requires index_id")
        if not self.index_path:
            raise ValueError("prepared graph requires index_path")


@dataclass(frozen=True)
class ConnectorPreparation:
    """Graph snapshot and run paths produced by a warehouse connector."""

    prepared_graph: PreparedGraph
    scratch_dir: str
    artifact_dir: str
    custom_training_table: PreparedCustomTrainingTable | None = None


@runtime_checkable
class TrainingConnector(Protocol):
    """Warehouse seam required by the streamlined training workflow."""
    def prepare(
        self,
        *,
        graph: Mapping[str, Any],
        predictive_query: str,
        model_plan: ModelPlan,
        custom_training_table: CustomTrainingTable | None,
        run_id: str,
    ) -> ConnectorPreparation:
        """Stage source tables and return one prepared DiskGraph snapshot."""


@dataclass(frozen=True)
class PublishedModel:
    """Environment-native model identity returned after publication."""

    name: str
    version: str
    uri: str


@runtime_checkable
class TrainingPublisher(Protocol):
    """Publish a trained vital bundle to an environment-native registry."""
    def publish(
        self,
        model_name: str,
        training: CallableTrainingResult,
    ) -> PublishedModel:
        """Register one trained model and return its stable identity."""


def _require_separate_generation_plan(model_plan: ModelPlan) -> None:
    if model_plan.training_table_generation != TrainingTableGenerationPlan():
        raise ValueError(
            "model_plan.training_table_generation must use its defaults; "
            "pass training_table_generation_plan instead")


@dataclass(frozen=True)
class TrainingWorkflowRequest:
    """Portable inputs for materialization and original Kumo AutoML."""

    query: ValidatedNativeQuery
    graph: NativeGraph
    prepared_graph: PreparedGraph
    run_id: str
    scratch_dir: str
    artifact_dir: str
    model_plan: ModelPlan = field(default_factory=ModelPlan)
    #: Generation settings, or the effective settings recorded by existing rows.
    training_table_generation_plan: TrainingTableGenerationPlan = field(
        default_factory=TrainingTableGenerationPlan)
    search: search_options.AutoMLSearchOptions = field(
        default_factory=search_options.AutoMLSearchOptions)
    custom_training_table: PreparedCustomTrainingTable | None = None
    #: Explicit train-table anchor timestamps. Empty derives them from the
    #: DiskGraph column statistics of the query's own time columns.
    timestamps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, ValidatedNativeQuery):
            raise TypeError("query must be ValidatedNativeQuery")
        if not isinstance(self.graph, NativeGraph):
            raise TypeError("graph must be NativeGraph")
        if not isinstance(self.prepared_graph, PreparedGraph):
            raise TypeError("prepared_graph must be PreparedGraph")
        for label, value in (
            ("run_id", self.run_id),
            ("scratch_dir", self.scratch_dir),
            ("artifact_dir", self.artifact_dir),
        ):
            if not value:
                raise ValueError(f"{label} must be non-empty")
        if not isinstance(self.model_plan, ModelPlan):
            raise TypeError("model_plan must be a ModelPlan")
        _require_separate_generation_plan(self.model_plan)
        if not isinstance(self.training_table_generation_plan,
                          TrainingTableGenerationPlan):
            raise TypeError("training_table_generation_plan must be a "
                            "TrainingTableGenerationPlan")
        object.__setattr__(
            self, "query",
            resolve_query_forecast(self.query, self.graph,
                                   self.training_table_generation_plan))
        if not isinstance(self.search, search_options.AutoMLSearchOptions):
            raise TypeError("search must be AutoMLSearchOptions")
        if (self.custom_training_table is not None and not isinstance(
                self.custom_training_table, PreparedCustomTrainingTable)):
            raise TypeError(
                "custom_training_table must be PreparedCustomTrainingTable")
        if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in self.timestamps):
            raise TypeError("timestamps must contain integer epoch values")

    @classmethod
    def from_spec(
            cls,
            *,
            graph: Mapping[str, Any],
            predictive_query: str,
            prepared_graph: PreparedGraph,
            run_id: str,
            scratch_dir: str,
            artifact_dir: str,
            model_plan: ModelPlan | None = None,
            training_table_generation_plan: TrainingTableGenerationPlan | None = None,
            search: search_options.AutoMLSearchOptions | None = None,
            custom_training_table: PreparedCustomTrainingTable | None = None,
            timestamps: tuple[int, ...] = (),
    ) -> "TrainingWorkflowRequest":
        """Validate plain graph/query inputs before any materialization."""

        resolved_graph = normalize_graph(graph)
        return cls(
            query=parse_and_validate_query(predictive_query, resolved_graph),
            graph=resolved_graph,
            prepared_graph=prepared_graph,
            run_id=run_id,
            scratch_dir=scratch_dir,
            artifact_dir=artifact_dir,
            model_plan=ModelPlan() if model_plan is None else model_plan,
            training_table_generation_plan=(
                TrainingTableGenerationPlan() if training_table_generation_plan
                is None else training_table_generation_plan),
            search=search or search_options.AutoMLSearchOptions(),
            custom_training_table=custom_training_table,
            timestamps=timestamps,
        )


def prepare_training_request(
        *,
        connector: TrainingConnector,
        graph: Mapping[str, Any],
        predictive_query: str,
        run_id: str,
        model_plan: ModelPlan | None = None,
        training_table_generation_plan: TrainingTableGenerationPlan | None = None,
        search: search_options.AutoMLSearchOptions | None = None,
        custom_training_table: CustomTrainingTable | None = None,
        timestamps: tuple[int, ...] = (),
) -> TrainingWorkflowRequest:
    """Prepare warehouse data through a connector and enter core training."""

    resolved_plan = ModelPlan() if model_plan is None else model_plan
    _require_separate_generation_plan(resolved_plan)
    prepared = connector.prepare(
        graph=graph,
        predictive_query=predictive_query,
        model_plan=resolved_plan,
        custom_training_table=custom_training_table,
        run_id=run_id,
    )
    if not isinstance(prepared, ConnectorPreparation):
        raise TypeError("connector.prepare must return ConnectorPreparation")
    return TrainingWorkflowRequest.from_spec(
        graph=graph,
        predictive_query=predictive_query,
        prepared_graph=prepared.prepared_graph,
        run_id=run_id,
        scratch_dir=prepared.scratch_dir,
        artifact_dir=prepared.artifact_dir,
        model_plan=resolved_plan,
        training_table_generation_plan=training_table_generation_plan,
        search=search,
        custom_training_table=prepared.custom_training_table,
        timestamps=timestamps,
    )


@dataclass(frozen=True)
class MaterializedTrainingTask:
    """DiskGraph task bundle plus its single resolved numerical task."""

    manifest_path: str
    runtime_manifest_path: str
    bundle_dir: str
    storage_kind: str
    task_type: str
    task_family: str
    train_table_extension_path: str
    task: ResolvedNumericalTask
    #: The caller's plan resolved against this task. Training reads its
    #: options from here rather than resolving the caller's plan again.
    model_plan: ModelPlan

    #: Exactly the keys `to_stage_output` writes. Staged runners validate a
    #: cached stage output against this before reusing it, so the set has one
    #: home: a runner holding a stale copy would reject every output and
    #: silently re-run ingest and materialization instead of failing loudly.
    STAGE_OUTPUT_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "manifest_path",
        "runtime_manifest_path",
        "bundle_dir",
        "storage_kind",
        "task_type",
        "task_family",
        "train_table_extension_path",
        "resolved_task",
        "model_plan",
    })

    def to_stage_output(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "runtime_manifest_path": self.runtime_manifest_path,
            "bundle_dir": self.bundle_dir,
            "storage_kind": self.storage_kind,
            "task_type": self.task_type,
            "task_family": self.task_family,
            "train_table_extension_path": self.train_table_extension_path,
            "resolved_task": self.task.to_dict(),
            "model_plan": to_json_dict(self.model_plan),
        }

    @classmethod
    def from_stage_output(
        cls,
        output: Mapping[str, Any],
    ) -> "MaterializedTrainingTask":
        if set(output) != cls.STAGE_OUTPUT_FIELDS:
            raise ValueError(
                "materialized training output fields must exactly match "
                f"{sorted(cls.STAGE_OUTPUT_FIELDS)!r}")
        return cls(
            manifest_path=str(output["manifest_path"]),
            runtime_manifest_path=str(output["runtime_manifest_path"]),
            bundle_dir=str(output["bundle_dir"]),
            storage_kind=str(output["storage_kind"]),
            task_type=str(output["task_type"]),
            task_family=str(output["task_family"]),
            train_table_extension_path=str(
                output["train_table_extension_path"]),
            task=ResolvedNumericalTask.from_dict(output["resolved_task"]),
            model_plan=from_json_dict(dict(output["model_plan"]), ModelPlan),
        )


def build_training_task_plan(
    request: TrainingWorkflowRequest, ) -> NativePqueryTaskPlan:
    """Build the exact train-table and task-dataset plan for a request."""

    return build_native_pquery_task_plan(
        query=request.query,
        graph=request.graph,
        training_table_generation_plan=request.training_table_generation_plan,
        run_id=request.run_id,
        graph_index_path=request.prepared_graph.index_path,
        scratch_dir=request.scratch_dir,
        artifact_dir=request.artifact_dir,
        graph_index_id=request.prepared_graph.index_id,
        timestamps=request.timestamps,
    )


def _default_num_neighbors(options: NeighborSamplingPlan) -> tuple[int, ...]:
    """Read the first trial's per-hop fanout out of a caller's plan."""

    search_space = inferred_or_missing_type_to_none(options.num_neighbors)
    if not search_space:
        return ResolvedTaskSampling().default_num_neighbors
    num_neighbors = search_space[0]
    if isinstance(num_neighbors, (list, tuple)):  # Backward compatibility.
        return tuple(int(value) for value in num_neighbors)
    hops = num_neighbors.to_dict()
    return tuple(int(hops[index].default) for index in sorted(hops))


def infer_max_target_neighbors_per_entity(
    index_path: str,
    target_to_entity_paths: TargetToEntityPaths,
) -> int | None:
    """Derive the hop-1 fanout on the target->entity edge from the data.

    The degree policy is shared with the monolith's
    `infer_authoring_max_target_neighbors_per_entity`; all this supplies is
    where the row counts come from -- the ingested DiskGraph statistics,
    which are the only counts a portable run has.

    No `avg_entity_degree` is offered because there is none to offer:
    DiskGraph-native train tables never populate the degree statistics, so
    the monolith takes the same row-count branch for these tasks.
    """

    return infer_target_edge_fanout(
        target_to_entity_paths,
        row_count=lambda table_name: table_row_count(index_path, table_name),
    )


def _drop_unresolvable_inferred_fanout(plan: ModelPlan) -> None:
    """Fall back to the hop default where no degree estimate exists.

    `AutoTrainer` fills every `inferred` hop-1 entry from the resolved
    task's `max_target_neighbors_per_entity`; with no estimate to put there
    it would assert on the sentinel instead. Removing the per-edge entry
    leaves the hop's own default in force, which is what the portable path
    sampled before it began creating the entry at all.
    """

    sampling = plan.neighbor_sampling
    for num_neighbors in sampling.num_neighbors:
        if not isinstance(num_neighbors, NumNeighborsConfig):
            continue
        for hop_cfg in num_neighbors.to_dict().values():
            for key, value in hop_cfg.__pydantic_extra__.items():
                if _is_inferred(value):
                    delattr(hop_cfg, key)
    targets = sampling.max_target_neighbors_per_entity
    if isinstance(targets, list) and any(map(_is_inferred, targets)):
        # Backward compatibility: the flat list `AutoTrainer` still fills
        # from the same estimate. An `inferred` entry left here trips the
        # same assertion, so resolve it to the first hop's default.
        # `_default_num_neighbors` returns () for a hopless config, which
        # has no hop-1 default to fall back to; reuse its own empty-search
        # -space fallback rather than indexing off the end.
        defaults = (_default_num_neighbors(sampling)
                    or ResolvedTaskSampling().default_num_neighbors)
        default = defaults[0]
        sampling.max_target_neighbors_per_entity = [
            default if _is_inferred(value) else value for value in targets
        ]


def effective_max_target_neighbors(
    request: TrainingWorkflowRequest,
    inferred_max_target_neighbors: int | None,
) -> int | None:
    """The one number `AutoTrainer` will fill `inferred` entries from.

    The data estimate when there is one, and otherwise the number the
    caller typed. Both the plan and the task have to be resolved against
    the same answer: keeping an `inferred` entry the task cannot fill, or
    dropping one the task could have filled, are the two ways this goes
    quietly wrong.
    """

    if inferred_max_target_neighbors is not None:
        return inferred_max_target_neighbors
    targets = inferred_or_missing_type_to_none(
        request.model_plan.neighbor_sampling.max_target_neighbors_per_entity)
    return inferred_or_missing_type_to_none(targets[0]) if targets else None


def _resolved_task_options(
    request: TrainingWorkflowRequest,
    model_plan: ModelPlan,
    inferred_max_target_neighbors: int | None = None,
) -> dict[str, Any]:
    """Resolve the task-level settings a caller may override.

    ``metrics`` comes from the *resolved* plan's ``training_job``, so the
    task computes what the plan tunes on -- the way the platform's own
    `parse_pquery` writes the plan's list onto the task it just built.
    `validate` fills that list from a default that knows the query type, the
    run mode and the weight column, while `default_metrics` -- the task's
    fallback -- knows only the task type. Weighted link prediction is where
    they part: the plan tunes on `weighted_ndcg@k`, which the task would
    never compute. ``lag_timesteps`` and ``timeframe_step`` come from
    ``training_table_generation``, and the temporal offsets are derived from
    the query's own date-offset range.

    ``sampling`` is projected from the *caller's* neighbor-sampling plan, not
    the resolved one: `AutoTrainer` resolves `neighbor_sampling` from the
    data, so an option the caller left unset has to reach the task as the
    resolved-task default rather than as the plan default `validate` would
    fill in. ``replace``, ``disjoint``, and ``temporal_strategy`` have no
    model-plan counterpart at all and keep their resolved-task defaults.

    ``max_target_neighbors_per_entity`` is the exception, and carries
    ``inferred_max_target_neighbors`` rather than the caller's own number.
    It is not an option the task samples with: it is the single value
    `AutoTrainer` writes into every `inferred` hop-1 entry, so the platform
    puts its degree estimate -- never the caller's plan -- there too. It
    falls back to the caller's number only when no estimate could be
    derived from the data.

    A caller who sets the option and nothing else is therefore overridden
    by the estimate, and only a per-edge hop-1 fanout they name themselves
    survives -- `AutoTrainer` leaves a non-sentinel entry alone. The plan's
    own list is read only under the legacy list form of ``num_neighbors``,
    which `from_default` no longer produces. That is the platform's
    behaviour too (`bin/kumo-auto` overrides the same way), which is the
    point: the option is a request for inference, not a fanout.
    """
    table_options = request.training_table_generation_plan
    sampling_options = request.model_plan.neighbor_sampling

    unix_start_offset = unix_end_offset = unix_step_size = None
    if request.query.query_type == "temporal":
        offset = request.query.query.get_combined_date_offset_range()
        start = int(offset.start or 0)
        end = int(offset.end)
        timeframe_step = inferred_or_missing_type_to_none(
            table_options.timeframe_step)
        step = int(timeframe_step or end - start)
        unix_start_offset = int(compute_unix_step(start, offset.unit))
        unix_end_offset = int(compute_unix_step(end, offset.unit))
        unix_step_size = int(compute_unix_step(step, offset.unit))

    default_lags = 1 if request.query.task_type == "forecasting" else 0
    lag_timesteps = inferred_or_missing_type_to_none(
        table_options.lag_timesteps)
    if lag_timesteps is None:
        lag_timesteps = default_lags
    max_target_neighbors = effective_max_target_neighbors(
        request, inferred_max_target_neighbors)
    metrics = model_plan.training_job.metrics
    # `MissingType` and `InferredType` are `str` enums, so an unresolved plan
    # would not raise here -- `list('???')` is `['?', '?', '?']`, a metric
    # list of three nonsense names. An empty list is just as quiet: it is
    # falsy, so `from_manifest` would swap in `default_metrics` and put back
    # the plan/task divergence this resolution order exists to remove.
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(
            "the resolved model plan must carry a non-empty metric list "
            f"(got {metrics!r}); `ModelPlan.validate` fills it in")
    sampling = ResolvedTaskSampling(
        default_num_neighbors=_default_num_neighbors(sampling_options),
        adaptive=bool(sampling_options.adaptive_sampling),
        max_target_neighbors_per_entity=(None if max_target_neighbors is None
                                         else int(max_target_neighbors)),
    )
    return {
        "name": request.run_id,
        "metrics": list(metrics),
        "forecast_length": request.query.forecast_length,
        "lag_timesteps": int(lag_timesteps),
        "unix_start_offset": unix_start_offset,
        "unix_end_offset": unix_end_offset,
        "unix_step_size": unix_step_size,
        "sampling": sampling,
        "exclude_cols_dict": request.query.query.get_exclude_cols_dict(),
    }


def materialize_generated_training_task(
    request: TrainingWorkflowRequest,
    task_dataset: TaskDatasetMaterializeSpec,
    generated: GeneratedTrainTableResult,
    *,
    tt_id: str,
    runtime: Any | None = None,
) -> MaterializedTrainingTask:
    """Ingest generated rows and materialize a selected graph/query task.

    ``tt_id`` names a writable extension under the selected graph index.
    Independent calls that reuse one generated staging directory must supply
    distinct ids because DiskGraph ingest does not lock or reuse extensions.
    """

    ingested = ingest_train_table(
        staging_dir=generated.staging_dir,
        index_path=request.prepared_graph.index_path,
        tt_id=tt_id,
        runtime=runtime,
    )
    # The selected graph path and per-call ingest id determine the actual
    # extension. Always replace the builder's pre-ingest manifest placeholder.
    task_dataset = replace(task_dataset,
                           tt_manifest_path=ingested.manifest_path)
    result = materialize_task_dataset(
        graph_index_id=request.prepared_graph.index_id,
        graph_index_path=request.prepared_graph.index_path,
        spec=task_dataset,
    )
    manifest = DiskgraphTaskManifest.load(result.manifest_path)
    facts = plan_validation_facts(manifest)
    # The degree estimate is derived before the plan is resolved: it decides
    # whether the plan may keep an `inferred` per-edge fanout at all.
    inferred_max_target_neighbors = effective_max_target_neighbors(
        request,
        infer_max_target_neighbors_per_entity(
            request.prepared_graph.index_path,
            facts["target_to_entity_paths"],
        ),
    )
    # The plan is resolved first: the task is then built once, already
    # holding the metric list the plan named.
    model_plan = resolve_model_plan(
        request.model_plan,
        request.search,
        training_table_generation_plan=request.training_table_generation_plan,
        query_type=QueryType(request.query.query_type),
        # The manifest's time column is physical; real time comes from the query.
        has_train_table_time_col=query_is_temporal(request.query,
                                                   request.graph),
        graph=to_plan_graph(request.graph),
        inferred_max_target_neighbors=inferred_max_target_neighbors,
        **facts,
    )
    task = ResolvedNumericalTask.from_manifest(
        manifest,
        column_stats=manifest.load_column_stats(),
        **_resolved_task_options(request, model_plan,
                                 inferred_max_target_neighbors),
    )
    return MaterializedTrainingTask(
        manifest_path=result.manifest_path,
        runtime_manifest_path=result.runtime_manifest_path,
        bundle_dir=result.bundle_dir,
        storage_kind=result.storage_kind,
        task_type=task_dataset.task_type,
        task_family=task_dataset.task_family,
        train_table_extension_path=ingested.extension_path,
        task=task,
        model_plan=model_plan,
    )


def materialize_training_task(
    request: TrainingWorkflowRequest,
    *,
    runtime: Any | None = None,
) -> MaterializedTrainingTask:
    """One-shot replacement for the former materialize activity chain."""

    plan = build_training_task_plan(request)
    if request.custom_training_table is None:
        generated = generate_train_table(plan.train_table, runtime=runtime)
    else:
        generated = generate_custom_train_table(
            plan.train_table,
            request.custom_training_table,
            runtime=runtime,
        )
    return materialize_generated_training_task(
        request,
        plan.task_dataset,
        generated,
        tt_id=plan.train_table.tt_id,
        runtime=runtime,
    )


def resolve_model_plan(
    model_plan: ModelPlan,
    search: search_options.AutoMLSearchOptions | None = None,
    *,
    training_table_generation_plan: TrainingTableGenerationPlan | None = None,
    task_type: TaskType,
    query_type: QueryType,
    has_train_table_time_col: bool,
    entity_table_names: list[str],
    target_to_entity_paths: list[list[tuple[str, str, str]]],
    has_train_table_weight_col: bool,
    graph: PlanGraph | None = None,
    inferred_max_target_neighbors: int | None = None,
) -> ModelPlan:
    """Resolve one caller plan against the materialized task's defaults.

    The task facts arrive as the plain values `ModelPlan.validate` asks for.
    Generation settings are supplied separately; nondefault settings in the
    historical input model-plan section are rejected. The resolved artifact
    retains that section for legacy consumers.

    ``graph`` is what lets `validate` create the per-target-edge hop-1
    `inferred` entry and read column stypes for metric validation; without
    it the plan resolves as if the graph had no edges worth a special
    fanout. It travels with ``inferred_max_target_neighbors``, the estimate
    that entry is later filled from, because creating the entry without one
    leaves a sentinel `AutoTrainer` asserts on.

    Options the caller left *missing* are filled from the Kumo default for
    this task and query type, and the caller's own options are validated
    against them. Options left *inferred* stay inferred: that sentinel means
    "resolve this from the data later", which `AutoTrainer` does for
    `neighbor_sampling`, and which `_split_options` does for the training
    table's split. The caller's plan is never mutated.
    """

    _require_separate_generation_plan(model_plan)
    search = search or search_options.AutoMLSearchOptions()
    plan = copy.deepcopy(model_plan)
    # Legacy validation and resolved-plan artifacts still carry this section.
    plan.training_table_generation = (
        TrainingTableGenerationPlan() if training_table_generation_plan is None
        else copy.deepcopy(training_table_generation_plan))
    # `query_type` and `has_train_table_time_col` are independent, which is
    # why both are asked for. A static query over timestamped rows (fraud on
    # transactions.created_at) yields a train table with a time column, so
    # deriving the query type from that column reads a static run as
    # temporal. Link prediction diverges the other way: its temporal-ness is
    # the semantic task type and ignores a physical pseudo time column that
    # the train table may still carry.
    plan.validate(
        task_type=task_type,
        query_type=query_type,
        has_train_table_time_col=has_train_table_time_col,
        run_mode=search.run_mode,
        graph=graph,
        entity_table_names=list(entity_table_names),
        target_to_entity_paths=target_to_entity_paths,
        has_train_table_weight_col=has_train_table_weight_col,
    )
    if inferred_max_target_neighbors is None:
        _drop_unresolvable_inferred_fanout(plan)
    if search.num_trials is not None:
        plan.training_job.num_experiments = search.num_trials
    if search.seed is not None:
        plan.training_job.manual_seed = search.seed
    return plan


TrainingBindingsFactory = Callable[[ResolvedTrainingInput], TrainingBindings]


def train_prepared_graph(
    request: TrainingWorkflowRequest,
    *,
    home_dir: str,
    bindings_factory: TrainingBindingsFactory,
    options: CallableTrainingOptions | None = None,
    runtime: Any | None = None,
) -> CallableTrainingResult:
    """Materialize and train through the original Kumo AutoML implementation."""

    materialized = materialize_training_task(request, runtime=runtime)
    training_input = ResolvedTrainingInput(
        task=materialized.task,
        model_plan=materialized.model_plan,
        home_dir=home_dir,
        task_dataset_bundle_dir=materialized.bundle_dir,
    )
    return run_callable_training(
        training_input,
        bindings=bindings_factory(training_input),
        options=options or CallableTrainingOptions(),
    )


@dataclass(frozen=True)
class GNNTrainingResult:
    """Compact result returned by the configured two-argument workflow."""

    run_id: str
    model_name: str
    version: str
    model_uri: str
    metrics: Mapping[str, float]
    artifact_paths: Mapping[str, str]

    @property
    def val_metric(self) -> float | None:
        values = [
            float(value) for name, value in self.metrics.items()
            if name.startswith("val_") and "loss" not in name
        ]
        return values[0] if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_uri": self.model_uri,
            "val_metric": self.val_metric,
            "metrics": dict(self.metrics),
            "artifact_paths": dict(self.artifact_paths),
        }


@dataclass(frozen=True)
class GNNPredictionResult:
    """Compact result returned by the configured prediction workflow."""

    run_id: str
    status: str
    output_table: str
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "output_table": self.output_table,
            "row_count": self.row_count,
        }


@runtime_checkable
class GNNPredictionBackend(Protocol):
    """Environment adapter for the real Kumo batch-prediction pipeline."""
    def predict(
        self,
        *,
        graph: Mapping[str, Any],
        query: str,
        checkpoint_uri: str,
        output_table: str,
        model_plan: ModelPlan,
        task_type: str,
        prediction_timestamps: tuple[int, ...],
        top_k: int | None,
        return_embeddings: bool,
        write_mode: str,
        run_id: str,
    ) -> GNNPredictionResult:
        """Run inference and persist its environment-native output."""


@dataclass(frozen=True)
class GNNPredictionWorkflow:
    """Configured workflow for query, model, and output-table prediction."""

    backend: GNNPredictionBackend
    graph: Mapping[str, Any]
    model_plan: ModelPlan = field(default_factory=ModelPlan)
    task_type: str = "node_prediction"
    prediction_timestamps: tuple[int, ...] = ()
    top_k: int | None = None
    return_embeddings: bool = False
    write_mode: str = "overwrite"

    def __post_init__(self) -> None:
        if not isinstance(self.model_plan, ModelPlan):
            raise TypeError("model_plan must be a ModelPlan")

    def predict_gnn(
        self,
        query: str,
        checkpoint_uri: str,
        output_table: str,
        *,
        run_id: str | None = None,
    ) -> GNNPredictionResult:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if not checkpoint_uri:
            raise ValueError("checkpoint_uri must be non-empty")
        if not output_table:
            raise ValueError("output_table must be non-empty")
        resolved_run_id = run_id or uuid.uuid4().hex
        result = self.backend.predict(
            graph=self.graph,
            query=query,
            checkpoint_uri=checkpoint_uri,
            output_table=output_table,
            model_plan=self.model_plan,
            task_type=self.task_type,
            prediction_timestamps=self.prediction_timestamps,
            top_k=self.top_k,
            return_embeddings=self.return_embeddings,
            write_mode=self.write_mode,
            run_id=resolved_run_id,
        )
        if not isinstance(result, GNNPredictionResult):
            raise TypeError(
                "prediction backend must return GNNPredictionResult")
        if result.run_id != resolved_run_id:
            raise ValueError("prediction backend returned a different run_id")
        if result.output_table != output_table:
            raise ValueError(
                "prediction backend returned a different output_table")
        return result


@dataclass(frozen=True)
class GNNTrainingWorkflow:
    """Configured workflow whose call surface is query plus model name."""

    connector: TrainingConnector
    graph: Mapping[str, Any]
    bindings_factory: TrainingBindingsFactory
    publisher: TrainingPublisher
    model_plan: ModelPlan = field(default_factory=ModelPlan)
    #: Settings used to generate or prepare the training rows.
    training_table_generation_plan: TrainingTableGenerationPlan = field(
        default_factory=TrainingTableGenerationPlan)
    search: search_options.AutoMLSearchOptions = field(
        default_factory=search_options.AutoMLSearchOptions)
    custom_training_table: CustomTrainingTable | None = None
    options: CallableTrainingOptions = field(
        default_factory=CallableTrainingOptions)
    timestamps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model_plan, ModelPlan):
            raise TypeError("model_plan must be a ModelPlan")
        _require_separate_generation_plan(self.model_plan)
        if not isinstance(self.training_table_generation_plan,
                          TrainingTableGenerationPlan):
            raise TypeError("training_table_generation_plan must be a "
                            "TrainingTableGenerationPlan")

    def train_gnn(
        self,
        query: str,
        output_model_name: str,
        *,
        run_id: str | None = None,
    ) -> GNNTrainingResult:
        if not output_model_name:
            raise ValueError("output_model_name must be non-empty")
        resolved_run_id = run_id or uuid.uuid4().hex
        request = prepare_training_request(
            connector=self.connector,
            graph=self.graph,
            predictive_query=query,
            run_id=resolved_run_id,
            model_plan=self.model_plan,
            training_table_generation_plan=self.training_table_generation_plan,
            search=self.search,
            custom_training_table=self.custom_training_table,
            timestamps=self.timestamps,
        )
        training = train_prepared_graph(
            request,
            home_dir=request.scratch_dir,
            bindings_factory=self.bindings_factory,
            options=self.options,
        )
        published = self.publisher.publish(output_model_name, training)
        if not isinstance(published, PublishedModel):
            raise TypeError("publisher.publish must return PublishedModel")
        if published.name != output_model_name:
            raise ValueError("publisher returned a different model name")
        return GNNTrainingResult(
            run_id=resolved_run_id,
            model_name=published.name,
            version=published.version,
            model_uri=published.uri,
            metrics=training.metrics,
            artifact_paths=training.artifact_paths,
        )


__all__ = [
    "ConnectorPreparation",
    "GNNPredictionBackend",
    "GNNPredictionResult",
    "GNNPredictionWorkflow",
    "GNNTrainingResult",
    "GNNTrainingWorkflow",
    "MaterializedTrainingTask",
    "PreparedGraph",
    "PublishedModel",
    "TrainingBindingsFactory",
    "TrainingConnector",
    "TrainingPublisher",
    "TrainingWorkflowRequest",
    "build_training_task_plan",
    "materialize_generated_training_task",
    "materialize_training_task",
    "prepare_training_request",
    "resolve_model_plan",
    "train_prepared_graph",
]
