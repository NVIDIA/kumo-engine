# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`PredictiveQuery`: mirrors the cloud SDK's PQL surface, minus what's out
of scope for this package.

`PredictiveQuery(graph, query)` binds a `Graph` at construction, matching the
cloud SDK's `kumo.PredictiveQuery(graph=graph, query=QUERY)` -- not a
`GraphSnapshot`. `PredictiveQuery.from_snapshot(snapshot, query)` binds a
snapshot instead and parses against `snapshot.graph_config`.

Parsing (`kumo_core.pquery.parse_and_validate_query` against
`self.graph.native_graph`, needing no materialized index) happens lazily, on first
access to the `_parsed` `cached_property`, and only once:
`validate()`/`get_task_type()` both read that same property rather than
re-parsing on every call. Confirmed against the existing local backend's own
`_LocalPQueryBackend.validate` (`kumoai/backend/local/pquery.py`), which parses
against `to_portable(definition)` -- the schema-level conversion, not an
ingested snapshot -- that this doesn't need a `GraphSnapshot` either.

Construction itself stays I/O-free, consistent with `Table`/`Graph`: nothing
parses until `_parsed` is actually touched (by `validate()`, `get_task_type()`,
or a caller directly). A failed parse is *not* cached -- `functools.
cached_property` only caches a successful return -- so both `validate()` and
`get_task_type()` simply access `_parsed` and let a parse failure propagate;
neither wraps it into a non-raising result.

`suggest_model_plan` uses resolved graph/query metadata. Methods that generate
training or prediction tables also read the materialized index. These methods
accept an explicit `GraphSnapshot` of this query's graph and never trigger
graph ingest themselves.

Supported, matching `kumo.engine.pquery.predictive_query.PredictiveQuery` in the
cloud SDK: `validate`, `get_task_type`, `suggest_model_plan`,
`generate_training_table`, `generate_prediction_table`. All take
`kumoapi.model_plan`/`kumoapi.task` types unchanged (`RunMode`,
`SuggestModelPlanRequest`, `ModelPlan`, `TrainingTableGenerationPlan`,
`TaskType`) -- the same shared, backend-agnostic config the cloud pipeline
uses.

Not supported (yet): `suggest_prediction_table_plan` (explicitly deferred).
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan import ModelPlanValueError
from kumo_core.model_plan import \
    TrainingTableGenerationPlan as CoreGenerationPlan
from kumo_core.model_plan import (
    inferred_or_missing_type_to_none,
    unsupported_entity_candidate_errors,
)
from kumo_core.model_plan_conversion import public_model_plan_from_internal
from kumo_core.pquery import (
    ValidatedNativeQuery,
    parse_and_validate_query,
    query_is_temporal,
    resolve_query_split,
    target_to_entity_paths,
    to_plan_graph,
)
from kumo_core.training.splits import RandomLinkSplit
from kumo_core.training.table import generate_train_table
from kumo_core.training.workflow import (
    PreparedGraph,
    TrainingWorkflowRequest,
    build_training_task_plan,
)
from kumo_core.types import TaskType as CoreTaskType
from kumoapi.common import ValidationResponse
from kumoapi.json_serde import to_json_dict
from kumoapi.model_plan import ModelPlan, RunMode, TrainingTableGenerationPlan
from kumoapi.pquery import QueryType
from kumoapi.task import TaskType
from kumoapi.train import TrainingTableSpec

from kumo.engine._embeddings import no_encoder_inference_warnings
from kumo.engine._scratch import (
    new_run_id,
    prediction_table_scratch_dir,
    train_table_scratch_dir,
)
from kumo.engine.graph import Graph, GraphSnapshot
from kumo.engine.pquery.prediction_table import (
    PredictionTable,
    _compile_prediction_table,
)
from kumo.engine.pquery.training_table import TrainingTable, _finish

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictiveQuery:
    r"""An immutable PQL query bound to a `Graph` or a `GraphSnapshot`."""

    graph: Graph | GraphSnapshot
    query: str

    @classmethod
    def from_snapshot(cls, snapshot: GraphSnapshot,
                      query: str) -> 'PredictiveQuery':
        r"""Bind `query` to `snapshot`'s graph without its `Graph` object.

        Snapshots passed to this query's methods must come from the same
        graph (same `graph_key`). That only checks the snapshot belongs to
        this query's graph; it says nothing about compatibility with a
        trained model.
        """
        return cls(graph=snapshot, query=query)

    @cached_property
    def _parsed(self) -> ValidatedNativeQuery:
        r"""Parses and validates `self.query` against `self.graph.native_graph`,
        via `kumo_core.pquery.parse_and_validate_query` --
        the same parser `Trainer._predict` already calls directly, and what
        `kumo_core.training.workflow.TrainingWorkflowRequest.from_spec`
        uses internally for `fit`. No materialized `GraphSnapshot` needed
        (see module docstring).

        `functools.cached_property` writes directly into `self.__dict__` on
        first access, bypassing `__setattr__` -- this works even though
        `PredictiveQuery` is a frozen dataclass, and is why this needs no
        `object.__setattr__` dance the way `Graph`/`Table` construction-time
        caching would.

        Raises (an unspecified `kumo_core` exception, typically a
        `PredictiveQueryPlanError`) on an invalid query -- unknown table/
        column, malformed PQL, a dangling `FOR EACH` entity, and so on. The
        failure itself is not cached; a later access re-parses.

        `task_type`/`task_family`/`query_type` on the returned
        `ValidatedNativeQuery` are all plain `str`; `get_task_type` below
        wraps `task_type` into the shared, public `kumoapi.task.TaskType`.
        """
        return parse_and_validate_query(self.query, self.graph.native_graph)

    def _require_matching_snapshot(self, snapshot: GraphSnapshot) -> None:
        if isinstance(self.graph, GraphSnapshot):
            if snapshot.id == self.graph.id:
                return
            graph_key = self.graph.graph_key
        else:
            graph_key = self.graph.id
        if snapshot.graph_key != graph_key:
            raise ValueError(
                f"snapshot {snapshot.id!r} was not materialized from "
                f"this query's graph (id {graph_key!r}) -- pass a "
                "snapshot of that graph.")

    def validate(self, verbose: bool = True) -> ValidationResponse:
        r"""Return validation warnings and information for a valid query.

        Invalid queries raise the same exception as query parsing. The
        response is a copy, so modifying it does not alter cached diagnostics.

        Args:
            verbose: Whether to log non-error output of this validation, as
                the cloud SDK does: warnings and information at WARNING level,
                or a confirmation at INFO level when the response is empty.
        """
        response = deepcopy(self._parsed.validation_response)
        if verbose:
            if response.empty():
                logger.info("Query %s is configured correctly.", self.query)
            else:
                logger.warning(response.message())
        return response

    def get_task_type(self) -> TaskType:
        r"""Infers this query's task type (binary classification, regression,
        ...) from `self._parsed`. Mirrors the cloud SDK's `PredictiveQuery.
        get_task_type()`. Raises the same way `self._parsed` does, rather
        than returning a `ValidationResponse` -- there is no sensible task
        type to hand back for an unparseable query.
        """
        return TaskType(self._parsed.task_type)

    def _model_plan_validation_facts(self, run_mode: RunMode) -> dict[str, Any]:
        parsed = self._parsed
        graph = self.graph.native_graph
        entity_tables = [parsed.entity_table]
        if parsed.rhs_node_type is not None:
            entity_tables.append(parsed.rhs_node_type)
        return dict(
            task_type=CoreTaskType(parsed.task_type),
            query_type=QueryType(parsed.query_type),
            has_train_table_time_col=query_is_temporal(parsed, graph),
            run_mode=RunMode(run_mode),
            graph=to_plan_graph(graph),
            entity_table_names=entity_tables,
            target_to_entity_paths=target_to_entity_paths(parsed, graph),
        )

    def suggest_model_plan(
        self,
        snapshot: GraphSnapshot,
        *,
        run_mode: RunMode = RunMode.FAST,
        train_table_spec: Optional[TrainingTableSpec] = None,
    ) -> ModelPlan:
        r"""Suggests a modeling plan given this query and a materialized
        `snapshot` of this query's graph. Mirrors the cloud SDK's
        `PredictiveQuery.suggest_model_plan`.

        Uses resolved graph/query metadata without generating a training table
        or starting training. Only `train_table_spec.weight_col` presence
        affects suggestions; inferred values are resolved during training.
        """
        self._require_matching_snapshot(snapshot)
        parsed = self._parsed
        plan = CoreModelPlan.from_default(
            **self._model_plan_validation_facts(run_mode),
            has_train_table_weight_col=(train_table_spec is not None
                                        and train_table_spec.weight_col
                                        is not None),
            forecast_length=(parsed.forecast_length
                             if parsed.task_type == "forecasting" else None),
            top_k=parsed.query.top_k,
        )
        return public_model_plan_from_internal(plan)

    def generate_training_table(
        self,
        snapshot: GraphSnapshot,
        plan: Optional[TrainingTableGenerationPlan] = None,
        *,
        mlops: Optional[Any] = None,
    ) -> TrainingTable:
        r"""Compiles this query into a `TrainingTable` against a materialized
        `snapshot` of this query's graph. Mirrors the cloud SDK's
        `PredictiveQuery.generate_training_table`.

        `Trainer.fit` consumes the returned `TrainingTable` directly. Its
        generated artifacts, required split stages, and effective generation
        settings are retained without keeping a graph-bound task plan.

        Args:
            snapshot: a snapshot of this query's graph to compile against.
            plan: if not given, uses `kumo_core`'s generation defaults,
                matching the local backend's existing
                `suggest_training_table_plan` default.
            mlops: a `kumo.MLOps`; with MLOps on (this or a `kumo.mlops`
                block), the table is also stored under the artifact root.
        """
        self._require_matching_snapshot(snapshot)
        for name in ('train_table_path_override', 'task_path_override'):
            if plan is not None and getattr(plan, name) is not None:
                raise ValueError(
                    f"'{name}' is not supported by native training-table "
                    "generation; use TrainingTable.from_source_table() to "
                    "train on existing labeled rows.")
        from kumo_core.io.json import from_json_dict

        options = (CoreGenerationPlan() if plan is None else from_json_dict(
            to_json_dict(plan), CoreGenerationPlan))
        value = inferred_or_missing_type_to_none(options.split)
        if value is not None:
            split = resolve_query_split(self._parsed, value)
            if isinstance(split, RandomLinkSplit):
                options = replace(options, split=repr(split))

        # Validate before allocating a run directory or invoking native
        # generation. DiskGraph does not carry weight_col in TrainTableRequest,
        # so accepting one here would silently emit an unweighted artifact and
        # defer the actionable rejection until fit.
        # generate_training_table has no run_mode argument, so FAST only
        # preflights mode-independent constraints. Fit remains authoritative
        # for validation that depends on its selected run mode.
        errors, warnings = options.validate(
            **self._model_plan_validation_facts(RunMode.FAST),
            has_train_table_weight_col=False,
        )
        for warning in warnings:
            logger.warning(f"{warning.title}: {warning.message}")
        if errors:
            raise ModelPlanValueError(errors)

        run_id = new_run_id()
        run_dir = train_table_scratch_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        workflow = TrainingWorkflowRequest.from_spec(
            graph=snapshot.graph_config,
            predictive_query=self.query,
            prepared_graph=PreparedGraph(
                index_id=snapshot.id,
                index_path=str(snapshot.index_path),
            ),
            run_id=run_id,
            scratch_dir=str(run_dir),
            artifact_dir=str(run_dir),
            training_table_generation_plan=options,
        )
        with no_encoder_inference_warnings():
            task_plan = build_training_task_plan(workflow)
        generated = generate_train_table(task_plan.train_table)

        generation_plan = options
        if workflow.query.query_type == 'temporal':
            offset = workflow.query.query.get_combined_date_offset_range()
            timeframe_step = inferred_or_missing_type_to_none(
                generation_plan.timeframe_step)
            generation_plan = replace(
                generation_plan,
                timeframe_step=(int(timeframe_step) if timeframe_step
                                is not None else int(offset.end) -
                                int(offset.start or 0)),
            )
        if workflow.query.task_type == 'forecasting':
            generation_plan = replace(
                generation_plan,
                forecast_length=workflow.query.forecast_length,
                lag_timesteps=task_plan.train_table.lag_timesteps,
                year_over_year=('yoy_offsets_secs'
                                in (task_plan.train_table.compiler_options
                                    or {})),
            )

        table = TrainingTable(
            query=self.query,
            train_table_path=Path(generated.train_table_path),
            manifest_path=Path(generated.manifest_path),
            staging_dir=Path(generated.staging_dir),
            required_stages=task_plan.task_dataset.required_stages,
            generation_plan=generation_plan,
        )
        return _finish(table, snapshot, task_plan.train_table, mlops)

    def generate_prediction_table(
        self,
        snapshot: GraphSnapshot,
        *,
        prediction_time: Optional[datetime] = None,
        plan: Optional[TrainingTableGenerationPlan] = None,
    ) -> PredictionTable:
        r"""Compiles this query into a `PredictionTable` against a
        materialized `snapshot` of this query's graph. Mirrors the cloud SDK's
        `PredictiveQuery.generate_prediction_table`, and parallels
        `generate_training_table` above.

        `Trainer.predict` consumes the returned `PredictionTable` directly
        -- `request`/`graph_index_id` carry forward exactly what this call
        compiled and resolved (including the anchor timestamp(s)
        `prediction_time` resolved to), so `predict` doesn't recompile the
        query or resolve a different anchor time (see `PredictionTable`'s
        docstring).

        Args:
            snapshot: a snapshot of this query's graph to compile against.
            prediction_time: the anchor timestamp for a temporal query;
                when omitted, uses the latest source-time maximum across the
                query's target aggregations in the snapshot.
            plan: effective training-table generation settings. Forecast
                lag, year-over-year, and timeframe settings must match the
                fitted trainer. A loaded trainer exposes this plan through
                ``training_table_generation_plan``; pass it as
                ``query.generate_prediction_table(
                snapshot, plan=trainer.training_table_generation_plan)``.
        """
        self._require_matching_snapshot(snapshot)
        if errors := unsupported_entity_candidate_errors(plan):
            raise ModelPlanValueError(errors)

        run_id = new_run_id()
        run_dir = prediction_table_scratch_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        request, generated, _ = _compile_prediction_table(
            snapshot,
            self._parsed,
            prediction_time=prediction_time,
            run_id=run_id,
            scratch_dir=run_dir,
            artifact_dir=run_dir,
            generation_plan=plan,
        )

        return PredictionTable(
            query=self.query,
            table_path=Path(generated.table_path),
            staging_dir=Path(generated.staging_dir),
            row_count=generated.row_count,
            tt_id=generated.tt_id,
            timestamps=request.timestamps,
            required_stages=request.task_dataset.required_stages,
            compiler_options=request.compiler_options,
            forecast_timeframe_step=request.forecast_timeframe_step,
        )
