# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved numerical task values consumed by AutoML and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.train.autoregression import YOY_COL
from kumo_core.types import TaskType

EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class ResolvedEncoderInput:
    column_name: str
    dtype: str
    stype: str
    encoder: dict[str, object]
    stats: dict[str, object] | None = None


@dataclass(frozen=True)
class ResolvedTaskSplit:
    family: str = "precomputed"
    stage_counts: dict[str, int] = field(default_factory=dict)
    stage_seed_times: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedTaskSampling:
    default_num_neighbors: tuple[int, ...] = (10, 10)
    num_neighbors_per_edge_type: dict[EdgeType,
                                      tuple[int,
                                            ...]] = field(default_factory=dict)
    replace: bool = False
    disjoint: bool = True
    adaptive: bool = False
    temporal_strategy: str = "last"
    max_target_neighbors_per_entity: int | None = None


@dataclass(frozen=True)
class ResolvedTaskGraph:
    node_types: tuple[str, ...]
    edge_types: tuple[EdgeType, ...]
    seed_node_type: str
    target_node_type: str


@dataclass(frozen=True)
class ResolvedTrainTable:
    name: str
    target_column: str | None
    time_column: str | None
    weight_column: str | None
    foreign_key_columns: tuple[str, ...] = ("ENTITY", )


@dataclass
class ResolvedNumericalTask:
    """Service-free task after graph and train-table materialization.

    ``task_type`` remains mutable because the established Trainer normalizes a
    one-class multiclass task to binary classification during initialization.
    All authoring, connector, and persistence state is intentionally absent.
    """

    name: str
    task_type: TaskType
    metrics: list[str]
    graph: ResolvedTaskGraph
    train_table: ResolvedTrainTable
    split: ResolvedTaskSplit = field(default_factory=ResolvedTaskSplit)
    sampling: ResolvedTaskSampling = field(default_factory=ResolvedTaskSampling)
    target_to_entity_paths: list[list[EdgeType]] = field(default_factory=list)
    exclude_cols_dict: dict[str, list[str]] = field(default_factory=dict)
    unix_start_offset: int | None = None
    unix_end_offset: int | None = None
    unix_step_size: int | None = None
    forecast_length: int = 1
    lag_timesteps: int = 0
    encoder_inputs: dict[str, list[ResolvedEncoderInput]] = field(
        default_factory=dict)
    target_encoder_input: ResolvedEncoderInput | None = None

    @property
    def is_temporal(self) -> bool:
        # Static and temporal link prediction are semantic task types. A
        # physical pseudo-time column used for splitting must not override
        # that contract.
        if self.task_type == TaskType.STATIC_LINK_PREDICTION:
            return False
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            return True
        return self.train_table.time_column is not None

    @property
    def has_year_over_year_labels(self) -> bool:
        """Whether the materialized train table carries year-over-year labels.

        The model plan only ever *requests* the feature, and its request is
        resolved twice over: an unset request compiles to off, and the
        column is only emitted for a forecasting train table. Both task
        construction paths instead derive `encoder_inputs` from the task
        manifest's feature schema, which materialization writes from the
        table it actually built -- so the built table, read back here, is
        the one source of truth for whether the model has a year-over-year
        input to encode.
        """

        return any(encoder_input.column_name == YOY_COL
                   for encoder_input in self.encoder_inputs.get(
                       self.train_table.name, ()))

    @property
    def unqualified_target_to_entity_paths(self) -> list[list[EdgeType]]:
        return self.target_to_entity_paths

    def to_task_config(self) -> "ResolvedNumericalTask":
        """Return the exact numerical task accepted by the core Trainer."""

        return self

    def get_train_table_stats(self, after_split: bool) -> None:
        del after_split
        return None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        sampling = dict(payload["sampling"])
        sampling["num_neighbors_per_edge_type"] = [
            {
                "edge_type": list(edge_type),
                "fanouts": list(fanouts)
            } for edge_type, fanouts in
            self.sampling.num_neighbors_per_edge_type.items()
        ]
        payload["sampling"] = sampling
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str,
                                        object]) -> "ResolvedNumericalTask":
        expected = {
            "name",
            "task_type",
            "metrics",
            "graph",
            "train_table",
            "split",
            "sampling",
            "target_to_entity_paths",
            "exclude_cols_dict",
            "unix_start_offset",
            "unix_end_offset",
            "unix_step_size",
            "forecast_length",
            "lag_timesteps",
            "encoder_inputs",
            "target_encoder_input",
        }
        if set(payload) != expected:
            raise ValueError(
                "resolved task fields must exactly match the schema")
        graph = payload["graph"]
        train_table = payload["train_table"]
        split = payload["split"]
        sampling = payload["sampling"]
        if not all(
                isinstance(value, Mapping)
                for value in (graph, train_table, split, sampling)):
            raise TypeError("resolved task nested values must be mappings")
        sampling_values = dict(sampling)
        per_edge = {}
        for entry in sampling_values.pop("num_neighbors_per_edge_type"):
            edge_type = tuple(entry["edge_type"])
            per_edge[edge_type] = tuple(entry["fanouts"])
        sampling_values["default_num_neighbors"] = tuple(
            sampling_values["default_num_neighbors"])
        sampling_values["num_neighbors_per_edge_type"] = per_edge
        graph_values = dict(graph)
        graph_values["node_types"] = tuple(graph_values["node_types"])
        graph_values["edge_types"] = tuple(
            tuple(edge) for edge in graph_values["edge_types"])
        split_values = dict(split)
        split_values["stage_seed_times"] = {
            str(stage): tuple(bounds)
            for stage, bounds in split_values["stage_seed_times"].items()
        }
        return cls(
            name=str(payload["name"]),
            task_type=TaskType(str(payload["task_type"])),
            metrics=[str(metric) for metric in payload["metrics"]],
            graph=ResolvedTaskGraph(**graph_values),
            train_table=ResolvedTrainTable(
                **{
                    **dict(train_table),
                    "foreign_key_columns":
                    tuple(train_table["foreign_key_columns"]),
                }),
            split=ResolvedTaskSplit(**split_values),
            sampling=ResolvedTaskSampling(**sampling_values),
            target_to_entity_paths=[[
                tuple(edge) for edge in path
            ] for path in payload["target_to_entity_paths"]],
            exclude_cols_dict={
                str(table): [str(column) for column in columns]
                for table, columns in payload["exclude_cols_dict"].items()
            },
            unix_start_offset=payload["unix_start_offset"],
            unix_end_offset=payload["unix_end_offset"],
            unix_step_size=payload["unix_step_size"],
            forecast_length=int(payload["forecast_length"]),
            lag_timesteps=int(payload["lag_timesteps"]),
            encoder_inputs={
                str(table):
                [ResolvedEncoderInput(**dict(value)) for value in values]
                for table, values in payload["encoder_inputs"].items()
            },
            target_encoder_input=(
                ResolvedEncoderInput(**dict(payload["target_encoder_input"]))
                if payload["target_encoder_input"] is not None else None),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: DiskgraphTaskManifest,
        *,
        name: str,
        metrics: list[str] | None = None,
        forecast_length: int = 1,
        lag_timesteps: int = 0,
        unix_start_offset: int | None = None,
        unix_end_offset: int | None = None,
        unix_step_size: int | None = None,
        sampling: ResolvedTaskSampling | None = None,
        exclude_cols_dict: Mapping[str, Sequence[str]] | None = None,
        column_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "ResolvedNumericalTask":
        """``column_stats`` (``manifest.load_column_stats()``) is required
        only to fit encoders; without it the inputs carry no statistics."""
        task_type = TaskType(manifest.task_type)
        edges = _manifest_edge_types(manifest)
        target_paths = _manifest_target_to_entity_paths(manifest, edges)
        columns = manifest.train_table_columns
        node_schema = manifest.feature_schema.get("node_types", {})
        encoder_inputs = {
            str(table): [
                ResolvedEncoderInput(
                    **{k: v
                       for k, v in value.items() if k != "stats"},
                    stats=(None if column_stats is None else
                           column_stats[str(table)][value["column_name"]]),
                ) for value in config.get("column_specs", ())
            ]
            for table, config in node_schema.items()
        }
        target_value = manifest.feature_schema.get("target_encoder_input")
        return cls(
            name=name,
            task_type=task_type,
            metrics=list(metrics or default_metrics(task_type)),
            graph=ResolvedTaskGraph(
                node_types=_manifest_node_types(manifest),
                edge_types=edges,
                seed_node_type=manifest.seed_node_type,
                target_node_type=manifest.target_node_type,
            ),
            train_table=ResolvedTrainTable(
                name=manifest.train_table_name,
                target_column=columns.get("target_column"),
                time_column=columns.get("time_column"),
                weight_column=columns.get("weight_column"),
                foreign_key_columns=tuple(columns["foreign_key_columns"]),
            ),
            split=ResolvedTaskSplit(
                stage_counts=dict(manifest.stage_counts),
                stage_seed_times=dict(manifest.stage_seed_times),
            ),
            sampling=sampling or ResolvedTaskSampling(),
            target_to_entity_paths=target_paths,
            exclude_cols_dict={
                str(table): [str(column) for column in columns]
                for table, columns in (exclude_cols_dict or {}).items()
            },
            unix_start_offset=unix_start_offset,
            unix_end_offset=unix_end_offset,
            unix_step_size=unix_step_size,
            forecast_length=forecast_length,
            lag_timesteps=lag_timesteps,
            encoder_inputs=encoder_inputs,
            target_encoder_input=(ResolvedEncoderInput(
                **dict(target_value)) if target_value is not None else None),
        )


def _manifest_edge_types(
        manifest: DiskgraphTaskManifest) -> tuple[EdgeType, ...]:
    return tuple(
        (str(edge["src"]), str(edge["rel"]), str(edge["dst"]))
        for edge in manifest.graph_runtime_metadata.get("edge_types", ()))


def _manifest_node_types(manifest: DiskgraphTaskManifest) -> tuple[str, ...]:
    return tuple(
        str(value)
        for value in manifest.graph_runtime_metadata.get("node_types", ()))


def _manifest_entity_table_names(
        manifest: DiskgraphTaskManifest) -> list[str]:
    """Name the tables a plan is validated against as entities.

    `PredictiveQueryV2.entity_table_names` is the seed table, followed by
    the RHS table for link prediction. The order carries meaning: metric
    validation resolves a `diversity[col]@k` column against the *last*
    name, and the adaptive sample-size estimate seeds off the *first*. The
    graph's full node-type list satisfies neither, so it is not a stand-in.
    """

    names = [manifest.seed_node_type]
    if manifest.target_node_type != manifest.seed_node_type:
        names.append(manifest.target_node_type)
    return names


def _manifest_target_to_entity_paths(
    manifest: DiskgraphTaskManifest,
    edges: tuple[EdgeType, ...],
) -> list[list[EdgeType]]:
    """Read the query's target->entity paths off the materialized task.

    The paths are recorded at materialization because the graph's edge list
    cannot reproduce them. `target_node_type` is the RHS entity for link
    prediction and the seed itself for node prediction, so matching edges
    against it looks for a `products -> users` or `users -> users` edge that
    no such graph has -- and finds nothing precisely where the platform
    finds the `orders -> users` fact edge whose hop-1 fanout it infers.

    Older bundles carry no paths and fall back to that edge match, which
    stays wrong for those, but no more wrong than it was when they were
    written.
    """

    recorded = manifest.target_to_entity_paths
    if recorded is not None:
        return [[edge for edge in path] for path in recorded]
    return [[edge] for edge in edges if edge[0] == manifest.target_node_type
            and edge[2] == manifest.seed_node_type]


def plan_validation_facts(manifest: DiskgraphTaskManifest) -> dict[str, Any]:
    """Read the task facts a model plan is validated and defaulted against.

    `from_manifest` derives the same values for the task itself. Both read
    one manifest, so a plan is never resolved against a different task than
    the one training receives.

    """
    columns = manifest.train_table_columns
    return {
        "task_type":
        TaskType(manifest.task_type),
        "entity_table_names":
        _manifest_entity_table_names(manifest),
        "target_to_entity_paths":
        _manifest_target_to_entity_paths(manifest,
                                         _manifest_edge_types(manifest)),
        "has_train_table_weight_col":
        columns.get("weight_column") is not None,
    }


def default_metrics(task_type: TaskType) -> list[str]:
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return ["acc"]
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return ["auroc", "ap", "auprc"]
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return [
            f"{metric}_{average}" for metric in ("ap", "auroc", "auprc")
            for average in ("macro", "micro", "per_label")
        ]
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        return ["mae", "mse", "rmse"]
    if task_type.is_ranking:
        return [
            f"{metric}@{k}" for k in (1, 10, 100)
            for metric in ("map", "precision", "recall", "f1")
        ]
    raise ValueError(f"unsupported task type {task_type.value!r}")
