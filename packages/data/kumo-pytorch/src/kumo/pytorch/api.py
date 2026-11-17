# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public declarations and execution lifecycle for relational model inputs."""

from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    AllAnchors,
    AnchorTable,
    CandidateColumn,
    padded_positions,
    shard_positions,
)
from kumo.pytorch.artifact import Artifact, FitRecord, FittedState, load_weights, save_weights
from kumo.pytorch.assembly import PreparedBatchPlan, PreviewResult, prepare_batch_spec
from kumo.pytorch.contract.batch import *  # noqa: F403
from kumo.pytorch.contract.batch import __all__ as _batch_api
from kumo.pytorch.contract.batch.validation import InputExposureWarning
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    ActiveInterval,
    Aggregate,
    BuiltSample,
    CalendarPart,
    CandidateRequest,
    CandidateSetSpec,
    Compare,
    Constant,
    DatedEntityValue,
    Derived,
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    ExternalTarget,
    Field,
    GraphLayer,
    GraphSample,
    GroupBy,
    MapValues,
    NeighborhoodSpec,
    Reduce,
    RowFilter,
    Rows,
    RowTarget,
    SampleRequest,
    SampleSpec,
    TargetValue,
    TimeWindow,
)
from kumo.pytorch.contract.schema import (
    NON_TEMPORAL,
    DataSchema,
    NonTemporal,
    RelationSpec,
    TableSpec,
)
from kumo.pytorch.dataset import Dataset, InMemoryRows, Parquet
from kumo.pytorch.engine import (
    AnchorTask,
    Engine,
    SampleSource,
    available_engines,
    engine_from_name,
    register_engine,
)
from kumo.pytorch.engines import DiskGraphEngine, InMemoryEngine
from kumo.pytorch.engines.unsupported import Unsupported
from kumo.pytorch.preparation import Plan, check, prepare, preview
from kumo.pytorch.serving import ServingAssembler, ServingRowsSpec
from kumo.pytorch.spec import DataLoaderSpec, ValidationError, ValidationIssue
from kumo.pytorch.task import Task, TaskBatches

__all__ = [
    *_batch_api,
    "ALL_ANCHORS",
    "AllAnchors",
    "AnchorTable",
    "CandidateColumn",
    "padded_positions",
    "shard_positions",
    "DType",
    "InputExposureWarning",
    "ActiveInterval",
    "Aggregate",
    "BuiltSample",
    "CalendarPart",
    "CandidateRequest",
    "CandidateSetSpec",
    "Compare",
    "Constant",
    "DatedEntityValue",
    "Derived",
    "EntitySpec",
    "EventSpec",
    "EventTimeDelta",
    "ExternalTarget",
    "Field",
    "GraphLayer",
    "GraphSample",
    "GroupBy",
    "MapValues",
    "NeighborhoodSpec",
    "Reduce",
    "RowFilter",
    "Rows",
    "RowTarget",
    "SampleRequest",
    "SampleSpec",
    "TargetValue",
    "TimeWindow",
    "NON_TEMPORAL",
    "DataSchema",
    "NonTemporal",
    "RelationSpec",
    "TableSpec",
    "Dataset",
    "InMemoryRows",
    "Parquet",
    "AnchorTask",
    "Engine",
    "SampleSource",
    "available_engines",
    "engine_from_name",
    "register_engine",
    "DiskGraphEngine",
    "Unsupported",
    "InMemoryEngine",
    "Plan",
    "check",
    "prepare",
    "preview",
    "DataLoaderSpec",
    "ValidationError",
    "ValidationIssue",
    "FitRecord",
    "FittedState",
    "Task",
    "TaskBatches",
    "PreparedBatchPlan",
    "PreviewResult",
    "prepare_batch_spec",
    "Artifact",
    "load_weights",
    "save_weights",
    "ServingAssembler",
    "ServingRowsSpec",
]
