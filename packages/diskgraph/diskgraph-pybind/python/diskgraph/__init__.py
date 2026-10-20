# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ._c import (
    export_tt_artifacts,
    convert_kumo_config,
    ingest,
    ExportConfig,
    FkeyConfig,
    TimeUnit,
    SamplerEngine,
    RuntimeBatchIterator,
    discover_tts,
    PySampleIterator,
    DiskGraphTTBuilder,
    EdgeRef,
    RowFilter,
    StageFilter,
    execute_tt_plan,
    execute_tt_gen,
    execute_tt_ingest,
    execute_bp_plan,
    drop_tt,
    read_tt_parquet_batches,
    DiskgraphDataError,
    DiskgraphSourceValidationError,
    DiskgraphTargetCategoryLimitError,
    DiskgraphConfigError,
    DiskgraphSystemError,
    DiskgraphCorruptionError,
    DiskManager,
    estimate_ingest_size,
    read_col_stats,
    read_edge_stats,
    read_index_metadata,
    read_success_marker,
    write_success_marker,
    touch_success_marker,
    verify_snapshot_integrity,
    has_resumable_ingest,
    validated_index_resume_progress,
    format_versions,
    compute_col_stats_for_parquet,
    resolve_temporal,
    detect_multicat_delimiter,
    multicat_delimiter_is_proven,
    TIMESTAMP_FORMATS,
    MULTICAT_DELIMITERS,
    CUSTOM_TT_CONTRACT,
)
from ._exceptions import DiskgraphTaskError

# Lazy imports for modules that have external dependencies (kumoapi)
def __getattr__(name):
    if name == "DiskGraphPlanCompiler":
        from .compiler import DiskGraphPlanCompiler
        return DiskGraphPlanCompiler
    if name == "SplitSpec":
        from .split_contract import SplitSpec
        return SplitSpec
    if name == "generate_training_table":
        from .executor import generate_training_table
        return generate_training_table
    if name == "DiskgraphConfigConverter":
        from .config_converter import DiskgraphConfigConverter
        return DiskgraphConfigConverter
    if name == "build_prediction_table_edges_artifact":
        from .prediction_artifacts import build_prediction_table_edges_artifact
        return build_prediction_table_edges_artifact
    if name == "parse_prediction_table_edges_artifact":
        from .prediction_artifacts import parse_prediction_table_edges_artifact
        return parse_prediction_table_edges_artifact
    if name == "validate_prediction_table_edges_artifact":
        from .prediction_artifacts import validate_prediction_table_edges_artifact
        return validate_prediction_table_edges_artifact
    if name == "build_prediction_table_artifact_bundle":
        from .prediction_artifacts import build_prediction_table_artifact_bundle
        return build_prediction_table_artifact_bundle
    if name == "build_train_table_artifacts_for_existing_table":
        from .tt_artifacts import build_train_table_artifacts_for_existing_table
        return build_train_table_artifacts_for_existing_table
    if name == "build_prediction_table_artifacts_for_existing_table":
        from .tt_artifacts import (
            build_prediction_table_artifacts_for_existing_table,
        )
        return build_prediction_table_artifacts_for_existing_table

    _TD_ARTIFACTS = {
        "FeatureSchemaInputs",                          # feature_schema.json input shape
        "GraphRuntimeMetadataInputs",                   # graph_runtime_metadata.json input shape
        "SplitInfoInputs",                              # split_info.json input shape
        "TaskDatasetMetadataInputs",                    # metadata.json input shape
        "TrainTableColumnMappingInputs",                # seed-id / target / time / weight column names
        "build_node_prediction_task_dataset_bundle",    # bundle-writer entry point
        "task_dataset_artifact_root",                   # canonical artifact-root resolver
        "TASK_DATASET_SCHEMA_VERSION",                  # bundle schema version constant
        "TASK_DATASET_IMPLEMENTATION_KIND",             # mock/dev implementation marker
    }
    _TD_REOPEN = {
        "RuntimeIndex",  # reusable opened index; iterate many epochs/shards from one open
        "task_stage_size",              # stage-size query from runtime manifest
    }
    if name in _TD_ARTIFACTS:
        from . import task_dataset_artifacts as m
        return getattr(m, name)
    if name in _TD_REOPEN:
        from . import runtime_batch_iterator as m
        return getattr(m, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "export_tt_artifacts",
    "convert_kumo_config",
    "ingest",
    "ExportConfig",
    "FkeyConfig",
    "TimeUnit",
    "SamplerEngine",
    "RuntimeBatchIterator",
    "discover_tts",
    "PySampleIterator",
    "DiskGraphTTBuilder",
    "EdgeRef",
    "RowFilter",
    "StageFilter",
    "DiskgraphDataError",
    "DiskgraphSourceValidationError",
    "DiskgraphTargetCategoryLimitError",
    "DiskgraphConfigError",
    "DiskgraphSystemError",
    "DiskgraphCorruptionError",
    "DiskManager",
    "estimate_ingest_size",
    "read_col_stats",
    "read_edge_stats",
    "read_index_metadata",
    "read_success_marker",
    "write_success_marker",
    "touch_success_marker",
    "verify_snapshot_integrity",
    "has_resumable_ingest",
    "validated_index_resume_progress",
    "format_versions",
    "compute_col_stats_for_parquet",
    "DiskGraphPlanCompiler",
    "SplitSpec",
    "DiskgraphConfigConverter",
    "execute_tt_plan",
    "execute_tt_gen",
    "execute_tt_ingest",
    "execute_bp_plan",
    "drop_tt",
    "read_tt_parquet_batches",
    "generate_training_table",
    "build_prediction_table_edges_artifact",
    "parse_prediction_table_edges_artifact",
    "validate_prediction_table_edges_artifact",
    "build_prediction_table_artifact_bundle",
    "build_train_table_artifacts_for_existing_table",
    "build_prediction_table_artifacts_for_existing_table",
    # Task-dataset rework surface
    "FeatureSchemaInputs",
    "GraphRuntimeMetadataInputs",
    "SplitInfoInputs",
    "TaskDatasetMetadataInputs",
    "TrainTableColumnMappingInputs",
    "build_node_prediction_task_dataset_bundle",
    "task_dataset_artifact_root",
    "TASK_DATASET_SCHEMA_VERSION",
    "TASK_DATASET_IMPLEMENTATION_KIND",
    "RuntimeIndex",
    "task_stage_size",
    "DiskgraphTaskError",
    "DiskgraphCorruptionError",
    "verify_snapshot_integrity",
    "CUSTOM_TT_CONTRACT",
]
