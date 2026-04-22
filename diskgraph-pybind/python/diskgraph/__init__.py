from ._c import (
    export_tt_artifacts,
    convert_kumo_config,
    ingest,
    ExportConfig,
    FkeyConfig,
    TimeUnit,
    SamplerEngine,
    PySampleIterator,
    DiskGraphTTBuilder,
    EdgeRef,
    RowFilter,
    StageFilter,
    TTPlanIterator,
    execute_tt_plan,
    DiskgraphDataError,
    DiskgraphConfigError,
    DiskgraphSystemError,
    DiskManager,
    estimate_ingest_size,
    read_success_marker,
    write_success_marker,
    touch_success_marker,
)

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
    if name == "compute_dense_split_assignments":
        from .split_assignments import compute_dense_split_assignments
        return compute_dense_split_assignments
    if name == "execute_tt_plan_with_artifacts":
        from .tt_artifacts import execute_tt_plan_with_artifacts
        return execute_tt_plan_with_artifacts
    if name == "build_train_table_artifacts_for_existing_table":
        from .tt_artifacts import build_train_table_artifacts_for_existing_table
        return build_train_table_artifacts_for_existing_table
    if name == "build_prediction_table_edges_artifact":
        from .prediction_artifacts import build_prediction_table_edges_artifact
        return build_prediction_table_edges_artifact
    if name == "parse_prediction_table_edges_artifact":
        from .prediction_artifacts import parse_prediction_table_edges_artifact
        return parse_prediction_table_edges_artifact
    if name == "validate_prediction_table_edges_artifact":
        from .prediction_artifacts import validate_prediction_table_edges_artifact
        return validate_prediction_table_edges_artifact
    if name == "build_prediction_table_col_stats_artifact":
        from .prediction_artifacts import build_prediction_table_col_stats_artifact
        return build_prediction_table_col_stats_artifact
    if name == "parse_prediction_table_col_stats_artifact":
        from .prediction_artifacts import parse_prediction_table_col_stats_artifact
        return parse_prediction_table_col_stats_artifact
    if name == "validate_prediction_table_col_stats_artifact":
        from .prediction_artifacts import validate_prediction_table_col_stats_artifact
        return validate_prediction_table_col_stats_artifact
    if name == "build_prediction_table_artifact_bundle":
        from .prediction_artifacts import build_prediction_table_artifact_bundle
        return build_prediction_table_artifact_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "export_tt_artifacts",
    "convert_kumo_config",
    "ingest",
    "ExportConfig",
    "FkeyConfig",
    "TimeUnit",
    "SamplerEngine",
    "PySampleIterator",
    "DiskGraphTTBuilder",
    "EdgeRef",
    "RowFilter",
    "StageFilter",
    "TTPlanIterator",
    "DiskgraphDataError",
    "DiskgraphConfigError",
    "DiskgraphSystemError",
    "DiskManager",
    "estimate_ingest_size",
    "read_success_marker",
    "write_success_marker",
    "touch_success_marker",
    "DiskGraphPlanCompiler",
    "SplitSpec",
    "DiskgraphConfigConverter",
    "execute_tt_plan",
    "generate_training_table",
    "compute_dense_split_assignments",
    "execute_tt_plan_with_artifacts",
    "build_train_table_artifacts_for_existing_table",
    "build_prediction_table_edges_artifact",
    "parse_prediction_table_edges_artifact",
    "validate_prediction_table_edges_artifact",
    "build_prediction_table_col_stats_artifact",
    "parse_prediction_table_col_stats_artifact",
    "validate_prediction_table_col_stats_artifact",
    "build_prediction_table_artifact_bundle",
]
