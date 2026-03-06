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
)

# Lazy imports for modules that have external dependencies (kumoapi)
def __getattr__(name):
    if name == "DiskGraphPlanCompiler":
        from .compiler import DiskGraphPlanCompiler
        return DiskGraphPlanCompiler
    if name == "generate_training_table":
        from .executor import generate_training_table
        return generate_training_table
    if name == "DiskgraphConfigConverter":
        from .config_converter import DiskgraphConfigConverter
        return DiskgraphConfigConverter
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
    "DiskGraphPlanCompiler",
    "DiskgraphConfigConverter",
    "execute_tt_plan",
    "generate_training_table",
]
