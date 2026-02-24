from ._c import (
    SamplerEngine,
    PySampleIterator,
    DiskGraphTTBuilder,
    EdgeRef,
    RowFilter,
    StageFilter,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SamplerEngine",
    "PySampleIterator",
    "DiskGraphTTBuilder",
    "EdgeRef",
    "RowFilter",
    "StageFilter",
    "DiskGraphPlanCompiler",
    "execute_tt_plan",
    "generate_training_table",
]
