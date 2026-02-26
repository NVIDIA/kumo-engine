"""Convenience wrapper for the TT-gen executor.

Usage::

    from diskgraph.executor import generate_training_table

    for batch in generate_training_table(plan_json, index_path):
        # batch is a pyarrow.RecordBatch
        ...
"""

from typing import Iterator

import pyarrow as pa

from ._c import execute_tt_plan


def generate_training_table(
    plan_json: str,
    index_path: str,
) -> Iterator[pa.RecordBatch]:
    """Execute a TT-gen plan against a DiskGraph index.

    Returns a streaming iterator of ``pyarrow.RecordBatch`` objects.
    Each batch contains ~64K rows with columns: ENTITY, TIMESTAMP,
    TARGET, and optionally FREQUENCY / ENTITY_INCLUDE_BIT.

    Args:
        plan_json: JSON-serialised ``DiskGraphTTPlan`` (from
            ``DiskGraphTTBuilder.build()`` or ``DiskGraphPlanCompiler``).
        index_path: Path to the DiskGraph index directory.

    Yields:
        ``pyarrow.RecordBatch`` objects as they are produced by the
        Rust executor's rayon workers.
    """
    return execute_tt_plan(plan_json, index_path)
