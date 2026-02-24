"""Convenience wrapper for the TT-gen executor.

Usage::

    from diskgraph.executor import generate_training_table

    shard_paths = generate_training_table(plan_json, index_path, output_dir)
"""

from typing import List

from ._c import execute_tt_plan


def generate_training_table(
    plan_json: str,
    index_path: str,
    output_dir: str,
) -> List[str]:
    """Execute a TT-gen plan against a DiskGraph index.

    Args:
        plan_json: JSON-serialised ``DiskGraphTTPlan`` (from
            ``DiskGraphTTBuilder.build()`` or ``DiskGraphPlanCompiler``).
        index_path: Path to the DiskGraph index directory.
        output_dir: Directory where output Parquet shards will be written.
            Created automatically if it does not exist.

    Returns:
        List of output shard file paths (one per rayon thread).
    """
    return execute_tt_plan(plan_json, index_path, output_dir)
