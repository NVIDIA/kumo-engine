# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convenience wrapper for the TT-gen executor.

TT-NG writes artifacts and DG-native stage tables under
``index_path/train_tables/<tt_id>/`` and returns a summary dict.

Usage::

    from diskgraph.executor import generate_training_table

    summary = generate_training_table(plan_json, index_path, tt_id)
    # summary["tt_parquet_path"], summary["ingested_stage_tables"], ...
"""

from typing import Any, Dict

from ._c import execute_tt_plan


def generate_training_table(
    plan_json: str,
    index_path: str,
    tt_id: str,
) -> Dict[str, Any]:
    """Execute a TT-gen plan against a DiskGraph index.

    Args:
        plan_json: JSON-serialised ``DiskGraphTTPlan`` (from
            ``DiskGraphTTBuilder.build()`` or ``DiskGraphPlanCompiler``).
        index_path: Path to the DiskGraph index directory.
        tt_id: Stable, filename-safe training-table identifier. Artifacts are
            written under ``index_path/train_tables/<tt_id>/``.

    Returns:
        A dict containing TT artifact paths, stage parquet counts/paths, and
        DG-native ``ingested_stage_tables`` refs.
    """
    return execute_tt_plan(plan_json, index_path, tt_id)
