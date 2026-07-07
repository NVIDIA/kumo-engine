"""High-level entry point for ingesting a batch-prediction stage table
(LHS `predict` or RHS `rhs_predict`) as a DiskGraph extension.

Mirrors the `tt_artifacts.py` style: marshals plain Python args and
dispatches into the Rust pyo3 entry point at `_c.ingest_prediction_stage`,
which delegates to the source-agnostic `diskgraph_ingest::ingest_stage_extension`
primitive. The stage parquet is ingested under `extension_output_root` and its
`entity_src_col` is resolved against the parent index's `dst_table` primary key.
"""
from __future__ import annotations

from ._c import (  # type: ignore[import]
    DiskgraphConfigError,
    ingest_prediction_stage as _native_ingest_prediction_stage,
)

# Stage table names for the two batch-prediction sides.
LHS_STAGE = 'predict'
RHS_STAGE = 'rhs_predict'

_VALID_UNITS = {'seconds', 'millis', 'micros', 'nanos',
                'milliseconds', 'microseconds', 'nanoseconds',
                'second', 's', 'ms', 'us', 'ns'}


def ingest_prediction_stage(
    *,
    index_path: str,
    extension_output_root: str,
    stage_table_name: str,
    parquet_path: str,
    entity_src_col: str,
    dst_table: str,
    input_timestamp_unit: str = 'nanos',
) -> int:
    """Ingest one pre-materialized prediction-stage parquet as a DG extension.

    Args:
        index_path: Path to the parent DiskGraph index directory.
        extension_output_root: Output dir for the produced extension
            (e.g. ``<index>/pred_tables/<job_id>`` for LHS or
            ``<index>/pred_tables/<job_id>_rhs`` for RHS).
        stage_table_name: On-disk stage table name — typically ``predict``
            (LHS) or ``rhs_predict`` (RHS). See :data:`LHS_STAGE` /
            :data:`RHS_STAGE`.
        parquet_path: Path to the pre-materialized stage parquet
            (ENTITY + TIMESTAMP, no target).
        entity_src_col: Column in the stage parquet that references the
            parent entity (resolved via an external_parent FK).
        dst_table: Parent table name the FK points at (must have a PK).
        input_timestamp_unit: Unit assigned to a TIMESTAMP column stored as
            a plain integer (typed Arrow Timestamp arrays self-describe).
            Accepted: ``seconds`` | ``millis`` | ``micros`` | ``nanos``.

    Returns:
        The emitted row count for the stage (read back from the produced
        extension's ``node_count``).
    """
    for name, value in (
        ('index_path', index_path),
        ('extension_output_root', extension_output_root),
        ('stage_table_name', stage_table_name),
        ('parquet_path', parquet_path),
        ('entity_src_col', entity_src_col),
        ('dst_table', dst_table),
    ):
        if not value or not isinstance(value, str):
            raise DiskgraphConfigError(
                f'ingest_prediction_stage requires a non-empty str {name}')
    if input_timestamp_unit.lower() not in _VALID_UNITS:
        raise DiskgraphConfigError(
            'ingest_prediction_stage: input_timestamp_unit must be one of '
            f'seconds|millis|micros|nanos, got {input_timestamp_unit!r}')

    return _native_ingest_prediction_stage(
        index_path=index_path,
        extension_output_root=extension_output_root,
        stage_table_name=stage_table_name,
        parquet_path=parquet_path,
        entity_src_col=entity_src_col,
        dst_table=dst_table,
        input_timestamp_unit=input_timestamp_unit,
    )


__all__ = ['ingest_prediction_stage', 'LHS_STAGE', 'RHS_STAGE']
