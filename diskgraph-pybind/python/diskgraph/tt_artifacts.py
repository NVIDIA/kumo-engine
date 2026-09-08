"""High-level entry point for rehydrating TT artifacts from a customer-
supplied training-table parquet (the custom-TT path).

Exposes `build_train_table_artifacts_for_existing_table` matching the
signature kumo's `ensure_diskgraph_native_artifacts_for_train_table`
calls into. Marshals the SplitSpec dataclass to a JSON-serialized plan
SplitSpec shape and dispatches into the Rust pyo3 entry point at
`_c.build_train_table_artifacts_for_existing_table`.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from ._c import (  # type: ignore[import]
    DiskgraphConfigError,
    DiskgraphDataError,
    build_prediction_table_artifacts_for_existing_table as _native_build_predict,
    build_split_spec_json as _native_build_split_spec_json,
    build_train_table_artifacts_for_existing_table as _native_build,
)
from .split_contract import SplitSpec


def _split_spec_to_plan_json(
    split_spec: Optional[SplitSpec],
    *,
    split_family_hint: Optional[str],
) -> Optional[str]:
    """Serialize a Python `SplitSpec` to the JSON the Rust executor consumes
    (a `SplitSpec<Seconds>` from `diskgraph-core`), for
    `build_train_table_artifacts_for_existing_table`'s `split_spec_json`.

    Returns None when split_spec is None (no split path).

    The marshalling itself lives in Rust -- `make_split_spec` in
    `diskgraph-pybind/src/tt_builder.rs`, which extracts the dataclass
    field-by-field and is shared with the generated path's
    `DiskGraphTTBuilder.split_ratio`. That sharing is the point: the unit rules
    are asymmetric enough to invite drift (`offsets` must be pre-scaled into
    base units, `timeframe`/`anchor_start_offset`/`timeframe_step` must not be,
    months and years route to `month_offsets` instead, and only DateOffset /
    TimeRange splits carry unit fields at all), and a second copy of them here
    did drift.

    The only thing this function still does to the spec is resolve `kind`
    against the caller's family hint.
    """
    if split_spec is None:
        return None
    if not isinstance(split_spec, SplitSpec):
        raise DiskgraphDataError(
            'build_train_table_artifacts_for_existing_table requires '
            f'split_config as SplitSpec, got {type(split_spec).__name__}')
    if split_family_hint and split_spec.kind \
            and split_family_hint != split_spec.kind:
        raise DiskgraphConfigError(
            'diskgraph split contract mismatch: '
            f'split_family_hint={split_family_hint} conflicts with '
            f'split_config.kind={split_spec.kind}')
    resolved_family = split_spec.kind or split_family_hint
    if resolved_family != split_spec.kind:
        split_spec = dataclasses.replace(split_spec, kind=resolved_family)
    return _native_build_split_spec_json(split_spec)


def _resolve_common_config(
    *,
    fn_name: str,
    tt_id: str,
    index_path: Optional[str],
    train_table_col_stats_config: Optional[dict],
    split_stats_config: Optional[dict],
) -> tuple:
    """Validate + extract the config shared by the train and predict custom-TT
    entries: ``tt_id``, ``index_path``, ``target_stype`` (REQUIRED),
    ``weight_col`` (optional), and the entity-table metadata
    (``entity_table``/``entity_pk_column``, REQUIRED).

    Returns ``(target_stype, weight_col, entity_table_name, entity_pk_column)``.
    Raises ``DiskgraphConfigError`` on any missing required field.
    """
    if not tt_id or not isinstance(tt_id, str):
        raise DiskgraphConfigError(
            f'{fn_name} requires tt_id (the published extension subdir name)')
    if index_path is None or not isinstance(index_path, str):
        raise DiskgraphConfigError(
            f'{fn_name} requires index_path (the diskgraph index directory)')

    target_stype = None
    if isinstance(train_table_col_stats_config, dict):
        target_stype = train_table_col_stats_config.get('target_stype')
    if not target_stype or not isinstance(target_stype, str):
        raise DiskgraphConfigError(
            f"{fn_name} requires train_table_col_stats_config['target_stype']")

    weight_col: Optional[str] = None
    if isinstance(train_table_col_stats_config, dict):
        wc = train_table_col_stats_config.get('weight_col')
        if isinstance(wc, str) and wc:
            weight_col = wc

    entity_table_name: Optional[str] = None
    entity_pk_column: Optional[str] = None
    if isinstance(split_stats_config, dict):
        et = split_stats_config.get('entity_table')
        if isinstance(et, str) and et:
            entity_table_name = et
        epk = split_stats_config.get('entity_pk_column')
        if isinstance(epk, str) and epk:
            entity_pk_column = epk
    if not entity_table_name:
        raise DiskgraphConfigError(
            f"{fn_name} requires split_stats_config['entity_table'] (the "
            'entity-table name in the diskgraph index)')
    if not entity_pk_column:
        raise DiskgraphConfigError(
            f"{fn_name} requires split_stats_config['entity_pk_column'] (the "
            'PK column name in the entity table)')

    return target_stype, weight_col, entity_table_name, entity_pk_column


def build_train_table_artifacts_for_existing_table(
    *,
    output_path: str,
    split_config: SplitSpec,
    input_timestamp_unit: str,
    tt_id: str,
    split_family_hint: Optional[str] = None,
    split_stats_config: Optional[dict] = None,
    train_table_col_stats_config: Optional[dict] = None,
    train_table_edges_config: Optional[dict] = None,
    index_path: Optional[str] = None,
    producer_summary_json: Optional[str] = None,
    target_entity_table: Optional[str] = None,
    task_type: Optional[str] = None,
    lp_rhs_is_temporal: bool = False,
) -> dict:
    """Rebuild TT artifacts (manifest + sidecars) for an existing
    training-table parquet.

    Args:
        output_path: Path to the customer parquet (after the kumo bridge
            normalizes ENTITY → ENTITY1 / TARGET → TARGET1).
        split_config: SplitSpec describing the desired train/val/test
            split.
        split_family_hint: Optional split family name; must agree with
            `split_config.kind` if both are set.
        split_stats_config: Free-form dict; today consumed for
            `entity_table` and `entity_pk_column` (the entity-table
            metadata kumo's pquery exposes).
        train_table_col_stats_config: Free-form dict; today consumed for
            `target_stype` (REQUIRED) and optional `weight_col`.
        train_table_edges_config: Edge bundle config (kumo's
            transitional GraphStore bridge handles this separately;
            ignored here).
        index_path: REQUIRED — path to the diskgraph index dir.
        input_timestamp_unit: REQUIRED — the unit of any Int64 TIMESTAMP
            column in `output_path`'s parquet. Typed Arrow Timestamp(unit)
            arrays self-describe and ignore this; Int64 columns use it
            verbatim. Kumo's rehydration sets this to "nanoseconds" (its
            canonical TIMESTAMP representation). Accepted: "nanoseconds",
            "microseconds", "milliseconds", "seconds".
        tt_id: REQUIRED — carried into the durable ``tt_ingest_spec.json``
            so the separate ``execute_tt_ingest`` job can later publish
            ``<index_path>/train_tables/<tt_id>/``. Kumo passes its
            ``ttng_tt_id``. This gen-only entry does NOT publish an
            extension itself.
        producer_summary_json: Optional kumo producer summary; today
            ignored.
        task_type: Optional Kumo task type. Forecasting CTTs use this to
            preserve precomputed sequence columns through the native builder.
        lp_rhs_is_temporal: Whether LP target/RHS table has time/end-time
            columns. Temporal-RHS LP CTT ingest uses this to emit per-stage
            timestamp bucket indexes, matching the native TT-gen path.

    Returns:
        dict describing the durable GEN-ONLY bundle (mirrors
        ``execute_tt_gen``): ``staging_dir`` (the kept bundle dir kumo
        persists + re-ingests via the normal ``execute_tt_ingest``),
        ``tt_parquet_path``, ``ingest_spec_path``, ``row_count``,
        ``output_path``, ``artifacts_dir``, ``split_counts``,
        ``split_meta``. No extension is published here.
    """
    # Entity-table metadata is required because the Rust executor needs to
    # validate the customer parquet's ENTITY1 PKs against the index. The
    # kumo caller passes these via split_stats_config (the existing
    # free-form dict already in the contract).
    (target_stype, weight_col, entity_table_name,
     entity_pk_column) = _resolve_common_config(
        fn_name='build_train_table_artifacts_for_existing_table',
        tt_id=tt_id,
        index_path=index_path,
        train_table_col_stats_config=train_table_col_stats_config,
        split_stats_config=split_stats_config,
    )

    plan_split_json = _split_spec_to_plan_json(
        split_config, split_family_hint=split_family_hint)

    # The producer_summary_json + train_table_edges_config kwargs are
    # currently consumed elsewhere (kumo's separate edge-bundle pass).
    # Suppress unused-warnings.
    _ = producer_summary_json
    _ = train_table_edges_config

    return _native_build(
        output_path=output_path,
        index_path=index_path,
        tt_id=tt_id,
        entity_table_name=entity_table_name,
        entity_pk_column=entity_pk_column,
        target_stype=target_stype,
        input_timestamp_unit=input_timestamp_unit,
        split_spec_json=plan_split_json,
        weight_col=weight_col,
        target_entity_table=target_entity_table,
        task_type=task_type,
        lp_rhs_is_temporal=bool(lp_rhs_is_temporal),
    )


def build_prediction_table_artifacts_for_existing_table(
    *,
    output_path: str,
    input_timestamp_unit: str,
    tt_id: str,
    split_stats_config: Optional[dict] = None,
    train_table_col_stats_config: Optional[dict] = None,
    train_table_edges_config: Optional[dict] = None,
    index_path: Optional[str] = None,
    producer_summary_json: Optional[str] = None,
    target_entity_table: Optional[str] = None,
    task_type: Optional[str] = None,
) -> dict:
    """Rebuild TT artifacts (manifest + sidecars) for a customer PREDICTION
    parquet (ENTITY + TIMESTAMP, NO target).

    Mirrors :func:`build_train_table_artifacts_for_existing_table` but takes no
    ``split_config``: the prediction bundle is a single ``predict`` stage
    (``SplitMode::None``) built with ``is_prediction=true`` (the customer TARGET
    is not consumed).

    Args:
        output_path: Path to the customer prediction parquet (after the kumo
            bridge normalizes ENTITY → ENTITY1). No target column is required.
        input_timestamp_unit: REQUIRED — unit of any Int64 TIMESTAMP column
            (see the train entry for the accepted values).
        tt_id: REQUIRED — the gen-side id carried into
            ``tt_ingest_spec.json``. Predict fans out: the later
            ``execute_tt_ingest`` job may publish under a per-BP-job TARGET id
            that differs from this gen ``tt_id``. This gen-only entry does NOT
            publish an extension.
        split_stats_config: consumed for ``entity_table`` +
            ``entity_pk_column`` (as in the train entry).
        train_table_col_stats_config: consumed for ``target_stype`` (REQUIRED)
            and optional ``weight_col``.
        index_path: REQUIRED — path to the diskgraph index dir.

    Returns:
        The SAME durable GEN-ONLY bundle dict shape as the train entry
        (``staging_dir``/``tt_parquet_path``/``ingest_spec_path``/
        ``row_count``/``output_path``/``artifacts_dir``/``split_counts``/
        ``split_meta``), with a single ``predict`` stage.
    """
    (target_stype, weight_col, entity_table_name,
     entity_pk_column) = _resolve_common_config(
        fn_name='build_prediction_table_artifacts_for_existing_table',
        tt_id=tt_id,
        index_path=index_path,
        train_table_col_stats_config=train_table_col_stats_config,
        split_stats_config=split_stats_config,
    )

    # As on the train path, these kwargs are consumed elsewhere (kumo's
    # separate edge-bundle pass); suppress unused-warnings.
    _ = producer_summary_json
    _ = train_table_edges_config

    return _native_build_predict(
        output_path=output_path,
        index_path=index_path,
        tt_id=tt_id,
        entity_table_name=entity_table_name,
        entity_pk_column=entity_pk_column,
        target_stype=target_stype,
        input_timestamp_unit=input_timestamp_unit,
        weight_col=weight_col,
        target_entity_table=target_entity_table,
        task_type=task_type,
    )


__all__ = [
    'build_prediction_table_artifacts_for_existing_table',
    'build_train_table_artifacts_for_existing_table',
]
