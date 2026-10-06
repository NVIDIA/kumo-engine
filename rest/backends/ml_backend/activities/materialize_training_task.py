# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os.path as osp
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Tuple

from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.job_config import JobConfig
from kumo_core.model_plan import ModelPlan, _is_inferred
from kumo_core.types import ColStatType, TaskType
from temporalio import activity

from io_utils import fsutil, logging
from io_utils.exceptions import catch_and_log
from io_utils.json import save_json
from kumo.data import get_active_cache
from kumo.data.cache.exact_stats import ExactStats
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.graph.column import EncoderGenerator
from kumo.graph.table_data import TableData
from kumo.home import HomeDirManager
from kumo.pquery.predictive_query import (
    TRAIN_TABLE_NAME,
    PredictiveQuery,
    PredictiveQueryV2,
)
from kumo.task.splits.parsing import get_split
from kumo.train.utils.config_utils import save_trainer_job_config
from kumo.validation import ValidationResponse
from rest.backends.data_backend.utils import get_data_operation_description
from rest.backends.ml_backend import validation
from rest.backends.ml_backend.activities.common import (
    estimate_input_data_stats,
    make_materialization_info,
    save_input_data_stats_to_state_db,
)
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.timing import kumo_timing
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import TrainingSpec
from rest.utils.driver_common import (
    hydrate_predictive_query_for_activity,
    open_connectors,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import TrainingJobID
from statedb.config.diskgraph import get_diskgraph_index_path_for_job
from statedb.utils.errors import (
    DiskgraphConfigError,
    DiskgraphDataError,
    DiskgraphError,
    DiskgraphSystemError,
    InvalidGraphError,
    TrainingDataMaterializationError,
    TrainingJobOutputError,
)
from temporalio_commons.error_utils import is_error_non_retryable

from ...data_backend import connector_activity_registry
from ..ml_job_output_store import save_materialization_info

log = logging.getLogger(__name__)

# DiskGraph's pybind exceptions are optional on hosts without the wheel.
# Filtering unavailable classes keeps this module importable there; ``except
# ()`` simply matches nothing. Data/config failures are deterministic, while a
# system failure can change when Temporal retries the activity.
_DISKGRAPH_NON_RETRYABLE_ERRORS = tuple(
    error_type for error_type in (DiskgraphDataError, DiskgraphConfigError)
    if error_type is not None)
_DISKGRAPH_RETRYABLE_ERRORS = tuple(
    error_type for error_type in (DiskgraphSystemError, )
    if error_type is not None)
_DISKGRAPH_NATIVE_ERRORS = (_DISKGRAPH_NON_RETRYABLE_ERRORS
                            + _DISKGRAPH_RETRYABLE_ERRORS)


def _configure_forecasting_sequence_columns(
    task: Any,
    year_over_year: bool,
    include_future_labels: bool = False,
) -> None:
    if task.task_type == TaskType.FORECASTING:
        task._set_forecast_col_encoder(  # pylint: disable=protected-access
            year_over_year, include_future_labels=include_future_labels)


_TTNG_NODE_PREDICTION_TASK_TYPES = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
    TaskType.MULTILABEL_RANKING,
    TaskType.REGRESSION,
}
_TTNG_LINK_PREDICTION_TASK_TYPES = {
    TaskType.STATIC_LINK_PREDICTION,
    TaskType.TEMPORAL_LINK_PREDICTION,
}
_TTNG_FORECASTING_TASK_TYPES = {
    TaskType.FORECASTING,
}
_TTNG_TASK_DATASET_TASK_TYPES = (_TTNG_NODE_PREDICTION_TASK_TYPES
                                 | _TTNG_LINK_PREDICTION_TASK_TYPES
                                 | _TTNG_FORECASTING_TASK_TYPES)


def _ttng_task_family(task_type: TaskType) -> str:
    from kumo.diskgraph._diskgraph_legacy import task_family
    return task_family(task_type)


def _ttng_task_is_temporal(task: Any) -> bool:
    from kumo.diskgraph._diskgraph_legacy import task_is_temporal
    return task_is_temporal(task)


def _build_ttng_feature_schema(
    task: Any,
    fitted_encoders: Any,
) -> dict[str, Any]:
    from kumo.diskgraph._diskgraph_legacy import build_feature_schema
    return build_feature_schema(task, fitted_encoders)


def _ttng_train_table_col_stats_sidecar_path(
    train_table_path: str,
    diskgraph_index_path: str,
) -> str:
    """Return the published TT-NG column-stats sidecar path.

    The generator handoff bundle is private to ``execute_tt_ingest``. Kumo
    consumers read only the complete extension that ingest published.
    """
    tt_id = _ttng_train_table_tt_id(train_table_path)
    return osp.join(diskgraph_index_path, 'train_tables', tt_id,
                    'train_table_col_stats.json')


def _ttng_validate_train_table_col_stats_sidecar(
    task: Any,
    sidecar_path: str,
) -> None:
    """Fail when the sidecar omits exact target counts an encoder requires."""
    from kumo.diskgraph import diskgraph_native_tt_artifacts as tt_artifacts

    try:
        payload = tt_artifacts.parse_train_table_col_stats_artifact(
            sidecar_path)
    except Exception as e:
        raise TrainingDataMaterializationError(
            message=('TT-NG train-table column statistics are malformed at '
                     f'{sidecar_path!r}.'),
            error=e,
            non_retryable=True,
        ) from e

    physical_to_logical = {'TARGET1': 'TARGET'}
    available: dict[str, set[ColStatType]] = {}
    for stat_key in payload['stats']:
        parts = stat_key.rsplit('__', 3)
        if len(parts) != 4:
            continue
        column_name, _stype, stat_name, _approx = parts
        try:
            stat_type = ColStatType(stat_name)
        except ValueError:
            continue
        logical_name = physical_to_logical.get(column_name, column_name)
        available.setdefault(logical_name, set()).add(stat_type)

    target_col = task.train_table.target_col
    if target_col is None:
        return

    missing: list[str] = []
    logical_name = physical_to_logical.get(target_col.name, target_col.name)
    for stat_type in (
            ColStatType.CATEGORY_COUNTS,
            ColStatType.MULTI_CATEGORY_COUNTS,
    ):
        if (not stat_type.supports(target_col.dtype)
                or not stat_type.supports_stype(target_col.stype)):
            continue
        if stat_type not in available.get(logical_name, set()):
            missing.append(f'{target_col.name}: {stat_type.value}')

    if missing:
        raise TrainingDataMaterializationError(
            message=(
                'TT-NG train-table column statistics are incomplete at '
                f'{sidecar_path!r}; missing {missing}. The generated train '
                'table must be regenerated with a corrected DiskGraph build; '
                'retrying this training job cannot repair the artifact.'),
            non_retryable=True,
        )


def _ttng_prepare_train_table_col_stats(
    task: Any,
    train_table_path: str,
    diskgraph_index_path: str,
    ttng_tt_id: str,
) -> str | None:
    """Validate stats before re-ingest, then return the published sidecar."""
    published_sidecar = _ttng_train_table_col_stats_sidecar_path(
        train_table_path, diskgraph_index_path)

    # Link-prediction targets are graph node IDs and have no train-table target
    # encoder. New native LP bundles therefore omit this otherwise
    # cardinality-sized sidecar. Historical present sidecars remain accepted,
    # but still fail closed when malformed.
    if bool(getattr(task.task_type, 'is_link_pred', False)):
        if fsutil.exists(published_sidecar):
            _ttng_validate_train_table_col_stats_sidecar(
                task, published_sidecar)
        _ttng_ensure_train_table_extension(train_table_path,
                                           diskgraph_index_path, ttng_tt_id)
        if fsutil.exists(published_sidecar):
            _ttng_validate_train_table_col_stats_sidecar(
                task, published_sidecar)
            return published_sidecar
        return None

    preflight_sidecar = published_sidecar

    if not fsutil.exists(preflight_sidecar):
        from kumo.data.generic.diskgraph_train_table import (
            resolve_durable_ttng_bundle,
            ttng_bundle_dir_for_train_table,
        )
        try:
            durable_staging = resolve_durable_ttng_bundle(
                ttng_bundle_dir_for_train_table(train_table_path))
        except Exception as e:
            raise TrainingDataMaterializationError(
                message=('TT-NG could not resolve the durable train-table '
                         'artifact bundle before materialization.'),
                error=e,
                non_retryable=True,
            ) from e
        preflight_sidecar = osp.join(durable_staging,
                                     'train_table_col_stats.json')

    if not fsutil.exists(preflight_sidecar):
        raise TrainingDataMaterializationError(
            message=(
                'TT-NG train-table artifact is missing '
                'train_table_col_stats.json at both the published and '
                f'durable locations (last checked {preflight_sidecar!r}).'),
            non_retryable=True,
        )

    # This read is intentionally before the potentially expensive re-ingest.
    _ttng_validate_train_table_col_stats_sidecar(task, preflight_sidecar)
    _ttng_ensure_train_table_extension(train_table_path, diskgraph_index_path,
                                       ttng_tt_id)

    if not fsutil.exists(published_sidecar):
        raise TrainingDataMaterializationError(
            message=('TT-NG published extension is missing '
                     f'train_table_col_stats.json at {published_sidecar!r}; '
                     're-ingest the persisted staging bundle.'),
            non_retryable=True,
        )
    if published_sidecar != preflight_sidecar:
        _ttng_validate_train_table_col_stats_sidecar(task, published_sidecar)
    return published_sidecar


def _ttng_ensure_train_table_extension(
    train_table_path: str,
    diskgraph_index_path: str,
    ttng_tt_id: str,
) -> None:
    """Ensure the published ``train_tables/<tt_id>/`` extension exists locally.

    The index is keyed by GRAPH, not by job, so the gen and training pods
    resolve to the same index path -- but the per-job ``train_tables/<tt_id>/``
    extension is NOT part of the dataplane snapshot, so the snapshot restore on
    the training pod rebuilds the base index without it. The downstream
    ``materialize_task_dataset`` reads ``spec.tt_manifest_path`` from this
    extension, so re-ingest from the durable persisted staging bundle when it is
    absent (idempotent; mirrors the re-ingest in ``materialize_training_table``).
    """
    from kumo.data.generic.diskgraph_staging import (
        TTNGOrphanSkipError,
        ensure_stage_ingested,
    )
    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        ttng_bundle_dir_for_train_table,
    )

    # Delegate to the shared self-heal, which copies the durable bundle into a
    # pod-local staging dir before ingest; DG's std::fs cannot read the durable
    # (S3) path directly. Train tt_id is both source and target.
    staging_dir = ttng_bundle_dir_for_train_table(train_table_path)
    try:
        with kumo_timing(
                "training_materialize.ttng_reingest_train_table_extension"):
            ensure_stage_ingested(
                index_path=diskgraph_index_path,
                target_tt_id=ttng_tt_id,
                staging_dir=staging_dir,
                predict_stage_name=None,
            )
    except TTNGOrphanSkipError as e:
        # Not a bundle problem: the KUMO_TTNG_ORPHAN_FAIL_PCT threshold fired.
        # The message is self-contained — pass it through verbatim.
        raise TrainingDataMaterializationError(
            message=str(e),
            non_retryable=True,
        ) from e
    except _DISKGRAPH_NATIVE_ERRORS as e:
        # The activity interceptor preserves messages only for sanctioned
        # public exceptions. Convert Kumo-owned native failures here so the
        # actionable DiskGraph detail reaches the job status and API.
        raise DiskgraphError(
            message=('TT-NG: DiskGraph failed while re-ingesting the '
                     f'train_tables extension for tt_id={ttng_tt_id!r} from '
                     f'the staging bundle at {staging_dir!r}.'),
            error=e,
            non_retryable=isinstance(e, _DISKGRAPH_NON_RETRYABLE_ERRORS),
        ) from e
    except (TTNGBundleError, RuntimeError) as e:
        raise TrainingDataMaterializationError(
            message=(
                'TT-NG: train_tables extension is missing and the durable '
                f'staging bundle at {staging_dir!r} is missing/incomplete to '
                f're-ingest tt_id={ttng_tt_id!r}. The gen job must persist the '
                f'complete bundle before training materialize runs. Cause: {e}'
            ),
            non_retryable=True,
        ) from e


def _ensure_diskgraph_col_stats_in_cache(graph, job_input) -> None:
    """Ensure DiskGraph-computed col stats are present in the data backend cache.

    Fallback for cold-cache training paths (fresh activity subprocess /
    custom-TT jobs): the main ``diskgraph_ingest_stats_activity`` may
    not have populated every per-table cache entry. This seeds the
    remaining entries directly from the DG index. Idempotent;
    overwrites so the DG index stays authoritative.
    """
    index_path = get_diskgraph_index_path_for_job(job_input)
    from kumo.diskgraph import ingest_stats_cache
    ingest_stats_cache.seed_missing_col_stats_from_index(
        index_path, graph.tables)


def _infer_graph_encoders_from_cached_stats_or_fail(graph) -> None:
    """Compatibility helper for tests; prefer task-based helper in runtime."""
    cache = get_active_cache()

    encoder_stats: dict[str, ExactStats] = {}
    for table in graph.tables:
        cols = [col for col in table.cols if col.encoder is None]
        if not cols:
            continue

        loaded = cache.load_col_stats_cache(table.fingerprint(),
                                            table.data_version)
        if not loaded:
            raise RuntimeError(
                'diskgraph strict path requires precomputed col stats for '
                f"table '{table.name}' (version={table.data_version})")

        approx_request = {}
        exact_request = {}
        for col in cols:
            col_cfg = col.base_config()
            approx_types = [
                stat for stat in EncoderGenerator.REQUIRED_APPROX_STATS
                if stat.supports(col.dtype) and stat.supports_stype(col.stype)
            ]
            exact_types = [
                stat for stat in EncoderGenerator.REQUIRED_EXACT_STATS
                if stat.supports(col.dtype) and stat.supports_stype(col.stype)
            ]
            if approx_types:
                approx_request[col_cfg] = approx_types
            if exact_types:
                exact_request[col_cfg] = exact_types

        # DG-native: read stats straight from the cache populated by
        # DiskGraph ingest. The severed legacy factory has no
        # compute-on-miss path; a miss means ingest didn't run or the
        # stat type isn't emitted (caught later in per-col validation).
        approx_stats = cache.get_col_stats(
            table.fingerprint(),
            approx_request,
            data_version=table.data_version,
            approx=True,
        )
        exact_stats = cache.get_col_stats(
            table.fingerprint(),
            exact_request,
            data_version=table.data_version,
            approx=False,
        )

        per_col_stats = {}
        for col in cols:
            col_cfg = col.base_config()
            col_approx = approx_stats.get(col_cfg, {})
            col_exact = exact_stats.get(col_cfg, {})
            approx_required = len(approx_request.get(col_cfg, [])) > 0
            exact_required = len(exact_request.get(col_cfg, [])) > 0
            if (approx_required
                    or exact_required) and not (col_approx or col_exact):
                log.warning(
                    '[DG] Missing encoder stats for %s.%s: '
                    'approx_required=%s (types=%s), exact_required=%s (types=%s), '
                    'col_approx_keys=%s, col_exact_keys=%s, '
                    'col_cfg=%s, stype=%s, dtype=%s',
                    table.name,
                    col.name,
                    approx_required,
                    [
                        s.__class__.__name__
                        for s in approx_request.get(col_cfg, [])
                    ],
                    exact_required,
                    [
                        s.__class__.__name__
                        for s in exact_request.get(col_cfg, [])
                    ],
                    list(col_approx.keys()),
                    list(col_exact.keys()),
                    col_cfg,
                    col.stype,
                    col.dtype,
                )
                raise RuntimeError(
                    'diskgraph strict path requires precomputed encoder stats '
                    f"for column '{table.name}.{col.name}'")
            merged = {k: v.config() for k, v in col_approx.items()}
            merged.update({k: v.config() for k, v in col_exact.items()})
            per_col_stats[col.name] = merged

        encoder_stats[table.name] = ExactStats(table_stats={},
                                               col_stats=per_col_stats)

    graph.infer_encoders(encoder_stats=encoder_stats)


def _split_family_from_split_config(split_obj: Any) -> Optional[str]:
    if split_obj is None:
        return None
    cfg_fn = getattr(split_obj, 'config', None)
    if callable(cfg_fn):
        cfg = cfg_fn()
        if isinstance(cfg, dict):
            target = cfg.get('_target_')
            if isinstance(target, str) and target:
                return target.rsplit('.', 1)[-1]
        target = getattr(cfg, '_target_', None)
        if isinstance(target, str) and target:
            return target.rsplit('.', 1)[-1]
    return split_obj.__class__.__name__


def _coerce_split_config_for_diskgraph(split_obj: Any) -> Any:
    if split_obj is None or isinstance(split_obj, dict):
        return split_obj
    cfg_fn = getattr(split_obj, 'config', None)
    if callable(cfg_fn):
        cfg = cfg_fn()
        if is_dataclass(cfg):
            return asdict(cfg)
        if isinstance(cfg, dict):
            return cfg
    if is_dataclass(split_obj):
        return asdict(split_obj)
    return split_obj


def _custom_tt_split_col_name(job_input: Any) -> Optional[str]:
    """The split column the job's custom training table names, if any."""
    custom = getattr(getattr(job_input, 'spec', None), 'custom_train_table',
                     None)
    return getattr(custom, 'split_col_name', None)


def _custom_tt_split_override(
    split_obj: Any,
    split_column: Optional[str],
) -> tuple[Any, Optional[str]]:
    """Replace the plan's split with the one the supplied table carries.

    ``split_column`` is what the training-table specification names, per
    table. It used to come from a deployment-wide environment variable, which
    was the wrong scope for a per-table fact and meant a cloud-registered
    table naming its own split column had that name silently ignored.
    """
    split_column = (split_column or '').strip()
    if not split_column:
        return split_obj, _split_family_from_split_config(split_obj)

    from kumo.task.split import ManualSplit
    split = ManualSplit(
        col=split_column,
        train_value=0,
        val_value=1,
        test_value=2,
    )
    return split, 'ManualSplit'


def _ttng_train_table_tt_id(train_table_path: str) -> str:
    """Read the train ``tt_id`` from the selected bundle's ingest spec."""
    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        read_tt_id_from_staging_bundle,
        ttng_bundle_dir_for_train_table,
    )
    staging_dir = ttng_bundle_dir_for_train_table(train_table_path)
    try:
        return read_tt_id_from_staging_bundle(staging_dir)
    except TTNGBundleError as e:
        raise TrainingDataMaterializationError(message=str(e), error=e,
                                               non_retryable=True) from e


def _ttng_train_table_parse_target(
    train_table_path: str,
    *,
    is_custom_train_table: bool = False,
) -> tuple[str, Optional[str]]:
    """Return the physical dataset and its stable logical table name."""
    return train_table_path, ('tt' if is_custom_train_table else
                              TRAIN_TABLE_NAME)


def _bare_table_name(table_name: str) -> str:
    return table_name.split('.', 1)[1] if '.' in table_name else table_name


def _get_graph_table_by_bare_name(graph: Any, bare_table_name: str) -> Any:
    for table in getattr(graph, 'tables', []):
        candidates = {
            getattr(table, 'table_name', None),
            getattr(table, 'name', None),
            _bare_table_name(getattr(table, 'name', '') or ''),
        }
        if bare_table_name in candidates:
            return table
    raise TrainingDataMaterializationError(
        message=('TT-NG custom train table ingestion could not find graph '
                 f'table {bare_table_name!r}'),
        non_retryable=True,
    )


def _ttng_custom_tt_entity_metadata(task: Any) -> tuple[str, str]:
    train_table = task.train_table
    tt_fkeys = list(getattr(train_table, 'fkeys', None) or [])
    if not tt_fkeys:
        raise TrainingDataMaterializationError(
            message=('TT-NG custom train table ingestion requires '
                     'train_table.fkeys[0] to identify the seed entity table'),
            non_retryable=True,
        )

    dst_table = getattr(tt_fkeys[0], 'dst_table', None) or ''
    entity_table_name = _bare_table_name(dst_table)
    entity_table = _get_graph_table_by_bare_name(task.graph, entity_table_name)
    entity_pk_column = getattr(entity_table, 'pkey_name', None)
    if not entity_pk_column:
        raise TrainingDataMaterializationError(
            message=('TT-NG custom train table ingestion requires entity table '
                     f'{entity_table_name!r} to have a primary key'),
            non_retryable=True,
        )
    return entity_table_name, entity_pk_column


def _ttng_custom_tt_target_stype(pquery: PredictiveQuery) -> str:
    target_stype = getattr(pquery.validated_predictive_query, 'target_stype',
                           None)
    value = getattr(target_stype, 'value', None)
    if not isinstance(value, str) or not value:
        raise TrainingDataMaterializationError(
            message=('TT-NG custom train table ingestion requires '
                     'validated_predictive_query.target_stype'),
            non_retryable=True,
        )
    return value


def _build_ttng_custom_train_table_artifacts(
    *,
    train_table_path: str,
    split_config: Any,
    split_family: Optional[str],
    task: Any,
    pquery: PredictiveQuery,
    diskgraph_index_path: str,
    ttng_tt_id: str,
    time_col_name: str,
    weight_col_name: Optional[str],
) -> str:
    from diskgraph import (
        DiskgraphDataError,
        SplitSpec,
        build_train_table_artifacts_for_existing_table,
    )
    from kumo_core.pquery import static_link_fact
    from kumo_core.training.splits import uses_pseudo_time

    from kumo.data.generic.diskgraph_train_table import (
        TT_PARQUET_FILENAME,
        _build_split_spec,
        local_ttng_builder_input,
        persist_ttng_bundle,
        ttng_persisted_staging_dir,
    )

    entity_table_name, entity_pk_column = _ttng_custom_tt_entity_metadata(task)
    target_stype_value = _ttng_custom_tt_target_stype(pquery)
    task_type_value = (getattr(task.task_type, 'value', None)
                       or getattr(task.task_type, 'name', None)
                       or str(task.task_type))

    target_entity_table = None
    lp_rhs_is_temporal = False
    if bool(getattr(task.task_type, 'is_link_pred', False)):
        from kumo.pquery.utils import get_rhs_lp_table
        target_entity_table = get_rhs_lp_table(
            pquery.validated_predictive_query,
            pquery.graph,
            no_connector=True,
        )
        try:
            rhs_table = pquery.graph[target_entity_table]
            lp_rhs_is_temporal = bool(rhs_table.has_time_col()
                                      or rhs_table.has_end_time_col())
        except Exception:
            lp_rhs_is_temporal = False

    fact_pseudo_time = (static_link_fact(pquery.validated_predictive_query,
                                         target_entity_table)
                        if uses_pseudo_time(task_type_value) else None)

    split_spec = _build_split_spec(
        _coerce_split_config_for_diskgraph(split_config),
        split_family,
        SplitSpec,
    )
    if split_spec is None:
        raise RuntimeError(
            'TT-NG custom train table ingestion requires a valid split_config')

    col_stats_config: dict[str, Any] = {
        'target_stype': target_stype_value,
        'time_col': time_col_name,
    }
    if weight_col_name:
        col_stats_config['weight_col'] = weight_col_name

    # The DG builder writes its staging bundle to ``<output_path>.artifacts``
    # using local-filesystem ops, so ``output_path`` must be a LOCAL path.
    # ``train_table_path`` (the ingested customer parquet) is remote on the S3
    # storage lane; stage it into a local scratch dir so the builder reads the
    # parquet AND writes its ``.artifacts`` sibling locally (mirrors the PQL
    # path, where ``execute_tt_gen`` stages under the local index). We then
    # persist the local bundle to the durable (possibly remote) location.
    local_input = local_ttng_builder_input(train_table_path,
                                           'custom_train_table.parquet')

    with local_input as local_output_path:
        try:
            result = build_train_table_artifacts_for_existing_table(
                output_path=local_output_path,
                split_config=split_spec,
                tt_id=ttng_tt_id,
                split_family_hint=split_spec.kind,
                split_stats_config={
                    'entity_table': entity_table_name,
                    'entity_pk_column': entity_pk_column,
                },
                train_table_col_stats_config=col_stats_config,
                index_path=diskgraph_index_path,
                input_timestamp_unit='nanoseconds',
                target_entity_table=target_entity_table,
                task_type=task_type_value,
                lp_rhs_is_temporal=lp_rhs_is_temporal,
                fact_pseudo_time=fact_pseudo_time,
            )
        except DiskgraphDataError as error:
            raise TrainingDataMaterializationError(
                message=str(error),
                non_retryable=True,
            ) from error
        staging_dir = result.get('staging_dir')
        if not isinstance(staging_dir, str) or not staging_dir:
            raise RuntimeError(
                'TT-NG custom train table gen did not return staging_dir')

        # Persist the durable staging bundle (tt.parquet + tt_ingest_spec.json
        # + per-stage seed parquets) so execute_tt_ingest can rebuild the
        # extension on any pod after a snapshot restore -- same contract as the
        # normal-gen persist in generate_train_table.py. Custom TT is now
        # gen-only: the separate execute_tt_ingest (below / at materialize)
        # publishes the extension, exactly like the PQL path.
        persisted = ttng_persisted_staging_dir(train_table_path)
        published_bundle = persist_ttng_bundle(
            staging_dir,
            persisted,
            expected_tt_id=ttng_tt_id,
        )
    return osp.join(published_bundle, TT_PARQUET_FILENAME)


def build_ttng_custom_train_table_bundle(
    job_input: TrainingJobInput,
    settings: MLJobSettings,
    train_table_path: str,
    weight_col_name: Optional[str] = None,
) -> str:
    """C2: TT-NG custom-gen step. Build the durable staging bundle from the
    customer's parquet and persist it to ``ttng_persisted_staging_dir``. This
    is gen-only: it does NOT ingest (the train pod's ensure-extension step is
    the sole ingest).

    Plain function (like ``materialize_training_task``): it is invoked
    in-process by the pool wrapper
    ``SnapshotActivities.build_ttng_custom_train_table_bundle``, which is the
    ``@connector_authorized_activity`` and ``_ensure_snapshot``s the parent
    graph index onto local SSD first (the custom-TT gen wheel opens the parent
    index, absent on the [s3] lane without a snapshot restore). It only does
    the parse_pquery/DG-wheel build and never sets up per-job engine/data
    backends.
    """
    home_dir = get_job_dir(get_job_base_dir(), job_input.id)
    assert job_input.graph_data_compute_input is not None
    pquery, credentials = _create_pquery_with_credentials(
        job_input.spec,
        job_input.graph_data_compute_input.connectors,
    )
    pquery.graph.set_table_data_versions(job_input.id)
    open_connectors(pquery.graph, credentials)
    assert isinstance(job_input.spec.advanced_options, ModelPlan)
    task, _ = AutoTrainer.parse_pquery(
        pquery,
        model_plan=job_input.spec.advanced_options,
        train_table_path=train_table_path,
        job_id=job_input.id,
        job_dir=home_dir,
        ignore_encoders=True,
        weight_col_name=weight_col_name,
    )
    get_active_cache().table_version_cache.remove(task.train_table.data_source)
    train_table = task.train_table

    # Honour the split column the training-table specification names, so the
    # bundle we build carries the ManualSplit; no-op when the spec names none.
    split_obj, split_family = _custom_tt_split_override(
        getattr(train_table, 'split', None),
        _custom_tt_split_col_name(job_input))
    diskgraph_index_path = get_diskgraph_index_path_for_job(job_input)
    ttng_tt_id = str(job_input.id)
    time_col_name = getattr(train_table, 'time_col_name', None) or 'TIMESTAMP'

    with kumo_timing("training_materialize.ttng_custom_tt_source"):
        published_train_table_path = _build_ttng_custom_train_table_artifacts(
            train_table_path=train_table_path,
            split_config=split_obj,
            split_family=split_family,
            task=task,
            pquery=pquery,
            diskgraph_index_path=diskgraph_index_path,
            ttng_tt_id=ttng_tt_id,
            time_col_name=time_col_name,
            weight_col_name=weight_col_name,
        )

    # Mirror DiskGraph's native split stats for split-bearing custom tables.
    # Old/no-split bundles without the sidecar remain a no-op.
    from kumo.task.split import Split
    split_fpid = (Split.get_fpid_from_config(job_input.spec.split)
                  if job_input.spec.split is not None else None)
    _mirror_ttng_split_stats_to_job_dir(
        train_table_path=published_train_table_path,
        job_dir=home_dir,
        split_fpid=split_fpid,
        task_type=getattr(task, 'task_type', None),
    )
    return published_train_table_path


def _mirror_ttng_split_stats_to_job_dir(
    *,
    train_table_path: str,
    job_dir: str,
    split_fpid: Optional[str],
    task_type: Any,
) -> None:
    """Mirror DiskGraph's validated split-stats sidecar into the job dir."""
    if split_fpid is None:
        return

    from kumo.data.generic.diskgraph_train_table import (
        resolve_durable_ttng_bundle,
        ttng_bundle_dir_for_train_table,
    )
    from kumo.diskgraph import diskgraph_native_tt_artifacts
    from rest.backends.ml_backend.activities.split import (
        _assert_native_split_stats_contract,
        _split_train_table_stats_from_payload,
    )

    staging_dir = ttng_bundle_dir_for_train_table(train_table_path)
    staging_dir = resolve_durable_ttng_bundle(staging_dir)
    split_stats_path = osp.join(staging_dir, 'split_stats.json')
    if not fsutil.exists(split_stats_path):
        log.warning(
            'DiskGraph split-stats sidecar is absent; no split stats will be '
            'published for train table %s. New split-bearing bundles should '
            'publish this artifact natively.',
            train_table_path,
        )
        return

    parsed = diskgraph_native_tt_artifacts.parse_split_stats_artifact(
        split_stats_path)
    split_stats = _split_train_table_stats_from_payload(parsed)
    _assert_native_split_stats_contract(task_type, split_stats)
    out_path = HomeDirManager(job_dir).split_train_table_stats(split_fpid)
    save_json(split_stats, out_path)


@connector_activity_registry.connector_authorized_activity
def materialize_training_table(
    job_input: TrainingJobInput,
    settings: MLJobSettings,
    train_table_path: str,
    execute_split: bool = True,
    weight_col_name: Optional[str] = None,
    is_custom_train_table: bool = False,
) -> Tuple[str, ValidationResponse]:
    """Materialize a train table for a training job.

    Given a train table path (either customer-provided, or generated by the
    `generate_train_table` activity), rehydrate any missing pre-split stats
    or train-table edge bundle from the DiskGraph index, then materialize the
    train table splits and persist the config + data to the job home dir.

    Args:
        job_input (TrainingJobInput): The training job input.
        settings (MLJobSettings): The ml job setting.
        train_table_path (str): The train table path to load and materialize
            the train table.
        execute_split (bool): Whether to execute the train/val/test split as
            part of materialization.
        weight_col_name (Optional[str]): The weight column name, if present
            in the train table.
        is_custom_train_table (bool): Whether the train table is
            customer-provided rather than generated.

    Raises:
        TrainingDataMaterializationError: If rehydrating the missing
            train-table artifacts or materializing the splits fails.

    Returns:
        Tuple[str, ValidationResponse]: train table materialization path and
            the validation response.
    """
    validation_response = ValidationResponse()
    # TT-NG data plane. The durable staging bundle is produced+persisted by
    # gen (generate_train_table.py) or, for custom TT, by the custom-gen
    # activity (build_ttng_custom_train_table_bundle). The train pod's
    # ensure-extension step is the SOLE ingest. This activity only validates
    # the durable bundle and returns; parsing the source table here would both
    # duplicate the producer's work and require a raw staged copy that direct
    # Databricks ingestion intentionally does not create.
    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        assert_ttng_bundle_complete,
        ttng_bundle_dir_for_train_table,
    )
    staging_dir = ttng_bundle_dir_for_train_table(train_table_path)
    try:
        assert_ttng_bundle_complete(staging_dir)
    except TTNGBundleError as e:
        raise TrainingDataMaterializationError(
            message=(f'{e} The gen job (PQL) or the custom-gen step must '
                     'persist the complete bundle before materialize runs.'),
            error=e,
            non_retryable=True,
        ) from e
    return ("", validation_response)


def _get_trainer_dir_from_job_id(
    job_id: str,
    settings: MLJobSettings,
    context: str,
    current_job_id: Optional[str] = None,
) -> str:
    """Get the best trainer directory from a training job ID.

    Args:
        job_id: The training job ID to get the trainer directory from.
        settings: ML job settings containing deploy environment config.
        context: Context string for logging (e.g., "warm start" or "base trainer").
        current_job_id: Optional current job ID for error messages.

    Returns:
        The path to the best trainer directory.

    Raises:
        ValueError: If the trainer directory does not exist.
    """
    home_dir = get_job_dir(get_job_base_dir(), TrainingJobID(job_id))
    home_mgr = HomeDirManager(home_dir)

    # Get the best trainer directory from the job
    task_name = home_mgr.first_task_name()
    assert task_name is not None
    trainer_dir = home_mgr.get_best_trainer_dir(task_name)
    assert trainer_dir is not None

    log.info(f'Using {context} trainer directory: {trainer_dir}')
    if not fsutil.exists(trainer_dir):
        error_msg = f"{context.capitalize()} trainer directory not found at '{trainer_dir}'"
        if current_job_id:
            error_msg += f' for job {current_job_id}'
        error_msg += '. Please verify the training job completed successfully.'
        raise TrainingJobOutputError(
            message=error_msg,
            non_retryable=True,
        )

    return trainer_dir


@connector_activity_registry.connector_authorized_activity
def generate_training_config(
    job_input: TrainingJobInput,
    settings: MLJobSettings,
) -> str:
    r"""
    Generates the training config and returns the saved path. It also
    calculates the input data stats and saves them for downstream
    activities.

    De-globalized to run on the data-backend thread pool: it no longer wraps
    its work in ``temp_home_dir`` or per-job engine/data-backend setup (which
    mutate the process-global home dir + engine/data-backend singletons). The
    global home
    dir is never read on this path (``home_dir`` is threaded explicitly through
    ``HomeDirManager`` and ``save_trainer_job_config``); the engine /
    data-backend reads it needs are deploy-wide and established once at worker
    startup; ``estimate_input_data_stats`` and ``set_table_data_versions`` only
    touch the version-keyed ``get_active_cache()`` singleton. The one genuinely
    per-job global it used to set — the diskgraph index path — is now threaded
    explicitly into ``save_trainer_job_config``, so concurrent jobs sharing a
    thread can't collide.
    """
    # Extract whether we should disable explainability
    disable_explain = (
        job_input.spec.advanced_options.training_job.disable_explain)
    # Get model plan
    model_plan = job_input.spec.advanced_options
    task_metrics = model_plan.training_job.metrics
    if model_plan.training_job.tune_metric not in task_metrics:
        model_plan.training_job.tune_metric = task_metrics[0]
    for key, value in settings.train_cfg_overrides.items():
        setattr(model_plan, key, value)
    # Get home dir
    job_id = job_input.id
    home_dir = get_job_dir(get_job_base_dir(), job_id)
    home_dir_mgr = HomeDirManager(home_dir)

    # Handle warm start if specified
    warm_start_trainer_dir = None
    if job_input.warm_start_job_id:
        warm_start_trainer_dir = _get_trainer_dir_from_job_id(
            job_input.warm_start_job_id,
            settings,
            'warm start',
        )

    # Per-job diskgraph index path, threaded explicitly into the JobConfig
    # instead of via the process-global engine handle (not set on the thread
    # pool).
    diskgraph_index_path = get_diskgraph_index_path_for_job(job_input)

    # Retrieve the graph from pquery config
    assert job_input.graph_data_compute_input is not None
    predictive_query_config, _ = hydrate_predictive_query_for_activity(
        job_input.spec.predictive_query,
        job_input.graph_data_compute_input.connectors,
    )
    with kumo_timing("training_config.pquery_from_config"):
        pquery: PredictiveQuery = PredictiveQueryV2.from_config(
            predictive_query_config)
    graph = pquery.graph
    # Set table data versions to the one decided at the beginning
    # of the batch prediction workflow
    graph.set_table_data_versions(job_input.id)

    # Estimate data stats from the graph. Number of edges is retrieved
    # from data backend cache. Materialized table size is estimated from
    # table source size
    assert job_input.graph_version_info
    assert job_input.graph_data_compute_input
    graph_key_ver = VersionedGraphKey(
        job_input.graph_data_compute_input.graph_key,
        job_input.graph_version_info.graph_data_version)
    with kumo_timing("training_config.estimate_input_stats"):
        data_stats = estimate_input_data_stats(graph, graph_key_ver,
                                               deploy_env())
    log.info(f"Computed estimated InputDataStats: {data_stats} for job: "
             f"{job_id}")
    save_json(data_stats, home_dir_mgr.estimated_input_data_stats_path)

    # Generate JobConfig and save it. This will be used later by the
    # materialize_training_task activity to load materialization paths. On DG,
    # graph tables and edges are served from the on-disk index, so no edge
    # persist paths are produced here.
    with kumo_timing("training_config.save_trainer_job_config"):
        config_path = save_trainer_job_config(home_dir, pquery.name, graph,
                                              model_plan, job_id,
                                              diskgraph_index_path,
                                              disable_explain,
                                              warm_start_trainer_dir)
    log.info(f"Saved materialized task for job: {job_id}. JobConfig "
             f"saved at {config_path}")
    return config_path


def _maybe_materialize_ttng_task_dataset(
    *,
    task: Any,
    pquery: PredictiveQuery,
    train_table_path: str,
    job_input: TrainingJobInput,
    diskgraph_index_path: Optional[str],
    job_config_path: str,
) -> None:
    """DG-native task-dataset materialization for the TT-NG path.

    Writes the manifest bundle via
    :meth:`DiskgraphService.materialize_task_dataset` and records ``bundle_dir``
    on the ``JobConfig`` so the trainer dispatches to ``DiskgraphNestedTrainer``.
    No-op unless ``diskgraph_index_path`` is set. Supports the TT-NG
    node-prediction task types plus LP; others raise.
    """
    if diskgraph_index_path is None:
        return
    try:
        from kumo.diskgraph import DiskgraphHandle, get_service
        from kumo.task.task import collect_encoders
        is_link_pred = bool(getattr(task.task_type, 'is_link_pred', False))
        if task.task_type not in _TTNG_TASK_DATASET_TASK_TYPES:
            raise RuntimeError('TT-NG task dataset does not support this '
                               'task type; '
                               f'got task_type={task.task_type!r}')

        fitted_encoders = collect_encoders(task)

        rhs_node_type: Optional[str] = None
        if is_link_pred:
            from kumo.pquery.utils import get_rhs_lp_table
            rhs_node_type = get_rhs_lp_table(
                pquery.validated_predictive_query,
                pquery.graph,
                no_connector=True,
            )

        _ttng_tt_id = _ttng_train_table_tt_id(train_table_path)
        # Snapshot restore may omit the per-job train_tables extension; re-ingest
        # from durable staging so the manifest below exists.
        _ttng_ensure_train_table_extension(train_table_path,
                                           diskgraph_index_path, _ttng_tt_id)
        wheel_tt_manifest_path = osp.join(diskgraph_index_path, 'train_tables',
                                          _ttng_tt_id, 'manifest.json')

        from kumo.diskgraph._diskgraph_legacy import build_task_dataset_spec
        spec = build_task_dataset_spec(
            task=task,
            fitted_encoders=fitted_encoders,
            tt_manifest_path=wheel_tt_manifest_path,
            job_id=str(job_input.id),
            artifact_root=osp.join(
                diskgraph_index_path,
                'task_datasets',
                str(job_input.id),
            ),
            rhs_node_type=rhs_node_type,
        )
        # Normalize the display ``graph_key`` (e.g. "FINANCIAL") to its
        # ``GraphIDv2`` ("NA-<fpid>") form so ``_resolve(handle)`` lands on the
        # same on-disk snapshot dir that ``get_diskgraph_index_path_for_job``
        # (the writer/ingest side, used everywhere else in this module) produced.
        # Without this the ingest-binding validator computes the expected
        # ``graph_index_path`` off the raw display name and rejects the
        # (correctly NA-keyed) published manifest.
        handle = DiskgraphHandle(
            graph_key=str(
                _graph_key_to_v2(job_input.graph_data_compute_input.graph_key)),
            graph_data_version=(
                job_input.graph_version_info.graph_data_version),
            job_id=str(job_input.id),
        )
        result = get_service().materialize_task_dataset(handle, spec)

        # Persist bundle_dir on the JobConfig so kumo-auto can detect
        # the TT-NG artifact and dispatch accordingly.
        job_config = JobConfig.load(job_config_path)
        job_config.diskgraph_task_dataset_bundle_dir = result.bundle_dir
        save_json(job_config, job_config_path)

        log.info(
            '[TT-NG] materialize_task_dataset wrote manifest at '
            '%s (storage_kind=%s); recorded bundle_dir=%s on JobConfig '
            'at %s',
            result.manifest_path,
            result.storage_kind,
            result.bundle_dir,
            job_config_path,
        )
    except Exception as e:
        raise TrainingDataMaterializationError(
            message=('TT-NG task-dataset materialization failed '
                     f'(job_id={job_input.id})'),
            error=e,
            non_retryable=is_error_non_retryable(e),
        ) from e


def materialize_training_task(
    job_input: TrainingJobInput,
    settings: MLJobSettings,
    train_table_path: str,
    train_table_materialization_path: str,
    job_config_path: str,
    weight_col_name: Optional[str] = None,
) -> Tuple[str, str, ValidationResponse]:
    r"""
    This activity performs the following functions:
    * Run `infer_encoders` on the graph
    * Create Training Table
    * Materialize and save the task
    * Create :obj:`JobConfig` containing all the information required for
        running the training job.

    Returns the path to the saved :obj:`JobConfig` if using k8s backend.
    Returns :obj:`None` otherwise.
    """
    validator = validation.MaterializeTaskValidator(
        max_total_table_size=settings.max_total_table_size,
        max_total_graph_edges=settings.max_total_graph_edges,
    )
    validation_response = ValidationResponse()

    log.info("Starting materialize_task activity for job: %s", job_input.id)

    job_id = job_input.id
    home_dir = get_job_dir(get_job_base_dir(), job_id)
    # No process globals: neither temp_home_dir nor per-job engine setup runs.
    # home_dir is threaded explicitly to every consumer on the materialize path
    # (parse_pquery job_dir=, task.materialize persist_dir=/local_dir=,
    # HomeDirManager below), so concurrent jobs on the pool worker's shared
    # thread executor can't collide. The diskgraph index path is threaded
    # (job_config.diskgraph_index_path -> task.materialize;
    # get_diskgraph_index_path_for_job in the col-stats helpers); nothing reads
    # global engine state (engine_backend defaults to DISKGRAPH; engine_mode
    # reads escaped by the explicit index path).
    # Create predictive query
    with kumo_timing("training_task.create_pquery"):
        pquery, credentials = _create_pquery_with_credentials(
            job_input.spec,
            job_input.graph_data_compute_input.connectors,
        )

    # Set table data versions to the one decided at the beginning
    # of the training workflow
    pquery.graph.set_table_data_versions(job_input.id)
    # Open the connectors
    log.debug("Opening Connectors '%s' for training job: %s",
              pquery.graph.data_sources, job_input.id)
    with kumo_timing("training_task.open_connectors"):
        open_connectors(pquery.graph, credentials)

    with kumo_timing("training_task.ensure_col_stats_cache"):
        _ensure_diskgraph_col_stats_in_cache(pquery.graph, job_input)
    with kumo_timing("training_task.infer_graph_encoders"):
        _infer_graph_encoders_from_cached_stats_or_fail(pquery.graph)

    # Create task and AutoTrainer config.
    try:
        with kumo_timing("training_task.parse_pquery"):
            parse_table_path, parse_table_name = (
                _ttng_train_table_parse_target(
                    train_table_path,
                    is_custom_train_table=(job_input.spec.custom_train_table
                                           is not None),
                ))
            task, space = AutoTrainer.parse_pquery(
                pquery,
                model_plan=job_input.spec.advanced_options,
                train_table_path=parse_table_path,
                job_id=job_input.id,
                job_dir=home_dir,
                ignore_encoders=True,
                weight_col_name=weight_col_name,
                train_table_name=parse_table_name,
            )
        for key, value in settings.train_cfg_overrides.items():
            setattr(space, key, value)
        _configure_forecasting_sequence_columns(
            task,
            job_input.spec.predictive_query.year_over_year,
            include_future_labels=True,
        )
    except Exception as e:
        raise TrainingDataMaterializationError(
            message="Failed to prepare training data",
            error=e,
            non_retryable=is_error_non_retryable(e),
        ) from e

    validation_response.merge(pquery.get_validate_train_table_response()
                              or ValidationResponse())
    if task.graph.name is None:
        msg = ("Invalid graph configuration: Graph name cannot be None in "
               "training task")
        raise InvalidGraphError(message=msg)

    # Remove cached data version for previously generated train table
    get_active_cache().table_version_cache.remove(task.train_table.data_source)

    diskgraph_index_path = get_diskgraph_index_path_for_job(job_input)
    assert diskgraph_index_path, (
        "diskgraph_index_path must be set when diskgraph edge "
        "materialization is enabled.")

    # The wheel owns artifact production, so skip the legacy load_data
    # -> materialize -> edges chain. An empty TableData shell lets
    # downstream touch ``train_table.data``.
    task.train_table._data = TableData(
        task.train_table,
        data_version=task.train_table.data_version,
    )
    # Validate the producer's categorical-stat coverage before a missing local
    # extension triggers a potentially expensive re-ingest. The generator
    # handoff remains private to execute_tt_ingest; consumers seed only from
    # the complete published extension returned here.
    ttng_tt_id = _ttng_train_table_tt_id(train_table_path)
    sidecar_path = _ttng_prepare_train_table_col_stats(task, train_table_path,
                                                       diskgraph_index_path,
                                                       ttng_tt_id)

    # Seed col stats so the encoder build finds CATEGORY_COUNTS. This reads
    # only the published, self-describing sidecar -- never the transient gen
    # manifest.
    seeded = (task.seed_train_table_col_stats_from_sidecar(sidecar_path)
              if sidecar_path is not None else False)
    log.info('[TT-NG] Train-table col stats seeded: %s (sidecar=%s)', seeded,
             sidecar_path)
    task.infer_encoders()
    task.validate(raise_on_error=True, ignore_encoders=False)

    with kumo_timing("training_task.materialize_ttng_task_dataset"):
        _maybe_materialize_ttng_task_dataset(
            task=task,
            pquery=pquery,
            train_table_path=train_table_path,
            job_input=job_input,
            diskgraph_index_path=diskgraph_index_path,
            job_config_path=job_config_path,
        )

    home_dir_mgr = HomeDirManager(home_dir)

    # Diskgraph: stats from index metadata (no Spark artifacts)
    from kumo.diskgraph import DiskgraphHandle
    from kumo.diskgraph.input_stats import assemble_input_data_stats
    graph_input = job_input.graph_data_compute_input
    graph_version = job_input.graph_version_info
    graph_key_ver = VersionedGraphKey(graph_input.graph_key,
                                      graph_version.graph_data_version)
    with kumo_timing("training_task.assemble_input_stats"):
        data_stats = assemble_input_data_stats(
            DiskgraphHandle(
                graph_key=graph_input.graph_key,
                graph_data_version=graph_version.graph_data_version,
            ),
            graph=task.graph,
            graph_key_ver=graph_key_ver,
            index_path=diskgraph_index_path,
        )

    log.info(f"Computed InputDataStats: {data_stats} for job: "
             f"{job_id}")
    save_json(data_stats, home_dir_mgr.input_data_stats_path)

    # Save the InputDataStats in StateDB:
    with catch_and_log():
        save_input_data_stats_to_state_db(job_id, data_stats)

    # Run post materialization validation
    with kumo_timing("training_task.post_materialization_validation"):
        validation_response.merge(validator.post_materialization(data_stats))
    if not validation_response.ok:
        return ("", "", validation_response)

    log.debug("Saving task to home_dir: %s", home_dir)
    # Save the task to the job's home dir. home_dir is threaded explicitly
    # (Task.save -> graph.save(destination=)) so no process-global temp_home_dir
    # is needed and concurrent jobs on the shared thread executor don't collide.
    with kumo_timing("training_task.save_task"):
        task.save(destination=home_dir)

    # Save stats about the materialization to S3.
    materialization_info = make_materialization_info(
        job_input.start_time,
        task.graph,
        job_id=job_input.id,
        graph_id=job_input.spec.graph_id,
        predictive_query_id=job_input.spec.predictive_query_id,
        temporal_workflow_id=activity.info().workflow_run_id,
    )
    with kumo_timing("training_task.save_materialization_info"):
        save_materialization_info(materialization_info)

    return (job_config_path, get_data_operation_description(),
            validation_response)


def _create_pquery_with_credentials(
    spec: TrainingSpec,
    connector_refs,
):
    predictive_query_config, credentials = (
        hydrate_predictive_query_for_activity(
            spec.predictive_query,
            connector_refs,
        ))
    return (
        _create_pquery_from_hydrated_config(spec, predictive_query_config),
        credentials,
    )


def _create_pquery_from_hydrated_config(
    spec: TrainingSpec,
    predictive_query_config,
) -> PredictiveQuery:
    if isinstance(spec.advanced_options, ModelPlan):
        train_table_opts = spec.advanced_options.\
            training_table_generation
        if not _is_inferred(train_table_opts.split):
            # TODO (matthias) drop the above condition when split
            # can no longer be "INFERRED"
            split_conf = get_split(train_table_opts.split)
            assert split_conf is not None, (
                'Split is not InferredType in Model Plan '
                'but get_split returned None')
            if split_conf is not None and hasattr(split_conf,
                                                  'flexible_boundaries'):
                if split_conf.timeframe_step is not None:
                    split_conf.flexible_boundaries = True
                else:
                    # We should NEVER modify the customer-given
                    # split when timeframe_step is None
                    split_conf.flexible_boundaries = False

            predictive_query_config.split = split_conf.config()

    pquery = PredictiveQueryV2.from_config(predictive_query_config)
    return pquery
