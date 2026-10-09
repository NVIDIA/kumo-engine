# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os.path as osp
from typing import Any, Optional, Tuple

from kumo_core.job_config import PredictionConfig
from kumo_core.prediction_options import resolve_prediction_time_options
from kumo_core.types import PredictionArtifactType, TaskType
from kumoapi.typing import Dtype
from temporalio import activity

from io_utils import fsutil, logging
from io_utils.exceptions import catch_and_log
from io_utils.json import save_json
from kumo.connector import ParquetConnector
from kumo.data import get_active_cache, get_persist_dir
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.diskgraph import DiskgraphHandle
from kumo.home import HomeDirManager
from kumo.pquery import PredictiveQuery, PredictiveQueryV2
from kumo.pquery.predictive_query import train_table_source_name
from kumo.task.task import Task
from kumo.validation import ValidationResponse
from kumo.validation.train_table_validation import TrainingTableSplitError
from rest.backends.ml_backend import validation
from rest.backends.ml_backend.activities.common import (
    estimate_input_data_stats,
    make_materialization_info,
    save_input_data_stats_to_state_db,
)
from rest.backends.ml_backend.data_util import (
    get_job_base_dir,
    get_job_dir,
    get_max_partition_size_with_top_k,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.timing import kumo_timing
from rest.config.batch_prediction_job import (  # noqa E501
    BatchPredictionJobInput,
    requested_prediction_artifact_types,
)
from rest.utils.driver_common import (
    hydrate_predictive_query_for_activity,
    open_connectors,
)
from service_framework.deploy_env_config import deploy_env
from service_framework.deploy_stage import EnvVar
from statedb.config.diskgraph import get_diskgraph_index_path_for_job
from statedb.utils.errors import (
    IncompatiblePredictionTable,
    MissingTrainingJobError,
    PredictionDataMaterializationError,
)
from temporalio_commons.error_utils import is_error_non_retryable

from ...data_backend import connector_activity_registry
from ..ml_job_output_store import save_materialization_info

log = logging.getLogger(__name__)

PRED_TABLE = 'pred_table'
ENV_MAX_PARTITIONS_PER_WORKER = EnvVar('MAX_PARTITIONS_PER_WORKER', type=int,
                                       required=False, default_value=100)


def _configure_prediction_output_fields(
    prediction_config: PredictionConfig,
    job_input: BatchPredictionJobInput,
) -> None:
    requested_by_artifact: dict[str, list[str]] = {}
    for output in job_input.config.outputs:
        artifact_type = output.artifact_type.value
        requested_by_artifact.setdefault(
            artifact_type,
            sorted({
                getattr(field, 'value', field)
                for field in output.extra_fields
            }))
    prediction_config.output_extra_fields = requested_by_artifact

    time_options = resolve_prediction_time_options(job_input.config.task_type,
                                                   requested_by_artifact)
    prediction_config.ignore_time_col = time_options.ignore_time_col
    if time_options.pred_time_col_name is not None:
        prediction_config.pred_time_col_name = time_options.pred_time_col_name
    if time_options.include_job_timestamp:
        prediction_config.start_time = job_input.start_time


def return_embeddings(job_input: BatchPredictionJobInput) -> bool:
    return (PredictionArtifactType.EMBEDDINGS
            in requested_prediction_artifact_types(job_input.config))


def return_predictions(job_input: BatchPredictionJobInput) -> bool:
    return (PredictionArtifactType.PREDICTIONS
            in requested_prediction_artifact_types(job_input.config))


def _pred_num_partitions_from_rows(
    num_rows: int,
    pred_config: PredictionConfig,
    job_input: BatchPredictionJobInput,
    settings: MLJobSettings,
    task: Task,
    *,
    id_is_string: bool,
) -> int:
    """Compute partition count from a row count.

    Used by the TT-NG path, which sources ``num_rows`` from the ingested
    predict stage.
    """
    job_id = job_input.id
    job_config = job_input.config
    log.info(f"job_id: {job_id}, num_rows in prediction table: "
             f"{num_rows}")

    # First compute no.of partitions based on limits:
    max_rows_per_partition = settings.pred_max_num_rows_per_partition
    max_entities_per_partition = \
        settings.pred_max_num_entities_per_partition
    log.info("Environment settings - "
             f"max_rows_per_partition: {max_rows_per_partition}, "
             f"max_entities_per_partition: "
             f"{max_entities_per_partition}, "
             f"num_parallel_workers: "
             f"{job_config.num_parallel_workers_limit}, "
             f"{task.task_type=}")
    # Compute maximum partition size:
    max_partition_size = max_rows_per_partition
    if pred_config.return_embeddings:
        max_partition_size = max_entities_per_partition
    if pred_config.top_k is not None and pred_config.top_k > 0:
        max_partition_size = get_max_partition_size_with_top_k(
            max_partition_size, max_rows_per_partition, pred_config.top_k, task)

    # If the task type is MULTICLASS_CLASSIFICATION or
    # MULTILABEL_CLASSIFICATION, prediction table conversion from wide to
    # long format is invoked. This will raise a pyarrow overflow error if
    # we use the current partition sizes and the ID column in the
    # prediction table is a string. To avoid this, we need to split the
    # table into smaller partitions.
    if (task.task_type in {
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION
    }):
        if id_is_string:
            log.info(f"Reducing max_partition_size by a factor of 32 for "
                     f"job: {job_id} due to string ID column in prediction "
                     f"table.")
            max_partition_size = max_partition_size // 32
    elif task.task_type == TaskType.FORECASTING:
        lag_timesteps = 0
        yoy_timesteps = 0
        if job_input.predictive_query.lag_timesteps is not None:
            lag_timesteps = job_input.predictive_query.lag_timesteps
        if job_input.predictive_query.year_over_year:
            assert job_input.predictive_query.forecast_length is not None
            yoy_timesteps = job_input.predictive_query.forecast_length
        if lag_timesteps + yoy_timesteps > 0:
            max_partition_size = max_partition_size // (lag_timesteps +
                                                        yoy_timesteps)
            log.info(f"Reducing max_partition_size by a factor of "
                     f"{lag_timesteps + yoy_timesteps} for job: {job_id} as it "
                     f"has lag features for forecasting.")

    log.info(f"Final max partition size: {max_partition_size}")

    num_partitions = int(num_rows / max_partition_size) + 1

    # Now check for num_parallel_workers i.e. split the table into
    # multiples of num_parallel_workers:
    num_partitions = job_config.num_parallel_workers_limit * math.ceil(
        num_partitions / job_config.num_parallel_workers_limit)

    # Limit the number of partitions based on max partitions per worker
    max_partitions = (job_config.num_parallel_workers_limit *
                      ENV_MAX_PARTITIONS_PER_WORKER.parse_value())
    num_partitions = min(num_partitions, max_partitions)

    log.info(f"Prediction table for job: {job_id} will be partitioned "
             f"into {num_partitions} partitions")
    return num_partitions


@connector_activity_registry.connector_authorized_activity
def generate_prediction_config(
    job_input: BatchPredictionJobInput,
    settings: MLJobSettings,
    serving: bool,
    pred_table_path: str,
    home_dir: Optional[str] = None,
) -> str:
    r"""
    Generates the prediction config and returns the saved path. It also
    calculates the input data stats and saves them for downstream
    activities.
    """
    # Load job metadata
    job_id = job_input.id
    training_job_id = job_input.training_job_id
    if home_dir is None:
        home_dir = get_job_dir(get_job_base_dir(), job_id)
    home_dir_mgr = HomeDirManager(home_dir)
    train_dir = get_job_dir(get_job_base_dir(), training_job_id)

    diskgraph_index_path = get_diskgraph_index_path_for_job(job_input)

    # Retrieve the graph from pquery config
    predictive_query_config, _ = hydrate_predictive_query_for_activity(
        job_input.predictive_query,
        job_input.graph_data_compute_input.connectors,
    )
    with kumo_timing("prediction_config.pquery_from_config"):
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
    graph_key_ver = VersionedGraphKey(
        job_input.graph_data_compute_input.graph_key,
        job_input.graph_version_info.graph_data_version)
    with kumo_timing("prediction_config.estimate_input_stats"):
        data_stats = estimate_input_data_stats(graph, graph_key_ver,
                                               deploy_env())
    log.info(f"Computed estimated InputDataStats: {data_stats} for job: "
             f"{job_id}")
    save_json(data_stats, home_dir_mgr.estimated_input_data_stats_path)

    # Get the best trainer dir; on DG, graph-table/node/edge data is served
    # from the on-disk index, so no edge persist paths are generated here.
    pquery_name = job_input.predictive_query.name
    train_dir_mgr = HomeDirManager(train_dir)
    model_dir = train_dir_mgr.get_best_trainer_dir(pquery_name)

    # TT-NG keeps ``pred_table_path`` STABLE at the gen location: the predict
    # path re-ingests the durable ``.ttng_staging`` bundle written BESIDE the
    # pred table (resolved via ``ttng_persisted_pred_staging_dir``) and the DG
    # service ingests seeds from it -- the raw pred parquet is never read as
    # data. No relocation, no bundle copy.

    output_format = 'wide'  # Deprecated - remove once models are past 1.37.
    if job_input.config.task_type in [
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
    ]:
        output_format = 'long'

    # Generate PredictionConfig and save it. This will be used later by the
    # materialize_prediction_task activity to load materialization paths.
    pred_config = PredictionConfig(
        home_dir=home_dir,
        train_dir=train_dir,
        model_dir=model_dir,
        output_dir=home_dir_mgr.prediction_output_dir,
        task_name=pquery_name,
        return_embeddings=return_embeddings(job_input),
        return_predictions=return_predictions(job_input),
        pred_table_path=pred_table_path,
        data_dir=get_persist_dir(),
        top_k=job_input.config.num_classes_to_return,
        binary_threshold=job_input.config.binary_classification_threshold,
        tables_to_load=graph.table_names,
        nodes_to_load=graph.table_names,
        diskgraph_index_path=diskgraph_index_path,
        disable_explain=not job_input.config.explanations,
        output_format=output_format,
    )
    _configure_prediction_output_fields(pred_config, job_input)
    fsutil.makedirs(home_dir_mgr.jobs_dir, exist_ok=True)
    pred_config_path = home_dir_mgr.job_config_path(job_id)
    with kumo_timing("prediction_config.save_prediction_config"):
        save_json(pred_config, pred_config_path)
    log.info(f"Saved PredictionConfig for job: {job_id} at {pred_config_path}")
    return pred_config_path


def _load_ttng_prediction_bundle_identity(
        pred_table_path: str) -> tuple[str, str]:
    """Return the durable bundle path and its canonical source TT id."""
    from kumo.data.generic.diskgraph_staging import (
        ttng_persisted_pred_staging_dir,
    )
    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        assert_ttng_bundle_complete,
        read_tt_id_from_staging_bundle,
    )

    durable_staging_dir = ttng_persisted_pred_staging_dir(pred_table_path)
    try:
        assert_ttng_bundle_complete(durable_staging_dir)
        source_tt_id = read_tt_id_from_staging_bundle(durable_staging_dir)
    except TTNGBundleError as e:
        raise PredictionDataMaterializationError(
            message=(f'{e} The gen→ingest handoff did not persist a complete '
                     'bundle beside the pred table.'),
            error=e,
            non_retryable=True,
        ) from e
    return durable_staging_dir, source_tt_id


def _materialize_ttng_prediction_task_dataset(
    *,
    task: Any,
    pquery: Any,
    job_input: BatchPredictionJobInput,
    prediction_config: PredictionConfig,
    diskgraph_index_path: str,
    train_table_name: str,
    durable_staging_dir: str,
    source_tt_id: str,
) -> int:
    """TT-NG batch-prediction materialize (node + link prediction).

    Ingest-only: the generator already ran ``execute_tt_gen`` into the kept
    staging dir; this ingests it and records ``bundle_dir`` on the
    ``PredictionConfig`` so ``kumo-pred`` dispatches to ``DiskgraphNestedTrainer``.
    Returns the ingested predict-stage row count.
    """
    from kumo_core.task_dataset import TaskDatasetMaterializeSpec

    from kumo.diskgraph import DiskgraphHandle, get_service
    from kumo.task.task import collect_encoders
    from rest.backends.ml_backend.activities.materialize_training_task import (
        _build_ttng_feature_schema,
        _ttng_task_family,
        _ttng_task_is_temporal,
    )

    is_link_pred = bool(getattr(task.task_type, 'is_link_pred', False))

    fitted_encoders = collect_encoders(task)
    graph = task.graph
    graph_node_types = tuple(t.table_name for t in graph.tables)
    graph_edge_types = tuple((t.table_name, fkey.name, fkey.dst_table_name)
                             for t in graph.tables for fkey in t.fkeys)
    feature_schema = _build_ttng_feature_schema(task, fitted_encoders)
    if task.task_type == TaskType.FORECASTING:
        pred_table_name = task.train_table.table_name
        if train_table_name != pred_table_name:
            node_schema = feature_schema['node_types']
            pred_entry = node_schema.pop(pred_table_name, None)
            if pred_entry is not None:
                node_schema[train_table_name] = {
                    **pred_entry,
                    'mapper_key': train_table_name,
                }

    # Entity table = dst of the prediction table's entity FK (strip any
    # ``connector.`` prefix to match the wheel's bare-name convention).
    pred_fkeys = list(getattr(task.train_table, 'fkeys', None) or [])
    assert pred_fkeys, 'TT-NG BP: prediction table has no entity fkey'
    dst_tbl = pred_fkeys[0].dst_table or ''
    entity_table_name = (dst_tbl.split('.', 1)[1]
                         if '.' in dst_tbl else dst_tbl)

    # Task.is_temporal carries temporal node-prediction identity; the TaskType
    # enum only separates temporal/static for link prediction.
    is_temporal = _ttng_task_is_temporal(task)
    rhs_node_type: Optional[str] = None
    if is_link_pred:
        from kumo.pquery.utils import get_rhs_lp_table
        rhs_node_type = get_rhs_lp_table(
            pquery.validated_predictive_query,
            pquery.graph,
            no_connector=True,
        )

    from kumoapi.pquery import QueryType

    from kumo.data.generic.diskgraph_staging import (
        predict_staging_tt_id,
        ttng_staging_dir,
    )

    # Prediction anchor: deterministic max target date (matches the BP-gen
    # anchor); empty for static queries.
    bp_gen = pquery.get_batch_pred_table_generator()
    pred_time = None
    if bp_gen.query_type == QueryType.TEMPORAL:
        pred_time = bp_gen.generate_pred_timestamp()
    timestamps = [int(pred_time.value)] if pred_time else []  # ns

    # Predict seed time (int64-ns) for the RHS MIPS NodeRange loader (else it
    # seeds rhs_seed_time=None and the engine raises). Temporal -> anchor,
    # static -> 0. LP-only.
    predict_seed_time_ns: Optional[int] = None
    if is_link_pred:
        predict_seed_time_ns = timestamps[0] if timestamps else 0

    from kumo.data.generic.diskgraph_train_table import (
        TTNGBundleError,
        assert_ttng_bundle_complete,
        restore_durable_ttng_bundle,
    )
    predict_staging_dir = ttng_staging_dir(diskgraph_index_path, source_tt_id)

    # Bridge the gen→ingest handoff across delete_snapshot / across pods (mirrors
    # the train fix): prefer the index-local ephemeral staging, fall back to the
    # durable copy beside the pred table. Record the durable dir on the config so
    # each predict shard can reconstruct its own extension.
    prediction_config.diskgraph_pred_staging_dir = durable_staging_dir
    # TARGET is BP-job-keyed (pred_<job_input.id>), distinct from the gen-keyed
    # SOURCE staging above. This is a LOCAL fact of THIS BP job (not a
    # cross-boundary hand-off): each of this job's shards re-ingests the SAME
    # source bundle INTO this isolated target extension. Keeping source != target
    # is the intentional per-BP-job fan-out isolation -- do NOT collapse them.
    prediction_config.diskgraph_pred_target_tt_id = predict_staging_tt_id(
        str(job_input.id))
    had_complete_local_bundle = True
    try:
        assert_ttng_bundle_complete(predict_staging_dir)
    except TTNGBundleError:
        had_complete_local_bundle = False
    restore_durable_ttng_bundle(durable_staging_dir, predict_staging_dir)
    if not had_complete_local_bundle:
        log.info('[TT-NG] BP restored predict staging from durable %s -> %s',
                 durable_staging_dir, predict_staging_dir)

    handle = DiskgraphHandle(
        graph_key=job_input.graph_data_compute_input.graph_key,
        graph_data_version=job_input.graph_version_info.graph_data_version,
    )
    bundle_root = osp.join(diskgraph_index_path, 'task_datasets',
                           str(job_input.id))
    service = get_service()
    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path='',  # unused on the BP path (no wheel TT manifest)
        job_id=str(job_input.id),
        artifact_root=bundle_root,
        task_family=_ttng_task_family(task.task_type),
        task_type=task.task_type.value,
        is_temporal=is_temporal,
        seed_node_type=entity_table_name,
        train_table_name=train_table_name,
        entity_table_name=entity_table_name,
        rhs_node_type=rhs_node_type,
        feature_schema=feature_schema,
        graph_node_types=graph_node_types,
        graph_edge_types=graph_edge_types,
    )
    result = service.materialize_prediction_task_dataset(
        handle,
        spec,
        predict_staging_dir=predict_staging_dir,
        predict_seed_time_ns=predict_seed_time_ns,
    )
    log.info('[TT-NG] BP materialized task dataset at %s', result.bundle_dir)
    prediction_config.diskgraph_task_dataset_bundle_dir = result.bundle_dir
    return int(result.predict_row_count or 0)


def _custom_pred_entity_metadata(
    pquery: PredictiveQuery, ) -> Tuple[str, str, Dtype]:
    """Single-entity metadata for the custom pred bundle: (entity_table_name,
    entity_pk_column, entity_pk_dtype). Parity with the custom-TRAIN entity
    resolution -- the
    entity comes from the pquery's entity column (single entity only; multi-
    entity is the redesign, explicitly out of scope), and the PK from the
    entity table in the graph. Names are bare (no ``connector.`` prefix) to
    match the DG index convention.
    """
    from kumo.pquery.utils import table_name as _pq_table_name
    validated = pquery.validated_predictive_query
    entity_full = getattr(validated, 'entity_column', None)
    if not entity_full:
        raise PredictionDataMaterializationError(
            message=('TT-NG custom prediction table requires the pquery to '
                     'expose an entity column'),
            non_retryable=True,
        )
    entity_table_full = _pq_table_name(entity_full)
    entity_table_name = (entity_table_full.split('.', 1)[1]
                         if '.' in entity_table_full else entity_table_full)
    entity_table = pquery.graph.table(entity_table_full)
    entity_pk_column = getattr(entity_table, 'pkey_name', None)
    if not entity_pk_column:
        raise PredictionDataMaterializationError(
            message=('TT-NG custom prediction table requires entity table '
                     f'{entity_table_name!r} to have a primary key'),
            non_retryable=True,
        )
    entity_pk = entity_table.pkey
    if entity_pk is None or entity_pk.dtype is None:
        raise PredictionDataMaterializationError(
            message=('TT-NG custom prediction table requires entity table '
                     f'{entity_table_name!r} to have a typed primary key'),
            non_retryable=True,
        )
    return entity_table_name, entity_pk_column, entity_pk.dtype


def _custom_pred_target_stype(pquery: PredictiveQuery) -> Optional[str]:
    """Target stype is OPTIONAL for a prediction bundle (no target column);
    pass it through when the pquery carries it (LP / typed tasks read it), else
    ``None``."""
    target_stype = getattr(pquery.validated_predictive_query, 'target_stype',
                           None)
    value = getattr(target_stype, 'value', None)
    return value if isinstance(value, str) and value else None


def _validate_custom_pred_table_schema(
    *,
    parquet_path: str,
    entity_pk_dtype: Dtype,
    is_temporal: bool,
) -> None:
    """Validate a customer-supplied prediction parquet (closes the historical
    ``# TODO: validate`` in ``_generate_pred_table``).

    Requirements (single-entity parity):
    * ``ENTITY1`` is present, by that name and no other;
    * for a TEMPORAL task, a physical ``TIMESTAMP`` column is required -- the
      per-row anchor time is meaningful and must be supplied. For a STATIC
      task the ``TIMESTAMP`` may be omitted: the DG custom-source processor
      synthesizes a sentinel per-row timestamp at generation (mirroring the
      native PQL/custom-train static path);
    * NO target column -- a prediction table must not carry a label.
    """
    import pyarrow.parquet as pq
    from kumo_core.training import custom_table

    resolved = _resolve_single_pred_parquet_for_schema(parquet_path)
    schema = pq.read_schema(fsutil.s3a_as_s3(resolved))
    cols = set(schema.names)

    entity_name = custom_table.entity_column()
    anchor_name = custom_table.timestamp_column()

    # Named, not matched. The graph's own primary-key column name is not an
    # accepted spelling and neither is a singular `ENTITY`: DiskGraph looks
    # this column up as `ENTITY1`, so a table that does not carry one is
    # refused here rather than read as having no rows it can resolve.
    if entity_name not in cols:
        raise PredictionDataMaterializationError(
            message=(f'TT-NG custom prediction table is missing '
                     f'{entity_name!r}, which is the name DiskGraph reads the '
                     f'entity key under; found {sorted(cols)!r}'),
            non_retryable=True,
        )
    entity_type = schema.field(entity_name).type
    if not custom_table.reads(entity_type, 'entity'):
        raise PredictionDataMaterializationError(
            message=(f'TT-NG custom prediction table column {entity_name!r} '
                     f'is {entity_type}. An entity key is read as one of '
                     f"{custom_table.accepted('entity')}; DiskGraph resolves "
                     f'anything else to no node and drops the row without '
                     f'counting it.'),
            non_retryable=True,
        )
    entity_dtype = custom_table.entity_key_dtype(entity_type)
    if not custom_table.entity_key_resolves(entity_type, entity_pk_dtype):
        raise PredictionDataMaterializationError(
            message=(
                'TT-NG custom prediction table entity column '
                f'{entity_name!r} has type {entity_dtype.value}, but the '
                f'graph primary key requires {entity_pk_dtype.value}. Cast '
                'the entity column to the graph primary-key type.'),
            non_retryable=True,
        )

    if is_temporal and anchor_name not in cols:
        raise PredictionDataMaterializationError(
            message=(f'TT-NG temporal custom prediction table has no '
                     f'{anchor_name} column; a temporal task requires a '
                     f'per-row anchor time; found {sorted(cols)!r}'),
            non_retryable=True,
        )
    if anchor_name in cols:
        anchor_type = schema.field(anchor_name).type
        if not custom_table.reads(anchor_type, 'timestamp'):
            raise PredictionDataMaterializationError(
                message=(f'TT-NG custom prediction table column '
                         f'{anchor_name!r} is {anchor_type}. An anchor is '
                         f"read as one of {custom_table.accepted('timestamp')}"
                         f'; DiskGraph reads anything else as a null anchor '
                         f'and drops the row.'),
                non_retryable=True,
            )

    carried_targets = sorted(name for name in cols
                             if custom_table.column_role(name) == 'target')
    if carried_targets:
        raise PredictionDataMaterializationError(
            message=('TT-NG custom prediction table must not carry a target '
                     f'column; found {carried_targets!r}'),
            non_retryable=True,
        )


def _resolve_single_pred_parquet_for_schema(path: str) -> str:
    """Return a single parquet file to read a schema from (a multi-part dir
    shares one schema across parts, so any part suffices)."""
    if not fsutil.isdir(path):
        return path
    parts = [
        entry for entry in fsutil.listdir(path, suffix='parquet')
        if not fsutil.isdir(entry)
    ]
    if not parts:
        raise PredictionDataMaterializationError(
            message=('TT-NG custom prediction table dir has no parquet parts: '
                     f'{path!r}'),
            non_retryable=True,
        )
    return parts[0]


def _build_ttng_custom_pred_table_artifacts(
    *,
    pred_table_path: str,
    pquery: PredictiveQuery,
    diskgraph_index_path: str,
    ttng_tt_id: str,
) -> str:
    """Gen-only PREDICT twin of
    ``materialize_training_task._build_ttng_custom_train_table_artifacts``.

    Calls the DG PREDICT entry ``build_prediction_table_artifacts_for_existing_table``
    (is_prediction, single ``predict`` stage, no split/target) and persists the
    durable bundle to ``ttng_persisted_pred_staging_dir`` beside the pred table.
    """
    # Pass the pred-table path (single file OR multi-part dir) directly to DG,
    # which globs+reads all parts (enumerate_custom_tt_parts) -- same as the
    # custom-train path. No kumo-side single-parquet narrowing.
    from diskgraph import build_prediction_table_artifacts_for_existing_table
    from kumoapi.pquery import QueryType

    from kumo.data.generic import (
        _diskgraph_legacy_graphstore_bridge,
        diskgraph_staging,
    )
    from kumo.data.generic.diskgraph_train_table import (
        assert_ttng_bundle_complete,
        local_ttng_builder_input,
        persist_durable_ttng_bundle,
    )

    entity_table_name, entity_pk_column, entity_pk_dtype = (
        _custom_pred_entity_metadata(pquery))
    target_stype_value = _custom_pred_target_stype(pquery)
    task_type = pquery.get_task_type()
    task_type_value = getattr(task_type, 'value', None) or str(task_type)
    is_temporal = (pquery.get_batch_pred_table_generator().query_type ==
                   QueryType.TEMPORAL)

    target_entity_table = None
    if bool(getattr(task_type, 'is_link_pred', False)):
        from kumo.pquery.utils import get_rhs_lp_table
        target_entity_table = get_rhs_lp_table(
            pquery.validated_predictive_query,
            pquery.graph,
            no_connector=True,
        )

    col_stats_config: dict[str, Any] = {}
    if target_stype_value:
        col_stats_config['target_stype'] = target_stype_value

    durable = diskgraph_staging.ttng_persisted_pred_staging_dir(pred_table_path)
    # Match the custom-train path: the DG builder uses local filesystem ops,
    # while the staged customer table and durable bundle may live remotely.
    local_input = local_ttng_builder_input(pred_table_path,
                                           'custom_prediction_table.parquet')

    with local_input as local_output_path:
        with fsutil.TemporaryDirectory() as normalized_scratch:
            normalized_path = (_diskgraph_legacy_graphstore_bridge.
                               normalize_legacy_parquet_for_rehydration(
                                   local_output_path,
                                   output_dir=normalized_scratch,
                                   normalize_timestamp=is_temporal,
                               ))
            _validate_custom_pred_table_schema(
                parquet_path=normalized_path,
                entity_pk_dtype=entity_pk_dtype,
                is_temporal=is_temporal,
            )
            result = build_prediction_table_artifacts_for_existing_table(
                output_path=normalized_path,
                tt_id=ttng_tt_id,
                split_stats_config={
                    'entity_table': entity_table_name,
                    'entity_pk_column': entity_pk_column,
                },
                train_table_col_stats_config=col_stats_config,
                index_path=diskgraph_index_path,
                input_timestamp_unit='nanoseconds',
                target_entity_table=target_entity_table,
                task_type=task_type_value,
            )
            staging_dir = result.get('staging_dir')
            if not isinstance(staging_dir, str) or not staging_dir:
                raise RuntimeError(
                    'TT-NG custom prediction table gen did not return '
                    'staging_dir')
            # The native custom builder places its bundle beside
            # ``normalized_path``. Persist it before ``normalized_scratch`` is
            # cleaned up; otherwise copy_tree_threaded sees an empty source and
            # silently publishes no bundle.
            assert_ttng_bundle_complete(staging_dir)
            persist_durable_ttng_bundle(
                staging_dir,
                durable,
                expected_tt_id=ttng_tt_id,
            )
    log.info('[TT-NG] custom pred: persisted bundle %s -> %s (tt_id=%s)',
             staging_dir, durable, ttng_tt_id)
    return durable


def build_ttng_custom_pred_table_bundle(
    job_input: BatchPredictionJobInput,
    settings: MLJobSettings,
    pred_table_path: str,
) -> None:
    """PREDICT twin of
    ``materialize_training_task.build_ttng_custom_train_table_bundle``.

    Gen-only: builds the durable predict-stage staging bundle from the
    customer's prediction parquet and persists it to
    ``ttng_persisted_pred_staging_dir(pred_table_path)`` so
    ``materialize_prediction_task`` can re-ingest it on any pod. Does NOT
    ingest (each BP shard's ensure-extension is the sole ingest).
    """
    predictive_query_config, credentials = (
        hydrate_predictive_query_for_activity(
            job_input.predictive_query,
            job_input.graph_data_compute_input.connectors,
        ))
    pquery: PredictiveQuery = PredictiveQueryV2.from_config(
        predictive_query_config)
    pquery.graph.set_table_data_versions(job_input.id)
    open_connectors(pquery.graph, credentials)

    diskgraph_index_path = get_diskgraph_index_path_for_job(job_input)
    from kumo.data.generic.diskgraph_staging import predict_staging_tt_id

    # SOURCE (gen-keyed) id: identical to the PQL-gen id the consumer resolves
    # from the bundle spec (predict_staging_tt_id(generate_pred_table_job_id or
    # job_input.id)). Custom pred has no separate gen job, so key by the BP job.
    ttng_tt_id = predict_staging_tt_id(str(job_input.id))

    with kumo_timing('prediction_materialize.ttng_custom_pred_source'):
        _build_ttng_custom_pred_table_artifacts(
            pred_table_path=pred_table_path,
            pquery=pquery,
            diskgraph_index_path=diskgraph_index_path,
            ttng_tt_id=ttng_tt_id,
        )


def materialize_prediction_task(
    job_input: BatchPredictionJobInput,
    settings: MLJobSettings,
    serving: bool,
    pred_config_path: str,
    home_dir: Optional[str] = None,
) -> Tuple[str, str, ValidationResponse]:
    r"""
    This activity performs the following operations when using k8s backend:
    - Load the task from the training job's home_dir
    - Create prediction table from pquery.
    - Replace training table with the prediction table.
    - Generate edges from prediction table to the rest of the graph.
    - Save all the information in :obj:`BatchPredictionJobConfig` and return
        the path.
    """
    from rest.backends.data_backend.utils import get_data_operation_description
    validator = validation.MaterializeTaskValidator(
        max_total_table_size=settings.max_total_table_size,
        max_total_graph_edges=settings.max_total_graph_edges,
    )
    response = ValidationResponse()

    # Parse the required components from job_input, settings.
    job_id = job_input.id
    training_job_id = job_input.training_job_id
    prediction_config = PredictionConfig.load(pred_config_path)

    if home_dir is None:
        home_dir = get_job_dir(get_job_base_dir(), job_id)
    home_dir_mgr = HomeDirManager(home_dir)
    train_dir = get_job_dir(get_job_base_dir(), training_job_id)
    train_dir_mgr = HomeDirManager(train_dir)
    if not fsutil.exists(train_dir):
        raise MissingTrainingJobError(
            message=(f"Training job '{training_job_id}' not found. "
                     "The job may have been deleted or not yet completed."))

    # The "materializing" status is published on the pool worker's event loop
    # by the ``SnapshotActivities.materialize_prediction_task`` wrapper before
    # this runs; no StateDB write happens on this diskgraph worker thread.
    diskgraph_index_path = prediction_config.diskgraph_index_path
    # TT-NG-only: prediction is always DiskGraph-backed. An absent index path
    # means the graph binding (graph_data_compute_input / graph_version_info)
    # was missing or corrupt when the prediction config was built -- there is no
    # legacy fallback to run. Reject at the activity boundary with a precise,
    # non-retryable error rather than proceeding into a half-bound state.
    if not diskgraph_index_path:
        raise PredictionDataMaterializationError(
            message=(
                'No DiskGraph index is bound to this prediction job; the graph '
                'binding is missing or corrupt. Re-ingest the graph for this '
                'version and re-run prediction.'),
            non_retryable=True,
        )
    # No process globals: neither temp_home_dir nor per-job engine setup
    # runs. home_dir is threaded explicitly (HomeDirManager,
    # Task.load(home_dir=train_dir), task.materialize persist_dir=/local_dir=
    # below), so concurrent jobs on the pool worker's shared thread executor
    # can't collide. The diskgraph index path is threaded to task.materialize
    # and the DG services; nothing reads the global engine state
    # (engine_backend defaults to DISKGRAPH; engine_mode reads are escaped by
    # the explicit index path).
    # Create pquery object after setting pquery backend, so the pquery
    # backend setting gets correctly passed.
    predictive_query_config, credentials = (
        hydrate_predictive_query_for_activity(
            job_input.predictive_query,
            job_input.graph_data_compute_input.connectors,
        ))
    with kumo_timing("prediction_task.pquery_from_config"):
        pquery: PredictiveQuery = PredictiveQueryV2.from_config(
            predictive_query_config)

    # Load the task config from the training job dir. Note that we force
    # ignore all training directory data here since we will re-materialize
    # the data at the latest data version available:

    # NOTE: A bit hacky but a way to fetch the pquery name and make it work
    # with both named (V1) and unnamed (V2) object models.
    pq_dir_name = train_dir_mgr.first_task_name()
    if not pq_dir_name:
        raise MissingTrainingJobError(
            message=(f"Training job '{training_job_id}' appears incomplete. "
                     "The required model data was not found. Please ensure the "
                     "training job completed successfully before running batch "
                     "prediction."))
    with kumo_timing("prediction_task.load_training_task"):
        task = Task.load(pq_dir_name, home_dir=train_dir)
    # NOTE: Update the task name to NA because when starting kumo-pred,
    # the GPU worker will try to load the pred task config using the
    # task name saved in the prediction job JSON file. Which
    # will be NA.
    task.name = pquery.name

    # Weight column is not present and not needed
    # during prediction.
    if task.train_table is not None:
        task.train_table._weight_col = None

    # Validate forecasting compatibility
    if task.task_type == TaskType.FORECASTING:
        if task.lag_timesteps != job_input.predictive_query.lag_timesteps:
            raise IncompatiblePredictionTable(
                message=("This prediction table is not compatible with "
                         "the model that was requested. You specified "
                         f"lag_timesteps = {task.lag_timesteps} in your "
                         "TrainingTableGenerationPlan when training the "
                         "model, but you specified lag_timesteps = "
                         f"{job_input.predictive_query.lag_timesteps} in "
                         "your PredictionTableGenerationPlan when "
                         "creating the prediction table. These must match."))
        if (task.forecast_length != job_input.predictive_query.forecast_length):
            raise IncompatiblePredictionTable(
                message=("This prediction table is not compatible with "
                         "the model that was requested. You specified "
                         f"forecast_length = {task.forecast_length} in "
                         "your TrainingTableGenerationPlan when training "
                         "the model, but you specified forecast_length = "
                         f"{job_input.predictive_query.forecast_length} "
                         "in your PredictionTableGenerationPlan when "
                         "creating the prediction table. These must match."))

    # TODO(manan): this is a hacky way of ensuring that the graph that is
    # input at prediction time is the one that is to be used with the
    # materialized prediction task. Sadly, this is a bit more involved
    # than just `task.graph = pquery.graph`, since the graph informs
    # other properties of the task, including training table properties.
    # This is likely brittle and needs further investment:
    pquery.graph.copy_encoders(task.graph)
    task.graph = pquery.graph
    task.target_to_entity_paths = pquery.target_to_entity_paths

    if pquery.get_task_type() == TaskType.STATIC_LINK_PREDICTION:
        task.target_table_name = (
            pquery.get_table_generator().get_target_table_name())

    for fkey in task.train_table.fkeys:
        new_dst_table = task.graph.table(fkey.dst_table_name)
        fkey.dst_table = new_dst_table.name
        fkey.dst_data_source = new_dst_table.data_source

    # Set table data versions to the one decided at the beginning
    # of the batch prediction workflow
    log.info(f"Using batch prediction job id {job_input.id} to update table "
             f"versions for prediction task.")
    task.graph.set_table_data_versions(job_input.id)

    # Generate prediction table ###########################################
    log.debug("Opening Connectors '%s' for batch prediction: %s",
              task.graph.data_sources, job_input.id)
    with kumo_timing("prediction_task.open_connectors"):
        open_connectors(task.graph, credentials)

    # TT-NG: prediction input stays unpartitioned (num_partitions == 1);
    # output_num_partitions is set below from the predict-stage row count
    # after _materialize_ttng_prediction_task_dataset. The pred table is
    # served as-is -- no partitioning.
    prediction_config.num_partitions = 1
    prediction_config.output_num_partitions = None
    partitioned_bp_table_path = prediction_config.pred_table_path

    connector = ParquetConnector(
        name=train_table_source_name(partitioned_bp_table_path),
        # Removing last occurrence of 'pred_table' name from the path
        # since that will already be included when the final path if
        # generated from parquet connector based on the name of the table
        root_dir=osp.dirname(partitioned_bp_table_path),
    )
    # Remove cached data version for previously generated tables:
    get_active_cache().table_version_cache.remove(connector.name)

    # Capture the train-table name before the pred-table swap; the TT-NG
    # predict manifest reuses it so node types match the checkpoint.
    ttng_train_table_name = task.train_table.table_name
    task = task._replace_train_table(PRED_TABLE, data_source=connector)
    task.fixed_split([1.0, 0.0, 0.0])

    durable_staging_dir, source_tt_id = (_load_ttng_prediction_bundle_identity(
        prediction_config.pred_table_path))
    # ``Task.validate`` touches ``train_table.data_version``. For TT-NG the
    # durable bundle, not the raw prediction parquet, is authoritative. Set its
    # identity before validation so a direct warehouse source does not trigger
    # a bogus lookup for ``pred_table`` in the temporary ParquetConnector.
    task.train_table._data_version = source_tt_id

    try:
        task.validate(raise_on_error=True, ignore_encoders=True)
        task._set_forecast_col_encoder(
            job_input.predictive_query.year_over_year)

        # TT-NG inference: ingest the kept predict staging dir + write the
        # manifest; the trainer serves via the wheel iterator.
        with kumo_timing("prediction_task.materialize_ttng_prediction_dataset"):
            predict_row_count = (_materialize_ttng_prediction_task_dataset(
                task=task,
                pquery=pquery,
                job_input=job_input,
                prediction_config=prediction_config,
                diskgraph_index_path=diskgraph_index_path,
                train_table_name=ttng_train_table_name,
                durable_staging_dir=durable_staging_dir,
                source_tt_id=source_tt_id,
            ))
        prediction_config.output_num_partitions = (
            _pred_num_partitions_from_rows(predict_row_count, prediction_config,
                                           job_input, settings, task,
                                           id_is_string=False))
        log.info("[TT-NG] prediction task-dataset materialized")
    except TrainingTableSplitError as e:
        response.errors.append(e.as_error())
        return ("", "", response)
    except Exception as e:
        raise PredictionDataMaterializationError(
            message="Failed to prepare prediction data",
            error=e,
            non_retryable=is_error_non_retryable(e),
        ) from e

    # TT-NG prediction is always DiskGraph-backed (diskgraph_index_path is
    # asserted non-empty above), so input stats come from the DG index.
    from kumo.data.cache.graph_cache import VersionedGraphKey
    from kumo.diskgraph.input_stats import assemble_input_data_stats
    dg_handle = DiskgraphHandle(
        graph_key=(job_input.graph_data_compute_input.graph_key),
        graph_data_version=(job_input.graph_version_info.graph_data_version),
    )
    with kumo_timing("prediction_task.assemble_input_stats"):
        data_stats = assemble_input_data_stats(
            dg_handle,
            graph=task.graph,
            graph_key_ver=VersionedGraphKey(dg_handle.graph_key,
                                            dg_handle.graph_data_version),
            index_path=diskgraph_index_path,
        )
    log.info(f"Computed InputDataStats: {data_stats} for job: "
             f"{job_id}")
    with catch_and_log():
        save_input_data_stats_to_state_db(job_id, data_stats)

    with kumo_timing("prediction_task.post_materialization_validation"):
        response.merge(validator.post_materialization(data_stats))
    if not response.ok:
        return ("", "", response)

    log.info("Saving task to prediction home_dir")
    # home_dir is threaded explicitly into Task.save (-> graph.save(destination=))
    # so the task/graph configs land in THIS prediction job's home_dir without a
    # process-global temp_home_dir; concurrent jobs on the shared thread executor
    # can't collide.
    with kumo_timing("prediction_task.save_task"):
        # Save the task in the prediction home_dir.
        task.save(destination=home_dir)
    log.info("Task saved successfully")

    # Save the persist paths of tables and edges.
    # In diskgraph mode, graph data is not materialized through the traditional
    # path so persist paths are not available on the graph object.
    home_dir_mgr = HomeDirManager(home_dir)
    # Save the input data stats to the job directory:
    with kumo_timing("prediction_task.save_input_stats"):
        save_json(data_stats, home_dir_mgr.input_data_stats_path)
    # Save the updated prediction config to the job directory.
    with kumo_timing("prediction_task.save_prediction_config"):
        save_json(prediction_config, pred_config_path)

    # Save stats about the materialization to S3.
    materialization_info = make_materialization_info(
        job_input.start_time,
        task.graph,
        job_id=job_input.id,
        graph_id=job_input.graph_id,
        predictive_query_id=job_input.predictive_query_id,
        temporal_workflow_id=activity.info().workflow_run_id,
    )
    with kumo_timing("prediction_task.save_materialization_info"):
        save_materialization_info(materialization_info)

    return (pred_config_path, get_data_operation_description(), response)
