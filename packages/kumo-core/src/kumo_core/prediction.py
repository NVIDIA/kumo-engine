# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable DiskGraph batch-prediction table and task materialization."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)
from uuid import uuid4

import diskgraph
from pandas import Timestamp

from .pquery import (
    NativeGraph,
    ValidatedNativeQuery,
    _temporal_target_time_range,
)
from .prediction_options import resolve_prediction_time_options
from .task_dataset import (
    TaskDatasetMaterializeResult,
    TaskDatasetMaterializeSpec,
    write_flat_manifest,
    write_thin_manifest,
)
from .train.autoregression import FUTURE_LABELS_COL
from .training.table import apply_reference_resolution
from .training.validation import validate_anchor_time
from .types.enums import PredictionArtifactType

if TYPE_CHECKING:
    from .embedding_models import EmbeddingModelStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionTableRequest:
    query: ValidatedNativeQuery
    graph: NativeGraph
    timestamps: tuple[int, ...]
    index_path: str
    tt_id: str
    task_dataset: TaskDatasetMaterializeSpec
    compiler_options: Mapping[str, Any] | None = None
    forecast_timeframe_step: int | None = None


@dataclass(frozen=True)
class GeneratedPredictionTable:
    staging_dir: str
    table_path: str
    row_count: int
    tt_id: str


@dataclass(frozen=True)
class PreparedCustomPredictionTable:
    """Connector-staged prediction rows ready for DiskGraph."""

    parquet_path: str
    time_column: str = "TIMESTAMP"

    def __post_init__(self) -> None:
        if not self.parquet_path:
            raise ValueError("custom prediction table requires a path")


def forecast_task_options(
    request: PredictionTableRequest, ) -> Dict[str, int] | None:
    """Return the resolved-task scalars required to replay a forecast.

    Prediction-table compilation owns the temporal query and the exact lag
    features it emitted. Passing these small values to every scoring process
    keeps spawned workers aligned with the fitted checkpoint.
    """
    if request.query.task_type != "forecasting":
        return None
    options = dict(request.compiler_options or {})
    lag_offsets = options.get("lag_offsets_secs")
    if not isinstance(lag_offsets, (list, tuple)) or not lag_offsets:
        raise ValueError("Forecast prediction requires historical lag offsets")
    offset = request.query.query.get_combined_date_offset_range()
    start = int(offset.start or 0)
    end = int(offset.end)
    step = request.forecast_timeframe_step or end - start
    from .utils.time import compute_unix_step
    return {
        "forecast_length": int(request.query.forecast_length),
        "lag_timesteps": len(lag_offsets),
        "unix_start_offset": int(compute_unix_step(start, offset.unit)),
        "unix_end_offset": int(compute_unix_step(end, offset.unit)),
        "unix_step_size": int(compute_unix_step(step, offset.unit)),
    }


def _validate_forecast_checkpoint(trainer: Any, task: Any) -> None:
    """Reject a checkpoint whose fitted forecast inputs differ from task."""
    past_encoder = trainer.past_encoder
    trained_lags = (getattr(past_encoder, "length", None)
                    if past_encoder is not None else getattr(
                        trainer.head, "past_channels", None))
    if (trained_lags != task.lag_timesteps
            or (trainer.year_over_year_encoder is not None)
            != task.has_year_over_year_labels or getattr(
                trainer.head, "unix_step_size", None) != task.unix_step_size):
        raise ValueError(
            "Forecast checkpoint lag/year-over-year/timing mismatch")
    fitted = (trainer.encoders.encoders.get(task.train_table.name)
              if trainer.encoders is not None else None)
    if fitted is None or any(
            value.column_name not in fitted
            for value in task.encoder_inputs.get(task.train_table.name, ())):
        raise ValueError("Forecast checkpoint lacks feature encoders")


def generate_prediction_table(
    request: PredictionTableRequest,
    *,
    runtime: Any | None = None,
) -> GeneratedPredictionTable:
    """Compile PQL in prediction mode and retain its TT-NG staging bundle."""

    runtime = runtime or diskgraph
    query = request.query
    options: dict[str, Any] = {
        "query": query.query,
        "timestamps": list(request.timestamps),
        "task_type": query.task_type,
        "input_timestamp_unit": "nanoseconds",
        "is_prediction": True,
        "target_stype": query.target_stype,
        "entity_time_col": request.graph.table(query.entity_table).time_column,
    }
    forecast_options = dict(request.compiler_options or {})
    allowed = {"lag_offsets_secs", "yoy_offsets_secs", "future_offsets_secs"}
    if (forecast_options.keys() - allowed
            or forecast_options and query.task_type != "forecasting"):
        raise ValueError(
            "prediction compiler_options require a forecasting query and "
            "only support forecast offsets")
    if query.query_type == "temporal" and request.timestamps:
        minimum, _, _ = _temporal_target_time_range(query, request.index_path)
        aggregation = query.query.get_all_target_aggregations()[0]
        table_name = str(aggregation.time_col).rsplit(".", 1)[0]
        for timestamp in request.timestamps:
            error = validate_anchor_time(Timestamp(timestamp, unit="ns"),
                                         minimum, table_name)
            if error is not None:
                raise ValueError(f"{error.title}. {error.message}")
    compiler = runtime.DiskGraphPlanCompiler(index_path=request.index_path)
    # Future targets belong to training only. Prediction retains the fitted
    # historical feature layout without asking the compiler for labels.
    forecast_options.pop("future_offsets_secs", None)
    options.update(forecast_options)
    if query.task_family == "link_prediction":
        options.update({
            "lp_prediction_mode": "LHS",
            "list_distinct_target_is_array": True,
        })
    plan_json = compiler.compile(**options)
    plan_json = apply_reference_resolution(
        plan_json,
        entity_node_type=query.entity_table,
        rhs_node_type=None,
        resolve_link_targets=False,
        is_static_link=False,
        rhs_is_temporal=False,
    )
    generated = runtime.execute_tt_gen(
        plan_json,
        request.index_path,
        request.tt_id,
    )
    staging_dir = generated.get("staging_dir")
    table_path = generated.get("tt_parquet_path")
    row_count = int(generated.get("row_count") or 0)
    if not isinstance(staging_dir, str) or not staging_dir:
        raise RuntimeError(
            "prediction execute_tt_gen result missing staging_dir")
    if not isinstance(table_path, str) or not table_path:
        raise RuntimeError(
            "prediction execute_tt_gen result missing tt_parquet_path")
    if row_count <= 0:
        raise RuntimeError("prediction query produced no rows")
    return GeneratedPredictionTable(
        staging_dir=staging_dir,
        table_path=table_path,
        row_count=row_count,
        tt_id=request.tt_id,
    )


def generate_custom_prediction_table(
    request: PredictionTableRequest,
    custom_table: PreparedCustomPredictionTable,
    *,
    runtime: Any | None = None,
) -> GeneratedPredictionTable:
    """Build the standard predict-stage bundle from caller-supplied rows."""
    if not isinstance(custom_table, PreparedCustomPredictionTable):
        raise TypeError(
            "custom_table must be PreparedCustomPredictionTable")
    if not Path(custom_table.parquet_path).exists():
        raise RuntimeError("custom prediction table parquet does not exist: "
                           f"{custom_table.parquet_path!r}")
    runtime = runtime or diskgraph
    entity = request.graph.table(request.query.entity_table)
    result = runtime.build_prediction_table_artifacts_for_existing_table(
        output_path=custom_table.parquet_path,
        input_timestamp_unit="nanoseconds",
        tt_id=request.tt_id,
        split_stats_config={
            "entity_table": request.query.entity_table,
            "entity_pk_column": entity.primary_key or "",
        },
        train_table_col_stats_config={
            "target_stype": request.query.target_stype,
            "time_col": custom_table.time_column,
        },
        index_path=request.index_path,
        target_entity_table=request.query.rhs_node_type,
        task_type=request.query.task_type,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError(
            "custom prediction-table builder did not return a mapping")
    staging_dir = result.get("staging_dir")
    table_path = result.get("tt_parquet_path")
    row_count = int(result.get("row_count") or 0)
    if not isinstance(staging_dir, str) or not staging_dir:
        raise RuntimeError(
            "custom prediction-table builder result missing staging_dir")
    if not isinstance(table_path, str) or not table_path:
        raise RuntimeError(
            "custom prediction-table builder result missing tt_parquet_path")
    if row_count <= 0:
        raise RuntimeError("custom prediction table contains no rows")
    return GeneratedPredictionTable(
        staging_dir=staging_dir,
        table_path=table_path,
        row_count=row_count,
        tt_id=request.tt_id,
    )


def materialize_prediction_task(
    request: PredictionTableRequest,
    generated: GeneratedPredictionTable,
    *,
    graph_index_id: str,
    runtime: Any | None = None,
) -> TaskDatasetMaterializeResult:
    """Ingest the retained prediction table and write a predict-only bundle."""

    runtime = runtime or diskgraph
    forecasting = (request.query.task_type == "forecasting"
                   and request.compiler_options is not None)
    ingested = runtime.execute_tt_ingest(
        staging_dir=generated.staging_dir,
        index_path=request.index_path,
        tt_id=request.tt_id,
        predict_stage_name="predict",
    )
    if not isinstance(ingested, Mapping):
        raise RuntimeError("prediction execute_tt_ingest returned no manifest")
    task_dataset = ingested.get("task_dataset")
    if not isinstance(task_dataset, Mapping):
        raise RuntimeError(
            "prediction execute_tt_ingest result missing task_dataset")
    stages = task_dataset.get("stages")
    predict = stages.get("predict") if isinstance(stages, Mapping) else None
    if not isinstance(predict, Mapping):
        raise RuntimeError("prediction task dataset missing predict stage")
    table_name = predict.get("table_name")
    row_count = predict.get("row_count")
    index_path = predict.get("index_path")
    if not isinstance(table_name, str) or not table_name:
        raise RuntimeError("prediction stage missing table_name")
    if not isinstance(row_count, int) or row_count < 0:
        raise RuntimeError("prediction stage has invalid row_count")
    if not isinstance(index_path, str) or not index_path:
        raise RuntimeError("prediction stage missing index_path")
    extension_path = task_dataset.get("tt_extension_path")
    if not isinstance(extension_path, str) or not extension_path:
        extension_path = os.path.join(
            request.index_path,
            "train_tables",
            request.tt_id,
        )

    feature_schema = request.task_dataset.feature_schema
    if forecasting:
        feature_schema = deepcopy(feature_schema)
        features = feature_schema["node_types"][
            request.task_dataset.train_table_name]
        features["feature_columns"] = [
            column for column in features["feature_columns"]
            if column != FUTURE_LABELS_COL
        ]
        features["column_specs"] = [
            column for column in features["column_specs"]
            if column["column_name"] != FUTURE_LABELS_COL
        ]
    spec = replace(
        request.task_dataset,
        tt_manifest_path=os.path.join(extension_path, "manifest.json"),
        train_table_name=(request.task_dataset.train_table_name
                          if forecasting else table_name),
        feature_schema=feature_schema,
    )
    stage_counts = {"predict": row_count}
    stage_table_refs = {
        "predict": {
            "table_name": table_name,
            "index_path": index_path,
            "storage_kind": "diskgraph_table",
        }
    }
    stage_seed_times = None
    if request.timestamps:
        stage_seed_times = {
            "predict": [min(request.timestamps),
                        max(request.timestamps)]
        }
    manifest_path = write_thin_manifest(
        spec,
        graph_index_id=graph_index_id,
        graph_index_path=request.index_path,
        stats={},
        stage_counts=stage_counts,
        stage_seed_times=stage_seed_times,
        stage_table_refs=stage_table_refs,
        has_time=True,
        has_weight=False,
        column_stats=False,
    )
    bundle_dir = os.path.dirname(manifest_path)
    runtime_manifest_path = write_flat_manifest(
        spec,
        bundle_dir=bundle_dir,
        graph_index_path=request.index_path,
        stage_index_path=extension_path,
        stage_counts=stage_counts,
        stage_table_refs=stage_table_refs,
        has_time=True,
        has_weight=False,
    )
    return TaskDatasetMaterializeResult(
        manifest_path=manifest_path,
        runtime_manifest_path=runtime_manifest_path,
        bundle_dir=bundle_dir,
    )


__all__ = [
    "GeneratedPredictionTable",
    "PreparedCustomPredictionTable",
    "PredictionTableRequest",
    "forecast_task_options",
    "generate_prediction_table",
    "generate_custom_prediction_table",
    "materialize_prediction_task",
    "score_prediction",
]


def _prediction_worker_entry(
    score_worker: Callable[..., Any],
    worker_id: int,
    error_sender: Any,
    worker_kwargs: Mapping[str, Any],
) -> None:
    r"""Run one scorer in an isolated process and report a short failure."""
    try:
        result = score_worker(worker_id=worker_id, **worker_kwargs)
    except BaseException as error:
        error_sender.send((type(error).__name__, str(error)[:4096]))
        raise
    else:
        error_sender.send(result)
    finally:
        error_sender.close()


def _run_prediction_workers(
    worker_count: int,
    score_worker: Callable[..., Any],
    worker_kwargs: Mapping[str, Any],
) -> list[Any]:
    r"""Run scorers in spawned sibling processes and wait for every worker.

    Each scorer creates a PyTorch DataLoader process pool. A spawned scorer
    starts with independent Python, CUDA, lock, and DataLoader state.
    """
    context = mp.get_context('spawn')
    workers = []
    try:
        for worker_id in range(worker_count):
            error_receiver, error_sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_prediction_worker_entry,
                args=(score_worker, worker_id, error_sender, worker_kwargs),
                name=f'kumo-bp-{worker_id}',
            )
            try:
                process.start()
            except BaseException:
                error_sender.close()
                error_receiver.close()
                raise
            error_sender.close()
            workers.append((worker_id, process, error_receiver))
    except BaseException:
        for _, process, error_receiver in workers:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
            error_receiver.close()
        raise

    failures = []
    results = []
    for worker_id, process, error_receiver in workers:
        process.join()
        try:
            report = error_receiver.recv() if error_receiver.poll() else None
        except EOFError:
            report = None
        error_receiver.close()
        results.append(report)
        exit_code = process.exitcode
        process.close()
        if exit_code:
            detail = (report[1] or report[0]) if report else (
                f'exited with code {exit_code}')
            error = RuntimeError(
                f'Prediction worker {worker_id} failed: {detail}')
            logger.error('%s', error)
            failures.append(error)
    if failures:
        raise failures[0]
    return results


def _prediction_worker_devices(
    num_workers: int,
    accelerator: str = 'auto',
) -> Tuple[Optional[int], ...]:
    r"""Assign concurrent scorers to distinct visible CUDA devices.

    ``None`` leaves Lightning's automatic non-CUDA selection in charge. A
    CUDA run with more workers than visible devices is rejected because every
    process loads its own checkpoint copy.
    """
    if num_workers == 1:
        return (None, )
    if accelerator in {'cpu', 'mps'}:
        return (None, ) * num_workers

    import torch

    count = torch.cuda.device_count()
    if count == 0:
        return (None, ) * num_workers
    if num_workers > count:
        raise ValueError(
            f'num_workers={num_workers} requires {num_workers} visible CUDA '
            f'devices, but only {count} are visible')
    return tuple(range(num_workers))


class _PredictionWorkerConfig(NamedTuple):
    worker_count: int
    devices: Tuple[Optional[int], ...]
    manifest_path: str
    bundle_dir: str
    model_dir: str
    run_id: str
    artifacts: FrozenSet[PredictionArtifactType]
    top_k: Optional[int]
    extra_fields: Optional[Dict[str, List[str]]]
    job_start_time: Optional[datetime]
    pred_time_col_name: Optional[str]
    output_dir: str
    attempt_id: str
    graph_snapshot_id: str
    source_config_sha256: str
    runtime_config_sha256: str
    binary_threshold: Optional[float]
    forecast_options: Optional[Dict[str, int]]
    embedding_models: Optional[EmbeddingModelStore] = None
    ignore_time_col: bool = False
    accelerator: str = 'auto'


def _score_prediction_shard(
    *,
    worker_id: int,
    config: _PredictionWorkerConfig,
    manifest: Any = None,
    encoders: Any = None,
) -> Any:
    r"""Load and score one prediction-table shard in its worker process.

    An in-process caller passes the ``manifest`` and ``encoders`` it already
    loaded, so each is parsed and held once.
    """
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.train.trainer import load_encoders
    from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime

    if manifest is None:
        manifest = DiskgraphTaskManifest.load(config.manifest_path)
    task = ResolvedNumericalTask.from_manifest(
        manifest,
        name=config.run_id,
        **(config.forecast_options or {}),
    )
    load_options = {'accelerator': config.accelerator}
    device = config.devices[worker_id]
    if device is not None:
        load_options['devices'] = [device]
    if config.binary_threshold is not None:
        load_options['binary_threshold'] = config.binary_threshold
    trainer = DiskgraphNestedTrainer.load(
        task,
        config.model_dir,
        diskgraph_task_dataset_bundle_dir=config.bundle_dir,
        encoders=(encoders if encoders is not None else load_encoders(
            config.model_dir)),
        infer_config_path=False,
        diskgraph_runtime=LocalDiskgraphTrainingRuntime(),
        embedding_models=config.embedding_models,
        **load_options,
    )
    if config.forecast_options is not None:
        _validate_forecast_checkpoint(trainer, task)
    from kumo_core.train.artifact_stats import PredictionArtifactStatsCollector
    shard_attempt = f'{config.attempt_id}-{worker_id}'
    generation = (Path(config.output_dir) / '_attempts' /
                  f'runtime-{worker_id:05d}' / shard_attempt)
    collector = PredictionArtifactStatsCollector(
        str(generation),
        tuple(sorted(config.artifacts, key=lambda kind: kind.value)))
    trainer.predict(
        output_dir=str(generation),
        artifact_stats_collector=collector,
        task=task,
        return_predictions=(PredictionArtifactType.PREDICTIONS
                            in config.artifacts),
        return_embeddings=(PredictionArtifactType.EMBEDDINGS
                           in config.artifacts),
        top_k=config.top_k,
        output_extra_fields=config.extra_fields or None,
        job_start_time=config.job_start_time,
        pred_time_col_name=config.pred_time_col_name,
        ignore_time_col=config.ignore_time_col,
        partition_id=worker_id,
        total_partitions=config.worker_count,
        output_partition_ids=[worker_id],
        runtime_num_workers=config.worker_count,
        runtime_worker_id=worker_id,
        task_manifest=manifest,
    )
    return _publish_scored_shard(
        collector, output_dir=config.output_dir, run_id=config.run_id,
        attempt_id=shard_attempt, graph_snapshot_id=config.graph_snapshot_id,
        source_config_sha256=config.source_config_sha256,
        runtime_config_sha256=config.runtime_config_sha256, worker_id=worker_id,
        worker_count=config.worker_count)


def score_prediction(
    *,
    materialized: Any,
    model_dir: str,
    run_id: str,
    scratch_dir: str,
    artifacts: FrozenSet[PredictionArtifactType],
    top_k: Optional[int] = None,
    extra_fields: Optional[Dict[str, List[str]]] = None,
    started: Optional[datetime] = None,
    num_workers: int = 1,
    output_dir: str | None = None,
    graph_snapshot_id: str,
    source_config_sha256: str,
    binary_threshold: float | None = None,
    forecast_options: Mapping[str, int] | None = None,
    embedding_models: EmbeddingModelStore | None = None,
    encoders: Any = None,
    accelerator: str = 'auto',
) -> Any:
    """Score partitions and publish bounded metadata using the hosted protocol.

    Rows stay in producer Parquet files. Workers return only small immutable
    manifest references; selection publishes after every scorer succeeds.
    ``encoders`` are the model's, if the caller already loaded them; only an
    in-process scorer reuses them.
    """
    from .prediction_options import validate_prediction_workers
    validate_prediction_workers(num_workers)
    from .task_dataset import DiskgraphTaskManifest

    prediction_dir = Path(output_dir) if output_dir is not None else (
        Path(scratch_dir) / 'predict_output' / run_id)
    output_dir = str(prediction_dir)
    from kumo_core.train.artifact_manifest import (
        current_prediction_selection_ref,
        hash_prediction_config,
        load_current_prediction_selection_metadata,
        publish_prediction_selection,
    )
    if (prediction_dir / '_selected_artifacts.json').exists():
        selected = load_current_prediction_selection_metadata(output_dir)
        if (selected.job_id != run_id
                or selected.graph_snapshot_id != graph_snapshot_id
                or selected.source_config_sha256 != source_config_sha256
                or {item.artifact_type
                    for item in selected.artifacts} != set(artifacts)):
            raise ValueError(
                'Prediction output is already owned by another selection')
        return current_prediction_selection_ref(output_dir)
    runtime_config_sha256 = hash_prediction_config(
        str(Path(model_dir) / 'config.json'))
    manifest = DiskgraphTaskManifest.load(materialized.manifest_path)
    seed_count = manifest.stage_counts.get('predict')
    # Preserve the caller's existing no-output contract without launching a
    # scorer for an empty population or creating zero-sized partitions.
    if seed_count == 0:
        raise RuntimeError(
            'prediction was asked for output but wrote no parquet')
    worker_count = min(num_workers,
                       seed_count) if seed_count is not None else num_workers
    # Select per artifact, as hosted does, so the default output matches.
    selection = {
        artifact.value: (extra_fields or {}).get(artifact.value, [])
        for artifact in artifacts
    }
    time_options = resolve_prediction_time_options(manifest.task_type,
                                                   selection)
    attempt_id = uuid4().hex
    config = _PredictionWorkerConfig(
        worker_count=worker_count,
        devices=_prediction_worker_devices(worker_count, accelerator),
        manifest_path=materialized.manifest_path,
        bundle_dir=materialized.bundle_dir,
        model_dir=model_dir,
        run_id=run_id,
        artifacts=artifacts,
        top_k=top_k,
        extra_fields=selection,
        job_start_time=started if time_options.include_job_timestamp else None,
        pred_time_col_name=time_options.pred_time_col_name,
        ignore_time_col=time_options.ignore_time_col,
        output_dir=output_dir,
        attempt_id=attempt_id,
        graph_snapshot_id=graph_snapshot_id,
        source_config_sha256=source_config_sha256,
        runtime_config_sha256=runtime_config_sha256,
        binary_threshold=binary_threshold,
        forecast_options=(dict(forecast_options)
                          if forecast_options is not None else None),
        embedding_models=embedding_models,
        accelerator=accelerator,
    )
    if worker_count == 1:
        references = [
            _score_prediction_shard(worker_id=0, config=config,
                                    manifest=manifest, encoders=encoders)
        ]
    else:
        # Spawned scorers load their own copies.
        del manifest, encoders
        from .embedding_models import require_importable
        require_importable(embedding_models)
        references = _run_prediction_workers(worker_count,
                                             _score_prediction_shard,
                                             {'config': config})
    return publish_prediction_selection(
        references, job_id=run_id, selection_id=run_id,
        selection_attempt_id=attempt_id, canonical_output_root=output_dir,
        graph_snapshot_id=graph_snapshot_id,
        source_config_sha256=source_config_sha256, expected_shards=worker_count,
        expected_output_partitions=worker_count)


def _publish_scored_shard(
    collector: Any,
    *,
    output_dir: str,
    run_id: str,
    attempt_id: str,
    graph_snapshot_id: str,
    source_config_sha256: str,
    runtime_config_sha256: str,
    worker_id: int = 0,
    worker_count: int = 1,
) -> Any:
    """Publish a local scorer partition through the hosted ownership format."""
    from kumo_core.train.artifact_manifest import (
        PREDICTION_ARTIFACT_MANIFEST_VERSION,
        PredictionShardManifest,
        inventory_generation,
        prediction_shard_producer_abi,
        publish_shard_manifest,
    )
    from kumo_core.train.artifact_stats import PredictionArtifactStatsRequest

    # Each scorer owns one disjoint output partition; use writer statistics,
    # not estimated entity counts or a reread of the output.
    request = PredictionArtifactStatsRequest(
        attempt_id=attempt_id,
        job_id=run_id,
        runtime_config_sha256=runtime_config_sha256,
        output_dir=collector.output_dir,
        runtime_shard_id=worker_id,
        runtime_shard_count=worker_count,
        output_partition_start=worker_id,
        output_partition_stop=worker_id + 1,
        expected_artifact_types=collector.expected_artifact_types,
    )
    payload = collector.build_payload(request)
    if worker_count == 1 and not payload.written_partition_count:
        # The existing single-worker scorer also rejected missing Parquet.
        # Do not change hosted all-empty selection semantics here.
        raise RuntimeError(
            'prediction was asked for output but wrote no parquet')
    manifest = PredictionShardManifest(
        schema_version=PREDICTION_ARTIFACT_MANIFEST_VERSION,
        job_id=run_id,
        attempt_id=attempt_id,
        graph_snapshot_id=graph_snapshot_id,
        source_config_sha256=source_config_sha256,
        runtime_config_sha256=runtime_config_sha256,
        producer_abi=prediction_shard_producer_abi(worker_count),
        canonical_output_root=output_dir,
        generation_root=collector.output_dir,
        runtime_shard_id=worker_id,
        runtime_shard_count=worker_count,
        output_partition_start=worker_id,
        output_partition_stop=worker_id + 1,
        requested_artifacts=collector.expected_artifact_types,
        artifacts=inventory_generation(collector.output_dir,
                                       collector.expected_artifact_types,
                                       payload),
    )
    return publish_shard_manifest(manifest)


def publish_scored_output(collector: Any, **kwargs: Any) -> Any:
    """Publish one local scorer through the shared shard/selection protocol."""
    from kumo_core.train.artifact_manifest import publish_prediction_selection

    reference = _publish_scored_shard(collector, **kwargs)
    return publish_prediction_selection(
        [reference],
        job_id=kwargs["run_id"],
        selection_id=kwargs["run_id"],
        selection_attempt_id=kwargs["attempt_id"],
        canonical_output_root=kwargs["output_dir"],
        graph_snapshot_id=kwargs["graph_snapshot_id"],
        source_config_sha256=kwargs["source_config_sha256"],
        expected_shards=1,
        expected_output_partitions=1,
    )
