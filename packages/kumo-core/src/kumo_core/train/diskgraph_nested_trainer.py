# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskgraphNestedTrainer — TT-in-DG variant of NestedTrainer.

Dispatched by ``AutoTrainer._train`` when
``JobConfig.diskgraph_task_dataset_bundle_dir`` is set. Bypasses the legacy
data plane (no feature_store / graph_sampler / fkey.pt); the wheel iterator
over a task-dataset bundle is the entire data plane.
"""
from __future__ import annotations

import json
import logging
import os.path as osp
from collections.abc import Iterator
from typing import Any, ContextManager, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumo_core import runtime_progress
from kumo_core.diskgraph.lightning_data import (
    DiskgraphLightningDataModule,
    suppress_batch_indices_warning,
    suppress_iterable_len_warning,
)
from kumo_core.diskgraph.metadata import normalize_diskgraph_edge_type
from kumo_core.diskgraph.stores import iter_primary_key_batches
from kumo_core.io.csv import save_csv
from kumo_core.io.filesystem import fsspec_mangle
from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.train.artifact_stats import PredictionArtifactStatsCollector
from kumo_core.train.mapping import (
    NODE_INDEX_COLUMN,
    TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
)
from kumo_core.train.nested_trainer import NestedTrainer
from kumo_core.train.output import ArrowPredOutput
from kumo_core.train.trainer import (
    _MAX_EVAL_TOP_K,
    create_multiclass_prediction_output,
    create_multilabel_classification_prediction_output,
    create_multilabel_ranking_prediction_output,
    generate_majority_sampling_weight,
)
from kumo_core.training.config import (
    DiskgraphNestedTrainerConfig,
    FitMode,
    LinkPredModuleType,
    WeightMode,
    _normalize_edge_type_key,
)
from kumo_core.training.options import allow_array_targets
from kumo_core.training.runtime import DiskgraphTrainingRuntime
from kumo_core.types import TaskType
from kumo_core.utils.names import get_table_name
from kumo_core.utils.registry import register

logger = logging.getLogger(__name__)


def _prediction_progress_callbacks(
    datamodule: DiskgraphLightningDataModule,
    log_dir: str | None,
    log_id: int,
) -> list[Any]:
    """Collect prediction progress only when a consumer needs it."""
    if log_dir is None and not runtime_progress.has_observer():
        return []
    from kumo_core.callbacks import PredictionProgressLogger

    loader = datamodule.predict_dataloader()
    try:
        total_iterations = len(loader)
    except TypeError:
        # Streaming loaders have no known total; completed batches still help.
        total_iterations = 0
    return [PredictionProgressLogger(log_dir, log_id, total_iterations)]


def _defer_predict_worker_shutdown(
        datamodule: DiskgraphLightningDataModule) -> ContextManager[None]:
    """Defer strict cleanup until prediction output has been committed."""
    return datamodule.defer_dataloader_worker_shutdown('predict')


def _shutdown_predict_workers(datamodule: DiskgraphLightningDataModule) -> None:
    """Run the strict prediction barrier when the DataModule owns workers."""
    datamodule.shutdown_dataloader_workers('predict')


def _abort_prediction_writer(writer: Any) -> None:
    """Abort output without replacing the exception that caused the abort."""
    try:
        writer.abort()
    except BaseException:
        logger.exception('Failed to abort prediction output')


def _forecasting_out_timestamps(pl_model: Any, time: Any) -> Any:
    # ``time`` is already canonical ns (see _seed_time_to_datetime64_ns) and
    # ``get_future_unix`` operates in ns -- pass through. A unit rescale here
    # would re-interpret ns as the engine unit and overflow int64.
    return pl_model.get_future_unix(time).flatten()


def _seed_time_to_datetime64_ns(time: Any) -> "np.ndarray":
    """Read the prediction/eval-output seed time as ``datetime64[ns]``.

    The runtime adapter already normalizes the engine seed time (engine unit,
    default seconds) to canonical ns at the batch boundary (see
    ``diskgraph_runtime_adapter.engine_int64_to_ns`` / ``#29499``), so ``time``
    is ALREADY int64 ns -- read it directly. Re-applying a seconds
    interpretation (the prior ``datetime64[s]`` via ``index_np_time_unit``)
    overflowed the ~1.4e18 ns value and wrapped the output to year ~1730.
    """
    # Seed time may be a device tensor that still tracks grad; detach and move
    # it to host before numpy() (which rejects grad-tracking or non-CPU
    # tensors).
    if hasattr(time, 'detach'):
        time = time.detach()
    if hasattr(time, 'cpu'):
        time = time.cpu()
    arr = time.numpy() if hasattr(time, 'numpy') else np.asarray(time)
    return arr.astype('int64').astype('datetime64[ns]')


def _arrow_timestamp_us(values: Any) -> pa.Array:
    return pa.array(values).cast(pa.timestamp('us'), safe=False)


def _arrow_string_array(values: Any) -> pa.Array:
    """Build the explicitly string-typed embedding ID representation."""
    array = np.asarray(values)
    if array.dtype.kind != 'O':
        return pa.array(array).cast(pa.string())

    inferred_dtype = pd.api.types.infer_dtype(array, skipna=True)
    if inferred_dtype.startswith('mixed'):
        # Arrow cannot infer a common type for genuinely mixed objects.
        return pa.array([
            None if value is None or pd.isna(value) else str(value)
            for value in array
        ], type=pa.string())

    # Homogeneous strings/nulls and other homogeneous scalar IDs stay inside
    # Arrow. Casting also fixes all-null batches to a stable string schema.
    return pa.array(array, from_pandas=True).cast(pa.string())


def _build_ttng_lp_chunk_output(
    prediction: dict[str, Any],
    *,
    rhs_node_ids: np.ndarray,
    rhs_pk_arr: np.ndarray,
    rhs_pk_type: pa.DataType,
    lhs_fkey_name: str,
    time_col: str,
    ignore_time_col: bool,
    job_start_time: Any,
    emit_predictions: bool,
    emit_embeddings: bool,
    lhs_table: str,
    rhs_table: str,
    embedding_extra_fields: set[str],
    embedding_ids_as_string: bool,
    class_metadata: dict[bytes, bytes] | None = None,
    rhs_pk_mixed: bool = False,
) -> ArrowPredOutput:
    """Build TT-NG LP output directly as Arrow without pandas frames."""
    import torch

    def _np(value: Any) -> Any:
        if value is None:
            return None
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) \
            else np.asarray(value)

    lhs_pkey = _np(prediction.get('pkey'))
    time = _np(prediction.get('time'))
    if lhs_pkey is None:
        raise RuntimeError('[TT-NG LP] predict outputs missing pkey')
    n_rows = lhs_pkey.shape[0]

    if emit_predictions:
        target = _np(prediction.get('target'))
        score = _np(prediction.get('score'))
        if target is None:
            raise RuntimeError('[TT-NG LP] predict outputs missing target')
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        target_rows, k = target.shape
        if target_rows != n_rows:
            raise RuntimeError(
                f'[TT-NG LP] prediction rows ({target_rows}) do not match '
                f'anchors ({n_rows})')

        flat = target.reshape(-1)
        missing = flat < 0
        positions = np.searchsorted(rhs_node_ids, flat)
        valid = positions < rhs_node_ids.shape[0]
        if rhs_node_ids.shape[0]:
            valid &= rhs_node_ids[np.minimum(positions,
                                             rhs_node_ids.shape[0] - 1)] == flat
        if not bool((valid | missing).all()):
            raise RuntimeError(
                '[TT-NG LP] prediction contains a non-canonical RHS node ID')
        if missing.all():
            class_values = pa.nulls(flat.size, type=rhs_pk_type)
        else:
            values = rhs_pk_arr[np.where(missing, 0, positions)]
            if rhs_pk_mixed:
                values = [
                    None if value is None or pd.isna(value) else str(value)
                    for value in values
                ]
            class_values = pa.array(values, type=rhs_pk_type, mask=missing,
                                    from_pandas=True)
        arrays: list[pa.Array] = [
            pa.array(np.repeat(lhs_pkey, k)),
            class_values,
        ]
        names = [lhs_fkey_name, 'CLASS']
        if score is not None:
            arrays.append(pa.array(score.reshape(-1)))
            names.append('SCORE')
        if (time is not None and time.shape[0] == n_rows
                and not ignore_time_col):
            arrays.append(
                _arrow_timestamp_us(
                    np.repeat(_seed_time_to_datetime64_ns(time), k)))
            names.append(time_col)
        if job_start_time is not None:
            job_time = pd.Timestamp(job_start_time).tz_localize(
                None).to_datetime64()
            arrays.append(_arrow_timestamp_us(np.full(flat.shape[0], job_time)))
            names.append('JOB_TIMESTAMP')
        table = pa.Table.from_arrays(arrays, names=names,
                                     metadata=class_metadata)
    else:
        # Keep the unrequested prediction stream empty while this batch emits
        # LHS embeddings.
        table = pa.table({})

    embedding: pa.Table | None = None
    if emit_embeddings:
        lhs_emb = prediction.get('lhs_embedding')
        if lhs_emb is None:
            raise RuntimeError(
                '[TT-NG LP] predict output missing requested LHS embeddings')
        matrix = torch.as_tensor(lhs_emb).detach().cpu().numpy()
        if matrix.shape[0] != lhs_pkey.shape[0]:
            raise RuntimeError(
                f'[TT-NG LP] LHS embedding rows ({matrix.shape[0]}) '
                f'do not match anchors ({lhs_pkey.shape[0]})')
        string_ids = (embedding_ids_as_string
                      or lhs_pkey.dtype.kind in {'O', 'S', 'U'})
        embedding_arrays: list[pa.Array] = [
            (_arrow_string_array(lhs_pkey)
             if string_ids else pa.array(lhs_pkey)),
            pa.array(np.full(lhs_pkey.shape[0], lhs_table)),
        ]
        embedding_names = ['ID', 'TYPE']
        if lhs_table == rhs_table:
            embedding_arrays.append(
                pa.array(np.full(lhs_pkey.shape[0], lhs_fkey_name)))
            embedding_names.append('NAME')
        flat_embedding = pa.array(matrix.reshape(-1))
        offsets = pa.array(
            np.arange(matrix.shape[0] + 1, dtype=np.int32) * matrix.shape[1])
        embedding_arrays.append(
            pa.ListArray.from_arrays(offsets, flat_embedding))
        embedding_names.append('EMBEDDING')
        if ('ANCHOR_TIMESTAMP' in embedding_extra_fields and time is not None
                and time.shape[0] == n_rows):
            embedding_arrays.append(pa.array(_seed_time_to_datetime64_ns(time)))
            embedding_names.append('ANCHOR_TIMESTAMP')
        embedding = pa.Table.from_arrays(embedding_arrays,
                                         names=embedding_names)
    return ArrowPredOutput(table, embedding)


def _build_mapper_bank_from_encoders(
    encoders: Any,
    feature_schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the mapper bank for ``DiskgraphLightningDataModule`` from
    fitted encoders. Keyed by ``mapper_key`` (the bare table name)."""
    from kumo_core.encoders.tensor_mapper import TensorMapper

    bank: dict[str, dict[str, TensorMapper]] = {}
    nodes_meta = feature_schema.get("node_types") or {}
    for meta in nodes_meta.values():
        mapper_key = meta.get("mapper_key")
        if mapper_key is None or mapper_key in bank:
            continue
        encoder_dict = encoders[mapper_key] if mapper_key in encoders else {}
        bank[mapper_key] = {
            col: TensorMapper.from_encoder(encoder)
            for col, encoder in encoder_dict.items()
            if col in meta.get("feature_columns", encoder_dict)
        }
    return bank


def _time_col_dict_from_schema(
    feature_schema: dict[str, Any], ) -> dict[str, str]:
    nodes_meta = feature_schema.get("node_types") or {}
    out: dict[str, str] = {}
    for ns, meta in nodes_meta.items():
        time_col = meta.get("time_column")
        if time_col:
            out[ns] = time_col
    return out


def _train_stages_for_fit_mode(fit_mode: FitMode) -> tuple[str, ...]:
    if fit_mode == FitMode.refit_trainval:
        return ("train", "val")
    if fit_mode == FitMode.refit_full:
        return ("train", "val", "test")
    return ("train", )


def _runtime_options_from_sampler(sampler_kwargs: Any) -> dict[str, bool]:
    return {
        "replace": bool(getattr(sampler_kwargs, "replace", False)),
        "disjoint": bool(getattr(sampler_kwargs, "disjoint", True)),
        "adaptive": bool(getattr(sampler_kwargs, "adaptive_sampling", False)),
    }


def _ttng_estimated_edges(
    sampling_plan: Mapping[str, Any],
    batch_size: int,
) -> int:
    """Estimate TT-NG per-batch edge volume for DiskGraph auto-config.

    This mirrors the legacy sampler's conservative max-fanout-per-hop
    estimate, but reads the *compiled TT-NG plan* and actual training batch
    size rather than legacy ``NumNeighbors`` and its 1024-item default.
    """
    per_seed_fanout = 1
    for layer in sampling_plan.get("layers") or ():
        candidates = []
        for config in (layer.get("fanouts") or {}).values():
            if config is not None:
                candidates.append(int(config.get("fanout", 0)))
        default = layer.get("default_fanout")
        if default is not None:
            candidates.append(int(default.get("fanout", 0)))
        per_seed_fanout *= max(candidates, default=1)
    return int(batch_size * per_seed_fanout * 0.25)


def _ttng_diskgraph_runtime_config(
    runtime: DiskgraphTrainingRuntime | None,
    graph_runtime_metadata: Mapping[str, Any] | None,
    sampling_plan: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any] | None:
    """Return the existing DiskGraph policy for the TT-NG wheel path.

    TT-NG bypasses the legacy neighbor sampler, so it must explicitly apply
    the same policy before constructing its independent wheel iterators.
    Missing index metadata remains valid for small unit-only manifests; a real
    staged index must provide its finalized manifest or configuration fails
    rather than silently falling back to unbounded defaults.
    """
    index_path = (graph_runtime_metadata or {}).get("graph_index_path")
    if not index_path:
        return None
    if runtime is None:
        raise RuntimeError(
            "diskgraph_runtime is required for DiskGraph training")

    estimated_edges = _ttng_estimated_edges(sampling_plan, batch_size)
    config = runtime.auto_config(
        index_path=str(index_path),
        estimated_edges=estimated_edges,
    )
    logger.info(
        "[TT-NG] DiskGraph auto_config index=%s estimated_edges=%d; "
        "result: mode=%s "
        "num_workers=%d rayon_threads=%d",
        index_path,
        estimated_edges,
        config["mode"],
        config["num_workers"],
        config["rayon_threads"],
    )
    return config


def _bool_key(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return None


def _pkey_frame(pkey: Any, fkey_names: list[str]) -> pd.DataFrame:
    pkey_arr = pkey if isinstance(pkey, np.ndarray) else np.asarray(pkey)
    pkey_arr = (pkey_arr.reshape(1, -1)
                if pkey_arr.ndim == 1 else pkey_arr.transpose())
    return pd.DataFrame({
        fkey_name: pkey_arr[i]
        for i, fkey_name in enumerate(fkey_names)
    })


def _binary_distribution_from_stats(
    stats: dict[str, Any] | Any, ) -> dict[bool, float] | None:
    target_distribution = (stats or {}).get("target_distribution") or {}
    raw = target_distribution.get("percentage") or target_distribution.get(
        "count")
    if not isinstance(raw, dict):
        return None
    parsed: dict[bool, float] = {}
    for key, value in raw.items():
        bool_key = _bool_key(key)
        if bool_key is not None:
            parsed[bool_key] = float(value)
    if set(parsed) != {False, True}:
        return None
    total = sum(parsed.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in parsed.items()}


def _binary_distribution_from_counts(
    false_count: Any,
    true_count: Any,
) -> dict[bool, float] | None:
    try:
        counts = {False: int(false_count), True: int(true_count)}
    except (TypeError, ValueError):
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in counts.items()}


def _binary_distribution_from_stage_label_index(
    runtime_manifest_path: str,
    stage: str,
) -> dict[bool, float] | None:
    with open(runtime_manifest_path, encoding='utf-8') as f:
        runtime = json.load(f)
    stage_ref = (runtime.get('stages') or {}).get(stage)
    if not isinstance(stage_ref, Mapping):
        return None
    index = stage_ref.get('binary_label_index')
    if not isinstance(index, Mapping):
        return None
    return _binary_distribution_from_counts(
        index.get('false_count'),
        index.get('true_count'),
    )


def _has_train_table_weight_column(tm: "DiskgraphTaskManifest") -> bool:
    weight_column = (tm.train_table_columns or {}).get('weight_column')
    return bool(weight_column)


def _uses_rhs_feature_store(module_kwargs: Any) -> bool:
    rhs_emb_mode = getattr(module_kwargs, "rhs_emb_mode", None)
    if getattr(rhs_emb_mode, "use_rhs_feature", False):
        return True
    if hasattr(rhs_emb_mode, "value"):
        rhs_emb_mode = rhs_emb_mode.value
    elif hasattr(rhs_emb_mode, "name"):
        rhs_emb_mode = rhs_emb_mode.name
    return str(rhs_emb_mode).lower() in {"feature", "shallow_feature", "fusion"}


def _uses_rhs_lookup(module_kwargs: Any) -> bool:
    module_type = getattr(module_kwargs, "module", None)
    if module_type not in {
            LinkPredModuleType.SHALLOW_RHS, LinkPredModuleType.HYBRID
    }:
        return False
    rhs_emb_mode = getattr(module_kwargs, "rhs_emb_mode", None)
    if getattr(rhs_emb_mode, "use_rhs_lookup", False):
        return True
    if hasattr(rhs_emb_mode, "value"):
        rhs_emb_mode = rhs_emb_mode.value
    elif hasattr(rhs_emb_mode, "name"):
        rhs_emb_mode = rhs_emb_mode.name
    return str(rhs_emb_mode).lower() in {"lookup", "fusion"}


def _rhs_primary_key_index_map(
    primary_keys: np.ndarray,
    node_ids: np.ndarray,
    expected_count: int,
) -> pd.DataFrame:
    """Build the checkpoint raw-PK to canonical graph-node-ID mapping."""
    if len(primary_keys) != len(node_ids):
        raise RuntimeError(
            '[TT-NG LP] RHS primary-key and node-ID catalogs differ in size')
    _validate_rhs_node_ids(node_ids, expected_count)
    index = pd.Index(primary_keys, name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME)
    if index.hasnans:
        raise RuntimeError('[TT-NG LP] RHS primary-key catalog contains nulls')
    if index.has_duplicates:
        raise RuntimeError(
            '[TT-NG LP] RHS primary-key catalog contains duplicates')
    return pd.DataFrame(
        {NODE_INDEX_COLUMN: node_ids.astype(np.int64, copy=False)},
        index=index,
    )


def _validate_rhs_node_ids(node_ids: np.ndarray, expected_count: int) -> None:
    if node_ids.ndim != 1 or node_ids.dtype.kind not in {'i', 'u'}:
        raise RuntimeError('[TT-NG LP] RHS node-ID catalog is not integral')
    if len(node_ids) and (int(node_ids.min()) < 0
                          or int(node_ids.max()) >= expected_count):
        raise RuntimeError('[TT-NG LP] RHS node-ID catalog is out of range')
    if len(node_ids) > 1 and bool(np.any(node_ids[1:] == node_ids[:-1])):
        raise RuntimeError('[TT-NG LP] RHS node-ID catalog contains duplicates')
    if len(node_ids) > 1 and not bool(np.all(node_ids[1:] > node_ids[:-1])):
        raise RuntimeError('[TT-NG LP] RHS node-ID catalog is not ordered')


def _rhs_canonical_node_ids(
    index_path: str,
    rhs_node_type: str,
    expected_count: int,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Materialize canonical physical IDs without retaining raw PK values."""
    node_ids = np.empty(expected_count, dtype=np.int64)
    offset = 0
    for raw_node_ids, raw_batch in iter_primary_key_batches(
            index_path, rhs_node_type, stop=expected_count,
            batch_size=batch_size):
        batch_node_ids = np.asarray(raw_node_ids)
        batch = np.asarray(raw_batch)
        if (batch_node_ids.ndim != 1
                or batch_node_ids.dtype.kind not in {'i', 'u'}):
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned invalid IDs')
        if batch.ndim != 1 or len(batch_node_ids) != len(batch):
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned misaligned IDs')
        if offset + len(batch_node_ids) > expected_count:
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned more canonical '
                f'rows than the DiskGraph node count ({expected_count})')
        node_ids[offset:offset + len(batch_node_ids)] = batch_node_ids
        offset += len(batch_node_ids)

    node_ids = node_ids[:offset]
    _validate_rhs_node_ids(node_ids, expected_count)
    return node_ids


def _rhs_primary_key_catalog(
    index_path: str,
    rhs_node_type: str,
    expected_count: int,
    batch_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one ordered catalog through bounded PK-only rowpack reads."""
    primary_keys: np.ndarray | None = None
    node_ids = np.empty(expected_count, dtype=np.int64)
    offset = 0
    for raw_node_ids, raw_batch in iter_primary_key_batches(
            index_path, rhs_node_type, stop=expected_count,
            batch_size=batch_size):
        batch_node_ids = np.asarray(raw_node_ids)
        batch = np.asarray(raw_batch)
        if batch.ndim != 1:
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned a non-vector batch')
        if batch.dtype.kind not in {'i', 'u', 'O', 'S', 'U'}:
            raise RuntimeError('[TT-NG LP] unsupported RHS primary-key dtype: '
                               f'{batch.dtype}')
        if batch.dtype.kind == 'O':
            inferred = pd.api.types.infer_dtype(batch, skipna=True)
            if inferred not in {
                    'integer', 'string', 'bytes', 'decimal', 'empty'
            }:
                raise RuntimeError(
                    '[TT-NG LP] unsupported RHS primary-key object dtype: '
                    f'{inferred}')
        if bool(pd.isna(batch).any()):
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key catalog contains nulls')
        if batch_node_ids.ndim != 1 or len(batch_node_ids) != len(batch):
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned misaligned IDs')
        if offset + len(batch) > expected_count:
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader returned more canonical '
                f'rows than the DiskGraph node count ({expected_count})')
        if primary_keys is None:
            primary_keys = np.empty(expected_count, dtype=batch.dtype)
        elif batch.dtype != primary_keys.dtype:
            raise RuntimeError(
                '[TT-NG LP] RHS primary-key reader changed dtype between '
                f'batches: {primary_keys.dtype} then {batch.dtype}')
        node_ids[offset:offset + len(batch)] = batch_node_ids
        primary_keys[offset:offset + len(batch)] = batch
        offset += len(batch)

    if primary_keys is None:
        primary_keys = np.empty(0, dtype=np.int64)
    return node_ids[:offset], primary_keys[:offset]


def _build_runtime_feature_store(
    *,
    runtime: DiskgraphTrainingRuntime,
    tm: DiskgraphTaskManifest,
    encoders: Any,
    entity_table_names: list[str],
) -> tuple[Any, Any]:
    """Build the lightweight DiskGraph FS/GS used for RHS feature reads."""
    index_path = (tm.graph_runtime_metadata or {}).get("graph_index_path")
    if not index_path:
        raise RuntimeError(
            "[TT-NG] RHS feature mode requires graph_runtime_metadata."
            "graph_index_path; got None")

    return runtime.create_feature_and_graph_stores(
        index_path=index_path,
        encoders=encoders,
        entity_table_names=entity_table_names,
        time_col_dict=_time_col_dict_from_schema(tm.feature_schema),
    )


def _load_index_metadata(
    graph_runtime_metadata: dict[str, Any] | None, ) -> dict[str, Any]:
    """Helper to parse ``index_metadata.json`` from the index path if it exists."""
    graph_index_path = (graph_runtime_metadata or {}).get('graph_index_path')
    if not graph_index_path:
        return {}
    idx_meta_path = osp.join(graph_index_path, 'index_metadata.json')
    if not osp.exists(idx_meta_path):
        return {}
    import json
    try:
        with open(idx_meta_path) as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning("[TT-NG] Failed to load index_metadata.json: %s", e)
        return {}


def _has_train_table_time_col(train_table_columns: Mapping[str, Any]) -> bool:
    return bool(train_table_columns.get("time_column"))


def _get_num_nodes_from_index_metadata(
    graph_runtime_metadata: dict[str, Any] | None,
    node_type: str,
) -> int:
    """Load node capacity (max_id + 1) for a specific node type from ``index_metadata.json``."""
    idx_meta = _load_index_metadata(graph_runtime_metadata)
    node_types = idx_meta.get('node_types') or []
    for nt in node_types:
        if nt.get('name') == node_type:
            max_id = nt.get('max_id', -1)
            return max_id + 1 if max_id >= 0 else 0
    return 0


def _edge_types_from_index_metadata(
    graph_runtime_metadata: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    """Load Kumo-canonical edge types from ``index_metadata.json``.

    DiskGraph stores verbose FK relation names in ``index_metadata.json`` while
    runtime batches are adapted to Kumo's short edge names. Normalize here too
    so HeteroConv metadata and runtime ``HeteroData`` use identical keys. Falls
    back to manifest edge types if index metadata is missing.
    """
    short_fallback: list[tuple[str, str, str]] = [
        (entry['src'], entry['rel'], entry['dst'])
        for entry in ((graph_runtime_metadata or {}).get('edge_types') or [])
    ]
    idx_meta = _load_index_metadata(graph_runtime_metadata)
    relations = idx_meta.get('edge_relations') or []
    if not relations:
        return short_fallback
    edge_types: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in relations:
        name = rel.get('name')
        src_type = rel.get('src_type')
        dst_type = rel.get('dst_type')
        if not name or not src_type or not dst_type:
            continue
        edge_type = normalize_diskgraph_edge_type(src_type, name, dst_type)
        if edge_type not in seen:
            edge_types.append(edge_type)
            seen.add(edge_type)
    return edge_types


# Used for test only.
_DEFAULT_TTNG_SAMPLING_PLAN: dict[str, Any] = {
    'layers': [
        {
            'fanouts': {},
            'default_fanout': {
                'fanout': 10,
                'mode': 'temporal_most_recent',
                'absent_key_policy': 'uniform_fallback',
            },
        },
        {
            'fanouts': {},
            'default_fanout': {
                'fanout': 10,
                'mode': 'temporal_most_recent',
                'absent_key_policy': 'uniform_fallback',
            },
        },
    ],
}


# @register required: ConfigMixin.config() needs data_cls when AutoTrainer
# serializes at a trial boundary, else the first checkpoint raises ValueError.
@register(data_cls=DiskgraphNestedTrainerConfig)
class DiskgraphNestedTrainer(NestedTrainer):
    """TT-NG NestedTrainer subclass driven by a task-dataset bundle.

    Adds a keyword-only ``diskgraph_task_dataset_bundle_dir`` (consumed here,
    not forwarded to the parent).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bundle_dir = kwargs.pop('diskgraph_task_dataset_bundle_dir', None)
        if bundle_dir is None:
            raise ValueError('DiskgraphNestedTrainer requires '
                             'diskgraph_task_dataset_bundle_dir; got None')
        super().__init__(*args, **kwargs)
        self.diskgraph_task_dataset_bundle_dir: str = bundle_dir
        self._runtime_num_workers = 1
        self._runtime_worker_id = 0

    def set_runtime_worker_shard(
        self,
        *,
        runtime_num_workers: int,
        runtime_worker_id: int,
    ) -> None:
        runtime_num_workers = max(1, int(runtime_num_workers))
        runtime_worker_id = int(runtime_worker_id)
        if not 0 <= runtime_worker_id < runtime_num_workers:
            raise ValueError(
                'runtime_worker_id must be in [0, runtime_num_workers)')
        self._runtime_num_workers = runtime_num_workers
        self._runtime_worker_id = runtime_worker_id
        datamodule = getattr(self, 'datamodule', None)
        if datamodule is not None:
            datamodule.runtime_num_workers = runtime_num_workers
            datamodule.runtime_worker_id = runtime_worker_id

    def _build_ttng_sampling_plan(self,
                                  tm: DiskgraphTaskManifest) -> dict[str, Any]:
        """Compile ``num_neighbors`` into a DiskGraph sampling plan."""
        raw_values = self.sampler_kwargs.num_neighbors.get_values()
        temporal_strategy = getattr(self.sampler_kwargs, "temporal_strategy",
                                    "last")
        mode = "temporal_most_recent" if temporal_strategy == "last" else "uniform"

        idx_meta = _load_index_metadata(tm.graph_runtime_metadata)
        relations = idx_meta.get("edge_relations") or []

        edge_type_to_rel_name = {}
        for rel in relations:
            name = rel.get("name")
            src_type = rel.get("src_type")
            dst_type = rel.get("dst_type")
            if name and src_type and dst_type:
                edge_type_to_rel_name[(src_type, name, dst_type)] = name
                edge_type_to_rel_name[normalize_diskgraph_edge_type(
                    src_type, name, dst_type)] = name

        rel_name_to_csc_counterpart = {}
        all_rel_names = {
            rel.get("name")
            for rel in relations if rel.get("name")
        }
        for name in all_rel_names:
            if name.endswith("_reverse"):
                counterpart = name[:-len("_reverse")]
            else:
                counterpart = name + "_reverse"
            if counterpart in all_rel_names:
                rel_name_to_csc_counterpart[name] = counterpart

        def _hop(fanout: int) -> dict[str, Any]:
            return {
                # -1 means all neighbors.
                "fanout": 1_000_000_000 if fanout < 0 else fanout,
                "mode": mode,
                "absent_key_policy": "uniform_fallback",
            }

        # Case A: Simple list-based num_neighbors (e.g. [10, 5])
        if isinstance(raw_values, list):
            return {
                "layers": [{
                    "fanouts": {},
                    "default_fanout": _hop(fanout),
                } for fanout in raw_values]
            }

        # Case B: Dictionary-based per-relation num_neighbors
        per_type_values = raw_values or {}
        default_values = self.sampler_kwargs.num_neighbors.default

        num_hops = max((len(v) for v in per_type_values.values()), default=0)
        if default_values:
            num_hops = max(num_hops, len(default_values))

        layers = []
        unresolved = set()
        resolved_positive_fanout = False
        for hop in range(num_hops):
            default_fanout_val = (default_values[hop] if default_values
                                  and hop < len(default_values) else 0)
            fanouts = {}
            for edge_type, hops_list in per_type_values.items():
                raw_fanout = hops_list[hop] if hop < len(hops_list) else 0
                # A 0 is emitted only to override a default.
                if raw_fanout == 0 and default_fanout_val == 0:
                    continue
                edge_type_key = _normalize_edge_type_key(edge_type)
                rel_name = edge_type_to_rel_name.get(edge_type_key)
                if rel_name is None:
                    unresolved.add(str(edge_type))
                    continue
                csc_name = rel_name_to_csc_counterpart.get(rel_name, rel_name)
                fanouts[csc_name] = _hop(raw_fanout)
                resolved_positive_fanout |= raw_fanout != 0

            default_fanout = (_hop(default_fanout_val)
                              if default_fanout_val != 0 else None)
            layers.append({
                "fanouts": fanouts,
                "default_fanout": default_fanout,
            })

        if unresolved:
            if not resolved_positive_fanout and not any(
                    layer["default_fanout"] is not None for layer in layers):
                raise RuntimeError(
                    "[TT-NG] Per-relation num_neighbors did not resolve any "
                    "DiskGraph edge relations; refusing to train with an "
                    f"empty sampling plan. Unresolved: {sorted(unresolved)}")
            logger.warning(
                "[TT-NG] num_neighbors entries do not match any DiskGraph "
                "edge relation and are not sampled: %s", sorted(unresolved))

        return {"layers": layers}

    def init(
        self,
        *,
        task_manifest: DiskgraphTaskManifest | None = None,
    ) -> "DiskgraphNestedTrainer":
        """Build the datamodule over the bundle, then the PL trainer + model.

        Idempotent: AutoTrainer appends a callback onto ``pl_trainer.callbacks``
        after the first call, so re-entry must not rebuild ``pl_trainer`` (it
        would drop the callback). ``task_manifest`` is this bundle's manifest
        when the caller already loaded it.
        """
        if self._fully_initialized and not self.fit_mode.is_refit:
            return self

        # Mirror Trainer.init normalization: MulticlassClassification with
        # out_channels=1 is treated as binary.
        if (self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION
                and self.head is not None and self.head.out_channels == 1):
            logger.warning(
                '[TT-NG] Setting the task type to binary classification '
                'since there only exists two classes')
            self.task.task_type = TaskType.BINARY_CLASSIFICATION

        manifest_path = osp.join(self.diskgraph_task_dataset_bundle_dir,
                                 'manifest.json')
        tm = task_manifest
        if tm is None:
            logger.info('[TT-NG] Loading task-dataset bundle from %s',
                        manifest_path)
            tm = DiskgraphTaskManifest.load(manifest_path)

        self._configure_model_triple_from_manifest(
            tm,
            time_col_dict_fn=_time_col_dict_from_schema,
            has_train_table_time_col=_has_train_table_time_col(
                tm.train_table_columns),
        )

        mapper_bank = _build_mapper_bank_from_encoders(self.encoders,
                                                       tm.feature_schema)

        train_batch_size = int(self.loader_kwargs.batch_size)
        eval_batch_size = int((self.eval_loader_kwargs
                               or {}).get('batch_size', train_batch_size))

        sampling_plan = self._build_ttng_sampling_plan(tm)
        logger.info('[TT-NG] Generated dynamic, model-driven sampling plan: %s',
                    sampling_plan)
        runtime_options = _runtime_options_from_sampler(self.sampler_kwargs)
        auto_config = _ttng_diskgraph_runtime_config(
            getattr(self, "diskgraph_runtime", None),
            tm.graph_runtime_metadata,
            sampling_plan,
            train_batch_size,
        )
        if auto_config is not None:
            self.loader_kwargs.num_workers = auto_config["num_workers"]
            runtime_options.update(
                num_threads=auto_config["rayon_threads"],
                fused=(auto_config["mode"] == "fused"),
            )

        # Two-tower LP engine negatives (M per anchor); gated to LP + TWO_TOWER
        # (others use implicit negatives), applied to the train loader only.
        is_lp = self.task.task_type.is_link_pred
        module_type = (getattr(self.module_kwargs, "module", None) if getattr(
            self, "module_kwargs", None) is not None else None)
        bidirectional_output = bool(
            is_lp and module_type is not None
            and getattr(module_type, "use_bidirectional_sampling", False))
        request_negs = (is_lp and module_type == LinkPredModuleType.TWO_TOWER
                        and bool(tm.target_node_type))
        neg_rhs_node_type = tm.target_node_type if request_negs else None
        neg_amount_m = ((getattr(self.module_kwargs, "rhs_sample_size", None)
                         or 1) if request_negs else None)

        # Temporally-valid RHS set for ShallowRHS / Hybrid temporal LP; the
        # datamodule builds the provider lazily. None for non-LP.
        rhs_node_type = (tm.target_node_type if is_lp else None)
        unix_start_offset = getattr(self.task, 'unix_start_offset', None)
        unix_end_offset = getattr(self.task, 'unix_end_offset', None)

        # Seed off entity_node_id for every task type; LP also reads positive
        # labels off target_node_ids.
        resolve_entity_node_type = tm.seed_node_type
        resolve_target_node_type = (tm.target_node_type if is_lp else None)
        target_encoder = (None if is_lp or getattr(
            self.encoders, 'target', None) is None else self.target_encoder)

        weighted_seed_sampling: dict[str, float | bool] | None = None
        self.majority_sampling_weights = None
        has_weight_column = _has_train_table_weight_column(tm)
        if not getattr(self, '_prediction_mode', False):
            majority_sampling_weights = None
            if self.majority_sampling_ratio is not None:
                if self.weight_mode not in {
                        WeightMode.SAMPLE,
                        WeightMode.WEIGHTED_LOSS,
                }:
                    raise NotImplementedError(
                        '[TT-NG] majority_sampling_ratio requires '
                        'WeightMode.SAMPLE or WeightMode.WEIGHTED_LOSS on the '
                        'TT-NG path.')
                sample_mode = self.weight_mode == WeightMode.SAMPLE
                if sample_mode:
                    target_distribution = (
                        _binary_distribution_from_stage_label_index(
                            tm.runtime_manifest_path, 'train'))
                else:
                    target_distribution = _binary_distribution_from_stats(
                        tm.stats)
                if target_distribution is None:
                    source = ('train-stage binary_label_index counts'
                              if sample_mode else
                              'binary target_distribution in the task-dataset '
                              'stats sidecar')
                    raise RuntimeError(
                        f'[TT-NG] majority_sampling_ratio requires {source}.')
                majority_sampling_weights = generate_majority_sampling_weight(
                    self.majority_sampling_ratio,
                    target_distribution,
                )

            if self.weight_mode == WeightMode.SAMPLE and (
                    has_weight_column or majority_sampling_weights is not None):
                weighted_seed_sampling = {
                    'use_sample_weights': has_weight_column,
                }
                if majority_sampling_weights is not None:
                    weighted_seed_sampling.update({
                        'false_weight':
                        float(majority_sampling_weights.get(False, 1.0)),
                        'true_weight':
                        float(majority_sampling_weights.get(True, 1.0)),
                    })
            elif self.weight_mode == WeightMode.WEIGHTED_LOSS:
                self.majority_sampling_weights = majority_sampling_weights

        self.datamodule = DiskgraphLightningDataModule(
            manifest=tm,
            mapper_bank=mapper_bank,
            sampling_plan=sampling_plan,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            max_steps_per_epoch=self.max_steps_per_epoch,
            min_steps_per_epoch=self.min_steps_per_epoch,
            max_val_steps=self.max_val_steps,
            max_test_steps=self.max_test_steps,
            train_stages=_train_stages_for_fit_mode(self.fit_mode),
            runtime_options=runtime_options,
            runtime_num_workers=getattr(self, '_runtime_num_workers', 1),
            runtime_worker_id=getattr(self, '_runtime_worker_id', 0),
            num_workers=self.loader_kwargs.num_workers,
            pin_memory=self.loader_kwargs.pin_memory,
            persistent_workers=self.loader_kwargs.persistent_workers,
            prefetch_factor=self.loader_kwargs.prefetch_factor,
            task_type=self.task.task_type.value,
            neg_rhs_node_type=neg_rhs_node_type,
            neg_amount_m=neg_amount_m,
            rhs_node_type=rhs_node_type,
            unix_start_offset=unix_start_offset,
            unix_end_offset=unix_end_offset,
            resolve_entity_node_type=resolve_entity_node_type,
            resolve_target_node_type=resolve_target_node_type,
            target_encoder=target_encoder,
            bidirectional_output=bidirectional_output,
            base_seed=(0 if getattr(self, 'random_state_seed', None) is None
                       else self.random_state_seed),
            # Single-timestamp grouping mirrors legacy's TimestampSampler gate,
            # which requires the module to seed off LHS nodes (shallow_rhs / id /
            # hybrid; not two_tower). two_tower / non-LP -> use_lhs_node_loader
            # False -> never grouped.
            use_lhs_node_loader=bool(
                is_lp and module_type is not None
                and getattr(module_type, 'use_lhs_node_loader', False)),
            weighted_seed_sampling=weighted_seed_sampling,
        )

        self.create_pl_trainer()

        # Advance the DLDM shuffle stream per epoch (PL's set_epoch doesn't
        # reach an IterableDataset).
        self.pl_trainer.callbacks = self.pl_trainer.callbacks + [
            DiskgraphLightningDataModule.shuffle_stream_callback(),
            DiskgraphLightningDataModule.worker_pool_lifecycle_callback(),
        ]

        entity_table_name = tm.metadata['entity_table_name']
        entity_table_names = [entity_table_name]
        is_lp = self.task.task_type.is_link_pred
        # RHS node type comes off the manifest (TaskConfig has none).
        rhs_table_name = tm.target_node_type if is_lp else None
        if is_lp and rhs_table_name:
            entity_table_names.append(rhs_table_name)

        if is_lp and _uses_rhs_feature_store(
                getattr(self, "module_kwargs", None)):
            if self.diskgraph_runtime is None:
                raise RuntimeError(
                    "diskgraph_runtime is required for RHS feature mode")
            self.feature_store, self.graph_store = _build_runtime_feature_store(
                runtime=self.diskgraph_runtime,
                tm=tm,
                encoders=self.encoders,
                entity_table_names=entity_table_names,
            )
            self.datamodule._data = (self.feature_store, self.graph_store)

        # Lookup embeddings must be keyed by raw primary key in checkpoints.
        # DiskGraph node IDs can change between graph snapshots, so an identity
        # map would resize successfully during BP while silently assigning
        # learned rows to the wrong entities.
        rhs_index_map = pd.DataFrame()
        num_rhs_nodes = 0
        if is_lp and rhs_table_name:
            num_rhs_nodes = _get_num_nodes_from_index_metadata(
                tm.graph_runtime_metadata, rhs_table_name)
            if num_rhs_nodes > 0:
                index_path = (tm.graph_runtime_metadata
                              or {}).get('graph_index_path')
                if not index_path:
                    raise RuntimeError(
                        '[TT-NG LP] RHS catalog requires graph_index_path')
                if _uses_rhs_lookup(self.module_kwargs):
                    catalog_ids, catalog = _rhs_primary_key_catalog(
                        index_path, rhs_table_name, num_rhs_nodes)
                    rhs_index_map = _rhs_primary_key_index_map(
                        catalog, catalog_ids, num_rhs_nodes)
                    self._rhs_primary_key_catalog = catalog
                    self._rhs_primary_key_node_ids = catalog_ids
                    self._rhs_primary_key_index_map = rhs_index_map
                else:
                    # Non-lookup models need only canonical physical IDs until
                    # prediction output requires raw primary keys.
                    self._rhs_primary_key_node_ids = _rhs_canonical_node_ids(
                        index_path, rhs_table_name, num_rhs_nodes)
                    rhs_index_map = pd.DataFrame(index=range(num_rhs_nodes))
                    rhs_index_map[NODE_INDEX_COLUMN] = range(num_rhs_nodes)

        self.create_pl_model(
            entity_table_names=entity_table_names,
            train_table_name=tm.metadata['train_table_name'],
            rhs_index_map=rhs_index_map,
            num_rhs_nodes=num_rhs_nodes,
            ckpt_path=self.checkpoint_path,
        )
        if is_lp and hasattr(self, '_rhs_primary_key_node_ids'):
            canonical_node_ids = torch.from_numpy(
                self._rhs_primary_key_node_ids.astype(np.int64, copy=False))
            # Ordered, unique IDs in [0, num_rhs_nodes) are necessarily the
            # dense identity when their counts match, so no runtime filter is
            # needed in that common case.
            if len(canonical_node_ids) < num_rhs_nodes:
                self.pl_model.register_buffer(
                    'canonical_rhs_node_idx',
                    canonical_node_ids,
                    persistent=False,
                )
        if self.task.task_type == TaskType.FORECASTING:
            self.pl_model.use_precomputed_forecasting_sequences = True

        # Stamp the RHS catalog size so the MIPS-readiness check resolves without
        # datamodule.lp_train_table (which two-tower STATIC would hit and raise).
        if is_lp and rhs_table_name and num_rhs_nodes > 0:
            self.pl_model.num_rhs_nodes = num_rhs_nodes

        self._fully_initialized = True
        return self

    def explain(self, *args: Any, **kwargs: Any) -> None:
        """Lazily wire wheel-backed FS/GS into the legacy ``Trainer.explain``
        flow for the duration of the call (TT-NG wires none of them otherwise).
        """
        # Idempotent across repeat ``explain()`` calls on the same
        # trainer instance.
        if getattr(self, 'feature_store', None) is None:
            tm = DiskgraphTaskManifest.load(
                osp.join(self.diskgraph_task_dataset_bundle_dir,
                         'manifest.json'))
            index_path = (tm.graph_runtime_metadata
                          or {}).get('graph_index_path')
            if not index_path:
                raise RuntimeError(
                    '[TT-NG] explain() requires graph_runtime_metadata.'
                    'graph_index_path; got None')
            time_col_dict = dict(self.datamodule.time_col_dict)
            entity_table_name = (
                tm.metadata or {}).get('entity_table_name') or tm.seed_node_type
            if self.diskgraph_runtime is None:
                raise RuntimeError(
                    "diskgraph_runtime is required for DiskGraph explain")
            self.feature_store, self.graph_store = (
                self.diskgraph_runtime.create_feature_and_graph_stores(
                    index_path=index_path,
                    encoders=self.encoders,
                    entity_table_names=[entity_table_name],
                    time_col_dict=time_col_dict,
                ))

        # Expose (fs, gs) on the datamodule (SubgraphCollector reads .data).
        prev_data = self.datamodule._data
        self.datamodule._data = (self.feature_store, self.graph_store)
        try:
            return super().explain(*args, **kwargs)
        finally:
            self.datamodule._data = prev_data

    def _eval(
        self,
        stage: Any,
        return_embeddings: bool = False,
    ) -> Any:
        """Run PL validate/test, then build the formatted EvalOutput
        (``_build_eval_df`` reconstructs the per-row df from ``pl_model.outputs``).
        """
        from kumo_core.train.output import EvalOutput
        from kumoapi.train import TrainingStage
        assert stage in {
            TrainingStage.VAL, TrainingStage.TEST
        }, (f'[TT-NG] _eval received unexpected stage={stage!r}')

        if self._prediction_mode:
            raise RuntimeError(
                f'Calling `{stage.value}()` after `predict()` is not '
                f'allowed')

        if return_embeddings:
            raise NotImplementedError(
                '[TT-NG] return_embeddings is not yet wired on '
                'DiskgraphNestedTrainer.')

        self.pl_model.eval_top_k = None

        with suppress_iterable_len_warning():
            if stage == TrainingStage.VAL:
                metric_results = self.pl_trainer.validate(
                    model=self.pl_model,
                    datamodule=self.datamodule,
                    ckpt_path=fsspec_mangle(self.best_ckpt),
                )
            else:
                metric_results = self.pl_trainer.test(
                    model=self.pl_model,
                    datamodule=self.datamodule,
                    ckpt_path=fsspec_mangle(self.best_ckpt),
                )

        if metric_results:
            metrics = metric_results[0]
        else:
            is_empty_link_test = (
                stage == TrainingStage.TEST
                and self.task.task_type.is_link_pred
                and self.datamodule.task_manifest.stage_counts.get('test') == 0
            )
            if not is_empty_link_test:
                raise RuntimeError(
                    '[TT-NG] evaluation returned no metric result for '
                    f'stage={stage.value} task_type={self.task.task_type.value}'
                )
            metrics = {}

        if self.task.task_type.is_link_pred:
            # Link prediction has no row-wise evaluation frame. A zero-row
            # loader may also skip the hook that creates ``outputs``.
            outputs = None
            self.pl_model.outputs = None
        else:
            outputs, self.pl_model.outputs = self.pl_model.outputs, None

        path = osp.join(self.log_dir, f'{stage.value}_result.csv')
        save_csv(metrics, path, overwrite=True)

        if self.task.task_type.is_link_pred:
            return EvalOutput(metrics, None, None)

        df = self._build_eval_df(outputs)
        return EvalOutput(metrics, df, None)

    def _build_eval_df(self, outputs: Any) -> pd.DataFrame | None:
        from kumo_core.train.formatting import format_prediction
        from kumo_core.train.loss import DistributionLoss, MultiQuantileLoss

        # Link prediction has no scalar target encoder (encoders.target is None;
        # target_col=None by design), so there is no per-row scalar prediction
        # frame to build here — the LP eval metrics (recall@K / map@K) are
        # produced by the progress logger during validate and already written to
        # `{stage}_result.csv` by the caller. Return None (the eval-DF path is
        # only ported for node-pred). Same #29405 root cause as trainer.py's
        # `target_encoder` null-guard; this is the second unguarded site.
        if self.encoders.target is None or self.encoders.target.encoder is None:
            return None

        # Output column names come from train_table_columns, not index_metadata
        # (whose entity-table source names may lack a time col).
        ttc = self.datamodule.task_manifest.train_table_columns
        fkey_name = ttc['seed_id_column']
        time_col = ttc.get('time_column')
        target_col_name = self.encoders.target.col_name

        df = _pkey_frame(outputs['pkey'], [fkey_name])

        if outputs.get('time') is not None and time_col is not None:
            # ``outputs['time']`` is adapter-normalized canonical ns; read it
            # directly (see ``_seed_time_to_datetime64_ns``).
            df[time_col] = _seed_time_to_datetime64_ns(outputs['time'])

        if self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return create_multilabel_classification_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outputs['prob'],
                pred=outputs['pred'],
                target=outputs.get('target'),
                top_k=self._eval_top_k_from_outputs(),
            )

        if self.task.task_type == TaskType.MULTILABEL_RANKING:
            if outputs.get('target') is not None:
                # The ranking formatter drops this column and uses the encoded
                # target tensor to produce LABEL. Avoid decoding numeric
                # multicategorical labels through a string separator.
                df[target_col_name] = None
            return create_multilabel_ranking_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outputs['prob'],
                target=outputs.get('target'),
                top_k=self._eval_top_k_from_outputs(),
            )

        target = self.target_encoder.decode(outputs['target'])
        df[target_col_name] = target
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return create_multiclass_prediction_output(
                df=df,
                target_encoder=self.target_encoder,
                prob=outputs['prob'],
                top_k=self._eval_top_k_from_outputs(),
            )

        pred = outputs['pred']
        quantiles = quantile_names = None
        if isinstance(self.pl_model.loss, DistributionLoss):
            pred = self.pl_model.loss.to_prediction(pred)
            quantiles = outputs.get('prob')
            quantile_names = [
                str(n) for n in self.pl_model.loss.distribution.quantiles
            ]
        elif isinstance(self.pl_model.loss, MultiQuantileLoss):
            quantiles = outputs.get('prob')
            quantile_names = [str(n) for n in self.pl_model.loss.quantiles]

        out_timestamps = None
        if self.task.task_type == TaskType.FORECASTING:
            out_timestamps = _forecasting_out_timestamps(
                self.pl_model, outputs['time'])

        return format_prediction(
            df=df,
            target_col_name=target_col_name,
            target_encoder=self.target_encoder,
            task_type=self.task.task_type,
            pred=pred,
            prob=outputs.get('prob'),
            quantiles=quantiles,
            quantile_names=quantile_names,
            time_col_name=time_col,
            out_timestamps=out_timestamps,
            array_targets=allow_array_targets(),
        )

    def _eval_top_k_from_outputs(self) -> int:
        top_k = min(self.num_targets, _MAX_EVAL_TOP_K)
        return max(1, top_k)

    def predict(
        self,
        *,
        output_dir: str,
        task: Any,
        job_start_time: Any = None,
        return_embeddings: bool = False,
        return_predictions: bool = True,
        ignore_time_col: bool = False,
        pred_time_col_name: str | None = None,
        output_extra_fields: dict[str, list[str]] | None = None,
        max_output_file_size: int = 1024**3,
        progress_log_dir: str | None = None,
        progress_log_id: int = 0,
        partition_id: int = 0,
        total_partitions: int = 1,
        output_partition_ids: list[int] | None = None,
        runtime_num_workers: int = 1,
        runtime_worker_id: int = 0,
        top_k: int | None = None,
        artifact_stats_collector: PredictionArtifactStatsCollector
        | None = None,
        task_manifest: DiskgraphTaskManifest | None = None,
    ) -> Any:
        """TT-NG batch prediction over the wheel-backed predict stage.

        Serves the model over the ``predict`` stage via the wheel iterator and
        writes to ``output_dir``. Mirrors the node-pred branch of
        ``Trainer.predict_v2`` but sources metadata from the bundle / ``self.task``
        and keys output on ``pkey`` (no ``arange``).
        """
        emit_embeddings = bool(return_embeddings)
        emit_predictions = bool(return_predictions)
        if not emit_embeddings and not emit_predictions:
            raise ValueError(
                'Batch prediction must request predictions or embeddings.')

        import math

        import torch
        from kumo_core.train.formatting import (
            format_embeddings,
            format_prediction,
        )
        # yapf: enable
        from kumo_core.train.loss import DistributionLoss, MultiQuantileLoss
        from kumo_core.train.output import (
            PredOutput,
            StreamingPredOutputWriter,
        )
        from kumo_core.train.trainer import temporary_callbacks
        from pytorch_lightning.callbacks import BasePredictionWriter

        DiskgraphNestedTrainer.set_runtime_worker_shard(
            self,
            runtime_num_workers=runtime_num_workers,
            runtime_worker_id=runtime_worker_id,
        )
        if self.task.task_type in (TaskType.STATIC_LINK_PREDICTION,
                                   TaskType.TEMPORAL_LINK_PREDICTION):
            link_prediction_kwargs: dict[str, Any] = dict(
                output_dir=output_dir, task=task, job_start_time=job_start_time,
                return_embeddings=return_embeddings,
                return_predictions=return_predictions,
                ignore_time_col=ignore_time_col,
                pred_time_col_name=pred_time_col_name,
                output_extra_fields=output_extra_fields,
                max_output_file_size=max_output_file_size,
                progress_log_dir=progress_log_dir,
                progress_log_id=progress_log_id, partition_id=partition_id,
                total_partitions=total_partitions,
                output_partition_ids=output_partition_ids, top_k=top_k,
                task_manifest=task_manifest)
            if artifact_stats_collector is not None:
                link_prediction_kwargs["artifact_stats_collector"] = (
                    artifact_stats_collector)
            return self._predict_lp(**link_prediction_kwargs)

        self._prediction_mode = True
        if emit_predictions:
            top_k = self._preprocess_top_k(top_k)
        self.init(task_manifest=task_manifest)
        self.pl_model.eval_top_k = None
        output_partition_ids = output_partition_ids or [partition_id]

        callbacks = _prediction_progress_callbacks(
            self.datamodule, progress_log_dir, progress_log_id)
        embedding_extra_fields = set((output_extra_fields
                                      or {}).get('EMBEDDINGS', ()))

        def _build_output(prediction: dict[str, Any]) -> PredOutput:
            # predict_step emits decoded ``pred`` + ``prob`` just like the
            # legacy predict_v2 path.
            pkey = prediction.get('pkey')
            time = prediction.get('time')
            if pkey is None:
                raise RuntimeError('[TT-NG] predict output missing pkey')

            fkey_names = list(task.train_table.foreign_key_columns)
            base_df = _pkey_frame(pkey, fkey_names)

            seed_node_type = self.datamodule._seed_node_type
            time_col = (pred_time_col_name
                        or self.datamodule.time_col_dict.get(seed_node_type)
                        or 'TIMESTAMP')
            if time is not None and not ignore_time_col:
                # ``time`` is adapter-normalized canonical ns; read it directly
                # (see ``_seed_time_to_datetime64_ns``).
                base_df[time_col] = _seed_time_to_datetime64_ns(time)
            if job_start_time is not None:
                base_df['JOB_TIMESTAMP'] = pd.Timestamp(
                    job_start_time).tz_localize(None)

            main_df = base_df.copy()
            if self.task.task_type == TaskType.FORECASTING:
                if ('ANCHOR_TIMESTAMP' in embedding_extra_fields
                        and time_col in main_df):
                    main_df['ANCHOR_TIMESTAMP'] = main_df[time_col]
                main_df = main_df.drop(columns=[time_col], errors='ignore')
            if not emit_predictions:
                df = pd.DataFrame()
            else:
                pred = prediction.get('pred')
                prob = prediction.get('prob')
                if pred is None:
                    raise RuntimeError('[TT-NG] predict output missing pred')
                df = base_df

            if (emit_predictions and self.task.task_type
                    == TaskType.MULTICLASS_CLASSIFICATION):
                if top_k is None:
                    raise RuntimeError(
                        '[TT-NG] multiclass prediction requires top_k')
                df = create_multiclass_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=prob,
                    top_k=top_k,
                )
            elif (emit_predictions and self.task.task_type
                  == TaskType.MULTILABEL_CLASSIFICATION):
                df = create_multilabel_classification_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=prob,
                    pred=pred,
                    target=None,
                    top_k=top_k or self.target_encoder.num_categories,
                )
            elif (emit_predictions
                  and self.task.task_type == TaskType.MULTILABEL_RANKING):
                df = create_multilabel_ranking_prediction_output(
                    df=df,
                    target_encoder=self.target_encoder,
                    prob=prob,
                    target=None,
                    top_k=top_k or self.target_encoder.num_categories,
                )
            elif emit_predictions:
                quantiles = quantile_names = None
                if isinstance(self.pl_model.loss, DistributionLoss):
                    pred = self.pl_model.loss.to_prediction(pred)
                    quantiles = prob
                    quantile_names = [
                        str(n)
                        for n in self.pl_model.loss.distribution.quantiles
                    ]
                elif isinstance(self.pl_model.loss, MultiQuantileLoss):
                    quantiles = prob
                    quantile_names = [
                        str(n) for n in self.pl_model.loss.quantiles
                    ]

                out_timestamps = None
                if self.task.task_type == TaskType.FORECASTING:
                    out_timestamps = _forecasting_out_timestamps(
                        self.pl_model, time)

                df = format_prediction(
                    df=df,
                    target_col_name='TARGET',
                    target_encoder=self.target_encoder,
                    task_type=self.task.task_type,
                    pred=pred,
                    prob=prob,
                    quantiles=quantiles,
                    quantile_names=quantile_names,
                    top_k=top_k,
                    time_col_name=time_col,
                    out_timestamps=out_timestamps,
                    array_targets=allow_array_targets(),
                )

            embedding = None
            if return_embeddings:
                embedding_arr = prediction.get('embedding')
                if embedding_arr is None:
                    raise RuntimeError(
                        '[TT-NG] predict output missing requested embeddings')
                if isinstance(embedding_arr, torch.Tensor):
                    embedding_arr = embedding_arr.detach().cpu().numpy()
                embedding = format_embeddings(embedding_arr, main_df)
            return PredOutput(df, embedding)

        datamodule = self.datamodule

        class _ChunkedPredictionWriter(BasePredictionWriter):
            def __init__(self, partition_ids: list[int],
                         flush_every_batches: int) -> None:
                super().__init__(write_interval='batch')
                self.partition_ids = partition_ids
                self.flush_every_batches = flush_every_batches
                self.batch_count = 0
                self.saved_output_count = 0
                self.last_output: PredOutput | None = None
                self.stream: StreamingPredOutputWriter | None = None
                self.partition_idx: int | None = None
                self.pending_prediction: dict[str, Any] | None = None

            def _finish_partition(self) -> None:
                if self.stream is None or self.partition_idx is None:
                    return
                partition_id = self.partition_ids[self.partition_idx]
                self.stream.close()
                if artifact_stats_collector is not None:
                    artifact_stats_collector.add_partition(
                        partition_id, self.stream.artifact_stats)
                self.stream = None

            def write_prediction(self, prediction: dict[str, Any]) -> None:
                partition_idx = min(
                    self.batch_count // self.flush_every_batches,
                    len(self.partition_ids) - 1,
                )
                if partition_idx != self.partition_idx:
                    self._finish_partition()
                    self.stream = StreamingPredOutputWriter(
                        output_dir,
                        self.partition_ids[partition_idx],
                        max_size_per_file=max_output_file_size,
                        output_extra_fields=output_extra_fields,
                        job_start_time=job_start_time,
                        write_predictions=emit_predictions,
                        artifact_stats_sample_rows=({
                            artifact_type:
                            artifact_stats_collector.sample_rows_needed(
                                artifact_type)
                            for artifact_type in
                            artifact_stats_collector.expected_artifact_types
                        } if artifact_stats_collector is not None else None),
                    )
                    self.partition_idx = partition_idx
                    self.saved_output_count += 1
                out = _build_output(prediction)
                assert self.stream is not None
                self.stream.write(out)
                self.last_output = out
                self.batch_count += 1

            def write_on_batch_end(self, trainer, pl_module, prediction,
                                   batch_indices, batch, batch_idx,
                                   dataloader_idx) -> None:
                if prediction is None:
                    return
                # Lightning invokes callback hooks before the module's
                # on_predict_batch_end hook. The module hook replaces raw
                # logits with decoded pred/prob values, so retain at most one
                # batch and write it on the next callback after that mutation.
                if self.pending_prediction is not None:
                    self.write_prediction(self.pending_prediction)
                self.pending_prediction = prediction

            def on_predict_end(self, trainer, pl_module) -> None:
                if self.pending_prediction is not None:
                    self.write_prediction(self.pending_prediction)
                    self.pending_prediction = None
                self.close()
                _shutdown_predict_workers(datamodule)

            def close(self) -> None:
                self._finish_partition()

            def abort(self) -> None:
                if self.stream is not None:
                    self.stream.abort()
                for part_id in self.partition_ids[:self.saved_output_count]:
                    StreamingPredOutputWriter.cleanup_partition(
                        output_dir, part_id)

        total_batches = len(self.datamodule.predict_dataloader())
        flush_every_batches = max(
            1, math.ceil(total_batches / len(output_partition_ids)))
        writer = _ChunkedPredictionWriter(output_partition_ids,
                                          flush_every_batches)
        callbacks.append(writer)
        had_flag = hasattr(self.pl_model, '_kumo_return_predict_embeddings')
        old_flag = getattr(self.pl_model, '_kumo_return_predict_embeddings',
                           None)
        if emit_embeddings:
            self.pl_model._kumo_return_predict_embeddings = True
        try:
            with _defer_predict_worker_shutdown(self.datamodule):
                with temporary_callbacks(self.pl_trainer, callbacks), \
                     suppress_iterable_len_warning(), \
                     suppress_batch_indices_warning():
                    returned_outputs = self.pl_trainer.predict(
                        model=self.pl_model,
                        datamodule=self.datamodule,
                        ckpt_path=fsspec_mangle(self.best_ckpt),
                        return_predictions=False,
                    )
                # Unit-test trainers and older Lightning adapters may return
                # outputs without invoking BasePredictionWriter callbacks.
                if writer.batch_count == 0 and returned_outputs is not None:
                    for prediction in returned_outputs:
                        writer.write_prediction(prediction)
                writer.close()
                # Real Lightning already ran this barrier in writer
                # on_predict_end, before Trainer._teardown. Fake/older adapters
                # that omit callbacks use this idempotent fallback.
                _shutdown_predict_workers(self.datamodule)
        except BaseException:
            _abort_prediction_writer(writer)
            try:
                _shutdown_predict_workers(self.datamodule)
            except BaseException:
                logger.exception(
                    'Failed to close prediction DataLoader after error')
            raise
        finally:
            if emit_embeddings:
                if had_flag:
                    self.pl_model._kumo_return_predict_embeddings = old_flag
                else:
                    delattr(self.pl_model, '_kumo_return_predict_embeddings')
            # The per-call writer class is a reference cycle; release what its
            # methods close over so this trainer and its data module free on
            # return rather than at the next garbage collection.
            datamodule = _build_output = None
        if writer.saved_output_count == 0:
            raise RuntimeError('[TT-NG] predict produced no batches')
        logger.info('[TT-NG] saved batch prediction output to %s', output_dir)
        assert writer.last_output is not None
        return writer.last_output

    def _predict_lp(
        self,
        *,
        output_dir: str,
        task: Any,
        job_start_time: Any = None,
        return_embeddings: bool = False,
        return_predictions: bool = True,
        ignore_time_col: bool = False,
        pred_time_col_name: str | None = None,
        output_extra_fields: dict[str, list[str]] | None = None,
        max_output_file_size: int = 1024**3,
        progress_log_dir: str | None = None,
        progress_log_id: int = 0,
        partition_id: int = 0,
        total_partitions: int = 1,
        output_partition_ids: list[int] | None = None,
        top_k: int | None = None,
        artifact_stats_collector: PredictionArtifactStatsCollector
        | None = None,
        task_manifest: DiskgraphTaskManifest | None = None,
    ) -> Any:
        """TT-NG batch prediction for link prediction (all 4 model families).

        Ranks RHS candidates per anchor (``predict_step`` -> ``pkey`` / ``target``
        [N,k] dense node-ids / ``score``), maps node-ids back to RHS pkeys via a
        PK-only rowpack scan (position == node-id), and emits the
        legacy LP table (LHS fkey / ``CLASS`` / ``SCORE``, one row per anchor,k)
        when predictions are requested. With ``return_embeddings`` also emits
        the ``create_lp_embedding_output`` table (LHS per anchor + the filtered
        RHS catalog for two-tower/shallow-RHS).
        """
        import math

        import numpy as np
        import torch
        from kumo_core.train.output import (
            PredOutput,
            StreamingPredOutputWriter,
        )
        from kumo_core.train.trainer import temporary_callbacks
        from pytorch_lightning.callbacks import BasePredictionWriter

        emit_embeddings = bool(return_embeddings)
        emit_predictions = bool(return_predictions)
        if not emit_embeddings and not emit_predictions:
            raise ValueError(
                'Batch prediction must request predictions or embeddings.')

        self._prediction_mode = True
        self.init(task_manifest=task_manifest)
        output_partition_ids = output_partition_ids or [partition_id]

        has_emb_hook_fn = (self.pl_model._get_register_embedding_hook_fn()
                           is not None)
        if return_embeddings and not has_emb_hook_fn:
            raise ValueError(
                'Embeddings were requested, but this link-prediction model '
                'does not implement an embedding hook.')
        if emit_embeddings and self.pl_model.has_embedding_hook():
            self.pl_model.reset_embedding_hook()

        # Request outputs are authoritative. Embeddings-only link prediction
        # must not synthesize a top-k or enter MIPS, even when model defaults
        # and an RHS catalog are available.
        num_rhs = int(getattr(self.pl_model, 'num_rhs_nodes', 0) or 0)
        if not emit_predictions:
            self.pl_model.pred_top_k = 0
        else:
            if top_k is not None and top_k <= 0:
                raise ValueError(f'top_k must be greater than 0, got {top_k}')
            pred_top_k = (top_k or getattr(self.pl_model, 'pred_top_k', None)
                          or getattr(self.pl_model, 'eval_top_k', None) or 100)
            if num_rhs:
                pred_top_k = min(int(pred_top_k), num_rhs)
            self.pl_model.pred_top_k = int(pred_top_k)

        # kumo-pred reads a PredictionProgress file after predict.
        _cbs = _prediction_progress_callbacks(
            self.datamodule, progress_log_dir, progress_log_id)

        def _np(x: Any) -> Any:
            if x is None:
                return None
            return x.cpu().numpy() if isinstance(x, torch.Tensor) \
                else np.asarray(x)

        def _rhs_pkeys() -> tuple[np.ndarray, np.ndarray]:
            # Canonical RHS node-id -> raw PK, materialized once per process.
            rhs_name = self.datamodule.task_manifest.target_node_type
            catalog = getattr(self, '_rhs_primary_key_catalog', None)
            node_ids = getattr(self, '_rhs_primary_key_node_ids', None)
            if catalog is None or node_ids is None:
                graph_metadata = self.datamodule.task_manifest.graph_runtime_metadata
                index_path = (graph_metadata or {}).get('graph_index_path')
                if not index_path:
                    raise RuntimeError(
                        '[TT-NG LP] RHS lookup requires graph_index_path')
                node_ids, catalog = _rhs_primary_key_catalog(
                    index_path, rhs_name, num_rhs)
                self._rhs_primary_key_catalog = catalog
                self._rhs_primary_key_node_ids = node_ids
            return node_ids, catalog

        def _rhs_bounds(part_id: int, rhs_count: int) -> tuple[int, int]:
            per_partition = max(1, math.ceil(rhs_count / total_partitions))
            start = part_id * per_partition
            return start, min(rhs_count, start + per_partition)

        catalog_node_ids, rhs_pk_arr = _rhs_pkeys()
        if hasattr(self.pl_model, 'set_rhs_index_map'):
            rhs_index_map = pd.DataFrame(
                {
                    NODE_INDEX_COLUMN: catalog_node_ids.astype(
                        np.int64, copy=False)
                },
                index=pd.Index(
                    rhs_pk_arr,
                    name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
                ),
            )
            # Lightning restores ``best_ckpt`` inside predict(). Queue the
            # current raw-key map so the module's post-load hook can reorder
            # old learned rows and mean-initialize genuinely new entities.
            self.pl_model.set_rhs_index_map(rhs_index_map)
        # Infer once without retaining another copy of the full RHS catalog.
        rhs_pk_mixed = (emit_predictions and rhs_pk_arr.dtype.kind == 'O'
                        and pd.api.types.infer_dtype(
                            rhs_pk_arr, skipna=True).startswith('mixed'))
        if not emit_predictions:
            rhs_pk_type = pa.null()
        elif rhs_pk_mixed:
            rhs_pk_type = pa.string()
        elif rhs_pk_arr.dtype.kind == 'O':
            rhs_pk_type = pa.infer_type(rhs_pk_arr, from_pandas=True)
        else:
            rhs_pk_type = pa.from_numpy_dtype(rhs_pk_arr.dtype)
        class_metadata = None
        if pa.types.is_integer(rhs_pk_type):
            class_dtype = (pd.UInt64Dtype()
                           if pa.types.is_unsigned_integer(rhs_pk_type) else
                           pd.Int64Dtype())
            class_metadata = pa.Table.from_pandas(
                pd.DataFrame({'CLASS': pd.Series(dtype=class_dtype)}),
                preserve_index=False).schema.metadata
        embedding_ids_as_string = rhs_pk_arr.dtype.kind in {'O', 'S', 'U'}
        fkey_columns = list(task.train_table.foreign_key_columns)
        lhs_fkey_name = fkey_columns[0] if fkey_columns else 'ENTITY'
        seed_node_type = self.datamodule._seed_node_type
        time_col = (pred_time_col_name
                    or self.datamodule.time_col_dict.get(seed_node_type)
                    or 'TIMESTAMP')
        embedding_extra_fields = set((output_extra_fields
                                      or {}).get('EMBEDDINGS', ()))

        def _build_chunk_output(prediction: dict[str, Any]) -> ArrowPredOutput:
            lhs_table, rhs_table = self.pl_model.entity_table_names
            return _build_ttng_lp_chunk_output(
                prediction,
                rhs_node_ids=catalog_node_ids,
                rhs_pk_arr=rhs_pk_arr,
                rhs_pk_type=rhs_pk_type,
                rhs_pk_mixed=rhs_pk_mixed,
                lhs_fkey_name=lhs_fkey_name,
                time_col=time_col,
                ignore_time_col=ignore_time_col,
                job_start_time=job_start_time,
                emit_predictions=emit_predictions,
                emit_embeddings=emit_embeddings,
                lhs_table=lhs_table,
                rhs_table=rhs_table,
                embedding_extra_fields=embedding_extra_fields,
                embedding_ids_as_string=embedding_ids_as_string,
                class_metadata=class_metadata,
            )

        def _rhs_embedding_tables(
            output_partition_id: int,
            rhs_emb: torch.Tensor,
            rhs_node_ids: np.ndarray,
        ) -> Iterator[pa.Table]:
            rhs_count = rhs_emb.shape[0]
            if rhs_node_ids.shape != (rhs_count, ):
                raise RuntimeError(
                    '[TT-NG LP] RHS embedding/catalog row count mismatch: '
                    f'{rhs_count} embeddings for {rhs_node_ids.shape[0]} ids')
            start, end = _rhs_bounds(output_partition_id, rhs_count)
            if start >= end:
                return
            row_width = (rhs_emb[0].numel() * rhs_emb.element_size() + 128)
            rows_per_frame = max(1, (64 * 1024 * 1024) // row_width)
            _, rhs_table = self.pl_model.entity_table_names
            rhs_fkey = (fkey_columns[1]
                        if len(fkey_columns) > 1 else lhs_fkey_name)
            for frame_start in range(start, end, rows_per_frame):
                frame_end = min(end, frame_start + rows_per_frame)
                embedding_values = (
                    rhs_emb[frame_start:frame_end].detach().cpu().numpy())
                frame_node_ids = rhs_node_ids[frame_start:frame_end]
                positions = np.searchsorted(catalog_node_ids, frame_node_ids)
                valid = positions < catalog_node_ids.shape[0]
                if catalog_node_ids.shape[0]:
                    valid &= catalog_node_ids[np.minimum(
                        positions,
                        catalog_node_ids.shape[0] - 1)] == frame_node_ids
                if not bool(valid.all()):
                    raise RuntimeError(
                        '[TT-NG LP] filtered RHS node IDs are not canonical')
                raw_pkeys = rhs_pk_arr[positions]
                count = frame_end - frame_start
                arrays: list[pa.Array] = [
                    (_arrow_string_array(raw_pkeys)
                     if embedding_ids_as_string else pa.array(raw_pkeys)),
                    pa.array(np.full(count, rhs_table)),
                ]
                names = ['ID', 'TYPE']
                if self.pl_model.entity_table_names[0] == rhs_table:
                    arrays.append(pa.array(np.full(count, rhs_fkey)))
                    names.append('NAME')
                flat = pa.array(embedding_values.reshape(-1))
                offsets = pa.array(
                    np.arange(count + 1, dtype=np.int32) *
                    embedding_values.shape[1])
                arrays.append(pa.ListArray.from_arrays(offsets, flat))
                names.append('EMBEDDING')
                if 'ANCHOR_TIMESTAMP' in embedding_extra_fields:
                    arrays.append(pa.nulls(count, type=pa.timestamp('ns')))
                    names.append('ANCHOR_TIMESTAMP')
                yield pa.Table.from_arrays(arrays, names=names)

        datamodule = self.datamodule

        class _LPChunkedPredictionWriter(BasePredictionWriter):
            def __init__(self, partition_ids: list[int],
                         flush_every_batches: int) -> None:
                super().__init__(write_interval='batch')
                self.partition_ids = partition_ids
                self.flush_every_batches = flush_every_batches
                self.batch_count = 0
                self.saved_output_count = 0
                self.last_output: PredOutput | ArrowPredOutput | None = None
                self.rhs_emb: torch.Tensor | None = None
                self.rhs_node_ids: np.ndarray | None = None
                self.stream: StreamingPredOutputWriter | None = None
                self.partition_idx: int | None = None

            def capture_rhs_embedding(self, pl_module: Any) -> None:
                if not emit_embeddings or self.rhs_emb is not None:
                    return
                self.rhs_emb = pl_module.mips_index._key.detach()
                valid_rhs = getattr(pl_module, 'valid_rhs_node_idx', None)
                self.rhs_node_ids = (np.arange(
                    self.rhs_emb.shape[0], dtype=np.int64) if valid_rhs is None
                                     else _np(valid_rhs).astype(
                                         np.int64, copy=False))

            def _finish_partition(self) -> None:
                if self.stream is None or self.partition_idx is None:
                    return
                if emit_embeddings and self.rhs_emb is not None:
                    assert self.rhs_node_ids is not None
                    part_id = self.partition_ids[self.partition_idx]
                    for table in _rhs_embedding_tables(part_id, self.rhs_emb,
                                                       self.rhs_node_ids):
                        self.stream.write_embedding(table)
                self.stream.close()
                if artifact_stats_collector is not None:
                    artifact_stats_collector.add_partition(
                        self.partition_ids[self.partition_idx],
                        self.stream.artifact_stats,
                    )
                self.stream = None

            def write_prediction(self, prediction: dict[str, Any],
                                 pl_module: Any) -> None:
                self.capture_rhs_embedding(pl_module)
                partition_idx = min(
                    self.batch_count // self.flush_every_batches,
                    len(self.partition_ids) - 1,
                )
                new_partition = partition_idx != self.partition_idx
                if new_partition:
                    self._finish_partition()
                    self.stream = StreamingPredOutputWriter(
                        output_dir,
                        self.partition_ids[partition_idx],
                        max_size_per_file=max_output_file_size,
                        output_extra_fields=output_extra_fields,
                        job_start_time=job_start_time,
                        write_predictions=emit_predictions,
                        artifact_stats_sample_rows=({
                            artifact_type:
                            artifact_stats_collector.sample_rows_needed(
                                artifact_type)
                            for artifact_type in
                            artifact_stats_collector.expected_artifact_types
                        } if artifact_stats_collector is not None else None),
                    )
                    self.partition_idx = partition_idx
                    self.saved_output_count += 1
                out = _build_chunk_output(prediction)
                assert self.stream is not None
                self.stream.write(out)
                self.last_output = out
                self.batch_count += 1

            def write_on_batch_end(self, trainer, pl_module, prediction,
                                   batch_indices, batch, batch_idx,
                                   dataloader_idx) -> None:
                if prediction is not None:
                    self.write_prediction(prediction, pl_module)

            def on_predict_end(self, trainer, pl_module) -> None:
                self.close()
                _shutdown_predict_workers(datamodule)

            def close(self) -> None:
                self._finish_partition()

            def abort(self) -> None:
                if self.stream is not None:
                    self.stream.abort()
                    self.stream = None
                for part_id in self.partition_ids[:self.saved_output_count]:
                    StreamingPredOutputWriter.cleanup_partition(
                        output_dir, part_id)

        total_batches = len(self.datamodule.predict_dataloader())
        flush_every_batches = max(
            1, math.ceil(total_batches / len(output_partition_ids)))
        writer = _LPChunkedPredictionWriter(output_partition_ids,
                                            flush_every_batches)
        _cbs.append(writer)
        had_flag = hasattr(self.pl_model, '_kumo_return_predict_embeddings')
        old_flag = getattr(self.pl_model, '_kumo_return_predict_embeddings',
                           None)
        if emit_embeddings:
            self.pl_model._kumo_return_predict_embeddings = True
        try:
            with _defer_predict_worker_shutdown(self.datamodule):
                with temporary_callbacks(self.pl_trainer, _cbs), \
                     suppress_iterable_len_warning(), \
                     suppress_batch_indices_warning():
                    returned_outputs = self.pl_trainer.predict(
                        model=self.pl_model,
                        datamodule=self.datamodule,
                        ckpt_path=fsspec_mangle(self.best_ckpt),
                        return_predictions=False,
                    )
                if writer.batch_count == 0 and returned_outputs is not None:
                    for prediction in returned_outputs:
                        writer.write_prediction(prediction, self.pl_model)
                writer.close()
                _shutdown_predict_workers(self.datamodule)
        except BaseException:
            _abort_prediction_writer(writer)
            try:
                _shutdown_predict_workers(self.datamodule)
            except BaseException:
                logger.exception(
                    'Failed to close prediction DataLoader after error')
            raise
        finally:
            if emit_embeddings:
                if had_flag:
                    self.pl_model._kumo_return_predict_embeddings = old_flag
                else:
                    delattr(self.pl_model, '_kumo_return_predict_embeddings')
            # See predict(): release what the per-call writer class closes over.
            datamodule = _build_chunk_output = None
        if writer.saved_output_count == 0:
            raise RuntimeError('[TT-NG LP] predict produced no batches')
        logger.info('[TT-NG LP] saved batch prediction output to %s',
                    output_dir)
        assert writer.last_output is not None
        if isinstance(writer.last_output, ArrowPredOutput):
            return writer.last_output.to_pandas()
        return writer.last_output

    def _validate_mask(self, stage: Any) -> bool:
        # Parent verifies via feature_store (absent on TT-NG); an empty stage
        # just yields zero batches.
        return True

    def _configure_model_triple_from_manifest(
        self,
        tm: Any,
        *,
        time_col_dict_fn: Any,
        has_train_table_time_col: bool,
    ) -> None:
        """Stamp model/encoder/head metadata from the bundle (counterpart to
        ``Trainer.initialize_model``). Each ``is_fully_specified`` guard lets
        AutoML pre-specify any subset.
        """
        train_table_name = tm.metadata['train_table_name']
        feature_schema = tm.feature_schema

        meta = tm.graph_runtime_metadata or {}
        node_types: list[str] = sorted(meta.get('node_types') or [])

        # Read verbose edge names from index_metadata.json so HeteroConv layers
        # match the keys the wheel emits.
        edge_types = _edge_types_from_index_metadata(meta)

        if not self.model.is_fully_specified:
            self.model.metadata = (node_types, edge_types)
            self.model.validate()

        if not self.encoder.is_fully_specified:
            # collect_encoders keys by bare table_name (matching the wheel);
            # setdefault empty entries for encoder-less node types.
            table_encoder_dict = dict(self.encoders.encoders)
            for ns in (*node_types, train_table_name):
                table_encoder_dict.setdefault(ns, {})
            # The adapter emits an empty seed-block TF, so clear its encoders.
            table_encoder_dict[train_table_name] = {}
            self.encoder.table_encoder_dict = table_encoder_dict
            self.encoder.out_channels = self.model.channels
            if (not has_train_table_time_col
                    or self.task.task_type == TaskType.STATIC_LINK_PREDICTION):
                self.encoder.time_col_dict = {}
            else:
                self.encoder.time_col_dict = time_col_dict_fn(feature_schema)
            # Mask leakage exclusions (same-table PQL target); {} leaked it.
            self.encoder.exclude_cols_dict = {
                get_table_name(k): v
                for k, v in (self.task.exclude_cols_dict or {}).items()
            }
            self.encoder.validate()

        past_encoder = getattr(self, "past_encoder", None)
        year_over_year_encoder = getattr(self, "year_over_year_encoder", None)
        if past_encoder is not None and not past_encoder.is_fully_specified:
            past_encoder.in_channels = 1
            past_encoder.out_channels = self.model.channels
            past_encoder.length = self.task.lag_timesteps

        if (year_over_year_encoder is not None
                and not year_over_year_encoder.is_fully_specified):
            from kumo_core.encoders.time import TimeEncoder

            year_over_year_encoder.in_channels = 1 + TimeEncoder.out_channels
            year_over_year_encoder.out_channels = self.model.channels

        if not self.head.is_fully_specified:
            from kumo_core.train.loss import (
                DistributionLoss,
                MultiQuantileLoss,
                loss_resolver,
            )

            self.head.in_channels = self.model.channels
            is_lp = self.task.task_type.is_link_pred
            if is_lp:
                # Set LinkPredHead's lazy flags before validate(): MLP scorer for
                # bidirectional (ID/Hybrid), RHS projector for two-tower only.
                module_type = (getattr(
                    self.module_kwargs, "module", None) if getattr(
                        self, "module_kwargs", None) is not None else None)
                use_bidirectional = bool(module_type is not None and getattr(
                    module_type, "use_bidirectional_sampling", False))
                self.head.use_mlp_head = use_bidirectional
                self.head.use_rhs_projector = (
                    module_type == LinkPredModuleType.TWO_TOWER)
            else:
                loss = loss_resolver(
                    self.task.task_type,
                    self.loss,
                    target_encoder=self.target_encoder,
                )
                if self.task.task_type == TaskType.FORECASTING:
                    seed_node_type = getattr(tm, "seed_node_type", None)
                    num_entities = (_get_num_nodes_from_index_metadata(
                        meta, seed_node_type) if seed_node_type else 0)
                    if not num_entities:
                        num_entities = int((tm.stats or {}).get("num_entities")
                                           or 0)
                    if not num_entities:
                        raise ValueError(
                            "TT-NG forecasting requires num_entities from "
                            "index metadata or task-dataset stats")
                    self.head.num_entities = num_entities
                    self.head.past_channels = (past_encoder.out_channels
                                               if past_encoder is not None else
                                               (self.task.lag_timesteps or 0))
                    self.head.past_year_channels = (
                        year_over_year_encoder.out_channels
                        if year_over_year_encoder is not None else 0)
                    self.head.unix_step_size = self.task.unix_step_size
                if isinstance(loss, MultiQuantileLoss):
                    self.head.out_channels = len(loss.quantiles)
                else:
                    self.head.out_channels = self.num_targets
                if isinstance(loss, DistributionLoss):
                    self.head.param_keys = (
                        loss.distribution.distribution_parameters)
                else:
                    self.head.param_keys = []
            self.head.validate()

    @classmethod
    def load(
        cls,
        task: Any,
        root_dir: str,
        version: int | None = -1,
        *,
        diskgraph_task_dataset_bundle_dir: str,
        encoders: Any = None,
        **kwargs: Any,
    ) -> "DiskgraphNestedTrainer":
        """Load a trainer from a saved AutoML version directory.

        ``from_config`` reassigns ``cls`` to the parent, so let the parent's
        ``load`` run, then rebind ``__class__`` and stamp on the bundle dir.
        """
        if diskgraph_task_dataset_bundle_dir is None:
            raise ValueError('DiskgraphNestedTrainer.load requires '
                             'diskgraph_task_dataset_bundle_dir; got None')
        # TT-NG doesn't use feature_store / graph_store; drop them if passed.
        kwargs.pop('feature_store', None)
        kwargs.pop('graph_store', None)

        base = NestedTrainer.load(
            task,
            root_dir,
            version=version,
            feature_store=None,
            graph_store=None,
            encoders=encoders,
            **kwargs,
        )
        base.__class__ = cls
        base.diskgraph_task_dataset_bundle_dir = (  # type: ignore[attr-defined]
            diskgraph_task_dataset_bundle_dir)
        if kwargs.get('random_state_seed') is not None:
            base.random_state_seed = kwargs['random_state_seed']
        elif not hasattr(base, 'random_state_seed'):
            base.random_state_seed = None
        return base  # type: ignore[return-value]
