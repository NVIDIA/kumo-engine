# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct Rust-backed runtime batch iterators from a runtime manifest.

Kumo passes a flat ``kind="diskgraph_task_dataset"`` runtime manifest plus
iterator options, and this module constructs a Rust-backed
:class:`diskgraph._c.RuntimeBatchIterator` over the merged parent +
seed-table extension catalog. The iterator yields PyDicts matching PR
28765's ``runtime_batch_to_hetero`` contract.

Cross-boundary FK edges are pre-resolved at Phase 1 ingest, so we don't
need to surface ``seed_id_column`` / ``seed_node_type`` / ``time_column``
on the Python side anymore — the engine reads them from FK metadata and
the seed rowpack directly.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

from ._exceptions import DiskgraphConfigError


TaskDatasetRef = str


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise DiskgraphConfigError(f"{path} must contain a JSON object")
    return payload


def _manifest_view(
    manifest_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the flat Kumo runtime manifest needed for native iteration."""
    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "diskgraph_task_dataset":
        raise DiskgraphConfigError(
            "runtime manifest kind must be 'diskgraph_task_dataset'"
        )
    split_info = {
        "stage_counts": {
            stage: int((ref or {}).get("row_count", 0))
            for stage, ref in (manifest.get("stages") or {}).items()
        }
    }
    return manifest, split_info


def _use_seed_timestamps(manifest: Mapping[str, Any]) -> bool:
    task = manifest.get("task") or {}
    if not isinstance(task, Mapping) or "is_temporal" not in task:
        return True
    return bool(task.get("is_temporal"))


def _stage_tables(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("stages") or {}
    if not isinstance(raw, Mapping):
        raise DiskgraphConfigError("manifest.stages must be an object")

    out: dict[str, dict[str, Any]] = {}
    for stage, value in raw.items():
        if not isinstance(value, Mapping):
            raise DiskgraphConfigError(
                f"stages[{stage!r}] must be a table-ref object"
            )
        table_name = value.get("table_name")
        if not table_name:
            raise DiskgraphConfigError(
                f"stages[{stage!r}].table_name is required"
            )
        index_path = value.get("index_path")
        if not index_path:
            raise DiskgraphConfigError(
                f"stages[{stage!r}].index_path is required"
            )
        out[str(stage)] = {
            "table_name": str(table_name),
            "index_path": str(index_path),
        }
        binary_label_index = value.get("binary_label_index")
        if isinstance(binary_label_index, Mapping):
            out[str(stage)]["binary_label_index"] = dict(binary_label_index)
        sample_weight_index = value.get("sample_weight_index")
        if isinstance(sample_weight_index, Mapping):
            out[str(stage)]["sample_weight_index"] = dict(sample_weight_index)
    return out


def _graph_index_path(manifest: Mapping[str, Any]) -> str:
    """Parent DG index root from manifest."""
    graph = manifest.get("graph") or {}
    candidates = [
        manifest.get("graph_index_path"),
        graph.get("index_path"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise DiskgraphConfigError(
        "runtime manifest must include graph.index_path"
    )


def _extension_ref_from_stage_path(parent_root: str, stage_index_path: str) -> str:
    """Recover the seed-table extension reference from a stage ``index_path``.

    The stage path itself is the source of truth: return a relative
    ``train_tables/<tt_id>`` or ``pred_tables/<job_id>`` ref under the parent.
    """
    abs_parent = os.path.abspath(parent_root.rstrip("/"))
    abs_stage = os.path.abspath(stage_index_path.rstrip("/"))
    for extension_dir in ("train_tables", "pred_tables"):
        expected_prefix = os.path.join(abs_parent, extension_dir) + os.sep
        if not abs_stage.startswith(expected_prefix):
            continue
        extension_id = abs_stage[len(expected_prefix):].split(os.sep, 1)[0]
        if not extension_id:
            raise DiskgraphConfigError(
                "could not extract extension id from stage index_path "
                f"{stage_index_path}"
            )
        return f"{extension_dir}/{extension_id}"
    raise DiskgraphConfigError(
        "stage index_path must live under <graph_index_path>/train_tables/ "
        "or <graph_index_path>/pred_tables/: "
        f"parent={abs_parent}, stage={abs_stage}"
    )


def _sampling_plan_json(sampling_plan: Any) -> str:
    if isinstance(sampling_plan, str):
        return sampling_plan
    return json.dumps(sampling_plan or {"layers": []}, sort_keys=True)


def _weighted_seed_sampling_json(
    ref: Mapping[str, Any],
    stage: str,
    weighted_seed_sampling: Any,
) -> Optional[str]:
    if weighted_seed_sampling is None:
        return None
    if not isinstance(weighted_seed_sampling, Mapping):
        raise DiskgraphConfigError("weighted_seed_sampling must be a mapping")
    root = str(ref["index_path"])

    def _path(value: Any, *, context: str) -> str:
        if not value:
            raise DiskgraphConfigError(
                f"weighted_seed_sampling for stage {stage!r} missing {context}"
            )
        return os.path.join(root, str(value))

    def _weighted_bucket(
        index: Mapping[str, Any],
        key: str,
        *,
        multiplier: float = 1.0,
    ) -> dict[str, Any]:
        bucket = index.get(key)
        if not isinstance(bucket, Mapping):
            raise DiskgraphConfigError(
                f"sample_weight_index for stage {stage!r} missing {key!r} bucket"
            )
        total_weight = float(bucket.get("total_weight", 0.0))
        return {
            "name": key,
            "node_ids_path": _path(bucket.get("node_ids"),
                                   context=f"{key}.node_ids"),
            "count": int(bucket.get("count", 0)),
            "mass": total_weight * multiplier,
            "cdf_path": _path(bucket.get("cdf"), context=f"{key}.cdf"),
            "total_weight": total_weight,
        }

    def _class_bucket(
        binary_index: Mapping[str, Any],
        key: str,
        *,
        class_weight: float,
    ) -> dict[str, Any]:
        node_key = f"{key}_node_ids"
        count_key = f"{key}_count"
        count = int(binary_index.get(count_key, 0))
        return {
            "name": key,
            "node_ids_path": _path(binary_index.get(node_key),
                                   context=node_key),
            "count": count,
            "mass": count * class_weight,
        }

    use_sample_weights = bool(
        weighted_seed_sampling.get("use_sample_weights", False))
    has_class_weights = ("false_weight" in weighted_seed_sampling
                         or "true_weight" in weighted_seed_sampling)
    false_weight = float(weighted_seed_sampling.get("false_weight", 1.0))
    true_weight = float(weighted_seed_sampling.get("true_weight", 1.0))

    buckets: list[dict[str, Any]]
    if use_sample_weights:
        sample_index = ref.get("sample_weight_index")
        if not isinstance(sample_index, Mapping):
            raise DiskgraphConfigError(
                f"weighted_seed_sampling requires sample_weight_index for stage {stage!r}"
            )
        if has_class_weights:
            buckets = [
                _weighted_bucket(sample_index, "false_bucket",
                                 multiplier=false_weight),
                _weighted_bucket(sample_index, "true_bucket",
                                 multiplier=true_weight),
            ]
        else:
            buckets = [_weighted_bucket(sample_index, "all")]
    else:
        binary_index = ref.get("binary_label_index")
        if not isinstance(binary_index, Mapping):
            raise DiskgraphConfigError(
                f"weighted_seed_sampling requires binary_label_index for stage {stage!r}"
            )
        buckets = [
            _class_bucket(binary_index, "false",
                          class_weight=false_weight),
            _class_bucket(binary_index, "true", class_weight=true_weight),
        ]

    payload = {
        "buckets": buckets,
        "num_samples": int(weighted_seed_sampling["num_samples"]),
    }
    return json.dumps(payload, sort_keys=True)


_UNSET: Any = object()


def _validate_worker_iterator_inputs(
    *,
    batch_size: int,
    num_workers: int,
    worker_id: int,
    shuffle: bool,
    rnd_seed: Optional[int],
    epoch: int,
) -> None:
    if batch_size <= 0:
        raise DiskgraphConfigError("batch_size must be >= 1")
    if num_workers <= 0:
        raise DiskgraphConfigError("num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise DiskgraphConfigError(
            f"worker_id {worker_id} out of range for num_workers {num_workers}"
        )
    if shuffle and rnd_seed is None:
        raise DiskgraphConfigError("rnd_seed is required when shuffle=True")
    if epoch < 0:
        raise DiskgraphConfigError("epoch must be >= 0")


def task_stage_size(dataset_ref: TaskDatasetRef, stage: str) -> int:
    """Return the number of TT rows for a stage from the runtime manifest."""
    _, split_info = _manifest_view(dataset_ref)
    return int((split_info.get("stage_counts") or {}).get(stage, 0))


class RuntimeIndex:
    """An opened graph index, reusable across epochs and shards.

    Opening an index is expensive -- it mmaps every shard, builds every
    row-pack and sort-key decoder, and spawns a rayon pool (~1.6 s on a
    large graph). Selecting *which rows* to iterate is not. They used to
    happen together in :func:`make_runtime_batch_iterator`, so a training
    run paid the open once per epoch in every DataLoader worker, and again
    on every intra-epoch replay.

    Hold one of these for as long as the process serves the same dataset,
    and call :meth:`iter` per epoch (or per chunk, when work is handed out
    dynamically). Measured on a 343 GB graph: the first construction still
    costs ~1.55 s, every one after costs ~0.1 ms.

    ``rnd_seed`` and ``num_threads`` are fixed at open time because the
    engine is built from them. ``num_threads`` especially is not a
    performance-only knob -- the engine's parallel neighbour sampling
    produces different (individually reproducible) batches for different
    thread counts, so two iterators that disagree on it are not sampling
    the same graph.
    """

    def __init__(
        self,
        dataset_ref: TaskDatasetRef,
        rnd_seed: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        # Resolved lazily, on first iter(). Reading the manifest in the
        # constructor would put file I/O ahead of argument validation, and a
        # caller who passes an out-of-range worker_id should get a config
        # error rather than a FileNotFoundError for a path that was never the
        # problem. test_task_dataset_mock pins that ordering.
        self._dataset_ref = dataset_ref
        self._resolved: Any = None
        self._rnd_seed = int(rnd_seed) if rnd_seed is not None else None
        self._num_threads = int(num_threads) if num_threads is not None else None
        # Stages normally share one train-table extension, but nothing
        # guarantees it, and a Rust RuntimeIndex is bound to a single one.
        # Keyed here so a dataset whose stages differ still works, and still
        # opens each extension only once.
        self._indexes: dict = {}

    def _resolve(self) -> None:
        if self._resolved is not None:
            return
        manifest, split_info = _manifest_view(self._dataset_ref)
        self._manifest = manifest
        self._split_info = split_info
        self._tables = _stage_tables(manifest)
        self._parent_root = _graph_index_path(manifest)
        self._resolved = True

    def _index_for(self, extension_ref: str):
        idx = self._indexes.get(extension_ref)
        if idx is None:
            from ._c import RuntimeIndex as _RuntimeIndex
            idx = _RuntimeIndex(
                self._parent_root,
                extension_ref,
                self._rnd_seed,
                self._num_threads,
            )
            self._indexes[extension_ref] = idx
        return idx

    def stage_size(self, stage: str) -> int:
        self._resolve()
        return int((self._split_info.get("stage_counts") or {}).get(stage, 0))

    def iter(
        self,
        stage: str,
        sampling_plan: Any,
        num_workers: int,
        worker_id: int,
        shuffle: bool,
        epoch: int,
        batch_size: int,
        rnd_seed: Optional[int] = _UNSET,
        drop_last: bool = False,
        fused: Optional[bool] = None,
        swap_edge_direction: bool = True,
        neg_rhs_node_type: Optional[str] = None,
        neg_amount_m: Optional[int] = None,
        neg_rnd_seed: Optional[int] = None,
        replace: bool = False,
        disjoint: bool = True,
        adaptive: bool = False,
        entity_node_type: Optional[str] = None,
        target_node_type: Optional[str] = None,
        rhs_seed_node_type: Optional[str] = None,
        rhs_seed_time: Optional[int] = None,
        group_by_timestamp: bool = False,
        weighted_seed_sampling: Any = None,
        selected_stage_node_ids: Any = None,
        pseudo_seed_time: Optional[int] = None,
        disable_pseudo_time: bool = False,
        projections: dict[str, list[str]] | None = None,
    ) -> tuple[Any, int]:
        """Iterate one (stage, epoch, shard) of this index.

        Arguments match :func:`make_runtime_batch_iterator` except that
        ``dataset_ref`` and ``num_threads`` are fixed by the index. ``rnd_seed``
        defaults to the one the index was opened with, which is what callers
        almost always want; pass it explicitly only to vary the shuffle
        independently of the engine seed. Static LP cutoffs default to the
        manifest stage. Prediction can set ``disable_pseudo_time=True``.

        ``projections`` maps a node type to the columns the caller will
        actually consume, and is resolved once when the iterator is built:

        * ``None`` (the default) -- select all, everywhere. Pre-pushdown
          behavior.
        * a node type ABSENT from the mapping -- select all for that type.
        * ``{"users": []}`` -- select NOTHING for ``users``. The type is skipped
          outright: no rowpack decode, and no ``raw_features`` block for it in
          the emitted batch. Its EDGES are untouched, so it is still traversed
          normally; this declines a type's features, never its topology.
        * ``{"users": ["a", "b"]}`` -- materialize exactly those, intersected
          with what the snapshot actually stores.

        Explicit ``fused=True`` ignores non-empty projections with a warning.
        With ``fused=None``, non-empty projections select non-fused sampling.

        Was named ``feature_columns``, under which ``[]`` meant select-all. The
        rename is deliberate and unaliased: the two signatures are otherwise
        identical, so the parameter NAME is the only thing that makes a new
        caller against an old wheel fail loudly instead of silently reverting
        to select-all.
        """
        seed = self._rnd_seed if rnd_seed is _UNSET else rnd_seed
        _validate_worker_iterator_inputs(
            batch_size=batch_size, num_workers=num_workers, worker_id=worker_id,
            shuffle=shuffle, rnd_seed=seed, epoch=epoch,
        )
        self._resolve()
        ref = self._tables.get(stage)
        if ref is None:
            raise DiskgraphConfigError(
                f"stage {stage!r} not present in manifest.stages")
        if self.stage_size(stage) == 0:
            raise DiskgraphConfigError(
                f"stage {stage!r} not available on this dataset")

        extension_ref = _extension_ref_from_stage_path(
            self._parent_root, ref["index_path"])
        iterator = self._index_for(extension_ref).iter(
            ref["table_name"],
            _sampling_plan_json(sampling_plan),
            int(num_workers),
            int(worker_id),
            bool(shuffle),
            int(seed) if seed is not None else None,
            int(epoch),
            int(batch_size),
            bool(drop_last),
            fused if fused is None else bool(fused),
            bool(swap_edge_direction),
            neg_rhs_node_type,
            int(neg_amount_m) if neg_amount_m is not None else None,
            int(neg_rnd_seed) if neg_rnd_seed is not None else None,
            bool(replace),
            bool(disjoint),
            bool(adaptive),
            entity_node_type,
            target_node_type,
            rhs_seed_node_type,
            int(rhs_seed_time) if rhs_seed_time is not None else None,
            bool(group_by_timestamp),
            _weighted_seed_sampling_json(ref, stage, weighted_seed_sampling),
            selected_stage_node_ids,
            _use_seed_timestamps(self._manifest),
            int(pseudo_seed_time) if pseudo_seed_time is not None else None,
            bool(disable_pseudo_time),
            projections,
        )
        return iterator, iterator.num_samples()


