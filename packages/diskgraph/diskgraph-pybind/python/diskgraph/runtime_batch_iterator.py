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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._c import RuntimeBatchIterator

from ._exceptions import DiskgraphConfigError
from .sampling_options import InheritEngineSeed, SamplingOptions
from .sampling_plan import SamplingPlan

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


def _validate_worker_iterator_inputs(
    *,
    batch_size: int,
    num_workers: int,
    worker_id: int,
    shuffle: bool,
    rnd_seed: int | None,
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
        rnd_seed: int | None = None,
        num_threads: int | None = None,
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

    def sampler(self, stage: str):
        """Share the stage's combined native reader without opening another index."""
        self._resolve()
        ref = self._tables[stage]
        extension_ref = _extension_ref_from_stage_path(self._parent_root, ref["index_path"])
        return self._index_for(extension_ref).sampler()

    def stage_size(self, stage: str) -> int:
        self._resolve()
        return int((self._split_info.get("stage_counts") or {}).get(stage, 0))

    def iter(self, options: SamplingOptions) -> tuple[RuntimeBatchIterator, int]:
        """Iterate one (stage, epoch, shard) of this index.

        ``dataset_ref`` and ``num_threads`` are fixed by the index. ``rnd_seed``
        defaults to the one the index was opened with, which is what callers
        almost always want; pass it explicitly only to vary the shuffle
        independently of the engine seed. ``sampling_plan`` accepts a
        :class:`diskgraph.SamplingPlan`; other types raise ``TypeError``.
        Static LP cutoffs default to the manifest stage.
        Prediction can set ``disable_pseudo_time=True``.

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

        Args:
            options: Typed sampling and seed scheduling configuration.

        Returns:
            Native iterator and worker-local sample or timestamp-bucket count.

        Raises:
            TypeError: If options or the traversal plan has the wrong type.
            DiskgraphConfigError: If the stage or worker configuration is invalid.
        """
        if not isinstance(options, SamplingOptions):
            raise TypeError("options must be a diskgraph.SamplingOptions")
        if not isinstance(options.sampling_plan, SamplingPlan):
            raise TypeError("sampling_plan must be a diskgraph.SamplingPlan")
        seed = (
            self._rnd_seed
            if options.rnd_seed is InheritEngineSeed.DEFAULT
            else options.rnd_seed
        )
        _validate_worker_iterator_inputs(
            batch_size=options.batch_size,
            num_workers=options.num_workers,
            worker_id=options.worker_id,
            shuffle=options.shuffle,
            rnd_seed=seed,
            epoch=options.epoch,
        )
        self._resolve()
        ref = self._tables.get(options.stage)
        if ref is None:
            raise DiskgraphConfigError(
                f"stage {options.stage!r} not present in manifest.stages"
            )
        if self.stage_size(options.stage) == 0:
            raise DiskgraphConfigError(
                f"stage {options.stage!r} not available on this dataset"
            )

        extension_ref = _extension_ref_from_stage_path(
            self._parent_root, ref["index_path"]
        )
        iterator = self._index_for(extension_ref).iter(
            ref["table_name"],
            options.sampling_plan,
            int(options.num_workers),
            int(options.worker_id),
            bool(options.shuffle),
            int(seed) if seed is not None else None,
            int(options.epoch),
            int(options.batch_size),
            bool(options.drop_last),
            options.fused if options.fused is None else bool(options.fused),
            bool(options.swap_edge_direction),
            options.neg_rhs_node_type,
            int(options.neg_amount_m) if options.neg_amount_m is not None else None,
            int(options.neg_rnd_seed) if options.neg_rnd_seed is not None else None,
            bool(options.replace),
            bool(options.disjoint),
            bool(options.adaptive),
            options.entity_node_type,
            options.target_node_type,
            options.rhs_seed_node_type,
            int(options.rhs_seed_time) if options.rhs_seed_time is not None else None,
            bool(options.group_by_timestamp),
            options.weighted_seed_sampling,
            options.selected_stage_node_ids,
            _use_seed_timestamps(self._manifest),
            int(options.pseudo_seed_time)
            if options.pseudo_seed_time is not None
            else None,
            bool(options.disable_pseudo_time),
            options.projections,
        )
        return iterator, iterator.num_samples()


