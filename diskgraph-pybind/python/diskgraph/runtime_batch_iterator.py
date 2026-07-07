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


def _stage_tables(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw = manifest.get("stages") or {}
    if not isinstance(raw, Mapping):
        raise DiskgraphConfigError("manifest.stages must be an object")

    out: dict[str, dict[str, str]] = {}
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


def _sampling_plan_json(sampling_plan: Any) -> str:
    if isinstance(sampling_plan, str):
        return sampling_plan
    return json.dumps(sampling_plan or {"layers": []}, sort_keys=True)


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


def make_runtime_batch_iterator(
    dataset_ref: TaskDatasetRef,
    stage: str,
    sampling_plan: Any,
    num_workers: int,
    worker_id: int,
    shuffle: bool,
    rnd_seed: Optional[int],
    epoch: int,
    batch_size: int,
    drop_last: bool = False,
    num_threads: Optional[int] = None,
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
    selected_stage_node_ids: Any = None,
) -> tuple[Any, int]:
    """Create a Rust-backed iterator over a stage table in the merged catalog.

    Args:
        dataset_ref: Runtime task-dataset manifest path.
        stage: ``train`` / ``val`` / ``test``.
        sampling_plan: Existing DiskGraph ``SamplingPlan`` as JSON or dict.
        num_workers: Total independent consumers across ranks/workers.
        worker_id: Current global worker id.
        shuffle: Whether to shuffle TT row offsets before sharding.
        rnd_seed: Required when ``shuffle=True``.
        epoch: Trainer epoch for per-pass deterministic shuffle variation.
        batch_size: Number of TT rows per emitted sampled batch.
        drop_last: Whether to drop this worker's final partial batch.
        num_threads: rayon pool size per worker; ``None`` (default) ⇒ engine
            default (1).
        fused: ``None`` (default) lets DG pick via the
            ``engine::choose_fused_mode`` heuristic over (reader, plan,
            batch_size). Explicit ``True`` / ``False`` overrides the
            heuristic for callers that have profiled their workload.
        swap_edge_direction: PyG CSC convention. Defaults to ``True``.
        neg_rhs_node_type: RHS (target) node type to draw engine-side global
            negatives from (LP training). ``None`` (default) disables negative
            sampling. When set, ``neg_amount_m`` is required.
        neg_amount_m: Negatives drawn per positive (M). Required when
            ``neg_rhs_node_type`` is set; the rhs node set becomes
            ``[B positives, B*M negatives]`` anchor-major.
        neg_rnd_seed: RNG base for the negative draw, independent of the
            shuffle seed. ``None`` (default) ⇒ ``thread_rng`` (random negatives,
            the training/prod default — fresh each epoch); an int ⇒ deterministic
            replay-safe draw (tests).
        replace: Sample neighbors with replacement.
        disjoint: Keep per-seed sampled neighborhoods disjoint.
        adaptive: Enable DiskGraph adaptive sampling.
        entity_node_type: FK-free LP resolution (§S5) — when set, the seed
            set is read from the ``entity_node_id`` rowpack column instead of
            walking ``seed.edges.bin``. ``None`` keeps the legacy FK path.
        target_node_type: FK-free LP resolution (§S5) — when set, the LP
            positive LABELS are read from the ``target_node_ids`` rowpack
            column and emitted as ragged ``lp_pos_*`` fields (labels only,
            never rooted). ``None`` keeps the legacy raw-target path.
        rhs_seed_node_type: NodeRange seed mode (two-tower MIPS RHS pass) —
            when set, the iterator seeds the FULL ``[0, num_rhs)`` catalog of
            this node type directly at the fixed ``rhs_seed_time`` (required),
            bypassing stage-table seeding / entity-target resolution /
            negatives. Across all batches the seeds cover ``[0, num_rhs)``
            exactly once, in order. ``None`` keeps the stage-table path.
        rhs_seed_time: Fixed anchor timestamp applied to every NodeRange seed.
            Required when ``rhs_seed_node_type`` is set.
        group_by_timestamp: TT-NG temporal-LP timestamp grouping (LOAD side).
            When ``True``, every emitted batch holds node-ids of a single
            timestamp, scheduled from the per-stage ``bucket_index.bin`` the
            ingest side recorded; FAILS LOUD if that artifact is missing (no
            silent fallback). ``False`` (default) keeps the flat worker-sharded
            path byte-for-byte. Incompatible with ``rhs_seed_node_type``.
        selected_stage_node_ids: Optional explicit stage-table node ids to
            emit, in caller order. These are ``task.tt_stage_node_ids`` values,
            not batch-local ``task.input_ids`` or business primary keys. When
            ``num_workers > 1``, each worker emits the ids whose caller-order
            position satisfies ``position % num_workers == worker_id``.

    Returns:
        ``(RuntimeBatchIterator, num_worker_samples)``.
    """
    if neg_rhs_node_type is not None and neg_amount_m is None:
        raise DiskgraphConfigError(
            "neg_amount_m is required when neg_rhs_node_type is set"
        )
    if rhs_seed_node_type is not None and rhs_seed_time is None:
        raise DiskgraphConfigError(
            "rhs_seed_time is required when rhs_seed_node_type is set (NodeRange mode)"
        )
    _validate_worker_iterator_inputs(
        batch_size=batch_size,
        num_workers=num_workers,
        worker_id=worker_id,
        shuffle=shuffle,
        rnd_seed=rnd_seed,
        epoch=epoch,
    )

    manifest, split_info = _manifest_view(dataset_ref)
    tables = _stage_tables(manifest)
    ref = tables.get(stage)
    if ref is None:
        raise DiskgraphConfigError(f"stage {stage!r} not present in manifest.stages")

    stage_count = int((split_info.get("stage_counts") or {}).get(stage, 0))
    if stage_count == 0:
        raise DiskgraphConfigError(f"stage {stage!r} not available on this dataset")

    parent_root = _graph_index_path(manifest)
    extension_ref = _extension_ref_from_stage_path(parent_root, ref["index_path"])

    from ._c import RuntimeBatchIterator

    iterator = RuntimeBatchIterator(
        parent_root,
        extension_ref,
        ref["table_name"],
        _sampling_plan_json(sampling_plan),
        int(num_workers),
        int(worker_id),
        bool(shuffle),
        int(rnd_seed) if rnd_seed is not None else None,
        int(epoch),
        int(batch_size),
        bool(drop_last),
        int(num_threads) if num_threads is not None else None,  # num_threads — engine default when None
        # fused: None means "DG, decide via heuristic"; bool forces.
        fused if fused is None else bool(fused),
        bool(swap_edge_direction),
        neg_rhs_node_type,
        int(neg_amount_m) if neg_amount_m is not None else None,
        int(neg_rnd_seed) if neg_rnd_seed is not None else None,
        bool(replace),
        bool(disjoint),
        bool(adaptive),
        # FK-free LP resolution (design §S5): read the seed set from the
        # ``entity_node_id`` rowpack column (no ``seed.edges.bin`` walk) and
        # emit ragged ``lp_pos_*`` labels from ``target_node_ids``. Both
        # ``None`` preserves the legacy FK-edge / raw-target path.
        entity_node_type,
        target_node_type,
        # NodeRange seed mode (two-tower MIPS RHS pass): seed the full
        # ``[0, num_rhs)`` catalog of ``rhs_seed_node_type`` directly at the
        # fixed ``rhs_seed_time``, bypassing stage-table seeding / resolution /
        # negatives. Both ``None`` preserves the stage-table seeding path.
        rhs_seed_node_type,
        int(rhs_seed_time) if rhs_seed_time is not None else None,
        bool(group_by_timestamp),  # see docstring: single-ts grouped batches.
        selected_stage_node_ids,
    )
    return iterator, iterator.num_samples()


__all__ = [
    "TaskDatasetRef",
    "make_runtime_batch_iterator",
    "task_stage_size",
]
