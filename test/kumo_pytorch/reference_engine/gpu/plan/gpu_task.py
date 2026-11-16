"""`GpuTask`: upstream Kumo PyTorch's `Task` on the GPU: its fitted state and its batches."""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:

    from kumo.pytorch.anchors import SplitSelection
    from kumo.pytorch.contract.sample import SampleRequest
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.plan.gpu_plan import GpuPlan
    from test.kumo_pytorch.reference_engine.gpu.plan.state import Fitted
from test.kumo_pytorch.reference_engine.gpu.batching.admission import admitted
from test.kumo_pytorch.reference_engine.gpu.plan.groups import _Sizer, _groups, _in_groups, _measured

__all__ = ['GpuTask']


def _uncounted(counted: dict[str, Any], parents: list[str], rows: Any) -> Any:
    """The distinct anchor rows among `rows` (CuPy) not counted yet."""

    import cupy

    return cupy.unique(rows[counted[parents[0]][rows] < 0]) if rows.size else rows


def _chunked_counts(contract: DataLoaderSpec, sampler: Any, index: Any, anchors: GpuAnchors, rows: Any,
                    parents: list[str], budget: int) -> Iterator[tuple[Any, dict[str, Any]]]:
    """(chunk of `rows`, each parent's event counts there) for the anchors at `rows`, a group of anchors at a
    time within `budget` bytes (`test.kumo_pytorch.reference_engine.gpu.plan.groups._Sizer`): a selection holds every event it counts.
    An anchor whose entity the table lacks counts none (building its batch refuses it)."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.batching.admission import event_counts

    sizer, start = _Sizer(budget), 0
    while start < rows.size:
        chunk = rows[start:start + sizer.size()]
        with sizer.measured(int(chunk.size)):
            entity, found = index.find(anchors, chunk)
            counts = event_counts(contract, sampler, entity[found], anchors.cutoffs[chunk][found], parents)
            full = {}
            for parent in parents:
                full[parent] = cupy.zeros(chunk.size, dtype=cupy.int64)
                full[parent][found] = counts[parent]
        yield chunk, full
        start += int(chunk.size)


class GpuTask:
    """Anchors bound to a plan, with fitted encoders and ID spaces: upstream's `Task`, on the GPU."""

    def __init__(self, plan: GpuPlan, anchors: GpuAnchors, fit_selection: SplitSelection | None,
                 encoders: dict[str, Any], composites: dict[str, Any], spaces: dict[str, Any], targets: Any,
                 written: dict[str, Any], manifest: dict[str, Any], *, source: Any,
                 reused_fit: Mapping[str, Any], has_targets: bool = True, inference: bool,
                 field_maps: dict[str, dict[tuple[str, ...], int]] | None = None, statistics: Any = None) -> None:
        self.plan = plan
        self.contract = plan.contract
        self.anchors = anchors
        self.fit_selection = fit_selection
        self._statistics = statistics
        self._encoders = encoders  # encoder identity to its fitted `test.kumo_pytorch.reference_engine.gpu.encoding.encoders.GpuEncoder`
        # as upstream's `Task.encoders` names them: a shared encoder by its name, the others by identity
        self.encoders = {json.loads(key)[1] if json.loads(key)[0] == "shared" else key: encoder
                         for key, encoder in encoders.items()}
        self.spaces = spaces  # ID space name to its final numbering (upstream's `FittedSpace`)
        # Composite input name to its outer IDs (`test.kumo_pytorch.reference_engine.gpu.encoding.composite.CompositeVocabulary`; `.entries()` spells
        # them as a fitted contract's `Composite.vocabulary`)
        self.composites = composites
        # fitted target state (`test.kumo_pytorch.reference_engine.gpu.encoding.targets.TargetState`): thresholds of derived labels, matching groups
        self.targets = targets
        self.inference = inference
        # the same state as upstream writes it (`test.kumo_pytorch.reference_engine.gpu.plan.state`), and its manifest, which every batch names;
        # upstream's objects of it (`fitted_state`, `bound_contract()`, `fingerprints`) are made when first asked for
        self._written = written
        self._manifest = manifest
        self._source = source  # the anchors as given (requests or an anchor table), which `fingerprints` names
        self._reused_fit = dict(reused_fit)  # each reused half's fit record (`fitted.fit`); the rest fitted here
        self._has_targets = has_targets  # false for inference from an input-only state: it stays input-only
        self.field_maps = field_maps or {}  # each field-name dictionary: field identity to its ID
        # each counted EventSource input's events per anchor-table row (-1: not counted yet), for the batch
        # memory check (`test.kumo_pytorch.reference_engine.gpu.batching.admission`), counted once per task
        self._event_counts_cache: dict[str, Any] = {}
        self.output_schema: Mapping[str, Any] = {
            "inputs": {name: value["summary"] for name, value in manifest["inputs"].items()},
            "targets": {} if inference else dict(manifest["targets"]),
        }

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Upstream's manifest of the fitted state: inputs' and targets' encodings, ID spaces, the length
        summary, and the fitted state's and artifact's fingerprints."""

        return self._manifest

    @property
    def fitted_state(self) -> Any:
        """Upstream's `FittedState` of this task: what ``materialize(fitted=...)`` reuses and an artifact holds,
        with where it was fitted (``fit``)."""

        return self._upstream.state

    def bound_contract(self) -> DataLoaderSpec:
        """The contract with every fitted input encoding pinned into its declarations (upstream's)."""

        return self._upstream.bound_contract

    @cached_property
    def _upstream(self) -> Fitted:
        """The fitted state rebuilt by upstream from its written form (`test.kumo_pytorch.reference_engine.gpu.plan.state.as_upstream`)."""

        from kumo.pytorch.anchors import describe_selection

        from test.kumo_pytorch.reference_engine.gpu.plan.state import as_upstream

        here = None if self.fit_selection is None else {
            "contract": self.contract.fingerprint, "dataset": self.fingerprints["dataset"],
            "anchors": self.fingerprints["anchors"], "fit_split": describe_selection(self.fit_selection)}
        fit = {half: self._reused_fit[half] if half in self._reused_fit else here for half in ("inputs", "targets")}
        if self._statistics is not None and "inputs" not in self._reused_fit:
            fit["inputs"] = {**here, "statistics": self._statistics}
        return as_upstream(self.contract, self._written, fit,
                           has_targets=self._has_targets)

    @cached_property
    def fingerprints(self) -> Mapping[str, str]:
        """Upstream's identity of this task (``Task.fingerprints``): its contract's, inputs', dataset's, anchors',
        fitted state's halves', output schema's, and of all of them."""

        from test.kumo_pytorch.reference_engine.gpu.plan.state import fingerprint

        written = self._written
        result = {
            "contract": self.contract.fingerprint,
            "inputs": self.contract.input_fingerprint,
            "dataset": fingerprint(self.plan.dataset),
            "anchors": fingerprint(self._source),
            "input_state": fingerprint({key: value for key, value in written.items() if key != "targets"}),
            "target_state": fingerprint(written.get("targets", {})),
            "output_schema": fingerprint(self.output_schema),
        }
        result["task"] = fingerprint(result)
        return result

    def to_artifact(self) -> Any:
        """Upstream's `Artifact` of this task: its contract, bound contract, fitted state, fingerprints, output
        schema and manifest, reading and writing nothing. ``Artifact.save`` writes it; ``Artifact.load`` reads
        it back equal to this, in any process."""

        from kumo.pytorch.artifact import Artifact

        if not self.fitted_state.has_targets:
            raise ValueError(
                "this inference task holds only the input half of its fitted state, and an "
                "artifact holds the complete state; save the training task's artifact, or "
                "reuse Artifact.load(directory) of the one it was trained with"
            )
        return Artifact(contract=self.contract, bound_contract=self.bound_contract(), fitted_state=self.fitted_state,
                        fingerprints=dict(self.fingerprints),
                        output_schema=json.loads(json.dumps(self.output_schema)),
                        manifest=json.loads(json.dumps(self._manifest)))

    def count(self, split: str | Collection[str | None] | Any | None = None) -> int:
        """Anchors in `split` (every anchor when omitted), as upstream's `Task.count`."""

        from kumo.pytorch.anchors import ALL_ANCHORS, split_selection

        selection = split_selection(split)
        return self.anchors.count(ALL_ANCHORS if selection is None else selection)

    def batches(self, requests: Sequence[SampleRequest] | None = None, *,
                split: str | Collection[str | None] | Any | None = None, batch_size: int | None = None,
                epoch: int = 0, seed: int = 0, shuffle: bool = False, rank: int = 0, world_size: int = 1,
                drop_last: bool | None = None, chunk_batches: int = 8,
                output_format: str = "inspectable") -> Any:
        """Batches over this task's anchors (those of `split`), or over `requests` built in code.

        Upstream ``Task.batches``'s arguments and rules: the order is
        `shard_positions`'; each rank gets ``ceil(count / world_size)`` anchors, a
        short rank topped up by repeating its first anchor, marked in
        ``metadata["is_padding"]``; `drop_last` drops the rank's last partial
        batch. Every anchor of the call is worked out in one pass, except
        graph neighborhoods: they are sampled `chunk_batches` batches at a time.
        """

        from kumo.pytorch.anchors import ALL_ANCHORS, split_selection

        from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors

        contract = self.plan.contract
        policy = contract.batch.batching
        if requests is not None and split is not None:
            raise ValueError("split selects materialized anchors; requests carry their own")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must identify one member of a positive world size")
        if world_size > 1 and policy.max_tokens is not None:  # upstream's refusal, word for word
            raise ValueError(
                "Batching.max_tokens with world_size > 1 is not supported yet: padding "
                "equalizes anchors per rank, but a token budget makes the number of "
                "batches depend on lengths, so ranks can take different numbers of steps "
                "and hang synchronous data-parallel training. Use max_samples only for "
                "distributed runs until batches are planned globally and assigned whole "
                "to ranks."
            )
        if drop_last is None and self.inference:
            drop_last = False  # inference scores every anchor, whatever the training policy
        if chunk_batches <= 0:
            raise ValueError("chunk_batches must be positive")
        if output_format not in {"inspectable", "model"}:
            raise ValueError("output_format must be 'inspectable' or 'model'")
        maximum = batch_size if batch_size is not None else policy.max_samples
        if requests is None:
            anchors, chosen = self.anchors, split_selection(split)
            selection = ALL_ANCHORS if chosen is None else chosen
            anchors.count(selection)  # an unknown split is refused now, not mid-iteration
            if maximum is None:
                raise ValueError("reading materialized anchors requires a batch size: pass batch_size "
                                 "or set Batching.max_samples")
        else:
            anchors, selection = GpuAnchors.from_requests(requests), ALL_ANCHORS
            if maximum is None and policy.max_tokens is None:
                raise ValueError("set max_samples/max_tokens or pass batch_size")
        if maximum is not None and maximum <= 0:
            raise ValueError("batch_size must be positive")
        positions, padding = self._positions(anchors, selection, seed=seed, epoch=epoch, shuffle=shuffle, rank=rank,
                                             world_size=world_size)
        rows = anchors.rows(selection)[positions]
        drop_last = policy.drop_last if drop_last is None else drop_last
        options = {"encoders": self._encoders, "composites": self.composites, "spaces": self.spaces,
                   "target_state": self.targets, "anchors": anchors, "maximum": maximum,
                   "model": output_format == "model", "fingerprint": contract.fingerprint,
                   "artifact": self._manifest["artifact_fingerprint"], "seed": seed, "epoch": epoch,
                   "chunk": chunk_batches, "field_maps": self.field_maps}
        if self.inference:  # no target computed or emitted, so no row a target reads, no masking and no shift
            from kumo.pytorch.preparation import _inference_view

            contract = _inference_view(contract)
        from kumo.pytorch.anchors import describe_selection
        from kumo.pytorch.assembly import _fingerprint
        from kumo.pytorch.task import TaskBatches

        config = {"split": describe_selection(selection) if requests is None else None,
                  "requests": None if requests is None else _fingerprint(tuple(requests)),
                  "batch_size": maximum, "epoch": epoch, "seed": seed, "shuffle": shuffle,
                  "rank": rank, "world_size": world_size, "drop_last": drop_last,
                  "chunk_batches": chunk_batches, "output_format": output_format,
                  "gpu_route": type(self.plan).__name__,
                  "gpu_partitions": getattr(self.plan, "num_partitions", 1)}
        return TaskBatches(
            lambda: self._emit(contract, rows, padding, options, drop_last=drop_last, shuffle=shuffle,
                               from_requests=requests is not None),
            task_fingerprint=self.fingerprints["task"], config=config)

    def _positions(self, anchors: GpuAnchors, selection: Any, *, seed: int, epoch: int, shuffle: bool, rank: int,
                   world_size: int) -> tuple[Any, Any]:
        """This rank's anchors, as positions among `selection`'s, in order, and which are padding (CuPy):
        `shard_positions`' order dealt to ranks, each topped up to ``ceil(count / world_size)`` by repeating its
        first anchor (the first one of all for a rank with none)."""

        import cupy

        count = anchors.count(selection)
        positions = anchors.positions(selection, seed=seed, epoch=epoch, shuffle=shuffle, rank=rank,
                                      world_size=world_size)
        target = -(-count // world_size) if count else 0
        short = target - int(positions.size)
        padding = cupy.zeros(target, dtype=bool)
        if short:
            first = positions[:1] if positions.size else cupy.zeros(1, dtype=cupy.int64)
            positions = cupy.concatenate([positions, cupy.repeat(first, short)])
            padding[target - short:] = True
        return positions, padding

    def _emit(self, contract: DataLoaderSpec, rows: Any, padding: Any, options: dict[str, Any], *, drop_last: bool,
              shuffle: bool, from_requests: bool) -> Iterator[Any]:
        """The batches over the anchors at `rows`, in that order (`batches` chose them): `options` are
        `test.kumo_pytorch.reference_engine.gpu.batching.build.build_batches`'s for this call."""

        from test.kumo_pytorch.reference_engine.gpu.batching.build import build_batches

        from test.kumo_pytorch.reference_engine.gpu.batching.compose import BatchPlan

        build = partial(build_batches, sampler=self.plan.sampler, index=self.plan.index, drop_last=drop_last,
                        **options)
        if contract.batch.batching.max_tokens is None and not contract.batch.batching.bucket_by_length:
            cut = BatchPlan.fixed(int(rows.size), options["maximum"], drop_last)
            return admitted(contract, lambda: _in_groups(partial(build, contract, targets=not self.inference), rows,
                                                         padding, cut, self.plan.build_bytes),
                            lambda: self._admission(contract, options["anchors"], rows, cut.bounds[:cut.emit]))
        return admitted(contract, lambda: self._planned(build, contract, rows, padding,
                                                        _groups(int(rows.size), options, from_requests),
                                                        options["maximum"], drop_last), lambda: None)

    def _admission(self, contract: DataLoaderSpec, anchors: GpuAnchors, rows: Any, bounds: Sequence[tuple[int, int]],
                   where: Any = None) -> Any:
        """What the batch memory check needs for a call of fixed-size batches (`bounds`) over the anchors at
        `rows`: each counted input's width per batch and the GPU memory free for a batch, beside the data (and
        the rest of the model's peak, given `prepare(model_bytes=)`). None when no input is counted."""

        from test.kumo_pytorch.reference_engine.gpu.batching.admission import Admission, padded_event_inputs
        from test.kumo_pytorch.reference_engine.gpu.data.memory import free_bytes

        inputs = padded_event_inputs(contract)
        if not inputs or not bounds:
            return None
        parents = sorted(set(inputs.values()))
        counts, resident = self._event_counts(contract, anchors, rows, parents, where)
        return Admission.of(inputs, counts, bounds, free_bytes() - resident - self._model_rest(), resident)

    def _event_counts(self, contract: DataLoaderSpec, anchors: GpuAnchors, rows: Any, parents: list[str],
                      where: Any = None) -> tuple[dict[str, Any], int]:
        """Each anchor's events at `rows` per parent input, counted on the resident tables (and kept for this
        task's anchor table), and the bytes data loaded for the call would keep beside a batch: none here."""

        counted = self._counted(anchors, parents)
        todo = _uncounted(counted, parents, rows)
        if todo.size:
            for chunk, counts in _chunked_counts(contract, self.plan.sampler, self.plan.index, anchors, todo, parents,
                                                 self.plan.build_bytes):
                for parent in parents:
                    counted[parent][chunk] = counts[parent]
        return {parent: counted[parent][rows] for parent in parents}, 0

    def _counted(self, anchors: GpuAnchors, parents: list[str]) -> dict[str, Any]:
        """The per-row event counts kept for `anchors` (this task's anchor table), or new ones (requests)."""

        import cupy

        size = int(anchors.cutoffs.size)
        cache = self._event_counts_cache if anchors is self.anchors else {}
        for parent in parents:
            if parent not in cache:
                cache[parent] = cupy.full(size, -1, dtype=cupy.int64)
        return cache

    def _model_rest(self) -> int:
        """The part of the model's measured peak (`prepare(model_bytes=)`) not yet allocated: it must stay free."""

        model = getattr(self.plan, "model_bytes", None)
        if not model:
            return 0
        import torch

        return max(0, int(model) - int(torch.cuda.memory_allocated()))

    def _planned(self, build: Any, contract: DataLoaderSpec, rows: Any, padding: Any, groups: list[tuple[int, int]],
                 maximum: int | None, drop_last: bool, first: int = 0) -> Iterator[Any]:
        """Batches under a token budget or length buckets: each sample's length measured on the length input
        alone (no target changes it), the batches composed (`test.kumo_pytorch.reference_engine.gpu.batching.compose.plan_batches`), then built in that
        order."""

        import cupy

        from test.kumo_pytorch.reference_engine.gpu.batching.compose import plan_batches

        policy = contract.batch.batching
        lengths = _measured(build, contract, rows, padding, self.plan.build_bytes)
        order, plan = plan_batches(lengths, contract.batch.inputs[policy.length_input].layout, groups, maximum,
                                   policy.max_tokens, policy.bucket_by_length, drop_last)
        order = cupy.asarray(order)
        yield from _in_groups(partial(build, contract, targets=not self.inference), rows[order], padding[order], plan,
                              self.plan.build_bytes, first)

    def close(self) -> None:
        self.anchors.close()

    def __enter__(self) -> GpuTask:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()
