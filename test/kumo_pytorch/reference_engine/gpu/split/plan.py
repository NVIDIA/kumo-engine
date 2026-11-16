"""Prepare a contract on a split dataset: one too big for the GPU, split by its entity (`test.kumo_pytorch.reference_engine.gpu.split`)
and streamed through the GPU a partition at a time (`test.kumo_pytorch.reference_engine.gpu.split.streaming`).

`test.kumo_pytorch.reference_engine.gpu.prepare` comes here when the columns a contract reads do not fit on the
GPU (`test.kumo_pytorch.reference_engine.gpu.split.routing.choose_route`). The split lives in a folder the caller
names, as upstream's DiskGraph engine keeps its index in one
(``DiskGraphEngine(index=..., ingest="if_missing")``):

- A split already there is reused when it holds what the contract reads: the
  same entity, every table and column the contract reads with the same types,
  written from the same files (each table's paths, sizes and modification
  times: `test.kumo_pytorch.reference_engine.gpu.split.writing.source_fingerprint`), the rows the sample reaches
  along its routes (`test.kumo_pytorch.reference_engine.gpu.split.streaming.PartitionLayout.check_sample`), and the
  same assignment of entities to partitions.
- Otherwise, with ``split="if_missing"``, a split is written for this contract
  (the columns it reads and its sample's routes) into a temporary folder beside
  the named one, and renamed into place once complete; with ``split="never"``
  the dataset is refused.
- A split that does not match is never overwritten, nor is a folder that holds
  anything other than a split.

The plan (`SplitPlan`) keeps the shared tables on the GPU; each partition's own
tables are streamed through it. Each anchor belongs to the partition that holds
its entity: the split's own hash of the entity's key (`partition_of`), so no
table is read to place it.

Fitting (`test.kumo_pytorch.reference_engine.gpu.split.fit`) reads the fit split's anchors partition by
partition and merges what each fit needs: counts, moments, sorted values,
tuples; a second pass when something fitted depends on the first (Composite
IDs, derived labels). Stored-column encoders are fitted from original tables first.

Batches (`SplitTask.batches`) take upstream's order of the anchors (the anchor
table's, or a permutation of it seeded by ``seed`` and ``epoch``) regrouped by
partition: the partitions one after another (shuffled by the same seed and
epoch when ``shuffle``), each partition's anchors keeping their order. Each
partition is read once a pass, and only those holding anchors of the call. A
batch is cut from that order as upstream cuts its order: as many batches, of
the same sizes, ``drop_last`` dropping only the last. A batch that a partition
boundary crosses is sampled part by part, each part while its partition is on
the GPU, then built from the joined parts (`test.kumo_pytorch.reference_engine.gpu.split.parts`). Under a token
budget or length buckets, batches are composed within each partition (a chunk
never spans two). Each sample equals the one the dataset loaded whole gives.

With several ranks (``world_size``), rank r takes the r-th contiguous slice of
that order, ``ceil(count / world_size)`` anchors as upstream gives each rank, a
short rank topped up by repeating its last anchor (marked as padding): each
partition is read by one rank, but at a slice's edge, so the ranks together
read the dataset about once a pass. Ranks preparing the same split at once
take turns at a lock beside its folder: one writes it, the others reuse it.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from test.kumo_pytorch.reference_engine.gpu.plan.gpu_plan import GpuPlan
from test.kumo_pytorch.reference_engine.gpu.batching.admission import admitted
from test.kumo_pytorch.reference_engine.gpu.plan.gpu_task import GpuTask

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import EntityIndex
    from test.kumo_pytorch.reference_engine.gpu.split.parts import SampledPart
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler
    from test.kumo_pytorch.reference_engine.gpu.split.streaming import Partition, PartitionLayout
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ["SPLIT_POLICIES", "SplitPlan", "SplitTask", "ensure_split", "prepare_split", "split_mismatches"]

#: ``prepare(..., split=)``: write a split that is missing, or never write one.
SPLIT_POLICIES = ("if_missing", "never")


def prepare_split(contract: DataLoaderSpec, dataset: Dataset, engine: Any, schema: DataSchema, sample: SampleSpec,
                  split_dir: str | Path, *, if_missing: bool, inference: bool, num_partitions: int | None = None,
                  model_bytes: int | None = None, room: int | None = None,
                  peak_room: int | None = None) -> SplitPlan:
    """`test.kumo_pytorch.reference_engine.gpu.prepare` for a dataset that does not fit on the GPU: the split in `split_dir` (`ensure_split`), and
    its shared tables loaded.

    :param schema: the tables and columns the contract reads (`test.kumo_pytorch.reference_engine.gpu.plan.columns._narrowed`).
    :param sample: the sample the split serves (without the targets' reads for inference).
    :param num_partitions, model_bytes, room, peak_room: `ensure_split`'s.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.tables import load_tables

    layout = ensure_split(schema, dataset, sample, split_dir, if_missing=if_missing, num_partitions=num_partitions,
                          model_bytes=model_bytes, room=room, peak_room=peak_room)
    shared = [table.name for table in schema.tables if layout.kind(table.name) == "shared"]
    tables = load_tables(schema, dataset, only=shared) if shared else {}
    return SplitPlan(contract, dataset, engine, layout, schema, tables, inference=inference)


def ensure_split(schema: DataSchema, dataset: Dataset, sample: SampleSpec, directory: str | Path, *,
                 if_missing: bool = True, num_partitions: int | None = None, model_bytes: int | None = None,
                 room: int | None = None, peak_room: int | None = None) -> PartitionLayout:
    """The split of `dataset` in `directory` for `sample`: reused when it matches (`split_mismatches`), else written
    there when `if_missing`. A mismatched split, or a folder holding anything else, is refused and left as it is.

    :param num_partitions: write exactly this many partitions; a split found with another count does not match.
    :param model_bytes: the model's measured peak, which a split written here is sized beside.
    :param room: the bytes each partition may hold now, indexes included
        (`test.kumo_pytorch.reference_engine.gpu.split.routing.Route.partition_bytes`), and `peak_room` the room for two
        partitions and one's peak (`Route.partition_peak_bytes`); a split found whose
        largest partition takes more does not match.
    """

    from test.kumo_pytorch.reference_engine.gpu.split.writing import LAYOUT, write_partitions
    from test.kumo_pytorch.reference_engine.gpu.split.streaming import PartitionLayout

    directory = Path(directory).expanduser().resolve()
    found = _existing(directory, schema, dataset, sample, num_partitions, room, peak_room)
    if found is not None:
        return found
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _locked(directory):  # one process writes it; the others, waiting here, then find it
        found = _existing(directory, schema, dataset, sample, num_partitions, room, peak_room)
        if found is not None:
            return found
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"{directory} holds no {LAYOUT}, so it is not a split (or one left unfinished), and it is "
                             "not empty: remove it, or pass another split_dir")
        if not if_missing:
            raise FileNotFoundError(f"there is no split at {directory}, and split='never' writes none")
        with tempfile.TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
            built = Path(temporary) / "split"
            write_partitions(schema, dataset, sample.entity.table, built, sample=sample, num_partitions=num_partitions,
                             model_bytes=model_bytes)
            if directory.exists():
                directory.rmdir()  # empty (checked above); raises if anything appeared in it meanwhile
            built.rename(directory)
    return PartitionLayout.open(directory)


def _existing(directory: Path, schema: DataSchema, dataset: Dataset, sample: SampleSpec, num_partitions: int | None,
              room: int | None, peak_room: int | None) -> PartitionLayout | None:
    """The split in `directory` when it serves `sample` (`split_mismatches`); refused when it does not; None when
    there is none."""

    from test.kumo_pytorch.reference_engine.gpu.split.writing import LAYOUT
    from test.kumo_pytorch.reference_engine.gpu.split.streaming import PartitionLayout

    if not (directory / LAYOUT).is_file():
        return None
    layout = PartitionLayout.open(directory)
    problems = split_mismatches(layout, schema, dataset, sample, num_partitions=num_partitions, room=room,
                                peak_room=peak_room)
    if problems:
        raise ValueError(f"the split at {directory} does not serve this contract and dataset, and a split is never "
                         f"overwritten: {'; '.join(problems)}. Remove it, or pass another split_dir")
    return layout


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """An exclusive lock on writing the split in `directory`, held by one process at a time (several ranks
    preparing the same contract): a lock file beside it, ``.<name>.lock``, left there."""

    import fcntl

    with open(directory.parent / f".{directory.name}.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def split_mismatches(layout: PartitionLayout, schema: DataSchema, dataset: Dataset, sample: SampleSpec, *,
                     num_partitions: int | None = None, room: int | None = None,
                     peak_room: int | None = None) -> list[str]:
    """Why the split in `layout` cannot serve `sample` over `schema`'s tables and columns of `dataset` (none: it can):
    with `num_partitions`, another partition count; with `room` and `peak_room` (`ensure_split`), a partition
    bigger than they allow, counted from its rows as `sample` reads them now (`test.kumo_pytorch.reference_engine.gpu.split.routing._partition_bytes`)."""

    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.split.routing import _loads, _partition_bytes
    from test.kumo_pytorch.reference_engine.gpu.split.writing import _columns_json, source_fingerprint
    from test.kumo_pytorch.reference_engine.gpu.data.sources import resolve_sources

    root = sample.entity.table
    if layout.root != root:
        return [f"it is split by {layout.root!r}, and the sample is about {root!r}"]
    problems = []
    if num_partitions is not None and layout.num_partitions != num_partitions:
        problems.append(f"it has {layout.num_partitions} partitions, and {num_partitions} were asked for")
    if num_partitions is None and room is not None:
        loads = _loads(schema, sample, resolve_sources(schema, dataset))
        for partition in layout.record.get("partitions", ()):
            rows = {name: count + partition["halo_rows"].get(name, 0) for name, count in partition["rows"].items()
                    if name in loads}
            size = _partition_bytes(loads, rows)
            held, peak = size["held_bytes"], 2 * size["held_bytes"] + size["extra_bytes"]
            if held > room or (peak_room is not None and peak > peak_room):
                problems.append(
                    f"its partition {partition['id']} holds {held / 1e9:.2f} GB with its indexes and needs "
                    f"{size['extra_bytes'] / 1e9:.2f} GB more at its peak, more than the {room / 1e9:.2f} GB each of "
                    "the two partitions on the GPU at once may hold here"
                    + ("" if peak_room is None else f", or the {peak_room / 1e9:.2f} GB for two and one's peak"))
                break
    if layout.record.get("floats") != "exact":
        problems.append("it was written by an earlier test.kumo_pytorch.reference_engine.gpu, whose files may hold 0.0 for a float column's -0.0")
    assignment = layout.record.get("assignment", {})
    if (assignment.get("hash"), assignment.get("seed")) != ("murmurhash3_x86_32",
                                                             int(plc.hashing.LIBCUDF_DEFAULT_HASH_SEED)):
        problems.append("its entities were assigned to partitions by another hash")
    written_tables = layout.record["tables"]
    for name, (source, columns) in resolve_sources(schema, dataset).items():
        written = written_tables.get(name)
        if written is None:
            problems.append(f"it has no table {name!r}")
            continue
        for column, dtype in _columns_json(columns).items():
            if column not in written["columns"]:
                problems.append(f"it has no column {name}.{column}")
            elif written["columns"][column] != dtype:
                problems.append(f"it wrote {name}.{column} with another type")
        recorded = written.get("fingerprint")
        if recorded is None:
            problems.append(f"it records no fingerprint of the {name} files")
        elif recorded != source_fingerprint(source):
            problems.append(f"the {name} files changed after it was written")
    if not problems:
        try:
            layout.check_sample(schema, sample)
        except ValueError as error:
            problems.append(str(error))
    return problems


class SplitPlan(GpuPlan):
    """A contract bound to a split dataset: upstream's `Plan`, for a dataset streamed a partition at a time.

    `tables` holds the shared tables, on the GPU throughout; each partition's own
    tables are read when its anchors are fitted or batched. Anchors are bound as
    `GpuPlan.materialize` binds them, each placed in its entity's partition.
    """

    def __init__(self, contract: DataLoaderSpec, dataset: Dataset, engine: Any, layout: PartitionLayout,
                 schema: DataSchema, shared: dict[str, GpuTable], *, inference: bool = False) -> None:
        super().__init__(contract, dataset, engine, shared, None, None, inference=inference)
        self.layout = layout
        self.schema = schema  # the tables and columns the contract reads; each partition loads its own of them

    @property
    def num_partitions(self) -> int:
        return self.layout.num_partitions

    def partition_of(self, anchors: GpuAnchors, rows: Any) -> Any:
        """The partition holding the entity of each anchor at table `rows` (CuPy, or a slice), as a CuPy int32
        array: the split's hash of the key (`partition_of_keys`), an integer in its root table's type, a text by
        its bytes (each distinct text hashed once), a key of another type from the ID normalized to the key's type
        (`hashed_keys`). An entity no partition holds still gets one, which lacks it; so does an ID of another
        type than the root's keys, which no partition holds."""

        import cupy

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view
        from test.kumo_pytorch.reference_engine.gpu.split.hashing import partition_of_keys
        from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import _device_type

        root = self.layout.root
        key = next(table.primary_key for table in self.schema.tables if table.name == root)
        dtype = self.layout.columns(root)[key]
        if not (dtype.integer or dtype.name == "string"):
            return self._typed_partition_of(anchors, rows, dtype)
        values, texts = anchors.entity_values()
        values = values[rows]
        if not values.size or (texts is not None) != (dtype.name == "string"):
            return cupy.zeros(values.size, dtype=cupy.int32)
        if texts is not None:
            return _view(partition_of_keys(texts, self.num_partitions)).astype(cupy.int32)[values]
        keys = cupy.ascontiguousarray(values.astype(_device_type(dtype)))
        return _view(partition_of_keys(_column(keys), self.num_partitions)).astype(cupy.int32)

    def _typed_partition_of(self, anchors: GpuAnchors, rows: Any, dtype: Any) -> Any:
        """`partition_of` for keys of types other than integers and text: each anchor's ID normalized to the key's
        type in Python (as upstream normalizes it), as a column of it, then hashed as the split hashed the keys. An
        ID upstream refuses goes to partition 0, which then refuses it; a NaN is hashed, and refused by the
        partition it lands in, which holds no key equal to it."""

        import cupy
        import pyarrow as pa

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view
        from test.kumo_pytorch.reference_engine.gpu.split.hashing import partition_of_keys
        from test.kumo_pytorch.reference_engine.gpu.data.text import arrow_column
        from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import decimal_stored

        ids = anchors._python_ids()
        picked = [ids[row] for row in (range(*rows.indices(len(ids))) if isinstance(rows, slice)
                                       else rows.get().tolist())]
        if not picked:
            return cupy.zeros(0, dtype=cupy.int32)
        normalized = []
        for value in picked:
            try:
                normalized.append(dtype.normalize(value))
            except ValueError:
                normalized.append(None)
        array = pa.array(normalized, dtype.to_arrow())
        column = arrow_column(array) if dtype.name == "bytes" else _arrow_key_column(array, dtype)
        column = decimal_stored(column, dtype, "anchors") if dtype.name.startswith("decimal") else column
        parts = _view(partition_of_keys(column, self.num_partitions, dtype)).astype(cupy.int32)
        valid = cupy.asarray(array.is_valid().to_numpy(zero_copy_only=False))
        return cupy.where(valid, parts, 0).astype(cupy.int32)


    def stream(self, order: Sequence[int]) -> Iterator[Partition]:
        """The partitions in `order`, each with the plan's shared tables (`test.kumo_pytorch.reference_engine.gpu.split.streaming.stream_partitions`:
        the next one read while one is in use)."""

        from test.kumo_pytorch.reference_engine.gpu.split.streaming import stream_partitions

        return stream_partitions(self.schema, self.layout, order=order, shared=self.tables)

    def reader(self, partition: Partition) -> tuple[Sampler, EntityIndex]:
        """What reads `partition`'s tables: a sampler with unchanged anchor cutoffs and its entities' index."""

        from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import EntityIndex
        from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

        root = self.layout.root
        tables = dict(partition.tables)
        sampler = Sampler(self.contract, tables)
        return sampler, EntityIndex(tables[root], next(t.primary_key for t in self.schema.tables if t.name == root))

    def _fit_inputs(self, bound: Any, rows: Any, reused: Any, names: dict[str, Any], *, budget: Any = None, pinned=None) -> tuple[Any, ...]:
        """Everything fitted partition by partition (`test.kumo_pytorch.reference_engine.gpu.split.fit`)."""

        from test.kumo_pytorch.reference_engine.gpu.split.fit import fit_split_state

        return fit_split_state(self, bound, rows, names, reused, budget=budget, pinned=pinned)

    def _measure_fit(self, bound: Any, rows: Any, encoders: Any, composites: Any, spaces: Any, targets: Any,
                     names: dict[str, Any]) -> dict[str, Any]:
        """Each fitted sample's length in ``Batching.length_input``, partition by partition, in `rows`' order."""

        from test.kumo_pytorch.reference_engine.gpu.split.fit import measure_split_lengths

        return measure_split_lengths(self, bound, rows, encoders, composites, spaces, targets, names)

    def _task(self, *arguments: Any, **options: Any) -> SplitTask:
        return SplitTask(self, *arguments, **options)


class SplitTask(GpuTask):
    """Anchors bound to a split dataset's plan: upstream's `Task`, each anchor in its entity's partition."""

    def __init__(self, plan: SplitPlan, anchors: Any, *arguments: Any, **options: Any) -> None:
        super().__init__(plan, anchors, *arguments, **options)
        # each anchor row's partition (the anchor table's rows, or the requests in order)
        self.partitions = plan.partition_of(anchors, slice(None))
        self._partition_resident: dict[int, int] = {}  # bytes each partition kept on the GPU while counted

    def partition_counts(self, split: Any = None) -> list[int]:
        """How many anchors of `split` (all when omitted) each partition holds."""

        import cupy
        from kumo.pytorch.anchors import ALL_ANCHORS, split_selection

        selection = split_selection(split)
        rows = self.anchors.rows(ALL_ANCHORS if selection is None else selection)
        return cupy.bincount(self.partitions[rows], minlength=self.plan.num_partitions).get().tolist()

    def _positions(self, anchors: Any, selection: Any, *, seed: int, epoch: int, shuffle: bool, rank: int,
                   world_size: int) -> tuple[Any, Any]:
        """This rank's anchors: a contiguous slice of upstream's order regrouped by partition, ``ceil(count /
        world_size)`` of them (rank r the r-th), so each partition is read by one rank but at a slice's edge. A
        short rank is topped up by repeating its last anchor (its partition's still on the GPU), marked as
        padding; a rank with none, by the first anchor of all, as `GpuTask._positions` tops it up."""

        import cupy

        if world_size == 1:
            return super()._positions(anchors, selection, seed=seed, epoch=epoch, shuffle=shuffle, rank=0,
                                      world_size=1)
        whole, _ = super()._positions(anchors, selection, seed=seed, epoch=epoch, shuffle=shuffle, rank=0,
                                      world_size=1)
        where = self.plan.partition_of(anchors, anchors.rows(selection)[whole])
        whole = whole[self._regrouped(where, seed, epoch, shuffle)]
        target = -(-int(whole.size) // world_size) if whole.size else 0
        mine = whole[rank * target:(rank + 1) * target]
        short = target - int(mine.size)
        padding = cupy.zeros(target, dtype=bool)
        if short:
            last = mine[-1:] if mine.size else cupy.zeros(1, dtype=cupy.int64)
            mine = cupy.concatenate([mine, cupy.repeat(last, short)])
            padding[target - short:] = True
        return mine, padding

    def _regrouped(self, where: Any, seed: int, epoch: int, shuffle: bool) -> Any:
        """The order that regroups anchors in partitions `where` (CuPy) by partition: the partitions in
        `shard_positions`' order over them (shuffled by `seed` and `epoch`), each keeping its anchors' order."""

        import cupy
        from kumo.pytorch.anchors import shard_positions

        count, size = self.plan.num_partitions, int(where.size)
        order = list(shard_positions(count, seed=seed, epoch=epoch, shuffle=shuffle))
        place = cupy.empty(count, dtype=cupy.int64)
        place[cupy.asarray(order, dtype=cupy.int64)] = cupy.arange(count, dtype=cupy.int64)
        return cupy.argsort(place[where] * max(size, 1) + cupy.arange(size, dtype=cupy.int64))

    def _emit(self, contract: DataLoaderSpec, rows: Any, padding: Any, options: dict[str, Any], *, drop_last: bool,
              shuffle: bool, from_requests: bool) -> Iterator[Any]:
        """The anchors at `rows` (upstream's order, or a rank's slice of it regrouped) regrouped by partition,
        then batched. Regrouping keeps a slice's order: its anchors are grouped already, its padding (repeats of
        its last anchor) in the last group."""

        import cupy
        from kumo.pytorch.anchors import shard_positions

        plan = self.plan
        count, size = plan.num_partitions, int(rows.size)
        where = plan.partition_of(options["anchors"], rows) if from_requests else self.partitions[rows]
        order = list(shard_positions(count, seed=options["seed"], epoch=options["epoch"], shuffle=shuffle))
        regrouped = self._regrouped(where, options["seed"], options["epoch"], shuffle)
        rows, padding, where = rows[regrouped], padding[regrouped], where[regrouped]
        held = cupy.bincount(where, minlength=count).get().tolist() if size else [0] * count
        segments, start = [], 0
        for k in order:  # (partition, its first and last-plus-one place in the regrouped order)
            if held[k]:
                segments.append((k, start, start + held[k]))
                start += held[k]
        policy = contract.batch.batching
        if policy.max_tokens is None and not policy.bucket_by_length:
            from test.kumo_pytorch.reference_engine.gpu.batching.compose import BatchPlan

            cut = BatchPlan.fixed(size, options["maximum"], drop_last)
            # requests are new anchors each call: counting them would read every partition again each call
            return admitted(contract, lambda: self._fixed(contract, rows, padding, segments, options, drop_last),
                            lambda: None if from_requests else
                            self._admission(contract, options["anchors"], rows, cut.bounds[:cut.emit], where))
        return admitted(contract, lambda: self._composed(contract, rows, padding, segments, options, drop_last,
                                                         from_requests), lambda: None)

    def _event_counts(self, contract: DataLoaderSpec, anchors: Any, rows: Any, parents: list[str],
                      where: Any = None) -> tuple[dict[str, Any], int]:
        """Each anchor's events at `rows` (in partitions `where`), counted partition by partition, one on the GPU
        at a time, and what two partitions keep beside a batch: the two biggest of the call's, as measured while
        each was counted (RMM's count of what stays allocated: its tables and the indexes its sampling builds), kept
        for later calls."""

        from test.kumo_pytorch.reference_engine.gpu.data.memory import release_unused
        from test.kumo_pytorch.reference_engine.gpu.plan.gpu_task import _chunked_counts, _uncounted
        from test.kumo_pytorch.reference_engine.gpu.split.streaming import stream_partitions

        import gc

        import cupy

        counted = self._counted(anchors, parents)
        held = cupy.unique(where).get().tolist() if rows.size else []
        todo = _uncounted(counted, parents, rows)
        todo_where = self.partitions[todo] if anchors is self.anchors else self.plan.partition_of(anchors, todo)
        resident = self._partition_resident
        order = sorted(set(cupy.unique(todo_where).get().tolist()) | {k for k in held if k not in resident})
        if order:
            import rmm.statistics

            rmm.statistics.enable_statistics()  # once; keeps the pool beneath it
            stream = stream_partitions(self.plan.schema, self.plan.layout, order=order, read_ahead=False,
                                       shared=self.plan.tables)
            for k in order:
                rmm.statistics.push_statistics()  # what stays allocated: the partition and its indexes
                try:
                    partition = next(stream)
                    sampler, index = self.plan.reader(partition)
                    mine = todo[todo_where == k]
                    if mine.size:
                        for chunk, counts in _chunked_counts(contract, sampler, index, anchors, mine, parents,
                                                             self.plan.build_bytes):
                            for parent in parents:
                                counted[parent][chunk] = counts[parent]
                finally:
                    kept = rmm.statistics.pop_statistics().current_bytes
                resident[k] = max(0, int(kept))
                del partition, sampler, index
                gc.collect()  # freed before the next one is measured (its tables sit in reference cycles)
            next(stream, None)  # the last partition released
            release_unused()
        biggest = sorted((resident[k] for k in held), reverse=True)[:2]
        return {parent: counted[parent][rows] for parent in parents}, int(sum(biggest))

    def _fixed(self, contract: DataLoaderSpec, rows: Any, padding: Any, segments: list[tuple[int, int, int]],
               options: dict[str, Any], drop_last: bool) -> Iterator[Any]:
        """``max_samples`` batches cut from the regrouped order: those inside a partition built while it is on
        the GPU, one a boundary crosses built from its parts, each sampled while its partition was."""

        from functools import partial

        from test.kumo_pytorch.reference_engine.gpu.batching.compose import BatchPlan
        from test.kumo_pytorch.reference_engine.gpu.batching.build import build_batches
        from test.kumo_pytorch.reference_engine.gpu.plan.groups import _in_groups

        size, maximum = int(rows.size), options["maximum"]
        bounds = {k: (start, stop) for k, start, stop in segments}
        carried: list[SampledPart] = []  # the parts of a batch begun in earlier partitions, from `opened`
        opened = 0
        for partition in self.plan.stream([k for k, _, _ in segments]):
            start, stop = bounds[partition.id]
            sampler, index = self.plan.reader(partition)
            del partition  # held by the sampler until this partition is done
            build = partial(build_batches, contract, sampler, index, targets=not self.inference, **options)
            position = start
            if carried:  # the open batch takes this partition's first anchors
                end = min(stop, (start // maximum + 1) * maximum)
                carried.append(self._part(contract, sampler, index, rows[start:end], options))
                position = end
                if end % maximum == 0 or end == size:
                    yield from build(rows=rows[opened:end], padding=padding[opened:end], parts=carried,
                                     drop_last=drop_last and end == size, first=opened)
                    carried = []
            whole = stop if stop == size else max(position, stop // maximum * maximum)
            if whole > position:  # the batches inside this partition, a group at a time when they are many
                last = drop_last and whole == size
                yield from _in_groups(partial(build, drop_last=last), rows[position:whole], padding[position:whole],
                                      BatchPlan.fixed(whole - position, maximum, last), self.plan.build_bytes,
                                      position)
                position = whole
            if position < stop:  # a batch the next partitions finish
                opened, carried = position, [self._part(contract, sampler, index, rows[position:stop], options)]
            del sampler, index, build  # this partition freed before the one after the next is read

    def _composed(self, contract: DataLoaderSpec, rows: Any, padding: Any, segments: list[tuple[int, int, int]],
                  options: dict[str, Any], drop_last: bool, from_requests: bool) -> Iterator[Any]:
        """Batches under a token budget or length buckets, composed within each partition
        (`GpuTask._planned`); ``drop_last`` applies to the last partition's."""

        from functools import partial

        from test.kumo_pytorch.reference_engine.gpu.batching.build import build_batches
        from test.kumo_pytorch.reference_engine.gpu.plan.groups import _groups

        size = int(rows.size)
        bounds = {k: (start, stop) for k, start, stop in segments}
        for partition in self.plan.stream([k for k, _, _ in segments]):
            start, stop = bounds[partition.id]
            sampler, index = self.plan.reader(partition)
            del partition
            last = drop_last and stop == size
            build = partial(build_batches, sampler=sampler, index=index, drop_last=last, **options)
            yield from self._planned(build, contract, rows[start:stop], padding[start:stop],
                                     _groups(stop - start, options, from_requests), options["maximum"], last, start)
            del sampler, index, build

    def _part(self, contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, rows: Any,
              options: dict[str, Any]) -> SampledPart:
        from test.kumo_pytorch.reference_engine.gpu.split.parts import sample_part

        return sample_part(contract, sampler, index, options["anchors"], rows, options["encoders"], options["spaces"],
                           model=options["model"], seed=options["seed"], epoch=options["epoch"])


def _arrow_key_column(array: Any, dtype: Any) -> Any:
    """An Arrow array of key `dtype` as the libcudf column a split reads such keys into: a decimal256 as a
    decimal128 (precision up to 38) or, wider, as its bytes (`test.kumo_pytorch.reference_engine.gpu.data.decimal_storage._wide_decimal`); others as they are
    (a timestamp key is whole UTC seconds, the only unit `DType` takes)."""

    import pyarrow as pa
    import pylibcudf as plc

    if dtype.name == "decimal256" and dtype.precision > 38:
        import numpy as np

        from test.kumo_pytorch.reference_engine.gpu.data.text import arrow_column

        words = np.frombuffer(array.buffers()[1], dtype=np.uint8).reshape(len(array), 32)[:, ::-1]
        return arrow_column(pa.array([bytes(row) for row in words], pa.binary()))
    if dtype.name == "decimal256":
        array = array.cast(pa.decimal128(dtype.precision, dtype.scale))
    return plc.Column.from_arrow(array)
