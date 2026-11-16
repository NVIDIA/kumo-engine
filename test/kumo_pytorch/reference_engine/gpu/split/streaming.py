"""Stream a split dataset (`test.kumo_pytorch.reference_engine.gpu.split`) through a GPU, one partition at a time.

    layout = PartitionLayout.open("/data/split")
    for partition in stream_partitions(schema, layout):
        ...  # partition.tables: every table, the shared ones and this partition's own
        ...  # partition.home_rows[t]: how many of t's first rows are the partition's own

Shared tables are loaded once and stay on the GPU; each partition's own tables
are loaded through the direct Parquet route, used, and freed. While one
partition is in use, the next one is read in a background thread, so reading
overlaps the work on the current one. Two partitions are then on the GPU at
once, which the split's choice of partition count leaves room for.

The background read uses the same CUDA stream as the caller's work, so its GPU
steps are ordered with that work and no memory can be reused while still in
use. Its file reading still overlaps the caller's GPU work.

Release a partition (drop every reference to its tables) before asking for the
next one plus one, or three partitions are held. It is then collected (its
tables sit in reference cycles) and its memory given back to CUDA
(`test.kumo_pytorch.reference_engine.gpu.data.memory.release_unused`), where PyTorch can allocate it.

A split made for a contract's sample also holds, in each partition, rows that
belong elsewhere but that its root rows reach along the sample's routes. They
follow the partition's own rows in each table: ``home_rows`` says where.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ["Partition", "PartitionLayout", "stream_partitions"]


@dataclass(frozen=True)
class Partition:
    """One partition's tables on the GPU: the shared tables and its own.

    :param home_rows: for each split table, how many of its first rows belong to
        this partition; the rest are copies its routes reach. Shared tables are
        not listed. Only the root's first ``home_rows`` rows are this partition's
        entities.
    """

    id: int
    tables: Mapping[str, GpuTable]
    home_rows: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionLayout:
    """A split dataset as `test.kumo_pytorch.reference_engine.gpu.split.writing.write_partitions` wrote it."""

    directory: Path
    record: Mapping[str, Any]

    @classmethod
    def open(cls, directory: str | Path) -> PartitionLayout:
        from test.kumo_pytorch.reference_engine.gpu.split.writing import LAYOUT

        directory = Path(directory)
        path = directory / LAYOUT
        if not path.is_file():
            raise ValueError(f"{directory} holds no {LAYOUT}; write it with test.kumo_pytorch.reference_engine.gpu.split.writing.write_partitions")
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1:
            raise ValueError(f"{path}: unknown layout schema_version {record.get('schema_version')!r}")
        return cls(directory, record)

    @property
    def num_partitions(self) -> int:
        return int(self.record["num_partitions"])

    @property
    def root(self) -> str:
        return str(self.record["root"])

    def kind(self, table: str) -> str:
        return str(self.record["tables"][table]["kind"])

    def home_rows(self, k: int, table: str) -> int:
        """Rows of split `table` that belong to partition `k`; they come first in its files."""

        return int(self.record["partitions"][k]["rows"][table])

    def halo_rows(self, k: int, table: str) -> int:
        """Rows of split `table` copied into partition `k` for the routes of the sample it was split for."""

        return int(self.record["partitions"][k].get("halo_rows", {}).get(table, 0))

    def columns(self, table: str) -> dict[str, Any]:
        from kumo.pytorch.contract.dtypes import DType

        return {name: DType(**dtype) for name, dtype in self.record["tables"][table]["columns"].items()}

    def partition_dataset(self, k: int, tables: Sequence[str]) -> Dataset:
        """Partition `k`'s own files for `tables` (its partitioned tables)."""

        from test.kumo_pytorch.reference_engine.gpu.split.writing import _part
        from kumo.pytorch.dataset import Dataset, Parquet

        self._check_partition(k)
        return Dataset({name: Parquet(str(self.directory / _part(k) / name)) for name in tables})

    def shared_dataset(self, tables: Sequence[str]) -> Dataset:
        """The original sources of the shared `tables`."""

        from kumo.pytorch.dataset import Dataset, Parquet

        sources = {name: self.record["tables"][name]["source"] for name in tables}
        return Dataset({name: Parquet(source["path"], columns=source["columns"])
                        for name, source in sources.items()})

    def typed_schema(self, schema: DataSchema) -> DataSchema:
        """`schema` with every table's columns typed as the split wrote them.

        A table may declare a subset of the split's columns, each with the
        split's type or with its type omitted; omitting the columns takes them all.
        """

        tables = []
        for table in schema.tables:
            if table.name not in self.record["tables"]:
                raise ValueError(f"the split at {self.directory} has no table {table.name!r}")
            written = self.columns(table.name)
            if table.columns is None:
                tables.append(replace(table, columns=written))
                continue
            columns = {}
            for name, dtype in table.columns.items():
                if name not in written:
                    raise ValueError(f"the split at {self.directory} has no column {table.name}.{name}")
                if dtype is not None and dtype != written[name]:
                    raise ValueError(f"{table.name}.{name}: declared {dtype!r}, the split wrote {written[name]!r}")
                columns[name] = written[name]
            tables.append(replace(table, columns=columns))
        return replace(schema, tables=tuple(tables))

    def check_sample(self, schema: DataSchema, sample: SampleSpec) -> None:
        """Refuse `sample` unless each partition holds every row it reads from that partition's root rows.

        The split holds them when it was made for a sample whose routes include
        this one's (`test.kumo_pytorch.reference_engine.gpu.split.writing.write_partitions` with ``sample=``), or
        when a route only reaches shared tables or the partition's own rows
        (down the chains from the root). A split is made once and reused for
        every sample it holds.
        """

        from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import Step, _downward, _refuse_split_candidates
        from test.kumo_pytorch.reference_engine.gpu.sampling.paths import Hop, sample_routes

        if sample.entity.table != self.root:
            raise ValueError(f"the sample is about {sample.entity.table!r}, but the split at {self.directory} "
                             f"is by {self.root!r}")
        tables = self.record["tables"]
        split = {name for name, table in tables.items() if table["kind"] == "partitioned"}
        _refuse_split_candidates(schema, sample, split, self.root)
        owned = {name: tuple(tuple(Step(**step) for step in chain) for chain in tables[name]["chains"])
                 for name in split}
        made = [tuple(Hop(**hop) for hop in route) for route in self.record.get("routes", [])]
        walked = {route[:n] for route in made for n in range(1, len(route) + 1)}
        for route in sample_routes(schema, sample):
            for n in range(1, len(route) + 1):
                prefix = route[:n]
                table = prefix[-1].table
                if table not in tables:
                    raise ValueError(f"the split at {self.directory} has no table {table!r}")
                if table in split and prefix not in walked and not _downward(owned, prefix):
                    steps = "".join(f" -{hop.relation}-> {hop.table}" for hop in prefix)
                    raise ValueError(
                        f"the split at {self.directory} was not made for this sample: its partitions do not hold "
                        f"the {table} rows reached along {self.root}{steps} from their {self.root} rows. Split the "
                        "dataset again, passing this sample"
                    )

    def _check_partition(self, k: int) -> None:
        if type(k) is not int or not 0 <= k < self.num_partitions:
            raise ValueError(f"partition {k!r} is not one of 0..{self.num_partitions - 1}")


def stream_partitions(schema: DataSchema, layout: PartitionLayout, *, order: Sequence[int] | None = None,
                      read_ahead: bool = True, shared: Mapping[str, GpuTable] | None = None) -> Iterator[Partition]:
    """Each partition in `order` (all, ascending, when omitted), with the shared tables.

    :param read_ahead: read the next partition in the background while the
        current one is in use.
    :param shared: the shared tables, already on the GPU (a plan's); read from
        their files when omitted.
    """

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.data.memory import device, release_unused
    from test.kumo_pytorch.reference_engine.gpu.data.tables import load_tables

    gpu = device()  # the read-ahead thread works on this GPU too, not on CUDA's default one
    typed = layout.typed_schema(schema)
    names = [table.name for table in typed.tables]
    own = [name for name in names if layout.kind(name) == "partitioned"]
    shared_names = [name for name in names if layout.kind(name) == "shared"]
    order = list(range(layout.num_partitions)) if order is None else list(order)
    for k in order:
        layout._check_partition(k)
    if shared is None:
        shared = load_tables(typed, layout.shared_dataset(shared_names), only=shared_names) if shared_names else {}
    elif sorted(shared) != sorted(shared_names):
        raise ValueError(f"the shared tables given are {sorted(shared)}, and the split shares {sorted(shared_names)}")

    def load(k: int) -> dict[str, GpuTable]:
        with cupy.cuda.Device(gpu):
            tables = load_tables(typed, layout.partition_dataset(k, own), only=own, split=True) if own else {}
        for name, table in tables.items():
            expected = layout.home_rows(k, name) + layout.halo_rows(k, name)
            if table.num_rows != expected:
                raise ValueError(f"partition {k} of {layout.directory} holds {table.num_rows} {name} rows, "
                                 f"but its layout records {expected}")
        return tables

    def home(k: int) -> dict[str, int]:
        return {name: layout.home_rows(k, name) for name in own}

    def released() -> None:
        """The partition just released, freed now and given back to CUDA for PyTorch's allocations. Its tables
        sit in reference cycles (a sampler's), which only Python's cyclic collector frees: without collecting,
        a released partition stayed on the GPU until a collection happened to run (14 GiB on MBD)."""

        gc.collect()
        release_unused()

    if not read_ahead:
        for k in order:
            yield Partition(k, {**shared, **load(k)}, home(k))
            released()
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test.kumo_pytorch.reference_engine.gpu-read-ahead") as pool:
        pending = pool.submit(load, order[0]) if order else None
        for position, k in enumerate(order):
            tables = pending.result()
            pending = pool.submit(load, order[position + 1]) if position + 1 < len(order) else None
            yield Partition(k, {**shared, **tables}, home(k))
            del tables
            released()
