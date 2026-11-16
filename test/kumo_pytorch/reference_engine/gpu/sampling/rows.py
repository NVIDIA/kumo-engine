"""The rows a sample reads, and a call's shape, as the sampler holds them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
if TYPE_CHECKING:

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable


@dataclass
class _Column:
    """One column over selected rows.

    :param values: CuPy values (text: codes into `texts`); None for decimals.
    :param valid: CuPy bool, whether each value is non-NULL.
    :param stored: the (table, column) these values are some of, when they are.
    :param sort_keys: a callable giving the columns upstream orders these rows by (a table's primary key, a
        grouping's keys), aligned with them.
    :param words: a decimal's unscaled integers (`test.kumo_pytorch.reference_engine.gpu.data.decimals`), and `numbers` its float64 numbers.
    """

    dtype: Any
    values: Any
    valid: Any
    texts: Any = None
    stored: tuple[str, str] | None = None
    sort_keys: Any = None
    words: Any = None
    numbers: Any = None

    def take(self, order: Any) -> _Column:
        values = None if self.values is None else self.values[order]
        keys = self.sort_keys
        return _Column(self.dtype, values, self.valid[order], self.texts, self.stored,
                       None if keys is None else (lambda: [key.take(order) for key in keys()]),
                       None if self.words is None else self.words[order],
                       None if self.numbers is None else self.numbers[order])

class _TableRows:
    """Rows of one loaded table selected for anchors: (anchor, row) pairs, grouped by anchor."""

    def __init__(self, table: GpuTable, key: str, owner: Any, rows: Any) -> None:
        self.table, self.key, self.owner, self.rows = table, key, owner, rows

    def column(self, name: str, *, keyed: bool = True) -> _Column:
        import cupy

        table, dtype = self.table, self.table.dtypes[name]
        present = table.present(name)
        valid = cupy.ones(self.rows.size, dtype=bool) if present is None else present[self.rows]
        keys = (lambda: [self.column(self.key, keyed=False)]) if keyed else None
        if dtype.name.startswith("decimal"):  # its words and numbers, at these rows only
            from test.kumo_pytorch.reference_engine.gpu.data.decimals import gathered_words

            return _Column(dtype, None, valid, None, (table.name, name), keys,
                           gathered_words(table.columns[name], self.rows), table.numbers(name)[self.rows])
        values = table.values(name)[self.rows]
        return _Column(dtype, values, valid, table.dictionaries.get(name), (table.name, name), keys)

    def take(self, order: Any) -> _TableRows:
        return _TableRows(self.table, self.key, self.owner[order], self.rows[order])

class _GroupRows:
    """One row per group of a `GroupBy`, grouped by anchor: its keys and reduced values."""

    def __init__(self, owner: Any, columns: dict[str, _Column]) -> None:
        self.owner, self.columns = owner, columns

    def column(self, name: str) -> _Column:
        return self.columns[name]

    def take(self, order: Any) -> _GroupRows:
        return _GroupRows(self.owner[order], {name: column.take(order) for name, column in self.columns.items()})

@dataclass
class _Call:
    """One scope's values for a call: the named declarations they come from (the entity's values, or one
    event component's fields), the base rows they are read from (each anchor's entity row, or each event's
    row), each base row's anchor cutoff and (for events) its time, the queued checks, and what is worked out."""

    scope: Any
    table: str
    rows: Any
    cutoffs: Any
    flags: list[tuple[Any, Any]]
    times: Any = None
    sources: dict[str, Any] = field(default_factory=dict)
    computed: dict[str, Values] = field(default_factory=dict)
