"""Exact stored-column category counts on the GPU, over original table rows.

Resident tables are reused. Other tables are read in bounded chunks from their
original sources, so partition halos never contribute observations. Intermediate
storage scales with distinct keys; the read limit is not a total fitting-memory
limit. Vocabulary finalization copies distinct keys to the host, using the
existing canonical float ordering and fitted-state format.
"""
from __future__ import annotations


CHUNK_BYTES = 64 * 1024 * 1024


class CategoryCounts:
    """Distinct typed GPU keys and int64 counts, merged before vocabulary limits."""

    def __init__(self):
        self.keys = self.counts = None

    def add(self, keys, counts):
        import pylibcudf as plc

        if keys is None or not keys.size():
            return
        if self.keys is None:
            self.keys, self.counts = keys, counts
            return
        keys = plc.concatenate.concatenate([self.keys, keys])
        counts = plc.concatenate.concatenate([self.counts, counts])
        grouped, results = plc.groupby.GroupBy(plc.Table([keys])).aggregate([
            plc.groupby.GroupByRequest(counts, [plc.aggregation.sum()])])
        self.keys, self.counts = grouped.columns()[0], results[0].columns()[0]

    def encoder(self, identity, config, dtype):
        import cupy
        import pylibcudf as plc
        from test.kumo_pytorch.reference_engine.gpu.data.text import is_text, text_column
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view, counted_vocabulary

        if self.keys is None:
            keys, counts = cupy.empty(0, dtype=cupy.int64), cupy.empty(0, dtype=cupy.int64)
            texts = text_column([]) if is_text(dtype) else None
        elif is_text(dtype):
            # Text codes must follow canonical byte order for frequency ties.
            ordered = plc.sorting.sort(plc.Table([self.keys, self.counts]),
                                       [plc.types.Order.ASCENDING] * 2, [plc.types.NullOrder.AFTER] * 2)
            texts, frequencies = ordered.columns()
            keys, counts = cupy.arange(texts.size(), dtype=cupy.int32), _view(frequencies)
        else:
            keys, counts, texts = _view(self.keys), _view(self.counts), None
        return counted_vocabulary(identity, config, dtype, keys, counts, texts)


def _clock_mask(table, clock, cutoff):
    import cupy

    if clock is None:
        return None
    keep = table.present(clock)
    if cutoff is not None:
        times = table.values(clock)
        if table.dtypes[clock].name == "date32":
            times = times.astype(cupy.int64) * 86400
        before = times <= cutoff // 10**9
        keep = before if keep is None else keep & before
    return keep


def _column_counts(table, name, keep):
    import cupy
    import pylibcudf as plc
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

    valid = table.present(name)
    selected = valid if keep is None else keep if valid is None else keep & valid
    column = table.columns[name]
    if selected is not None:
        column = plc.stream_compaction.apply_retention_mask(plc.Table([column]), _column(selected)).columns()[0]
    if table.dtypes[name].name.startswith("float"):
        from test.kumo_pytorch.reference_engine.gpu.data.keys import widened

        values = widened(_view(column))
        if bool((~cupy.isfinite(values)).any()):
            raise ValueError(f"{table.dtypes[name].name} vocabulary values must be finite")
        column = _column(cupy.where(values == 0, values.dtype.type(0), values))
    grouped, results = plc.groupby.GroupBy(plc.Table([column]), plc.types.NullPolicy.EXCLUDE).aggregate([
        plc.groupby.GroupByRequest(column, [plc.aggregation.count()])])
    keys = grouped.columns()[0]
    if name in table.dictionaries:
        keys = plc.copying.gather(plc.Table([table.dictionaries[name]]), keys,
                                  plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    counts = plc.unary.cast(results[0].columns()[0], plc.DataType(plc.TypeId.INT64))
    return keys, counts


def count_columns(table, source, names, cutoff, *, loaded=None):
    from kumo.pytorch.contract.schema import table_clock
    from test.kumo_pytorch.reference_engine.gpu.data.tables import read_chunks

    clock = table_clock(table)
    columns = {name: table.columns[name] for name in dict.fromkeys([*names, *([clock] if clock else [])])}
    parts = (loaded,) if loaded is not None else read_chunks(table.name, source, columns, chunk_bytes=CHUNK_BYTES)
    result = {name: CategoryCounts() for name in names}
    for part in parts:
        keep = _clock_mask(part, clock, cutoff)
        for name, counts in result.items():
            counts.add(*_column_counts(part, name, keep))
    return result


def pin_categories(candidates, tables, dataset, cutoff, loaded):
    columns = {}
    requested = {column for encoder in candidates.values() for column in encoder.columns}
    for name in sorted({table for table, _ in requested}):
        names = sorted(column for table, column in requested if table == name)
        counts = count_columns(tables[name], dataset.tables[name], names, cutoff, loaded=loaded.get(name))
        columns.update({(name, column): value for column, value in counts.items()})
    fitted = {}
    for identity, encoder in candidates.items():
        merged = CategoryCounts()
        for column in encoder.columns:
            counts = columns[column]
            merged.add(counts.keys, counts.counts)
        fitted[identity] = merged.encoder(identity, encoder.config, encoder.dtype)
    return fitted
