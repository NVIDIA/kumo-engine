# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cutoff-aware stored-column fitting using Kumo's shared statistics format.

Kumo's learned stored-column encoders are fitted once per table row. Category
counts run on the GPU, reusing resident tables or streaming original sources.
Numerical statistics retain Kumo's native collector and bucket quantile grid;
that host pass uses temporary projections removed after fitting. Text training
also retains its host path. Partition halos never multiply fitting observations.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _batches(table, source, columns, cutoff):
    from test.kumo_pytorch.reference_engine.gpu.data.sources import (
        _physical_names,
    )

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from kumo.pytorch.contract.schema import table_clock
    from kumo.pytorch.dataset import _local_parquet_files

    clock = table_clock(table)
    names = list(dict.fromkeys([*columns, *([clock] if clock else [])]))
    physical = _physical_names(table.name, source, names)
    for file in _local_parquet_files(source.path):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=65536, columns=list(physical.values())):
            arrays = {}
            for name in names:
                values = batch.column(physical[name])
                dtype = table.columns[name].to_arrow()
                if pa.types.is_timestamp(dtype) and dtype.unit == 's':
                    # Source timestamps round upward; casting first would truncate
                    # fractional times and expose future rows at an earlier cutoff.
                    values = pc.ceil_temporal(values, unit='second')
                arrays[name] = pc.cast(values, dtype, safe=True)
            keep = None
            if clock is not None:
                keep = pc.is_valid(arrays[clock])
                if cutoff is not None:
                    # Kumo's reference statistics use a whole-second index clock.
                    times = pc.cast(pc.ceil_temporal(arrays[clock], unit='second'),
                                    pa.timestamp('s', tz='UTC'), safe=False)
                    keep = pc.and_(keep, pc.less_equal(times, pa.scalar(cutoff // 10**9,
                                                                        type=times.type)))
            selected = pa.RecordBatch.from_arrays([arrays[name] for name in columns], names=list(columns))
            yield selected if keep is None else selected.filter(keep)


def pin_table_encoders(contract, dataset, anchors, rows, encoder_cutoff=..., *, loaded=None):
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        reused_encoder,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.table_categories import (
        pin_categories,
    )

    import diskgraph
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.pytorch.column_stats import (
        EMPTY_STATS,
        LATEST_ANCHOR,
        QUANTILE_GRID,
        ColumnStats,
        cutoff_nanos,
        fit_from_stats,
        table_column_encoders,
    )
    from kumo.pytorch.contract.batch.encodings import Text, Vocabulary
    from kumo.pytorch.encoding import FittedEncoder, train_text
    from kumo.pytorch.materialization import _diskgraph_dtype

    candidates = table_column_encoders(contract)
    cutoff = LATEST_ANCHOR if encoder_cutoff is ... else encoder_cutoff
    if cutoff is LATEST_ANCHOR:
        cutoff = None
        if contract.requires_anchor_time:
            latest = rows[anchors.cutoffs[rows] == anchors.cutoffs[rows].max()]
            cutoff = max(cutoff_nanos(request.as_of_time) for start in range(0, int(latest.size), 65536)
                         for request in anchors.fetch_rows(latest[start:start + 65536]))
    else:
        cutoff = cutoff_nanos(cutoff)
    if not candidates:
        return {}, None
    tables = {table.name: table for table in contract.schema.tables}
    categorical = {identity: encoder for identity, encoder in candidates.items()
                   if isinstance(encoder.config, Vocabulary)}
    fitted = pin_categories(categorical, tables, dataset, cutoff, loaded or {})
    candidates = {identity: encoder for identity, encoder in candidates.items() if identity not in categorical}
    if not candidates:
        return fitted, {'cutoff': cutoff, 'col_stats': []}
    requests = sorted({request for encoder in candidates.values() for request in encoder.requests})
    columns: dict[Any, Any] = {}
    with tempfile.TemporaryDirectory(prefix='kumo-gpu-fit-') as directory:
        for number, name in enumerate(sorted({item[0] for item in requests})):
            requested = [item for item in requests if item[0] == name]
            names = sorted({item[1] for item in requested})
            table = tables[name]
            path = Path(directory) / f'{number}.parquet'
            schema = pa.schema([(column, table.columns[column].to_arrow()) for column in names])
            count = 0
            with pq.ParquetWriter(path, schema) as writer:
                for batch in _batches(table, dataset.tables[name], names, cutoff):
                    writer.write_batch(batch)
                    count += batch.num_rows
            if not count:
                columns.update({item: EMPTY_STATS for item in requested})
                continue
            output = Path(directory) / str(number)
            output.mkdir()
            diskgraph.write_column_stats_for_parquet(
                [str(path)],
                json.dumps([dict(name=column, **_diskgraph_dtype(pa, table.columns[column].to_arrow()))
                            for column in names]),
                json.dumps([[stype for _, column, stype in requested if column == name] for name in names]),
                str(output), quantiles=QUANTILE_GRID, timestamp_unit='seconds', num_threads=1,
            )
            stored = json.loads((output / 'col_stats.json').read_text())['anchors']
            columns.update({item: stored[item[1]] for item in requested})
    stats = ColumnStats(columns, {'value': None if cutoff is None else cutoff // 10**9, 'unit': 'seconds'})
    for identity, encoder in candidates.items():
        if isinstance(encoder.config, Text):
            def texts():
                for table, column in encoder.columns:
                    for batch in _batches(tables[table], dataset.tables[table], [column], cutoff):
                        yield [value for value in batch.column(0).to_pylist() if value is not None]
            state = FittedEncoder(identity=identity, config=encoder.config, dtype=encoder.dtype,
                                  tokenizer_json=train_text(encoder.config, texts()))
        else:
            state = fit_from_stats(identity, encoder, stats)
        fitted[identity] = reused_encoder(state)
    return fitted, {'cutoff': cutoff, 'col_stats': []}
