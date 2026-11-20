# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation-time native sampling compatibility checks."""
from ._phase import phase
from .bundle import quote


def validate_contract(workload):
    """Preparation-time eligibility gate shared by all engines; no hidden fallback.

    Null sort keys are admitted by the upstream sampler, and subsecond clocks
    cannot be indexed exactly. Reject those destinations before any measurement.
    Fractional non-clock features remain supported by the existing ingest bridge.
    """
    destinations = {workload.bundle.endpoint(s)[1] for layer in workload.layers for s in layer
                    if s['fanout'] != 0}
    bundle = workload.bundle
    import duckdb  # Optional frozen-bundle benchmark dependency.

    with phase('multihop_contract_validation', 'preparation'):
        with duckdb.connect(config={'threads': 2, 'memory_limit': '1GB'}) as con:
            for name in sorted(destinations):
                table = bundle.tables[name]
                tc = table.get('time_column')
                if not tc or not table['rows']:
                    continue
                con.read_parquet(bundle.paths(name)).create_view('clock_source')
                nulls, fractional, low, high = con.execute(
                    f'SELECT count(*) - count({quote(tc)}), '
                    f'count(*) FILTER (WHERE epoch_us({quote(tc)}) % 1000000 != 0), '
                    f'min(epoch_us({quote(tc)})), max(epoch_us({quote(tc)})) FROM clock_source'
                ).fetchall()[0]
                if nulls or fractional or (high - low) // 1000000 > 2 * (2**31 - 1):
                    raise ValueError(f'native_multihop requires non-null, whole-second, indexable clocks: {name}.{tc}; '
                                     'use temporal_query or local_native for exact fallbacks')
    for layer in workload.layers:
        for step in layer:
            total = bundle.tables[bundle.endpoint(step)[1]]['rows']
            fanout = total if step['fanout'] is None else min(total, step['fanout'])
            if step['direction'] == 'forward':
                fanout = min(fanout, 1)
            if fanout > 2**32 - 1:
                raise ValueError('native_multihop fanout exceeds uint32; use a bounded fanout')
