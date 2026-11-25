# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import Mock, patch

import pyarrow as pa

from kumo.pytorch import api
from kumo.pytorch.engines.node_values import computed_node_arrays
from tests.test_engine_drawn_candidates import POSITIVE, QUERY, data
from tests.test_scored_walks import walk_contract, write_parquet


def test_single_hop_counts_batch_distinct_coordinates_and_reuse_indexed_windows(tmp_path):
    schema, rows = data()
    rows["membership"].extend(dict(id=100 + i, pin=POSITIVE, board=10, at=11) for i in range(100))
    base = walk_contract(schema)
    past = api.Rows(("pin_links",), api.TimeWindow("at", stop=0))
    value = api.Aggregate(past, "count")
    fields = {
        "degree": value,
        "active": api.Compare(value, "greater_than", 1),
        "sum": api.Aggregate(past, "sum", "id"),
    }
    neighborhood = replace(base.sample.neighborhoods["graph"], node_values={"pins": fields})
    spec = replace(base, sample=replace(base.sample, neighborhoods={"graph": neighborhood}))
    dataset, _ = write_parquet(schema, rows, (api.SampleRequest(1, QUERY, 10),), tmp_path)
    with api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan:
        source = plan._source
        sampler = source._sampler
        pin = sampler.lookup_node_ids("pins", pa.array([POSITIVE]))[0]
        proxy = Mock(wraps=sampler)
        results = []

        def lengths(*args):
            result = sampler.query_temporal_lengths(*args)
            results.append(result)
            return result

        proxy.query_temporal_lengths.side_effect = lengths
        resolved = spec.resolve().sample.neighborhoods["graph"]
        with patch.object(source, "_sampler_context", proxy):
            values = computed_node_arrays(
                source,
                resolved,
                "pins",
                ("degree", "active", "sum"),
                [pin] * 4,
                [0, 1, 2, 0],
                [20, 10, 20],
            )
        by_time = [
            [r["id"] for r in rows["membership"] if r["pin"] == POSITIVE and r["at"] <= t]
            for t in (20, 10, 20, 20)
        ]
        assert values["degree"].to_pylist() == list(map(len, by_time))
        assert values["active"].to_pylist() == [True] * 4
        assert values["sum"].to_pylist() == list(map(sum, by_time))
        for query in (proxy.query_temporal_lengths, proxy.query_temporal):
            query.assert_called_once()
            assert query.call_args.args[0] == {"pins": [pin, pin]}
            assert query.call_args.args[1] == {"pins": [10, 20]}
        assert results[0]["stats"]["frames_counted"] == 1
        assert results[0]["stats"]["edges_visited"] == 0
        assert results[0]["stats"]["rowpack_timestamps"] == 0


def test_clockless_graph_needs_no_timestamp_anchor_column(tmp_path):
    schema, rows = data()
    schema = replace(
        schema,
        tables=tuple(replace(table, time_column=api.NON_TEMPORAL) for table in schema.tables),
    )
    spec = walk_contract(schema)
    base = spec.sample.neighborhoods["graph"]
    hood = replace(
        base,
        node_values={
            "pins": {"degree": api.Aggregate(api.Rows(("pin_links",)), "count")},
        },
    )
    graph = api.Input(
        api.NeighborhoodSource("graph"),
        api.Graph({"pins": api.Fields((api.Encode("degree", api.Raw()),))}),
        None,
    )
    spec = replace(
        spec,
        sample=replace(spec.sample, neighborhoods={"graph": hood}),
        batch=replace(spec.batch, inputs={"graph": graph}),
    )
    dataset, anchors = write_parquet(schema, rows, (api.SampleRequest(1, QUERY, 10),), tmp_path)
    with (
        api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=1)) as plan,
        plan.materialize(replace(anchors, as_of_time=None)) as task,
    ):
        batch = next(task.batches(output_format="columnar"))
        pins = batch.inputs["graph"].nodes["pins"]
        assert pins.features["degree"].to_pylist() == [
            sum(row["pin"] == node for row in rows["membership"]) for node in pins.ids.to_pylist()
        ]
        assert len(pins.ids) >= 1
