# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original-row correctness checks for direct Parquet graph/count bindings."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow.dataset as ds

from .bindings import contract_for, dataset_definition, graph_layout, profile
from .data import schema_for
from .feature_checks import FeatureChecks
from .graph_features import graph_oracle
from .population import check_delivery
from .population_anchors import projected_rows
from .targets import (
    source_target_rows,
    verify_stored_batch,
    verify_target_projection,
)


def epoch(value):
    return value.replace(tzinfo=UTC).timestamp()


def count_graph_features(binding, projection, mapping, state, *, mode):
    import kumo.pytorch.api as r

    definition = dataset_definition(binding)
    spec = definition["anchor_generation"]
    inputs = profile(binding, "input")
    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    ids = sorted({row["entity_id"] for row in rows})
    root = spec["root_table"]
    key = definition["tables"][root]["key"]
    events = spec["event_table"]
    sources = {}
    for table, column in [(root, key), (events, spec["event_root"])]:
        source = ds.dataset(
            [f["path"] for f in mapping["sources"][table]], format="parquet"
        )
        sources[table] = source.to_table(filter=ds.field(column).isin(ids)).to_pylist()
    roots = {row[key]: row for row in sources[root]}
    assert len(roots) == len(sources[root]) and set(roots) == set(ids)
    originals = source_target_rows(binding, mapping["anchor_source"]["path"], rows)
    for row in rows:
        original = originals[row["anchor_row_position"]]
        verify_target_projection(binding, row, original)
        assert str(row["entity_id"]) == row["registered_entity_id"]
        start = row["prediction_time_ns"] / 10**9
        declared = datetime.fromisoformat(spec["prediction_times"][row["split"]])
        assert start == declared.timestamp(), (
            "Anchor prediction clock differs from its declared split"
        )
        assert epoch(roots[row["entity_id"]][spec["root_time"]]) < start
        assert row["as_of_time"] == int(start) - 1
        count = sum(
            event[spec["event_root"]] == row["entity_id"]
            and start
            <= epoch(event[spec["event_time"]])
            < start + spec["window_seconds"]
            for event in sources[events]
        )
        assert original["count"] == row["original_count"] == count, (
            f"{row['sample_id']}: stored count {original['count']} differs from original rows {count}"
        )
    schema = schema_for(
        r.Dataset({n: r.Parquet(p) for n, p in mapping["table_paths"].items()}), binding
    )
    contract = contract_for(schema, binding, max_events=inputs["sampling_limit"])
    oracle = graph_oracle()
    source = oracle.SourceRows(schema, sources)

    def graph(batch):
        requests = [
            SimpleNamespace(
                sample_id=s,
                entity_id=anchors[s]["entity_id"],
                as_of_time=anchors[s]["as_of_time"],
            )
            for s in batch.metadata["sample_ids"]
        ]
        oracle.assert_graph(
            batch.inputs["graph"], contract, "graph", requests, source, state
        )

    return FeatureChecks(
        binding["feature_checks"],
        mode,
        {
            "identity_cutoff": lambda batch: check_delivery(
                batch,
                iter(anchors[s] for s in batch.metadata["sample_ids"]),
                graph_clocks={
                    n: g["timestamp"]
                    for n, g in graph_layout(inputs)[0].items()
                    if g.get("timestamp")
                },
            ),
            "graph_values": graph,
            "stored_targets": lambda batch: verify_stored_batch(
                binding,
                batch,
                [anchors[s] for s in batch.metadata["sample_ids"]],
                state,
            ),
        },
        {row["sample_id"]: row["split"] for row in rows},
    )
