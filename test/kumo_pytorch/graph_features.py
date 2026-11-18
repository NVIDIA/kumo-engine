# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse package graph semantics against independently reconstructed source rows."""

import importlib.util
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

from .bindings import contract_for, dataset_definition, graph_layout, profile


@lru_cache(maxsize=1)
def graph_oracle():
    path = (
        Path(__file__).resolve().parents[2]
        / "packages/experimental/kumo-pytorch/tests/graph_semantics.py"
    )
    spec = importlib.util.spec_from_file_location("_kumo_test_graph_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_check(binding, mapping, anchors, events, state, *, root_rows=None):
    import kumo.pytorch.api as r

    from .data import schema_for

    inputs = profile(binding, "input")
    if root_rows is None:
        root = dataset_definition(binding)["source_options"]["root"]
        selected = {row["entity_id"] for row in anchors.values()}
        roots = {}
        for source in mapping["sources"][root["table"]]:
            for batch in pq.ParquetFile(source["path"]).iter_batches(
                columns=[root["key"], root["time"]]
            ):
                for row in batch.to_pylist():
                    if row[root["key"]] in selected:
                        assert row[root["key"]] not in roots
                        roots[row[root["key"]]] = row[root["time"]]
        assert set(roots) == selected
        offset = profile(binding, "anchor").get("root_time_offset_seconds")
        if offset is not None:
            for anchor in anchors.values():
                assert (
                    anchor["prediction_time_ns"]
                    == (roots[anchor["entity_id"]] + offset) * 10**9
                ), "Prediction clock disagrees with original root availability"
        stamp = lambda value: datetime.fromtimestamp(value, timezone.utc)
        tables = {
            inputs["root_table"]: [
                {"root_id": key, "available_at": stamp(clock)}
                for key, clock in roots.items()
            ],
            **{name: [] for name in graph_layout(inputs)[0]},
        }
    else:
        tables = {
            inputs["root_table"]: root_rows,
            **{name: [] for name in graph_layout(inputs)[0]},
        }
        stamp = lambda value: datetime.fromtimestamp(value, timezone.utc)
    for key, rows in events.items():
        for event in rows:
            if event["family"] in tables:
                tables[event["family"]].append(
                    {
                        **event["values"],
                        "event_id": event["mapped_id"],
                        "source_id": event["source_id"],
                        "root_id": key,
                        "event_time": stamp(event["time"]),
                        "event_time_seconds": event["time"],
                    }
                )
    # Prepared schemas describe types only; all answer values came from originals.
    schema = schema_for(
        r.Dataset(
            {name: r.Parquet(path) for name, path in mapping["table_paths"].items()}
        ),
        binding,
    )
    contract = contract_for(
        schema, binding, max_events=max(1, max(map(len, events.values()), default=0))
    )
    oracle = graph_oracle()
    source = oracle.SourceRows(schema, tables)

    def verify(batch):
        requests = [
            SimpleNamespace(
                sample_id=sample,
                entity_id=anchors[sample]["entity_id"],
                as_of_time=anchors[sample]["as_of_time"],
            )
            for sample in batch.metadata["sample_ids"]
        ]
        oracle.assert_graph(
            batch.inputs["graph"], contract, "graph", requests, source, state
        )

    return verify
