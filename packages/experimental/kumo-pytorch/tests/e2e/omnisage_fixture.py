# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Parquet inputs and public DiskGraph task for the OmniSage loader comparison."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch import api
from tests.test_scored_walks import star

KEEP = {"pins": 8, "boards": 4}
RESTART = 0.35
WALK_BUDGET = 1024
DRAW_COUNT = 2
FUTURE_WINDOW = 4
CANDIDATE_NAMES = ("supplied", "scored", "future")


@dataclass
class Workload:
    schema: api.DataSchema
    dataset: api.Dataset
    anchors: api.AnchorTable
    paths: dict[str, Path]
    width: int


def make_fixture(root, *, pins, anchors, width, client=None, component_pins=8):
    """Persist the oracle inputs once. Upload identical bytes when an S3 client is supplied."""
    if pins < 16 or min(anchors, width) < 1:
        raise ValueError("supply at least 16 pins, one anchor, and a positive array width")
    if component_pins < 1:
        raise ValueError("component_pins must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base, _ = star()
    integer = api.DType("int64")
    clock = api.DType("timestamp", unit="s", timezone="UTC")
    vectors_by_table = {
        "pins": {"image": api.DType("float32", shape=(width,))},
        "boards": {"annotation": api.DType("float32", shape=(min(128, width),))},
    }
    schema = replace(
        base,
        tables=(
            *(
                replace(table, columns={**table.columns, **vectors_by_table[table.name]})
                if table.name in vectors_by_table
                else table
                for table in base.tables
            ),
            api.TableSpec(
                "actions",
                "id",
                {"id": integer, "actor": integer, "pin": integer, "at": clock},
                time_column="at",
            ),
        ),
        relations=(*base.relations, api.RelationSpec("acted", "pins", "id", "actions", "actor")),
    )
    ids = np.arange(pins, dtype=np.int64)
    # Distinct, bounded float32 values retain exact Parquet/NumPy/Torch parity.
    vectors = ((ids[:, None] * 13 + np.arange(width) * 7) % 1021).astype(np.float32) / 1024
    tables = {
        "pins": pa.table(
            {
                "id": ids,
                "image": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), width),
            }
        )
    }
    # A late-only board and an isolated-at-all-anchor-times pin reveal hindsight leakage.
    boards_per_component = max(5, component_pins // 3)
    sentinel = boards_per_component * ((pins + component_pins - 1) // component_pins)
    board_ids = np.arange(sentinel + 1)
    board_width = min(128, width)
    annotation = ((board_ids[:, None] * 19 + np.arange(board_width) * 11 + 1) % 997).astype(
        np.float32
    ) / 1024
    tables["boards"] = pa.table(
        {
            "id": board_ids,
            "annotation": pa.FixedSizeListArray.from_arrays(
                pa.array(annotation.ravel()), board_width
            ),
        }
    )
    events = []
    actions = []
    for pin in range(pins):
        if pin != pins - 1:
            base_board = boards_per_component * (pin // component_pins)
            for offset, at in enumerate((2, 8, 12, 16, 22)):
                board = base_board + (pin + offset) % boards_per_component
                events.append((len(events), pin, board, at))
            # At each cutoff there are three distinct eligible IDs, including one at
            # the inclusive upper boundary, alongside repeats and ineligible sentinels.
            for cutoff in (8, 16, 24):
                for delta, shift in ((0, 5), (1, 1), (2, 1), (3, 2), (4, 3), (5, 7)):
                    actions.append((len(actions), pin, (pin + shift) % pins, cutoff + delta))
        events.append((len(events), pin, sentinel, 28))
    for name, keys, rows in (
        ("membership", ("id", "pin", "board", "at"), events),
        ("actions", ("id", "actor", "pin", "at"), actions),
    ):
        tables[name] = pa.table(
            {
                key: pa.array(values, type=clock.to_arrow() if key == "at" else pa.int64())
                for key, values in zip(keys, zip(*rows, strict=True), strict=True)
            }
        )
    order = np.random.default_rng(7321).permutation(anchors)
    root_ids = ((order // 3) * 131) % pins
    root_ids[(order // 3) % 19 == 1] = pins - 1
    cutoffs = np.asarray((8, 16, 24))[order % 3]
    positives = []
    for sid, pin in zip(order.tolist(), root_ids.tolist(), strict=True):
        group = [(pin + 1) % pins, (pin + 2) % pins]
        if sid % 17 == 0:
            group = []
        elif sid % 11 == 0:
            group = group[:1]
        positives.append([{"pin": candidate} for candidate in group])
    tables["anchors"] = pa.table(
        {
            "sid": order,
            "entity": root_ids,
            "at": pa.array(cutoffs, type=clock.to_arrow()),
            "positives": pa.array(positives, type=pa.list_(pa.struct([("pin", pa.int64())]))),
        }
    )
    paths = {}
    sources = {}
    for name, table in tables.items():
        path = root / f"{name}.parquet"
        pq.write_table(table, path)
        paths[name] = path
        if client is None:
            uri = str(path)
        else:
            key = f"combined/{path.name}"
            client.upload_file(str(path), "omnisage-fixture", key)
            uri = f"s3://omnisage-fixture/{key}"
        sources[name] = api.Parquet(uri)
    binding = api.AnchorTable(
        sources.pop("anchors"),
        "entity",
        "at",
        sample_id="sid",
        candidates={"supplied": api.CandidateColumn("positives", "pin")},
    )
    return Workload(schema, api.Dataset(sources), binding, paths, width)


def spec(workload, batch_size):
    """Use the same scored walk, feature projections, and cutoff semantics in all four graphs."""
    rows = (("pins", "pin_links"), ("boards", "board_links"))
    walk = api.WalkSample("walk", ("membership",), KEEP, RESTART, WALK_BUDGET)
    hood = api.NeighborhoodSpec(
        (api.GraphLayer((walk,)),),
        node_values={
            table: {
                "degree": api.Aggregate(api.Rows((link,), api.TimeWindow("at", stop=0)), "count")
            }
            for table, link in rows
        },
    )
    root = api.EntitySpec("pins")
    graphs = {"graph": hood}
    draws = {
        "supplied": None,
        "scored": api.DrawNeighbors("graph", ("walk.pins",), count=DRAW_COUNT, weight="score"),
        "future": api.DrawRows(
            api.Rows(("acted",), api.TimeWindow("at", start=0, stop=FUTURE_WINDOW)),
            column="pin",
            count=DRAW_COUNT,
        ),
    }
    rep = api.Graph(
        {
            "pins": api.Fields(
                tuple(api.Encode(field, api.Raw()) for field in ("degree", "image"))
            ),
            "boards": api.Fields(
                tuple(api.Encode(field, api.Raw()) for field in ("degree", "annotation"))
            ),
        }
    )
    return api.DataLoaderSpec(
        workload.schema,
        api.SampleSpec(
            root,
            neighborhoods=graphs,
            candidate_sets={
                name: api.CandidateSetSpec(root, neighborhoods=graphs, draw=draw)
                for name, draw in draws.items()
            },
        ),
        api.BatchSpec(
            inputs={
                "query": api.Input(api.NeighborhoodSource("graph"), rep, None),
                **{
                    name: api.Input(api.NeighborhoodSource("graph", candidates=name), rep, None)
                    for name in draws
                },
            },
            batching=api.Batching(max_samples=batch_size),
        ),
    )
