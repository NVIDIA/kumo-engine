# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded T01 weekly session counts from viewer and session Parquet files."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from e2e.datasets import bind_tables
from e2e.datasets.staging import verify
from kumo.pytorch import api

T01_MANIFEST = Path(__file__).parent / "pins" / "streaming-25b.json"


def load_streaming(root):
    """Project the two verified T01 viewers and their forecast and historical graph rows."""
    files = verify(root, T01_MANIFEST)
    selected = [("viewer_id", "in", [3, 873])]
    # Viewer 3 has no earlier history. Include all ties at viewer 873's 32nd graph second.
    history = [*selected, ("start_time", ">", datetime(2025, 8, 18, 10, 43, 18))]
    columns = [
        "session_id",
        "viewer_id",
        "start_time",
        "watch_seconds",
        "completion_frac",
        "device_type",
    ]
    return (
        pq.read_table(files["viewers"], columns=["viewer_id", "signup_time"], filters=selected),
        pq.read_table(files["sessions"], columns=columns, filters=history, use_threads=False),
    )


def prepare_streaming(directory, cohort_size):
    """Bind a bounded viewer cohort to its three weekly forecast windows."""
    first, week = datetime(2025, 12, 10), timedelta(weeks=1)
    viewers = pq.read_table(
        directory / "viewers.parquet", columns=["viewer_id"], filters=[("signup_time", "<", first)]
    )
    cohort = viewers["viewer_id"].sort().slice(0, cohort_size).to_pylist()
    if cohort_size < 1 or len(set(cohort)) != cohort_size:
        raise ValueError("The cohort requires enough distinct eligible viewers")

    anchors = []
    for i, split in enumerate(("train", "valid", "test")):
        start = first + i * week
        owners = pq.read_table(
            directory / "sessions.parquet",
            columns=["viewer_id"],
            filters=[
                ("viewer_id", "in", cohort),
                ("start_time", ">=", start),
                ("start_time", "<", start + week),
            ],
        )
        counts = Counter(owners["viewer_id"].to_pylist())
        for viewer in cohort:
            anchors.append(
                dict(
                    viewer_id=viewer,
                    sample_id=f"{split}:{viewer}",
                    split=split,
                    as_of_time=start - timedelta(seconds=1),
                    future_count=counts[viewer],
                    registered_entity_id=str(viewer),
                    anchor_row_position=len(anchors),
                    prediction_time_ns=int(start.replace(tzinfo=UTC).timestamp()) * 10**9,
                )
            )
    rows = pa.Table.from_pylist(anchors)
    pq.write_table(
        rows.append_column("count", rows["future_count"].cast(pa.float32(), safe=False)),
        directory / "anchors.parquet",
    )

    relation = api.RelationSpec("viewers_sessions", "viewers", "viewer_id", "sessions", "viewer_id")
    schema = api.DataSchema(
        (
            api.TableSpec("viewers", "viewer_id", time_column="signup_time"),
            api.TableSpec("sessions", "session_id", time_column="start_time"),
        ),
        (relation,),
    )
    layer = api.GraphLayer((api.GraphSample(relation.name, 32, time_column="start_time"),))
    sample = api.SampleSpec(
        api.EntitySpec("viewers", entity_type="viewer"),
        neighborhoods={"history_graph": api.NeighborhoodSpec((layer,))},
        targets={"count": api.ExternalTarget(api.DType("float32"))},
    )
    fields = api.Fields(
        (
            api.Encode("watch_seconds", api.ZScore()),
            api.Encode("completion_frac", api.ZScore()),
            api.Encode("device_type", api.Vocabulary()),
        )
    )
    graph = api.Input(
        api.NeighborhoodSource("history_graph"), api.Graph({"sessions": fields}), None
    )
    batch = api.BatchSpec(
        inputs={"graph": graph}, targets={"count": api.LabelTarget("count", api.Raw())}
    )
    return bind_tables(
        directory, api.DataLoaderSpec(schema, sample, batch), "viewer_id", targets=("count",)
    )
