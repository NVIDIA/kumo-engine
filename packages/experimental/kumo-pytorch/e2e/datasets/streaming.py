# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T01 weekly session counts from the compact extract or full Streaming source."""

import hashlib
import json
import struct
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e2e.datasets import bind_tables
from e2e.datasets.staging import verify
from kumo.pytorch import api

T01_MANIFEST = Path(__file__).parent / "pins" / "streaming-25b.json"
FIRST_WEEK = datetime(2025, 12, 10)


def load_streaming(root):
    """Load the verified T01 viewer and session extracts."""
    files = verify(root, T01_MANIFEST)
    return pq.read_table(files["viewers"]), pq.read_table(files["sessions"])


def weekly_counts(viewers, sessions, cohort_size):
    """Reduce the source files to the lowest eligible IDs and their exact weekly counts."""
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    ids = pa.array([], type=pa.int64())
    owner = ds.field("viewer_id")
    for fragment in ds.dataset(viewers, format="parquet").get_fragments():
        eligible = ds.field("signup_time") < FIRST_WEEK
        if len(ids) == cohort_size:
            eligible &= owner <= pc.max(ids).as_py()
        for batch in fragment.to_batches(columns=["viewer_id"], filter=eligible):
            candidates = pa.concat_arrays([ids, batch.column(0)])
            ids = candidates.take(pc.bottom_k_unstable(candidates, cohort_size))
    cohort = ids.sort().to_pylist()
    if len(set(cohort)) != cohort_size:
        raise ValueError("The cohort requires enough distinct eligible viewers")
    events = ds.dataset(sessions, format="parquet")
    clock = ds.field("start_time")
    # The explicit ID range also permits row-group pruning on ordered source shards.
    owners = (owner >= cohort[0]) & (owner <= cohort[-1]) & owner.isin(ids)
    counts = []
    for i in range(3):
        start = FIRST_WEEK + timedelta(weeks=i)
        weekly = Counter()
        for batch in events.to_batches(
            columns=["viewer_id"],
            filter=owners & (clock >= start) & (clock < start + timedelta(weeks=1)),
        ):
            weekly.update(batch.column(0).to_pylist())
        counts.append(weekly)
    return cohort, counts


def prepare_streaming(directory, cohort_size, *, source_manifest=None, replay_viewers=None):
    """Bind T01 using either the two local files or all seven verified source directories."""
    schema = api.DataSchema(
        (
            api.TableSpec("viewers", "viewer_id", time_column="signup_time"),
            api.TableSpec("sessions", "session_id", time_column="start_time"),
        ),
        (api.RelationSpec("viewers_sessions", "viewers", "viewer_id", "sessions", "viewer_id"),),
    )
    paths = {t.name: directory / f"{t.name}.parquet" for t in schema.tables}
    if source_manifest is not None:
        tables = json.loads(source_manifest.read_text())["schema"]["tables"]
        relations = []
        for name, table in tables.items():
            for key in table["foreign_keys"]:
                parent, column = key["references"].split(".")
                relations.append(
                    api.RelationSpec(f"{parent}_{name}", parent, column, name, key["column"])
                )
        schema = api.DataSchema(
            tuple(
                api.TableSpec(n, t["primary_key"], time_column=t["time_column"] or api.NON_TEMPORAL)
                for n, t in tables.items()
            ),
            tuple(relations),
        )
        paths = {name: source_manifest.parent / name for name in tables}
    cohort, counts = weekly_counts(paths["viewers"], paths["sessions"], cohort_size)
    if source_manifest is not None:
        digest = hashlib.sha256()
        for viewer in cohort:
            digest.update(struct.pack("<4q", viewer, *(week[viewer] for week in counts)))
        expected = json.loads(T01_MANIFEST.read_text())["viewer_weekly_counts_sha256"]
        if digest.hexdigest() != expected:
            raise ValueError("Streaming full-cohort weekly counts differ from the pinned digest")
    if replay_viewers is not None:
        if not set(replay_viewers).issubset(cohort):
            raise ValueError("Replay viewers must belong to the verified cohort")
        cohort = replay_viewers
    anchors = []
    for i, split in enumerate(("train", "valid", "test")):
        start = FIRST_WEEK + timedelta(weeks=i)
        for viewer in cohort:
            anchors.append(
                dict(
                    viewer_id=viewer,
                    sample_id=f"{split}:{viewer}",
                    split=split,
                    as_of_time=start - timedelta(seconds=1),
                    future_count=counts[i][viewer],
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

    layer = api.GraphLayer((api.GraphSample("viewers_sessions", 32, time_column="start_time"),))
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
        directory,
        api.DataLoaderSpec(schema, sample, batch),
        "viewer_id",
        targets=("count",),
        source_paths=paths,
    )
