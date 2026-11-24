# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connector → snapshot → retained FM Arrow batch → local SDM prediction.

Run from this directory after installing the matching engine/native wheels and
the optional SDM environment. See README.md for the checkpoint and dataset.
"""
from datetime import datetime, timezone
from pathlib import Path

from fixture import write_fixture
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo import fm
from kumo.pytorch import api


def build_index_and_load_batch(root="scratch/connector-to-prediction"):
    """Write example Parquet files, build or reuse the index, and load an FM batch.

    Configures engine storage under root/producer and retains files for reuse.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    kumo.configure(scratch_dir=root / "producer")
    source = kumo.FileConnector(root_dir=str(write_fixture(root / "source")))
    graph = kumo.Graph(
        tables={
            "users": kumo.Table.from_source_table(
                source["users.parquet"], primary_key="user_id"),
            "orders": kumo.Table.from_source_table(
                source["orders.parquet"], primary_key="order_id",
                time_column="order_time").with_column("user_id", stype=Stype.ID),
        },
        edges=[kumo.Edge(src_table="orders", fkey="user_id", dst_table="users")],
    )
    engine = api.DiskGraphEngine.from_snapshot(
        graph.snapshot(), num_threads=2)
    relation = engine.schema.relations[0].name

    horizon = 30 * 24 * 60 * 60
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        targets={"next_orders": api.Compare(
            api.Aggregate(api.Rows(path=(relation,), time=api.TimeWindow(
                "order_time", start=0, stop=horizon)), "count"),
            "greater_than", 0)},
        neighborhoods={"history": api.NeighborhoodSpec(layers=(
            api.GraphLayer(samples=(api.GraphSample(
                relation=relation, direction="reverse", strategy="recent", fanout=8),)),
        ))},
    )

    with fm.prepare(engine=engine, sample=sample, target="next_orders",
                    task="binary_classification", classes={False: "no", True: "yes"},
                    features={"users": ["user_id"],
                              "orders": ["order_id", "order_time"]}) as loader:
        batch = loader.load(
            context=fm.Context(entities=range(32), times=[
                datetime(2025, 6, 1, tzinfo=timezone.utc)]),
            query=fm.Query(entities=range(8),
                           at=datetime(2025, 11, 1, tzinfo=timezone.utc)),
        )
    return batch


def main():
    import torch
    from adapter import arrow_to_sdm
    from sdm.models import KumoRelational

    batch = build_index_and_load_batch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {name: value.to(device) for name, value in arrow_to_sdm(batch).items()}
    model = KumoRelational(task="classification", device=device).eval()
    with torch.inference_mode():
        model.fit(inputs["x_context"], inputs["y_context"],
                  inputs["related_context_tables"], num_estimators=1, num_hops=2,
                  generator=torch.Generator(device=device).manual_seed(42))
        predictions = model.predict(inputs["x_query"], inputs["related_query_tables"])
    print(predictions.to_arrow())
    print("Same retained batch, HTTP bytes:", len(batch.encode("arrays")),
          len(batch.encode("arrow_ipc")))


if __name__ == "__main__":
    main()
