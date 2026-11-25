# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic synthetic tables demonstrating one mapping for all recipe modes."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from tfm.config import DataConfig
from tfm.contracts.relational import RelationalDataset, RelationalForeignKey, RelationalTable
from tfm.data.adapters.registry import Adapter
from tfm.data.adapters.result import AdapterResult


def load(
    config: DataConfig, *, seed: int, flat_fields: bool = False, fixture_dir: Path | None = None
) -> AdapterResult:
    """Generate a deterministic, public fixture; IDs are metadata, never features."""
    del seed, flat_fields, fixture_dir
    if config.n_entities < 8:
        raise ValueError("The relational synthetic example needs at least eight entities")
    origin = datetime(2024, 1, 1, tzinfo=UTC)
    entities, containers, activity, labels = [], [], [], []
    for i in range(config.n_entities):
        owner = f"entity{i}"
        group = f"container{i}"
        split = (
            "train"
            if i < config.n_entities // 2
            else ("validation" if i < config.n_entities * 3 // 4 else "test")
        )
        cutoff = origin + timedelta(days={"train": 10, "validation": 11, "test": 12}[split])
        entities.append({"id": owner})
        containers.append(
            {"id": group, "owner": owner, "created": origin, "known": origin + timedelta(days=1)}
        )
        for j, day in enumerate((2, 4, 12, 13)):
            activity.append(
                {
                    "id": f"activity{i}-{j}",
                    "container": group,
                    "at": origin + timedelta(days=day),
                    "known": origin + timedelta(days=day),
                    "value": float((i + 1) * (j + 1)),
                }
            )
        if config.load_labels:
            labels.append(
                {
                    "id": f"label{i}",
                    "owner": owner,
                    "cutoff": cutoff,
                    "outcome": float(i % 2),
                    "available": cutoff + timedelta(days=3),
                    "split": split,
                }
            )
    return AdapterResult(
        relational=RelationalDataset(
            (
                RelationalTable("entities", ("id",), immutable=True),
                RelationalTable("containers", ("id",), event_time="created", available_at="known"),
                RelationalTable(
                    "activity",
                    ("id",),
                    numeric={"amount": "value"},
                    event_time="at",
                    available_at="known",
                ),
                RelationalTable("outcomes", ("id",), immutable=True, role="labels"),
            ),
            (
                RelationalForeignKey("owns", "entities", "containers", ("owner",)),
                RelationalForeignKey("activity", "containers", "activity", ("container",)),
            ),
            {
                "entities": tuple(entities),
                "containers": tuple(containers),
                "activity": tuple(activity),
                "outcomes": tuple(labels),
            },
            observed_through=origin + timedelta(days=30),
        )
    )


adapter = Adapter(load)
