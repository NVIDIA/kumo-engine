# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasets for the package end-to-end flows."""


def bind_tables(directory, contract, root_key, targets=()):
    """Resolve a prepared Parquet dataset with its registered anchors."""
    from kumo.pytorch import api

    def source(name):
        return api.Parquet(str(directory / f"{name}.parquet"))

    dataset = api.Dataset({table.name: source(table.name) for table in contract.schema.tables})
    anchors = api.AnchorTable(
        source("anchors"),
        root_key,
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={name: name for name in targets},
        metadata=("registered_entity_id", "prediction_time_ns", "anchor_row_position"),
    )
    return contract.resolve(dataset=dataset), dataset, anchors
