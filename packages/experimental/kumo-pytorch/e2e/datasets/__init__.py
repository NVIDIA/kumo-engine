# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasets for the package end-to-end flows."""

from dataclasses import replace

CALENDAR = {"hour": range(24), "weekday": range(7), "day": range(1, 32)}


def history_batch(events, encoders, capacity, *, masked=False, max_tokens=None):
    """Build the masked-field or next-token batch for a complete event history."""
    from kumo.pytorch import api

    source = api.EventSource(tuple(events))
    roles = {name: f"[{name.upper()}]" for name in ("entity", "event", "mask", "pad", "unknown")}
    if masked:
        from tests.reference_models.batch_definitions import PRAGMA_BATCH

        fields = api.Fields(
            tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
            row=api.Tokens(max_tokens=max_tokens),
            field_names="fields",
            field_space="history",
            forbidden_fields=("entity_id", "sample_id", "label", "split"),
        )
        inputs = {
            "history": replace(
                PRAGMA_BATCH.inputs["history"], source=source, representation=fields
            ),
            "history_calendar": PRAGMA_BATCH.inputs["history_calendar"],
            "history_timing": PRAGMA_BATCH.inputs["history_timing"],
        }
        targets = {
            "masked_values": replace(PRAGMA_BATCH.targets["masked_values"], fields=tuple(encoders))
        }
        names = PRAGMA_BATCH.field_names
    else:
        roles["separator"] = "[SEP]"
        encoders = {"age_range": api.Buckets(bins=64), **encoders}
        for name, domain in CALENDAR.items():
            encoders[name] = api.Vocabulary(values=tuple((i, i + 2) for i in domain))
        fields = api.Fields(
            tuple(api.Encode(name, encoder, "history") for name, encoder in encoders.items()),
            row=api.Tokens(missing="error"),
            history=api.Flatten(
                (len(encoders) + 1) * capacity - 1, keep="first", separator="separator"
            ),
        )
        inputs = {"history": api.Input(source, fields, api.Padded())}
        targets = {"next_token": api.NextTokenTarget("history")}
        names = {}
    return api.BatchSpec(
        inputs,
        targets,
        id_spaces={"history": api.IdSpace(roles)},
        field_names=names,
        batching=api.Batching(max_samples=32),
    )


def bind_tables(directory, contract, root_key, targets=(), *, source_paths=None):
    """Resolve a prepared Parquet dataset with its registered anchors."""
    from kumo.pytorch import api

    if source_paths is None:
        source_paths = {t.name: directory / f"{t.name}.parquet" for t in contract.schema.tables}
    dataset = api.Dataset({name: api.Parquet(str(path)) for name, path in source_paths.items()})
    anchors = api.AnchorTable(
        api.Parquet(str(directory / "anchors.parquet")),
        root_key,
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={name: name for name in targets},
        metadata=("registered_entity_id", "prediction_time_ns", "anchor_row_position"),
    )
    return contract.resolve(dataset=dataset), dataset, anchors
