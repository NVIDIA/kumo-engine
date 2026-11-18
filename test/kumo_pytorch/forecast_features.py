# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent original-row oracles for partitioned graph/count forecasts."""

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .bindings import dataset_definition, graph_layout, profile
from .feature_checks import FeatureChecks
from .graph_features import graph_check
from .population import check_delivery
from .population_anchors import projected_rows
from .targets import (
    generated_checks,
    source_target_rows,
    verify_stored_batch,
    verify_target_projection,
)


def source_reference(binding, mapping, anchors):
    """Use originals, independent of converter outputs and normalization helpers."""
    options = dataset_definition(binding)["source_options"]
    root = options["root"]
    membership = {}
    for source in mapping["sources"][root["table"]]:
        parts = [
            p.split("=", 1)[1]
            for p in Path(source["path"]).parts
            if p.startswith(options["partition"] + "=")
        ]
        assert len(parts) == 1 and parts[0] in options["splits"]
        for name in (
            pq.ParquetFile(source["path"])
            .read(columns=[root["key"]])
            .column(0)
            .to_pylist()
        ):
            assert isinstance(name, str) and name and name not in membership
            membership[name] = parts[0]
    identities = {
        name: i
        for i, name in enumerate(
            sorted(membership, key=lambda name: name.encode("utf-8"))
        )
    }
    selected = {row["registered_entity_id"] for row in anchors.values()}
    for row in anchors.values():
        name = row["registered_entity_id"]
        assert identities[name] == row["entity_id"], "Client identity mapping differs"
        assert row["split"] == options["splits"][membership[name]], (
            "Client split membership differs"
        )
    names = pa.array(sorted(selected), pa.string())
    events = {identities[name]: [] for name in selected}
    spans = {}
    for family, spec in options["events"].items():
        offset, lo, hi = 0, None, None
        columns = [
            spec["root"],
            spec["time"],
            options["partition"],
            *spec["numerical"],
            *spec["categorical"],
        ]
        for source in mapping["sources"][family]:
            for batch in pq.ParquetFile(source["path"]).iter_batches(columns=columns):
                stamp = pc.cast(
                    pc.cast(batch.column(spec["time"]), pa.timestamp("ns")), pa.int64()
                )
                assert not stamp.null_count
                first, last = pc.min(stamp).as_py(), pc.max(stamp).as_py()
                if first is not None:
                    lo = first if lo is None else min(lo, first)
                    hi = last if hi is None else max(hi, last)
                data = (
                    pa.Table.from_batches([batch])
                    .append_column("__clock", stamp)
                    .append_column(
                        "__row",
                        pa.array(range(offset, offset + len(batch)), pa.int64()),
                    )
                )
                offset += len(batch)
                data = data.filter(pc.is_in(data[spec["root"]], value_set=names))
                for row in data.to_pylist():
                    name = row[spec["root"]]
                    assert str(row[options["partition"]]) == membership[name]
                    values = {}
                    for field in spec["numerical"]:
                        value = row[field]
                        values[field] = (
                            float(value)
                            if value is not None and math.isfinite(value)
                            else None
                        )
                    for field in spec["categorical"]:
                        value = row[field]
                        assert value is None or type(value) in (int, float)
                        values[field] = (
                            None
                            if value is None or not math.isfinite(value)
                            else str(int(value))
                            if value == int(value)
                            else repr(value)
                        )
                    events[identities[name]].append(
                        {
                            "family": family,
                            "source_id": row["__row"],
                            "mapped_id": row["__row"],
                            "time": row["__clock"] // 10**9,
                            "time_ns": row["__clock"],
                            "values": values,
                        }
                    )
        spans[family] = (lo, hi)
    return events, spans


def future_counts(anchors, events, spans, checks):
    answers = {}
    for sample, row in anchors.items():
        start = row["prediction_time_ns"]
        answers[sample] = {}
        for name, check in checks.items():
            stop = start + check["window_seconds"] * 10**9
            first, last = spans[check["family"]]
            assert first is not None and first <= start and stop <= last, (
                "Future window lacks recorded-span coverage"
            )
            answers[sample][name] = sum(
                event["family"] == check["family"] and start <= event["time_ns"] < stop
                for event in events[row["entity_id"]]
            )
    return answers


def count_features(binding, projection, mapping, state, *, mode):
    rows = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    anchors = {row["sample_id"]: row for row in rows}
    originals = source_target_rows(binding, mapping["anchor_source"]["path"], rows)
    events, spans = source_reference(binding, mapping, anchors)
    checks = generated_checks(binding)
    answers = future_counts(anchors, events, spans, checks)
    for row in rows:
        original = originals[row["anchor_row_position"]]
        verify_target_projection(binding, row, original)
        assert row["as_of_time"] == original["prediction_time_ns"] // 10**9 - 1, (
            f"{row['sample_id']}: execution clock differs from original prediction T-1s"
        )
        for name, check in checks.items():
            assert (
                original[check["stored"]]
                == row["original_" + check["stored"]]
                == answers[row["sample_id"]][name]
            ), (
                f"{row['sample_id']}: original stored count={original[check['stored']]!r}, "
                f"projected integer={row['original_' + check['stored']]!r}, "
                f"independent future count={answers[row['sample_id']][name]}"
            )
    inputs = profile(binding, "input")

    def generated(batch):
        import torch

        for name in checks:
            target = batch.targets[name]
            assert (
                target["values"].dtype == torch.int64
                and target["valid"].dtype == torch.bool
            )
            assert (
                target["values"].shape
                == target["valid"].shape
                == (len(batch.metadata["sample_ids"]),)
            )
            assert target["valid"].all()
            assert target["values"].tolist() == [
                answers[s][name] for s in batch.metadata["sample_ids"]
            ], "Generated count disagrees with original future transaction rows"

    return FeatureChecks(
        binding["feature_checks"],
        mode,
        {
            "identity_cutoff": lambda batch: check_delivery(
                batch,
                iter(anchors[s] for s in batch.metadata["sample_ids"]),
                graph_clocks={
                    node: fields["timestamp"]
                    for node, fields in graph_layout(inputs)[0].items()
                },
            ),
            "graph_values": graph_check(
                binding,
                mapping,
                anchors,
                events,
                state,
                root_rows=[{"root_id": key} for key in events],
            ),
            "stored_targets": lambda batch: verify_stored_batch(
                binding,
                batch,
                [anchors[s] for s in batch.metadata["sample_ids"]],
                state,
            ),
            "generated_targets": generated,
        },
        {row["sample_id"]: row["split"] for row in rows},
    )
