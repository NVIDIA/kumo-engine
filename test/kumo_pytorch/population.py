# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared population assertions; all campaigns use nightly run."""

import hashlib
import json
import time
from itertools import islice

import torch

from .aml_inputs import SPLITS
from .bindings import graph_layout
from .bindings import profile as binding_profile
from .bindings import task_definition
from .contracts import GRAPH_TIMESTAMP_FIELD
from .diagnostics import mismatch
from .population_anchors import add_identity, projected_rows
from .stage_timing import StageTiming
from .targets import verify_stored_batch, verify_target_fitting


def write_report(directory, report):
    # Durable partial evidence if a later batch or native process fails.
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "report.json")


def check_delivery(batch, expected, *, graph_clocks=None):
    """Assert identities, order, clocks, metadata and finite native outputs."""
    ids = list(batch.metadata["sample_ids"])
    rows = list(islice(expected, len(ids)))
    assert ids and ids == [row["sample_id"] for row in rows], (
        "Missing, duplicate or reordered anchors; "
        + mismatch(ids, [row["sample_id"] for row in rows], rows, "sample_id")
    )
    assert not batch.metadata["is_padding"].any(), "Unexpected sample padding"
    assert batch.metadata["entity_ids"].tolist() == [
        row["entity_id"] for row in rows
    ], mismatch(
        batch.metadata["entity_ids"].tolist(),
        [row["entity_id"] for row in rows],
        rows,
        "entity_id",
    )
    assert batch.metadata["as_of_time"].tolist() == [
        row["as_of_time"] for row in rows
    ], mismatch(
        batch.metadata["as_of_time"].tolist(),
        [row["as_of_time"] for row in rows],
        rows,
        "as_of_time",
        precision="integer seconds",
    )
    expected_metadata = [
        {
            name: row[name]
            for name in (
                "prediction_time_ns",
                "registered_entity_id",
                "anchor_row_position",
                "split",
            )
        }
        for row in rows
    ]
    assert list(batch.metadata["request_metadata"]) == expected_metadata, mismatch(
        list(batch.metadata["request_metadata"]),
        expected_metadata,
        rows,
        "request_metadata",
    )

    def finite(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), "Nonfinite emitted values"
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)

    finite(batch.inputs)
    finite(batch.targets)
    if "graph" in batch.inputs:
        for name, field in (
            graph_clocks
            if graph_clocks is not None
            else {"transactions": GRAPH_TIMESTAMP_FIELD}
        ).items():
            nodes = batch.inputs["graph"]["nodes"].get(name)
            if nodes is None:
                continue
            times = nodes["features"][field]["values"].flatten()
            cutoffs = batch.metadata["as_of_time"][nodes["batch"]]
            assert (times <= cutoffs).all(), (
                "Graph history leaked beyond the anchor cutoff"
            )
    return rows


def check_batch(batch, expected, *, native, family, state):
    rows = check_delivery(batch, expected)
    ids = list(batch.metadata["sample_ids"])
    target_name = (
        ("masked_values" if family == "pragma" else "next_token") if native else "label"
    )
    assert set(batch.targets) == {target_name}, "Wrong task targets"
    target = batch.targets[target_name]
    assert target["values"].shape == target["valid"].shape
    assert target["valid"].dtype == torch.bool
    if not native:
        labels = [row["label"] for row in rows]
        assert target["valid"].flatten().tolist() == [v is not None for v in labels], (
            mismatch(
                target["valid"].flatten().tolist(),
                [v is not None for v in labels],
                rows,
                "label.valid",
                precision="exact bool",
            )
        )
        assert all(
            label is None or value == label
            for value, label in zip(
                target["values"].flatten().tolist(), labels, strict=True
            )
        ), "Stored labels differ; " + mismatch(
            [
                value if label is not None else None
                for value, label in zip(
                    target["values"].flatten().tolist(), labels, strict=True
                )
            ],
            labels,
            rows,
            "label",
            precision=str(target["values"].dtype),
        )
    else:
        history = batch.inputs["history"]
        assert target["valid"].shape == history["valid"].shape
        assert not (target["valid"] & ~history["valid"]).any(), (
            "Padding became a target"
        )
        assert (target["values"][~target["valid"]] == -100).all()
        assert (target["values"][target["valid"]] >= 0).all()
        if family == "pragma":
            mask_id = state["spaces"]["history"]["roles"]["mask"]
            assert torch.equal(target["valid"], history["value_ids"] == mask_id)
        else:
            assert torch.equal(target["valid"], history["valid"])
            valid = history["valid"][:, 1:]
            assert torch.equal(
                target["values"][:, :-1][valid], history["value_ids"][:, 1:][valid]
            )
    return ids, int(target["valid"].sum())


def verify_batch_for(binding, batch, expected, state):
    """One delivery oracle for stored labels and native pretraining targets."""
    task = task_definition(binding)
    if task["kind"] != "stored":
        family = binding["representation"]
        _, eligible = check_batch(
            batch, expected, native=True, family=family, state=state
        )
        return {task["kind"]: eligible}
    inputs = binding_profile(binding, "input")
    rows = check_delivery(
        batch,
        expected,
        graph_clocks={
            node: fields["timestamp"]
            for node, fields in graph_layout(inputs)[0].items()
            if fields.get("timestamp")
        }
        if "graph" in inputs
        else None,
    )
    return verify_stored_batch(task, batch, rows, state)


def check_population(
    task,
    binding,
    projection,
    population,
    directory,
    report,
    *,
    batch_size,
    seed=0,
    features=None,
):
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    assert isinstance(task._anchors, NativeAnchors), (
        "Population requires native AnchorTable iteration"
    )
    timing = StageTiming(report, lambda: write_report(directory, report))
    with timing.span("population.setup", checkpoint=True):
        frozen = task.fitted_state.fingerprint
        state = task.fitted_state.to_dict()
        report["fitted_state_sha256"] = frozen
        task.to_artifact().save(directory / "fitted")
        assert {s: task.count(s) for s in SPLITS} == population["counts"], (
            "Native population counts differ"
        )
        if binding["targets"]:
            verify_target_fitting(
                task_definition(binding), projected_rows(projection, "train"), state
            )
    names = tuple(binding["targets"]) or (task_definition(binding)["kind"],)
    splits = report["splits"] = {}
    for split in SPLITS:
        splits[split] = {
            "status": "pending",
            "expected": population["counts"][split],
            "delivered": 0,
            "batches": 0,
            "eligible_targets": dict.fromkeys(names, 0),
        }
    for split in SPLITS:
        result = splits[split]
        result["status"] = "running"
        expected, digest = projected_rows(projection, split), hashlib.sha256()
        with timing.span("population.plan", checkpoint=True):
            batches = iter(
                task.batches(
                    split=split,
                    epoch=0,
                    seed=seed,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    output_format="model",
                )
            )
        first = True
        while True:
            # Planning can run inside the first next(); retain that latency
            # separately as a subset of population.next, not an additive stage.
            with timing.span("population.next", checkpoint=first):
                try:
                    started = time.perf_counter()
                    batch = next(batches)
                except StopIteration:
                    break
                if first:
                    result["first_batch_seconds"] = time.perf_counter() - started
                    first = False
            with timing.span("population.verify"):
                if features is not None:
                    features(batch, phase="population")
                eligible = verify_batch_for(binding, batch, expected, state)
                for sample in batch.metadata["sample_ids"]:
                    add_identity(digest, sample)
                for name, count in eligible.items():
                    result["eligible_targets"][name] += count
                result["delivered"] += len(batch.metadata["sample_ids"])
                result["batches"] += 1
            if result["batches"] % 100 == 0:
                write_report(directory, report)
        missing = next(expected, None)
        assert missing is None, (
            f"Iteration dropped anchors after split position {result['delivered']}: first expected={missing!r}, actual=<end of iteration>"
        )
        assert result["delivered"] == result["expected"], "Incomplete split"
        result["ordered_sample_id_sha256"] = digest.hexdigest()
        assert digest.hexdigest() == population["ordered_sample_id_sha256"][split]
        assert all(n > 0 for n in result["eligible_targets"].values()), (
            "Split has no eligible targets"
        )
        assert task.fitted_state.fingerprint == frozen, (
            "Batch iteration changed fitted state"
        )
        result["status"] = "passed"
        write_report(directory, report)
    return splits


if __name__ == "__main__":
    raise SystemExit(
        "Use: python -m test.kumo_pytorch.nightly run --binding ID --mode smoke|full ..."
    )
