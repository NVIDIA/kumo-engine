# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent account-involvement references; never supplied as model inputs.

No imports from the source mapper, task builders, model loss or anchor generator.
The source reader retains only selected accounts' transactions, while split
boundaries and observation coverage use every source transaction.
"""

import json
import math
from collections import Counter
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY = 86400


def source_reference(
    path, entity_ids, variant="hi-small", *, include_values=False, include_labels=True
):
    roots = set(entity_ids)
    numbers = pa.array(sorted({json.loads(root)[3] for root in roots}))
    timeline, events = Counter(), []
    offset = 0
    columns = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
    ]
    if include_labels:
        columns.append("is_laundering")
    value_fields = (
        "amount_paid",
        "payment_currency",
        "amount_received",
        "receiving_currency",
        "payment_format",
    )
    if include_values:
        columns.extend(value_fields)

    def clock(text):
        return int(
            datetime.strptime(text, "%Y/%m/%d %H:%M")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    def account(bank, number):
        return json.dumps(
            [variant, "account", str(int(bank)), number],
            separators=(",", ":"),
            ensure_ascii=False,
        )

    for batch in pq.ParquetFile(path).iter_batches(
        columns=columns, batch_size=65536, use_threads=False
    ):
        for item in pc.value_counts(batch.column("timestamp")).to_pylist():
            timeline[item["values"]] += item["counts"]
        mask = pc.or_(
            pc.is_in(batch.column("from_account"), value_set=numbers),
            pc.is_in(batch.column("to_account"), value_set=numbers),
        )
        positions = pc.indices_nonzero(mask)
        for position, row in zip(
            positions.to_pylist(), batch.take(positions).to_pylist(), strict=True
        ):
            sender = account(row["from_bank"], row["from_account"])
            receiver = account(row["to_bank"], row["to_account"])
            endpoints = {sender, receiver}
            if not endpoints & roots:
                continue
            if include_labels and row["is_laundering"] not in ("0", "1"):
                raise ValueError("Expected binary source laundering flag")
            events.append(
                {
                    "id": offset + position,
                    "time": clock(row["timestamp"]),
                    "accounts": endpoints,
                    "sender": sender,
                    "receiver": receiver,
                }
            )
            if include_labels:
                events[-1]["laundering"] = row["is_laundering"] == "1"
            if include_values:
                events[-1]["values"] = {name: row[name] for name in value_fields}
        offset += batch.num_rows
    if not timeline:
        raise ValueError("Empty source observation")
    cumulative, cuts = 0, []
    for stamp in sorted(timeline):
        cumulative += timeline[stamp]
        while len(cuts) < 2 and cumulative > offset * (6, 8)[len(cuts)] // 10:
            cuts.append(clock(stamp))
    return {
        "events": events,
        "cuts": cuts,
        "start": clock(min(timeline)),
        "end": clock(max(timeline)),
        "source_rows": offset,
    }


def expected_anchor(row, reference):
    cutoff = row["as_of_time"] // 10**9
    if row["as_of_time"] % (DAY * 10**9):
        raise AssertionError("Prediction time is not daily UTC midnight")
    touching = [
        event for event in reference["events"] if row["entity_id"] in event["accounts"]
    ]
    if cutoff < reference["start"] or not any(
        event["time"] < cutoff for event in touching
    ):
        raise AssertionError("Account anchor has no prior activity")
    train_end, valid_end = reference["cuts"]
    split, end = (
        ("train", train_end)
        if cutoff < train_end
        else ("valid", valid_end)
        if cutoff < valid_end
        else ("test", reference["end"])
    )
    if cutoff + DAY > min(end, reference["end"]):
        return None, None
    positive = any(
        event["laundering"] and cutoff <= event["time"] < cutoff + DAY
        for event in touching
    )
    return split, int(positive)


def verify_anchors(rows, reference):
    labels = {}
    for row in rows:
        split, label = expected_anchor(row, reference)
        if row["forecast_split"] != split:
            raise AssertionError(
                f"{row['sample_id']}: forecast split/maturity differs from source"
            )
        if row["involvement"] != label:
            raise AssertionError(
                f"{row['sample_id']}: involvement target differs from source"
            )
        if (
            "graph_as_of_time" in row
            and row["graph_as_of_time"] != row["as_of_time"] - 10**9
        ):
            raise AssertionError("Graph cutoff differs from original prediction time")
        if row["sample_id"] in labels:
            raise AssertionError("Duplicate anchor identity")
        labels[row["sample_id"]] = label
    return labels


def binary_evaluation(logits, labels, padding):
    """Python scalar BCE, independent of torch loss/mask/reduction helpers."""
    if not (len(logits) == len(labels) == len(padding)):
        raise AssertionError("Evaluation row counts differ")
    terms, invalid, padded = [], 0, 0
    for score, label, is_padding in zip(logits, labels, padding, strict=True):
        if is_padding:
            padded += 1
        elif label is None:
            invalid += 1
        else:
            if label not in (0, 1) or not math.isfinite(score):
                raise AssertionError("Invalid eligible binary prediction/label")
            terms.append(
                max(score, 0) - score * label + math.log1p(math.exp(-abs(score)))
            )
    total = math.fsum(terms)
    return {
        "loss_sum": total,
        "eligible": len(terms),
        "invalid": invalid,
        "padding": padded,
        "mean_loss": total / len(terms) if terms else None,
    }


def history(row, reference):
    return sorted(
        (
            event
            for event in reference["events"]
            if row["entity_id"] in event["accounts"]
            and event["time"] * 10**9 < row["as_of_time"]
        ),
        key=lambda event: (event["time"], event["id"]),
    )


def relationship_counts(row, reference):
    past = history(row, reference)
    result = {}
    for name, duration in (
        ("all_history", None),
        ("24h", DAY),
        ("7d", 7 * DAY),
        ("30d", 30 * DAY),
    ):
        window = [
            event
            for event in past
            if duration is None
            or event["time"] >= row["as_of_time"] // 10**9 - duration
        ]
        result[f"{name}_total_touching_count"] = len(window)
        for role in ("sender", "receiver"):
            result[f"{name}_{role}_count"] = sum(
                event[role] == row["entity_id"] for event in window
            )
    return result


def combined_features(binding, projection, mapping, state, *, mode):
    """Build the existing source oracle for a bounded account/transfer binding.

    Dataset identity comes from its declaration. This factory owns neither
    execution nor targets; the shared runner verifies target values and losses.
    """
    from .bindings import dataset_definition, resolve
    from .feature_checks import FeatureChecks
    from .population_anchors import projected_rows
    from .reference_tokens import (
        verify_aggregate_layout,
        verify_graph_values,
        verify_history_values,
    )
    from .targets import (
        source_target_rows,
        verify_stored_batch,
        verify_target_fitting,
        verify_target_projection,
    )

    assert binding["representation"] == "combined"
    projected = [
        row
        for split in ("train", "valid", "test")
        for row in projected_rows(projection, split)
    ]
    rows = [
        {
            "sample_id": row["sample_id"],
            "entity_id": row["registered_entity_id"],
            "as_of_time": row["prediction_time_ns"],
            "split": row["split"],
        }
        for row in projected
    ]
    anchors = {row["sample_id"]: row for row in rows}
    reference = source_reference(
        mapping["sources"]["transactions"]["path"],
        {row["entity_id"] for row in rows},
        dataset_definition(binding)["source_options"]["variant"],
        include_values=True,
        include_labels=False,
    )
    # Derive identities from the original binding, not the converter's output.
    names = (
        pq.read_table(
            mapping["sources"]["accounts_binding"]["path"], columns=["entity_id"]
        )
        .column(0)
        .to_pylist()
    )
    identities = {
        name: i for i, name in enumerate(sorted(names, key=lambda s: s.encode("utf-8")))
    }
    training = [row for row in rows if row["split"] == "train"]
    projected_by_id = {row["sample_id"]: row for row in projected}
    originals = source_target_rows(binding, mapping["anchor_source"]["path"], projected)

    def selected_rows(batch):
        return [projected_by_id[sample] for sample in batch.metadata["sample_ids"]]

    def input_check(name):
        return lambda batch: verify_inputs(
            batch, "combined", anchors, reference, identities, training, checks=(name,)
        )

    def targets(batch):
        for row in selected_rows(batch):
            verify_target_projection(
                binding, row, originals[row["anchor_row_position"]]
            )
        verify_target_fitting(
            binding, (row for row in projected if row["split"] == "train"), state
        )
        verify_stored_batch(binding, batch, selected_rows(batch), state)

    checks = {
        "identity_cutoff": input_check("identity"),
        "graph_membership": input_check("graph"),
        "graph_values": lambda batch: verify_graph_values(
            batch, anchors, reference, state, identities
        ),
        "history_membership": input_check("history"),
        "history_values": lambda batch: verify_history_values(
            batch, anchors, reference, state
        ),
        "aggregate_counts": input_check("aggregates"),
        "aggregate_layout": verify_aggregate_layout,
        "stored_targets": targets,
    }
    source_targets = binding["feature_checks"].get("source_targets", {}).get(mode)
    if source_targets:
        checks["forecast_targets"] = lambda batch: resolve(source_targets)(
            [originals[row["anchor_row_position"]] for row in selected_rows(batch)],
            reference,
        )
    return FeatureChecks(
        binding["feature_checks"],
        mode,
        checks,
        {row["sample_id"]: row["split"] for row in projected},
    )


def verify_inputs(
    batch,
    family,
    anchors,
    reference,
    identities,
    training_rows,
    *,
    history_reference=None,
    graph_reference=None,
    checks=("identity", "graph", "history", "aggregates"),
):
    """Check graph membership/edges, full flat history and all twelve counts.

    Encoded token values have separate engine/encoding coverage. Here their
    event grouping/count and the permitted input field boundary are explicit.
    This source comparison uses unpadded batches. Generic distributed delivery
    and padding identity are covered by the existing package tests.
    Explicit history/graph variants may supply selected source rows separately;
    aggregate counts always use the complete reference.
    """
    if batch.metadata["is_padding"].any():
        raise AssertionError("Source input comparison requires an unpadded batch")
    rows = [anchors[sid] for sid in batch.metadata["sample_ids"]]
    graph = family in {"graphsage", "combined"}
    if "identity" in checks:
        expected_keys = (
            {"graph"}
            if family == "graphsage"
            else {"history"}
            if family == "nuformer"
            else {"graph", "history", "aggregates"}
        )
        if set(batch.inputs) != expected_keys:
            raise AssertionError("Unexpected input branch or target leakage")
        clocks = [row["as_of_time"] // 10**9 - int(graph) for row in rows]
        if batch.metadata["as_of_time"].tolist() != clocks:
            raise AssertionError("Execution cutoff differs from original anchor")
        if list(batch.metadata["entity_ids"]) != [
            identities[row["entity_id"]] for row in rows
        ]:
            raise AssertionError("Root identity/order differs from original anchor")
    histories = [history(row, reference) for row in rows]
    if graph and "graph" in checks:
        import torch

        graph_histories = (
            histories
            if graph_reference is None
            else [history(row, graph_reference) for row in rows]
        )
        payload = batch.inputs["graph"]
        nodes = payload["nodes"]
        if set(nodes) - {"accounts", "transactions"} or nodes["accounts"]["features"]:
            raise AssertionError("Unpermitted graph node features")
        transactions = nodes.get("transactions")
        if transactions is not None:
            permitted = {
                "amount_paid",
                "amount_received",
                "payment_currency",
                "receiving_currency",
                "payment_format",
                "event_time_seconds",
            }
            if set(transactions["features"]) != permitted:
                raise AssertionError("Unpermitted graph fields")
        for owner, events in enumerate(graph_histories):
            actual = (
                []
                if transactions is None
                else [
                    int(identity)
                    for identity, member in zip(
                        transactions["ids"], transactions["batch"], strict=True
                    )
                    if int(member) == owner
                ]
            )
            if sorted(actual) != sorted(event["id"] for event in events):
                raise AssertionError("Graph history differs from strict source cutoff")
        expected_edges = Counter(
            (role, owner, identities[event[role]], event["id"])
            for owner, events in enumerate(graph_histories)
            for event in events
            for role in ("sender", "receiver")
        )
        actual_edges = Counter()
        for role, edge in payload["edges"].items():
            if (
                edge["source_type"] != "accounts"
                or edge["destination_type"] != "transactions"
            ):
                raise AssertionError("Unexpected graph edge direction")
            indices = edge["index"]
            assert (
                indices.dtype == torch.int64
                and indices.ndim == 2
                and indices.shape[0] == 2
            ), "Graph edge indices must be an int64 matrix with two rows"
            assert (indices >= 0).all(), "Graph edge indices must be nonnegative"
            assert (indices[0] < len(nodes["accounts"]["ids"])).all(), (
                "Graph source edge index exceeds node table"
            )
            assert (
                indices[1]
                < (len(transactions["ids"]) if transactions is not None else 0)
            ).all(), "Graph destination edge index exceeds node table"
            for account_index, event_index in edge["index"].T.tolist():
                owner = int(nodes["accounts"]["batch"][account_index])
                if owner != int(transactions["batch"][event_index]):
                    raise AssertionError("Graph edge crosses example owners")
                actual_edges[
                    (
                        role,
                        owner,
                        int(nodes["accounts"]["ids"][account_index]),
                        int(transactions["ids"][event_index]),
                    )
                ] += 1
        if actual_edges != expected_edges:
            raise AssertionError("Graph relationships differ from source endpoints")
    if family != "graphsage" and "history" in checks:
        stream = batch.inputs["history"]
        sequence_histories = (
            histories
            if history_reference is None
            else [history(row, history_reference) for row in rows]
        )
        for owner, events in enumerate(sequence_histories):
            expected = [
                item
                for index in range(len(events))
                for item in ([index] * 10 + ([-1] if index + 1 < len(events) else []))
            ]
            valid = stream["valid"][owner]
            if valid.tolist() != [True] * len(expected) + [False] * (
                len(valid) - len(expected)
            ):
                raise AssertionError(
                    "Token history truncation, shift or padding differs"
                )
            actual = stream["source_event_indices"][owner][valid].tolist()
            # Row assembly uses per-sample indices; native columnar assembly
            # indexes the batch's concatenated source rows. Compare event groups
            # relative to this sample, preserving separator positions exactly.
            base = actual[0] if actual else 0
            if (
                base < 0
                or [index - base if index >= 0 else index for index in actual]
                != expected
            ):
                raise AssertionError("Token event grouping/order differs")
    if family == "combined" and "aggregates" in checks:
        fitted = [relationship_counts(row, reference) for row in training_rows]
        actual = batch.inputs["aggregates"]
        if set(actual) != set(fitted[0]):
            raise AssertionError("Aggregate field allowlist differs")
        for owner, row in enumerate(rows):
            for field, count in relationship_counts(row, reference).items():
                mean = sum(item[field] for item in fitted) / len(fitted)
                scale = (
                    math.sqrt(
                        sum((item[field] - mean) ** 2 for item in fitted) / len(fitted)
                    )
                    or 1.0
                )
                expected = (count - mean) / scale
                if not actual[field]["valid"][owner] or not math.isclose(
                    actual[field]["values"][owner].item(),
                    expected,
                    rel_tol=2e-6,
                    abs_tol=2e-6,
                ):
                    raise AssertionError(
                        "Aggregate window/count/training normalization differs"
                    )


def verify_batch(batch, result, expected_labels):
    if set(batch.targets) != {"label"}:
        raise AssertionError("Supervised involvement must emit only the label target")
    ids = batch.metadata["sample_ids"]
    padding = batch.metadata["is_padding"].tolist()
    # Padded requests have generated identities and never enter the objective.
    # Their identity/content mechanics belong to the package regression tests.
    labels = [
        None if padded else expected_labels[sid]
        for sid, padded in zip(ids, padding, strict=True)
    ]
    target = batch.targets["label"]
    actual = target["values"].detach().flatten().tolist()
    validity = target["valid"].flatten().tolist()
    if (
        len(actual) != len(ids)
        or len(validity) != len(ids)
        or any(
            not padded and valid != (label is not None)
            for valid, label, padded in zip(validity, labels, padding, strict=True)
        )
    ):
        raise AssertionError("Target validity/shape differs from expected labels")
    if any(
        label is not None and value != label
        for value, label in zip(actual, labels, strict=True)
    ):
        raise AssertionError("Emitted involvement target differs from source")
    logits = result.logits.detach().flatten().double().tolist()
    metrics = binary_evaluation(logits, labels, batch.metadata["is_padding"].tolist())
    if result.weight != metrics["eligible"]:
        raise AssertionError("Loss denominator includes ineligible examples")
    # The consumer computes float32 BCE. Its intermediate log-sigmoid can
    # round on the scale of the logit, before cancellation leaves a small loss.
    # Bound that rounding by one float32 ULP per eligible logit in the mean;
    # ignored/padded scores must not enlarge the tolerance.
    roundoff = math.fsum(
        math.ldexp(1.0, max(-149, math.frexp(abs(score))[1] - 24))
        for score, label, padded in zip(logits, labels, padding, strict=True)
        if label is not None and not padded and score != 0
    ) / max(metrics["eligible"], 1)
    if not math.isclose(
        result.loss.item(),
        metrics["mean_loss"] or 0.0,
        rel_tol=2e-6,
        abs_tol=max(2e-7, roundoff),
    ):
        raise AssertionError("Loss differs from independent eligible-example BCE")
    return metrics
