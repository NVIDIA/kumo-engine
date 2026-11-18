# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar target encodings, per-output validity and independent loss oracles.

Numerical outputs use float32, with explicit finite-range checks. Categorical
outputs fit their own vocabulary on training anchors; unseen non-null values
remain valid UNKNOWN labels. Missing values invalidate only that output.
"""

from collections import Counter
from copy import deepcopy
from decimal import Decimal
from math import isfinite
from struct import pack, unpack

from .diagnostics import mismatch


def source_target_rows(binding, path, projected):
    """Read only selected source anchor row groups, retaining exact target values."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from .bindings import profile

    anchor = profile(binding, "anchor")
    rules = target_rules(binding)
    columns = {
        "sample_id": anchor["sample_id"],
        "registered_entity_id": anchor["entity_id"],
        "prediction_time_ns": anchor["time"],
        "split": anchor["split"],
        **{name: rule["column"] for name, rule in rules.items()},
    }
    wanted = {row["anchor_row_position"] for row in projected}
    source, offset, result = pq.ParquetFile(path), 0, {}
    for group in range(source.num_row_groups):
        end = offset + source.metadata.row_group(group).num_rows
        positions = sorted(position for position in wanted if offset <= position < end)
        if positions:
            table = source.read_row_group(group, columns=list(columns.values()))
            table = table.take([position - offset for position in positions])
            clock = columns["prediction_time_ns"]
            table = table.set_column(
                table.schema.get_field_index(clock),
                clock,
                pc.cast(pc.cast(table[clock], pa.timestamp("ns")), pa.int64()),
            )
            for position, row in zip(positions, table.to_pylist(), strict=True):
                result[position] = {
                    **{name: row[column] for name, column in columns.items()},
                    "anchor_row_position": position,
                }
        offset = end
        if len(result) == len(wanted):
            break
    assert set(result) == wanted, "Selected anchor positions are absent from source"
    return result


def verify_target_projection(binding, projected, original):
    """Verify source identity and the declared float32 transport independently."""
    for name in ("sample_id", "registered_entity_id", "prediction_time_ns", "split"):
        assert projected[name] == original[name], mismatch(
            [projected[name]], [original[name]], [projected], name
        )
    for name, rule in target_rules(binding).items():
        value = original[name]
        expected = (
            value
            if value is None or rule["kind"] == "category"
            else unpack("f", pack("f", float(value)))[0]
        )
        assert projected[name] == expected, (
            f"Source target {name} projection changed; original={value!r}; "
            + mismatch(
                [projected[name]],
                [expected],
                [projected],
                name,
                precision="exact category" if rule["kind"] == "category" else "float32",
            )
        )


def verify_next_outgoing(rows, reference):
    """Both forecast labels come from one earliest outgoing source transfer."""
    for row in rows:
        cutoff = row["prediction_time_ns"] // 10**9
        eligible = (
            event
            for event in reference["events"]
            if event["sender"] == row["registered_entity_id"]
            and cutoff <= event["time"] < cutoff + 86400
        )
        first = min(
            eligible, key=lambda event: (event["time"], event["id"]), default=None
        )
        values = first["values"] if first else {}
        amount = values.get("amount_paid")
        expected = (
            Decimal(str(amount)) if amount is not None else None,
            values.get("payment_currency"),
        )
        assert (row["amount"], row["currency"]) == expected, (
            f"{row['sample_id']}: forecast targets differ from the earliest outgoing transfer; "
            + mismatch(
                [(row["amount"], row["currency"])],
                [expected],
                [row],
                "forecast targets",
                precision="original Decimal and category before float32 conversion",
            )
        )


def target_rules(workload):
    from .bindings import task_definition

    task = task_definition(workload)
    if task["kind"] != "stored":
        return {}
    return deepcopy(task["targets"])


def target_value(value, rule):
    """Validate source meaning before the documented float32 projection."""
    if value is None:
        return None
    kind = rule["kind"]
    column = rule["column"]
    if kind == "category":
        if not isinstance(value, str):
            raise ValueError(f"{column} must be a string or null")
        return value
    if kind == "decimal":
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"{column} must preserve a finite source decimal")
        value = float(value)
    elif type(value) not in (int, float):
        raise ValueError(f"{column} must be numeric or null")
    if not isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise ValueError(f"{column} cannot be represented as finite float32")
    if kind == "binary" and value not in (0, 1):
        raise ValueError(f"{column} must be binary or null")
    if kind == "delay_seconds" and value < 0:
        raise ValueError(f"{column} must be nonnegative seconds")
    return value


def fitted_target_encoder(state, name):
    bindings = state["targets"][name]["bindings"]
    assert len(bindings) == 1 and bindings[0][0] == "value" and bindings[0][2] is None
    return state["encoders"][bindings[0][1]]


def verify_target_fitting(workload, rows, state):
    """Independently count labels on the projected training population only."""
    rules = target_rules(workload)
    categories = {
        name: Counter() for name, rule in rules.items() if rule["kind"] == "category"
    }
    count = 0
    for row in rows:
        assert row["split"] == "train"
        count += 1
        for name, values in categories.items():
            if row[name] is not None:
                values[row[name]] += 1
    assert count > 0 and set(state["targets"]) == set(rules)
    for name, rule in rules.items():
        encoder = fitted_target_encoder(state, name)
        if rule["kind"] == "category":
            counts = categories[name]
            ordered = sorted(counts, key=lambda value: (-counts[value], value))
            assert encoder["kind"] == "vocabulary"
            assert dict(encoder["entries"]) == {
                value: i + 2 for i, value in enumerate(ordered)
            }
            assert encoder["roles"] == {"missing": 0, "unknown": 1}
            assert encoder["vocabulary_size"] == len(ordered) + 2
        else:
            assert encoder["kind"] == "raw" and encoder["dtype"]["name"] == "float32"


def verify_stored_batch(workload, batch, rows, state):
    """Check emitted scalar targets against independently streamed anchor rows."""
    import torch

    rules = target_rules(workload)
    assert set(batch.targets) == set(rules)
    counts = {}
    for name, rule in rules.items():
        target = batch.targets[name]
        values, valid = target["values"], target["valid"]
        assert values.shape == valid.shape == (len(rows),)
        assert valid.dtype == torch.bool
        raw = [row[name] for row in rows]
        assert valid.tolist() == [v is not None for v in raw], mismatch(
            valid.tolist(),
            [v is not None for v in raw],
            rows,
            f"{name}.valid",
            precision="exact bool",
        )
        if rule["kind"] == "category":
            entries = dict(fitted_target_encoder(state, name)["entries"])
            expected = [0 if v is None else entries.get(v, 1) for v in raw]
            assert values.dtype == torch.int64
            assert values.tolist() == expected, mismatch(
                values.tolist(),
                expected,
                rows,
                name,
                precision="exact int64 category IDs",
            )
        else:
            assert values.dtype == torch.float32
            expected = torch.tensor(
                [0 if v is None else v for v in raw],
                dtype=values.dtype,
                device=values.device,
            )
            torch.testing.assert_close(
                values[valid],
                expected[valid],
                rtol=0,
                atol=0,
                msg=mismatch(
                    [
                        v if eligible else None
                        for v, eligible in zip(
                            values.tolist(), valid.tolist(), strict=True
                        )
                    ],
                    [
                        v if eligible else None
                        for v, eligible in zip(
                            expected.tolist(), valid.tolist(), strict=True
                        )
                    ],
                    rows,
                    name,
                    precision="float32 (rtol=0, atol=0)",
                ),
            )
        counts[name] = int((valid & ~batch.metadata["is_padding"]).sum())
    return counts


def verify_stored_loss(workload, batch, result):
    """Audit loss algebra and eligibility independently of the model consumer."""
    import torch

    rules = target_rules(workload)
    assert (
        set(result.logits) == set(result.weights) == set(result.loss_sums) == set(rules)
    )
    total = result.loss.new_zeros(())
    for name, rule in rules.items():
        target = batch.targets[name]
        selected = target["valid"] & ~batch.metadata["is_padding"]
        labels, logits = target["values"][selected], result.logits[name][selected]
        count = int(selected.sum())
        assert result.weights[name] == count
        if rule["kind"] == "category":
            losses = torch.logsumexp(logits, dim=1) - logits.gather(
                1, labels[:, None]
            ).squeeze(1)
        elif rule["kind"] == "binary":
            x = logits[:, 0]
            losses = torch.logaddexp(torch.zeros_like(x), x) - labels * x
        else:
            difference = (logits[:, 0].double() - labels.double()).abs()
            losses = torch.where(
                difference < 1, 0.5 * difference.square(), difference - 0.5
            )
        torch.testing.assert_close(result.loss_sums[name], losses.sum())
        if count:
            total = total + losses.mean()
    assert result.weight == sum(result.weights.values())
    torch.testing.assert_close(result.loss, total)
