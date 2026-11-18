# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent reductions over original event rows, before and after fitting."""

import json
import math
import statistics
from collections import Counter, defaultdict

from .bindings import aggregate_fields


def reduce_rows(rows, operation, column, cutoff):
    if operation == "count":
        return len(rows)
    values = [row[column] for row in rows if row[column] is not None]
    if operation == "count_distinct":
        return len(set(values))
    if not values:
        return 0 if operation == "sum" and not rows else None
    if operation == "sum":
        return sum(values)
    if operation in {"min", "max"}:
        return (min if operation == "min" else max)(values)
    if operation == "mean":
        return statistics.mean(values)
    if operation in {"std_population", "var_population"}:
        return (
            statistics.pstdev if operation == "std_population" else statistics.pvariance
        )(values)
    if operation == "mode":
        counts = Counter(values)
        return min(
            counts,
            key=lambda value: (-counts[value], type(value).__name__, repr(value)),
        )
    if operation == "recency":
        return float(cutoff - max(values))
    raise AssertionError(f"No independent reduction rule: {operation}")


def aggregate_answers(inputs, events, anchor):
    """Evaluate the declared windows with Python rows, never engine results."""
    cutoff = anchor["as_of_time"]
    families = {tuple(path): name for name, path, _, _ in inputs["families"]}

    def eligible(family, start):
        return [
            {**event["values"], "event_time": event["time"]}
            for event in events[anchor["entity_id"]]
            if event["family"] == family
            and event["time"] <= cutoff
            and (start is None or event["time"] > cutoff + start)
        ]

    def source(spec):
        if "keys" not in spec:
            return eligible(spec["family"], spec.get("start"))
        groups = defaultdict(list)
        for row in source(spec["source"]):
            groups[tuple(row[key] for key in spec["keys"])].append(row)
        return [
            {
                **dict(zip(spec["keys"], keys, strict=True)),
                **{
                    name: reduce_rows(
                        rows, query["operation"], query.get("column"), cutoff
                    )
                    for name, query in spec["values"].items()
                },
            }
            for keys, rows in groups.items()
        ]

    answers = {
        name: len(eligible(families[tuple(path)], start))
        for name, path, _, start in inputs["aggregates"]
    }
    queries = inputs.get("aggregate_queries", {})
    pending = set()

    def answer(name):
        if name in answers:
            return answers[name]
        assert name not in pending, "Cyclic aggregate dependencies"
        pending.add(name)
        query = queries[name]
        if "constant" in query:
            value = query["constant"]
        elif "derived" in query:
            left, right = (answer(n) for n in query["inputs"])
            if left is None or right is None:
                value = None
            elif query["derived"] == "add":
                value = left + right
            else:
                assert query["derived"] == "divide", "No independent derived rule"
                value = left / right if right else None
        else:
            value = reduce_rows(
                source(query["source"]), query["operation"], query.get("column"), cutoff
            )
        pending.remove(name)
        answers[name] = value
        return value

    for name in queries:
        answer(name)
    return answers


def aggregate_check(inputs, events, anchors, state):
    """Build expected values once; assert train-only fit statistics independently."""
    import torch

    answers = {
        sample: aggregate_answers(inputs, events, row)
        for sample, row in anchors.items()
    }
    queries = inputs.get("aggregate_queries", {})
    exact_counts = {entry[0] for entry in inputs["aggregates"]}
    for name, query in queries.items():
        operation = query.get("operation")
        grouped = query.get("source", {}).get("values", {})
        if operation in {"count", "count_distinct"} or (
            operation in {"min", "max"}
            and grouped.get(query.get("column"), {}).get("operation")
            in {"count", "count_distinct"}
        ):
            exact_counts.add(name)
        if query.get("categorical"):
            values = Counter(
                answers[s][name]
                for s, row in anchors.items()
                if row["split"] == "train" and answers[s][name] is not None
            )
            key = json.dumps(
                ["inputs", "aggregate_checks", "fields", name], separators=(",", ":")
            )
            encoder = state["encoders"][key]
            assert dict(encoder["entries"]) == {
                value: i + 2
                for i, value in enumerate(
                    sorted(values, key=lambda value: (-values[value], value))
                )
            }, f"{name}: train-only vocabulary differs"
    fitted = {}
    for name in aggregate_fields(inputs):
        values = [
            answers[s][name]
            for s, row in anchors.items()
            if row["split"] == "train" and answers[s][name] is not None
        ]
        mean = statistics.mean(values) if values else 0.0
        scale = (statistics.pstdev(values) or 1.0) if values else 1.0
        key = json.dumps(
            ["inputs", "aggregates", "fields", name], separators=(",", ":")
        )
        encoder = state["encoders"][key]
        assert encoder["kind"] == "zscore"
        # Native and Python floating reductions may associate differently.
        # 2e-6 matches the suite's existing float32 normalization bound.
        for field, expected in (("mean", mean), ("scale", scale)):
            assert math.isclose(encoder[field], expected, rel_tol=2e-6, abs_tol=2e-6), (
                f"{name}: train-only {field} differs"
            )
        fitted[name] = mean, scale

    def verify(batch):
        ids = batch.metadata["sample_ids"]
        raw, normalized = batch.inputs["aggregate_checks"], batch.inputs["aggregates"]
        assert set(raw) == set(next(iter(answers.values())))
        assert set(normalized) == set(fitted)
        for name, item in raw.items():
            expected = [answers[s][name] for s in ids]
            key = json.dumps(
                ["inputs", "aggregate_checks", "fields", name], separators=(",", ":")
            )
            encoder = state["encoders"][key]
            categorical = queries.get(name, {}).get("categorical", False)
            assert encoder["kind"] == ("vocabulary" if categorical else "raw")
            dtype = (
                torch.int64
                if categorical or name in exact_counts
                else getattr(torch, queries[name]["dtype"])
            )
            assert item["values"].dtype == dtype
            assert item["values"].shape == item["valid"].shape == (len(ids),)
            assert item["valid"].dtype == torch.bool
            assert item["valid"].tolist() == [
                value is not None for value in expected
            ], f"{name}: validity"
            for index, value in enumerate(expected):
                if value is None:
                    continue
                if categorical:
                    value = dict(encoder["entries"]).get(
                        value, encoder["roles"]["unknown"]
                    )
                actual = item["values"][index].item()
                assert (
                    math.isclose(actual, value, rel_tol=2e-12, abs_tol=2e-12)
                    if item["values"].is_floating_point()
                    else actual == value
                ), (
                    f"{ids[index]}/{name}: raw aggregate expected={value!r}, actual={actual!r}"
                )
        for name, (mean, scale) in fitted.items():
            item = normalized[name]
            assert (
                item["values"].dtype == torch.float32
                and item["valid"].dtype == torch.bool
            )
            assert item["values"].shape == item["valid"].shape == (len(ids),)
            assert torch.equal(item["valid"], raw[name]["valid"])
            for index, sample in enumerate(ids):
                value = answers[sample][name]
                if value is not None:
                    expected = (value - mean) / scale
                    assert math.isclose(
                        item["values"][index].item(),
                        expected,
                        rel_tol=2e-6,
                        abs_tol=2e-6,
                    ), f"{sample}/{name}: normalized aggregate differs"

    return verify
