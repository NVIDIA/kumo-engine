# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph encoders do not require Python records for the other fitting inputs."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api, assembly
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from tests.test_model_history_limits import _sample
from tests.test_model_history_limits import history_data as history_data
from tests.test_preview import _equal as assert_equal


def _contract(schema, shared, target):
    sample = _sample()
    events = {
        "orders": replace(
            sample.events["orders"],
            fields={
                **sample.events["orders"].fields,
                "execution_age": api.EventTimeDelta("seconds"),
                "offset": api.Constant(1.0, api.DType("float64")),
                "age": api.Derived("add", ("execution_age", "offset")),
                "answer": api.EventTimeDelta("seconds"),
            },
        )
    }
    if target == "aligned":
        events["other"] = replace(events["orders"], limit=3, order="descending")
    sample = replace(
        sample,
        events=events,
        entity_values={
            "count": api.Aggregate(events["orders"].rows, "count", dtype=api.DType("int64"))
        },
        neighborhoods={
            "graph": api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("orders", 4),)),))
        },
        targets={**sample.targets, "label": api.ExternalTarget(api.DType("string"))},
    )
    source = api.EventSource(
        tuple(events), max_events=3 if target == "aligned" else None, order="descending"
    )
    inputs = {
        "graph": api.Input(
            api.NeighborhoodSource("graph"),
            api.Graph(
                node_fields={
                    "orders": api.Fields(
                        (
                            api.Encode(
                                "amount", api.UseEncoding("shared") if shared else api.ZScore()
                            ),
                        )
                    )
                }
            ),
            None,
        ),
        "history": api.Input(
            source,
            api.Fields(
                (
                    api.Encode("id", api.Raw()),
                    api.Encode("amount", api.Raw()),
                    api.Encode("kind", api.Vocabulary()),
                    api.Encode("age", api.Buckets(bins=4)),
                )
            ),
            api.Padded(),
        ),
        "counts": api.Input(
            api.EntitySource(), api.Fields((api.Encode("count", api.ZScore()),)), api.Separate()
        ),
    }
    if shared:
        inputs["shared_history"] = api.Input(
            source, api.Fields((api.Encode("age", api.UseEncoding("shared")),)), api.Padded()
        )
    targets = {
        "external": api.LabelTarget("label", api.Vocabulary()),
        "computed": api.LabelTarget("future", api.ZScore()),
        "aligned": api.AlignedEventTarget("history", (api.Encode("answer", api.Buckets(bins=3)),)),
    }
    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs,
            {"target": targets[target]},
            encodings={"shared": api.ZScore()} if shared else {},
        ),
    )


@pytest.mark.parametrize("anchor_table", [False, True], ids=["requests", "parquet"])
@pytest.mark.parametrize(
    "shared,target", [(False, "external"), (True, "computed"), (False, "aligned")]
)
def test_mixed_fitting_keeps_columnar_values_and_matches_records(
    history_data, tmp_path, monkeypatch, anchor_table, shared, target
):
    schema, dataset, _memory, original = history_data
    contract = _contract(schema, shared, target)
    requests = [
        replace(request, targets=(api.TargetValue("label", str(i % 2)),))
        for i, request in enumerate(original)
    ]
    requests.append(
        api.SampleRequest(99, 1, 100, split="valid", targets=(api.TargetValue("label", "heldout"),))
    )
    anchors = requests
    if anchor_table:
        path = tmp_path / "anchors.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample": [request.sample_id for request in requests],
                    "entity": [request.entity_id for request in requests],
                    "time": [request.as_of_time for request in requests],
                    "split": [request.split for request in requests],
                    "label": [request.targets[0].value for request in requests],
                }
            ),
            path,
            row_group_size=2,
        )
        anchors = api.AnchorTable(
            api.Parquet(str(path)),
            sample_id="sample",
            entity_id="entity",
            as_of_time="time",
            split="split",
            targets={"label": "label"},
        )
    # Exercise chunk boundaries without manufacturing a large fitting population.
    monkeypatch.setattr(assembly, "_FIT_CHUNK", 2)
    with api.prepare(contract, dataset, api.DiskGraphEngine(num_threads=1)) as plan:
        with monkeypatch.context() as previous:
            previous.setattr(DiskGraphSource, "build_model", None)
            reference = plan.materialize(anchors, fit_split="train")
        with reference:
            expected_state = reference.fitted_state
            expected = {
                fmt: list(reference.batches(batch_size=2, output_format=fmt))
                for fmt in ("model", "inspectable")
            }

        calls = []
        graph_calls = []
        build_model = DiskGraphSource.build_model
        fit_graph = DiskGraphSource.fit_graph_fields

        def columnar(source, sample, selected):
            calls.append(tuple(request.sample_id for request in selected))
            return build_model(source, sample, selected)

        def graph_values(source, sample, spec, roots):
            graph_calls.append(tuple(root.sample_id for root in roots))
            return fit_graph(source, sample, spec, roots)

        def records(*args, **kwargs):
            pytest.fail("mixed fitting reconstructed Python sample records")

        with monkeypatch.context() as candidate:
            candidate.setattr(DiskGraphSource, "build", records)
            candidate.setattr(DiskGraphSource, "build_model", columnar)
            candidate.setattr(DiskGraphSource, "fit_graph_fields", graph_values)
            actual = plan.materialize(anchors, fit_split="train")
        with actual:
            assert calls and all(0 < len(chunk) <= 2 and 99 not in chunk for chunk in calls)
            assert graph_calls == (
                [tuple(request.sample_id for request in original)] if shared else []
            )
            assert actual.fitted_state == expected_state
            assert actual.fitted_state.fingerprint == expected_state.fingerprint
            for fmt, expected_batches in expected.items():
                batches = list(actual.batches(batch_size=2, output_format=fmt))
                assert len(batches) == len(expected_batches)
                for batch, expected_batch in zip(batches, expected_batches, strict=True):
                    assert_equal(batch.inputs, expected_batch.inputs)
                    assert_equal(batch.targets, expected_batch.targets)
                    assert_equal(batch.metadata, expected_batch.metadata)
