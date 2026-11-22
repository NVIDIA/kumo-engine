# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliver each eligible original source row through the selected DiskGraph binding."""

import json
import os
from datetime import UTC
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from e2e.datasets.binding_sources import full_source
from kumo.pytorch import api
from tests.e2e.binding_plan import BINDINGS
from tests.e2e.binding_report import BindingReport

_SPLITS = ("train", "valid", "test")


def _source_heads(binding):
    if binding in ("S01", "E03"):
        return {"masked_values": None}
    if binding in ("S02", "E04"):
        return {"next_token": None}
    if binding == "M03":
        return {k: ds.field("next_" + k + "_valid") for k in ("event_type", "event_subtype")}
    if binding == "M04":
        return dict.fromkeys(("count", "generated_count"), ds.field("future_count").is_valid())
    if binding == "L01":
        return {k: ds.field("next_" + k).is_valid() for k in ("amount", "currency")}
    return {"label": ds.field("label" if binding == "S04" else "involvement").is_valid()}


def _anchor_ids(path, split_column, split):
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["sample_id", split_column]):
        indices = pc.indices_nonzero(pc.equal(batch[split_column], split))
        yield from zip(
            pc.add(indices.cast(pa.int64()), offset).to_pylist(),
            batch["sample_id"].take(indices).to_pylist(),
            strict=True,
        )
        offset += len(batch)


def _check_population(binding, files, native, index):
    spec, data, table = native
    _, _, selector, representation = BINDINGS[binding]
    raw = ds.dataset(files["anchors"])
    manifest = json.loads(files["anchor_manifest"].read_text())
    assert manifest["statistics"]["rows"] == raw.count_rows()
    column = "forecast_split" if binding in ("E01", "E02", "E05", "L01") else "split"
    splits = {s: ds.field(column) == s for s in _SPLITS}
    counts = {s: raw.count_rows(filter=pred) for s, pred in splits.items()}
    assert min(counts.values()) > 0 and sum(counts.values()) == raw.count_rows(
        filter=ds.field(column).is_valid()
    )
    targets = {
        name: None
        if eligible is None
        else {s: raw.count_rows(filter=pred & eligible) for s, pred in splits.items()}
        for name, eligible in _source_heads(binding).items()
    }
    last_train = max(
        pc.max(batch["as_of_time"]).as_py()
        for batch in raw.to_batches(columns=["as_of_time"], filter=splits["train"])
        if len(batch)
    )
    cutoff = int(last_train.replace(tzinfo=UTC).timestamp()) - 1
    report = BindingReport(
        binding,
        selector,
        representation,
        counts,
        targets,
        spec.batch.inputs,
        mode="full",
        scope="full_source",
    )
    engine = api.DiskGraphEngine(index=index, num_threads=1, num_shards=1)
    with (
        api.prepare(spec, data, engine) as plan,
        plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as task,
    ):
        for split, count in counts.items():
            assert task.count(split) == count
            expected = _anchor_ids(files["anchors"], column, split)
            for batch in task.batches(
                split=split,
                batch_size=32 if binding == "L01" else 4,
                shuffle=False,
                output_format="model",
            ):
                meta = batch.metadata
                ids = zip(
                    (r["anchor_row_position"] for r in meta["request_metadata"]),
                    meta["sample_ids"],
                    strict=True,
                )
                assert list(ids) == list(islice(expected, batch.batch_size))
                assert set(meta["splits"]) == {split} and not any(meta["is_padding"])
                report.observe(batch)
            assert next(expected, None) is None
    report.publish()


@pytest.mark.parametrize("binding", BINDINGS)
@pytest.mark.binding_report("full")
def test_full_binding(binding, tmp_path):
    if os.getenv("KUMO_PYTORCH_FULL_BINDING") != binding:
        pytest.skip("select the registered full-population binding on manual dispatch")
    if not os.environ.get("KUMO_PYTORCH_DIAGNOSTICS"):
        pytest.fail("set KUMO_PYTORCH_DIAGNOSTICS to validate full-population evidence")
    files = full_source(binding, Path(os.environ["KUMO_PYTORCH_DATA_ROOT"]))
    if binding.startswith(("E", "L")):
        from e2e.datasets.aml import prepare_full_aml

        native = prepare_full_aml(files, tmp_path / "prepared", binding)
    else:
        from e2e.datasets.population_tables import prepare_full

        native = prepare_full(binding, files, tmp_path / "prepared")
    _check_population(binding, files, native, tmp_path / "index")
