# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the original binding's optimizer, fitted artifact and loader checkpoint."""

import json
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from e2e.datasets.binding_sources import PINS, _sources
from kumo.pytorch import api
from tests.e2e.binding_lifecycle import run_lifecycle
from tests.e2e.binding_plan import NIGHTLY_TRAINING
from tests.e2e.binding_report import BindingReport
from tests.e2e.binding_training import run_training, train_full_binding
from tests.e2e.cuda_metrics import training_gpu_window
from tests.e2e.full_binding_run import training_contract


def _evidence(source, mode, counts=None, targets=None):
    i = source.identity
    return BindingReport(
        i["binding"],
        i["selector"],
        i["representation"],
        counts or source.counts,
        targets or source.targets,
        source.features,
        mode=mode,
    )


def _lifecycle_table(task, table, binding, root, work):
    if task.count("train") >= 2:
        return table
    # These correctness cohorts have one training row. Select one more genuine train
    # anchor for the same complete root history, from the already-pinned source.
    names, _ = _sources(binding)
    pin = json.loads((PINS / (names[-1] + ".json")).read_text())
    path = root / pin["objects"]["anchors"]["key"]
    native = pq.read_table(table.source.path)
    (row,) = native.filter(pc.equal(native["split"], "train")).to_pylist()
    columns = ["sample_id", "entity_id", "split", "as_of_time"]
    if binding == "M04":
        columns += ["future_count", "graph_as_of_time", "outcome_end"]
    offset = 0
    for raw in pq.ParquetFile(path).iter_batches(columns=columns):
        matching = pc.and_(
            pc.equal(raw["entity_id"], row["registered_entity_id"]), pc.equal(raw["split"], "train")
        )
        indices = pc.indices_nonzero(
            pc.and_(matching, pc.not_equal(raw["sample_id"], row["sample_id"]))
        )
        if len(indices):
            pos = int(indices[0].as_py())
            record = raw.slice(pos, 1).to_pylist()[0]
            extended = row | {
                k: value for k, value in record.items() if k in row and k != "entity_id"
            }
            extended.update(
                anchor_row_position=offset + pos,
                prediction_time_ns=raw["as_of_time"].cast(pa.int64())[pos].as_py(),
            )
            if binding == "M04":
                extended.update(
                    as_of_time=record["graph_as_of_time"],
                    count=float(record["future_count"]),
                    original_count=record["future_count"],
                )
            if "cutoff_epoch_seconds" in row:
                extended["cutoff_epoch_seconds"] = extended["prediction_time_ns"] // 10**9
            path = work / "lifecycle-anchors.parquet"
            pq.write_table(
                pa.concat_tables([native, pa.Table.from_pylist([extended], schema=native.schema)]),
                path,
            )
            return replace(table, source=api.Parquet(str(path)))
        offset += len(raw)
    raise ValueError(f"{binding} needs two original training anchors for mid-epoch resume")


def exercise_binding(task, data, table, engine, work, source):
    binding = source.identity["binding"]
    if device := os.getenv("KUMO_PYTORCH_BINDING_SMOKE_DEVICE"):
        assert binding in NIGHTLY_TRAINING and device in ("cuda", "cpu")
        cutoff_ns = task.fitted_state.fit["inputs"]["statistics"]["cutoff"]
        cutoff = None if cutoff_ns is None else cutoff_ns // 10**9
        with (
            api.prepare(training_contract(task.contract), data, engine) as plan,
            plan.materialize(table, fit_split="train", encoder_cutoff=cutoff) as recent,
        ):
            result = train_full_binding(
                recent,
                binding,
                batch_size=2,
                device=device,
                split_batches=lambda split, epoch: recent.batches(
                    split=split, epoch=epoch, batch_size=2, drop_last=False, output_format="model"
                ),
                train_window=partial(training_gpu_window, device)
                if device == "cuda"
                else nullcontext,
            )
        for row in [*result["epochs"], *result["validation"], result["test"]]:
            assert row["samples"] == source.counts[row["split"]]
            for head, loss in row["heads"].items():
                expected = source.targets[head]
                if expected is not None:
                    assert loss["eligible"] == expected[row["split"]]
        return
    training = _evidence(source, "training")
    checks = run_training(task, work, training, binding)
    training.publish(checks=checks, expected_checks={"updates": [100, 100], "reproducible": True})
    raw = _lifecycle_table(task, table, binding, Path(os.environ["KUMO_PYTORCH_DATA_ROOT"]), work)
    table = raw
    with api.prepare(task.contract, data, engine) as plan:
        # On scalar-bearing contracts the implicit values cache has a temporary path.
        # Keep the sidecar so its anchor identity also survives checkpoint reload.
        if task.contract.sample.entity_values or any(
            isinstance(t, api.Aggregate) for t in task.contract.sample.targets.values()
        ):
            table = plan._materialize_values(raw, work / "values", stats_split="train").anchors
        with plan.materialize(table, fitted=task.fitted_state) as labelled:
            counts = {s: labelled.count(s) for s in source.counts}
            targets = {
                k: None if v is None else {s: v[s] for s in counts}
                for k, v in source.targets.items()
            }
            if counts["train"] > source.counts["train"]:
                for expected in targets.values():
                    if expected is not None:
                        expected["train"] += 1
            lifecycle = _evidence(source, "lifecycle", counts, targets)
            threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(17)
                    checks = run_lifecycle(
                        labelled,
                        data,
                        table,
                        engine.index,
                        work,
                        lifecycle,
                        binding=binding,
                        raw_table=raw,
                    )
            finally:
                torch.set_num_threads(threads)
    lifecycle.publish(
        checks=checks,
        expected_checks=dict(
            fresh_process=True,
            inference=counts["valid"] + counts["test"],
            mid_epoch_resume=counts["train"] - 1,
            finetuning=counts["train"] - 1,
        ),
    )
