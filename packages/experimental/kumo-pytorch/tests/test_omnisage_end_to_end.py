# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the combined OmniSage workload against Parquet source truth."""

import math
import os
from contextlib import nullcontext
from itertools import chain

import pytest
import torch

from kumo.pytorch import api
from tests.e2e.omnisage_fixture import make_fixture, spec
from tests.e2e.omnisage_verify import Oracle
from tests.e2e.s3_endpoint import local_s3


def _check_train_step(batch, model_file):
    from tests.e2e.omnisage_training import released_model, train_step

    model = released_model(model_file)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = [p.detach().clone() for p in model.parameters()]
    loss, pairs = train_step(batch, model, optimizer)
    assert pairs > 0 and math.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(
        not torch.equal(before, p) for before, p in zip(initial, model.parameters(), strict=True)
    )


@pytest.mark.parametrize("on_s3", [False, True], ids=["local", "s3"])
def test_omnisage_diskgraph_loaders(tmp_path, on_s3):
    pytest.importorskip("diskgraph")
    if on_s3:
        pytest.importorskip("moto.server")

    from tests.e2e.omnisage_custom import CustomLoader

    with local_s3() if on_s3 else nullcontext() as client:
        data = tmp_path / "source"
        model_file = os.getenv("OMNISAGE_MODEL_FILE")
        workload = make_fixture(
            data, pins=32, anchors=48, width=1152 if model_file else 16, client=client
        )
        oracle = Oracle(workload)
        if on_s3:
            # The verifier has the source truth. Neither loader can use the local copies.
            data.rename(tmp_path / "original-parquet")
        engine = api.DiskGraphEngine(num_threads=1, random_seed=42)
        with (
            api.prepare(spec(workload, 7), workload.dataset, engine) as plan,
            plan.materialize(workload.anchors) as task,
            CustomLoader(
                engine.index,
                workload.anchors.source,
                7,
                width=workload.width,
                seed=42,
                num_threads=1,
            ) as custom,
        ):
            for batches in (task.batches(output_format="model"), custom.batches()):
                if model_file:
                    batches = iter(batches)
                    first = next(batches)
                    _check_train_step(first, model_file)
                    batches = chain((first,), batches)
                counts = oracle.verify_epoch(batches)
                assert counts["anchors"] == 48 and counts["batches"] == 7
                assert counts["graph_roots"] == 314
                assert counts["supplied_candidates"] == 86
                assert counts["scored_candidates"] == counts["future_candidates"] == 90
                assert counts["scored_edges"] > counts["graph_roots"]
