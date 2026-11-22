# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliver each eligible original source row through the selected DiskGraph binding."""

import os
from pathlib import Path

import pytest

from e2e.datasets.binding_sources import full_source
from kumo.pytorch import api
from tests.e2e.binding_plan import BINDINGS
from tests.e2e.binding_report import BindingReport
from tests.e2e.full_binding_run import full_batches, full_tables, source_population


def _check_population(binding, files, native, index):
    spec, data, table = native
    _, _, selector, representation = BINDINGS[binding]
    column, counts, targets, cutoff = source_population(binding, files)
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
            for batch in full_batches(
                task, files["anchors"], column, 32 if binding == "L01" else 4, split
            ):
                report.observe(batch)
    report.publish()


@pytest.mark.parametrize("binding", BINDINGS)
@pytest.mark.binding_report("full")
def test_full_binding(binding, tmp_path):
    if os.getenv("KUMO_PYTORCH_FULL_BINDING") != binding:
        pytest.skip("select the registered full-population binding on manual dispatch")
    if not os.environ.get("KUMO_PYTORCH_DIAGNOSTICS"):
        pytest.fail("set KUMO_PYTORCH_DIAGNOSTICS to validate full-population evidence")
    files = full_source(binding, Path(os.environ["KUMO_PYTORCH_DATA_ROOT"]))
    native = full_tables(binding, files, tmp_path / "prepared")
    _check_population(binding, files, native, tmp_path / "index")
