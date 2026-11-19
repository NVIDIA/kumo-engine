# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an explicitly recorded subset of unchanged upstream tests on DiskGraph."""

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "packages/experimental/kumo-pytorch"
EXCLUDED_TESTS = {
    "test_synthetic_banking_materialization_sampling_and_reopen_are_reproducible",
    "test_diskgraph_refuses_future_intermediate_edge_that_leaks_old_rows",
    "test_diskgraph_temporal_sequences_aggregates_and_labels_match_reference",
    "test_diskgraph_pragma_batches_match_reference_without_record_adaptation",
    "test_both_engines_and_every_thread_count_train_the_same_tokenizer",
}
EXCLUDED_PROPERTIES = {
    "differential",
    "graph_differential",
    "sequence_differential",
    "anchor_splits",
    "fitted_state",
}
NATIVE_FILES = {
    "test_diskgraph_backend.py",
    "test_diskgraph_graph_output.py",
    "test_diskgraph_unsupported.py",
    "test_materialized_native_batches.py",
    "test_native_anchor_handoff.py",
    "test_native_length_planning.py",
    "test_padding_identity_native.py",
    "test_clock_precision.py",
    "test_sampled_graph_topology.py",
    "test_inspectable_graph_batches.py",
    "test_text_from_index.py",
    "test_history_cap_pushdown.py",
    "test_graph_self_relation.py",
    "test_time_boundaries.py",
    "test_mixed_graph_fitting.py",
    "test_api_reference.py",
    "test_graph_edge_uniqueness.py",
}


class Selection:
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        chosen, excluded = [], []
        for item in items:
            params = getattr(getattr(item, "callspec", None), "params", {})
            obligation = params.get("obligation")
            engine = params.get("engine_name", params.get("engine"))
            property_name = None
            if obligation:
                property_name, engine, _ = obligation
            name = getattr(item, "originalname", None) or item.name
            reason = "Not an original DiskGraph-specific case"
            keep = (
                engine == "diskgraph"
                or item.path.name in NATIVE_FILES
                or name == "test_diskgraph_refuses_it_with_its_code"
            )
            if "conformance" in item.path.parts:
                keep = engine == "diskgraph" and name in {
                    "test_public_behavior",
                    "test_scalar_compatibility_and_real_execution",
                    "test_property",
                    "test_corpus",
                }
            if engine == "in_memory" or "in_memory" in item.name:
                keep, reason = False, "InMemory variant"
            if name in EXCLUDED_TESTS or (
                name == "test_property" and property_name in EXCLUDED_PROPERTIES
            ):
                keep, reason = False, "Uses InMemory as a reference or setup engine"
            if keep:
                chosen.append(item)
            else:
                excluded.append({"nodeid": item.nodeid, "reason": reason})
        path = Path(os.environ["BASELINE_SELECTION"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "selected": [i.nodeid for i in chosen],
                    "excluded": excluded,
                    "source_revision": "3fde15c7f9a9652f40ed77b2704479e9ce868f4b",
                },
                indent=2,
            )
            + "\n"
        )
        config.hook.pytest_deselected(items=[i for i in items if i not in chosen])
        items[:] = chosen
        if not chosen:
            raise pytest.UsageError("No original DiskGraph tests selected")

    def pytest_sessionstart(self, session):
        from kumo.pytorch.api import InMemoryEngine

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "BASELINE SELECTION ERROR: this original case uses InMemory; "
                "exclude it explicitly rather than counting this as a package defect"
            )

        InMemoryEngine.__signature__ = inspect.signature(InMemoryEngine)
        InMemoryEngine.__init__ = forbidden


if __name__ == "__main__":
    os.chdir(PACKAGE)
    sys.path[:0] = [str(PACKAGE / "src"), str(PACKAGE)]
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "--require-imports",
                "--conformance-engine=diskgraph",
                *sys.argv[1:],
            ],
            plugins=[Selection()],
        )
    )
