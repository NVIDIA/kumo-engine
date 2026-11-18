# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use the package's test-only refusal policy from Engine-owned runners."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

_path = (
    Path(__file__).resolve().parents[2]
    / "packages/experimental/kumo-pytorch/conformance/refusals.py"
)
_spec = importlib.util.spec_from_file_location("_kumo_test_refusals", _path)
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
clean_refusal = _policy.clean_refusal
record_refusal = _policy.record_refusal
pytest_plugin = _policy.pytest_plugin


def case_failure(error, *, allow_refusal=True):
    """Persist the distinction between a clean refusal and failed execution."""
    import traceback

    refusal = record_refusal(error) if allow_refusal else None
    if refusal is not None:
        return {"status": "passed", "outcome": "refused", "refusal": refusal}
    return {
        "status": "failed",
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


@contextmanager
def executing_case():
    """A later lifecycle refusal must not replace an already executing workload."""
    try:
        yield
    except Exception as error:
        if clean_refusal(error) is not None:
            raise AssertionError(
                "Capability refused after case execution began"
            ) from error
        raise
