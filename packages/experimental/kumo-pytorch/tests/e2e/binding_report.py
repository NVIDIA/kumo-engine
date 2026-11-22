# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a dataset test's retained measurements against its source oracle."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

_PENDING = {}


def require_fields(actual, expected, where):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"{where}: expected {key}={value!r}, observed {actual.get(key)!r}")


class BindingReport:
    """Compare counts from source rows with observations from verified batches."""

    def __init__(
        self,
        binding,
        selector,
        representation,
        counts,
        targets,
        features,
        *,
        mode="correctness",
        scope="bounded",
    ):
        self.identity = dict(
            binding=binding,
            selector=selector,
            representation=representation,
            mode=mode,
            scope=scope,
        )
        self.counts = dict(counts)
        self.targets = (
            targets  # head -> per-split source count, or None for generated token targets
        )
        self.features = set(features)
        self.splits = {}

    def observe(self, batch):
        if not os.environ.get("KUMO_PYTORCH_DIAGNOSTICS"):
            return
        splits = batch.metadata["splits"]
        if any(batch.metadata["is_padding"]):
            raise ValueError("Unexpected padding in binding evidence")
        for split, count in Counter(splits).items():
            row = self.splits.setdefault(
                split, {"delivered": 0, "eligible": Counter(), "features": Counter()}
            )
            row["delivered"] += count
            for head in self.targets:
                mask = batch.targets[head]["valid"]
                if count != len(splits):
                    mask = mask[[i for i, name in enumerate(splits) if name == split]]
                row["eligible"][head] += int(mask.sum())
            row["features"].update(
                {name: count for name in self.features.intersection(batch.inputs)}
            )

    def validate(self, report, revision, checks=None, *, allowed_refusal=None):
        if not revision:
            raise ValueError("Missing candidate revision")
        require_fields(
            report, dict(self.identity, revision=revision, tracked_diff_sha256=None), "binding"
        )
        splits = report["splits"]
        if report["outcome"] == "refused":
            if (
                not allowed_refusal
                or report.get("refusal") != allowed_refusal
                or report.get("stage") != "prepare"
                or report.get("full_population_verified") is not False
                or any(row["delivered"] != 0 for row in splits.values())
            ):
                raise ValueError("Invalid prepare refusal")
            return "refused"
        require_fields(
            report,
            {
                "outcome": "executed",
                "stage": "complete",
                "full_population_verified": self.identity["mode"] == "full",
            },
            "execution",
        )
        if set(splits) != self.counts.keys():
            raise ValueError(f"Expected splits {list(self.counts)}, observed {list(splits)}")
        for split, count in self.counts.items():
            row = splits[split]
            require_fields(
                row, {"delivered": count, "features": dict.fromkeys(self.features, count)}, split
            )
            if row["eligible"].keys() != self.targets.keys():
                raise ValueError(f"{split}: eligible heads differ")
            for head, expected in self.targets.items():
                n = row["eligible"][head]
                if type(n) is not int or (n != expected[split] if expected is not None else n <= 0):
                    raise ValueError(f"{split}: wrong eligible {head} count {n}")
        require_fields(report["checks"], checks or {}, self.identity["mode"])
        return "executed"

    def publish(self, *, checks=None, expected_checks=None, refusal=None):
        if os.environ.get("KUMO_PYTORCH_DIAGNOSTICS"):
            node = os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]
            modes = _PENDING.setdefault(node, {})
            mode = self.identity["mode"]
            if mode in modes:
                raise ValueError(f"Duplicate {mode} report for {node}")
            modes[mode] = (self, checks or {}, expected_checks or {}, refusal)


def finish_binding_reports(item, report, *, call_passed=True):
    """Require every declared report from a passing call, including after fixture retries."""
    modes = _PENDING.pop(item.nodeid, {})
    if not call_passed or not report.passed:
        return
    from tests.e2e.runs import _revision

    expected = {
        mark.args[0] if mark.args else item.callspec.params["mode"]
        for mark in item.iter_markers("binding_report")
    }
    try:
        if modes.keys() != expected:
            raise ValueError(
                f"Expected binding reports {sorted(expected)}, received {sorted(modes)}"
            )
        directory = Path(os.environ["KUMO_PYTORCH_DIAGNOSTICS"])
        directory.mkdir(parents=True, exist_ok=True)
        node = hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
        source = _revision()
        for mode, (plan, checks, required, refusal) in modes.items():
            path = directory / f"binding-{mode}-{node}.json"
            data = dict(
                plan.identity,
                **source,
                splits=plan.splits,
                expected=dict(
                    counts=plan.counts,
                    targets=plan.targets,
                    features=sorted(plan.features),
                    checks=required,
                ),
                checks=checks,
                outcome="refused" if refusal else "executed",
                stage="prepare" if refusal else "complete",
                full_population_verified=mode == "full"
                and plan.identity["scope"] == "full_source"
                and not refusal,
            )
            if refusal:
                data["refusal"] = refusal[0]
            try:
                plan.validate(
                    data,
                    os.getenv("GITHUB_SHA", source["revision"]),
                    required,
                    allowed_refusal=refusal[1] if refusal else None,
                )
            except (KeyError, TypeError, ValueError) as error:
                data.update(
                    outcome="invalid",
                    stage="validation",
                    full_population_verified=False,
                    error=str(error),
                )
                raise
            finally:
                pending = path.with_suffix(".tmp")
                pending.write_text(json.dumps(data) + "\n")
                pending.replace(path)
    except (KeyError, OSError, TypeError, ValueError) as error:
        report.outcome = "failed"
        report.longrepr = f"{item.nodeid}: {error}"
