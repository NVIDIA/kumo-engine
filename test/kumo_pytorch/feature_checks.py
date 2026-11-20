# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named source assertions and evidence shared by dataset bindings."""

from collections import Counter

SPLITS = ("train", "valid", "test")
PHASES = ("population", "training", "evaluation")


def required_assertions(requirements, mode):
    declarations = requirements["assertions"]
    assert declarations and all(
        row["description"] and row["reference"] and row["features"]
        for row in declarations.values()
    ), "Feature assertions require a meaning and an independent reference"
    selected = {
        name: row
        for name, row in declarations.items()
        if mode in row.get("modes", requirements["modes"])
    }
    assert {feature for row in selected.values() for feature in row["features"]} == set(
        requirements["features"]
    ), "Every required feature needs a named assertion in this campaign"
    return selected


class FeatureChecks:
    def __init__(self, requirements, mode, checks, sample_splits):
        declarations = required_assertions(requirements, mode)
        assert set(checks) == set(declarations), (
            "Missing or unregistered feature assertion"
        )
        self.checks = checks
        self.sample_splits = sample_splits
        self.coverage = {
            feature: dict.fromkeys(SPLITS, 0) for feature in requirements["features"]
        }
        self.evidence = {
            name: {
                **declaration,
                "status": "pending",
                "checked_anchors": {
                    phase: dict.fromkeys(SPLITS, 0) for phase in PHASES
                },
            }
            for name, declaration in declarations.items()
        }

    def __call__(self, batch, *, phase="population"):
        assert phase in PHASES
        counts = Counter(
            self.sample_splits[sample] for sample in batch.metadata["sample_ids"]
        )
        assert counts and not (counts.keys() - set(SPLITS))
        failures = []
        for name, check in self.checks.items():
            evidence = self.evidence[name]
            try:
                check(batch)
            except Exception as error:
                evidence.update(
                    status="failed", error=f"{type(error).__name__}: {error}"
                )
                failures.append((name, error))
            else:
                if evidence["status"] != "failed":
                    evidence["status"] = "passed"
                for split, count in counts.items():
                    evidence["checked_anchors"][phase][split] += count
        if failures:
            raise AssertionError(
                "Feature assertions failed: " + ", ".join(name for name, _ in failures)
            ) from failures[0][1]
        if phase == "population":
            for feature in self.coverage.values():
                for split, count in counts.items():
                    feature[split] += count


def validate_evidence(report, requirements, mode):
    declarations = required_assertions(requirements, mode)
    evidence = report["feature_assertions"]
    assert set(evidence) == set(declarations), "Missing individual feature evidence"
    delivered = {split: row["delivered"] for split, row in report["splits"].items()}
    for name, declaration in declarations.items():
        row = evidence[name]
        assert row["status"] == "passed", f"Feature assertion {name} did not pass"
        assert all(row[key] == value for key, value in declaration.items()), (
            "Feature assertion meaning changed"
        )
        counts = row["checked_anchors"]
        assert set(counts) == set(PHASES)
        assert all(
            set(group) == set(SPLITS)
            and all(type(n) is int and n >= 0 for n in group.values())
            for group in counts.values()
        )
        assert counts["population"] == delivered, (
            f"Feature assertion {name} missed selected anchors"
        )
        if mode == "training":
            assert all(
                type(repeat["observed_train_anchors"]) is int
                and repeat["observed_train_anchors"] >= repeat["completed_updates"]
                for repeat in report["repeats"]
            )
            assert counts["training"]["train"] == sum(
                repeat["observed_train_anchors"] for repeat in report["repeats"]
            ), f"Feature assertion {name} missed training input observations"
            assert counts["training"]["valid"] == counts["training"]["test"] == 0
            assert counts["evaluation"] == {
                "train": 0,
                "valid": delivered["valid"],
                "test": delivered["test"],
            }
        else:
            assert all(
                not any(counts[phase].values()) for phase in ("training", "evaluation")
            )
