# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swap one building-block slot at a time and score every variant the same way.

Each variant is the base config plus a few dotted-key overrides. The data
section and the seed are shared, so every variant sees the same book, cutoffs,
label windows, and probe user split; only the swapped slot differs.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tfm.config import RecipeConfig
from tfm.runtime.pipeline import run_evaluate, run_train, setup_run

SLOTS = ("baseline", "event representation", "sequence architecture", "objective", "readout")
REPORTED_ARMS = ("raw", "embedding", "combined")


@dataclass(frozen=True)
class Variant:
    """The base config with `overrides` applied; `slot` names what changed."""

    name: str
    slot: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise ValueError(f"Variant {self.name!r}: slot {self.slot!r} is not one of {SLOTS}.")


def apply_overrides(config: RecipeConfig, overrides: Mapping[str, Any]) -> RecipeConfig:
    """`{"model.composition.readout": "mean"}` sets one key; a mapping value replaces a section."""
    raw = config.model_dump(mode="json")
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"Override {dotted!r}: no section {key!r}.")
            node = node[key]
        node[leaf] = value
    return RecipeConfig.model_validate(raw)


def load_variants(path: Path) -> list[Variant]:
    with path.open("r", encoding="utf-8") as handle:
        rows = yaml.safe_load(handle)["variants"]
    return [
        Variant(
            name=row["name"],
            slot=row["slot"],
            overrides=row.get("overrides") or {},
            note=row.get("note", ""),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class VariantResult:
    variant: Variant
    seeds: tuple[int, ...]
    metrics: tuple[dict[str, dict[str, float]], ...]

    def values(self, arm: str, metric: str) -> list[float]:
        return [m[arm][metric] for m in self.metrics if arm in m and metric in m[arm]]


def run_comparison(
    base: RecipeConfig,
    variants: Sequence[Variant],
    run_root: Path,
    *,
    seeds: Sequence[int],
    device: str | None = None,
) -> list[VariantResult]:
    """Train and evaluate every variant at every seed, in order."""
    configs = {v.name: apply_overrides(base, v.overrides) for v in variants}
    results = []
    for variant in variants:
        per_seed = []
        for seed in seeds:
            config = configs[variant.name].model_copy(update={"seed": seed})
            run = setup_run(
                config=config,
                run_dir=run_root / variant.name / f"seed_{seed}",
                device=device,
                recipe="ablation_building_blocks",
            )
            run_train(run)
            per_seed.append(run_evaluate(run))
        results.append(VariantResult(variant, tuple(seeds), tuple(per_seed)))
    return results


def _cell(values: list[float]) -> str:
    if not values:
        return "skipped"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def comparison_markdown(results: Sequence[VariantResult], *, task: str = "") -> str:
    """One row per variant: fraud-probe ROC-AUC per arm, as median [min, max] over seeds."""
    prefix = f"{task}_" if task else ""
    seeds = results[0].seeds if results else ()
    lines = [
        f"Fraud probe ROC-AUC{f' ({task})' if task else ''}, "
        f"{'median [min, max] over seeds ' if len(seeds) > 1 else 'seed '}"
        f"{', '.join(map(str, seeds))}.",
        "",
        "| Variant | Slot changed | " + " | ".join(REPORTED_ARMS) + " | Eval positives |",
        "|---|---|" + "---|" * len(REPORTED_ARMS) + "---|",
    ]
    for result in results:
        cells = [_cell(result.values(prefix + arm, "roc_auc")) for arm in REPORTED_ARMS]
        positives = result.values(prefix + "embedding", "n_eval_positive")
        support = str(int(min(positives))) if positives else "-"
        lines.append(
            f"| `{result.variant.name}` | {result.variant.slot} | "
            + " | ".join(cells)
            + f" | {support} |"
        )
    notes = [f"- `{r.variant.name}`: {r.variant.note}" for r in results if r.variant.note]
    if notes:
        lines += ["", *notes]
    return "\n".join(lines) + "\n"


def comparison_record(results: Sequence[VariantResult]) -> dict[str, Any]:
    return {
        "variants": [
            {
                "name": r.variant.name,
                "slot": r.variant.slot,
                "overrides": dict(r.variant.overrides),
                "seeds": list(r.seeds),
                "metrics": list(r.metrics),
            }
            for r in results
        ]
    }
