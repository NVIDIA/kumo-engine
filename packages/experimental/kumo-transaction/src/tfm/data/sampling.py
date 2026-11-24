# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity subsampling that keeps minority-class users when labels exist."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from tfm.contracts.errors import SchemaError


def stratified_sample_ids(
    population: Sequence[object],
    positives: Sequence[object],
    *,
    cap: int,
    seed: int,
) -> list[object]:
    """Return ``cap`` ids, preferring about half from ``positives`` when possible."""
    if cap <= 0:
        raise SchemaError("Entity sample cap must be positive.")
    seen: dict[object, None] = {}
    ordered: list[object] = []
    for item in population:
        if item in seen:
            continue
        seen[item] = None
        ordered.append(item)
    pos_set = {item for item in positives if item in seen}
    rng = Random(seed)
    pos = [item for item in ordered if item in pos_set]
    neg = [item for item in ordered if item not in pos_set]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), max(1, cap // 2)) if pos else 0
    chosen = pos[:n_pos]
    chosen.extend(neg[: max(0, cap - len(chosen))])
    if len(chosen) < cap:
        chosen.extend(pos[n_pos : n_pos + (cap - len(chosen))])
    return chosen[:cap]
