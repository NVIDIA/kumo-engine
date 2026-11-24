# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve composition-slot names from configuration into components"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tfm.attention.eager import EagerAttention
from tfm.contracts.errors import IncompatibleCompositionError, UnsupportedCombinationError
from tfm.heads.binary import Binary
from tfm.objectives.next_event import NextEvent
from tfm.pooling.last_token import LastTokenPool
from tfm.position.none import NoPE

POSITION_BUILDERS: dict[str, Callable[..., Any]] = {"nope": NoPE}
ATTENTION_BUILDERS: dict[str, Callable[..., Any]] = {"eager": EagerAttention}
OBJECTIVE_BUILDERS: dict[str, Callable[..., Any]] = {
    "next_event": NextEvent,
    "causal_next_event": NextEvent,
}
READOUT_BUILDERS: dict[str, Callable[..., Any]] = {"last_token": LastTokenPool}
HEAD_BUILDERS: dict[str, Callable[..., Any]] = {"binary": Binary}

PLANNED: dict[str, dict[str, str]] = {
    "position": {
        "time_rope": "experimental; wired only inside ProfileHistoryModel (recipe 02)",
        "event_rope": "Tier 2, not implemented",
        "learned_abs": "Tier 2, not implemented",
    },
    "attention": {
        "varlen_flash": "Tier 1/2, not implemented; requires the attention slot to be injected",
        "flex": "Tier 2, not implemented",
    },
    "objective": {
        "token_event_key_mlm": "experimental; recipe 02 only, key_value_time layout",
        "multi_gran_mlm": "Tier 2, not implemented",
        "infonce": "Tier 2, not implemented",
        "electra": "Tier 2, not implemented",
    },
    "readout": {
        "mean": "Tier 2, not implemented",
        "attention_pool": "Tier 2, not implemented",
        "latent_attention_pool": "Tier 2, not implemented",
        "usr_token": "experimental; ProfileHistoryModel readout (recipe 02)",
    },
    "head": {
        "survival": "Tier 2, not implemented",
        "regression": "Tier 2, not implemented",
        "multitask": "Tier 2, not implemented",
        "xgboost_export": "optional extra, not implemented",
    },
}


def normalise_objective_specs(value: object) -> list[dict[str, Any]]:
    """Accept a name, a mapping, or a list of either; return a list of mappings."""
    if value is None:
        return [{"name": "next_event"}]
    if isinstance(value, str):
        return [{"name": value}]
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, Mapping):
                out.append(dict(item))
            else:  # a pydantic model, or anything else exposing a name
                out.append({"name": getattr(item, "name", str(item))})
        return out
    raise UnsupportedCombinationError(
        f"objective must be a name, a mapping, or a list of either; got {type(value).__name__}."
    )


def resolve_slot(
    slot: str,
    name: str,
    builders: Mapping[str, Callable[..., Any]],
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Build the component named for ``slot``, or fail with a specific reason."""
    key = str(name).strip().lower()
    if key in builders:
        try:
            return builders[key](**dict(params or {}))
        except TypeError as exc:
            raise IncompatibleCompositionError(
                f"Composition slot {slot!r} component {name!r} does not accept "
                f"{sorted(dict(params or {}))}: {exc}"
            ) from exc
    planned = PLANNED.get(slot, {})
    if key in planned:
        raise IncompatibleCompositionError(
            f"Composition slot {slot!r} requested {name!r}, which is {planned[key]}. "
            f"Milestone 1 implements {sorted(builders)}. "
            "Change the configuration or use the recipe that owns that component."
        )
    raise UnsupportedCombinationError(
        f"Composition slot {slot!r} requested unknown component {name!r}. "
        f"Implemented: {sorted(builders)}. Recognised but unimplemented: {sorted(planned)}."
    )


def resolve_objectives(value: object) -> list[Any]:
    """Build every objective the composition names, forwarding their parameters."""
    built: list[Any] = []
    for spec in normalise_objective_specs(value):
        params = {k: v for k, v in spec.items() if k != "name"}
        built.append(
            resolve_slot("objective", spec.get("name", "next_event"), OBJECTIVE_BUILDERS, params)
        )
    return built
