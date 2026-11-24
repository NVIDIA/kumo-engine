# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple name registry for supported composition components."""

from __future__ import annotations

from tfm.contracts.capabilities import Capabilities
from tfm.contracts.errors import UnsupportedCombinationError

_REGISTRY: dict[str, Capabilities] = {}


def register(cap: Capabilities) -> Capabilities:
    _REGISTRY[cap.name] = cap
    return cap


def get_capability(name: str) -> Capabilities:
    if name not in _REGISTRY:
        raise UnsupportedCombinationError(f"Unknown component {name!r}.")
    return _REGISTRY[name]


def advertised_capabilities() -> dict[str, Capabilities]:
    return dict(_REGISTRY)


def _cap(name: str, *, causal: bool, bidirectional: bool) -> None:
    register(
        Capabilities(
            name=name,
            causal=causal,
            bidirectional=bidirectional,
            requires_timestamps=False,
            support_tier="reference",
        )
    )


_cap("decoder", causal=True, bidirectional=False)
_cap("flat", causal=True, bidirectional=False)
_cap("nope", causal=True, bidirectional=True)
_cap("eager_attention", causal=True, bidirectional=True)
_cap("last_token_pool", causal=True, bidirectional=True)
_cap("next_event", causal=True, bidirectional=False)
_cap("binary", causal=True, bidirectional=True)
register(
    Capabilities(
        name="transaction_field_tokenizer",
        causal=True,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="reference",
    )
)
register(
    Capabilities(
        name="key_value_time",
        causal=False,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="time_rope",
        causal=True,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="profile_history",
        causal=False,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="profile_event_decoder",
        causal=True,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="next_event_fields",
        causal=True,
        bidirectional=False,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="token_event_key_mlm",
        causal=False,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
register(
    Capabilities(
        name="usr_token_pool",
        causal=False,
        bidirectional=True,
        requires_timestamps=True,
        support_tier="experimental",
    )
)
