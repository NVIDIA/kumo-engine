# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence

from fastapi import routing


def routes_for_prefix(routes: Sequence[Any], prefix: str) -> list[Any]:
    """Return effective routes whose full path starts with ``prefix``."""
    iter_routes = getattr(routing, 'iter_route_contexts', iter)
    return [
        route for route in iter_routes(routes)
        if getattr(route, 'path', '').startswith(prefix)
    ]
