# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The HTTP surface the typed API classes need from a Kumo client.

Each ``*API`` class in this package issues requests and parses responses; it
does not need authentication, retry policy, project resolution, or any of the
other state a :class:`~kumoai_studio.client.client.KumoClient` carries. Declaring that
narrow surface here lets the API modules depend on :class:`Transport` instead
of on ``KumoClient``.

That inverts a dependency that was previously circular. ``client.py`` owns the
registry of typed APIs, so it must import them; before, they imported it back
for a constructor annotation, which forced ``client.py`` to defer every one of
those imports into a property body. This module is a leaf -- it imports nothing
from the package -- so the cycle is gone and ``client.py`` imports the API
classes normally.

:class:`~kumoai_studio.client.client.KumoClient` satisfies this protocol structurally;
no registration or subclassing is required.
"""

from typing import Any, Protocol, runtime_checkable

import requests

from kumoai_studio.client.endpoints import Endpoint

__all__ = ['Transport']


@runtime_checkable
class Transport(Protocol):
    r"""Issues authenticated HTTP requests against the Kumo service.

    The method names keep their leading underscore because they name an
    internal transport rather than public SDK surface, and because the typed
    API classes already call them that way.
    """
    def _request(self, endpoint: Endpoint, **kwargs: Any) -> requests.Response:
        r"""Issues a request described by a typed :class:`Endpoint`."""
        ...

    def _get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        ...

    def _post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        ...

    def _patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        ...
