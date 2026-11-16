# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from asgiref.compatibility import guarantee_single_callable
from starlette.datastructures import MutableHeaders
from starlette.types import Message


class StripHeaderMiddleware:
    """The ASGI application middleware.

    Args:
        app: The ASGI application callable to forward requests to.
        exclude_paths: HTTP routes to exclude from header stripping
    """
    def __init__(
        self,
        app,
        headers: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        self.headers = headers or []
        self.exclude_paths = exclude_paths or []
        self._app = guarantee_single_callable(app)

    async def __call__(self, scope, receive, send):
        """The ASGI application

        Args:
            scope: A ASGI environment.
            receive: An awaitable callable yielding dictionaries
            send: An awaitable callable taking a single dictionary as argument.
        """
        async def send_wrapper(message: Message) -> None:
            if ('path' in scope and scope['path'] not in self.exclude_paths
                    and 'type' in message
                    and message['type'] == "http.response.start"):
                response_headers = MutableHeaders(scope=message)
                for header in self.headers:
                    if header in response_headers:
                        del response_headers[header]
            await send(message)

        return await self._app(scope, receive, send_wrapper)
