# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Set, Tuple, Type

from asgiref.compatibility import guarantee_single_callable
from fastapi import Request
from kumo_core.version import get_kumo_version
from pyinstrument import Profiler
from pyinstrument.renderers.base import FrameRenderer
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.jsonrenderer import JSONRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer

from async_utils.sync import blocking_func
from io_utils import fsutil, logging
from service_framework.deploy_env_config import deploy_env

log = logging.getLogger(__name__)

profile_format_dict: Dict[str, Tuple[Type[FrameRenderer], str]] = {
    "html": (HTMLRenderer, '.html'),
    "json": (JSONRenderer, '.json'),
    'speedscope': (SpeedscopeRenderer, '.json'),  # default
}

PROFILING_THREAD_POOL = ThreadPoolExecutor(
    thread_name_prefix='DataBackendCache-Threadpool')


class ProfilingMiddleware:
    """The ASGI application middleware.

    This class is an ASGI middleware that enables request profiling when query
    parameter 'profile' is true.

    Args:
        app: The ASGI application callable to forward requests to.
    """
    def __init__(self, app):
        self._app = guarantee_single_callable(app)
        self._output_base_dir = os.path.join(
            deploy_env().control_plane_storage_root, 'profiling',
            get_kumo_version())
        self._output_tasks: Set[asyncio.Task] = set()

    async def __call__(self, scope, receive, send):
        """The ASGI application

        Args:
            scope: A ASGI environment.
            receive: An awaitable callable yielding dictionaries
            send: An awaitable callable taking a single dictionary as argument.
        """
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        req = Request(scope)
        # Enable profiling when query param has 'profile=true'
        if req.query_params.get('profile') != 'true':
            return await self._app(scope, receive, send)

        log.debug('Profling is enabled.')
        with Profiler(interval=0.001, async_mode="enabled") as profiler:
            response = await self._app(scope, receive, send)

        # we dump the profiling into a file
        format = req.query_params.get("profile_format", "speedscope")
        if format not in profile_format_dict:
            log.warning('Profile format %s is not supported, use default',
                        format)
            format = 'speedscope'
        renderer_cls, suffix = profile_format_dict[format]
        renderer = renderer_cls()

        method = req.method
        req_path = req.url.path.strip('/').replace('/', '_')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self._output_base_dir,
                                   f'{method}-{req_path}-{ts}{suffix}')

        @blocking_func(PROFILING_THREAD_POOL)
        def _write_output():
            fsutil.write_text(output_file, profiler.output(renderer=renderer))
            log.info('Wrote profiling result to %s', output_file)

        output_task = asyncio.ensure_future(_write_output.coroutine())
        self._output_tasks.add(output_task)
        output_task.add_done_callback(self._output_tasks.discard)

        return response
