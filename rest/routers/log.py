# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as pylogging
from enum import Enum
from http import HTTPStatus

from fastapi.param_functions import Depends
from fastapi.routing import APIRouter

from io_utils import kumo_dataclass, logging
from kumo.connector.connector import Optional
from rest.config.status import EmptyResponse
from rest.deps.authentication import Authentication

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def python_level(self) -> int:
        if self == LogLevel.INFO:
            return pylogging.INFO
        elif self == LogLevel.WARNING:
            return pylogging.WARNING
        elif self == LogLevel.ERROR:
            return pylogging.ERROR
        raise RuntimeError(f"Invalid LogLevel {self}")


@kumo_dataclass
class LogErrorArgs:
    message: str
    level: LogLevel
    stacktrace: Optional[str]


class LogResource:
    """Allows the frontend javascript code to report error logs to the backend.
    """
    def __init__(self, auth: Authentication) -> None:
        self.router = APIRouter(
            prefix="/v0/log",
            tags=["Log"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description": "Not found"
                },
                HTTPStatus.OK: {
                    "content": {
                        "image/svg+xml": {}
                    }
                }
            },
        )

        self.router.add_api_route(
            "/log_ui_error",
            self.log_ui_error,
            response_model=EmptyResponse,
            methods=['POST'],
        )

    def log_ui_error(
        self,
        args: LogErrorArgs,
    ) -> EmptyResponse:
        """Log a UI crash report to the backend, so that it can be viewed in
        our logsearch tool.

        Args:
            message (str): a 1-line description of the error
            level (str): info|warning|error
            stacktrace (str): the location in the code that raised the error
                message
        """
        # TODO: generate structured logs to Loki, rather than stderr.
        logger.log(args.level.python_level, "Received error from UI: %s\n%s",
                   args.message, args.stacktrace)
        return EmptyResponse()
